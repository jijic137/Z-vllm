import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert denominator > 0, f"denominator must be positive, got {denominator}"
    assert numerator % denominator == 0
    return numerator // denominator


def _tp_rank_size(tp_group: "dist.ProcessGroup | None") -> tuple[int, int]:
    """取 TP 组内的 (rank, world_size)。

    PyTorch 对单 rank 的 NCCL/RCCL 子组，get_world_size 可能返回 -1
    （元数据未初始化）——纯 EP（moe_tp_size=1）时每个专家组即单卡组，
    此处归一化为 (0, 1)。
    """
    rank = dist.get_rank(tp_group)
    size = dist.get_world_size(tp_group)
    if size in (-1, 1):
        return 0, 1
    return rank, size


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        tp_group: "dist.ProcessGroup | None" = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        # tp_group 为 None 表示全局 TP 组（attention 等）；MoE 专家传入专家内 TP 子组
        self.tp_group = tp_group
        self.tp_rank, self.tp_size = _tp_rank_size(tp_group)
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: "dist.ProcessGroup | None" = None,
    ):
        world_size = _tp_rank_size(tp_group)[1]
        super().__init__(input_size, divide(output_size, world_size), bias, 0, tp_group)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        tp_group: "dist.ProcessGroup | None" = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, tp_group)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        tp_group: "dist.ProcessGroup | None" = None,
    ):
        tp_size = _tp_rank_size(tp_group)[1]
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        if total_num_kv_heads % tp_size == 0:
            self.num_kv_heads = total_num_kv_heads // tp_size
        else:
            # TP 超过 KV head 数时（如 8 卡跑 4 KV head 的模型）：复制 KV head，每 rank 持有一个
            assert tp_size % total_num_kv_heads == 0
            self.num_kv_heads = 1
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        if self.num_kv_heads * tp_size == total_num_kv_heads:
            super().__init__(hidden_size, output_size, bias, tp_group)
        else:
            # 复制模式下每 rank 宽度 != 总量/tp，直接按每 rank head 数初始化
            LinearBase.__init__(self, hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_size, bias, 0, tp_group)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + (0 if loaded_shard_id == "k" else self.num_kv_heads * self.head_size)
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        if loaded_shard_id == "q" or self.num_kv_heads * self.tp_size == self.total_num_kv_heads:
            loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        else:
            # KV head 复制：rank r 取 head 组 r % total_num_kv_heads
            loaded_weight = loaded_weight.chunk(self.total_num_kv_heads, self.tp_dim)[self.tp_rank % self.total_num_kv_heads]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: "dist.ProcessGroup | None" = None,
    ):
        world_size = _tp_rank_size(tp_group)[1]
        super().__init__(divide(input_size, world_size), output_size, bias, 1, tp_group)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y, group=self.tp_group)
        return y
