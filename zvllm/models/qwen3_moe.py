import torch
from torch import nn
import torch.distributed as dist

from zvllm.layers.activation import SiluAndMul
from zvllm.layers.fused_moe import BMM_MAX_TOKENS, FUSED_MAX_TOKENS, fused_moe_bmm, fused_moe_triton, triton_available
from zvllm.layers.layernorm import RMSNorm
from zvllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from zvllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from zvllm.models.qwen3 import Qwen3Attention, Qwen3MLP


class Qwen3MoeExpert(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        tp_group: "dist.ProcessGroup | None",
    ) -> None:
        super().__init__()
        # 单个专家：与 dense MLP 同构，但 linear 按"专家内 TP"组切分
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [moe_intermediate_size] * 2,
            bias=False,
            tp_group=tp_group,
        )
        self.down_proj = RowParallelLinear(
            moe_intermediate_size,
            hidden_size,
            bias=False,
            tp_group=tp_group,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.gate_up_proj(x))
        return self.down_proj(x)


class Qwen3MoeSparseMoeBlock(nn.Module):
    """
    Router + 专家。支持两个正交的并行维度（乘积等于全局 TP）：
    - 专家内 TP（moe_tp）：每个专家的 linear 在组内 rank 间切分，每 rank 累加
      全部本地专家的 partial 结果，层内一次组内 all_reduce 即完成专家计算；
    - 专家并行（moe_ep）：专家按组均分，每 rank 只算自己拥有的专家。
      "对专家加权和"是线性运算，全 world 一次 all_reduce 与 all-to-all 等价
      （省掉数据搬运，代价是通信量为 token 数的 topk 倍）。
    小批次（T <= FUSED_MAX_TOKENS）走 fused_moe 融合路径（grouped GEMM，零 host 同步）；
    大批次保留遍历式 forward（可读优先）：每专家一个 gather + GEMM + scatter 累加。
    """

    def __init__(
        self,
        config,
        moe_tp_size: int,
        moe_ep_size: int,
        tp_group: "dist.ProcessGroup | None",
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.topk = config.num_experts_per_tok
        self.tp_size = moe_tp_size
        self.ep_size = moe_ep_size
        self.tp_group = tp_group
        # 本 rank 属于 EP 组 (rank // moe_tp)，拥有连续一段全局专家号
        ep_rank = dist.get_rank() // moe_tp_size
        num_local = self.num_experts // moe_ep_size
        self.local_expert_ids = list(range(ep_rank * num_local, (ep_rank + 1) * num_local))
        # ModuleDict 保留全局专家号进参数名（mlp.experts.{e}.*），与 checkpoint 命名对齐
        self.experts = nn.ModuleDict({
            str(e): Qwen3MoeExpert(config.hidden_size, config.moe_intermediate_size, tp_group)
            for e in self.local_expert_ids
        })
        # router 全量复制在每个 rank：输入跨 rank 一致，top-k 结果天然一致，无需通信
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.local_start = self.local_expert_ids[0]
        # stacked 专家权重：w13 [E, 2I, H]（前 I 行 gate、后 I 行 up），w2 [E, H, I]。
        # 大 buffer 是权重的唯一存储：每个专家的 linear 权重把 param.data 重指向 buffer
        # 切片（view），loader 的 copy_ 经 view 直接写入大 buffer。融合路径（grouped GEMM）
        # 直接用 buffer：零拷贝、零额外显存——若首次 forward 才惰性 torch.cat，会完整复制
        # 一份专家权重（30B EP=2 约 +29GB/rank，必 OOM，2026-08-19 真机实测）；遍历路径
        # 则继续用 per-expert view 读同一份权重。
        first = self.experts[str(self.local_expert_ids[0])]
        gw, dw = first.gate_up_proj.weight, first.down_proj.weight
        num_local = len(self.local_expert_ids)
        self.w13 = torch.empty(num_local, *gw.shape, dtype=gw.dtype, device=gw.device)
        self.w2 = torch.empty(num_local, *dw.shape, dtype=dw.dtype, device=dw.device)
        for i, e in enumerate(self.local_expert_ids):
            expert = self.experts[str(e)]
            expert.gate_up_proj.weight.data = self.w13[i]
            expert.down_proj.weight.data = self.w2[i]

    def _forward_loop(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """遍历式实现：每专家一个 gather + GEMM + scatter 累加（含 host 同步，大批次 GEMM 足够大时无所谓）"""
        out = torch.zeros(x.shape[0], self.topk, x.shape[1], device=x.device, dtype=x.dtype)
        for e_id, expert in self.experts.items():
            mask = topk_ids == int(e_id)
            if not mask.any():
                continue
            idx = mask.nonzero()
            y = expert(x[idx[:, 0]])
            w = topk_weights[idx[:, 0], idx[:, 1]].to(x.dtype).unsqueeze(-1)
            out[idx[:, 0], idx[:, 1]] = y * w
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 路由（softmax → top-k → 归一化，与 Qwen3-MoE 参考实现一致）
        router_probs = torch.softmax(self.gate(x).float(), dim=-1)
        topk_weights, topk_ids = torch.topk(router_probs, self.topk, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        # 输出按 (token, slot) 组织，非本地专家的位置为 0，跨 rank 求和即 all_reduce
        T = x.shape[0]
        use_triton = T <= FUSED_MAX_TOKENS and x.is_cuda and triton_available()
        use_bmm = T <= BMM_MAX_TOKENS and not use_triton
        if use_triton or use_bmm:
            w13, w2 = self.w13, self.w2
            out = (fused_moe_triton if use_triton else fused_moe_bmm)(
                x, topk_weights, topk_ids, self.local_start, w13, w2
            )
        else:
            out = self._forward_loop(x, topk_weights, topk_ids)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        if self.ep_size > 1:
            dist.all_reduce(out)
        return out.sum(dim=1)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_id: int,
        moe_tp_size: int,
        moe_ep_size: int,
        tp_group: "dist.ProcessGroup | None",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        if layer_id < getattr(config, "first_k_dense_replace", 0):
            # 前 first_k_dense_replace 层是 dense MLP（HF Qwen3MoeConfig 缺省为 0，
            # 如 Qwen3-30B-A3B 全 48 层 MoE；235B 等在 config.json 中显式置 1）
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        else:
            self.mlp = Qwen3MoeSparseMoeBlock(config, moe_tp_size, moe_ep_size, tp_group)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config,
        moe_tp_size: int,
        moe_ep_size: int,
        tp_group: "dist.ProcessGroup | None",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, i, moe_tp_size, moe_ep_size, tp_group)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    has_moe = True

    def __init__(
        self,
        config,
        moe_tp_size: int,
        moe_ep_size: int,
        tp_group: "dist.ProcessGroup | None",
    ) -> None:
        super().__init__()
        # EP > 1 时 checkpoint 里有不属于本 rank 的专家权重，loader 需要跳过
        self.skip_unowned_weights = moe_ep_size > 1
        self.model = Qwen3MoeModel(config, moe_tp_size, moe_ep_size, tp_group)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
