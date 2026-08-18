import os
from dataclasses import dataclass


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    moe_tp_size: int | None = None    # 专家内 TP，缺省 = tensor_parallel_size（纯 TP 模式）
    moe_ep_size: int = 1              # 专家并行 EP，需满足 moe_tp_size * moe_ep_size == tensor_parallel_size
    hf_config: "PretrainedConfig | None" = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    master_port: int = 2333              # NCCL 进程组初始化端口
    shm_name: str = "zvllm"              # 多卡方法调用的共享内存名
    shm_size: int = 2**20                # 共享内存大小（字节）
    max_graph_bs: int = 512              # CUDA Graph 捕获的最大 batch size

    def __post_init__(self):
        from transformers import AutoConfig

        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if getattr(self.hf_config, "model_type", "") == "qwen3_moe":
            self.moe_tp_size = self.moe_tp_size or self.tensor_parallel_size
            assert self.moe_tp_size * self.moe_ep_size == self.tensor_parallel_size, \
                f"moe_tp_size * moe_ep_size 必须等于 tensor_parallel_size（当前 {self.moe_tp_size} * {self.moe_ep_size} != {self.tensor_parallel_size}）"
            assert self.hf_config.num_experts % self.moe_ep_size == 0, \
                f"num_experts（{self.hf_config.num_experts}）必须能被 moe_ep_size（{self.moe_ep_size}）整除"
