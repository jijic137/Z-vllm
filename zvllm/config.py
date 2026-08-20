import os
from glob import glob
from dataclasses import dataclass

from zvllm.models import SUPPORTED_MODEL_TYPES


def _checkpoint_has_qkv_bias(model_path: str) -> bool:
    """从 checkpoint 推断 LLaMA 家族注意力 qkv 是否带 bias。

    部分发行版的 config 里没有 attention_bias 字段（如 Qwen2-0.5B：config.json
    无此字段，但权重里 q/k/v bias 均非零），config 默认值无法区分"无 bias"与
    "有 bias 未声明"。这里只读 safetensors 头部元数据判断 q_proj.bias 键是否存在，
    不加载任何张量数据；主进程调用一次，多卡 worker 拿到的是已就绪的 Config。"""
    from safetensors import safe_open
    for file in sorted(glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            if any(k.endswith("self_attn.q_proj.bias") for k in f.keys()):
                return True
    return False


@dataclass(slots=True)
class Config:
    model: str
    model_source: str = "auto"    # model 非本地路径时的权重来源：auto / modelscope / hf
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
    weight_bits: int = 16                # 16=bf16（默认）；8=int8 加载时量化（per-group 128 对称，不支持 int4）
    kv_bits: int = 16                    # KV cache 精度：16=模型 dtype（默认）；8=int8 per-token 量化（不支持 int4）
    spec_decode: str = "off"             # 投机解码：off / ngram（序列自身历史的 n-gram 草稿，零训练）
    spec_gamma: int = 4                  # 每步最多草稿 token 数（验证步实际处理 1+gamma 个 token）
    spec_ngram: int = 4                  # n-gram 匹配长度：尾部 n 个 token 在自身历史中复现时取后续 token 作草稿

    def __post_init__(self):
        assert self.weight_bits in (8, 16), \
            (f"weight_bits 只支持 8（int8 加载时量化）或 16（bf16），"
             f"int4 当前版本不支持（当前 {self.weight_bits}）")
        assert self.kv_bits in (8, 16), \
            (f"kv_bits 只支持 8（int8 per-token KV 量化）或 16（模型 dtype），"
             f"int4 当前版本不支持（当前 {self.kv_bits}）")
        assert self.spec_decode in ("off", "ngram"), \
            f"spec_decode 必须是 off / ngram（当前 {self.spec_decode!r}）"
        if self.spec_decode == "ngram":
            assert 1 <= self.spec_gamma <= 8, f"spec_gamma 需在 1..8（当前 {self.spec_gamma}）"
            assert 2 <= self.spec_ngram <= 8, f"spec_ngram 需在 2..8（当前 {self.spec_ngram}）"
        from transformers import AutoConfig
        from zvllm.utils.model_download import resolve_model_path

        assert self.model_source in ("auto", "modelscope", "hf"), \
            f"model_source 必须是 auto / modelscope / hf（当前 {self.model_source}）"
        # 主进程解析/下载一次，多卡 worker 经 pickle 拿到的是已就绪的本地路径
        self.model = resolve_model_path(self.model, self.model_source)
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 早期校验模型家族，避免不支持的架构落到 worker 里才失败（或静默出错）
        model_type = getattr(self.hf_config, "model_type", "")
        assert model_type in SUPPORTED_MODEL_TYPES, \
            f"不支持的 model_type: {model_type!r}（当前支持 {', '.join(SUPPORTED_MODEL_TYPES)}）"
        if model_type in ("llama", "qwen2") and getattr(self.hf_config, "attention_bias", None) is None:
            # config 未声明 attention_bias（Qwen2-0.5B 发行版）：从 checkpoint 推断，
            # 避免加载时 qkv bias 参数缺失而崩溃（静默跳过非零 bias 会直接算错输出）
            self.hf_config.attention_bias = _checkpoint_has_qkv_bias(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if getattr(self.hf_config, "model_type", "") == "qwen3_moe":
            if self.moe_tp_size is None:
                # 未显式指定专家内 TP：自动推导（纯 EP 模式）；显式指定 moe_tp_size 则为混合 TP*EP
                assert self.tensor_parallel_size % self.moe_ep_size == 0, \
                    f"tensor_parallel_size（{self.tensor_parallel_size}）必须能被 moe_ep_size（{self.moe_ep_size}）整除"
                self.moe_tp_size = self.tensor_parallel_size // self.moe_ep_size
            assert self.moe_tp_size * self.moe_ep_size == self.tensor_parallel_size, \
                f"moe_tp_size * moe_ep_size 必须等于 tensor_parallel_size（当前 {self.moe_tp_size} * {self.moe_ep_size} != {self.tensor_parallel_size}）"
            assert self.hf_config.num_experts % self.moe_ep_size == 0, \
                f"num_experts（{self.hf_config.num_experts}）必须能被 moe_ep_size（{self.moe_ep_size}）整除"
