"""模型注册：按 HF config 的 model_type 构建对应模型类。

新增模型家族时在这里登记：同构家族（如 llama / qwen2）可共用一个实现类。
"""
from zvllm.models.qwen3 import Qwen3ForCausalLM
from zvllm.models.qwen3_moe import Qwen3MoeForCausalLM
from zvllm.models.llama import LlamaForCausalLM

SUPPORTED_MODEL_TYPES = ("qwen3", "qwen3_moe", "llama", "qwen2")


def build_model(hf_config, engine_config, moe_tp_group=None):
    model_type = getattr(hf_config, "model_type", "")
    if model_type == "qwen3_moe":
        return Qwen3MoeForCausalLM(hf_config, engine_config.moe_tp_size, engine_config.moe_ep_size, moe_tp_group)
    if model_type == "qwen3":
        return Qwen3ForCausalLM(hf_config)
    if model_type in ("llama", "qwen2"):
        # 两家族架构同构（无 q/k norm 的 LLaMA 家族），共用 LlamaForCausalLM
        return LlamaForCausalLM(hf_config)
    raise ValueError(
        f"不支持的 model_type: {model_type!r}（当前支持 {', '.join(SUPPORTED_MODEL_TYPES)}）")
