from zvllm.sampling_params import SamplingParams


def __getattr__(name):
    # 延迟导入 LLM：engine 子模块（scheduler/block_manager/sequence）
    # 不需要 torch/transformers 即可导入，便于纯 CPU 单元测试
    if name == "LLM":
        from zvllm.llm import LLM
        return LLM
    raise AttributeError(f"module 'zvllm' has no attribute {name!r}")
