from zvllm import LLM, SamplingParams
from zvllm.utils.model_download import resolve_model_path
from transformers import AutoTokenizer


def main():
    # hub 模型 ID：非本地目录时自动下载（默认先魔搭、后 HF；本地目录则直接使用）
    model = "Qwen/Qwen3-30B-A3B"
    path = resolve_model_path(model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    # MoE 两种典型并行方式（moe_tp_size * moe_ep_size 必须等于卡数）：
    # 1) 纯专家内 TP：每张卡持有每个专家的 1/N
    # llm = LLM(model, tensor_parallel_size=8)
    # 2) 纯 EP：每张卡完整持有 N_experts/N 个专家（小 GEMM 更少，推荐）
    llm = LLM(model, tensor_parallel_size=8, moe_ep_size=8)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
