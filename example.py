import os
import sys
from zvllm import LLM, SamplingParams
from zvllm.utils.model_download import resolve_model_path
from transformers import AutoTokenizer


def main():
    # hub 模型 ID：非本地目录时自动下载（默认先魔搭、后 HF；本地目录则直接使用）
    # 用法：python example.py [model]（本地目录或 hub 模型 ID，默认 Qwen/Qwen3-0.6B）
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    path = resolve_model_path(model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(model, enforce_eager=True, tensor_parallel_size=1)

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
