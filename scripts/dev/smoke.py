"""开发用冒烟脚本：加载任意受支持的模型并跑一次生成，用于验证各代码路径。

用法示例：
  python scripts/dev/smoke.py PrimeIntellect/qwen3-moe-tiny
  python scripts/dev/smoke.py Qwen/Qwen3-0.6B --no-eager --kv-bits 8 --spec ngram

与 example.py 的区别：默认不加 chat template（兼容 base 模型）、关闭进度条、
支持直接透传引擎开关，便于逐条验证新功能。
"""
import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")

from transformers import AutoTokenizer  # noqa: E402

from zvllm import LLM, SamplingParams  # noqa: E402
from zvllm.utils.model_download import resolve_model_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--ep", type=int, default=1, help="moe_ep_size")
    parser.add_argument("--weight-bits", type=int, default=16, choices=[8, 16])
    parser.add_argument("--kv-bits", type=int, default=16, choices=[8, 16])
    parser.add_argument("--spec", default="off", choices=["off", "ngram"])
    parser.add_argument("--no-eager", action="store_true", help="允许 CUDA graph / torch.compile")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        args.model,
        enforce_eager=not args.no_eager,
        tensor_parallel_size=args.tp,
        moe_ep_size=args.ep,
        weight_bits=args.weight_bits,
        kv_bits=args.kv_bits,
        spec_decode=args.spec,
        gpu_memory_utilization=args.gpu_mem,
    )
    prompts = [args.prompt, "1 + 1 ="]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    for prompt, output in zip(prompts, outputs):
        print(f"\n=== {prompt!r}\n--- {output['text']!r}")


if __name__ == "__main__":
    main()
