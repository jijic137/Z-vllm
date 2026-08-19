"""MoE EP decode 吞吐 bench（README「已验证」口径：贪心 64 token，总墙钟 / 64）。

用法（服务器 8×W7900D，权重已缓存于 ~/.cache/modelscope）：
  python bench_moe_ep.py --tp 2 --ep 2     # 纯 EP=2（每卡 64 专家）
  python bench_moe_ep.py --tp 4 --ep 4     # 纯 EP=4（每卡 32 专家）
  python bench_moe_ep.py --tp 8 --ep 8     # 纯 EP=8（每卡 16 专家）
  python bench_moe_ep.py --tp 8 --ep 4     # 4 个 EP 组 × 专家内 TP=2

注意：引擎是单进程启动模型——主进程为 rank0 并自行 spawn tp-1 个 worker 进程，
不要用 torchrun（每个 rank 进程都会再 spawn 一组 worker，两组进程抢同一个
master_port 的 TCPStore 而死锁，2026-08-19 真机踩坑验证）。

MoE 模型在 model_runner 中恒为 enforce_eager（CUDA graph 暂不支持 MoE），
因此本脚本显式传 enforce_eager=True 只是把隐式行为写明。
"""
import argparse
import os
import time

from transformers import AutoTokenizer

from zvllm import LLM, SamplingParams
from zvllm.utils.model_download import resolve_model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--tp", type=int, default=None, help="缺省 = WORLD_SIZE")
    ap.add_argument("--ep", type=int, default=None, help="缺省 = tp（纯 EP）")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    tp = args.tp or world
    ep = args.ep or tp

    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "introduce yourself"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(path, tensor_parallel_size=tp, moe_ep_size=ep, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens)

    # 预热（含融合权重一次性构建）
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))

    times, outputs = [], None
    for _ in range(args.runs):
        t0 = time.time()
        outputs = llm.generate([prompt], sp)
        times.append(time.time() - t0)
    best = min(times)
    print(f"OUTPUT: {outputs[0]['text']!r}")
    print(f"MOE_BENCH: tp={tp} ep={ep} tokens={args.tokens} "
          f"best={best:.2f}s tok/s={args.tokens / best:.2f} all={['%.2f' % t for t in times]}")


if __name__ == "__main__":
    main()
