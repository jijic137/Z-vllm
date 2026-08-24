# -*- coding: utf-8 -*-
"""MoE EP 并发 decode 吞吐 bench（T>1 区间：N 条并发贪心序列 × 每序列 64 token）。

回答的问题：T=1 时 EP 2/4/8 是平台期（9.74/9.49/9.58 tok/s），那么并发区间
（T=16/32/64）聚合吞吐随 N 怎么涨、EP scaling 曲线什么形状、平台期是否被
all_reduce 通信量（T×topk×H/层）击穿。

用法（服务器 8×W7900D，权重已缓存；单进程启动模型，勿用 torchrun）：
  python bench_moe_conc.py --tp 2 --ep 2 --n 1,16,32,64
  python bench_moe_conc.py --tp 8 --ep 8 --n 1,16,32,64

指标口径：
- decode 阶段单独计时（step() 的 num_tokens<0 步），与 T=1 bench（总墙钟/64，
  prefill 仅 ~15 token 可忽略）可比；
- per_stream_tps = tokens / decode 墙钟（单条流视角，直接与 T=1 基线对比）；
- agg_tps = n × per_stream_tps（聚合吞吐，continuous batching 的真实收益）；
- steady_ms 为 decode 步延迟的中位 50% 均值（去掉首尾波动）。
"""
import argparse
import os
import statistics
import time

from transformers import AutoTokenizer

from zvllm import LLM, SamplingParams
from zvllm.utils.model_download import resolve_model_path


def measure_once(llm, prompts, sp):
    for p in prompts:
        llm.add_request(p, sp)
    prefill_time = decode_time = 0.0
    decode_tokens = 0
    step_lats = []
    t_start = time.time()
    while not llm.is_finished():
        t0 = time.time()
        _outputs, num_tokens = llm.step()
        dt = time.time() - t0
        if num_tokens > 0:
            prefill_time += dt
        else:
            decode_time += dt
            decode_tokens += -num_tokens
            step_lats.append(dt)
    wall = time.time() - t_start
    lats = sorted(step_lats)
    lo, hi = len(lats) // 4, max(len(lats) // 4 + 1, 3 * len(lats) // 4)
    steady_ms = 1000 * sum(lats[lo:hi]) / max(1, hi - lo)
    return dict(wall=wall, prefill=prefill_time, decode=decode_time,
                dec_tokens=decode_tokens, steady_ms=steady_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--tp", type=int, default=None, help="缺省 = WORLD_SIZE")
    ap.add_argument("--ep", type=int, default=None, help="缺省 = tp（纯 EP）")
    ap.add_argument("--n", default="1,16,32,64", help="并发序列数，逗号分隔")
    ap.add_argument("--tokens", type=int, default=64, help="每序列最大生成 token")
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    tp = args.tp or world
    ep = args.ep or tp
    n_list = [int(x) for x in args.n.split(",") if x.strip()]

    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"introduce yourself (variant {i})"}],
            tokenize=False, add_generation_prompt=True)
        for i in range(max(n_list))
    ]

    llm = LLM(path, tensor_parallel_size=tp, moe_ep_size=ep, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)

    # 预热（含 triton 首编、融合路径一次性触发）
    llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=8),
                 use_tqdm=False)

    for n in n_list:
        results = [measure_once(llm, prompts[:n], sp) for _ in range(args.runs)]
        best = min(results, key=lambda r: r["decode"])
        per_stream = args.tokens / best["decode"]
        print(f"MOE_CONC: tp={tp} ep={ep} n={n} tokens={args.tokens} "
              f"wall={best['wall']:.2f}s prefill={best['prefill']:.2f}s "
              f"decode={best['decode']:.2f}s agg_tps={n * per_stream:.2f} "
              f"per_stream_tps={per_stream:.2f} steady_ms={best['steady_ms']:.1f} "
              f"all_decode={['%.2f' % r['decode'] for r in results]}",
              flush=True)
    print(f"MOE_CONC_DONE: tp={tp} ep={ep}", flush=True)


if __name__ == "__main__":
    main()
