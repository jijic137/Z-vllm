"""KV int8 vs bf16 引擎级 A/B（贪心生成 + 速度 + KV 块数 + token 级分歧）。

走真实引擎路径（scheduler + 分页 KV + 采样）。KV 量化只作用于 decode 步的分页读
（首个 prefill chunk 仍全精度），因此本脚本是 KV-int8 的核心质量 + 性能测试。
用法：HIP_VISIBLE_DEVICES=2 ~/zvllm-env/bin/python kv_ab.py <model_dir>
"""
import sys
import time

import torch

from zvllm import LLM, SamplingParams

PROMPTS = [
    "introduce yourself in one sentence",
    "list the first ten prime numbers",
    "用一句话解释什么是分页注意力",
    "write a haiku about the sea",
]
WARMUP = ["warmup only"]
MAX_TOKENS = 384    # 固定解码长度，ignore_eos 强制跑满，保证两配置工作量一致


def run_config(model_dir: str, kv_bits: int, port: int):
    torch.cuda.reset_peak_memory_stats()
    llm = LLM(model_dir, enforce_eager=True, tensor_parallel_size=1,
              kv_bits=kv_bits, master_port=port)
    n_blocks = llm.config.num_kvcache_blocks
    # 预热：触发 triton kernel 编译 / 分页缓存分配，避免计入计时
    llm.generate(WARMUP, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
                 use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    token_ids = [o["token_ids"] for o in outs]
    texts = [o["text"] for o in outs]
    n_tok = sum(len(t) for t in token_ids)
    llm.exit()
    torch.cuda.empty_cache()
    return dict(texts=texts, token_ids=token_ids, n_tok=n_tok, wall=t1 - t0,
                mem_gb=mem_gb, n_blocks=n_blocks)


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return len(a)    # 完全一致


def main():
    model_dir = sys.argv[1]
    torch.manual_seed(0)
    print(f"[KV-AB] model={model_dir}  prompts={len(PROMPTS)}  max_tokens={MAX_TOKENS} (贪心)", flush=True)
    bf = run_config(model_dir, 16, 2361)
    q8 = run_config(model_dir, 8, 2362)

    for name, r in (("KV-bf16", bf), ("KV-int8", q8)):
        print(f"\n[{name}] tokens={r['n_tok']}  wall={r['wall']:.2f}s  "
              f"throughput={r['n_tok']/r['wall']:.1f} tok/s  peak_mem={r['mem_gb']:.2f}GB  "
              f"kv_blocks={r['n_blocks']}", flush=True)
        for i, t in enumerate(r["texts"]):
            print(f"  [{i}] {t[:80]!r}")

    print("\n[逐 prompt token 级对比]（int8 允许与 bf16 出现少量分歧）", flush=True)
    for i, (a, b) in enumerate(zip(bf["token_ids"], q8["token_ids"])):
        m = min(len(a), len(b))
        match = sum(1 for x, y in zip(a, b) if x == y)
        fd = first_divergence(a, b)
        print(f"  [prompt {i}] len bf16={len(a)} int8={len(b)}  first_diverge={fd}  "
              f"match={match}/{m} ({100*match/max(1,m):.1f}%)")

    speedup = (bf["n_tok"] / bf["wall"]) / (q8["n_tok"] / q8["wall"])
    block_ratio = q8["n_blocks"] / bf["n_blocks"]
    print(f"\n[KV-AB] 吞吐 bf16 vs int8: {speedup:.2f}x（>1 表示 int8 更快）", flush=True)
    print(f"[KV-AB] kv_blocks int8/bf16 = {block_ratio:.2f}x（期望 ~2x，每 token KV 省一半）", flush=True)


if __name__ == "__main__":
    main()
