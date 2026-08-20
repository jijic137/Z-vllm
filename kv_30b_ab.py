"""30B-A3B EP=2 的 KV int8 vs bf16 A/B（验证 int8 KV 与 EP 多卡兼容 + 大模型行为）。

测：num_kvcache_blocks（每 rank）、物理 VRAM、tok/s、短贪心 A/B（首分歧 + 一致率）。
用法：HIP_VISIBLE_DEVICES=2,3 ~/zvllm-env/bin/python kv_30b_ab.py
"""
import re
import subprocess
import sys
import time

import torch

from zvllm import LLM, SamplingParams

MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B"
PROMPTS = [
    "introduce yourself in one sentence",
    "list the first ten prime numbers",
    "用一句话解释什么是专家并行",
    "write a haiku about the sea",
]
MAX_TOKENS = 128    # 短生成，验证 EP 兼容 + 质量信号，控制时间


def vram_gib():
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d+)\]\s*:\s*VRAM Total Used Memory \(B\):\s*(\d+)", line)
        if m:
            res[int(m.group(1))] = int(m.group(2)) / 2 ** 30
    return res


def run_config(kv_bits, port, shm):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=2, moe_ep_size=2,
              kv_bits=kv_bits, max_model_len=4096, master_port=port, shm_name=shm)
    t_load = time.perf_counter() - t0
    n_blocks = llm.config.num_kvcache_blocks
    print(f"[KV30B kv{kv_bits}] engine ready in {t_load:.0f}s  kv_blocks/rank={n_blocks}  "
          f"VRAM={vram_gib()}", flush=True)

    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
                 use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)
    t1 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    n = sum(len(o["token_ids"]) for o in outs)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    token_ids = [o["token_ids"] for o in outs]
    texts = [o["text"] for o in outs]
    print(f"[KV30B kv{kv_bits}] gen {n} tok in {t2 - t1:.2f}s = {n / (t2 - t1):.1f} tok/s  "
          f"peak_alloc={peak_gb:.2f}GB", flush=True)
    for i, t in enumerate(texts):
        print(f"  [{i}] {t[:110]!r}")
    llm.exit()
    torch.cuda.empty_cache()
    return dict(n_blocks=n_blocks, tok_per_s=n / (t2 - t1), token_ids=token_ids, texts=texts)


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else len(a)


def main():
    torch.manual_seed(0)
    print(f"[KV30B] EP=2 tp=2 prompts={len(PROMPTS)} max_tokens={MAX_TOKENS} (贪心)", flush=True)
    bf = run_config(16, 2351, "zkv30b16")
    q8 = run_config(8, 2352, "zkv30b8")

    print("\n[30B 逐 prompt token 级对比]", flush=True)
    for i, (a, b) in enumerate(zip(bf["token_ids"], q8["token_ids"])):
        m = min(len(a), len(b))
        match = sum(1 for x, y in zip(a, b) if x == y)
        fd = first_divergence(a, b)
        print(f"  [prompt {i}] len bf16={len(a)} int8={len(b)}  first_diverge={fd}  "
              f"match={match}/{m} ({100 * match / max(1, m):.1f}%)")

    ratio = q8["n_blocks"] / bf["n_blocks"]
    spd = bf["tok_per_s"] / q8["tok_per_s"]
    print(f"\n[KV30B] kv_blocks int8/bf16 = {ratio:.2f}x（期望 ~2x）", flush=True)
    print(f"[KV30B] 吞吐 bf16 vs int8 = {spd:.2f}x（>1 表示 int8 更快）", flush=True)


if __name__ == "__main__":
    main()
