"""用干净 kernel 重扫 IC 膝点（working set → 读带宽）。
用法：cd /root/zvllm && HIP_VISIBLE_DEVICES=4 ~/zvllm-env/bin/python bench_knee2.py
"""
import statistics

import torch
import triton
import triton.language as tl

DEV = "cuda"


def t(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


@triton.jit
def sweep_read(a_ptr, out_ptr, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(iters):
        acc += tl.load(a_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, acc)


BLOCK = 1024
for mb in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    n = mb * 524288  # bf16 元素
    a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    out = torch.empty(n, device=DEV, dtype=torch.float32)
    it = max(16, int(1073741824 // (n * 2)))  # ~1GB 流量
    ms = t(lambda: sweep_read[(n // BLOCK,)](a, out, it, BLOCK=BLOCK))
    gbs = n * 2 * it / (ms / 1e3) / 1e9
    print(f"KNEE2_MB={mb} GBPS={gbs:.0f}", flush=True)
    del a, out
    torch.cuda.empty_cache()
print("DONE")
