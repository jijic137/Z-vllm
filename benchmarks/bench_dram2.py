"""DRAM / L1 带宽复测（干净的数字，供 roofline 用）。
用法：cd /root/zvllm && HIP_VISIBLE_DEVICES=4 ~/zvllm-env/bin/python bench_dram2.py
"""
import statistics

import torch
import triton
import triton.language as tl

DEV = "cuda"


def t(fn, iters=5, warmup=2):
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


# ---- DRAM：16GB buffer，torch 高度优化的 copy/triad kernel ----
n = 8 * 1024**3  # 8Gi bf16 元素 = 16GB
a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
b = torch.randn(n, device=DEV, dtype=torch.bfloat16)
ms = t(lambda: a.copy_(b))
print(f"DRAM_COPY_GBPS={2 * n * 2 / (ms / 1e3) / 1e9:.0f}")
ms = t(lambda: a.add_(b))
print(f"DRAM_TRIAD_GBPS={3 * n * 2 / (ms / 1e3) / 1e9:.0f}")
del a, b
torch.cuda.empty_cache()

# ---- L1：每 program 私有 2KB 切片，小 BLOCK 避免寄存器溢出 ----


@triton.jit
def l1_read(a_ptr, out_ptr, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(iters):
        acc += tl.load(a_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, acc)


BLOCK = 1024  # 2KB bf16 / program
NPROG = 48
n = NPROG * BLOCK
a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
out = torch.empty(n, device=DEV, dtype=torch.float32)
ITERS = 40000
ms = t(lambda: l1_read[(NPROG,)](a, out, ITERS, BLOCK=BLOCK), iters=3, warmup=1)
print(f"BW_L1_READ_GBPS={n * 2 * ITERS / (ms / 1e3) / 1e9:.0f}")
print("DONE")
