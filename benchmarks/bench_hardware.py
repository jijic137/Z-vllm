"""W7900D (gfx1100) 硬件画像：各级带宽实测 + bf16 GEMM 峰值 + 项目 kernel roofline 数据点。

输出 KEY=VALUE 行，供 roofline 绘图使用。用法：
  cd /root/zvllm && HIP_VISIBLE_DEVICES=4 ~/zvllm-env/bin/python bench_hardware.py
"""
import statistics
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


def med_time(fn, iters=10, warmup=3):
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


def main():
    print(f"PYTORCH={torch.__version__} TRITON={triton.__version__}")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU_NAME={p.name}")
    print(f"GPU_SM={p.multi_processor_count}")
    print(f"GPU_MEM_GIB={p.total_memory / 2**30:.1f}")
    clk_khz = getattr(p, "clock_rate", 0)
    mclk_khz = getattr(p, "memory_clock_rate", 0)
    bus = getattr(p, "memory_bus_width", 0)
    print(f"GPU_CLK_KHZ={clk_khz} MEM_CLK_KHZ={mclk_khz} MEM_BUS_BITS={bus}")
    if mclk_khz and bus:
        theo_gbs = mclk_khz * 2 * (bus / 8) / 1e6
        print(f"DRAM_BW_THEORETICAL_GBPS={theo_gbs:.0f}")

    # ---------- 1. 各级带宽（Triton 流式 kernel） ----------
    @triton.jit
    def read_kernel(a_ptr, out_ptr, iters, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for _ in range(iters):
            acc += tl.load(a_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, acc)

    @triton.jit
    def r2w1_kernel(a_ptr, b_ptr, c_ptr, iters, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        for _ in range(iters):
            x = tl.load(a_ptr + offs).to(tl.float32)
            y = tl.load(b_ptr + offs).to(tl.float32)
            tl.store(c_ptr + offs, (x + y).to(tl.bfloat16))

    BLOCK = 16384  # 32KB bf16 / program

    def bw_read(mb, iters, nprog_cap=None):
        n = mb * 524288  # bf16 元素
        a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
        grid = n // BLOCK
        out = torch.empty(grid * BLOCK, device=DEV, dtype=torch.float32)
        ms = med_time(lambda: read_kernel[(grid,)](a, out, iters, BLOCK=BLOCK),
                      iters=5, warmup=2)
        gbs = (n * 2 * iters) / (ms / 1e3) / 1e9
        del a, out
        torch.cuda.empty_cache()
        return gbs

    def bw_r2w1(mb, iters):
        n = mb * 524288
        a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(n, device=DEV, dtype=torch.bfloat16)
        c = torch.empty(n, device=DEV, dtype=torch.bfloat16)
        grid = n // BLOCK
        ms = med_time(lambda: r2w1_kernel[(grid,)](a, b, c, iters, BLOCK=BLOCK),
                      iters=5, warmup=2)
        gbs = (n * 2 * 3 * iters) / (ms / 1e3) / 1e9
        del a, b, c
        torch.cuda.empty_cache()
        return gbs

    # L1：54 个 program 各持私有 32KB 切片（驻留本 CU L1）
    l1_mb = 54 * 32 / 1024  # ~1.69MB
    n = int(l1_mb * 524288)
    a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    out = torch.empty(n, device=DEV, dtype=torch.float32)
    ms = med_time(lambda: read_kernel[(54,)](a, out, 20000, BLOCK=BLOCK), iters=3, warmup=1)
    BW_L1 = (n * 2 * 20000) / (ms / 1e3) / 1e9
    print(f"BW_L1_READ_GBPS={BW_L1:.0f}")
    del a, out
    torch.cuda.empty_cache()

    BW_IC = bw_read(32, 200)
    print(f"BW_IC_READ_GBPS={BW_IC:.0f}")

    BW_DRAM_R = bw_read(2048, 4)
    BW_DRAM_RW = bw_r2w1(2048, 4)
    print(f"BW_DRAM_READ_GBPS={BW_DRAM_R:.0f}")
    print(f"BW_DRAM_2R1W_GBPS={BW_DRAM_RW:.0f}")

    # IC 容量膝点扫描
    knee = []
    for mb in (8, 16, 32, 64, 128, 256, 512):
        it = max(1, int(2048 / mb))  # 目标 ~2GB 流量
        gbs = bw_read(mb, it)
        knee.append((mb, gbs))
        print(f"KNEE_MB={mb} GBPS={gbs:.0f}", flush=True)

    # ---------- 2. bf16 GEMM 峰值与 M 扫描 ----------
    peaks = []
    for s in (2048, 4096, 8192, 16384):
        a = torch.randn(s, s, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(s, s, device=DEV, dtype=torch.bfloat16)
        ms = med_time(lambda: a @ b, iters=10, warmup=3)
        tflops = 2 * s**3 / (ms / 1e3) / 1e12
        peaks.append(tflops)
        print(f"GEMM_S={s} TFLOPS={tflops:.1f}", flush=True)
        del a, b
        torch.cuda.empty_cache()
    print(f"GEMM_PEAK_TFLOPS={max(peaks):.1f}")

    K = N = 2048
    b = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    for m in (1, 8, 32, 128, 512, 2048, 8192):
        a = torch.randn(m, K, device=DEV, dtype=torch.bfloat16)
        nrep = max(1, 512 // m)  # 小 M 多跑几次摊 launch
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            for _ in range(nrep):
                a @ b
        torch.cuda.synchronize(); s.record()
        for _ in range(nrep):
            a @ b
        e.record(); torch.cuda.synchronize()
        ms = s.elapsed_time(e) / nrep
        tflops = 2 * m * K * N / (ms / 1e3) / 1e12
        ai = 2 * m * K * N / (2 * (m * K + K * N + m * N))
        print(f"GEMM_M={m} TFLOPS={tflops:.3f} AI={ai:.3f}", flush=True)
        del a
        torch.cuda.empty_cache()

    # MoE 专家 GEMV（30B-A3B：hidden 2048, moe_inter 768）
    for k, n2, tag in ((2048, 1536, "GEMM1"), (768, 2048, "GEMM2")):
        a = torch.randn(1, k, device=DEV, dtype=torch.bfloat16)
        w = torch.randn(k, n2, device=DEV, dtype=torch.bfloat16)
        ms = med_time(lambda: a @ w, iters=50, warmup=10)
        tflops = 2 * k * n2 / (ms / 1e3) / 1e12
        ai = 2 * k * n2 / (2 * (k + k * n2 + n2))
        print(f"MOE_GEMV_{tag} TFLOPS={tflops:.4f} AI={ai:.3f} MS={ms:.4f}", flush=True)
        del a, w
        torch.cuda.empty_cache()

    # ---------- 3. 项目 decode attention kernel（30B-A3B / 0.6B 维度） ----------
    from zvllm.layers.attention import triton_paged_decode
    BS = 256
    scale = 1.0 / (128 ** 0.5)

    def attn_case(H, Hkv, D, L, int8, tag):
        nb = (L + BS - 1) // BS + 1
        if int8:
            kc = torch.randint(-127, 127, (nb, BS, Hkv, D), device=DEV, dtype=torch.int8)
            vc = torch.randint(-127, 127, (nb, BS, Hkv, D), device=DEV, dtype=torch.int8)
            ks = torch.rand(nb, BS, device=DEV, dtype=torch.float32) * 0.01 + 0.005
            vs = torch.rand(nb, BS, device=DEV, dtype=torch.float32) * 0.01 + 0.005
        else:
            kc = torch.randn(nb, BS, Hkv, D, device=DEV, dtype=torch.bfloat16)
            vc = torch.randn(nb, BS, Hkv, D, device=DEV, dtype=torch.bfloat16)
            ks = torch.empty(0, device=DEV)
            vs = torch.empty(0, device=DEV)
        q = torch.randn(1, H, D, device=DEV, dtype=torch.bfloat16)
        cl = torch.tensor([L], device=DEV, dtype=torch.int32)
        nblk = (L + BS - 1) // BS
        bt = torch.arange(nblk, device=DEV, dtype=torch.int32).reshape(1, -1)

        def run():
            return triton_paged_decode(q, kc, vc, ks, vs, cl, bt, H, Hkv, scale)

        ms = med_time(run, iters=20, warmup=5)
        flops = 4 * L * D * H          # QK + PV
        kv_bytes = 2 * L * Hkv * D * (1 if int8 else 2)
        bytes_total = kv_bytes + (L * H * D + 1 * H * D) * 2
        tflops = flops / (ms / 1e3) / 1e12
        ai = flops / bytes_total
        print(f"ATTN_{tag} L={L} TFLOPS={tflops:.4f} AI={ai:.3f} MS={ms:.4f} "
              f"KV_GBPS={kv_bytes / (ms / 1e3) / 1e9:.0f}", flush=True)
        del kc, vc, q
        torch.cuda.empty_cache()

    for L in (256, 1024, 4096):
        attn_case(32, 4, 128, L, False, "Q3_30B_BF16")
        attn_case(32, 4, 128, L, True, "Q3_30B_INT8")
    for L in (256, 1024, 4096):
        attn_case(16, 8, 128, L, False, "Q3_0P6B_BF16")
        attn_case(16, 8, 128, L, True, "Q3_0P6B_INT8")

    # ---------- 4. int8 / fp8 GEMM 硬件能力探针（W8A8 判据） ----------
    @triton.jit
    def gemm_i8(a_ptr, b_ptr, c_ptr, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        grid_n = N // BN
        pm, pn = pid // grid_n, pid % grid_n
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
        b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k in range(0, K, BK):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
            a_ptrs += BK
            b_ptrs += BK * N
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)

    S = 2048
    ai8 = torch.randint(-127, 127, (S, S), device=DEV, dtype=torch.int8)
    bi8 = torch.randint(-127, 127, (S, S), device=DEV, dtype=torch.int8)
    ci8 = torch.empty(S, S, device=DEV, dtype=torch.int32)
    try:
        ms = med_time(lambda: gemm_i8[(S // 64 * S // 64,)](
            ai8, bi8, ci8, S, S, S, BM=64, BN=64, BK=32), iters=20, warmup=5)
        tflops = 2 * S**3 / (ms / 1e3) / 1e12
        print(f"INT8_GEMM_TFLOPS={tflops:.1f}", flush=True)
    except Exception as ex:
        print(f"INT8_GEMM_ERR={type(ex).__name__}: {str(ex)[:200]}", flush=True)
    del ai8, bi8, ci8
    torch.cuda.empty_cache()

    abf = torch.randn(S, S, device=DEV, dtype=torch.bfloat16)
    bbf = torch.randn(S, S, device=DEV, dtype=torch.bfloat16)
    ms_bf = med_time(lambda: abf @ bbf, iters=20, warmup=5)
    print(f"BF16_GEMM_2048_TFLOPS={2 * S**3 / (ms_bf / 1e3) / 1e12:.1f}", flush=True)

    fp8_err = None
    for dt_name in ("float8_e4m3fnuz", "float8_e4m3fn"):
        dt = getattr(torch, dt_name, None)
        if dt is None:
            continue
        try:
            a8 = torch.randn(S, S, device=DEV).to(dt)
            b8 = torch.randn(S, S, device=DEV).to(dt)
            one = torch.ones((), device=DEV, dtype=torch.float32)
            out = torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one,
                                   out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            ms = med_time(lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one,
                                                   out_dtype=torch.bfloat16),
                          iters=20, warmup=5)
            print(f"FP8_GEMM_TFLOPS={2 * S**3 / (ms / 1e3) / 1e12:.1f} ({dt_name})", flush=True)
            break
        except Exception as ex:
            fp8_err = f"{dt_name}: {type(ex).__name__}: {str(ex)[:160]}"
    if fp8_err:
        print(f"FP8_GEMM_ERR={fp8_err}", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()
