"""int8 KV cache + Triton paged decode GPU kernel 测试（服务器 W7900D ROCm 上运行）。

运行: CUDA_VISIBLE_DEVICES=2 python3 test_kv_gpu.py

测试项:
1. store_kvcache_int8  vs fp32 参照公式：位级一致（含 -1 slot 跳过不写）
2. triton_paged_decode int8 vs f64 naive attention（反量化 K/V）：隔离 kernel 误差
3. triton_paged_decode bf16 vs sdpa_decode / f64 naive：回归等价性
   并报告 int8 输出 vs bf16 输出的量化噪声量级
4. CUDA/HIP graph 捕获探测（Triton kernel 是否可被 torch.cuda.graph 捕获，决定 dense 模型 graph 可行性）

形状覆盖: (H,Hkv)=(16,2) ratio8 / (16,8) ratio2；D=128；BS=256；L=300/256/257/1/512（tail/整块/全 tail/多块）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from zvllm.layers.attention import (
    store_kvcache_int8,
    triton_paged_decode,
    sdpa_decode,
)

DEV = "cuda"
BS = 256
NB = 16          # 物理块数
LENSES = [300, 256, 257, 1, 512]
SHAPES = [(16, 2), (16, 8)]   # (H, Hkv): 30B tp4 / 0.6B
D = 128

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def ref_quantize(x_f32: torch.Tensor):
    """与 store_kvcache_int8_kernel 同公式，全程 fp32（GPU kernel 先 .to(tl.float32) 再运算）。
    逐行（每 token 一行，跨全部 KV head 展平）对称 int8：scale = max(|x|,1e-8)/127。"""
    scale = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.where(x_f32 >= 0, torch.floor(x_f32 / scale + 0.5), torch.ceil(x_f32 / scale - 0.5))
    return q.clamp(-127, 127).to(torch.int8), scale.squeeze(-1)


def make_tables(lenses):
    """block_tables[b, j] = b*3 + j（行间不重叠），剩余 -1 padding。"""
    bs = len(lenses)
    max_blk = max((L + BS - 1) // BS for L in lenses)
    tables = torch.full((bs, max_blk), -1, dtype=torch.int32)
    for b, L in enumerate(lenses):
        for j in range((L + BS - 1) // BS):
            tables[b, j] = b * 3 + j
    assert int(tables.max()) < NB, "tables overflow NB"
    return tables


def ref_decode_f64(q, kq, ksc, vq, vsc, tables, H, Hkv, sf):
    """f64 naive decode 参照：按块表 gather -> 反量化 -> GQA repeat -> softmax(qK^T*scale)V。"""
    q = q.cpu()
    ratio = H // Hkv
    out = torch.empty(len(LENSES), H, D, dtype=torch.float64)
    for b, L in enumerate(LENSES):
        nblk = (L + BS - 1) // BS
        k_parts, v_parts = [], []
        for j in range(nblk):
            bt = int(tables[b, j])
            off0 = j * BS
            ln = min(BS, L - off0)
            ksc_s = ksc[bt * BS: bt * BS + ln].view(ln, 1, 1)
            vsc_s = vsc[bt * BS: bt * BS + ln].view(ln, 1, 1)
            k_parts.append((kq[bt, :ln].float() * ksc_s).double())
            v_parts.append((vq[bt, :ln].float() * vsc_s).double())
        kf = torch.cat(k_parts).repeat_interleave(ratio, dim=1)   # [L, H, D]
        vf = torch.cat(v_parts).repeat_interleave(ratio, dim=1)
        qf = q[b].double()
        logits = torch.einsum('lhd,hd->lh', kf, qf) * sf                # [L, H]（sf 已是 1/sqrt(D)，与 kernel 的 scale 一致）
        out[b] = torch.einsum('lh,lhd->hd', torch.softmax(logits, dim=0), vf)
    return out


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def test_store_int8_bitexact():
    torch.manual_seed(0)
    N, Hkv, Dh = 97, 8, 128
    Dt = Hkv * Dh
    key = torch.randn(N, Hkv, Dh, dtype=torch.float32).to(torch.bfloat16)
    value = torch.randn(N, Hkv, Dh, dtype=torch.float32).to(torch.bfloat16)
    key, value = key.to(DEV), value.to(DEV)
    k_cache = torch.zeros(NB, BS, Hkv, Dh, dtype=torch.int8, device=DEV)
    v_cache = torch.zeros(NB, BS, Hkv, Dh, dtype=torch.int8, device=DEV)
    k_scale = torch.zeros(NB, BS, dtype=torch.float32, device=DEV)
    v_scale = torch.zeros(NB, BS, dtype=torch.float32, device=DEV)
    slots = torch.arange(N, dtype=torch.int32)
    slots = slots.to(DEV)
    slots[3] = -1    # -1 slot：kernel 应跳过，不写任何位置
    slots[50] = -1

    t0 = time.time()
    store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slots)
    torch.cuda.synchronize()
    dt = time.time() - t0

    qk, sk = ref_quantize(key.float().reshape(N, Dt).cpu())
    qv, sv = ref_quantize(value.float().reshape(N, Dt).cpu())
    kc = k_cache.view(NB * BS, Dt).cpu()
    vc = v_cache.view(NB * BS, Dt).cpu()
    ksc = k_scale.reshape(NB * BS).cpu()
    vsc = v_scale.reshape(NB * BS).cpu()

    ok = True
    for i in range(N):
        s = int(slots[i])
        if s == -1:
            if kc[i].any() or vc[i].any() or ksc[i] != 0 or vsc[i] != 0:
                ok = False
                print(f"  -1 slot {i} 被写入!", flush=True)
                break
            continue
        if not (torch.equal(kc[i], qk[i]) and torch.equal(ksc[i], sk[i])
                and torch.equal(vc[i], qv[i]) and torch.equal(vsc[i], sv[i])):
            ok = False
            print(f"  token {i} 位级不一致: k_q max diff={int((kc[i].long()-qk[i].long()).abs().max())}, "
                  f"scale diff={float(abs(ksc[i]-sk[i]))}", flush=True)
            break
    report("store_kvcache_int8 位级一致 (N=97, 2x -1 slot)", ok, f"{dt*1e3:.1f}ms")


def fill_caches(Hkv):
    """随机 fp32 K/V -> 参照公式量化 -> 填入 [NB,BS,Hkv,D] int8 cache + fp32 scale（独立于 store kernel）。"""
    torch.manual_seed(1)
    kf = torch.randn(NB * BS, Hkv, D, dtype=torch.float32)
    vf = torch.randn(NB * BS, Hkv, D, dtype=torch.float32)
    kq, ksc = ref_quantize(kf.reshape(NB * BS, Hkv * D))
    vq, vsc = ref_quantize(vf.reshape(NB * BS, Hkv * D))
    return (kf, vf,
            kq.reshape(NB, BS, Hkv, D).contiguous().to(DEV),
            vq.reshape(NB, BS, Hkv, D).contiguous().to(DEV),
            ksc.to(DEV), vsc.to(DEV))


def test_decode_int8():
    for H, Hkv in SHAPES:
        kf, vf, kq, vq, ksc, vsc = fill_caches(Hkv)
        torch.manual_seed(2)
        q = torch.randn(len(LENSES), H, D, dtype=torch.float32).to(torch.bfloat16).to(DEV)
        tables = make_tables(LENSES).to(DEV)
        lens = torch.tensor(LENSES, dtype=torch.int32, device=DEV)
        sf = D ** -0.5

        t0 = time.time()
        o = triton_paged_decode(q, kq, vq, ksc, vsc, lens, tables, H, Hkv, sf)
        torch.cuda.synchronize()
        dt = time.time() - t0

        ref = ref_decode_f64(q, kq.cpu(), ksc.cpu(), vq.cpu(), vsc.cpu(), tables.cpu(), H, Hkv, sf).to(DEV)
        e = rel_err(o.squeeze(1), ref)
        ok = e < 5e-3
        report(f"triton decode int8 (H={H},Hkv={Hkv},ratio={H//Hkv}) vs f64 naive", ok, f"rel={e:.2e} {dt*1e3:.1f}ms")
        if H == 16 and Hkv == 8:
            # int8 量化噪声量级：同形状 bf16 cache 跑一遍
            kb = kf.to(torch.bfloat16).reshape(NB, BS, Hkv, D).contiguous().to(DEV)
            vb = vf.to(torch.bfloat16).reshape(NB, BS, Hkv, D).contiguous().to(DEV)
            ob = triton_paged_decode(q, kb, vb, q, q, lens, tables, H, Hkv, sf)
            torch.cuda.synchronize()
            print(f"  [info] int8 vs bf16 输出相对差 (量化噪声量级): {rel_err(o.squeeze(1), ob.squeeze(1)):.2e}", flush=True)


def test_decode_bf16():
    for H, Hkv in SHAPES:
        torch.manual_seed(3)
        kf = torch.randn(NB * BS, Hkv, D, dtype=torch.float32)
        vf = torch.randn(NB * BS, Hkv, D, dtype=torch.float32)
        kb = kf.to(torch.bfloat16).reshape(NB, BS, Hkv, D).contiguous().to(DEV)
        vb = vf.to(torch.bfloat16).reshape(NB, BS, Hkv, D).contiguous().to(DEV)
        q = torch.randn(len(LENSES), H, D, dtype=torch.float32).to(torch.bfloat16).to(DEV)
        tables = make_tables(LENSES).to(DEV)
        lens = torch.tensor(LENSES, dtype=torch.int32, device=DEV)
        sf = D ** -0.5

        o_t = triton_paged_decode(q, kb, vb, q, q, lens, tables, H, Hkv, sf)
        o_s = sdpa_decode(q, kb, vb, lens, tables, BS, H, Hkv, sf)
        torch.cuda.synchronize()

        # f64 naive 参照用 cache 里的 bf16 值（与两条实现读到的数据一致）
        kq64 = kb.cpu().float()
        vq64 = vb.cpu().float()
        ones = torch.ones(NB * BS, dtype=torch.float32)
        ref = ref_decode_f64(q, kq64, ones, vq64, ones, tables.cpu(), H, Hkv, sf).to(DEV)
        e_t = rel_err(o_t.squeeze(1), ref)
        e_s = rel_err(o_s.squeeze(1), ref)
        ok = e_t < 2e-3 and e_s < 2e-3
        report(f"triton decode bf16 (H={H},Hkv={Hkv}) vs f64 naive / sdpa_decode", ok,
               f"triton rel={e_t:.2e}, sdpa rel={e_s:.2e}")


def test_graph_capture_probe():
    """探测：Triton kernel 能否被 torch.cuda.graph 捕获（ROCm hipGraph）。信息性测试，失败不算 FAIL。"""
    H, Hkv = 8, 4
    torch.manual_seed(4)
    kb = torch.randn(NB, BS, Hkv, D, dtype=torch.bfloat16, device=DEV)
    vb = torch.randn(NB, BS, Hkv, D, dtype=torch.bfloat16, device=DEV)
    q = torch.randn(2, H, D, dtype=torch.bfloat16, device=DEV)
    tables = make_tables([300, 256]).to(DEV)
    lens = torch.tensor([300, 256], dtype=torch.int32, device=DEV)
    sf = D ** -0.5

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            triton_paged_decode(q, kb, vb, q, q, lens, tables, H, Hkv, sf)
    torch.cuda.current_stream().wait_stream(stream)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = triton_paged_decode(q, kb, vb, q, q, lens, tables, H, Hkv, sf)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        o2 = o.clone()
        g.replay()
        torch.cuda.synchronize()
        e = rel_err(o2, o)
        print(f"[PROBE] CUDA graph 捕获 Triton decode: OK（replay 稳定, 两次输出 rel={e:.2e}）", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[PROBE] CUDA graph 捕获 Triton decode: FAIL -> {type(ex).__name__}: {ex}", flush=True)


def main():
    print(f"device: {torch.cuda.get_device_name(0)} | torch={torch.__version__}", flush=True)
    try:
        import triton
        print(f"triton: {triton.__version__}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"triton import failed: {ex}", flush=True)
        sys.exit(1)

    t0 = time.time()
    test_store_int8_bitexact()
    test_decode_int8()
    test_decode_bf16()
    test_graph_capture_probe()
    print(f"\n耗时 {time.time()-t0:.1f}s（含 triton 编译）", flush=True)
    print("ALL PASS" if all(RESULTS) else "HAS FAILURES", flush=True)
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
