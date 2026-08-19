"""SDPA attention 兜底后端纯 CPU 单测（无需 GPU/triton/flash_attn）。运行：python tests/test_attention.py

对拍对象：naive 右对齐因果 attention（float32 直接实现）。
覆盖：prefill 无 prefix、prefill 带 prefix（分页 gather + 右对齐 mask）、
decode（-1 padding、跨块 Lmax、GQA）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from zvllm.layers.attention import sdpa_prefill, sdpa_decode

torch.manual_seed(0)

H, HKV, D = 4, 2, 8      # GQA ratio = 2
SCALE = D ** -0.5


def repeat_kv(t, ratio, dim=1):
    return t if ratio == 1 else t.repeat_interleave(ratio, dim=dim)


def ref_attention(q, k, v):
    """naive 右对齐因果 attention（float32）。q: [Lq,H,D]，k,v: [Lk,H,D]（GQA 已展开）"""
    Lq, Lk = q.size(0), k.size(0)
    s = torch.einsum("lhd,mhd->hlm", q, k) * SCALE          # [H, Lq, Lk]
    cols = torch.arange(Lk)[None, :]
    rows = torch.arange(Lq)[:, None]
    s = s.masked_fill((cols > Lk - Lq + rows).unsqueeze(0), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hlm,mhd->lhd", p, v)               # [Lq, H, D]


def check(name, got, expect, atol=2e-4):
    got, expect = got.float(), expect.float()
    diff = (got - expect).abs().max().item()
    assert diff < atol, f"{name}: max diff {diff} > {atol}"
    print(f"  ok {name} (max diff {diff:.2e})")


def test_prefill_no_prefix():
    print("test_prefill_no_prefix")
    lens = [5, 3, 7]
    T = sum(lens)
    q = torch.randn(T, H, D)
    k = torch.randn(T, HKV, D)
    v = torch.randn(T, HKV, D)
    cu = torch.tensor([0, 5, 8, 15], dtype=torch.int32)
    got = sdpa_prefill(q, k, v, cu, cu, 0, None, None, None, H, HKV, SCALE)
    assert got.shape == (T, H, D), got.shape
    start = 0
    for i, L in enumerate(lens):
        ref = ref_attention(q[start:start + L], repeat_kv(k[start:start + L], H // HKV, 1),
                            repeat_kv(v[start:start + L], H // HKV, 1))
        check(f"seq{i} (L={L})", got[start:start + L], ref)
        start += L


def test_prefill_prefix():
    print("test_prefill_prefix")
    BS, nb = 4, 5
    k_cache = torch.randn(nb, BS, HKV, D)
    v_cache = torch.randn(nb, BS, HKV, D)
    # seq0: 上下文 6（块 0,1，块 0 前 4 个为 prefix），本次新算 2 个 token（位置 4,5 -> 块 1 偏移 0,1）
    # seq1: 上下文 3（块 3），全部为新 token
    new_k0, new_v0 = torch.randn(2, HKV, D), torch.randn(2, HKV, D)
    new_k1, new_v1 = torch.randn(3, HKV, D), torch.randn(3, HKV, D)
    k_cache[1, 0:2], v_cache[1, 0:2] = new_k0, new_v0      # 模拟 store_kvcache
    k_cache[3, 0:3], v_cache[3, 0:3] = new_k1, new_v1
    tables = torch.tensor([[0, 1, -1], [3, -1, -1]], dtype=torch.int32)
    q = torch.cat([torch.randn(2, H, D), torch.randn(3, H, D)])
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 6, 9], dtype=torch.int32)
    dummy = torch.zeros(0, HKV, D)
    got = sdpa_prefill(q, dummy, dummy, cu_q, cu_k, BS, k_cache, v_cache, tables, H, HKV, SCALE)
    assert got.shape == (5, H, D), got.shape
    ctx_k0 = torch.cat([k_cache[0, :4], new_k0], 0)
    ctx_v0 = torch.cat([v_cache[0, :4], new_v0], 0)
    check("seq0 (Lk=6, Lq=2)", got[0:2],
          ref_attention(q[0:2], repeat_kv(ctx_k0, H // HKV, 1), repeat_kv(ctx_v0, H // HKV, 1)))
    check("seq1 (Lk=3, Lq=3)", got[2:5],
          ref_attention(q[2:5], repeat_kv(new_k1, H // HKV, 1), repeat_kv(new_v1, H // HKV, 1)))


def test_decode():
    print("test_decode")
    BS, nb = 4, 3
    k_cache = torch.randn(nb, BS, HKV, D)
    v_cache = torch.randn(nb, BS, HKV, D)
    # seq0: 上下文 5 -> 块 [0,1]（块 1 只用 1 个 slot）；seq1: 上下文 2 -> 块 [2]，表尾 -1 padding
    tables = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
    context_lens = torch.tensor([5, 2], dtype=torch.int32)
    q = torch.randn(2, H, D)
    got = sdpa_decode(q, k_cache, v_cache, context_lens, tables, BS, H, HKV, SCALE)
    assert got.shape == (2, 1, H, D), got.shape
    ctx_k0 = torch.cat([k_cache[0], k_cache[1, :1]], 0)
    ctx_v0 = torch.cat([v_cache[0], v_cache[1, :1]], 0)
    check("seq0 (Lk=5, 跨块)", got[0, 0],
          ref_attention(q[0:1], repeat_kv(ctx_k0, H // HKV, 1), repeat_kv(ctx_v0, H // HKV, 1)))
    ctx_k1, ctx_v1 = k_cache[2, :2], v_cache[2, :2]
    check("seq1 (Lk=2, 短于 Lmax)", got[1, 0],
          ref_attention(q[1:2], repeat_kv(ctx_k1, H // HKV, 1), repeat_kv(ctx_v1, H // HKV, 1)))


def test_decode_single_seq():
    print("test_decode_single_seq")
    BS = 4
    k_cache = torch.randn(2, BS, HKV, D)
    v_cache = torch.randn(2, BS, HKV, D)
    tables = torch.tensor([[0, 1]], dtype=torch.int32)
    context_lens = torch.tensor([8], dtype=torch.int32)     # 正好占满 2 块
    q = torch.randn(1, H, D)
    got = sdpa_decode(q, k_cache, v_cache, context_lens, tables, BS, H, HKV, SCALE)
    assert got.shape == (1, 1, H, D), got.shape
    ctx_k, ctx_v = k_cache[0:2].reshape(2 * BS, HKV, D), v_cache[0:2].reshape(2 * BS, HKV, D)
    check("single (Lk=8, 对齐块边界)", got[0, 0],
          ref_attention(q[0:1], repeat_kv(ctx_k, H // HKV, 1), repeat_kv(ctx_v, H // HKV, 1)))


if __name__ == "__main__":
    test_prefill_no_prefix()
    test_prefill_prefix()
    test_decode()
    test_decode_single_seq()
    print("ALL PASSED")
