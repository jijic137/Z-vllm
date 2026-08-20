"""int8 KV cache 纯 CPU 单测（无需 GPU/triton/flash_attn）。运行：python tests/test_kv_quant.py

对拍对象：
- 量化/反量化：与 store_kvcache_int8_kernel 同公式，验证往返误差性质
  （kernel 本体在 GPU 上与同公式参照做位级对拍，见服务器脚本；这里验证公式基础）
- _gather_context int8 分支：与手工反量化对拍
- sdpa_prefill int8 路径：与 naive 右对齐因果 attention（反量化后的 K/V）对拍
- Config.kv_bits 校验（int4 必须拒绝）
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from zvllm.layers.attention import _gather_context, sdpa_prefill

H, HKV, D = 4, 2, 8      # GQA ratio = 2
SCALE = D ** -0.5


def repeat_kv(t, ratio, dim=1):
    return t if ratio == 1 else t.repeat_interleave(ratio, dim=dim)


def quantize_per_token(x: torch.Tensor):
    """与 store_kvcache_int8_kernel 同公式：scale = max(|x|, 1e-8)/127，对称 int8 四舍五入。"""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.where(x >= 0, torch.floor(x / scale + 0.5), torch.ceil(x / scale - 0.5))
    return q.clamp(-127, 127).to(torch.int8), scale.squeeze(-1)


def dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.unsqueeze(-1)


def test_roundtrip():
    print("test_roundtrip")
    for mag in (1e-3, 1.0, 1e3):
        x = torch.randn(64, 128) * mag
        q, scale = quantize_per_token(x)
        assert q.dtype == torch.int8
        assert (q >= -127).all() and (q <= 127).all()
        err = (dequant(q, scale) - x).abs().amax(dim=-1)
        assert (err <= 0.5 * scale + 1e-6).all(), (err.max().item(), scale.min().item())
    x = torch.zeros(4, 16)    # 零向量：scale 落到下界，量化全 0、反量化全 0
    q, scale = quantize_per_token(x)
    assert (q == 0).all() and (dequant(q, scale) == 0).all()
    print("  ok roundtrip (|dequant-x| <= 0.5*scale，零向量安全)")


def test_gather_int8():
    print("test_gather_int8")
    nb, BS = 4, 8
    k_orig = torch.randn(nb * BS, HKV, D)
    v_orig = torch.randn(nb * BS, HKV, D)
    kq, ks = quantize_per_token(k_orig.reshape(-1, HKV * D))
    vq, vs = quantize_per_token(v_orig.reshape(-1, HKV * D))
    k_cache = kq.reshape(nb, BS, HKV, D)
    v_cache = vq.reshape(nb, BS, HKV, D)
    k_scale = ks.reshape(nb, BS)
    v_scale = vs.reshape(nb, BS)
    block_ids = torch.tensor([0, 1], dtype=torch.int32)
    total_len = 10    # 块 0 整块 8 + 块 1 前 2
    k, v = _gather_context(block_ids, k_cache, v_cache, total_len, BS, k_scale, v_scale, torch.float32)
    expect_k = dequant(kq, ks)[:total_len].reshape(total_len, HKV, D)
    expect_v = dequant(vq, vs)[:total_len].reshape(total_len, HKV, D)
    assert (k - expect_k).abs().max().item() < 1e-6
    assert (v - expect_v).abs().max().item() < 1e-6
    # bf16 路径（不传 scale）行为不变
    k2, v2 = _gather_context(block_ids, k_orig.reshape(nb, BS, HKV, D),
                             v_orig.reshape(nb, BS, HKV, D), total_len, BS)
    assert (k2 - k_orig.reshape(nb * BS, HKV, D)[:total_len]).abs().max().item() < 1e-6
    print("  ok gather int8（含 bf16 路径回归）")


def test_prefill_int8():
    print("test_prefill_int8")
    BS, nb = 4, 2
    k_orig = torch.randn(nb * BS, HKV, D)
    v_orig = torch.randn(nb * BS, HKV, D)
    kq, ks = quantize_per_token(k_orig.reshape(-1, HKV * D))
    vq, vs = quantize_per_token(v_orig.reshape(-1, HKV * D))
    k_cache = kq.reshape(nb, BS, HKV, D)
    v_cache = vq.reshape(nb, BS, HKV, D)
    k_scale = ks.reshape(nb, BS)
    v_scale = vs.reshape(nb, BS)
    # 单序列：上下文 Lk=5（块 0 整块 + 块 1 偏移 0），本次新算 Lq=2（新 token 已写入缓存槽位 4,5）
    tables = torch.tensor([[0, 1]], dtype=torch.int32)
    q = torch.randn(2, H, D)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 5], dtype=torch.int32)
    dummy = torch.zeros(0, HKV, D)
    got = sdpa_prefill(q, dummy, dummy, cu_q, cu_k, BS, k_cache, v_cache, tables,
                       H, HKV, SCALE, k_scale, v_scale)
    # 参照：反量化后的缓存 K/V + 右对齐因果
    ctx_k = repeat_kv(dequant(kq, ks)[:5].reshape(5, HKV, D), H // HKV, 1)
    ctx_v = repeat_kv(dequant(vq, vs)[:5].reshape(5, HKV, D), H // HKV, 1)
    Lq, Lk = 2, 5
    s = torch.einsum("lhd,mhd->hlm", q, ctx_k) * SCALE
    cols = torch.arange(Lk)[None, :]
    rows = torch.arange(Lq)[:, None]
    s = s.masked_fill((cols > Lk - Lq + rows).unsqueeze(0), float("-inf"))
    expect = torch.einsum("hlm,mhd->lhd", torch.softmax(s, dim=-1), ctx_v)
    diff = (got - expect).abs().max().item()
    assert diff < 2e-4, f"prefill int8 max diff {diff}"
    print(f"  ok prefill int8 (max diff {diff:.2e})")


def test_kv_bits_validation():
    print("test_kv_bits_validation")
    from zvllm.config import Config
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "config.json").write_text("{}")
        try:
            Config(td, kv_bits=4)
            raise AssertionError("kv_bits=4 应当被拒绝")
        except AssertionError as e:
            assert "kv_bits" in str(e), str(e)
    print("  ok kv_bits 校验（int4 被拒绝）")


if __name__ == "__main__":
    test_roundtrip()
    test_gather_int8()
    test_prefill_int8()
    test_kv_bits_validation()
    print("ALL PASSED")
