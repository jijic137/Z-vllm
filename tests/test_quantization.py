"""Weight-only W8 量化纯 CPU 单测（无需 GPU/triton/flash_attn）。运行：python tests/test_quantization.py

覆盖：
- quantize/dequant 往返：逐元素误差上界（量化步长 + bf16 scale 舍入 + 乘积 bf16 舍入），
  全零组无 NaN/Inf 且反量化严格为 0
- 分片一致性：Column（切 out 维）/ Row（切 in 维）本地分片量化 == 全局量化结果切片
  ——"先切分片再量化"的数学基础
- 量化 linear forward（CPU 物化兜底路径）与 F.linear(x, dequant) 逐位一致，
  与原 bf16 权重的误差在界内
- QKV / MergedColumn 子区域分片加载（回归：_load_quantized 必须写 narrowed 区域，
  不能整参覆盖）
- Row in 维整组对齐断言：in=768（6 组）在 tp=4 下启动即拒绝、tp=2 接受
- Config weight_bits：int4 在触碰 model 路径之前被拒；默认 16
- fused_moe_bmm 量化路径 vs 手工反量化参照（对拍闭环：参照不复用 dequantize_weight）
- 模型接线：Qwen3 dense 的 int8/scale 参数计数与 embed/lm_head 保持原 dtype；
  MoE 块 scale stacked buffer 的 view 重指向 + 量化 forward vs 手工参照
"""
import dataclasses
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import torch.distributed as dist

import zvllm.layers.linear as linear_mod
from zvllm.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear,
)
from zvllm.layers.fused_moe import fused_moe_bmm
from zvllm.quantization import WEIGHT_GROUP, quantize_weight, dequantize_weight


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


if not dist.is_initialized():
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{free_port()}",
                            world_size=1, rank=0)

torch.manual_seed(0)


def manual_dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """手工反量化（float64 中间量，不依赖 quantization.dequantize_weight，保持对拍独立）。"""
    shape = q.shape
    q_g = q.to(torch.float64).reshape(*shape[:-1], shape[-1] // WEIGHT_GROUP, WEIGHT_GROUP)
    s_g = s.to(torch.float64).unsqueeze(-1).expand_as(q_g)
    return (q_g * s_g).reshape(shape).to(torch.bfloat16)


def test_roundtrip():
    print("往返误差界")
    w = (torch.randn(128, 1024) * 0.3).bfloat16()
    q, s = quantize_weight(w)
    assert q.dtype == torch.int8 and q.shape == w.shape
    assert s.dtype == torch.bfloat16 and s.shape == (128, 1024 // WEIGHT_GROUP)
    assert q.abs().max().item() <= 127
    w_hat = dequantize_weight(q, s).to(torch.float64)
    w_d = w.to(torch.float64)
    s_d = s.to(torch.float64)
    err = (w_d - w_hat).abs()
    # 逐元素上界：量化步长 s/2 + clamp 极端（s bf16 舍入使 w/s 略超 127）≤ 127·2^-9·s
    # + 反量化乘积的 bf16 舍入 ≤ 127·2^-9·s
    bound = (s_d.unsqueeze(-1).expand(-1, -1, WEIGHT_GROUP).reshape(w_d.shape)
             * (0.5 + 2 * 127 * 2 ** -9) + 1e-3)
    assert (err <= bound).all(), f"往返逐元素误差超界: {err.max().item()}"
    # 平均误差应是量化步长量级（均匀舍入期望 s/4 + 少量乘积舍入）
    assert err.mean().item() <= s_d.mean().item() * 0.6, \
        f"平均误差 {err.mean().item():.2e} 超 0.6*mean(s)={s_d.mean().item() * 0.6:.2e}"
    print(f"  max {err.max().item():.2e}, mean {err.mean().item():.2e}"
          f" (mean s {s_d.mean().item():.2e}): OK")


def test_zero_group():
    print("全零组保护")
    w = torch.zeros(4, 128, dtype=torch.bfloat16)
    w[1, :64] = (torch.randn(64) * 0.3).bfloat16()
    q, s = quantize_weight(w)
    assert torch.isfinite(s.float()).all()
    for r in (0, 2, 3):
        assert (s[r] == 1.0).all(), f"全零组 {r} 的 scale 应为保护值 1.0"
        assert (q[r] == 0).all()
    w_hat = dequantize_weight(q, s)
    assert torch.equal(w_hat[0], w[0]), "全零组反量化必须严格为 0"
    print("  无 NaN/Inf，全零组反量化严格 0: OK")


def test_shard_equivalence():
    print("分片一致性（先切再量化 == 全局量化后切）")
    W = (torch.randn(512, 1024) * 0.3).bfloat16()
    q_full, s_full = quantize_weight(W)
    for rank in range(4):    # Column：切 out 维，组天然完整
        shard = W.narrow(0, rank * 128, 128)
        q_l, s_l = quantize_weight(shard)
        assert torch.equal(q_l, q_full.narrow(0, rank * 128, 128))
        assert torch.equal(s_l, s_full.narrow(0, rank * 128, 128))
    for rank in range(2):    # Row：切 in 维（每 rank 512 = 4 个整组）
        off = rank * 512
        shard = W.narrow(1, off, 512)
        q_l, s_l = quantize_weight(shard)
        assert torch.equal(q_l, q_full.narrow(1, off, 512))
        assert torch.equal(s_l, s_full.narrow(1, off // WEIGHT_GROUP, 512 // WEIGHT_GROUP))
    print("  Column(out 切分) / Row(in 切分) 本地 == 全局切片: OK")


def test_linear_forward():
    print("量化 linear forward（CPU 物化兜底路径）")
    in_size, out_size = 1024, 512
    W = (torch.randn(out_size, in_size) * 0.3).bfloat16()
    x = (torch.randn(7, in_size) * 0.5).bfloat16()
    q, s = quantize_weight(W)
    W_hat = dequantize_weight(q, s)
    ref_exact = F.linear(x, W_hat)
    ref_orig = F.linear(x, W)

    col = ColumnParallelLinear(in_size, out_size, quantized=True)
    col.weight_loader(col.weight, W)
    assert torch.equal(col.weight.data, q)
    assert torch.equal(col.weight_scale.data, s)
    got = col(x)
    assert torch.equal(got, ref_exact), "量化 forward 必须与 F.linear(x, dequant) 逐位一致"
    err = (got.float() - ref_orig.float()).abs().max().item()
    assert err < 0.3, f"量化 vs 原 bf16 权重误差 {err} 过大"
    print(f"  Column: 与 dequant 参照逐位一致，与原权重 max diff {err:.2e}: OK")

    row = RowParallelLinear(in_size, out_size, quantized=True)
    row.weight_loader(row.weight, W)
    assert torch.equal(row(x), ref_exact)
    print("  Row: 与 dequant 参照逐位一致: OK")


def test_packed_shard_loading():
    print("QKV / MergedColumn 子区域分片加载（回归）")
    hidden, head, n_heads, n_kv = 512, 64, 8, 4
    Wq = (torch.randn(n_heads * head, hidden) * 0.3).bfloat16()
    Wk = (torch.randn(n_kv * head, hidden) * 0.3).bfloat16()
    Wv = (torch.randn(n_kv * head, hidden) * 0.3).bfloat16()

    qkv = QKVParallelLinear(hidden, head, n_heads, n_kv, quantized=True)
    qkv.weight_loader(qkv.weight, Wq, "q")
    qkv.weight_loader(qkv.weight, Wk, "k")
    qkv.weight_loader(qkv.weight, Wv, "v")
    W_full = torch.cat([Wq, Wk, Wv], dim=0)
    q_full, s_full = quantize_weight(W_full)
    assert torch.equal(qkv.weight.data, q_full), "QKV 子区域量化 != 全局量化切片"
    assert torch.equal(qkv.weight_scale.data, s_full)
    x = (torch.randn(5, hidden) * 0.5).bfloat16()
    assert torch.equal(qkv(x), F.linear(x, dequantize_weight(q_full, s_full)))
    print(f"  QKV [{n_heads}q/{n_kv}kv]: 子区域加载 == 全局量化，forward 逐位一致: OK")

    inter = 256
    Wg = (torch.randn(inter, hidden) * 0.3).bfloat16()
    Wu = (torch.randn(inter, hidden) * 0.3).bfloat16()
    mlp = MergedColumnParallelLinear(hidden, [inter] * 2, quantized=True)
    mlp.weight_loader(mlp.weight, Wg, 0)
    mlp.weight_loader(mlp.weight, Wu, 1)
    W_full2 = torch.cat([Wg, Wu], dim=0)
    q_full2, s_full2 = quantize_weight(W_full2)
    assert torch.equal(mlp.weight.data, q_full2), "gate/up 子区域量化 != 全局量化切片"
    assert torch.equal(mlp.weight_scale.data, s_full2)
    print("  MergedColumn gate/up: 子区域加载 == 全局量化: OK")


def test_row_group_alignment_assert():
    print("Row in 维整组对齐断言")
    orig = linear_mod._tp_rank_size
    linear_mod._tp_rank_size = lambda g: (0, 4)
    try:
        try:
            RowParallelLinear(768, 512, quantized=True)
            raise SystemExit("in=768（6 组）tp=4 应被拒绝")
        except AssertionError as e:
            assert "整除" in str(e), f"断言文案不符: {e}"
    finally:
        linear_mod._tp_rank_size = orig
    linear_mod._tp_rank_size = lambda g: (0, 2)
    try:
        RowParallelLinear(768, 512, quantized=True)    # 768/2=384=3 整组，合法
    finally:
        linear_mod._tp_rank_size = orig
    print("  tp=4（6 组不可分）拒绝，tp=2（3 组）接受: OK")


def test_config_weight_bits():
    print("Config weight_bits 校验")
    from zvllm.config import Config
    try:
        Config(model="假路径仅用于触发 assert", weight_bits=4)
        raise SystemExit("weight_bits=4 应被拒绝")
    except AssertionError as e:
        assert "int4" in str(e), f"断言文案不符: {e}"
    defaults = {f.name: f.default for f in dataclasses.fields(Config)}
    assert defaults["weight_bits"] == 16
    print("  int4 在触碰 model 路径前被拒，默认 16: OK")


def test_fused_moe_bmm_quantized():
    print("fused_moe_bmm 量化路径 vs 手工反量化参照")
    H, I, E, K, T = 128, 128, 4, 2, 8
    x = (torch.randn(T, H) * 0.5).bfloat16()
    W13 = torch.randn(E, 2 * I, H) * 0.25
    W2 = torch.randn(E, H, I) * 0.25
    q13, s13 = quantize_weight(W13.bfloat16())
    q2, s2 = quantize_weight(W2.bfloat16())
    w13_ref = manual_dequant(q13, s13)
    w2_ref = manual_dequant(q2, s2)
    ids = torch.randint(0, E, (T, K))
    wts = torch.rand(T, K)
    wts = wts / wts.sum(dim=-1, keepdim=True)

    out = fused_moe_bmm(x, wts, ids, 0, q13, q2, s13, s2)
    ref = torch.zeros(T, K, H)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            gu = x[t].to(torch.float64) @ w13_ref[e].to(torch.float64).t()
            g, u = gu[:I], gu[I:]
            h = g * torch.sigmoid(g) * u
            y = h @ w2_ref[e].to(torch.float64).t()
            ref[t, k] = y * wts[t, k]
    err = (out.sum(1).to(torch.float64) - ref.sum(1)).abs().max().item()
    assert err < 0.2, f"bmm 量化路径 vs 手工参照 diff {err} 过大"
    out2 = fused_moe_bmm(x, wts, ids, 0, q13, q2, s13, s2)
    assert torch.equal(out, out2), "bmm 量化路径不确定"
    try:
        fused_moe_bmm(x, wts, ids, 0, q13, q2)
        raise SystemExit("int8 权重未传 scales 应被拒绝")
    except AssertionError as e:
        assert "scale" in str(e)
    print(f"  max diff {err:.2e} < 0.2，确定性 OK，缺 scales 拒绝: OK")


def test_model_wiring():
    print("模型接线（Qwen3 dense + Qwen3 MoE 块）")
    from transformers import Qwen3Config as HFQwen3Config
    from zvllm.models.qwen3 import Qwen3ForCausalLM
    from zvllm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)    # 与 model_runner 构造期口径一致
    try:
        hf = HFQwen3Config(
            vocab_size=2048, hidden_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=256, head_dim=64, rms_norm_eps=1e-6,
            max_position_embeddings=512,
        )
        model = Qwen3ForCausalLM(hf, quantized=True)
        int8 = [n for n, p in model.named_parameters() if p.dtype == torch.int8]
        scales = [n for n, p in model.named_parameters() if n.endswith("weight_scale")]
        # 2 层 × (qkv, o, gate_up, down) = 8 个 int8 权重 + 8 个 scale
        assert len(int8) == 8, f"int8 参数 {len(int8)} != 8: {int8}"
        assert len(scales) == 8, f"scale 参数 {len(scales)} != 8: {scales}"
        assert model.model.embed_tokens.weight.dtype == torch.bfloat16
        assert model.lm_head.weight.dtype == torch.bfloat16
        print(f"  Qwen3 dense: {len(int8)} int8 + {len(scales)} scale，embed/lm_head 保持 bf16: OK")

        # MoE 块：scale stacked buffer 重指向 + loader 写入 + 量化 forward
        mc = SimpleNamespace(hidden_size=256, moe_intermediate_size=128,
                             num_experts=4, num_experts_per_tok=2)
        block = Qwen3MoeSparseMoeBlock(mc, moe_tp_size=1, moe_ep_size=1,
                                       tp_group=None, quantized=True)
        torch.manual_seed(11)
        loaded = {}
        for e in block.local_expert_ids:
            Wg = (torch.randn(128, 256) * 0.25).bfloat16()
            Wu = (torch.randn(128, 256) * 0.25).bfloat16()
            Wd = (torch.randn(256, 128) * 0.25).bfloat16()
            loaded[e] = (Wg, Wu, Wd)
            expert = block.experts[str(e)]
            expert.gate_up_proj.weight_loader(expert.gate_up_proj.weight, Wg, 0)
            expert.gate_up_proj.weight_loader(expert.gate_up_proj.weight, Wu, 1)
            expert.down_proj.weight_loader(expert.down_proj.weight, Wd)
        # 大 buffer 内容 == 各专家全局量化；scale view 重指向生效
        for i, e in enumerate(block.local_expert_ids):
            Wg, Wu, Wd = loaded[e]
            q13_e, s13_e = quantize_weight(torch.cat([Wg, Wu], dim=0))
            q2_e, s2_e = quantize_weight(Wd)
            assert torch.equal(block.w13[i], q13_e), f"专家 {e} w13 buffer 与量化结果不符"
            assert torch.equal(block.w13_scale[i], s13_e)
            assert torch.equal(block.w2[i], q2_e)
            assert torch.equal(block.w2_scale[i], s2_e)
        first = block.experts[str(block.local_expert_ids[0])]
        assert first.gate_up_proj.weight_scale.data.data_ptr() == block.w13_scale[0].data_ptr()
        assert first.down_proj.weight_scale.data.data_ptr() == block.w2_scale[0].data_ptr()
        # forward vs 手工参照（路由与块内同式）
        block.gate.weight.data = (torch.randn(4, 256) * 0.5).bfloat16()
        x = (torch.randn(8, 256) * 0.5).bfloat16()
        got = block(x)
        probs = torch.softmax(block.gate(x).float(), dim=-1)
        tw, ti = torch.topk(probs, 2, dim=-1)
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
        ref = torch.zeros(8, 2, 256)
        for t in range(8):
            for k in range(2):
                e = int(ti[t, k])
                gu = x[t].to(torch.float64) @ manual_dequant(
                    block.w13[e], block.w13_scale[e]).to(torch.float64).t()
                g, u = gu[:128], gu[128:]
                h = g * torch.sigmoid(g) * u
                y = h @ manual_dequant(block.w2[e], block.w2_scale[e]).to(torch.float64).t()
                ref[t, k] = y * tw[t, k]
        err = (got.to(torch.float64) - ref.sum(1)).abs().max().item()
        assert err < 0.2, f"MoE 块量化 forward vs 手工参照 diff {err} 过大"
        print(f"  MoE 块: scale buffer 重指向/内容一致，forward max diff {err:.2e} < 0.2: OK")
    finally:
        torch.set_default_dtype(default_dtype)


def main():
    test_roundtrip()
    test_zero_group()
    test_shard_equivalence()
    test_linear_forward()
    test_packed_shard_loading()
    test_row_group_alignment_assert()
    test_config_weight_bits()
    test_fused_moe_bmm_quantized()
    test_model_wiring()
    print("ALL PASSED: test_quantization")


if __name__ == "__main__":
    main()
