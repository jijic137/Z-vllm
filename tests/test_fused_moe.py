"""MoE 融合路径纯 CPU 单测（无需 GPU/triton/flash_attn）。运行：python tests/test_fused_moe.py

对拍对象：float64 逐专家朴素参照（与 HF Qwen3-MoE 参考实现同构）。
覆盖：
- 融合 bmm 路径 vs float64 参照（bf16 容差内，T=1/3/7）
- top-k 路由选择与参照严格一致
- EP 语义：local_start>0 时非本地专家贡献严格为 0
- triton 路径胶水逻辑（CPU 可跑，不启动 kernel）：_align_pairs 块纯度 +
  scatter 掩码严格选中本地 pair 集合（回归 dummy 桶未初始化行被误读的 bug）
- 确定性：重复运行 bit 级一致
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from zvllm.layers.fused_moe import fused_moe_bmm

torch.manual_seed(0)

H, I, E_GLOBAL, K = 32, 16, 8, 4

# 数值尺度：输入/权重缩放到 |y| ~ O(4)，bf16 量化误差 ~1e-2，atol=0.1 约 10 倍量化间距；
# gate 权重放大 1.5 倍使 top-k logits margin ~O(1) >> bf16 logit 噪声，路由选择稳定
X_SCALE, W_SCALE, GATE_SCALE = 0.5, 0.5, 1.5
ATOL = 0.1


def route(x: torch.Tensor, gate_w: torch.Tensor, k: int = K):
    """与 Qwen3MoeSparseMoeBlock 完全一致的路由：float softmax → topk → 归一化"""
    probs = torch.softmax(x @ gate_w.t(), dim=-1).float()
    w, ids = torch.topk(probs, k, dim=-1)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w, ids


def ref_forward(x: torch.Tensor, gate_w: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
                local_start: int, num_local: int) -> torch.Tensor:
    """float64 朴素参照：只计算 local_start..local_start+num_local-1 的专家"""
    w, ids = route(x.double(), gate_w.double())
    T = x.shape[0]
    out = torch.zeros(T, K, H, dtype=torch.float64)
    for e in range(local_start, local_start + num_local):
        m = ids == e
        if not m.any():
            continue
        idx = m.nonzero()
        xe = x.double()[idx[:, 0]]
        gu = xe @ w13[e].double().t()  # [n, 2I]
        g, u = gu[:, :I], gu[:, I:]
        h = g * torch.sigmoid(g) * u
        y = h @ w2[e].double().t()
        out[idx[:, 0], idx[:, 1]] = y * w[idx[:, 0], idx[:, 1]].unsqueeze(-1)
    return out


def check(name: str, got: torch.Tensor, expect: torch.Tensor, atol: float):
    diff = (got.double() - expect).abs().max().item()
    assert diff < atol, f"{name}: max diff {diff} > {atol}"
    print(f"  {name}: max diff {diff:.2e} < {atol}")


def test_triton_layout_and_scatter():
    """triton 融合路径的 CPU 回归：_route_local/_align_pairs/scatter 掩码（不启动 kernel）。

    不变量：
    - expert_ids 取值 [-1, num_local)：dummy 桶/越界块映射为 -1（kernel 跳过，
      不得以 e=num_local 去读 stacked 权重造成 OOB）
    - 块纯度：e>=0 的 M 块内所有真实 pair 都来自专家 e（grouped GEMM 正确性基础）
    - scatter 掩码选中的 pair 多重集 == 本地 pair 集合（非本地/padding 一律不取）
    """
    from zvllm.layers.fused_moe import _align_pairs, _route_local

    BM = 16
    torch.manual_seed(7)
    cases = (
        (1, 8, 128, 64, 64),    # 30B decode EP=2 形状（T=1, K=8, 半本地）
        (3, 4, 8, 4, 4),        # 小型 EP
        (17, 8, 128, 0, 128),   # EP=1：无 dummy 桶
        (7, 8, 128, 96, 32),    # 偏斜本地区间
    )
    for T, k, e_global, local_start, num_local in cases:
        topk_ids = torch.randint(0, e_global, (T, k))
        e_ids, is_local = _route_local(topk_ids, local_start, num_local)
        sorted_ids, expert_ids = _align_pairs(e_ids, num_local, BM)
        n = T * k
        m_max = sorted_ids.shape[0]
        assert m_max % BM == 0 and expert_ids.shape == (m_max // BM,)
        assert expert_ids.min().item() >= -1 and expert_ids.max().item() < num_local, \
            f"T={T}: expert_ids 越界（dummy 桶未映射为 -1？）"
        for b in range(expert_ids.shape[0]):
            e = expert_ids[b].item()
            if e < 0:
                continue
            ids_b = sorted_ids[b * BM:(b + 1) * BM]
            real = ids_b[ids_b < n]
            assert (e_ids[real] == e).all(), f"T={T}: 块 {b} 混入非专家 {e} 的 pair"
        pair_valid = sorted_ids < n
        ids_safe = torch.where(pair_valid, sorted_ids, torch.zeros_like(sorted_ids)).to(torch.int64)
        row_valid = pair_valid & is_local[ids_safe]
        rows = ids_safe[row_valid]
        expect = torch.nonzero(is_local).flatten()
        assert torch.equal(torch.sort(rows).values, torch.sort(expect).values), \
            f"T={T}: scatter 选中 pair 与本地 pair 集合不一致"
        print(f"  layout/scatter T={T} local=[{local_start},{local_start + num_local}) "
              f"E={e_global}: OK（{rows.numel()}/{n} 对为本地）")


def main():
    gate_w = torch.randn(E_GLOBAL, H) * GATE_SCALE
    w13 = torch.randn(E_GLOBAL, 2 * I, H) * W_SCALE  # 前 I 行 gate、后 I 行 up
    w2 = torch.randn(E_GLOBAL, H, I) * W_SCALE
    for T in (1, 3, 7):
        x = torch.randn(T, H) * X_SCALE
        xb = x.bfloat16()
        # 与实现同精度链：bf16 输入/权重（CPU bf16 matmul）→ float softmax
        w32, ids32 = route(xb.float(), gate_w.bfloat16().float())
        out = fused_moe_bmm(xb, w32, ids32, 0, w13.bfloat16(), w2.bfloat16())
        ref = ref_forward(x, gate_w, w13, w2, 0, E_GLOBAL)
        print(f"T={T} (EP=1 全本地专家)")
        # 路由选择：bf16 topk 与 float64 topk 的专家集合应一致
        ref_w, ref_ids = route(x.double(), gate_w.double())
        assert (torch.sort(ids32, dim=-1).values == torch.sort(ref_ids.int(), dim=-1).values).all(), \
            f"T={T}: top-k 专家选择与参照不一致"
        check("bmm vs float64 参照", out.sum(1), ref.sum(1), ATOL)
        # 确定性
        out2 = fused_moe_bmm(xb, w32, ids32, 0, w13.bfloat16(), w2.bfloat16())
        assert torch.equal(out, out2), "bmm 路径不确定（重复运行 bit 级不一致）"
        # EP 语义：local_start=4 时前 4 个专家贡献必须为 0
        out_ep = fused_moe_bmm(xb, w32, ids32, 4, w13.bfloat16()[4:], w2.bfloat16()[4:])
        ref_ep = ref_forward(x, gate_w, w13, w2, 4, E_GLOBAL - 4)
        print(f"T={T} (EP 语义 local_start=4)")
        check("EP 子集 vs float64 参照", out_ep.sum(1), ref_ep.sum(1), ATOL)
    print("triton 胶水逻辑（layout + scatter 掩码）")
    test_triton_layout_and_scatter()
    print("ALL PASSED: test_fused_moe")


if __name__ == "__main__":
    main()
