"""MoE 融合前向：消除逐专家 gather/GEMM/scatter 循环与 host 同步。

背景（为什么要做）：原版 Qwen3MoeSparseMoeBlock.forward 对本地每个专家执行
mask.any()（GPU→CPU 同步）+ nonzero + gather + 两次小 GEMM + scatter。
decode 阶段每层 64 个本地专家 × 48 层 ≈ 3000 次同步 + 上万次 kernel 启动，
是 30B-A3B EP=2 decode 只有 3.7 tok/s 的主因（权重 HBM 读取本身只要 ~20ms/step）。

本模块提供两个数值等价的融合后端（路由 / top-k / 归一化 / 跨 rank all_reduce 逻辑
仍在 qwen3_moe.Qwen3MoeSparseMoeBlock.forward 中，这里只算"本地专家加权和"）：

- fused_moe_triton：CUDA 上的 Triton grouped GEMM。把 (token, slot) 对按本地专家号
  排序（非本地对放进 dummy 桶），M 维按块对齐；两个 grouped GEMM kernel（gate_up、
  down）+ torch silu&mul + 按路由权重 scatter。两个 GEMM 共用一个 kernel，用
  A_BY_TOKEN 区分行索引语义：gate_up 的输入 x [T,H] 按 token（pair//K）取行，
  down 的输入 h [M_max,I] 按排序行取行。排序 / 对齐全部是 on-device torch op，
  无 .any()/.item()/.nonzero()。
- fused_moe_bmm：纯 torch 兜底（CPU / 无 Triton / Triton 不支持当前设备）。每个专家
  padding 到 R=T*K 行，两次 bmm；padding 行输入为 0，GEMM 输出严格为 0，不贡献。

两者都不改变数学定义：out[t] = Σ_{(t,s) 路由到本地专家} w_{t,s} · expert_{e(t,s)}(x[t])。

两个后端都支持 W8 量化权重（int8 + per-group 128 scale）：Triton 路径在
kernel K 循环内反量化（不物化），bmm 路径先物化反量化（仅 CPU 单测 /
调试用；GPU 上量化配置由 MoE 块启动断言强制走 Triton）。
"""
import torch
import torch.nn.functional as F

from zvllm.layers.triton_utils import HAS_TRITON, triton_available  # noqa: F401（模型层经本模块引用）
from zvllm.quantization import WEIGHT_GROUP, dequantize_weight

if HAS_TRITON:
    import triton
    import triton.language as tl

# 融合路径 token 阈值：T 更大批次（prefill）保留原遍历实现，避免 [E, R=T*K, H] 缓冲膨胀
FUSED_MAX_TOKENS = 256
BMM_MAX_TOKENS = 32

# Triton 块尺寸（decode 导向：M 行很少，BN/BK 取大让单 program 干满权重流式读取）
_BM, _BN, _BK = 16, 64, 64


def _align_pairs(e_ids: torch.Tensor, num_experts: int, block_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把 (token, slot) 对按本地专家号分桶，输出 host 形状可知的 padding 布局。

    e_ids: [N] int64，取值 [0, num_experts]（num_experts 是 dummy 桶，收纳非本地对）。
    返回：
      sorted_token_ids: [M_max] int32，pair id（= token*K+slot）；padding 位置 = N（哨兵）
      expert_ids: [M_max // block_m] int32，每个 M 块的专家号；-1 = 无效块（kernel 跳过）
    M_max 为 host 可算上界（不依赖任何 device 标量），因此 grid 尺寸无需 host 同步。
    """
    N, E = e_ids.numel(), num_experts
    device = e_ids.device
    m_max = N + (E + 1) * block_m
    m_max = (m_max + block_m - 1) // block_m * block_m
    counts = torch.bincount(e_ids, minlength=E + 1)
    counts_pad = ((counts + block_m - 1) // block_m) * block_m
    cum_pad = counts_pad.cumsum(0)
    order = torch.argsort(e_ids, stable=True)
    e_sorted = e_ids[order]
    # 桶 e 之前的 padding 行数 = Σ_{e'<e} (counts_pad[e'] - counts[e'])
    shift_before = torch.cat(
        [torch.zeros(1, dtype=e_ids.dtype, device=device), (cum_pad - counts.cumsum(0))[:-1]]
    )
    pos = torch.arange(N, device=device, dtype=e_ids.dtype) + shift_before[e_sorted]
    sorted_token_ids = torch.full((m_max,), N, dtype=torch.int32, device=device)
    sorted_token_ids.scatter_(0, pos, order.to(torch.int32))
    row0 = torch.arange(m_max // block_m, device=device) * block_m
    # 包含 row0 的桶 = 第一个 cum_pad[e] > row0 的 e；row0 超出总行数 → E+1 → 无效块
    expert_ids = torch.searchsorted(cum_pad, row0, right=True).to(torch.int32)
    expert_ids = torch.where(expert_ids >= E, torch.full_like(expert_ids, -1), expert_ids)
    return sorted_token_ids, expert_ids


if HAS_TRITON:

    @triton.jit
    def _moe_grouped_gemm(
        a_ptr,      # bf16：A_BY_TOKEN 时 [T, K_IN]（x，按 token 取行）；否则 [M_max, K_IN]（h，按排序行取行）
        w_ptr,      # [E, N_OUT, K_IN] bf16 或 int8（stacked 专家权重，QUANT_W 时 int8）
        c_ptr,      # [M_max, N_OUT] bf16
        w_s_ptr,    # [E, N_OUT, K_IN // GROUP] bf16 per-group scale（QUANT_W=False 时不读，传任意指针）
        sorted_ids,  # [M_max] int32 pair id（哨兵 = N_PAIRS）
        expert_ids,  # [M_max // BM] int32（-1 = 跳过）
        N_PAIRS,     # T * K_TOPK
        K_TOPK: tl.constexpr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        A_BY_TOKEN: tl.constexpr,
        QUANT_W: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(expert_ids + pid_m)
        if e < 0:
            return
        offs_m = pid_m * BM + tl.arange(0, BM)
        pair_ids = tl.load(sorted_ids + offs_m)
        valid = pair_ids < N_PAIRS
        if A_BY_TOKEN:
            a_rows = tl.where(valid, pair_ids // K_TOPK, 0)
        else:
            a_rows = offs_m
        a_ptrs = a_ptr + a_rows[:, None].to(tl.int64) * K_IN + tl.arange(0, BK)[None, :]
        b_ptrs = w_ptr + e.to(tl.int64) * (N_OUT * K_IN) \
            + (pid_n * BN + tl.arange(0, BN))[:, None].to(tl.int64) * K_IN \
            + tl.arange(0, BK)[None, :]
        if QUANT_W:
            # b_s_base 保持 1D [BN]：tl.load 出 [BN] scale 后 [:, None] 广播为 [BN,1]。
            # 若这里带 [:, None]（2D 指针），load 出 [BN,1]，再 [:, None] 会膨胀成
            # rank-3 [BN,BN,BK]，tl.dot 直接编译失败（仅 GPU 编译期暴露）。
            b_s_base = w_s_ptr + e.to(tl.int64) * (N_OUT * (K_IN // GROUP)) \
                + (pid_n * BN + tl.arange(0, BN)).to(tl.int64) * (K_IN // GROUP)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K_IN, BK):
            a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
            if QUANT_W:
                # kernel 内反量化：int8 读 × 每行 group scale → bf16（K tile 恒在单个 group 内）
                w_q = tl.load(b_ptrs)
                s = tl.load(b_s_base + (k // GROUP))
                b = (w_q.to(tl.float32) * s.to(tl.float32)[:, None]).to(tl.bfloat16)
            else:
                b = tl.load(b_ptrs)
            acc = tl.dot(a, tl.trans(b), acc)
            a_ptrs += BK
            b_ptrs += BK
        c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * N_OUT + (pid_n * BN + tl.arange(0, BN))[None, :]
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=valid[:, None])


def _route_local(topk_ids: torch.Tensor, local_start: int, num_local: int) -> tuple[torch.Tensor, torch.Tensor]:
    """全局专家号 → 本地号；非本地对映射到 dummy 桶。返回 (e_ids[N], is_local[N])。"""
    local = topk_ids - local_start
    is_local = ((local >= 0) & (local < num_local)).reshape(-1)
    e_ids = torch.where(is_local, local.reshape(-1), torch.tensor(num_local, device=topk_ids.device, dtype=topk_ids.dtype))
    return e_ids, is_local


def fused_moe_triton(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    local_start: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton grouped GEMM 融合路径。

    x: [T, H] bf16；topk_weights: [T, K]（已归一化）；topk_ids: [T, K] int64 全局专家号
    w13: [E, 2I, H]（前 I 行 gate、后 I 行 up）；w2: [E, H, I]（均为本 rank 专家内 TP 分片）。
    bf16 或 W8 量化（int8 + w13_scale [E, 2I, H/G] / w2_scale [E, H, I/G]）。
    返回 [T, K, H]：本 rank 本地专家贡献，非本地 (token, slot) 位置为 0。
    """
    quant = w13.dtype == torch.int8
    assert (not quant) or (w13_scale is not None and w2_scale is not None), \
        "W8 量化 MoE 权重必须同时传入 scales（w13_scale / w2_scale）"
    T, H = x.shape
    K = topk_ids.shape[1]
    E, I = w13.shape[0], w2.shape[2]
    N = T * K
    device = x.device

    e_ids, is_local = _route_local(topk_ids, local_start, E)
    sorted_ids, expert_ids = _align_pairs(e_ids, E, _BM)
    m_max = sorted_ids.shape[0]

    c1 = torch.empty((m_max, 2 * I), device=device, dtype=x.dtype)
    _moe_grouped_gemm[(m_max // _BM, (2 * I) // _BN)](
        x, w13, c1, w13_scale if quant else w13, sorted_ids, expert_ids, N,
        K_TOPK=K, N_OUT=2 * I, K_IN=H, BM=_BM, BN=_BN, BK=_BK, A_BY_TOKEN=True,
        QUANT_W=quant, GROUP=WEIGHT_GROUP,
        num_warps=4, num_stages=3,
    )
    gate, up = c1.chunk(2, dim=1)
    h = F.silu(gate) * up  # [M_max, I]
    y = torch.empty((m_max, H), device=device, dtype=x.dtype)
    _moe_grouped_gemm[(m_max // _BM, H // _BN)](
        h, w2, y, w2_scale if quant else w2, sorted_ids, expert_ids, N,
        K_TOPK=K, N_OUT=H, K_IN=I, BM=_BM, BN=_BN, BK=_BK, A_BY_TOKEN=False,
        QUANT_W=quant, GROUP=WEIGHT_GROUP,
        num_warps=4, num_stages=3,
    )
    pair_valid = sorted_ids < N
    ids_safe = torch.where(pair_valid, sorted_ids, torch.zeros_like(sorted_ids)).to(torch.int64)
    # 非本地对进 dummy 桶：kernel 跳过其 M 块（expert_ids=-1），c1/y 对应行未初始化，
    # scatter 只取本地对；非本地位置保持 0，由跨 rank all_reduce 补齐完整加权和
    row_valid = pair_valid & is_local[ids_safe]
    rows = ids_safe[row_valid]
    out = torch.zeros((T, K, H), device=device, dtype=x.dtype)
    out[rows // K, rows % K] = y[row_valid] * topk_weights.reshape(-1)[rows].to(x.dtype).unsqueeze(-1)
    return out


def fused_moe_bmm(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    local_start: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """纯 torch padded bmm 融合路径（CPU / 无 Triton 兜底），输出布局同 fused_moe_triton。

    每个专家 padding 到 R=T*K 行：pair id 直接作行号，未路由到该专家的输入行全 0，
    bmm 输出严格为 0，scatter 时只写真实 (token, slot)，不破坏其他专家贡献。
    W8 量化权重先物化反量化（此路径仅用于 CPU 单测 / 调试）。
    """
    if w13.dtype == torch.int8:
        assert w13_scale is not None and w2_scale is not None, \
            "W8 量化 MoE 权重必须同时传入 scales（w13_scale / w2_scale）"
        w13 = dequantize_weight(w13, w13_scale, WEIGHT_GROUP)
        w2 = dequantize_weight(w2, w2_scale, WEIGHT_GROUP)
    T, H = x.shape
    K = topk_ids.shape[1]
    E, I = w13.shape[0], w2.shape[2]
    N, R = T * K, T * K
    device = x.device

    e_ids, is_local = _route_local(topk_ids, local_start, E)
    pair_rows = x.unsqueeze(1).expand(T, K, H).reshape(N, H)  # pair i 的输入行 = x[i // K]
    pair_ids_all = torch.arange(N, device=device)
    xp = torch.zeros((E, R, H), device=device, dtype=x.dtype)
    xp[e_ids[is_local], pair_ids_all[is_local], :] = pair_rows[is_local]

    c1 = torch.bmm(xp, w13.transpose(1, 2))  # [E, R, 2I]
    gate, up = c1.chunk(2, dim=2)
    h = F.silu(gate) * up
    y = torch.bmm(h, w2.transpose(1, 2))  # [E, R, H]

    # 真实行：pair p 路由到的专家恰为本行专家号
    e_of_row = torch.arange(E, device=device).unsqueeze(1).expand(E, R)
    p_of_row = pair_ids_all.unsqueeze(0).expand(E, R)
    real = (e_of_row == e_ids[p_of_row])  # [E, R] bool
    out = torch.zeros((T, K, H), device=device, dtype=x.dtype)
    out[p_of_row[real] // K, p_of_row[real] % K] = (
        y[real] * topk_weights.reshape(-1)[p_of_row[real]].to(x.dtype).unsqueeze(-1)
    )
    return out
