"""Weight-only W8 量化（int8 per-group 128 对称 RTN，加载时现场量化）。

设计口径（详见 阅读文档/权重量化-讨论稿.md，int4/W4 不在 v1 范围）：
- 量化发生在 weight_loader 载入 bf16 权重的那一刻：per-group 对称
  s_g = max|W_g| / 127，q = round(W / s_g)。零校准、零外部依赖，任何
  bf16 checkpoint 直接用；引擎不实现 AWQ/GPTQ 离线流程。
- 计算时"读到哪反量化到哪"：稠密路径走本模块的 Triton dequant GEMM，
  MoE 路径走 fused_moe 的 grouped GEMM dequant 分支。禁止整层物化反量化
  （瞬时双倍显存，30B-A3B 专家权重单卡 ~29GB → 58GB 直接 OOM）。
- 只量化 linear 权重（QKV/O/gate/up/down）；embed/lm_head 与 KV cache
  保持 bf16。
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from zvllm.layers.triton_utils import HAS_TRITON, triton_available

# 量化组大小（沿 in 维）。kernel BK=64 整除 128 → K tile 永远落在单个 group 内
WEIGHT_GROUP = 128


def quantize_weight(w: torch.Tensor, group: int = WEIGHT_GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [..., in] -> (int8 [..., in], bf16 scale [..., in // group])。

    对称 per-group：s_g = max|W_g| / 127，q = round(W / s_g) 并 clamp ±127
    （bf16 scale 舍入可能让 amax/s 略超 127）。全零组（s=0）以 s=1 保护：
    整组权重本来就是 0，q 恒 0，反量化结果恒 0，不会引入 nan。
    """
    assert w.shape[-1] % group == 0, \
        f"in 维（{w.shape[-1]}）必须被量化组大小（{group}）整除"
    shape = w.shape
    w_g = w.to(torch.float32).reshape(*shape[:-1], shape[-1] // group, group)
    amax = w_g.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).to(w.dtype)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w_g / scale).clamp(-127, 127).to(torch.int8)
    return q.reshape(shape), scale.reshape(*shape[:-1], shape[-1] // group)


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor, group: int = WEIGHT_GROUP) -> torch.Tensor:
    """int8 + bf16 scale -> bf16 反量化权重（CPU / 参照 / bmm 兜底路径）。"""
    shape = q.shape
    q_g = q.to(torch.float32).reshape(*shape[:-1], shape[-1] // group, group)
    w_g = q_g * scale.to(torch.float32).unsqueeze(-1)
    return w_g.reshape(shape).to(scale.dtype)


if HAS_TRITON:

    @triton.jit
    def _dequant_linear_kernel(
        a_ptr,      # bf16 [M, K] 激活
        w_ptr,      # int8 [N, K] packed 权重（F.linear 的 weight 布局）
        s_ptr,      # bf16 [N, K // GROUP] per-group scale
        c_ptr,      # bf16 [M, N]
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * K + tl.arange(0, BK)[None, :]
        w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * K + tl.arange(0, BK)[None, :]
        s_ptrs = s_ptr + offs_n.to(tl.int64) * (K // GROUP)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        mask_m = offs_m < M
        for k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            w_q = tl.load(w_ptrs)
            # K tile 整落在一个 group 内（GROUP=128 可被 BK=64 整除）→ scale 为 [BN] 向量
            s = tl.load(s_ptrs + (k // GROUP))
            w = (w_q.to(tl.float32) * s.to(tl.float32)[:, None]).to(tl.bfloat16)
            acc = tl.dot(a, tl.trans(w), acc)
            a_ptrs += BK
            w_ptrs += BK
        c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


def quant_linear(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x [M, K] bf16 × w int8 [N, K]ᵀ（+ scale）-> [M, N] bf16（F.linear 语义，无 bias）。

    GPU 上 Triton kernel 内反量化（不物化中间 bf16 权重）；CPU 或 Triton
    不可用时物化反量化 + mm（单元测试 / 调试路径）。
    """
    if x.numel() == 0:
        return x.new_empty((x.shape[0], w.shape[0]))
    if x.is_cuda and HAS_TRITON and triton_available():
        M, K = x.shape
        N = w.shape[0]
        assert K % 64 == 0 and N % 64 == 0, \
            f"W8 稠密 GEMM 要求 K/N 被 BK/BN=64 整除（K={K}, N={N}）"
        assert K % WEIGHT_GROUP == 0, f"K（{K}）必须被量化组（{WEIGHT_GROUP}）整除"
        c = torch.empty((M, N), device=x.device, dtype=x.dtype)
        _dequant_linear_kernel[((M + 15) // 16, N // 64)](
            x, w, scale, c, M,
            N=N, K=K, GROUP=WEIGHT_GROUP, BM=16, BN=64, BK=64,
            num_warps=4, num_stages=3,
        )
        return c
    return F.linear(x, dequantize_weight(w, scale))
