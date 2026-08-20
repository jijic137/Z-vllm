"""Triton 可用性探测（稠密 dequant GEMM 与 MoE grouped GEMM 共用）。

独立成模块避免 quantization ↔ fused_moe 的循环导入（两者都需要探测，
fused_moe 还需 quantization 的 WEIGHT_GROUP）。
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_triton_ok: bool | None = None


def triton_available() -> bool:
    """triton 可导入 + 在当前设备能跑通（惰性探测一次；gfx 等非主流架构可能编译失败）。"""
    global _triton_ok
    if not HAS_TRITON:
        return False
    if _triton_ok is None:
        try:

            @triton.jit
            def _probe(x_ptr, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1.0)

            x = torch.zeros(256, device="cuda", dtype=torch.float32)
            _probe[(1,)](x, BLOCK=256)
            _triton_ok = bool((x == 1.0).all().item())
        except Exception:
            _triton_ok = False
    return _triton_ok
