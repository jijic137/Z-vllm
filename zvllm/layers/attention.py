import torch
import torch.nn.functional as F
from torch import nn

from zvllm.utils.context import get_context

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:    # CPU 单测环境：store_kvcache kernel 不可用，仅 SDPA 兜底逻辑可测
    triton = None
    tl = None
    HAS_TRITON = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:    # ROCm/AMD 环境通常没有 flash-attn，退回 SDPA 后端
    HAS_FLASH_ATTN = False


if HAS_TRITON:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    assert HAS_TRITON, "store_kvcache requires triton"
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _repeat_kv(t: torch.Tensor, ratio: int, dim: int) -> torch.Tensor:
    """GQA：KV head 从 Hkv 扩展到 H"""
    return t if ratio == 1 else t.repeat_interleave(ratio, dim=dim)


def _gather_context(block_ids: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, total_len: int, block_size: int):
    """从分页缓存按 block 表 gather 出完整上下文 K/V，token-major [total_len, Hkv, D]"""
    device = block_ids.device
    idx = (block_ids.to(torch.int64)[:, None] * block_size
           + torch.arange(block_size, device=device, dtype=torch.int64)).reshape(-1)[:total_len]
    Hkv, D = k_cache.size(2), k_cache.size(3)
    k = k_cache.view(k_cache.size(0) * block_size, Hkv, D)[idx]
    v = v_cache.view(v_cache.size(0) * block_size, Hkv, D)[idx]
    return k, v


def sdpa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
                 block_size: int, k_cache: torch.Tensor | None, v_cache: torch.Tensor | None,
                 block_tables: torch.Tensor | None,
                 num_heads: int, num_kv_heads: int, scale: float) -> torch.Tensor:
    """prefill 阶段 SDPA 兜底（等价 flash_attn_varlen_func）。纯张量运算，CPU/GPU 通用。
    block_tables 为 None：所有序列无 prefix，直接用新算出的 k/v；
    否则从分页缓存 gather 完整上下文（新 token 已由 store_kvcache 写入缓存）。
    q: [Tq, H, D]；k, v: [Tk, Hkv, D]（仅 block_tables 为 None 时使用）
    返回 [Tq, H, D]
    """
    ratio = num_heads // num_kv_heads
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    device = q.device
    outs = []
    for i in range(len(cu_q) - 1):
        Lq = cu_q[i + 1] - cu_q[i]
        Lk = cu_k[i + 1] - cu_k[i]
        q_i = q[cu_q[i]:cu_q[i + 1]].unsqueeze(0).transpose(1, 2)          # [1, H, Lq, D]
        if block_tables is None:
            assert Lk == Lq
            k_i = _repeat_kv(k[cu_k[i]:cu_k[i + 1]], ratio, 1).unsqueeze(0).transpose(1, 2)
            v_i = _repeat_kv(v[cu_k[i]:cu_k[i + 1]], ratio, 1).unsqueeze(0).transpose(1, 2)
            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=True, scale=scale)
        else:
            k_full, v_full = _gather_context(block_tables[i, :(Lk + block_size - 1) // block_size],
                                             k_cache, v_cache, Lk, block_size)
            k_full = _repeat_kv(k_full, ratio, 1).unsqueeze(0).transpose(1, 2)
            v_full = _repeat_kv(v_full, ratio, 1).unsqueeze(0).transpose(1, 2)
            rows = torch.arange(Lq, device=device)[:, None]
            cols = torch.arange(Lk, device=device)[None, :]
            mask = cols <= Lk - Lq + rows                                  # 右对齐因果
            o_i = F.scaled_dot_product_attention(q_i, k_full, v_full, attn_mask=mask, scale=scale)
        outs.append(o_i)
    return torch.cat(outs, dim=2).squeeze(0).transpose(0, 1)              # [Tq, H, D]


def sdpa_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                context_lens: torch.Tensor, block_tables: torch.Tensor,
                block_size: int, num_heads: int, num_kv_heads: int, scale: float) -> torch.Tensor:
    """decode 阶段 SDPA 兜底（等价 flash_attn_with_kvcache）。
    Lmax = context_lens.max() 需要 host 同步：仅适用于 enforce_eager，与 CUDA Graph 捕获不兼容。
    q: [bs, H, D]；k_cache/v_cache: [nb, bs, Hkv, D]；context_lens: [bs]；
    block_tables: [bs, max_nb] int（-1 为 padding）
    返回 [bs, 1, H, D]
    """
    bs = q.size(0)
    Lmax = int(context_lens.max())
    device = q.device
    Hkv, D = k_cache.size(2), k_cache.size(3)
    ratio = num_heads // num_kv_heads
    pos = torch.arange(Lmax, device=device, dtype=torch.int64)
    blocks = block_tables[:, (pos // block_size).clamp(max=block_tables.size(1) - 1)]   # [bs, Lmax]
    blocks = blocks.clamp(min=0)                                                        # -1 padding -> 块 0（随后被 mask 掉）
    slots = blocks.to(torch.int64) * block_size + (pos % block_size).unsqueeze(0)       # [bs, Lmax]
    k_flat = k_cache.view(k_cache.size(0) * block_size, Hkv, D)
    v_flat = v_cache.view(v_cache.size(0) * block_size, Hkv, D)
    k_full = _repeat_kv(k_flat[slots], ratio, 2).permute(0, 2, 1, 3)                    # [bs, H, Lmax, D]
    v_full = _repeat_kv(v_flat[slots], ratio, 2).permute(0, 2, 1, 3)
    valid = pos.unsqueeze(0) < context_lens.to(torch.int64).unsqueeze(1)                # [bs, Lmax]
    o = F.scaled_dot_product_attention(q.unsqueeze(2), k_full, v_full,                   # [bs, H, 1, D]
                                       attn_mask=valid[:, None, None, :], scale=scale)
    return o.permute(0, 2, 1, 3)                                                          # [bs, 1, H, D]


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if HAS_FLASH_ATTN:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:    # SDPA 兜底（ROCm 等无 flash-attn 环境）
                o = sdpa_prefill(q, k, v, context.cu_seqlens_q, context.cu_seqlens_k,
                                 k_cache.size(1) if k_cache.numel() else 0,
                                 k_cache, v_cache, context.block_tables,
                                 self.num_heads, self.num_kv_heads, self.scale)
        else:    # decode
            if HAS_FLASH_ATTN:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
            else:    # SDPA 兜底（ROCm 等无 flash-attn 环境）
                o = sdpa_decode(q, k_cache, v_cache, context.context_lens, context.block_tables,
                                k_cache.size(1), self.num_heads, self.num_kv_heads, self.scale)
        return o
