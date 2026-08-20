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


    @triton.jit
    def store_kvcache_int8_kernel(
        key_ptr, key_stride, value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
        slot_mapping_ptr, D: tl.constexpr,
    ):
        """int8 KV 写路径：kernel 内 per-token 对称量化（max-abs / 127），写 int8 + fp32 per-token scale。"""
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets).to(tl.float32)
        value = tl.load(value_ptr + value_offsets).to(tl.float32)
        k_scale = tl.maximum(tl.max(tl.abs(key), 0), 1e-8) / 127.0
        v_scale = tl.maximum(tl.max(tl.abs(value), 0), 1e-8) / 127.0
        k_q = tl.where(key >= 0, tl.floor(key / k_scale + 0.5), tl.ceil(key / k_scale - 0.5))
        v_q = tl.where(value >= 0, tl.floor(value / v_scale + 0.5), tl.ceil(value / v_scale - 0.5))
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, tl.minimum(tl.maximum(k_q, -127.0), 127.0).to(tl.int8))
        tl.store(v_cache_ptr + cache_offsets, tl.minimum(tl.maximum(v_q, -127.0), 127.0).to(tl.int8))
        tl.store(k_scale_ptr + slot, k_scale)
        tl.store(v_scale_ptr + slot, v_scale)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    assert HAS_TRITON, "store_kvcache requires triton"
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_int8(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                       k_scale_cache: torch.Tensor, v_scale_cache: torch.Tensor, slot_mapping: torch.Tensor):
    assert HAS_TRITON, "store_kvcache_int8 requires triton"
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale_cache.dtype == torch.float32 and v_scale_cache.dtype == torch.float32
    assert slot_mapping.numel() == N
    store_kvcache_int8_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                                    k_cache, v_cache, k_scale_cache, v_scale_cache, slot_mapping, D)


def _repeat_kv(t: torch.Tensor, ratio: int, dim: int) -> torch.Tensor:
    """GQA：KV head 从 Hkv 扩展到 H"""
    return t if ratio == 1 else t.repeat_interleave(ratio, dim=dim)


def _gather_context(block_ids: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, total_len: int, block_size: int,
                    k_scale_cache: torch.Tensor | None = None, v_scale_cache: torch.Tensor | None = None,
                    out_dtype: torch.dtype | None = None):
    """从分页缓存按 block 表 gather 出完整上下文 K/V，token-major [total_len, Hkv, D]。
    int8 缓存：同步 gather per-token scale 并反量化到 out_dtype（缺省 bf16）后返回。"""
    device = block_ids.device
    idx = (block_ids.to(torch.int64)[:, None] * block_size
           + torch.arange(block_size, device=device, dtype=torch.int64)).reshape(-1)[:total_len]
    Hkv, D = k_cache.size(2), k_cache.size(3)
    k = k_cache.view(k_cache.size(0) * block_size, Hkv, D)[idx]
    v = v_cache.view(v_cache.size(0) * block_size, Hkv, D)[idx]
    if k_scale_cache is not None:
        dtype = out_dtype or torch.bfloat16
        ks = k_scale_cache.view(k_scale_cache.size(0) * block_size)[idx].to(dtype)
        vs = v_scale_cache.view(v_scale_cache.size(0) * block_size)[idx].to(dtype)
        k = k.to(dtype) * ks[:, None, None]
        v = v.to(dtype) * vs[:, None, None]
    return k, v


def sdpa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
                 block_size: int, k_cache: torch.Tensor | None, v_cache: torch.Tensor | None,
                 block_tables: torch.Tensor | None,
                 num_heads: int, num_kv_heads: int, scale: float,
                 k_scale_cache: torch.Tensor | None = None, v_scale_cache: torch.Tensor | None = None) -> torch.Tensor:
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
                                             k_cache, v_cache, Lk, block_size,
                                             k_scale_cache, v_scale_cache, q.dtype)
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


if HAS_TRITON:

    @triton.jit
    def paged_decode_attn_kernel(
        q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, out_ptr,
        context_lens_ptr, block_tables_ptr, bt_stride,
        softmax_scale,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
        IS_INT8: tl.constexpr,
    ):
        """Triton paged decode attention：每 (batch, q-head) 一个 program，online softmax，
        按 block 表直读 K/V，无 host 同步（兼容 CUDA Graph 捕获）。
        IS_INT8：读 int8 + fp32 per-token scale，kernel 内反量化；否则直读模型 dtype
        （与 sdpa_decode 数值等价，且无 Lmax 同步）。
        q: [bs, H, D] contiguous；k/v_cache: [nb, BS, Hkv, D]；scale: [nb, BS]；out: [bs, H, D] f32。"""
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        hkv = h // (H // Hkv)
        L = tl.load(context_lens_ptr + b)
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + pid * D + offs_d).to(tl.float32)
        offs_d64 = offs_d.to(tl.int64)
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([D], tl.float32)
        qk_scale = softmax_scale * 1.4426950408889634    # log2(e)
        kv_block_stride = BS * Hkv * D
        for blk in range(0, L // BS):
            bt = tl.load(block_tables_ptr + b * bt_stride + blk).to(tl.int64)
            base = bt * kv_block_stride + hkv * D
            offs_t = tl.arange(0, BS)    # 块内局部偏移：逻辑块的 token 总在物理块 0..BS-1
            k_offs = base + offs_t[:, None].to(tl.int64) * (Hkv * D) + offs_d64[None, :]
            k = tl.load(k_cache_ptr + k_offs)
            if IS_INT8:
                k = k.to(tl.float32) * tl.load(k_scale_ptr + bt * BS + offs_t.to(tl.int64))[:, None]
            else:
                k = k.to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * qk_scale
            m_new = tl.maximum(m_i, tl.max(s, 0))
            p = tl.math.exp2(s - m_new)
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 0)
            v = tl.load(v_cache_ptr + k_offs)
            if IS_INT8:
                v = v.to(tl.float32) * tl.load(v_scale_ptr + bt * BS + offs_t.to(tl.int64))[:, None]
            else:
                v = v.to(tl.float32)
            acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
            m_i = m_new
        tail = L - (L // BS) * BS
        if tail > 0:
            blk = L // BS
            bt = tl.load(block_tables_ptr + b * bt_stride + blk).to(tl.int64)
            base = bt * kv_block_stride + hkv * D
            offs_t = tl.arange(0, BS)    # 块内局部偏移
            valid = offs_t < tail
            k_offs = base + offs_t[:, None].to(tl.int64) * (Hkv * D) + offs_d64[None, :]
            k = tl.load(k_cache_ptr + k_offs, mask=valid[:, None], other=0.0)
            if IS_INT8:
                k = k.to(tl.float32) * tl.load(k_scale_ptr + bt * BS + offs_t.to(tl.int64),
                                               mask=valid, other=0.0)[:, None]
            else:
                k = k.to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * qk_scale
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 0))
            p = tl.math.exp2(s - m_new)
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 0)
            v = tl.load(v_cache_ptr + k_offs, mask=valid[:, None], other=0.0)
            if IS_INT8:
                v = v.to(tl.float32) * tl.load(v_scale_ptr + bt * BS + offs_t.to(tl.int64),
                                               mask=valid, other=0.0)[:, None]
            else:
                v = v.to(tl.float32)
            acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
        out = acc / l_i
        tl.store(out_ptr + pid * D + offs_d, out)


def triton_paged_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                        k_scale_cache: torch.Tensor, v_scale_cache: torch.Tensor,
                        context_lens: torch.Tensor, block_tables: torch.Tensor,
                        num_heads: int, num_kv_heads: int, scale: float) -> torch.Tensor:
    """Triton paged decode attention（按缓存 dtype 自动 int8 直读 / bf16）。
    返回 [bs, 1, H, D]（与 sdpa_decode 同形状）；无 host 同步，兼容 CUDA Graph 捕获。"""
    assert HAS_TRITON, "triton_paged_decode requires triton"
    bs, H, D = q.shape
    BS = k_cache.size(1)
    assert q.is_contiguous(), "triton_paged_decode requires contiguous q"
    out = torch.empty(bs * H * D, dtype=torch.float32, device=q.device)
    is_int8 = k_cache.dtype == torch.int8
    paged_decode_attn_kernel[(bs * H,)](
        q, k_cache, v_cache,
        k_scale_cache if is_int8 else q,    # bf16 分支不读 scale 指针，占位即可
        v_scale_cache if is_int8 else q,
        out, context_lens, block_tables, block_tables.stride(0),
        scale, H, num_kv_heads, D, BS, is_int8,
        num_warps=4,
    )
    return out.view(bs, 1, H, D).to(q.dtype)


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
        self.k_scale = self.v_scale = torch.tensor([])    # int8 KV 的 per-token scale [nb, BS]（bf16 时为空）

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        is_int8 = bool(k_cache.numel() and k_cache.dtype == torch.int8)
        if k_cache.numel() and v_cache.numel():
            if is_int8:
                store_kvcache_int8(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if HAS_FLASH_ATTN and not is_int8:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:    # SDPA 兜底（ROCm 等无 flash-attn 环境；int8 KV 需 gather 反量化）
                o = sdpa_prefill(q, k, v, context.cu_seqlens_q, context.cu_seqlens_k,
                                 k_cache.size(1) if k_cache.numel() else 0,
                                 k_cache, v_cache, context.block_tables,
                                 self.num_heads, self.num_kv_heads, self.scale,
                                 self.k_scale if is_int8 else None,
                                 self.v_scale if is_int8 else None)
        else:    # decode
            if is_int8 or (HAS_TRITON and not HAS_FLASH_ATTN):
                # Triton paged decode：无 host 同步（graph 友好）；ROCm 无 flash-attn 时亦替换 SDPA 兜底
                o = triton_paged_decode(q, k_cache, v_cache, self.k_scale, self.v_scale,
                                        context.context_lens, context.block_tables,
                                        self.num_heads, self.num_kv_heads, self.scale)
            elif HAS_FLASH_ATTN:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
            else:    # SDPA 兜底（CPU 单测环境）
                o = sdpa_decode(q, k_cache, v_cache, context.context_lens, context.block_tables,
                                k_cache.size(1), self.num_heads, self.num_kv_heads, self.scale)
        return o
