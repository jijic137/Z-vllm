from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # 投机验证步专用：per-序列 bool（与 cu_seqlens_q 分段一一对应），True 表示该段
    # 携带草稿、LM head 需保留整段 logit 行（接受-拒绝需要每个位置的 logit）；
    # 批内无草稿时为 None，走"仅保留输出行"的常规快速路径
    spec_flags: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, spec_flags=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, spec_flags)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
