"""logit 行布局纯 CPU 单测（LM head 行保留 × _sample_all 行游标，无需 GPU）。运行：python tests/test_logit_layout.py

背景：prefill 步 ParallelLMHead 只保留每段输出行（旧行为），投机验证步带草稿段
需保留整段 γ+1 行（接受-拒绝要逐位置 logit）。本文件对拍两条规则及其与
_sample_all 行游标的并行关系，防止行布局回归（真机上表现为越界 gather 非法访存）。
"""
import socket
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import torch.distributed as dist

from zvllm.engine.model_runner import ModelRunner
from zvllm.layers.embed_head import ParallelLMHead
from zvllm.layers.sampler import Sampler
from zvllm.utils.context import set_context, reset_context

VOCAB = 64


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


if not dist.is_initialized():
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{free_port()}", world_size=1, rank=0)

torch.manual_seed(0)


def make_head(seed=0):
    g = torch.Generator().manual_seed(seed)
    head = ParallelLMHead(VOCAB, VOCAB)
    with torch.no_grad():
        head.weight.normal_(generator=g)
    return head


def test_lmhead_plain_prefill_keeps_last_row():
    """spec_flags=None（常规 prefill）：每段只保留末行，行为与旧实现一致。"""
    head = make_head()
    x = torch.randn(5 + 3 + 1, VOCAB)    # 三段：5 / 3 / 1 行
    set_context(True, cu_seqlens_q=torch.tensor([0, 5, 8, 9], dtype=torch.int32))
    out = head(x)
    expect = F.linear(x[[4, 7, 8]], head.weight)
    assert out.shape == (3, VOCAB), out.shape
    assert torch.allclose(out, expect, atol=1e-5), (out - expect).abs().max()
    reset_context()
    print("  ok 常规 prefill 仅保留每段末行")


def test_lmhead_spec_segment_keeps_all_rows():
    """验证步：带草稿段（spec_flags=True）保留整段行，其余段仅末行。"""
    head = make_head()
    x = torch.randn(5 + 3 + 1, VOCAB)    # 段0 普通 prefill(5)，段1 草稿段(3)，段2 单行(1)
    cu = torch.tensor([0, 5, 8, 9], dtype=torch.int32)
    flags = torch.tensor([False, True, False])
    set_context(True, cu_seqlens_q=cu, spec_flags=flags)
    out = head(x)
    keep_idx = [4, 5, 6, 7, 8]
    expect = F.linear(x[keep_idx], head.weight)
    assert out.shape == (5, VOCAB), out.shape
    assert torch.allclose(out, expect, atol=1e-5), (out - expect).abs().max()
    reset_context()
    print("  ok 验证步：草稿段整段保留、普通段仅末行")


def test_lmhead_decode_no_slicing():
    """decode 步（is_prefill=False）：不做行过滤。"""
    head = make_head()
    x = torch.randn(4, VOCAB)
    set_context(False)
    out = head(x)
    assert out.shape == (4, VOCAB)
    assert torch.allclose(out, F.linear(x, head.weight), atol=1e-5)
    reset_context()
    print("  ok decode 步无行过滤")


class _FakeRunner:
    """提供 _sample_all/_accept_one/prepare_sample 所需的最小面（CPU 可跑）。"""

    def __init__(self):
        self.sampler = Sampler()

    _sample_all = ModelRunner._sample_all
    _accept_one = ModelRunner._accept_one

    @staticmethod
    def prepare_sample(seqs):
        """CPU 版五元组：与 ModelRunner.prepare_sample 同形同序，仅去掉 CUDA/pin。"""
        return (
            torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32),
            [seq.seq_id for seq in seqs],
            torch.tensor([seq.top_k for seq in seqs], dtype=torch.int64),
            torch.tensor([seq.top_p for seq in seqs], dtype=torch.float32),
            [seq.seed for seq in seqs],
        )


def _seq(temperature=0.0, top_k=0, top_p=1.0, seq_id=0, seed=None, drafts=None, n_scheduled=1):
    return SimpleNamespace(
        temperature=temperature, top_k=top_k, top_p=top_p,
        seq_id=seq_id, seed=seed,
        draft_tokens=list(drafts) if drafts else [],
        num_scheduled_tokens=n_scheduled,
    )


def test_sample_all_plain_layout():
    """纯 prefill/decode 批：每序列 1 行，行序即批序（不再按 seqlen 展开）。"""
    r = _FakeRunner()
    seqs = [_seq(seq_id=0, n_scheduled=100), _seq(seq_id=1, n_scheduled=1), _seq(seq_id=2, n_scheduled=1)]
    logits = torch.randn(3, VOCAB)
    logits[0].fill_(-5.0); logits[0, 7] = 9.0
    logits[1].fill_(-5.0); logits[1, 11] = 9.0
    logits[2].fill_(-5.0); logits[2, 3] = 9.0
    out = r._sample_all(seqs, logits)
    assert out == [[7], [11], [3]], out
    print("  ok 常规批行布局（1 行/序列，与旧 LM head 末行对齐）")


def test_sample_all_mixed_verify_layout():
    """混合验证批：[普通, 草稿(g=2)全接受, 普通]，游标 0 / 1..3 / 4。"""
    r = _FakeRunner()
    d1, d2, bonus = 21, 33, 45
    seqs = [_seq(seq_id=0, n_scheduled=50),
            _seq(seq_id=1, drafts=[d1, d2]),
            _seq(seq_id=2, n_scheduled=1)]
    logits = torch.randn(1 + 3 + 1, VOCAB)
    logits[0].fill_(-5.0); logits[0, 7] = 9.0          # 普通段输出行
    logits[1].fill_(-5.0); logits[1, d1] = 9.0         # 位置 L-1 预测 d1 -> 接受
    logits[2].fill_(-5.0); logits[2, d2] = 9.0         # 位置 L   预测 d2 -> 接受
    logits[3].fill_(-5.0); logits[3, bonus] = 9.0      # 位置 L+1 -> bonus
    logits[4].fill_(-5.0); logits[4, 11] = 9.0         # 普通段输出行
    out = r._sample_all(seqs, logits)
    assert out == [[7], [d1, d2, bonus], [11]], out
    print("  ok 混合验证批：全接受时产出 草稿+bonus，行游标正确")


def test_sample_all_mixed_verify_partial_reject():
    """部分拒绝：第 2 个草稿不匹配时，补目标分布 argmax 并截断。"""
    r = _FakeRunner()
    d1, reject_at, other = 21, 40, 52
    seqs = [_seq(seq_id=1, drafts=[d1, reject_at])]
    logits = torch.randn(3, VOCAB)
    logits[0].fill_(-5.0); logits[0, d1] = 9.0         # 接受 d1
    logits[1].fill_(-5.0); logits[1, other] = 9.0      # 目标预测 other != reject_at -> 拒绝
    logits[2].fill_(-5.0); logits[2, 63] = 9.0         # 该行不应被使用（bonus 位置被拒绝截断）
    out = r._sample_all(seqs, logits)
    assert out == [[d1, other]], out
    print("  ok 部分拒绝：草稿前缀 + 目标 argmax 补位")


if __name__ == "__main__":
    test_lmhead_plain_prefill_keeps_last_row()
    test_lmhead_spec_segment_keeps_all_rows()
    test_lmhead_decode_no_slicing()
    test_sample_all_plain_layout()
    test_sample_all_mixed_verify_layout()
    test_sample_all_mixed_verify_partial_reject()
    print("ALL LOGIT LAYOUT TESTS PASSED")
