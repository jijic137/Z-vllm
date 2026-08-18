"""Scheduler 纯逻辑单测（无需 GPU）。运行：python tests/test_scheduler.py

覆盖当前简化版调度语义（prefill/decode 不混合、仅队首可切块）。
实现"完整 chunked prefill + 混合调度"后，需同步更新本文件中标注 [OLD] 的用例。"""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zvllm.engine.sequence import Sequence, SequenceStatus
from zvllm.engine.scheduler import Scheduler

BS = 4
Sequence.block_size = BS


def make_scheduler(num_blocks=32, budget=8, max_seqs=2):
    cfg = SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=budget,
                          eos=999, kvcache_block_size=BS, num_kvcache_blocks=num_blocks)
    return Scheduler(cfg)


def full_prefill(sched, tokens, max_tokens=64, **sp):
    """把一条序列喂到 prefill 完成（可能跨多个 chunk），返回 seq"""
    from zvllm.sampling_params import SamplingParams
    seq = Sequence(tokens, SamplingParams(max_tokens=max_tokens, **sp))
    sched.add(seq)
    while seq in sched.waiting:
        seqs, is_prefill = sched.schedule()
        assert is_prefill and seq in seqs
        done = seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
        sched.postprocess(seqs, [998] * len(seqs), is_prefill)
        if done:
            assert seq in sched.running
    return seq


def decode_step(sched, seq, token=7):
    seqs, is_prefill = sched.schedule()
    assert not is_prefill and seqs == [seq]
    sched.postprocess(seqs, [token], is_prefill)
    return seqs


def test_prefill_basic():
    sched = make_scheduler()
    seq = full_prefill(sched, list(range(6)))
    assert seq.num_cached_tokens == 6 and len(seq) == 7
    assert seq.status == SequenceStatus.RUNNING
    assert not sched.waiting and list(sched.running) == [seq]
    assert sum(len(s.block_table) for s in sched.running) == 2  # 7 token -> 2 块
    print("test_prefill_basic OK")


def test_chunked_prefill():
    sched = make_scheduler(budget=8)
    from zvllm.sampling_params import SamplingParams
    seq = Sequence(list(range(20)), SamplingParams())
    sched.add(seq)
    # 20 token 分 3 步：8 + 8 + 4
    for chunk, cached_after in ((8, 8), (8, 16), (4, 20)):
        seqs, is_prefill = sched.schedule()
        assert is_prefill and seqs == [seq]
        assert seq.num_scheduled_tokens == chunk, (seq.num_scheduled_tokens, chunk)
        sched.postprocess(seqs, [998], is_prefill)
        assert seq.num_cached_tokens == cached_after
        if cached_after < 20:
            assert seq in sched.waiting, "未完成 prefill 的序列应留在 waiting"
            assert len(seq) == 20, "部分 prefill 不得追加 token"
    assert seq in sched.running and len(seq) == 21
    print("test_chunked_prefill OK")


def test_only_head_chunkable():
    # [OLD] 现有语义：仅队首允许切块；非队首必须整条塞进剩余预算，否则停等
    # （混合调度改造后：队首之后还能继续调度其他序列，见第 3 项）
    sched = make_scheduler(budget=16, max_seqs=4)
    from zvllm.sampling_params import SamplingParams
    s2 = Sequence(list(range(100, 112)), SamplingParams())   # 12 token
    s3 = Sequence(list(range(200, 208)), SamplingParams())   # 8 token
    sched.add(s2)
    sched.add(s3)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s2], f"非队首 s3 不应被部分调度：{[s.seq_id for s in seqs]}"
    assert s2.num_scheduled_tokens == 12
    sched.postprocess(seqs, [998], is_prefill)
    assert s2 in sched.running
    assert s3 in sched.waiting and not s3.block_table
    # 下一步 s3 成为队首，一次 prefill 完成
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s3] and s3.num_scheduled_tokens == 8
    sched.postprocess(seqs, [998], is_prefill)
    assert s3 in sched.running
    print("test_only_head_chunkable OK")


def test_prefill_decode_not_mixed():
    # [OLD] 现有语义：只要 waiting 非空，本步就全是 prefill，decode 让路
    sched = make_scheduler(budget=8)
    s1 = full_prefill(sched, list(range(6)))
    from zvllm.sampling_params import SamplingParams
    s2 = Sequence(list(range(50, 56)), SamplingParams())
    sched.add(s2)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s2] and s1 not in seqs
    sched.postprocess(seqs, [998], is_prefill)
    # 下一步 waiting 空了才 decode（s1、s2 一起）
    seqs, is_prefill = sched.schedule()
    assert not is_prefill and seqs == [s1, s2]
    sched.postprocess(seqs, [7, 8], is_prefill)
    print("test_prefill_decode_not_mixed OK")


def test_preemption_and_prefix_cache_partial_rescue():
    # 块耗尽 → 抢占最年轻 running；被抢占者重新 prefill 时 prefix cache 尽力兜底：
    # 本场景中 s1 的 decode 新块恰好复用了 s2 的第 2 个缓存块（哈希被清除），
    # 因此只有第 1 个满块命中，s2 需补算 5 token（确定性行为，非 best-effort 抖动）
    sched = make_scheduler(num_blocks=6, budget=8)
    from zvllm.sampling_params import SamplingParams
    s1 = full_prefill(sched, list(range(16)), max_tokens=2)      # 4 块，len=17
    s2 = Sequence(list(range(100, 108)), SamplingParams(max_tokens=8))  # 2 块
    sched.add(s2)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s2]
    sched.postprocess(seqs, [998], is_prefill)                    # len(s2)=9, 2 块，共 6 块耗尽
    assert len(s1.block_table) == 4 and len(s2.block_table) == 2
    assert len(sched.block_manager.free_block_ids) == 0
    victim_block = s2.block_table[1]    # s2 的第 2 块：将被 s1 的 decode 复用而失去缓存
    # s1 跨块 decode 需要新块 -> 抢占最年轻的 s2
    seqs, is_prefill = sched.schedule()
    assert not is_prefill and seqs == [s1], f"s2 应被抢占：{[s.seq_id for s in seqs]}"
    assert s2 in sched.waiting and s2.status == SequenceStatus.WAITING
    assert not s2.block_table
    assert s1.block_table[-1] == victim_block, "s1 的新块应恰好复用 s2 的第 2 块"
    sched.postprocess(seqs, [1], is_prefill)                      # s1 completions=2 -> 结束并释放
    assert s1.is_finished
    assert len(sched.block_manager.free_block_ids) == 6
    # 重调度：仅第 1 个满块命中 prefix cache（第 2 块已被复用清除），补算 5 token
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s2]
    assert s2.num_scheduled_tokens == 5, f"应只补算 5 token，实际 {s2.num_scheduled_tokens}"
    assert s2.num_cached_tokens == 4
    sched.postprocess(seqs, [998], is_prefill)
    assert s2 in sched.running and len(s2) == 10 and s2.completion_token_ids == [998, 998]
    print("test_preemption_and_prefix_cache_partial_rescue OK")


def test_eos_and_max_tokens():
    sched = make_scheduler()
    from zvllm.sampling_params import SamplingParams
    # eos 命中即结束
    s1 = Sequence([1, 2, 3], SamplingParams(max_tokens=8))
    sched.add(s1)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [999], True)
    assert s1.is_finished and not sched.running
    assert len(sched.block_manager.free_block_ids) == 32
    # max_tokens 到点结束
    s2 = Sequence([4, 5], SamplingParams(max_tokens=3))
    sched.add(s2)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [7], True)
    decode_step(sched, s2, token=8)
    assert not s2.is_finished
    decode_step(sched, s2, token=9)
    assert s2.is_finished and not sched.running
    # ignore_eos 时 eos 不结束
    s3 = Sequence([6, 7], SamplingParams(max_tokens=2, ignore_eos=True))
    sched.add(s3)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [999], True)
    assert not s3.is_finished
    print("test_eos_and_max_tokens OK")


def test_wait_head_blocks_prefill():
    # [OLD] 队首分不到块 -> prefill 阶段停等；若 running 也空了，现有实现直接
    # assert 崩溃（文档 §6.4 已知问题）。这里把两个边界都固化下来。
    sched = make_scheduler(num_blocks=2, budget=8)
    from zvllm.sampling_params import SamplingParams
    s_run = Sequence(list(range(100, 104)), SamplingParams(max_tokens=1))
    s_big = Sequence(list(range(40)), SamplingParams())   # 需 10 块
    sched.add(s_run)
    sched.add(s_big)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s_run], "队首 s_run 可分配，s_big 停等"
    sched.postprocess(seqs, [7], is_prefill)              # max_tokens=1 -> 结束
    assert s_run.is_finished
    # 现在队首 s_big 分不到块且 running 为空 -> 现有实现 assert 崩溃
    try:
        sched.schedule()
        raise SystemExit("应触发 assert scheduled_seqs（已知限制）")
    except AssertionError:
        pass
    print("test_wait_head_blocks_prefill OK")


if __name__ == "__main__":
    test_prefill_basic()
    test_chunked_prefill()
    test_only_head_chunkable()
    test_prefill_decode_not_mixed()
    test_preemption_and_prefix_cache_partial_rescue()
    test_eos_and_max_tokens()
    test_wait_head_blocks_prefill()
    print("ALL SCHEDULER TESTS PASSED")
