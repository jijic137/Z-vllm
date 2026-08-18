"""Scheduler 纯逻辑单测（无需 GPU）。运行：python tests/test_scheduler.py

覆盖混合调度语义：decode 优先、prefill/decode 同批混合（is_prefill 决定 varlen 路径）、
任意 waiting 序列可切块、decode 受步预算约束、块耗尽抢占 + prefix cache 部分兜底、
KV 装不下时的明确 RuntimeError（替代旧版 assert 崩溃）。
"""
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


def test_waiting_chunking():
    # 任意 waiting 序列都可切块（不再是"仅队首"）：队首占满预算后，
    # 下一条用剩余预算取 min(需要, 剩余)，跨步续传（块在首步即全量分配）
    sched = make_scheduler(budget=16, max_seqs=4)
    from zvllm.sampling_params import SamplingParams
    s2 = Sequence(list(range(100, 112)), SamplingParams())   # 12 token
    s3 = Sequence(list(range(200, 208)), SamplingParams())   # 8 token
    sched.add(s2)
    sched.add(s3)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s2, s3]
    assert s2.num_scheduled_tokens == 12
    assert s3.num_scheduled_tokens == 4, "s3 应只取剩余预算 4，实际 %d" % s3.num_scheduled_tokens
    sched.postprocess(seqs, [998, 998], is_prefill)
    assert s2 in sched.running
    assert s3 in sched.waiting and len(s3.block_table) == 2, "切块序列保留已分配块"
    assert len(s3) == 8, "部分 prefill 不得追加 token"
    # 下一步混合：s2 decode 1 + s3 补齐剩余 4 token
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s2, s3]
    assert not s2.is_prefill and s2.num_scheduled_tokens == 1
    assert s3.is_prefill and s3.num_scheduled_tokens == 4
    sched.postprocess(seqs, [998, 998], is_prefill)
    assert s3 in sched.running and len(s3) == 9
    assert len(s2) == 14
    print("test_waiting_chunking OK")


def test_mixed_step():
    # decode 优先：running 的 decode 与 waiting 的 prefill 同批执行，
    # 批内任一条是 prefill 即走 varlen 路径（is_prefill=True）
    sched = make_scheduler()
    s1 = full_prefill(sched, list(range(6)))
    from zvllm.sampling_params import SamplingParams
    s2 = Sequence(list(range(50, 56)), SamplingParams())
    sched.add(s2)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s1, s2]
    assert not s1.is_prefill and s1.num_scheduled_tokens == 1
    assert s2.is_prefill and s2.num_scheduled_tokens == 6
    sched.postprocess(seqs, [7, 8], is_prefill)
    assert len(s1) == 8 and len(s2) == 7
    assert list(sched.running) == [s1, s2]
    print("test_mixed_step OK")


def test_decode_budget_cap():
    # decode 受本步 token 预算约束；未调度到的序列按原顺序留在 running 队后
    sched = make_scheduler(budget=3, max_seqs=4)
    from zvllm.sampling_params import SamplingParams
    s1 = Sequence([10], SamplingParams(max_tokens=8))
    s2 = Sequence([20], SamplingParams(max_tokens=8))
    s3 = Sequence([30], SamplingParams(max_tokens=8))
    for s in (s1, s2, s3):
        sched.add(s)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [s1, s2, s3]
    sched.postprocess(seqs, [4, 5, 6], is_prefill)
    assert list(sched.running) == [s1, s2, s3]
    # budget=3：纯 decode 步一次出 3 条
    seqs, is_prefill = sched.schedule()
    assert not is_prefill and seqs == [s1, s2, s3]
    assert all(s.num_scheduled_tokens == 1 for s in seqs)
    sched.postprocess(seqs, [7, 8, 9], is_prefill)
    # budget=2：每步最多 2 条 decode，s3 留在队后
    sched.max_num_batched_tokens = 2
    for tokens in ((10, 11), (12, 13)):
        seqs, is_prefill = sched.schedule()
        assert not is_prefill and seqs == [s1, s2], \
            f"budget=2 应只 decode 2 条：{[s.seq_id for s in seqs]}"
        sched.postprocess(seqs, list(tokens), is_prefill)
        assert list(sched.running) == [s1, s2, s3], "未调度的 s3 应保持队后顺序"
    print("test_decode_budget_cap OK")


def test_unfittable_sequence_clear_error():
    # KV cache 连队首序列都装不下时，给出明确 RuntimeError 而非 assert 崩溃
    sched = make_scheduler(num_blocks=2)
    from zvllm.sampling_params import SamplingParams
    big = Sequence(list(range(40)), SamplingParams())   # 需 10 块 > 总量 2 块
    sched.add(big)
    try:
        sched.schedule()
        raise AssertionError("schedule() 应抛出 RuntimeError")
    except RuntimeError as e:
        assert "KV" in str(e), str(e)
    print("test_unfittable_sequence_clear_error OK")


def test_preemption_and_prefix_cache_partial_rescue():
    # 块耗尽 -> 抢占最年轻 running；被抢占者重 prefill 时 prefix cache 尽力兜底：
    # 本场景中 a1 的 decode 新块恰好复用了 a2 的第 2 个缓存块（哈希被清除），
    # 因此只有第 1 个满块命中，a2 需补算 5 token（确定性行为，非 best-effort 抖动）
    sched = make_scheduler(num_blocks=4, budget=8)
    from zvllm.sampling_params import SamplingParams
    a1 = Sequence(list(range(4)), SamplingParams(max_tokens=6))
    a2 = Sequence(list(range(100, 104)), SamplingParams(max_tokens=8))
    sched.add(a1)
    sched.add(a2)
    # 第 1 步：prefill 4+4=8，两条同批完成
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [a1, a2]
    sched.postprocess(seqs, [1, 2], is_prefill)
    # 第 2 步各自拿到第 2 块（4 块耗尽）；第 3-5 步同块内 decode
    for first in (3, 5, 7, 9):
        seqs, is_prefill = sched.schedule()
        assert not is_prefill and seqs == [a1, a2]
        sched.postprocess(seqs, [first, first + 1], is_prefill)
    assert len(a1.block_table) == 2 and len(a2.block_table) == 2
    assert len(sched.block_manager.free_block_ids) == 0
    victim_block = a2.block_table[1]    # a2 的第 2 块：将被 a1 的 decode 复用而失去缓存
    # 第 6 步：a1 的 decode 跨块需要新块 -> 空闲耗尽 -> 抢占最年轻的 a2
    seqs, is_prefill = sched.schedule()
    assert not is_prefill and seqs == [a1], f"a2 应被抢占：{[s.seq_id for s in seqs]}"
    assert a2 in sched.waiting and a2.status == SequenceStatus.WAITING
    assert not a2.block_table
    assert a1.block_table[-1] == victim_block, "a1 的新块应恰好复用 a2 的第 2 块"
    sched.postprocess(seqs, [3], is_prefill)                      # a1 completions=6 -> 结束并释放
    assert a1.is_finished
    assert len(sched.block_manager.free_block_ids) == 4
    # 重调度：仅第 1 个满块命中 prefix cache（第 2 块已被复用清除），补算 5 token
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [a2]
    assert a2.num_scheduled_tokens == 5, f"应只补算 5 token，实际 {a2.num_scheduled_tokens}"
    assert a2.num_cached_tokens == 4
    sched.postprocess(seqs, [998], is_prefill)
    assert a2 in sched.running and len(a2) == 10
    assert a2.completion_token_ids == [2, 4, 6, 8, 10, 998]
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
    # 队首可分块但第二条装不下：第二条停等；队首结束后 running 为空、
    # 新队首仍装不下 -> RuntimeError（旧实现在此处 assert 崩溃）
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
    try:
        sched.schedule()
        raise AssertionError("应抛出 RuntimeError（KV cache 太小）")
    except RuntimeError as e:
        assert "KV" in str(e), str(e)
    print("test_wait_head_blocks_prefill OK")


if __name__ == "__main__":
    test_prefill_basic()
    test_chunked_prefill()
    test_waiting_chunking()
    test_mixed_step()
    test_decode_budget_cap()
    test_unfittable_sequence_clear_error()
    test_preemption_and_prefix_cache_partial_rescue()
    test_eos_and_max_tokens()
    test_wait_head_blocks_prefill()
    print("ALL SCHEDULER TESTS PASSED")
