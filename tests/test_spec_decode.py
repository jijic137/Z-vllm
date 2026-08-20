"""投机解码（n-gram 草稿 + 接受-拒绝）纯 CPU 单测。运行：python tests/test_spec_decode.py

覆盖：
1) 贪心接受-拒绝与参考实现逐位一致（全接受 / 中途拒绝 / 首位拒绝 / γ=1）；
2) 采样接受-拒绝：seed 复现、输出不变量（bonus ≠ 被拒草稿、接受前缀 = 草稿前缀）；
3) 采样接受-拒绝的分布正确性：MC 统计输出频率 == 目标分布（残差采样数学）；
4) PLD 草稿生成：完整草稿 / 部分草稿 / 无匹配 / 跳过尾部自身出现（含全同 token 退化回归）；
5) 调度器：1+γ 计账、KV 预留、预算/块不足降级、max_tokens 余量上限、prefill 完成时建索引；
6) 回写口径：num_cached = 新长度-1（bonus KV 延迟）、哈希游标不含 bonus 块、
   prefill 完成恰满块时不再提前哈希（旧实现回归）、max_tokens/eos 截断接受列表；
7) 带草稿的 Sequence pickle 往返（TP worker 侧输入构造依赖）。

口径注意：prefill 完成必 append 一个首 token，故 prefill 后序列长 L = prompt_len + 1；
helper 的 first_token 参数控制该首 token（默认 998），chunked 半程步的 token 被忽略。
"""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from zvllm.sampling_params import SamplingParams
from zvllm.engine.sequence import Sequence
from zvllm.engine.scheduler import Scheduler
from zvllm.layers.sampler import Sampler
from zvllm.layers.spec_decode import accept_drafts

BS = 4
Sequence.block_size = BS
EOS = 999
PATTERN11 = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]


def make_scheduler(num_blocks=64, budget=64, max_seqs=4, **spec):
    kwargs = dict(spec_decode="off", spec_gamma=4, spec_ngram=4)
    kwargs.update(spec)
    cfg = SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=budget,
                          eos=EOS, kvcache_block_size=BS, num_kvcache_blocks=num_blocks,
                          max_model_len=4096, **kwargs)
    return Scheduler(cfg)


def prefill_until_running(sched, tokens, first_token=998, **sp):
    """跑完 chunked prefill 直到序列进入 running；每步"采样" first_token 作产出 token。

    prefill 完成步会 append 该首 token（最终 L = len(tokens)+1）；chunked 半程步
    postprocess 忽略传入 token（不并入序列）。"""
    seq = Sequence(tokens, SamplingParams(max_tokens=sp.pop("max_tokens", 64), **sp))
    sched.add(seq)
    while seq in sched.waiting:
        seqs, is_prefill = sched.schedule()
        assert is_prefill and seq in seqs
        sched.postprocess(seqs, [first_token] * len(seqs), is_prefill)
    return seq


# --------------------------------------------------------------- 接受-拒绝

def test_accept_greedy_matches_reference():
    # 参考贪心链：chain[pos] 是位置 pos 的目标 argmax
    chain = [3, 1, 4, 1, 5, 9, 2, 6, 8, 3]
    V, L = 10, 5      # 当前序列长 L（位置 0..L-1），验证位置 L-1..
    def rows_for(draft):
        # row i = 预测位置 L+i 的 logit，argmax = chain[L+i]
        rows = torch.full((len(draft) + 1, V), 0.0)
        for i in range(len(draft) + 1):
            rows[i, chain[L + i]] = 10.0
        return rows
    s = Sampler()
    for draft in ([9, 2, 6, 8], [9, 0, 6, 8], [0, 2, 6, 7], [9]):
        seq = Sequence(list(range(L)), SamplingParams(temperature=0))
        seq.draft_tokens = list(draft)
        out = accept_drafts(s, seq, rows_for(draft))
        # 参考：逐位与贪心链比对，首个不等处取目标 argmax 并停止
        ref = []
        for i, x in enumerate(draft):
            if x == chain[L + i]:
                ref.append(x)
            else:
                ref.append(chain[L + i])
                break
        else:
            ref.append(chain[L + len(draft)])
        assert out == ref, (draft, out, ref)
    print("  ok 贪心接受-拒绝与参考一致（全接受/中途拒绝/首位拒绝/γ=1）")


def test_accept_seeded_reproducible_and_invariants():
    V, L, g = 16, 6, 4
    seq = Sequence(list(range(L)), SamplingParams(temperature=1.0, seed=42))
    seq.draft_tokens = [5, 9, 2, 6]
    logits = torch.randn(g + 1, V)

    def run_once():
        s = Sampler()
        # 必须先注入 CPU generator，否则 _generator 会创建 CUDA generator（CPU 环境直接崩）
        s.generators[seq.seq_id] = torch.Generator().manual_seed(42)
        return accept_drafts(s, seq, logits.clone())

    o1, o2 = run_once(), run_once()
    assert o1 == o2, (o1, o2)
    assert 1 <= len(o1) <= g + 1
    k = len(o1) - 1
    assert o1[:k] == seq.draft_tokens[:k], "接受前缀必须是草稿前缀"
    if k < g:
        assert o1[k] != seq.draft_tokens[k], "bonus 不得等于被拒草稿 token（残差分布已清零）"
    print(f"  ok seed 复现 + 不变量（样本: {o1}, 接受 {k}/{g}）")


def test_accept_mc_distribution_matches_target():
    # γ=1：输出分布应严格等于目标分布 p（接受 p(x) / 拒绝后从 p 去掉 x 重归一采样）
    V, L = 4, 3
    logits = torch.tensor([[2.0, 0.5, 0.1, 0.0],
                           [2.0, 0.5, 0.1, 0.0]])     # 两行同分布（γ=1：2 行）
    p = torch.softmax(logits[0], dim=-1).tolist()
    s = Sampler()
    seq = Sequence(list(range(L)), SamplingParams(temperature=1.0))
    seq.draft_tokens = [1]
    N = 20000
    counts = [0] * V
    for _ in range(N):
        counts[accept_drafts(s, seq, logits)[0]] += 1
    freq = [c / N for c in counts]
    for i in range(V):
        assert abs(freq[i] - p[i]) < 0.03, (i, freq[i], p[i])
    print(f"  ok MC 分布正确（频率 {[round(f, 3) for f in freq]} vs p {[round(x, 3) for x in p]}）")


# --------------------------------------------------------------- PLD 草稿

def test_pld_draft_generation():
    # 完整草稿：尾 4-gram (1,2,3,4) 出现于 p=3,7,11（11 为尾部自身），p=7 后恰好 4 个
    sched = make_scheduler(spec_decode="ngram")
    seq = prefill_until_running(sched, PATTERN11, first_token=4)
    assert seq.ngram_positions is not None, "prefill 完成应建 n-gram 索引"
    assert sched._make_draft(seq) == [1, 2, 3, 4]
    # 部分草稿：尾 4-gram (4,1,2,4) 出现于 p=7,10（10 为尾部自身），p=7 后仅剩 3 个
    seq2 = prefill_until_running(make_scheduler(spec_decode="ngram"),
                                 [9, 8, 7, 6, 4, 1, 2, 4, 1, 2], first_token=4)
    assert sched._make_draft(seq2) == [1, 2, 4]
    # 无匹配
    seq3 = prefill_until_running(make_scheduler(spec_decode="ngram"), list(range(10, 30)))
    assert sched._make_draft(seq3) == []
    print("  ok PLD 草稿（完整/部分/无匹配）")


def test_pld_skips_trivial_tail_match():
    # 周期文本：n=2 时尾 (1,2) 出现于 p=1,3,5,7，p=7 是尾部自身出现须跳过；
    # p=5 后仅剩 2 个（不足 γ），p=3 后恰好 4 个 -> 应取 [1,2,1,2]
    sched = make_scheduler(spec_decode="ngram", spec_ngram=2)
    seq = prefill_until_running(sched, [1, 2, 1, 2, 1, 2, 1], first_token=2)
    d = sched._make_draft(seq)
    assert d == [1, 2, 1, 2], f"应取 p=3 后的 [1,2,1,2]，实际 {d}"
    # 全同 token 退化：trivial 口径（gram 结束位置 L-1）写错会误杀真实出现
    sched2 = make_scheduler(spec_decode="ngram", spec_ngram=4)
    seq2 = prefill_until_running(sched2, [9] * 11, first_token=9)
    assert sched2._make_draft(seq2) == [9, 9, 9, 9]
    sched3 = make_scheduler(spec_decode="ngram", spec_ngram=2)
    seq3 = prefill_until_running(sched3, [9, 9, 9], first_token=9)
    assert sched3._make_draft(seq3) == [9]
    print("  ok PLD 跳过尾部自身出现（周期 + 全同退化回归）")


# --------------------------------------------------------------- 调度

def test_spec_schedule_budget_and_reservation():
    sched = make_scheduler(budget=16, spec_decode="ngram")
    seq = prefill_until_running(sched, PATTERN11, first_token=4)
    seqs, is_prefill = sched.schedule()
    assert seqs == [seq] and not is_prefill
    assert seq.draft_tokens == [1, 2, 3, 4]
    assert seq.num_scheduled_tokens == 5, "本步按 1+γ 计账"
    # KV 预留：块表须覆盖位置 L+γ-1 = 15（块索引 3，共 4 块）
    assert len(seq.block_table) >= 4, f"草稿 KV 未预留：{len(seq.block_table)} 块"
    assert sched.spec_stats["proposed_tokens"] == 4 and sched.spec_stats["steps"] == 1
    print("  ok 调度：1+γ 计账 + 草稿 KV 预留 + 统计")


def test_spec_schedule_budget_cap():
    sched = make_scheduler(budget=3, spec_decode="ngram")    # 本步最多 3 token -> 草稿 ≤ 2
    seq = prefill_until_running(sched, PATTERN11, first_token=4)   # chunked: 3+3+3+2
    seqs, _ = sched.schedule()
    assert seq.num_scheduled_tokens == 3
    assert seq.draft_tokens == [1, 2], f"预算应把草稿截到 2，实际 {seq.draft_tokens}"
    print("  ok 调度：预算截断草稿")


def test_spec_schedule_max_tokens_cap():
    sched = make_scheduler(spec_decode="ngram")
    # prefill 已产出首 token：max_tokens=2 时余量 1 -> 草稿必空（min 公式回归）
    seq = prefill_until_running(sched, PATTERN11, first_token=4, max_tokens=2)
    seqs, _ = sched.schedule()
    assert seq.draft_tokens == [] and seq.num_scheduled_tokens == 1
    print("  ok 调度：max_tokens 余量 ≤1 时不出草稿")


def test_spec_schedule_degrades_when_blocks_short():
    sched = make_scheduler(num_blocks=4, spec_decode="ngram")
    seq = prefill_until_running(sched, [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], first_token=1)
    # 12 token prefill 占 3 块、free=1；满草稿 4 需再 2 块 -> 逐级降级到 3
    seqs, _ = sched.schedule()
    assert seqs == [seq]
    assert seq.draft_tokens == [2, 3, 4]
    assert seq.num_scheduled_tokens == 4
    assert len(seq.block_table) == 4
    print("  ok 调度：块不足时草稿降级")


# --------------------------------------------------------------- 回写口径

def test_postprocess_spec_bookkeeping():
    sched = make_scheduler(spec_decode="ngram")
    seq = prefill_until_running(sched, PATTERN11, first_token=4)
    seqs, _ = sched.schedule()
    assert seq.draft_tokens == [1, 2, 3, 4]
    emitted = sched.postprocess_multi(seqs, [[1, 2, 9]], False)    # 接受 2 草稿 + bonus 9
    assert emitted == [[1, 2, 9]]
    assert seq.completion_token_ids == [4, 1, 2, 9]
    assert len(seq) == 15
    # bonus 的 KV 延迟：可信长度 = 新长度-1
    assert seq.num_cached_tokens == len(seq) - 1
    assert seq.num_hashed_tokens == len(seq) - 1
    # 含 bonus 的块（位置 14 在块 3）未填满，不得入哈希表
    assert sched.block_manager.blocks[seq.block_table[14 // BS]].hash == -1
    # 新官方位置 (4,1,2,9) 未构成任何复现 4-gram：下轮草稿应为空
    assert sched._make_draft(seq) == []
    assert sched.spec_stats["accepted_tokens"] == 2
    print("  ok 回写：num_cached=新长度-1、哈希不含 bonus 块、索引更新、统计")


def test_decode_block_boundary_not_hashed_early():
    # 回归：prefill 完成恰把总长度凑满整块时，块内最后 token（首 token，位置 3）
    # 的 KV 尚未写入（下一步 decode 回喂才写）；旧实现用"新长度"作可信边界会
    # 提前把该块入哈希表 -> 复用方读到垃圾 KV
    sched = make_scheduler()    # spec off
    seq = prefill_until_running(sched, [1, 2, 3])    # L 3->4：块 0 恰好填满
    bm = sched.block_manager
    assert len(seq.block_table) == 1
    assert bm.blocks[seq.block_table[0]].hash == -1, "KV 未写满的块不得入哈希表"
    assert seq.num_hashed_tokens == 3
    # 下一步 decode（重写位置 3 的 KV 后）块 0 才可哈希
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [7], False)              # L 4->5：位置 3 KV 已写
    assert bm.blocks[seq.block_table[0]].hash != -1, "位置 3 KV 写入后块 0 应入哈希表"
    print("  ok 回归：解码满块延迟哈希（KV 写入前不入 prefix cache）")


def test_postprocess_truncates_at_max_tokens_and_eos():
    # max_tokens：prefill 后余量 2 -> 草稿截到 1；接受列表 [1,2] 用尽余量 -> length
    sched = make_scheduler(spec_decode="ngram")
    seq = prefill_until_running(sched, PATTERN11, first_token=4, max_tokens=3)
    seqs, _ = sched.schedule()
    assert seq.draft_tokens == [1], f"余量 2 -> 草稿 ≤1，实际 {seq.draft_tokens}"
    assert seq.num_scheduled_tokens == 2
    emitted = sched.postprocess_multi(seqs, [[1, 2]], False)
    assert emitted == [[1, 2]] and seq.finish_reason == "length" and seq.is_finished
    assert seq.completion_token_ids == [4, 1, 2]
    # eos：另起 scheduler（上面的引擎已空，再 schedule 会报"无序列可调度"）。
    # 接受 1 + bonus 999=eos -> stop，其后不得再追加
    sched2 = make_scheduler(spec_decode="ngram")
    seq2 = prefill_until_running(sched2, PATTERN11, first_token=4)
    seqs2, _ = sched2.schedule()
    emitted2 = sched2.postprocess_multi(seqs2, [[1, EOS, 2]], False)
    assert emitted2 == [[1, EOS]] and seq2.finish_reason == "stop"
    assert seq2.completion_token_ids == [4, 1, EOS], "eos 之后不得再追加"
    print("  ok max_tokens/eos 截断接受列表")


def test_pickle_with_drafts():
    import pickle
    seq = Sequence(list(range(8)), SamplingParams(temperature=0))
    seq.is_prefill = False
    seq.block_table = [3, 7]
    seq.draft_tokens = [10, 11, 12]
    s2 = pickle.loads(pickle.dumps(seq))
    assert s2.draft_tokens == [10, 11, 12]
    assert s2.last_token == 7 and s2.token_ids == []
    assert s2.num_hashed_tokens == 0
    print("  ok 带草稿 Sequence pickle 往返（worker 侧输入构造）")


if __name__ == "__main__":
    test_accept_greedy_matches_reference()
    test_accept_seeded_reproducible_and_invariants()
    test_accept_mc_distribution_matches_target()
    test_pld_draft_generation()
    test_pld_skips_trivial_tail_match()
    test_spec_schedule_budget_and_reservation()
    test_spec_schedule_budget_cap()
    test_spec_schedule_max_tokens_cap()
    test_spec_schedule_degrades_when_blocks_short()
    test_postprocess_spec_bookkeeping()
    test_decode_block_boundary_not_hashed_early()
    test_postprocess_truncates_at_max_tokens_and_eos()
    test_pickle_with_drafts()
    print("ALL SPEC DECODE TESTS PASSED")
