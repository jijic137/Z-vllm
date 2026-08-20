"""停止串（stop strings）与逐请求取消单测（无需 GPU）。运行：python tests/test_stop_strings.py

覆盖：
1) SamplingParams.stop 归一化（str -> list、None、非法元素拒绝）；
2) find_stop_match 纯函数（OpenAI 语义：生成文本以停止串结尾才命中，命中串保留在输出）；
3) Scheduler.finish / abort：释放 KV 块、移出队列、幂等（含 chunked prefill 半途取消）；
4) 引擎级停止串集成（真 Scheduler + 假 tokenizer：命中即终止、finish_reason="stop"、块释放）；
5) LLMEngine.abort_request：waiting / running 均可取消，已结束 / 未知 id 返回 False。
"""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zvllm.sampling_params import SamplingParams, find_stop_match
from zvllm.engine.sequence import Sequence
from zvllm.engine.scheduler import Scheduler
from zvllm.engine.llm_engine import LLMEngine

BS = 4
Sequence.block_size = BS


def make_scheduler(num_blocks=16, budget=8, max_seqs=4):
    cfg = SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=budget,
                          eos=999, kvcache_block_size=BS, num_kvcache_blocks=num_blocks)
    return Scheduler(cfg)


class CharTokenizer:
    """token id -> 单字符（id % 26 映射 A-Z），使停止串检查可构造、可断言。"""

    def decode(self, token_ids):
        return "".join(chr(65 + (i % 26)) for i in token_ids)


def make_engine(**kwargs):
    """无 GPU 的 LLMEngine 外壳：真 Scheduler + 假 tokenizer（只覆盖 stop/abort 路径）。"""
    engine = object.__new__(LLMEngine)
    engine.tokenizer = CharTokenizer()
    engine.scheduler = make_scheduler(**kwargs)
    return engine


def _expect_reject(stop):
    try:
        SamplingParams(stop=stop)
    except AssertionError as e:
        assert "非空字符串" in str(e), str(e)
        return
    raise AssertionError(f"stop={stop!r} 应被拒绝")


def test_stop_normalization():
    assert SamplingParams(stop="abc").stop == ["abc"]
    assert SamplingParams(stop=["a", "b"]).stop == ["a", "b"]
    sp = SamplingParams(stop=["a", "b"])
    sp.stop.append("c")   # 归一化产生独立 list，不污染调用方传入的列表
    assert SamplingParams(stop=["a", "b"]).stop == ["a", "b"]
    assert SamplingParams().stop is None
    _expect_reject("")
    _expect_reject(["a", ""])
    _expect_reject([123])
    print("test_stop_normalization OK")


def test_find_stop_match():
    assert find_stop_match("Hello world", ["world"]) == "world"
    assert find_stop_match("Hello", ["world"]) is None
    assert find_stop_match("para1\n\n", ["\n\n"]) == "\n\n"
    assert find_stop_match("", ["x"]) is None
    assert find_stop_match("abcXdef", ["X"]) is None, "只有尾部匹配才命中"
    assert find_stop_match("xy", ["y", "xy"]) == "y", "按列表顺序返回第一个命中"
    assert find_stop_match("END", ["END", "D"]) == "END"
    print("test_find_stop_match OK")


def test_abort_waiting_and_idempotent():
    sched = make_scheduler()
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=8))
    sched.add(seq)
    sched.abort(seq)
    assert seq.is_finished and seq.finish_reason == "abort"
    assert seq not in sched.waiting and seq not in sched.running
    assert len(sched.block_manager.free_block_ids) == 16
    sched.abort(seq)   # 幂等：重复取消不报错、不重复释放
    assert len(sched.block_manager.free_block_ids) == 16
    print("test_abort_waiting_and_idempotent OK")


def test_abort_running_frees_blocks():
    sched = make_scheduler()
    seq = Sequence(list(range(6)), SamplingParams(max_tokens=8))   # 6 token -> 2 块
    sched.add(seq)
    seqs, is_prefill = sched.schedule()
    sched.postprocess(seqs, [7], is_prefill)
    assert seq in sched.running and len(seq.block_table) == 2
    assert len(sched.block_manager.free_block_ids) == 14
    sched.abort(seq)
    assert seq.is_finished and seq.finish_reason == "abort"
    assert not sched.running and not sched.waiting
    assert len(sched.block_manager.free_block_ids) == 16, "取消须立即释放 KV 块"
    # 释放的块可被后续请求重新分配
    seq2 = Sequence(list(range(100, 106)), SamplingParams(max_tokens=8))
    sched.add(seq2)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [8], True)
    assert seq2 in sched.running and len(seq2.block_table) == 2
    print("test_abort_running_frees_blocks OK")


def test_abort_chunked_prefill_midway():
    # chunked prefill 半途取消：首步即全量分配的块同样要释放
    sched = make_scheduler(budget=4)
    seq = Sequence(list(range(8)), SamplingParams(max_tokens=8))    # 8 token，预算 4
    sched.add(seq)
    seqs, is_prefill = sched.schedule()
    assert seqs == [seq] and seq.num_scheduled_tokens == 4
    sched.postprocess(seqs, [7], is_prefill)
    assert seq in sched.waiting and len(seq.block_table) == 2, \
        "半程 prefill 的序列留在 waiting，块在首步已全量分配"
    sched.abort(seq)
    assert seq.is_finished and not seq.block_table
    assert len(sched.block_manager.free_block_ids) == 16
    print("test_abort_chunked_prefill_midway OK")


def test_finish_reasons_explicit():
    # postprocess 显式记录结束原因：eos 与 max_tokens 同时命中时 length 优先
    #（与旧 step() 的推导口径一致），否则按 eos -> "stop"
    sched = make_scheduler()
    s1 = Sequence([1, 2], SamplingParams(max_tokens=1))
    sched.add(s1)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [999], True)          # eos 且 max_tokens 同时命中
    assert s1.is_finished and s1.finish_reason == "length"
    s2 = Sequence([3, 4], SamplingParams(max_tokens=8))
    sched.add(s2)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [999], True)          # 仅 eos
    assert s2.is_finished and s2.finish_reason == "stop"
    print("test_finish_reasons_explicit OK")


def test_engine_stop_string_hit():
    engine = make_engine()
    seq = Sequence([0], SamplingParams(max_tokens=8, stop=["CD"]))
    engine.scheduler.add(seq)
    seqs, is_prefill = engine.scheduler.schedule()
    engine.scheduler.postprocess(seqs, [21], is_prefill)   # 第 1 个生成 token 'V'
    engine._check_stop_strings(seqs)
    assert not seq.is_finished, "V 尚不以 CD 结尾"
    seqs, _ = engine.scheduler.schedule()
    engine.scheduler.postprocess(seqs, [2], False)         # 'C' -> "VC"
    engine._check_stop_strings(seqs)
    assert not seq.is_finished
    seqs, _ = engine.scheduler.schedule()
    engine.scheduler.postprocess(seqs, [3], False)         # 'D' -> "VCD"
    engine._check_stop_strings(seqs)
    assert seq.is_finished and seq.finish_reason == "stop"
    assert seq.stop_reason == "CD"
    assert seq.completion_token_ids == [21, 2, 3], "停止串本身保留在输出中"
    assert not engine.scheduler.running and not engine.scheduler.waiting
    assert len(engine.scheduler.block_manager.free_block_ids) == 16
    print("test_engine_stop_string_hit OK")


def test_engine_no_stop_no_interference():
    engine = make_engine()
    seq = Sequence([0], SamplingParams(max_tokens=8))   # 未设置 stop
    engine.scheduler.add(seq)
    seqs, is_prefill = engine.scheduler.schedule()
    engine.scheduler.postprocess(seqs, [21], is_prefill)
    engine._check_stop_strings(seqs)
    assert not seq.is_finished and seq.finish_reason is None
    print("test_engine_no_stop_no_interference OK")


def test_engine_abort_request():
    engine = make_engine()
    seq = Sequence([1], SamplingParams(max_tokens=8))
    engine.scheduler.add(seq)
    assert engine.abort_request(seq.seq_id) is True
    assert seq.is_finished and seq.finish_reason == "abort"
    assert engine.abort_request(seq.seq_id) is False, "已取消的请求返回 False"
    assert engine.abort_request(99999) is False, "未知 id 返回 False"
    # running 中的请求也可取消
    seq2 = Sequence(list(range(6)), SamplingParams(max_tokens=8))
    engine.scheduler.add(seq2)
    seqs, _ = engine.scheduler.schedule()
    engine.scheduler.postprocess(seqs, [7], True)
    assert seq2 in engine.scheduler.running
    assert engine.abort_request(seq2.seq_id) is True
    assert len(engine.scheduler.block_manager.free_block_ids) == 16
    print("test_engine_abort_request OK")


if __name__ == "__main__":
    test_stop_normalization()
    test_find_stop_match()
    test_abort_waiting_and_idempotent()
    test_abort_running_frees_blocks()
    test_abort_chunked_prefill_midway()
    test_finish_reasons_explicit()
    test_engine_stop_string_hit()
    test_engine_no_stop_no_interference()
    test_engine_abort_request()
    print("ALL STOP-STRING/CANCEL TESTS PASSED")
