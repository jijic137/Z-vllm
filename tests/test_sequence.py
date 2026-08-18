"""Sequence 纯逻辑单测（无需 GPU）。运行：python tests/test_sequence.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zvllm.engine.sequence import Sequence, SequenceStatus
from zvllm.sampling_params import SamplingParams

Sequence.block_size = 4


def test_basics():
    seq = Sequence([10, 11, 12])
    assert len(seq) == 3
    assert seq[0] == 10 and seq[-1] == 12
    assert seq.last_token == 12
    assert seq.num_prompt_tokens == 3
    assert seq.prompt_token_ids == [10, 11, 12]
    assert seq.completion_token_ids == []
    assert seq.num_blocks == 1 and seq.last_block_num_tokens == 3
    assert seq.block(0) == [10, 11, 12]
    assert not seq.is_finished
    assert seq.status == SequenceStatus.WAITING
    print("test_basics OK")


def test_block_math():
    # 覆盖 prompt 长度 % block_size == 1 的回归场景（上游 issue #240/#66 的根源）
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17):
        seq = Sequence(list(range(n)))
        assert seq.num_blocks == (n + 3) // 4, n
        assert seq.last_block_num_tokens == n - (seq.num_blocks - 1) * 4, n
        assert 1 <= seq.last_block_num_tokens <= 4, n
        for i in range(seq.num_blocks):
            assert seq.block(i) == list(range(i * 4, min((i + 1) * 4, n))), (n, i)
    print("test_block_math OK")


def test_append():
    seq = Sequence([100, 101])
    for t in (7, 8, 9, 10):
        seq.append_token(t)
    assert len(seq) == 6
    assert seq.last_token == 10
    assert seq.completion_token_ids == [7, 8, 9, 10]
    assert seq.num_blocks == 2
    print("test_append OK")


def test_pickle_roundtrip():
    # engine 在多卡时会通过共享内存 pickle 传递 Sequence
    s1 = Sequence([1, 2, 3, 4, 5, 6])
    s1.block_table = [7, 3]
    s1.num_cached_tokens = 4
    import pickle
    s2 = pickle.loads(pickle.dumps(s1))
    assert s2.token_ids == [1, 2, 3, 4, 5, 6]
    assert s2.block_table == [7, 3]
    assert s2.num_cached_tokens == 4
    # decode 阶段只传 last_token，token_ids 为空
    s1.is_prefill = False
    s3 = pickle.loads(pickle.dumps(s1))
    assert s3.last_token == 6 and s3.token_ids == []
    assert len(s3) == 6
    print("test_pickle_roundtrip OK")


def test_sampling_params():
    sp = SamplingParams(temperature=0, top_k=1, top_p=1.0, seed=42)
    seq = Sequence([1], sp)
    assert (seq.temperature, seq.top_k, seq.top_p, seq.seed) == (0, 1, 1.0, 42)
    try:
        SamplingParams(temperature=-1)
        raise SystemExit("应拒绝 temperature < 0")
    except AssertionError:
        pass
    try:
        SamplingParams(top_p=0.0)
        raise SystemExit("应拒绝 top_p <= 0")
    except AssertionError:
        pass
    try:
        SamplingParams(top_k=-2)
        raise SystemExit("应拒绝 top_k < -1")
    except AssertionError:
        pass
    print("test_sampling_params OK")


if __name__ == "__main__":
    test_basics()
    test_block_math()
    test_append()
    test_pickle_roundtrip()
    test_sampling_params()
    print("ALL SEQUENCE TESTS PASSED")
