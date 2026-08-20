"""step 三阶段拆分（schedule / run_model / finalize_step）单测（无需 GPU）。
运行：python tests/test_step_split.py

背景：api_server 的两阶段 step loop 需要引擎把原 step() 拆成
阶段 A（调度，CPU，持锁）/ 阶段 B（GPU，不持锁）/ 阶段 C（状态回写，CPU，持锁），
使 GPU 计算窗口内请求入队与取消不被阻塞。本文件用无 GPU 的引擎外壳
（真 Scheduler + 假 tokenizer + 假 model_runner）锁定三阶段契约：
1) step() 原子组合（CLI 路径）行为与拆分前一致（prefill/decode/max_tokens）；
2) 阶段 B 期间被取消的序列在阶段 C 被跳过（token 丢弃、状态不二次变更、
   其余序列正常推进）；
3) chunked prefill 半程步不产出 token（旧 step() 会把采样 token 误发给客户端，
   该 token 从未并入序列）。
"""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zvllm.sampling_params import SamplingParams
from zvllm.engine.sequence import Sequence
from zvllm.engine.scheduler import Scheduler
from zvllm.engine.llm_engine import LLMEngine

BS = 4
Sequence.block_size = BS


def make_scheduler(num_blocks=16, budget=8, max_seqs=4):
    cfg = SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=budget,
                          eos=999, kvcache_block_size=BS, num_kvcache_blocks=num_blocks)
    return Scheduler(cfg)


class FakeTokenizer:
    def decode(self, token_ids):
        return "".join(chr(65 + (i % 26)) for i in token_ids)


class FakeModelRunner:
    """假 model_runner：call("run", seqs, is_prefill) 为每条序列返回固定 token。"""

    def __init__(self, token=7):
        self.token = token

    def call(self, method, *args):
        assert method == "run", method
        seqs, is_prefill = args
        return [self.token] * len(seqs)


def make_engine(**kwargs):
    """无 GPU 的 LLMEngine 外壳：真 Scheduler + 假 tokenizer + 假 model_runner。"""
    engine = object.__new__(LLMEngine)
    engine.tokenizer = FakeTokenizer()
    engine.scheduler = make_scheduler(**kwargs)
    engine.model_runner = FakeModelRunner()
    return engine


def test_step_composition_cli_path():
    # step() = schedule → run_model → finalize_step：CLI 原子路径行为不变
    engine = make_engine()
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=4))
    engine.scheduler.add(seq)
    outputs, num_tokens = engine.step()          # prefill 步
    assert num_tokens == 3
    assert outputs == [(seq.seq_id, [7], False, None)]
    assert seq.completion_token_ids == [7]
    for _ in range(2):                           # decode 步 x2
        outputs, num_tokens = engine.step()
        assert num_tokens == -1
        assert outputs == [(seq.seq_id, [7], False, None)]
    outputs, num_tokens = engine.step()          # 第 4 个 token 命中 max_tokens
    assert outputs == [(seq.seq_id, [7], True, "length")]
    assert engine.is_finished()
    print("test_step_composition_cli_path OK")


def test_finalize_skips_seq_aborted_during_gpu_phase():
    # 阶段 B（GPU）期间被取消的序列：阶段 C 跳过其回写，token 丢弃，
    # 状态保持取消时的定型（不二次释放块），其余序列正常推进
    engine = make_engine()
    a = Sequence([1], SamplingParams(max_tokens=8))
    b = Sequence([2], SamplingParams(max_tokens=8))
    engine.scheduler.add(a)
    engine.scheduler.add(b)
    seqs, is_prefill = engine.schedule()
    assert [s.seq_id for s in seqs] == [a.seq_id, b.seq_id]
    token_ids = engine.run_model(seqs, is_prefill)      # 阶段 B："GPU 计算"
    assert engine.abort_request(a.seq_id) is True       # 阶段 B 期间被取消
    outputs = engine.finalize_step(seqs, is_prefill, token_ids)
    assert outputs == [(b.seq_id, [7], False, None)], "被取消序列的采样 token 应丢弃"
    assert a.is_finished and a.finish_reason == "abort"
    assert b.completion_token_ids == [7]
    assert len(engine.scheduler.block_manager.free_block_ids) == 15, \
        "a 的块由取消释放一次，b 仍持有 1 块"
    outputs, _ = engine.step()                # b 正常继续 decode
    assert outputs == [(b.seq_id, [7], False, None)]
    assert len(engine.scheduler.block_manager.free_block_ids) == 15
    print("test_finalize_skips_seq_aborted_during_gpu_phase OK")


def test_finalize_chunked_prefill_yields_no_token():
    # chunked prefill 半程步：该序列本步未产出 token，outputs 不含它，
    # 采样 token 不得混入序列（旧 step() 会误发该 token 给客户端）
    engine = make_engine(budget=4)
    seq = Sequence(list(range(8)), SamplingParams(max_tokens=8))   # 8 token，预算 4
    engine.scheduler.add(seq)
    seqs, is_prefill = engine.schedule()
    assert seqs == [seq] and seq.num_scheduled_tokens == 4
    token_ids = engine.run_model(seqs, is_prefill)
    outputs = engine.finalize_step(seqs, is_prefill, token_ids)
    assert outputs == []
    assert seq.completion_token_ids == []
    assert seq in engine.scheduler.waiting and not seq.is_finished
    outputs, num_tokens = engine.step()       # 第二步走完 prefill，才首次产出 token
    assert num_tokens == 4
    assert outputs == [(seq.seq_id, [7], False, None)]
    assert seq.completion_token_ids == [7]
    assert seq in engine.scheduler.running
    print("test_finalize_chunked_prefill_yields_no_token OK")


def test_finalize_all_aborted_empty_batch():
    # 极端情况：批内序列在阶段 B 期间全部被取消，阶段 C 产出空输出、不崩溃
    engine = make_engine()
    a = Sequence([1], SamplingParams(max_tokens=8))
    engine.scheduler.add(a)
    seqs, is_prefill = engine.schedule()
    token_ids = engine.run_model(seqs, is_prefill)
    assert engine.abort_request(a.seq_id) is True
    outputs = engine.finalize_step(seqs, is_prefill, token_ids)
    assert outputs == []
    assert engine.is_finished()
    print("test_finalize_all_aborted_empty_batch OK")


if __name__ == "__main__":
    test_step_composition_cli_path()
    test_finalize_skips_seq_aborted_during_gpu_phase()
    test_finalize_chunked_prefill_yields_no_token()
    test_finalize_all_aborted_empty_batch()
    print("ALL STEP-SPLIT TESTS PASSED")
