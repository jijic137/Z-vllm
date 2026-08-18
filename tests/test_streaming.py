"""LLMEngine 流式/非流式聚合逻辑单测（无需 GPU）。运行：python tests/test_streaming.py

用脚本化的假引擎（实现 generate/generate_stream 依赖的最小接口）验证：
流式事件的顺序、增量文本累积、结束标志，以及非流式结果按输入顺序聚合。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zvllm.engine.llm_engine import LLMEngine


class FakeTokenizer:
    def decode(self, token_ids):
        return "".join(f"<{i}>" for i in token_ids)


class FakeSeq:
    def __init__(self, seq_id):
        self.seq_id = seq_id


class FakeEngine:
    """实现 LLMEngine.generate / generate_stream 依赖的最小接口。

    script：每个 step 依次返回的 outputs，元素为 (seq_id, new_token_ids, finished)。
    """

    def __init__(self, script):
        self.script = script
        self.step_count = 0

    def _add_requests(self, prompts, sampling_params):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        return [FakeSeq(i) for i in range(len(prompts))]

    def step(self):
        outputs = self.script[self.step_count]
        self.step_count += 1
        return outputs, 1

    def is_finished(self):
        return self.step_count >= len(self.script)


def make_engine(script):
    engine = FakeEngine(script)
    engine.tokenizer = FakeTokenizer()
    return engine


def test_generate_stream_events():
    # 两条请求各出 2 token，第二步同时结束；事件按 (步内批序) 产出
    engine = make_engine([
        [(0, [10], False), (1, [20], False)],
        [(0, [11], True), (1, [21], True)],
    ])
    events = list(LLMEngine.generate_stream(engine, ["p0", "p1"], "sp"))
    assert len(events) == 4
    assert events[0] == {"index": 0, "delta": "<10>", "text": "<10>",
                         "token_ids": [10], "finished": False}
    assert events[1] == {"index": 1, "delta": "<20>", "text": "<20>",
                         "token_ids": [20], "finished": False}
    assert events[2] == {"index": 0, "delta": "<11>", "text": "<10><11>",
                         "token_ids": [10, 11], "finished": True}
    assert events[3] == {"index": 1, "delta": "<21>", "text": "<20><21>",
                         "token_ids": [20, 21], "finished": True}
    # 每条请求的最后一个事件必须是 finished=True
    last_finished = {}
    for e in events:
        last_finished[e["index"]] = e["finished"]
    assert last_finished == {0: True, 1: True}
    print("test_generate_stream_events OK")


def test_stream_finish_order_independent():
    # 请求 1 先结束、请求 0 继续生成：结束计数与文本累积不受结束顺序影响
    engine = make_engine([
        [(0, [10], False), (1, [20], True)],
        [(0, [11], True)],
    ])
    events = list(LLMEngine.generate_stream(engine, ["p0", "p1"], "sp"))
    assert [e["index"] for e in events] == [0, 1, 0]
    assert events[1]["finished"] is True and events[1]["text"] == "<20>"
    assert events[2]["text"] == "<10><11>" and events[2]["finished"] is True
    print("test_stream_finish_order_independent OK")


def test_generate_blocking_aggregation():
    # 非流式：跨步累积 token，按输入顺序返回完整结果
    engine = make_engine([
        [(0, [10], False), (1, [20], False)],
        [(0, [11], True), (1, [21], True)],
    ])
    results = LLMEngine.generate(engine, ["p0", "p1"], "sp", use_tqdm=False)
    assert results == [
        {"text": "<10><11>", "token_ids": [10, 11]},
        {"text": "<20><21>", "token_ids": [20, 21]},
    ]
    print("test_generate_blocking_aggregation OK")


def test_generate_stream_dispatch():
    # generate(stream=True) 返回生成器且不阻塞执行
    import types
    engine = make_engine([[(0, [10], True)]])
    engine.generate_stream = types.MethodType(LLMEngine.generate_stream, engine)
    gen = LLMEngine.generate(engine, ["p0"], "sp", stream=True)
    assert isinstance(gen, types.GeneratorType)
    assert engine.step_count == 0, "生成器应先返回、消费时才执行"
    assert next(gen)["finished"] is True
    print("test_generate_stream_dispatch OK")


if __name__ == "__main__":
    test_generate_stream_events()
    test_stream_finish_order_independent()
    test_generate_blocking_aggregation()
    test_generate_stream_dispatch()
    print("ALL STREAMING TESTS PASSED")
