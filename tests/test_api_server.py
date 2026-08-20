"""OpenAI 服务 HTTP 层单测（无需 GPU）。运行：python tests/test_api_server.py

把脚本化的假引擎注入 InferenceServer，通过 FastAPI TestClient 验证：
/health、/v1/models、非流式 /v1/chat/completions、SSE 流式
（首个 role chunk / 内容 chunk / 最后 finish_reason chunk / [DONE]）、
/v1/completions（多 prompt 非流式、流式仅单 prompt）、非法参数 422/400。
"""
import json
import asyncio
import sys
import threading
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from zvllm.sampling_params import SamplingParams
from zvllm.entrypoints.openai.api_server import InferenceServer, create_app


class FakeTokenizer:
    def decode(self, token_ids):
        return "".join(f"<{i}>" for i in token_ids)

    def encode(self, text):
        return [ord(c) % 256 for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        assert tokenize is True
        return [5, 6, 7]


class FakeSeq:
    def __init__(self, seq_id):
        self.seq_id = seq_id


class FakeEngine:
    """实现 InferenceServer 依赖的 LLMEngine 最小接口（忠实模拟真实引擎语义）。

    specs：按 add_request 顺序的 (finish_reason, tokens) 列表。
    与真实引擎一致：step 只为已 add 的序列产出输出（调度器不认识未 add 的
    序列），is_finished 取决于是否还有未结束序列。每个 step 为每条已 add
    且未结束的序列产出一个新 token，token 列表耗尽即结束。状态加锁，模拟
    真实并发（后台 step 线程 + 事件循环）。

    三阶段接口与真实引擎对应：schedule（阶段 A，列队列）/ run_model
    （阶段 B，只"采样"不推进状态）/ finalize_step（阶段 C，推进状态并产出
    输出）；阶段 B 期间被取消（不在 _open 中）的序列在阶段 C 被跳过。
    """

    def __init__(self, specs):
        self.config = SimpleNamespace(model="/fake/Qwen3-0.6B")
        self.tokenizer = FakeTokenizer()
        self.specs = specs
        self._lock = threading.Lock()
        self._next_seq_id = 1
        self._open: dict[int, dict] = {}
        self.aborted: list[int] = []
        self.last_sp = None

    @property
    def scheduler(self):
        return SimpleNamespace(is_finished=self._is_finished)

    def _is_finished(self):
        with self._lock:
            return not self._open

    def add_request(self, prompt, sampling_params):
        with self._lock:
            reason, tokens = self.specs[self._next_seq_id - 1]
            self._open[self._next_seq_id] = {
                "pos": 0, "tokens": list(tokens), "reason": reason,
            }
            self.last_sp = sampling_params
            seq = FakeSeq(self._next_seq_id)
            self._next_seq_id += 1
            return seq

    def abort_request(self, request_id):
        with self._lock:
            if request_id in self._open:
                del self._open[request_id]
                self.aborted.append(request_id)
                return True
            return False

    def schedule(self):
        with self._lock:
            return [FakeSeq(i) for i in sorted(self._open)], False

    def run_model(self, seqs, is_prefill):
        with self._lock:
            return [self._open[s.seq_id]["tokens"][self._open[s.seq_id]["pos"]]
                    for s in seqs]

    def finalize_step(self, seqs, is_prefill, token_ids):
        with self._lock:
            outputs = []
            for s, tok in zip(seqs, token_ids):
                st = self._open.get(s.seq_id)
                if st is None:
                    continue    # 阶段 B 期间被取消：状态已定型，跳过
                st["pos"] += 1
                finished = st["pos"] >= len(st["tokens"])
                if finished:
                    del self._open[s.seq_id]
                outputs.append((s.seq_id, [tok], finished,
                                st["reason"] if finished else None))
            return outputs


def parse_sse(lines):
    payloads = []
    for line in lines:
        data = line[len("data:"):].strip()
        payloads.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return payloads


def test_health_and_models():
    engine = FakeEngine([])
    server = InferenceServer(engine)
    with TestClient(create_app(server)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        models = client.get("/v1/models").json()
        assert models["object"] == "list"
        assert models["data"][0]["id"] == "Qwen3-0.6B"
        assert models["data"][0]["object"] == "model"
    print("test_health_and_models OK")


def test_chat_completions_non_stream():
    engine = FakeEngine([("stop", [10, 11])])
    server = InferenceServer(engine)
    with TestClient(create_app(server)) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "Qwen3-0.6B",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2, "temperature": 0.7,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "Qwen3-0.6B"
        assert body["id"].startswith("chatcmpl-")
        choice = body["choices"][0]
        assert choice["message"] == {"role": "assistant", "content": "<10><11>"}
        assert choice["finish_reason"] == "stop"
        assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    print("test_chat_completions_non_stream OK")


def test_chat_completions_stream():
    engine = FakeEngine([("stop", [10, 11])])
    server = InferenceServer(engine)
    with TestClient(create_app(server)) as client:
        with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "stream": True,
        }) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = [line for line in r.iter_lines() if line.startswith("data:")]
    payloads = parse_sse(lines)
    assert payloads[-1] == "[DONE]"
    chunks = payloads[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # 首个 chunk 携带 role assistant（内容为空）
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[0]["choices"][0]["finish_reason"] is None
    # 内容 chunk
    assert chunks[1]["choices"][0]["delta"] == {"content": "<10>"}
    # 最后 chunk：delta 为空 + finish_reason
    assert chunks[2]["choices"][0]["delta"] == {}
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    print("test_chat_completions_stream OK")


def test_completions_non_stream_multi_prompt():
    engine = FakeEngine([("stop", [10, 11]), ("length", [20, 21])])
    server = InferenceServer(engine)
    with TestClient(create_app(server)) as client:
        r = client.post("/v1/completions", json={"prompt": ["ab", "cd"], "max_tokens": 2})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion"
        assert body["id"].startswith("cmpl-")
        assert [c["text"] for c in body["choices"]] == ["<10><11>", "<20><21>"]
        assert [c["finish_reason"] for c in body["choices"]] == ["stop", "length"]
        assert [c["index"] for c in body["choices"]] == [0, 1]
        assert body["usage"]["prompt_tokens"] == 4      # "ab" -> 2 token, "cd" -> 2 token
        assert body["usage"]["completion_tokens"] == 4
    print("test_completions_non_stream_multi_prompt OK")


def test_completions_stream_multi_prompt_rejected():
    engine = FakeEngine([])
    server = InferenceServer(engine)
    with TestClient(create_app(server)) as client:
        r = client.post("/v1/completions", json={"prompt": ["a", "b"], "stream": True})
        assert r.status_code == 400
    print("test_completions_stream_multi_prompt_rejected OK")


def test_invalid_sampling_params_rejected():
    engine = FakeEngine([])
    server = InferenceServer(engine)
    with TestClient(create_app(server)) as client:
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "temperature": -1,
        })
        assert r.status_code == 422
    print("test_invalid_sampling_params_rejected OK")


def test_stop_param_passthrough():
    # stop 字段应透传进 SamplingParams（str 归一化为 list），chat/completions 两端点
    engine = FakeEngine([("stop", [10]), ("stop", [20])])
    server = InferenceServer(engine)
    with TestClient(create_app(server)) as client:
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "stop": "<10>",
        })
        assert r.status_code == 200, r.text
        assert engine.last_sp.stop == ["<10>"]
        r = client.post("/v1/completions", json={"prompt": "ab", "max_tokens": 1,
                                                 "stop": ["x", "y"]})
        assert r.status_code == 200, r.text
        assert engine.last_sp.stop == ["x", "y"]
    print("test_stop_param_passthrough OK")


def test_astream_aborts_on_abandon():
    # 客户端放弃流（astream 被 aclose，如断连）时应触发 abort_request 并注销队列。
    # 后台 step 线程会紧循环重入服务锁，可能把测试线程的锁等待饿死到序列跑完
    #（非公平锁 + 忙循环，属测试环境假象；真实推理每步是 GPU 临界区，且断连
    # 时序列尚有大量 token 在途），故这里先停掉 step 线程，再手动登记在途序列
    # 并预置事件队列，使断言确定。step loop 与真实并发的交互由上方 HTTP 测试
    # 与真机 e2e 覆盖。
    engine = FakeEngine([])
    server = InferenceServer(engine)
    server.shutdown()
    server.step_thread.join(timeout=5)
    assert not server.step_thread.is_alive(), "step 线程应已退出"

    async def scenario():
        seq_id = 1
        engine._open[seq_id] = {"pos": 0, "tokens": [10, 11, 12], "reason": "stop"}
        q = asyncio.Queue()
        server.queues[seq_id] = q
        q.put_nowait(([10], False, None))   # 预置第一个 token 事件
        gen = server.astream(seq_id)
        item = await gen.__anext__()
        assert item == ([10], False, None)
        await gen.aclose()   # 模拟客户端断连
        assert engine.aborted == [seq_id], "放弃流应触发 abort_request"
        assert seq_id not in server.queues
        assert seq_id not in engine._open, "被取消的序列应移出引擎"

    asyncio.run(scenario())
    print("test_astream_aborts_on_abandon OK")


if __name__ == "__main__":
    test_health_and_models()
    test_chat_completions_non_stream()
    test_chat_completions_stream()
    test_completions_non_stream_multi_prompt()
    test_completions_stream_multi_prompt_rejected()
    test_invalid_sampling_params_rejected()
    test_stop_param_passthrough()
    test_astream_aborts_on_abandon()
    print("ALL API SERVER TESTS PASSED")
