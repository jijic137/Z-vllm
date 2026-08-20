"""DPClient CPU 单测：假副本（线程版，与真实 _replica_main 同消息协议）驱动
父进程的派发 / 事件 / abort / 看门狗逻辑，不依赖 GPU。"""
import queue
import threading
import time

import pytest

from zvllm.engine import dp_engine
from zvllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, s):
        return list(range(len(s))) if isinstance(s, str) else list(s)

    def decode(self, ids):
        return "".join(chr(65 + t % 26) for t in ids)


class _ThreadRunner:
    """Process 形态替身：把假副本跑在进程内线程里。"""

    def __init__(self, target, args):
        self._t = threading.Thread(target=target, args=args, daemon=True)

    def start(self):
        self._t.start()

    def is_alive(self):
        return self._t.is_alive()

    def join(self, timeout=None):
        self._t.join(timeout)

    def terminate(self):
        pass


def _fake_replica_main(cmd_q, out_q, model, gpus, master_port, shm_name, engine_kwargs):
    """假副本：与真实 _replica_main 同协议。

    每个 step 对每条在途请求产 1 个 token（token 号 = 累计产出数），
    产出 sp.max_tokens 个后 finished（reason=stop）；spec_stats 恒为固定值。"""
    out_q.put(("ready", 1024, 128))
    pending = 0
    reqs = {}
    while True:
        cmd = cmd_q.get()
        op = cmd[0]
        if op == "add":
            _, req_key, prompt, sp = cmd
            reqs[req_key] = [0, sp]
            pending += 1
            out_q.put(("added", req_key, req_key))
        elif op == "step":
            outputs = []
            for req_key, state in list(reqs.items()):
                state[0] += 1
                emitted = state[0]
                finished = emitted >= state[1].max_tokens
                outputs.append((req_key, [emitted], finished, "stop"))
                if finished:
                    del reqs[req_key]
                    pending -= 1
            out_q.put(("step_out", outputs, len(outputs),
                       {"proposed_tokens": 2, "accepted_tokens": 1, "steps": 1}))
        elif op == "abort_req":
            _, req_key = cmd
            ok = req_key in reqs
            if ok:
                del reqs[req_key]
                pending -= 1
            out_q.put(("aborted", req_key, ok))
        elif op == "exit":
            break
    out_q.put(("exited",))


def _hung_replica_main(cmd_q, out_q, model, gpus, master_port, shm_name, engine_kwargs):
    """挂死副本：added 正常回话，step 一律吞掉不回话（模拟 NCCL 死锁）。"""
    out_q.put(("ready", 1024, 128))
    stop = threading.Event()
    while True:
        cmd = cmd_q.get()
        if cmd[0] == "add":
            out_q.put(("added", cmd[1], cmd[1]))
        elif cmd[0] == "step":
            stop.wait(0.05)    # 吞掉 step，小步等待 exit
        elif cmd[0] == "exit":
            stop.set()
            break
    out_q.put(("exited",))


@pytest.fixture
def dp_factory(monkeypatch):
    monkeypatch.setattr(dp_engine, "resolve_model_path", lambda m, s="auto": "/fake")
    monkeypatch.setattr(dp_engine, "_load_tokenizer", lambda p: FakeTokenizer())

    def _spawn(self, r, cmd_q, out_q, gpus, master_port, shm_name):
        return _ThreadRunner(_fake_replica_main,
                             (cmd_q, out_q, self.model, gpus, master_port, shm_name, self.engine_kwargs))

    monkeypatch.setattr(dp_engine.DPClient, "_spawn_replica", _spawn)

    def _make_queues(self):
        # 线程版假副本跑在进程内：用 queue.Queue 替代原生管道 mp.Queue，
        # 避免 Windows 上跨线程管道读触发访问违例（父进程逻辑不受影响）。
        return queue.Queue(), queue.Queue()

    monkeypatch.setattr(dp_engine.DPClient, "_make_queues", _make_queues)

    clients = []

    def factory(**kw):
        c = dp_engine.DPClient("/fake", dp_size=2, tensor_parallel_size=2,
                               gpus=[0, 1, 2, 3], **kw)
        clients.append(c)
        return c

    yield factory
    for c in clients:
        c.close()


def _wait_until(pred, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_ready_metadata(dp_factory):
    c = dp_factory()
    assert c.max_model_len == 1024
    assert c.kv_blocks == [128, 128]


def test_generate_round_robin_and_order(dp_factory):
    c = dp_factory()
    sp = SamplingParams(temperature=0, max_tokens=3)
    out = c.generate([1, 2, 3, 4], sp, use_tqdm=False)
    assert len(out) == 4
    for o in out:
        assert o["token_ids"] == [1, 2, 3]      # 假模型每步产 1 个 token（按累计计数编号）
        assert o["finish_reason"] == "stop"
    reqs = list(c._requests.values())
    assert [r.replica for r in reqs] == [0, 1, 0, 1]    # round-robin
    assert c.is_finished()


def test_generate_stream_events(dp_factory):
    c = dp_factory()
    sp = SamplingParams(temperature=0, max_tokens=2)
    events = list(c.generate_stream([5, 6], sp))
    by_index = {}
    for ev in events:
        by_index.setdefault(ev["index"], []).append(ev)
    assert set(by_index) == {0, 1}
    for i, evs in by_index.items():
        assert len(evs) == 2
        # token_ids 为累计值、delta 为本步新 token（与 LLMEngine.generate_stream 同构）
        assert [e["token_ids"] for e in evs] == [[1], [1, 2]]
        assert [e["delta"] for e in evs] == ["B", "C"]
        assert evs[-1]["finished"] is True
        assert evs[-1]["finish_reason"] == "stop"


def test_abort_dispatched(dp_factory):
    c = dp_factory()
    # r0 的 max_tokens 取极大：保证 abort 到达前不可能跑完（每步需队列往返，
    # 10 万 token 至少需要 ~1s，abort 在数十 ms 内发出）
    r0 = c.add_request([1], SamplingParams(temperature=0, max_tokens=100000))
    r1 = c.add_request([1], SamplingParams(temperature=0, max_tokens=10))

    def dispatched():
        with c._lock:
            return (c._requests[r0].replica is not None
                    and c._requests[r1].replica is not None)

    assert _wait_until(dispatched)
    assert c.abort_request(r0) is True
    evs = list(c.stream(r0))
    assert evs[-1][1] is True and evs[-1][2] == "abort"
    assert sum(len(t) for t, _, _ in evs) < 10000    # 未到 max_tokens 即被取消
    evs1 = list(c.stream(r1))
    assert evs1[-1][2] == "stop"
    assert sum(len(t) for t, _, _ in evs1) == 10  # 另一请求不受影响


def test_abort_undispatched(dp_factory):
    c = dp_factory(_start_threads=False)     # 不启动 driver：请求必然处于未派发态
    rid = c.add_request([1], SamplingParams(temperature=0, max_tokens=5))
    assert c.abort_request(rid) is True       # 未派发 → 标记跳过，不占副本
    c._driver.start()
    evs = list(c.stream(rid))
    assert evs == [([], True, "abort")]


def test_on_event_callback(dp_factory):
    c = dp_factory()
    seen = []
    done = threading.Event()

    def on_event(toks, finished, reason):
        seen.append((list(toks), finished, reason))
        if finished:
            done.set()

    rid = c.add_request([1], SamplingParams(temperature=0, max_tokens=3), on_event=on_event)
    assert done.wait(10.0)
    assert [t for t, _, _ in seen] == [[1], [2], [3]]
    assert seen[-1][1] is True and seen[-1][2] == "stop"


def test_spec_stats_aggregate(dp_factory):
    c = dp_factory()
    c.generate([1, 2, 3, 4], SamplingParams(temperature=0, max_tokens=2), use_tqdm=False)
    # 假副本每次 step_out 回固定 stats；父进程按副本保留最后一次并跨副本求和
    assert c.spec_stats == {"proposed_tokens": 4, "accepted_tokens": 2, "steps": 2}


def test_watchdog_hang_fails_requests(monkeypatch, dp_factory):
    def _spawn_hung(self, r, cmd_q, out_q, gpus, master_port, shm_name):
        if r == 0:
            return _ThreadRunner(_hung_replica_main,
                                 (cmd_q, out_q, self.model, gpus, master_port, shm_name, self.engine_kwargs))
        return _ThreadRunner(_fake_replica_main,
                             (cmd_q, out_q, self.model, gpus, master_port, shm_name, self.engine_kwargs))

    monkeypatch.setattr(dp_engine.DPClient, "_spawn_replica", _spawn_hung)
    fatal = threading.Event()
    c = dp_engine.DPClient("/fake", dp_size=2, tensor_parallel_size=2, gpus=[0, 1, 2, 3],
                           step_timeout=0.5, on_fatal=lambda msg: fatal.set())
    try:
        rid = c.add_request([1], SamplingParams(temperature=0, max_tokens=4))    # 轮询到挂死的副本 0
        assert fatal.wait(15.0)
        evs = list(c.stream(rid))
        assert evs and evs[-1][1] is True and evs[-1][2] == "abort"
        with pytest.raises(RuntimeError):
            c.add_request([2], SamplingParams(temperature=0, max_tokens=1))    # 致命后拒绝新请求
    finally:
        c.close()
