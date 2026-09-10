"""DP（数据并行）多副本客户端：每副本一个独立进程，父进程是无 CUDA 的路由器。

架构（与 vLLM DP attention 的差别见 README「DP（多副本）推理」一节）：
- 副本进程的物理卡组互不相交（dp_size × tensor_parallel_size == 总卡数），每个
  副本持有完整的 LLMEngine：独立调度器、独立 KV 池、独立 NCCL 进程组
  （master_port / shm_name 按副本号错开）。副本内 MoE EP 通信沿用补零 all_reduce
  方案；副本之间没有任何集合通信。
- 父进程（本类）负责：请求分发（round-robin，副本各自排队吸收瞬时不均衡）、
  驱动各副本步进——driver 线程同一轮先给所有有在途请求的副本发 step 再收结果，
  各副本的 GPU 步在各自卡组上并行执行，不做跨副本串行等待——以及按请求
  分发事件（on_event 回调 / stream 迭代器）与 step 挂死看门狗。

线程模型：父进程内 driver 线程（派发 + 步进）、每副本一个 reader 线程（收结果），
公共状态全在 self._lock 下变更、self._cond 上会合；请求方的 add/abort/等待都
线程安全，可从任意工作线程调用。
"""
import atexit
import threading
import time
import queue
from collections import deque
from dataclasses import dataclass, field
from time import monotonic

import torch.multiprocessing as mp
from tqdm.auto import tqdm

from zvllm.sampling_params import SamplingParams
from zvllm.utils.model_download import resolve_model_path


def _parent_alive(pid: int) -> bool:
    """父进程是否存活（Linux 走 /proc；非 Linux 平台保守返回存活，不触发自退）。

    用于副本进程的孤儿守护：父进程被 SIGKILL 等强杀时不会走到 close()，
    副本需自行退出释放 GPU，而不是永远阻塞在 cmd_q.get() 上。"""
    import os
    if os.name != "posix":
        return True
    try:
        os.stat(f"/proc/{pid}")
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, use_fast=True)


def _replica_main(cmd_q, out_q, model, gpus, master_port, shm_name, engine_kwargs):
    """副本进程主函数（spawn 上下文启动）。

    必须先设 CUDA_VISIBLE_DEVICES 再导入/构建引擎：子进程本地 rank 0..n-1
    经该映射落到全局卡 gpus[0..n-1]。
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    from zvllm import LLM
    kw = dict(engine_kwargs)
    kw.pop("master_port", None)
    kw.pop("tensor_parallel_size", None)
    base_shm = kw.pop("shm_name", "zvllm")
    # 每副本 NCCL 端口 / 共享内存名按副本号错开，避免跨副本互相覆盖
    engine = LLM(model, tensor_parallel_size=len(gpus),
                 master_port=master_port, shm_name=f"{base_shm}_{shm_name}", **kw)
    out_q.put(("ready", engine.config.max_model_len, engine.config.num_kvcache_blocks))
    pending = 0
    req_to_seq = {}
    parent_pid = os.getppid()
    while True:
        try:
            cmd = cmd_q.get(timeout=5.0)
        except queue.Empty:
            # 超时窗口用于孤儿检测：父进程已死则自行退出（强杀场景不走 close）
            if not _parent_alive(parent_pid):
                break
            continue
        op = cmd[0]
        try:
            if op == "add":
                _, req_key, prompt, sp = cmd
                seq = engine.add_request(prompt, sp)
                pending += 1
                req_to_seq[req_key] = seq.seq_id
                out_q.put(("added", req_key, seq.seq_id))
            elif op == "step":
                if pending == 0:
                    # 兜底：父进程侧的在途计数与这边同源（同一批 add/finished/abort
                    # 事件），正常不会走到；走到就说明两边计数分歧，no-op 而非崩溃
                    out_q.put(("step_out", [], 0, dict(engine.spec_stats)))
                else:
                    outputs, num_tokens = engine.step()
                    done = sum(1 for _, _, finished, _ in outputs if finished)
                    pending -= done
                    out_q.put(("step_out", outputs, num_tokens, dict(engine.spec_stats)))
            elif op == "abort_req":
                _, req_key = cmd
                seq_id = req_to_seq.pop(req_key, None)
                ok = engine.abort_request(seq_id) if seq_id is not None else False
                if ok:
                    pending -= 1
                out_q.put(("aborted", req_key, ok))
            elif op == "exit":
                break
        except Exception as e:
            out_q.put(("fatal", repr(e)))
            raise
    engine.exit()
    out_q.put(("exited",))


@dataclass
class _Request:
    req_id: int
    index: int                      # 批次下标（generate/generate_stream 用；裸 add 为 -1）
    prompt: list[int]
    sp: SamplingParams
    on_event: object = None         # 可选回调 (new_token_ids, finished, reason)
    replica: int | None = None
    seq_id: int | None = None           # 副本内 Sequence id（added 后回填）
    skip: bool = False              # 派发前被 abort：driver 派发时直接补终止事件
    token_ids: list = field(default_factory=list)
    events: list = field(default_factory=list)    # 每步事件 (toks, finished, reason)，末事件恒 finished
    finished: bool = False
    finish_reason: str | None = None
    error: bool = False             # 所属副本 fatal 时被置位
    cond: object = field(default_factory=threading.Condition)


class DPClient:
    """DP 多副本推理客户端（父进程）。

    用法：
        client = DPClient("Qwen/Qwen3-30B-A3B", dp_size=2, tensor_parallel_size=2,
                          gpus=[2, 3, 4, 5], moe_ep_size=2)
        out = client.generate(["你好"], SamplingParams(max_tokens=32))
        # 或细粒度：
        rid = client.add_request("你好", SamplingParams(max_tokens=32))
        for toks, finished, reason in client.stream(rid): ...

    对外接口与 LLMEngine 对齐（generate / generate_stream / add_request /
    abort_request / is_finished / tokenizer / spec_stats），api_server 可直接
    以本类替代 LLM 作为后端。
    """

    def __init__(self, model, dp_size: int = 2, tensor_parallel_size: int = 2,
                 gpus: list[int] | None = None, master_port: int = 2345,
                 on_event: object = None, on_fatal: object = None,
                 step_timeout: float = 120.0, _start_threads: bool = True, **engine_kwargs):
        assert dp_size >= 1, f"dp_size 需 >= 1（当前 {dp_size}）"
        assert tensor_parallel_size >= 1, f"tensor_parallel_size 需 >= 1（当前 {tensor_parallel_size}）"
        total = dp_size * tensor_parallel_size
        if gpus is None:
            gpus = list(range(total))
        assert len(gpus) == total, \
            f"gpus 数量（{len(gpus)}）必须等于 dp_size × tensor_parallel_size（{total}）"
        assert len(set(gpus)) == len(gpus), f"gpus 不能重复：{gpus}"
        self.dp_size = dp_size
        self.tensor_parallel_size = tensor_parallel_size
        self.gpus = list(gpus)
        self.engine_kwargs = dict(engine_kwargs)
        self.engine_kwargs.pop("tensor_parallel_size", None)    # 每副本 TP 由本类按卡组显式传入
        self._on_fatal = on_fatal
        self.step_timeout = step_timeout

        self.model = resolve_model_path(model, engine_kwargs.get("model_source", "auto"))
        self.tokenizer = _load_tokenizer(self.model)
        self.max_model_len = None          # ready 后由副本回报
        self.kv_blocks = [0] * dp_size     # 每副本 KV 池块数（ready 后填充）

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._req_counter = 1000           # 请求 id 起点避开 Sequence 计数器空间
        self._requests: dict[int, _Request] = {}
        self._new_reqs: deque[_Request] = deque()
        self._rr = 0
        self._in_flight = [0] * dp_size
        self._cmd_q = [None] * dp_size
        self._out_q = [None] * dp_size
        self._seq_map = [dict() for _ in range(dp_size)]
        self._spec_stats = [dict() for _ in range(dp_size)]
        self._step_sent = [None] * dp_size
        self._last_reply = [None] * dp_size
        self._step_outstanding = [False] * dp_size    # 该副本有未回话的 step
        self._procs = [None] * dp_size
        self._fatal: str | None = None
        self._closed = False
        self._shutdown = threading.Event()

        for r in range(dp_size):
            cmd_q, out_q = self._make_queues()
            self._cmd_q[r], self._out_q[r] = cmd_q, out_q
            self._procs[r] = self._spawn_replica(
                r, cmd_q, out_q,
                self.gpus[r * tensor_parallel_size:(r + 1) * tensor_parallel_size],
                master_port + r, f"dp{r}")
            self._procs[r].start()
        # 阻塞等全部副本 ready（模型加载要分钟级，只校验子进程存活）
        for r in range(dp_size):
            while True:
                try:
                    msg = self._out_q[r].get(timeout=1.0)
                except Exception:
                    if not self._procs[r].is_alive():
                        raise RuntimeError(f"DP 副本 {r} 启动失败（进程已退出，看上方日志）")
                    continue
                if msg[0] == "ready":
                    with self._lock:
                        if self.max_model_len is None:
                            self.max_model_len = msg[1]
                        else:
                            assert self.max_model_len == msg[1], \
                                f"副本 max_model_len 不一致：{self.max_model_len} vs {msg[1]}"
                        self.kv_blocks[r] = msg[2]
                        self._cond.notify_all()
                    break
                if msg[0] == "fatal":
                    self._procs[r].join(timeout=10)
                    raise RuntimeError(f"DP 副本 {r} 启动失败：{msg[1]}")
                if msg[0] == "exited":
                    self._procs[r].join(timeout=10)
                    raise RuntimeError(f"DP 副本 {r} 启动失败：进程未 ready 即退出（看上方日志）")

        self._driver = threading.Thread(target=self._driver_loop, name="zvllm-dp-driver", daemon=True)
        self._readers = [threading.Thread(target=self._reader_loop, args=(r,),
                                          name=f"zvllm-dp-reader-{r}", daemon=True) for r in range(dp_size)]
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="zvllm-dp-watchdog", daemon=True)
        if _start_threads:
            self._driver.start()
            for t in self._readers:
                t.start()
            self._watchdog.start()
            # 这里用的是 atexit.register(self.close)（绑定方法强引用 self），但 DPClient
            # 本来就被自己的 driver/reader/watchdog 线程钉住——线程目标 self.*_loop 是绑定
            # 方法，线程对象又挂在 threading._active 上（2026-09-10 用 gc.get_referrers 实测
            # 确认），所以换成 weakref.finalize 并不会让"丢弃即回收"成立，退出时兜底语义也
            # 完全一致。结论：DPClient 必须显式 close()，别指望 GC 替它停副本。
            atexit.register(self.close)

    def _make_queues(self):
        """创建一对命令/结果队列（生产为 spawn 进程的 mp.Queue）。

        CPU 单测注入进程内 queue.Queue，避免在线程里跨线程使用原生管道
        （Windows 上 mp.Queue 的管道读会在纯线程中触发访问违例）。
        """
        ctx = mp.get_context("spawn")
        return ctx.Queue(), ctx.Queue()

    def _spawn_replica(self, r: int, cmd_q, out_q, gpus: list[int], master_port: int, shm_name: str):
        """创建副本进程（默认 spawn 真实引擎；CPU 单测注入线程版假副本）。

        副本必须非 daemon：副本内 LLMEngine 在 TP>1 时会再 spawn rank 子进程，
        daemon 进程不允许有子进程（multiprocessing AssertionError）。
        父进程被强杀时的孤儿副本由 _replica_main 的父进程存活守护兜底自退。
        """
        ctx = mp.get_context("spawn")
        return ctx.Process(
            target=_replica_main,
            args=(cmd_q, out_q, self.model, gpus, master_port, shm_name, self.engine_kwargs),
            name=f"zvllm-dp-replica-{r}", daemon=False)

    # ---------------------------------------------------------------- 对外接口

    def _encode_prompt(self, prompt) -> list[int]:
        """把 prompt 归一化成 token id 列表：str 走 tokenizer；单个 int 视为
        单 token prompt；列表/元组等可迭代对象原样展开。"""
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt)
        if isinstance(prompt, int):
            return [prompt]
        return list(prompt)

    def add_request(self, prompt, sampling_params: SamplingParams, on_event: object = None) -> int:
        """入队请求（str 或 token id 列表），返回父进程请求 id（req_id）。

        on_event 可选：该请求每步事件 (new_token_ids, finished, reason) 的回调，
        从 reader 线程调用（api_server 用它跨线程投递到 asyncio 队列；回调须在
        调用 add_request 前准备好——事件严格晚于本方法返回，无竞态）。"""
        return self._submit(self._encode_prompt(prompt), sampling_params, on_event).req_id

    def _submit(self, prompt: list[int], sp: SamplingParams, on_event: object, index: int = -1) -> _Request:
        assert len(prompt) > 0, "prompt 不能为空"
        with self._cond:
            if self._fatal is not None:
                raise RuntimeError(f"DP 客户端已致命（{self._fatal}），拒绝新请求")
            if self.max_model_len is not None:
                assert len(prompt) <= self.max_model_len, \
                    f"prompt 长度 {len(prompt)} 超过 max_model_len（{self.max_model_len}）"
            req = _Request(self._req_counter, index, prompt, sp, on_event)
            self._req_counter += 1
            self._requests[req.req_id] = req
            self._new_reqs.append(req)
            self._cond.notify_all()
            return req

    def stream(self, req_id: int):
        """请求级 token 流：逐条 yield (new_token_ids, finished, reason)（线程安全）。"""
        with self._lock:
            req = self._requests[req_id]
        i = 0
        while True:
            with req.cond:
                while i >= len(req.events):
                    req.cond.wait(1.0)
            ev = req.events[i]
            i += 1
            yield ev
            if ev[1]:
                break

    def generate(self, prompts, sampling_params, use_tqdm: bool = False):
        """生成一批 prompt 的补全，按输入顺序返回（与 LLMEngine.generate 同构）。"""
        sps = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        assert len(prompts) == len(sps), "prompts 与 sampling_params 数量不一致"
        reqs = [self._submit(self._encode_prompt(p), sps[i], None, index=i)
                for i, p in enumerate(prompts)]
        pbar = tqdm(total=len(reqs), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        try:
            for req in reqs:
                self._wait_done(req)
                pbar.update(1)
        finally:
            pbar.close()
        return [{"text": self.tokenizer.decode(req.token_ids),
                 "token_ids": list(req.token_ids),
                 "finish_reason": req.finish_reason} for req in reqs]

    def generate_stream(self, prompts, sampling_params):
        """流式生成：生成器，事件与 LLMEngine.generate_stream 同构（含 index 字段，
        跨副本事件交错到达，按到达顺序 yield）。"""
        sps = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        assert len(prompts) == len(sps), "prompts 与 sampling_params 数量不一致"
        reqs = [self._submit(self._encode_prompt(p), sps[i], None, index=i)
                for i, p in enumerate(prompts)]
        pos = [0] * len(reqs)
        all_done = [False] * len(reqs)
        done = 0
        while done < len(reqs):
            with self._lock:
                # 本轮新事件快照与完成计数同锁段判定：防止"finished 已置位但
                # 末事件尚未取走"的窗口导致提前终止、丢末事件（竞态窗口很小，
                # 旧实现偶发复现）
                new_events = [req.events[pos[i]:] for i, req in enumerate(reqs)]
                progressed = any(new_events)
                for i, req in enumerate(reqs):
                    pos[i] = len(req.events)
                    if not all_done[i] and req.finished and pos[i] >= len(req.events):
                        all_done[i] = True
                        done += 1
            for i, req in enumerate(reqs):
                for ev in new_events[i]:
                    toks, finished, reason = ev
                    yield {"index": i,
                           "delta": self.tokenizer.decode(toks),
                           "text": self.tokenizer.decode(req.token_ids),
                           "token_ids": list(req.token_ids),
                           "finished": finished,
                           "finish_reason": reason}
            if not progressed and done < len(reqs):
                with self._cond:
                    self._cond.wait(0.05)

    def abort_request(self, req_id: int) -> bool:
        """取消请求（幂等）：已派发则向所属副本发 abort；尚未派发则标记跳过
        （driver 派发时直接补终止事件，不占副本 KV）。返回是否受理。"""
        with self._lock:
            req = self._requests.get(req_id)
            if req is None or req.finished or self._fatal is not None:
                return False
            if req.replica is None:
                req.skip = True
            else:
                self._cmd_q[req.replica].put(("abort_req", req.req_id))
            return True

    def set_on_fatal(self, cb):
        """替换致命回调（服务层可在构造后挂接 fail-all 逻辑）。"""
        with self._lock:
            self._on_fatal = cb

    def is_finished(self) -> bool:
        with self._lock:
            return not self._new_reqs and not any(self._in_flight)

    @property
    def spec_stats(self) -> dict:
        """各副本投机解码统计求和：{proposed_tokens, accepted_tokens, steps}。"""
        with self._lock:
            agg = {}
            for s in self._spec_stats:
                for k, v in s.items():
                    agg[k] = agg.get(k, 0) + v
            return agg

    def _wait_done(self, req: _Request):
        with req.cond:
            while not req.finished:
                req.cond.wait(1.0)
        if req.error:
            raise RuntimeError(f"请求 {req.req_id} 失败：所属副本致命错误（{self._fatal}）")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()
        for q in self._cmd_q:
            if q is not None:
                try:
                    q.put(("exit",))
                except Exception:
                    pass
        for p in self._procs:
            if p is not None:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
        for t in (self._driver, *self._readers, self._watchdog):
            try:
                t.join(timeout=5)
            except RuntimeError:
                pass    # 线程从未启动（_start_threads=False 时部分线程不启动）

    # ---------------------------------------------------------------- 内部线程

    def _driver_loop(self):
        """派发新请求 + 驱动各活跃副本 step（同轮先全部发出再收，副本间 GPU 并行）。"""
        while not self._shutdown.is_set():
            if self._fatal is not None:
                # 已有副本致命：停止派发/步进，等 close（api_server 会 fail-all 后重启）
                time.sleep(0.02)
                continue
            skipped = []
            with self._lock:
                while self._new_reqs:
                    req = self._new_reqs.popleft()
                    if req.skip:
                        # 派发前被 abort：不派发，直接补终止事件
                        req.finished = True
                        req.finish_reason = "abort"
                        req.events.append(([], True, "abort"))
                        with req.cond:
                            req.cond.notify_all()
                        skipped.append(req)
                        continue
                    r = self._rr % self.dp_size
                    self._rr += 1
                    req.replica = r
                    self._in_flight[r] += 1
                    self._cmd_q[r].put(("add", req.req_id, req.prompt, req.sp))
                stepped = False
                for r in range(self.dp_size):
                    if self._in_flight[r] > 0 and not self._step_outstanding[r]:
                        # 每副本至多一个在途 step：上一 step 未回话不补发
                        # （补发会不断刷新 _step_sent，把看门狗饿死）
                        self._step_sent[r] = monotonic()
                        self._step_outstanding[r] = True
                        self._cmd_q[r].put(("step",))
                        stepped = True
                if not stepped:
                    self._cond.wait(timeout=0.02)
                    continue
            for req in skipped:
                if req.on_event is not None:
                    try:
                        req.on_event([], True, "abort")
                    except Exception as e:
                        print(f"[Z-vllm DP] on_event 回调异常（req {req.req_id}）: {e!r}", flush=True)
            # 等任一副本回话（reader 线程每收一条就 notify）；不空转
            with self._cond:
                self._cond.wait(timeout=0.1)

    def _reader_loop(self, r: int):
        while True:
            if self._shutdown.is_set():
                return
            try:
                msg = self._out_q[r].get(timeout=0.2)
            except Exception:
                continue
            if msg is None:    # mp.Queue 哨兵：全部写入方已退出
                return
            if msg[0] == "fatal":
                self._on_replica_fatal(r, msg[1])
                return
            events = []
            with self._lock:
                # 任何回话都证明副本存活且在消费命令队列：清在途 step 标记
                self._last_reply[r] = monotonic()
                self._step_outstanding[r] = False
                if msg[0] == "added":
                    _, req_id, seq_id = msg
                    self._seq_map[r][seq_id] = req_id
                    self._requests[req_id].seq_id = seq_id
                elif msg[0] == "step_out":
                    _, outputs, _num_tokens, spec_stats = msg
                    self._spec_stats[r] = spec_stats
                    for seq_id, toks, finished, reason in outputs:
                        # 非终态步骤只查表不摘除：一个请求的后续 token 还靠它定位
                        req_id = self._seq_map[r].get(seq_id)
                        if req_id is None:
                            continue
                        req = self._requests[req_id]
                        req.token_ids.extend(toks)
                        req.events.append((toks, finished, reason))
                        if finished:
                            self._seq_map[r].pop(seq_id, None)
                            req.finished = True
                            req.finish_reason = reason
                            self._in_flight[r] -= 1
                        events.append(req)
                elif msg[0] == "aborted":
                    _, req_id, ok = msg
                    if ok:
                        req = self._requests[req_id]
                        self._seq_map[r].pop(req.seq_id, None)
                        req.finished = True
                        req.finish_reason = "abort"
                        req.events.append(([], True, "abort"))
                        self._in_flight[r] -= 1
                        events.append(req)
                elif msg[0] == "ready":
                    pass    # 构造期已消费；理论上不可达
                self._cond.notify_all()
                for req in events:
                    with req.cond:
                        req.cond.notify_all()
            # on_event 回调在锁外调用，避免回调重入 client 方法时死锁
            for req in events:
                if req.on_event is not None:
                    try:
                        req.on_event(req.events[-1][0], req.events[-1][1], req.events[-1][2])
                    except Exception as e:
                        print(f"[Z-vllm DP] on_event 回调异常（req {req.req_id}）: {e!r}", flush=True)

    def _on_replica_fatal(self, r: int, msg: str):
        victims = []
        with self._lock:
            self._fatal = f"副本 {r}: {msg}"
            for req in self._requests.values():
                if not req.finished and req.replica == r:
                    req.finished = True
                    req.finish_reason = "abort"
                    req.error = True
                    req.events.append(([], True, "abort"))
                    victims.append(req)
            self._in_flight[r] = 0
            self._cond.notify_all()
            for req in victims:
                with req.cond:
                    req.cond.notify_all()
        print(f"[Z-vllm DP] FATAL 副本 {r}: {msg}（其全部在途请求已快速失败）", flush=True)
        if self._on_fatal is not None:
            try:
                self._on_fatal(f"副本 {r}: {msg}")
            except Exception as e:
                print(f"[Z-vllm DP] on_fatal 回调异常: {e!r}", flush=True)

    def _watchdog_loop(self):
        """看门狗：某副本的在途 step 超过 step_timeout 无任何回话即判挂死
        （NCCL 集合通信死锁 / GPU kernel 挂死无法进程内恢复）：失败其全部
        在途请求，交 on_fatal 决策（api_server 会 fail-all 并退出进程重启）。
        driver 对每副本至多维持一个未回话 step、超时无回话不补发，
        保证超时时 _step_sent 时间戳不被刷新、看门狗必然触发。"""
        while not self._shutdown.is_set():
            self._shutdown.wait(1.0)
            if self._shutdown.is_set() or self._fatal is not None:
                continue
            now = monotonic()
            for r in range(self.dp_size):
                if not self._step_outstanding[r]:
                    continue
                sent = self._step_sent[r]
                if sent is not None and now - sent > self.step_timeout:
                    self._on_replica_fatal(r, f"step 挂死（>{self.step_timeout:.0f}s 无回话）")
                    break
