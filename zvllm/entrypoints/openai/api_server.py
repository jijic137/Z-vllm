"""OpenAI 兼容推理服务（FastAPI + SSE 流式）。

启动（GPU 机器上）：
    pip install fastapi uvicorn        # 或 pip install z-vllm[serve]
    python -m zvllm.entrypoints.openai.api_server \
        --model ~/huggingface/Qwen3-0.6B/ --port 8000

端点：
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   （stream / 非 stream）
    POST /v1/completions        （非 stream 支持多 prompt；stream 仅单 prompt）

架构：单个全局引擎 + 后台 step-loop 线程。请求只负责入队序列、在自己的
每序列事件队列（asyncio.Queue）上等待；后台线程逐步推进引擎并把新生成 token
跨线程分发给所有在途请求，多个请求的 prefill/decode 因此能按调度器的混合批
自然交错。全部请求处理在事件循环上进行（阻塞部分经 to_thread 跑在工作线程），
长连接请求不占用工作线程。

健壮性设计（对应真机 e2e 暴露的两个问题：断连不触发取消、GPU step 卡死时
服务整体 wedged）：
1. 两阶段 step：step loop 只在阶段 A（调度）与阶段 C（状态回写）持服务锁，
   GPU 计算（阶段 B）不持锁——请求入队/取消永远不会被 GPU 计算阻塞。
2. 断连检测：uvicorn 在客户端断连时既不取消响应任务、send 也不报错，
   流式响应必须与 ASGI disconnect 消息（Request.is_disconnected）竞速；
   检测到断连立即取消对应序列（finish_reason="abort"）并释放其 KV 块，
   避免"孤儿序列"继续空转到 max_tokens。非流式请求的客户端断连暂不检测
   （请求会继续生成完）。
3. 看门狗：单步 GPU 阶段超过 STEP_TIMEOUT 未返回（如 GPU kernel 挂死）时，
   挂死 kernel 无法在进程内取消（torch 无进程内 reset），看门狗先通知全部
   在途请求快速失败，再退出进程，交由上层（systemd/手动）重启恢复，
   避免服务永久 wedged。

stop 已支持（生成文本以任一停止串结尾即终止，finish_reason="stop"）；
presence_penalty / frequency_penalty 仍被接受但忽略。
"""
import argparse
import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from zvllm import LLM, SamplingParams

# 单步 GPU 阶段超时：正常步为毫秒级，16k prefill 也在数十秒内，
# 超过即判 step 挂死（GPU 级故障，只能重启恢复）。
STEP_TIMEOUT = 60.0


class InferenceServer:
    """引擎封装：后台 step-loop 线程 + 看门狗线程 + 每序列事件队列（asyncio）。"""

    def __init__(self, engine: LLM):
        self.engine = engine
        self.model_name = os.path.basename(os.path.normpath(engine.config.model))
        self.lock = threading.Lock()
        self.queues: dict[int, asyncio.Queue] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown = threading.Event()
        self._hung = threading.Event()
        self._in_gpu_phase = False
        self._gpu_since = 0.0
        self.step_thread = threading.Thread(target=self._step_loop, daemon=True)
        self.step_thread.start()
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

    def attach_loop(self, loop: asyncio.AbstractEventLoop):
        """记录事件循环（服务启动时调用）；step 线程靠它跨线程投递事件。"""
        self._loop = loop

    def _push(self, seq_id: int, item):
        """step 线程 → 事件循环：跨线程 put 必须经 call_soon_threadsafe。"""
        q = self.queues.get(seq_id)
        if q is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(q.put_nowait, item)

    def _fail_all(self):
        """引擎致命（step 异常 / step 挂死）：向全部在途请求推送 None 哨兵，
        客户端快速失败而不是永久挂起。"""
        if self._loop is None:
            return
        for q in list(self.queues.values()):
            self._loop.call_soon_threadsafe(q.put_nowait, None)

    def _step_loop(self):
        while not self._shutdown.is_set():
            idle = False
            try:
                # 阶段 A（CPU，持锁）：调度。锁持有时间是纯 CPU，
                # GPU 计算期间入队/取消不会被阻塞。
                with self.lock:
                    if self.engine.scheduler.is_finished():
                        idle = True
                    else:
                        seqs, is_prefill = self.engine.schedule()
                if not idle:
                    # 阶段 B（GPU，不持锁）：前向 + 采样
                    self._in_gpu_phase = True
                    self._gpu_since = time.monotonic()
                    try:
                        token_ids = self.engine.run_model(seqs, is_prefill)
                    finally:
                        self._in_gpu_phase = False
                    # 阶段 C（CPU，持锁）：状态回写 + 停止串，再分发事件
                    with self.lock:
                        outputs = self.engine.finalize_step(seqs, is_prefill, token_ids)
                    for seq_id, new_token_ids, finished, reason in outputs:
                        self._push(seq_id, (new_token_ids, finished, reason))
            except Exception as e:
                # 引擎异常（如显存 OOM）：通知所有在途请求，避免客户端永久挂起
                print(f"[zvllm] step loop aborted: {e!r}")
                self._fail_all()
                break
            if idle:
                time.sleep(0.01)

    def _watchdog_loop(self):
        """看门狗：GPU 阶段超过 STEP_TIMEOUT 未返回即判 step 挂死。

        挂死的 GPU kernel 无法在进程内取消（torch 无进程内 reset），恢复的
        唯一途径是重启进程：先让全部在途请求快速失败，再退出进程，
        交由上层（systemd / 手动）重启。"""
        while not self._shutdown.is_set():
            time.sleep(1.0)
            if self._in_gpu_phase and not self._hung.is_set():
                if time.monotonic() - self._gpu_since > STEP_TIMEOUT:
                    self._hung.set()
                    print(f"[zvllm] FATAL: step 挂死（GPU 阶段 {STEP_TIMEOUT:.0f}s 未返回）。"
                          f"已通知全部在途请求并退出进程，请重启服务恢复 GPU。")
                    self._fail_all()
                    time.sleep(1.0)    # 留一点时间让客户端收到终止通知
                    os._exit(1)

    async def add_prompt(self, prompt_token_ids: list[int], sampling_params: SamplingParams) -> int:
        """入队新序列并注册其事件队列，返回 seq_id。

        队列注册与序列入队必须在同一锁段内原子完成：否则 step loop 可能在
        队列注册前就产出第一个 token 而丢失。asyncio.Queue 的构造不依赖事件
        循环（3.10+ 惰性绑定），可在工作线程创建。"""
        def _add():
            with self.lock:
                seq = self.engine.add_request(prompt_token_ids, sampling_params)
                self.queues[seq.seq_id] = asyncio.Queue()
                return seq.seq_id
        return await asyncio.to_thread(_add)

    async def abort(self, seq_id: int) -> bool:
        """取消在途请求（如客户端断连）：引擎释放其 KV 块，finish_reason="abort"。

        与 step loop 共用 self.lock 串行化保证引擎状态一致；GPU 阶段不持锁，
        这里的最长等待仅为阶段 A/C 的纯 CPU 时间。"""
        def _abort():
            with self.lock:
                return self.engine.abort_request(seq_id)
        return await asyncio.to_thread(_abort)

    async def _next_event(self, seq_id: int):
        """等待该序列的下一个事件（带看门狗）。

        step 线程异常退出时预期会推送 None 哨兵；若它以绕过哨兵的方式死亡
        （如致命 BaseException）或被看门狗判为挂死，这里及时抛出而不是让
        客户端永久挂起。"""
        q = self.queues[seq_id]
        while True:
            try:
                return await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if not self.step_thread.is_alive():
                    raise RuntimeError("推理引擎 step 线程已退出，请求中止")
                if self._hung.is_set():
                    raise RuntimeError("推理引擎 step 挂死，请求中止")

    async def wait_completion(self, seq_id: int) -> tuple[list[int], str]:
        """等待至序列结束，返回 (完整补全 token 列表, finish_reason)。"""
        token_ids: list[int] = []
        reason = None
        try:
            while True:
                item = await self._next_event(seq_id)
                if item is None:
                    raise RuntimeError("推理引擎异常退出，请求中止")
                new_token_ids, finished, finish_reason = item
                token_ids.extend(new_token_ids)
                if finished:
                    reason = finish_reason
                    break
        finally:
            self.queues.pop(seq_id, None)
        return token_ids, reason

    async def astream(self, seq_id: int):
        """逐条 yield (new_token_ids, finished, finish_reason)；结束或放弃时注销队列。

        客户端放弃流（断连）时由外层 aclose() 本生成器，finally 触发 abort，
        释放该序列占用的 KV 块。"""
        finished = False
        try:
            while True:
                item = await self._next_event(seq_id)
                if item is None:
                    raise RuntimeError("推理引擎异常退出，请求中止")
                yield item
                finished = item[1]
                if item[1]:
                    break
        finally:
            self.queues.pop(seq_id, None)
            if not finished:
                if await self.abort(seq_id):
                    print(f"[zvllm] aborted seq {seq_id}（客户端放弃流）")

    def shutdown(self):
        self._shutdown.set()


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = Field(1.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(-1, ge=-1)
    max_tokens: int = Field(64, gt=0)
    seed: int | None = None
    stream: bool = False
    # 以下字段为兼容 OpenAI 客户端而接受：stop 已支持；penalty 当前忽略
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    suffix: str | None = None
    temperature: float = Field(1.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(-1, ge=-1)
    max_tokens: int = Field(64, gt=0)
    seed: int | None = None
    stream: bool = False
    echo: bool = False
    stop: str | list[str] | None = None


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _sampling_params(req) -> SamplingParams:
    return SamplingParams(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                          max_tokens=req.max_tokens, seed=req.seed, stop=req.stop)


async def _wait_disconnect(request: Request) -> None:
    """等待客户端断连（已断连则立即返回）。

    uvicorn 在客户端断连时既不取消响应任务、对已断连连接的 send 也不报错——
    唯一可靠的检测是等待 ASGI receive 通道上的 http.disconnect 消息
    （Request.is_disconnected 等的就是这条消息，会阻塞到客户端断连为止）。
    该协程应运行在独立 task 里与 token 流竞速。"""
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(0.1)


def create_app(server: InferenceServer) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server.attach_loop(asyncio.get_running_loop())
        yield
        server.shutdown()

    app = FastAPI(title="Z-vllm", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [
            {"id": server.model_name, "object": "model",
             "created": int(time.time()), "owned_by": "zvllm"},
        ]}

    async def _add_or_400(prompt_token_ids: list[int], sp: SamplingParams) -> int:
        try:
            return await server.add_prompt(prompt_token_ids, sp)
        except AssertionError as e:
            raise HTTPException(status_code=400, detail=str(e))

    def _stream_response(request: Request, seq_id: int, make_chunks):
        """把序列 token 流包装成 SSE 响应，并与客户端断连竞速。

        make_chunks(new_token_ids, finished, reason) -> 该事件对应的 SSE 字符串列表。
        断连发生后在下一个 token 处跳出（最坏检测延迟一个 token 间隔，decode
        期间为毫秒级）：astream 的 aclose 触发 abort，释放序列 KV 块；
        [DONE] 只在客户端仍在连接时发送。"""

        async def stream():
            agen = server.astream(seq_id)
            disconnect = asyncio.create_task(_wait_disconnect(request))
            try:
                async for new_token_ids, finished, reason in agen:
                    if disconnect.done():
                        break    # 客户端断连：清理交给 finally
                    for sse_str in make_chunks(new_token_ids, finished, reason):
                        yield sse_str
                if not disconnect.done():
                    yield "data: [DONE]\n\n"
            finally:
                await agen.aclose()
                disconnect.cancel()

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        try:
            prompt = server.engine.tokenizer.apply_chat_template(
                req.messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"chat template 处理失败: {e!r}")
        sp = _sampling_params(req)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        seq_id = await _add_or_400(prompt, sp)
        if not req.stream:
            token_ids, reason = await server.wait_completion(seq_id)
            return {
                "id": request_id, "object": "chat.completion", "created": created,
                "model": server.model_name,
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": server.engine.tokenizer.decode(token_ids)},
                             "logprobs": None, "finish_reason": reason}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(token_ids),
                          "total_tokens": len(prompt) + len(token_ids)},
            }

        first = True

        def make_chunks(new_token_ids, finished, reason):
            nonlocal first
            chunks = []
            if first:
                # 首个 chunk 携带 role（OpenAI 流式协议）
                chunks.append(_sse({"id": request_id, "object": "chat.completion.chunk",
                                    "created": created, "model": server.model_name,
                                    "choices": [{"index": 0,
                                                 "delta": {"role": "assistant", "content": ""},
                                                 "finish_reason": None}]}))
                first = False
            delta = {} if finished else {"content": server.engine.tokenizer.decode(new_token_ids)}
            chunks.append(_sse({"id": request_id, "object": "chat.completion.chunk",
                                "created": created, "model": server.model_name,
                                "choices": [{"index": 0, "delta": delta,
                                             "finish_reason": reason if finished else None}]}))
            return chunks

        return _stream_response(request, seq_id, make_chunks)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
        if req.stream and len(prompts) != 1:
            raise HTTPException(status_code=400, detail="流式目前仅支持单个 prompt")
        tokenized = [server.engine.tokenizer.encode(p) for p in prompts]
        sp = _sampling_params(req)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        seq_ids = [await _add_or_400(t, sp) for t in tokenized]
        if not req.stream:
            choices, total_completion = [], 0
            for i, seq_id in enumerate(seq_ids):
                token_ids, reason = await server.wait_completion(seq_id)
                total_completion += len(token_ids)
                choices.append({"index": i,
                                "text": server.engine.tokenizer.decode(token_ids),
                                "logprobs": None, "finish_reason": reason})
            prompt_tokens = sum(len(t) for t in tokenized)
            return {
                "id": request_id, "object": "text_completion", "created": created,
                "model": server.model_name, "choices": choices,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": total_completion,
                          "total_tokens": prompt_tokens + total_completion},
            }

        def make_chunks(new_token_ids, finished, reason):
            return [_sse({"id": request_id, "object": "text_completion",
                          "created": created, "model": server.model_name,
                          "choices": [{"index": 0,
                                       "text": server.engine.tokenizer.decode(new_token_ids),
                                       "finish_reason": reason if finished else None}]})]

        return _stream_response(request, seq_ids[0], make_chunks)

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Z-vllm OpenAI 兼容推理服务")
    p.add_argument("--model", required=True, help="模型目录（HuggingFace 格式）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--moe-tp-size", type=int, default=None,
                   help="专家内 TP；MoE 模型需满足 moe_tp_size * moe_ep_size == tensor_parallel_size")
    p.add_argument("--moe-ep-size", type=int, default=1, help="专家并行 EP")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--kvcache-block-size", type=int, default=256)
    p.add_argument("--max-graph-bs", type=int, default=512, help="CUDA Graph 捕获的最大 batch size")
    p.add_argument("--master-port", type=int, default=2333, help="NCCL 进程组初始化端口")
    p.add_argument("--shm-name", default="zvllm", help="多卡方法调用的共享内存名")
    p.add_argument("--shm-size", type=int, default=2 ** 20, help="共享内存大小（字节）")
    p.add_argument("--enforce-eager", action="store_true", help="禁用 CUDA Graph（MoE 会自动强制）")
    return p.parse_args()


def main():
    args = parse_args()
    kwargs = dict(
        tensor_parallel_size=args.tensor_parallel_size,
        moe_tp_size=args.moe_tp_size,
        moe_ep_size=args.moe_ep_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.kvcache_block_size,
        max_graph_bs=args.max_graph_bs,
        master_port=args.master_port,
        shm_name=args.shm_name,
        shm_size=args.shm_size,
        enforce_eager=args.enforce_eager,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    print(f"Loading model {args.model} ...")
    engine = LLM(args.model, **kwargs)
    server = InferenceServer(engine)
    app = create_app(server)
    print(f"Z-vllm serving {server.model_name} on http://{args.host}:{args.port} (OpenAI 兼容)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
