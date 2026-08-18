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
每序列 token 队列上等待；后台线程逐步推进引擎并把新生成 token 分发给所有
在途请求，多个请求的 prefill/decode 因此能按调度器的混合批自然交错。
stop / presence_penalty / frequency_penalty 等 OpenAI 字段被接受但忽略。
"""
import argparse
import json
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from zvllm import LLM, SamplingParams


class InferenceServer:
    """引擎封装：后台 step-loop 线程 + 每序列 token 队列。"""

    def __init__(self, engine: LLM):
        self.engine = engine
        self.model_name = os.path.basename(os.path.normpath(engine.config.model))
        self.lock = threading.Lock()
        self.queues: dict[int, queue.Queue] = {}
        self._shutdown = threading.Event()
        self.step_thread = threading.Thread(target=self._step_loop, daemon=True)
        self.step_thread.start()

    def _step_loop(self):
        while not self._shutdown.is_set():
            try:
                with self.lock:
                    if self.engine.scheduler.is_finished():
                        idle = True
                    else:
                        idle = False
                        outputs, _ = self.engine.step()
                        for seq_id, new_token_ids, finished, reason in outputs:
                            q = self.queues.get(seq_id)
                            if q is not None:
                                q.put((new_token_ids, finished, reason))
            except Exception as e:
                # 引擎异常（如显存 OOM）：通知所有在途请求，避免客户端线程永久挂起
                print(f"[zvllm] step loop aborted: {e!r}")
                with self.lock:
                    for q in list(self.queues.values()):
                        q.put(None)
                break
            if idle:
                time.sleep(0.01)

    def add_prompt(self, prompt_token_ids: list[int], sampling_params: SamplingParams) -> int:
        """入队新序列并注册其 token 队列，返回 seq_id。"""
        q = queue.Queue()
        with self.lock:
            seq = self.engine.add_request(prompt_token_ids, sampling_params)
            self.queues[seq.seq_id] = q
        return seq.seq_id

    def _unregister(self, seq_id: int):
        with self.lock:
            self.queues.pop(seq_id, None)

    def _next_event(self, seq_id: int):
        """阻塞取该序列的下一个事件（带看门狗）。

        step 线程异常退出时预期会推送 None 哨兵；若它以绕过哨兵的方式死亡
        （如致命 BaseException），这里及时抛出而不是让客户端永久挂起。
        """
        q = self.queues[seq_id]
        while True:
            try:
                return q.get(timeout=1.0)
            except queue.Empty:
                if not self.step_thread.is_alive():
                    raise RuntimeError("推理引擎 step 线程已退出，请求中止")

    def wait_completion(self, seq_id: int) -> tuple[list[int], str]:
        """阻塞至序列结束，返回 (完整补全 token 列表, finish_reason)。"""
        token_ids: list[int] = []
        reason = None
        while True:
            item = self._next_event(seq_id)
            if item is None:
                raise RuntimeError("推理引擎异常退出，请求中止")
            new_token_ids, finished, finish_reason = item
            token_ids.extend(new_token_ids)
            if finished:
                reason = finish_reason
                break
        self._unregister(seq_id)
        return token_ids, reason

    def iter_stream(self, seq_id: int):
        """逐条 yield (new_token_ids, finished, finish_reason)；结束或放弃时注销队列。"""
        try:
            while True:
                item = self._next_event(seq_id)
                if item is None:
                    raise RuntimeError("推理引擎异常退出，请求中止")
                yield item
                if item[1]:
                    break
        finally:
            self._unregister(seq_id)

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
    # 以下字段为兼容 OpenAI 客户端而接受，当前未支持（忽略）：
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


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _sampling_params(req) -> SamplingParams:
    return SamplingParams(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                          max_tokens=req.max_tokens, seed=req.seed)


def create_app(server: InferenceServer) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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

    def _add_or_400(prompt_token_ids: list[int], sp: SamplingParams) -> int:
        try:
            return server.add_prompt(prompt_token_ids, sp)
        except AssertionError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        try:
            prompt = server.engine.tokenizer.apply_chat_template(
                req.messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"chat template 处理失败: {e!r}")
        sp = _sampling_params(req)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        seq_id = _add_or_400(prompt, sp)
        if not req.stream:
            token_ids, reason = server.wait_completion(seq_id)
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

        def stream():
            first = True
            for new_token_ids, finished, reason in server.iter_stream(seq_id):
                if first:
                    # 首个 chunk 携带 role（OpenAI 流式协议）
                    yield _sse({"id": request_id, "object": "chat.completion.chunk",
                                "created": created, "model": server.model_name,
                                "choices": [{"index": 0,
                                             "delta": {"role": "assistant", "content": ""},
                                             "finish_reason": None}]})
                    first = False
                delta = {} if finished else {"content": server.engine.tokenizer.decode(new_token_ids)}
                yield _sse({"id": request_id, "object": "chat.completion.chunk",
                            "created": created, "model": server.model_name,
                            "choices": [{"index": 0, "delta": delta,
                                         "finish_reason": reason if finished else None}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
        if req.stream and len(prompts) != 1:
            raise HTTPException(status_code=400, detail="流式目前仅支持单个 prompt")
        tokenized = [server.engine.tokenizer.encode(p) for p in prompts]
        sp = _sampling_params(req)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        seq_ids = [_add_or_400(t, sp) for t in tokenized]
        if not req.stream:
            choices, total_completion = [], 0
            for i, seq_id in enumerate(seq_ids):
                token_ids, reason = server.wait_completion(seq_id)
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

        def stream():
            for new_token_ids, finished, reason in server.iter_stream(seq_ids[0]):
                yield _sse({"id": request_id, "object": "text_completion",
                            "created": created, "model": server.model_name,
                            "choices": [{"index": 0,
                                         "text": server.engine.tokenizer.decode(new_token_ids),
                                         "finish_reason": reason if finished else None}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

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
