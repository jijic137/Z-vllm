import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from zvllm.config import Config
from zvllm.sampling_params import SamplingParams, find_stop_match
from zvllm.engine.sequence import Sequence
from zvllm.engine.scheduler import Scheduler


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 延迟导入：model_runner 会拉起 triton/模型层（GPU 专用依赖），
        # 纯 CPU 环境下也要能导入本模块做调度/流式逻辑的单元测试
        from zvllm.engine.model_runner import ModelRunner
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        """创建序列并入队，返回该 Sequence（调用方可用 seq_id 跟踪进度/流式输出）。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        assert len(prompt) > 0, "prompt 不能为空"
        assert len(prompt) <= self.config.max_model_len, \
            f"prompt 长度 {len(prompt)} 超过 max_model_len（{self.config.max_model_len}）"
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def schedule(self) -> tuple[list[Sequence], bool]:
        """阶段 A（CPU）：一轮调度，返回 (本步序列, 是否含 prefill)。

        只改调度器队列状态、纯 CPU、耗时微秒级。服务层（api_server）应在
        持服务锁时调用，使入队/取消与调度互斥。
        """
        return self.scheduler.schedule()

    def run_model(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """阶段 B（GPU）：前向 + 采样，返回每条序列采样的 token（与 seqs 位置对应）。

        不改任何调度器/序列状态，因此可以不在服务锁内执行——GPU 计算窗口里
        请求入队与取消不会被阻塞。
        """
        return self.model_runner.call("run", seqs, is_prefill)

    def finalize_step(self, seqs: list[Sequence], is_prefill: bool, token_ids: list[int]) -> list[tuple]:
        """阶段 C（CPU）：更新序列/调度器状态、检查停止串，返回本步 outputs。

        阶段 B 期间被取消的序列（status 已 FINISHED，如客户端断连 abort）跳过：
        其状态已由取消流程定型，本步为它采样的 token 丢弃。
        本步只是 prefill 一段（chunked prefill 未走完）的序列不产出 token，
        不进入 outputs（其采样 token 只是 logit 副产物，从未并入序列）。
        """
        live = [(seq, tok) for seq, tok in zip(seqs, token_ids) if not seq.is_finished]
        # 先按 scheduler.postprocess 的跳过口径算出"哪些序列本步会真正产出
        # token"（num_cached 尚未回写、num_tokens 尚未追加，必须在此刻评估），
        # 再回写状态
        emitted = [not (is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens)
                   for seq, _ in live]
        if live:
            self.scheduler.postprocess([seq for seq, _ in live],
                                       [tok for _, tok in live], is_prefill)
            self._check_stop_strings([seq for seq, _ in live])
        outputs = []
        for (seq, tok), will_emit in zip(live, emitted):
            if not will_emit:
                continue    # 本步只是 prefill 的一段：未产出 token
            reason = seq.finish_reason if seq.is_finished else None
            outputs.append((seq.seq_id, [tok], seq.is_finished, reason))
        return outputs

    def step(self):
        """执行一步原子调度 + 推理（CLI 路径）：schedule → run_model → finalize_step。

        返回 (outputs, num_tokens)：
        - outputs：本步批内每条实际产出 token 的序列的 (seq_id, new_token_ids,
          finished, finish_reason)；new_token_ids 为该序列本步新生成的 token
          （目前恒为 1 个），finished 表示该序列本步结束后是否终止
          （eos / max_tokens / 停止串），finish_reason 为 OpenAI 语义的结束原因
          （"stop" / "length" / "abort"），未结束时为 None。
        - num_tokens：本步处理的 token 数（prefill/混合步为正、纯 decode 步为负），用于吞吐展示。
        """
        seqs, is_prefill = self.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.run_model(seqs, is_prefill)
        return self.finalize_step(seqs, is_prefill, token_ids), num_tokens

    def _check_stop_strings(self, seqs):
        """OpenAI 停止串语义：生成文本以任一停止串结尾时立即终止（停止串本身保留在输出里）。

        每步对设置了 stop 的序列解码其补全文本并做尾部匹配；本引擎规模下
        逐步全量解码的开销可忽略（generate_stream 本就每步全量解码）。"""
        for seq in seqs:
            if seq.is_finished or not seq.stop:
                continue
            hit = find_stop_match(self.tokenizer.decode(seq.completion_token_ids), seq.stop)
            if hit is not None:
                seq.stop_reason = hit
                seq.finish_reason = "stop"
                self.scheduler.finish(seq)

    def abort_request(self, request_id: int) -> bool:
        """取消指定请求（request_id 为 add_request 返回的 Sequence.seq_id）。

        仍在 waiting/running 中的请求立即移出队列并释放 KV 块，finish_reason 记为
        "abort"；已结束或未知 id 返回 False（幂等）。

        线程安全：引擎本身非线程安全，并发调用方（如 api_server 的 step 线程）
        需自行用外部锁将本方法与 step() 串行化。
        """
        for seq in list(self.scheduler.waiting) + list(self.scheduler.running):
            if seq.seq_id == request_id:
                self.scheduler.abort(seq)
                return True
        return False

    def is_finished(self):
        return self.scheduler.is_finished()

    def _add_requests(self, prompts, sampling_params) -> list[Sequence]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(prompts) == len(sampling_params), "prompts 与 sampling_params 数量不一致"
        return [self.add_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        stream: bool = False,
    ):
        """生成一批 prompt 的补全。

        stream=False（默认）：阻塞至全部请求完成，按输入顺序返回
        list[{"text": 完整补全文本, "token_ids": 完整补全 token 列表,
              "finish_reason": "stop" / "length" / "abort"}]；
        stream=True：返回生成器，逐请求逐 token 产出事件
        （{"index", "delta", "text", "token_ids", "finished"}），见 generate_stream。
        """
        if stream:
            return self.generate_stream(prompts, sampling_params)
        seqs = self._add_requests(prompts, sampling_params)
        pbar = tqdm(total=len(seqs), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        completion_ids: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            outputs, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, new_token_ids, finished, _reason in outputs:
                completion_ids[seq_id] = completion_ids.get(seq_id, []) + new_token_ids
                if finished:
                    pbar.update(1)
        pbar.close()
        # 被 abort_request 取消的请求可能一个 token 都未产出，用 .get 兜底
        return [{"text": self.tokenizer.decode(completion_ids.get(seq.seq_id, [])),
                 "token_ids": completion_ids.get(seq.seq_id, []),
                 "finish_reason": seq.finish_reason} for seq in seqs]

    def generate_stream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        """流式生成：生成器，每个请求每产生一个新 token 就 yield 一个事件。

        事件字段：
        - index：该请求在 prompts 中的下标
        - delta：本步新生成 token 的文本（增量解码）
        - text：到目前为止的完整补全文本
        - token_ids：到目前为止的完整补全 token 列表
        - finished：该请求是否结束（每个请求的最后一个事件为 True）
        - finish_reason：OpenAI 语义的结束原因（"stop" / "length" / "abort"），未结束时为 None

        逐请求取消：消费途中可随时调用 llm.abort_request(seq_id)；被取消的请求
        会在下一轮循环立即收到 finished=True 的终止事件（finish_reason="abort"），
        其 KV 块随即释放，引擎可继续处理其余请求。
        """
        seqs = self._add_requests(prompts, sampling_params)
        index_by_seq_id = {seq.seq_id: i for i, seq in enumerate(seqs)}
        completion_ids: dict[int, list[int]] = {}
        pending = {seq.seq_id for seq in seqs}
        while pending:
            # 被外部 abort_request 取消的序列不会再产出 token：先补发终止事件，
            # 避免在已清空的引擎上多调一次 step()
            for seq in seqs:
                if seq.seq_id in pending and seq.is_finished:
                    pending.discard(seq.seq_id)
                    token_ids = completion_ids.get(seq.seq_id, [])
                    yield {
                        "index": index_by_seq_id[seq.seq_id],
                        "delta": "",
                        "text": self.tokenizer.decode(token_ids),
                        "token_ids": token_ids,
                        "finished": True,
                        "finish_reason": seq.finish_reason,
                    }
            if not pending:
                break
            outputs, _ = self.step()
            for seq_id, new_token_ids, finished, reason in outputs:
                token_ids = completion_ids.get(seq_id, []) + new_token_ids
                completion_ids[seq_id] = token_ids
                if finished:
                    pending.discard(seq_id)
                yield {
                    "index": index_by_seq_id[seq_id],
                    "delta": self.tokenizer.decode(new_token_ids),
                    "text": self.tokenizer.decode(token_ids),
                    "token_ids": token_ids,
                    "finished": finished,
                    "finish_reason": reason,
                }
        completion_ids.clear()
