import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from zvllm.config import Config
from zvllm.sampling_params import SamplingParams
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

    def step(self):
        """执行一步调度 + 推理。

        返回 (outputs, num_tokens)：
        - outputs：本步批内每条被调度序列的 (seq_id, new_token_ids, finished, finish_reason)；
          new_token_ids 为该序列本步新生成的 token（目前恒为 1 个），
          finished 表示该序列本步结束后是否终止（eos / max_tokens），
          finish_reason 为 OpenAI 语义的结束原因（"stop" / "length"），未结束时为 None。
        - num_tokens：本步处理的 token 数（prefill/混合步为正、纯 decode 步为负），用于吞吐展示。
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.is_finished:
                reason = "length" if seq.num_completion_tokens >= seq.max_tokens else "stop"
            else:
                reason = None
            outputs.append((seq.seq_id, [token_id], seq.is_finished, reason))
        return outputs, num_tokens

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
        list[{"text": 完整补全文本, "token_ids": 完整补全 token 列表}]；
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
        return [{"text": self.tokenizer.decode(completion_ids[seq.seq_id]),
                 "token_ids": completion_ids[seq.seq_id]} for seq in seqs]

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
        - finish_reason：OpenAI 语义的结束原因（"stop" / "length"），未结束时为 None

        注意：生成器未消费完就放弃（break / 异常）时，未完成请求仍驻留在引擎内，
        该引擎不适合再混入其他请求（本版本不提供逐请求取消）。
        """
        seqs = self._add_requests(prompts, sampling_params)
        index_by_seq_id = {seq.seq_id: i for i, seq in enumerate(seqs)}
        completion_ids: dict[int, list[int]] = {}
        pending = len(seqs)
        while pending:
            outputs, _ = self.step()
            for seq_id, new_token_ids, finished, reason in outputs:
                token_ids = completion_ids.get(seq_id, []) + new_token_ids
                completion_ids[seq_id] = token_ids
                if finished:
                    pending -= 1
                yield {
                    "index": index_by_seq_id[seq_id],
                    "delta": self.tokenizer.decode(new_token_ids),
                    "text": self.tokenizer.decode(token_ids),
                    "token_ids": token_ids,
                    "finished": finished,
                    "finish_reason": reason,
                }
        completion_ids.clear()
