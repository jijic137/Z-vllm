from collections import deque

from zvllm.config import Config
from zvllm.engine.sequence import Sequence, SequenceStatus
from zvllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """一步调度（prefill/decode 混合）：
        1) decode 优先：running 中每条序列出 1 个 token，受本步 token 预算约束；
           块耗尽时按旧策略抢占（牺牲最年轻的 running，prefix cache 兜底重算）。
        2) prefill：用剩余预算从 waiting 队首开始按 FCFS 调度；任意序列都可被切块
           （取 min(需要, 剩余)），队首完成后继续调度下一条——不再有"仅队首可切块"
           与"prefill 步不 decode"的队头阻塞。
        返回 (本步序列, 是否含 prefill)：后者决定 ModelRunner 走 varlen prefill
        路径还是 decode 快路径（CUDA graph）。纯 decode 步保持原快路径。"""
        scheduled_seqs = []
        num_batched_tokens = 0

        # 1) decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs \
                and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += 1
        num_decode = len(scheduled_seqs)

        # 2) prefill（含抢占后立即重入队首的序列）
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if not scheduled_seqs:
            # 调度不出任何序列。两种情况都是配置/容量错误，给出明确错误而非 assert 崩溃：
            # 1) 队首 waiting 序列的 KV 装不下（含 running 唯一序列抢占自己后回到队首的情况）；
            # 2) waiting/running 全空（引擎应先检查 is_finished，理论不可达）。
            head = self.waiting[0] if self.waiting else None
            if head is not None:
                raise RuntimeError(
                    f"调度失败：队首序列需 {head.num_blocks} 个 KV 块，超过 KV cache 总量 "
                    f"{len(self.block_manager.blocks)}（请增大显存/减小 max_model_len）"
                )
            raise RuntimeError("调度失败：无序列可调度（KV cache 可能过小）")
        # 把已调度的 decode 序列按原顺序放回队首（未调度到的 running 保持在队后）
        self.running.extendleft(reversed(scheduled_seqs[:num_decode]))
        return scheduled_seqs, any(seq.is_prefill for seq in scheduled_seqs)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence):
        """结束序列：释放其 KV 块并移出 running 队列（幂等）。

        由 postprocess（eos / max_tokens 命中）与引擎（停止串命中）调用；
        调用前须已设置 seq.finish_reason。"""
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        if seq in self.running:
            self.running.remove(seq)

    def abort(self, seq: Sequence):
        """取消请求：移出 waiting/running 队列并释放 KV 块，finish_reason 记 "abort"（幂等）。

        chunked prefill 半途的序列同样适用：已分配的块一并释放。"""
        seq.finish_reason = "abort"
        if seq in self.waiting:
            self.waiting.remove(seq)
        self.finish(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if seq.num_completion_tokens == seq.max_tokens:
                seq.finish_reason = "length"
            elif not seq.ignore_eos and token_id == self.eos:
                seq.finish_reason = "stop"
            else:
                continue
            self.finish(seq)
