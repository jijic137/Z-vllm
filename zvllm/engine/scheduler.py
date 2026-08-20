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
        # 投机解码（n-gram / PLD 草稿，零训练）：开关与参数缺省 off，兼容无字段的测试配置
        self.spec_enabled = getattr(config, "spec_decode", "off") == "ngram"
        self.spec_gamma = getattr(config, "spec_gamma", 4)
        self.spec_ngram = getattr(config, "spec_ngram", 4)
        self.max_model_len = getattr(config, "max_model_len", None)
        self.spec_stats = {"proposed_tokens": 0, "accepted_tokens": 0, "steps": 0}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """一步调度（prefill/decode 混合）：
        1) decode 优先：running 中每条序列出 1 个 token，受本步 token 预算约束；
           块耗尽时按旧策略抢占（牺牲最年轻的 running，prefix cache 兜底重算）。
        投机开启时 decode 序列先取 n-gram 草稿（≤spec_gamma），本步按 1+γ 个
        token 计预算并预留草稿 KV 块；草稿不足/块不足时逐级降级（先减草稿、再抢占）。
        2) prefill：用剩余预算从 waiting 队首开始按 FCFS 调度；任意序列都可被切块
           （取 min(需要, 剩余)），队首完成后继续调度下一条——不再有"仅队首可切块"
           与"prefill 步不 decode"的队头阻塞。
        返回 (本步序列, 是否含 prefill)：后者与批内草稿共同决定 ModelRunner 走
        varlen prefill 路径还是 decode 快路径（CUDA graph）。纯 decode 无草稿步
        保持原快路径。"""
        scheduled_seqs = []
        num_batched_tokens = 0

        # 1) decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs \
                and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.running.popleft()
            # 草稿上限：max_tokens 余量与本步 token 预算两个上界同时生效（取 min）
            draft = self._make_draft(seq) if self.spec_enabled else []
            if draft:
                max_emit = seq.max_tokens - seq.num_completion_tokens
                if self.max_model_len is not None:
                    max_emit = min(max_emit, self.max_model_len - len(seq))
                cap = max(0, min(max_emit - 1, self.max_num_batched_tokens - num_batched_tokens - 1))
                draft = draft[:cap]
            # KV 预留：装不下时先降级草稿长度（每步少验一个 token），再走抢占
            while not self.block_manager.can_append_n(seq, len(draft)):
                if draft:
                    draft.pop()
                    continue
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)    # 唯一序列：自身抢占后回队首，本步弃掉
                    break
            else:
                seq.num_scheduled_tokens = 1 + len(draft)
                seq.is_prefill = False
                seq.draft_tokens = draft
                self.block_manager.may_append_n(seq, len(draft))
                if draft:
                    self.spec_stats["proposed_tokens"] += len(draft)
                    self.spec_stats["steps"] += 1
                scheduled_seqs.append(seq)
                num_batched_tokens += seq.num_scheduled_tokens
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
                if self.spec_enabled and seq.ngram_positions is None:
                    self._build_ngram(seq)
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
        seq.draft_tokens = []
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
        """单 token 产出入口（非投机路径）；等价于 postprocess_multi([[t]])。"""
        self.postprocess_multi(seqs, [[t] for t in token_ids], is_prefill)

    def postprocess_multi(self, seqs: list[Sequence], token_lists: list[list[int]], is_prefill: bool) -> list[list[int]]:
        """回写一批序列的产出 token（每条可多 token：投机验证的接受前缀 + bonus）。

        逐 token 追加并检查 max_tokens/eos（中途命中即截断、标记结束）；按各路径的
        KV 可信长度推进 prefix cache 哈希游标；投机序列把新官方位置插入 n-gram 索引。
        返回每条序列实际并入/产出的 token 列表（chunked prefill 半程步为空列表）。"""
        emitted = []
        for seq, toks in zip(seqs, token_lists):
            emitted.append(self._postprocess_one(seq, toks, is_prefill))
        return emitted

    def _postprocess_one(self, seq: Sequence, toks: list[int], is_prefill: bool) -> list[int]:
        # 与 finalize 旧口径一致：chunked prefill 半程步不产出 token
        if is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens:
            safe_len = seq.num_cached_tokens + seq.num_scheduled_tokens
            self.block_manager.hash_blocks_upto(seq, safe_len)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            return []
        old_len = seq.num_tokens
        out = []
        for tok in toks:
            seq.append_token(tok)
            out.append(tok)
            if seq.num_completion_tokens == seq.max_tokens:
                seq.finish_reason = "length"
                break
            if not seq.ignore_eos and tok == self.eos:
                seq.finish_reason = "stop"
                break
        if seq.draft_tokens:
            # 投机：bonus token 的 KV 尚未写入（下一步回喂时重写），可信长度 = 新长度-1
            safe_len = seq.num_tokens - 1
            seq.num_cached_tokens = safe_len
            self.spec_stats["accepted_tokens"] += max(0, len(out) - 1)
        else:
            # prefill 完成步的新 token（位置 old_len）KV 尚未写入（下一步回喂才写）；
            # decode 只重写位置 old_len-1，新 token 同样延迟 -> 两者可信长度都是 old_len
            safe_len = old_len
            seq.num_cached_tokens += seq.num_scheduled_tokens
        # 新官方位置入 n-gram 索引（无索引/无新位置时为 no-op；γ=0 的步也要更新，
        # 否则后续步骤的草稿匹配范围会随时间退化）
        self._update_ngram(seq, old_len)
        self.block_manager.hash_blocks_upto(seq, safe_len)
        seq.num_scheduled_tokens = 0
        if seq.finish_reason is not None:
            self.finish(seq)
        return out

    # ---------------------------------------------------------------- 投机草稿

    def _build_ngram(self, seq: Sequence):
        """prefill 完成时建 n-gram 索引：{n-gram: 出现结束位置列表}（仅 rank0 进程）。"""
        n, toks = self.spec_ngram, seq.token_ids
        d = {}
        for p in range(n - 1, len(toks)):
            d.setdefault(tuple(toks[p - n + 1: p + 1]), []).append(p)
        seq.ngram_positions = d

    def _update_ngram(self, seq: Sequence, old_len: int):
        """decode 后把新官方位置 [old_len, len) 的 n-gram 增量插入索引。"""
        n, toks = self.spec_ngram, seq.token_ids
        d = seq.ngram_positions
        if d is None:
            return
        for p in range(max(n - 1, old_len), len(toks)):
            d.setdefault(tuple(toks[p - n + 1: p + 1]), []).append(p)

    def _make_draft(self, seq: Sequence) -> list[int]:
        """PLD：尾部 n 个 token 在序列自身历史中复现时，取该出现处后续 ≤γ 个 token。

        取能给出完整草稿的最新出现，否则给部分草稿的最新出现；跳过尾部自身出现。
        草稿是确定性点质量分布，接受-拒绝按 p(x) 概率接受（model_runner 侧执行）。"""
        n = self.spec_ngram
        toks = seq.token_ids
        L = len(toks)
        gamma = self.spec_gamma
        if L < n + 1:
            return []
        pos_list = seq.ngram_positions.get(tuple(toks[L - n:])) if seq.ngram_positions else None
        if not pos_list:
            return []
        trivial = L - 1    # 索引存 gram 的"结束"位置，尾部 gram 自身结束于 L-1
        for p in reversed(pos_list):
            if p != trivial and p + gamma <= L - 1:
                return toks[p + 1: p + 1 + gamma]
        for p in reversed(pos_list):
            if p != trivial and p + 1 <= L - 1:
                return toks[p + 1: L]
        return []
