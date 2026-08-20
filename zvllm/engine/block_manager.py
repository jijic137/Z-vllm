from collections import deque
import xxhash
import numpy as np

from zvllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.num_hashed_tokens = 0

    def _blocks_needed(self, seq: Sequence, n: int) -> int:
        """为追加 n 个新 token（位置 len..len+n-1）所需的未分配块数。

        最后一个已写 token 位置 len-1 所在块视为已分配（decode/verify 回喂时
        只是重写该槽）；n=0 即普通 decode 的"最后一个 token 所在块"口径。"""
        target = (len(seq) + n - 1) // self.block_size
        return max(0, target + 1 - len(seq.block_table))

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= self._blocks_needed(seq, 0)

    def can_append_n(self, seq: Sequence, n: int) -> bool:
        return len(self.free_block_ids) >= self._blocks_needed(seq, n)

    def may_append(self, seq: Sequence):
        self.may_append_n(seq, 0)

    def may_append_n(self, seq: Sequence, n: int):
        """预留覆盖位置 len-1..len+n-1 的块（投机验证步为草稿 token 预留 KV 槽）。"""
        target = (len(seq) + n - 1) // self.block_size
        while len(seq.block_table) <= target:
            seq.block_table.append(self._allocate_block())

    def hash_blocks_upto(self, seq: Sequence, safe_len: int):
        """把 KV 可信长度 safe_len 内新填满的块写入 prefix cache 哈希表（增量、游标单调）。

        safe_len 语义：位置 < safe_len 的 KV 均已写入且与官方 token 一致。各步的
        safe_len：prefill 段 = 段末（段内槽位本步全部写入）；decode = 旧长度
        （本步只重写位置 len-1，新 token 的 KV 延迟到下一步）；投机验证 = 新长度-1
        （bonus token 的 KV 延迟到下一步回喂时重写）。旧实现对 decode 用"新长度"
        作边界，恰在总长度凑满整块时会把最后一个未写入 KV 的块提前入哈希表，
        复用方读到垃圾 KV；这里按真实可信长度修正。"""
        start = seq.num_hashed_tokens // self.block_size
        end = safe_len // self.block_size
        if start >= end:
            seq.num_hashed_tokens = safe_len
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
        seq.num_hashed_tokens = safe_len
