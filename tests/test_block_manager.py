"""BlockManager 纯逻辑单测（无需 GPU）。运行：python tests/test_block_manager.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zvllm.engine.sequence import Sequence
from zvllm.engine.block_manager import BlockManager

BS = 4
Sequence.block_size = BS


def decode_slot(seq):
    """prepare_decode 使用的槽位公式：最后一个 token 的物理 KV slot"""
    return seq.block_table[-1] * BS + seq.last_block_num_tokens - 1


def run_decode_case(n0, steps, num_blocks=64):
    """decode 扩容回归（上游 nano-vllm issue #240 / #66）：
    旧条件 len(seq) % block_size == 1 对 prompt 长度 % block_size == 1 的序列
    多分配一块，使 decode 槽位公式指向错误块。这里对每个 n0 验证：
    1) block_table 长度始终 == num_blocks(len)；
    2) 槽位公式 == 位置 len-1 的真实 slot；
    3) 已用块数 == num_blocks（无泄漏、无多分配）。"""
    seq = Sequence(list(range(n0)))
    bm = BlockManager(num_blocks, BS)
    cached = bm.can_allocate(seq)
    assert cached == 0, (n0, cached)
    bm.allocate(seq, cached)
    assert len(seq.block_table) == seq.num_blocks
    seq.num_cached_tokens = n0
    for s in range(steps):
        N = len(seq)
        assert bm.can_append(seq), f"n0={n0} step={s}: can_append False"
        bm.may_append(seq)
        assert len(seq.block_table) == seq.num_blocks, \
            f"n0={n0} step={s}: table {len(seq.block_table)} != num_blocks {seq.num_blocks} (N={N})"
        slot = decode_slot(seq)
        true_block, true_off = (N - 1) // BS, (N - 1) % BS
        assert slot == seq.block_table[true_block] * BS + true_off, \
            f"n0={n0} step={s}: slot {slot} != true (N={N})"
        seq.append_token(10000 + N)
    # 收尾：最后一次 append 的 token 所在块按"下一次调度时分配"的语义尚未分配，
    # 走一次 can_append/may_append 补齐后再验证无泄漏、无多分配
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert len(seq.block_table) == seq.num_blocks
    used = num_blocks - len(bm.free_block_ids)
    assert used == seq.num_blocks, (n0, used, seq.num_blocks)
    print(f"run_decode_case n0={n0}: {steps} decode steps OK")


def test_decode_growth():
    for n0 in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 17):
        run_decode_case(n0, 12)
    # 补上"下一次调度才分配"的边界：表长不足时 _block_needed 恰好为 1
    seq = Sequence(list(range(5)))
    bm = BlockManager(64, BS)
    bm.allocate(seq, bm.can_allocate(seq))
    for _ in range(12):
        assert bm.can_append(seq)
        bm.may_append(seq)
        seq.append_token(99)
    assert len(seq.block_table) == 4 and bm._block_needed(seq) == 1
    bm.may_append(seq)
    assert len(seq.block_table) == seq.num_blocks == 5
    assert 64 - len(bm.free_block_ids) == seq.num_blocks
    print("test_decode_growth OK")


def test_prefix_cache():
    bm = BlockManager(32, BS)
    common = list(range(32))          # 8 个满块
    s1 = Sequence(common + [100, 101])
    s2 = Sequence(common + [200, 201, 202])
    bm.allocate(s1, bm.can_allocate(s1))
    # 模拟 prefill 完成后的登记（与 postprocess 相同顺序：先 hash 再更新 num_cached）
    s1.num_scheduled_tokens = 34
    bm.hash_blocks(s1)
    s1.num_cached_tokens = 34
    s1.num_scheduled_tokens = 0
    # s2 应命中 8 个缓存块，只需新分配 1 块
    cached = bm.can_allocate(s2)
    assert cached == 8, cached
    nfree_before = len(bm.free_block_ids)
    bm.allocate(s2, cached)
    assert len(bm.free_block_ids) == nfree_before - 1
    assert s2.block_table[:8] == s1.block_table[:8]      # 共享块
    assert s2.num_cached_tokens == 32
    # 共享块 ref_count = 2；释放 s1 后块仍在（s2 仍引用）
    shared = s1.block_table[0]
    assert bm.blocks[shared].ref_count == 2
    bm.deallocate(s1)
    assert bm.blocks[shared].ref_count == 1
    assert shared not in bm.free_block_ids
    bm.deallocate(s2)
    assert bm.blocks[shared].ref_count == 0
    assert shared in bm.free_block_ids
    print("test_prefix_cache OK")


def test_allocate_reuse_cleared_hash():
    # 曾被哈希登记的块重新分配后，旧哈希不得再被命中
    bm = BlockManager(8, BS)
    s1 = Sequence(list(range(8)))
    bm.allocate(s1, bm.can_allocate(s1))
    s1.num_scheduled_tokens = 8
    bm.hash_blocks(s1)
    s1.num_cached_tokens = 8
    s1.num_scheduled_tokens = 0
    victim = s1.block_table[0]
    old_hash = bm.blocks[victim].hash
    bm.deallocate(s1)
    # 先耗尽其余空闲块，再让新序列被迫复用 victim 物理块
    filler = Sequence(list(range(200, 224)))    # 6 块
    bm.allocate(filler, bm.can_allocate(filler))
    s2 = Sequence(list(range(300, 308)))
    bm.allocate(s2, bm.can_allocate(s2))
    assert victim in s2.block_table, "测试前提：victim 块被复用"
    assert bm.blocks[victim].hash == -1
    assert bm.hash_to_block_id.get(old_hash, -1) == -1, "复用后旧哈希必须失效"
    print("test_allocate_reuse_cleared_hash OK")


def test_can_allocate_shortage():
    bm = BlockManager(2, BS)
    s = Sequence(list(range(12)))    # 需要 3 块 > 2
    assert bm.can_allocate(s) == -1
    s = Sequence(list(range(8)))     # 恰好 2 块
    assert bm.can_allocate(s) == 0
    print("test_can_allocate_shortage OK")


if __name__ == "__main__":
    test_decode_growth()
    test_prefix_cache()
    test_allocate_reuse_cleared_hash()
    test_can_allocate_shortage()
    print("ALL BLOCK MANAGER TESTS PASSED")
