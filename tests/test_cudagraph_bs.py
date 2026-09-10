"""CUDA Graph batch size 覆盖性纯 CPU 单测（无需 GPU）。运行：python tests/test_cudagraph_bs.py

回归点：decode 步按"最小的不小于实际 batch 的已捕获图"选图 replay。早期实现只取
[1,2,4,8] + range(16, max_bs+1, 16)，最后一段残批（16*floor(max_bs/16) < bs <= max_bs）
没有对应的图，`next(...)` 直接抛 StopIteration——max_num_seqs=100 或 max_graph_bs=20
这类非 16 倍数配置即可让整个引擎崩溃（api_server 侧表现为 step loop 异常退出、
全部在途请求失败）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from zvllm.engine.model_runner import cudagraph_batch_sizes  # noqa: E402


@pytest.mark.parametrize("max_num_seqs,max_graph_bs", [
    (512, 512),      # 默认配置
    (20, 512),       # 只调小 max_num_seqs
    (512, 20),       # 只调小 max_graph_bs
    (100, 512),
    (17, 17),
    (64, 64),
    (1, 512),        # 退化为单序列
])
def test_covers_every_batch_size(max_num_seqs, max_graph_bs):
    sizes = cudagraph_batch_sizes(max_num_seqs, max_graph_bs)
    max_bs = min(max_num_seqs, max_graph_bs)
    assert sizes == sorted(set(sizes)), "必须是升序去重列表"
    assert max(sizes) == max_bs, f"最大已捕获 batch 应等于 max_bs（{max_bs}），实际 {max(sizes)}"
    for bs in range(1, max_bs + 1):
        chosen = next((x for x in sizes if x >= bs), None)
        assert chosen is not None, f"bs={bs} 在 {sizes} 中找不到可复用的图（会 StopIteration）"
        assert chosen == min(x for x in sizes if x >= bs), "应取最小的可用图"
    print(f"  ok max_num_seqs={max_num_seqs} max_graph_bs={max_graph_bs} -> {sizes}")


def test_regression_non_multiple_of_16():
    """旧实现的列表在 bs=97..100 / 17..20 上找不到图，这里固化新行为。"""
    old_style = [1, 2, 4, 8] + list(range(16, 101, 16))
    assert [b for b in range(1, 101) if next((x for x in old_style if x >= b), None) is None] \
        == [97, 98, 99, 100], "旧列表应恰好漏掉最后一段残批"
    new_style = cudagraph_batch_sizes(100, 512)
    assert all(next((x for x in new_style if x >= b), None) is not None for b in range(1, 101))
    print("  ok 残批 97..100 已覆盖")


if __name__ == "__main__":
    for mns, mgb in [(512, 512), (20, 512), (512, 20), (100, 512), (17, 17), (64, 64), (1, 512)]:
        test_covers_every_batch_size(mns, mgb)
    test_regression_non_multiple_of_16()
    print("ALL PASSED")
