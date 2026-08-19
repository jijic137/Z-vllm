"""Sampler 纯 CPU 单测（seed=None 路径不依赖 GPU；seed 路径仅 CUDA 环境执行）。运行：python tests/test_sampler.py

按 ModelRunner 的生产调用模式 sampler(logits, *sample_args) 构造入参，
防止 prepare_sample 五元组与 Sampler.forward 签名脱节的回归。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from zvllm.layers.sampler import Sampler


def sample_args(temperatures, seq_ids, top_k, top_p, seeds, device=None):
    """与 ModelRunner.prepare_sample 返回同形的五元组"""
    return (
        torch.tensor(temperatures, dtype=torch.float32, device=device),
        seq_ids,
        torch.tensor(top_k, dtype=torch.int64, device=device),
        torch.tensor(top_p, dtype=torch.float32, device=device),
        seeds,
    )


def test_greedy():
    s = Sampler()
    logits = torch.tensor([[1.0, 5.0, 3.0], [9.0, 0.0, 2.0]])
    out = s(logits, *sample_args([0.0, 0.0], [0, 1], [0, 0], [1.0, 1.0], [None, None]))
    assert out.tolist() == [1, 0], out.tolist()
    print("  ok greedy (temperature=0 -> argmax)")


def test_top_k1_is_argmax():
    s = Sampler()
    logits = torch.tensor([[0.5, 3.0, 1.5, 2.0]])
    out = s(logits, *sample_args([1.0], [7], [1], [1.0], [None]))
    assert out.tolist() == [1], out.tolist()
    print("  ok top_k=1 -> 必为 argmax")


def test_top_p_nucleus():
    # token 0 概率极低（~4.5e-5），top_p=0.9 的核集不含它，多次采样都不应出现
    s = Sampler()
    logits = torch.zeros(64, 8)
    logits[:, 1:] = 10.0
    out = s(logits, *sample_args([1.0] * 64, list(range(64)), [0] * 64, [0.9] * 64, [None] * 64))
    assert (out > 0).all(), out.tolist()
    print("  ok top_p 核集排除低概率 token")


def test_mixed_batch():
    # 同批混合：贪心 + top-k 采样 + 贪心
    s = Sampler()
    torch.manual_seed(3)
    logits = torch.randn(4, 16)
    logits[0, 5] = 100.0
    logits[3, 12] = 100.0
    out = s(logits, *sample_args([0.0, 1.0, 1.0, 0.0], [10, 11, 12, 13],
                                 [0, 2, 0, 0], [1.0, 1.0, 1.0, 1.0], [None] * 4))
    assert out.size(0) == 4 and out.dtype == torch.int64
    assert out[0].item() == 5 and out[3].item() == 12, out.tolist()
    assert out[1].item() in logits[1].topk(2).indices.tolist(), out.tolist()
    print("  ok 混合批（贪心 + top-k 同批）")


def test_seed_reproducible():
    if not torch.cuda.is_available():
        print("  skip seed 复现（无 CUDA 环境）")
        return
    torch.manual_seed(0)
    logits = torch.randn(2, 32, device="cuda")
    a = [1.0, 1.0], [5, 5], [0, 0], [1.0, 1.0], [42, 42]
    o1 = Sampler()(logits, *sample_args(*a, device="cuda"))
    o2 = Sampler()(logits, *sample_args(*a, device="cuda"))
    assert (o1 == o2).all(), (o1.tolist(), o2.tolist())
    print("  ok 同 seed 复现")


if __name__ == "__main__":
    test_greedy()
    test_top_k1_is_argmax()
    test_top_p_nucleus()
    test_mixed_batch()
    test_seed_reproducible()
    print("ALL SAMPLER TESTS PASSED")
