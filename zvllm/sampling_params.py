from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0    # 0 表示贪心（argmax）
    top_k: int = -1             # 只保留概率最高的 k 个 token，-1 不限制
    top_p: float = 1.0          # nucleus 采样，保留累积概率首次达到 p 的最小前缀集
    seed: int | None = None     # 给定后该序列的随机流可复现
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be >= 0 (0 means greedy)"
        assert 0 < self.top_p <= 1, "top_p must be in (0, 1]"
        assert self.top_k >= -1, "top_k must be >= -1 (-1 means no limit)"
