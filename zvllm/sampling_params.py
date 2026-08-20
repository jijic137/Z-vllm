from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0    # 0 表示贪心（argmax）
    top_k: int = -1             # 只保留概率最高的 k 个 token，-1 不限制
    top_p: float = 1.0          # nucleus 采样，保留累积概率首次达到 p 的最小前缀集
    seed: int | None = None     # 给定后该序列的随机流可复现
    max_tokens: int = 64
    ignore_eos: bool = False
    stop: str | list[str] | None = None   # 停止串：生成文本以其中任一项结尾时立即终止（OpenAI 语义）

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be >= 0 (0 means greedy)"
        assert 0 < self.top_p <= 1, "top_p must be in (0, 1]"
        assert self.top_k >= -1, "top_k must be >= -1 (-1 means no limit)"
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        elif self.stop is not None:
            self.stop = list(self.stop)
        if self.stop is not None:
            assert all(isinstance(s, str) and s for s in self.stop), \
                "stop 的元素必须是非空字符串"


def find_stop_match(text: str, stop: list[str]) -> str | None:
    """返回 stop 中第一个使 text 以其结尾的字符串；无命中返回 None。

    OpenAI 停止串语义：只检查生成文本尾部（endswith），命中的停止串本身
    保留在输出文本中，不做截断。"""
    for s in stop:
        if text.endswith(s):
            return s
    return None
