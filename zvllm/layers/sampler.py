import torch
from torch import nn


class Sampler(nn.Module):
    """采样：贪心（temperature=0）/ top-k / top-p / seed 复现。

    随机流按序列独立（generator 以 seq_id 为键），同一 seed 的序列可跨运行复现。
    compute_probs / gumbel_sample 拆分自原 forward（数学与 RNG 顺序不变），
    供投机解码的接受-拒绝复用同一套截断/温度/随机流语义。"""

    def __init__(self):
        super().__init__()
        self.generators: dict[int, torch.Generator] = {}

    def _generator(self, seq_id: int, seed: int) -> torch.Generator:
        gen = self.generators.get(seq_id)
        if gen is None:
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed)
            self.generators[seq_id] = gen
        return gen

    def compute_probs(self, logits: torch.Tensor, temperatures: torch.Tensor,
                      top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """截断（top-k/top-p）+ 温度缩放 + softmax -> 概率矩阵 [n, V]（不采样）。"""
        sampled = logits.float()
        n, vocab = sampled.size(0), sampled.size(1)

        # top-k：只保留概率最高的 k 个 token（k<=0 不限制）
        need_k = top_k > 0
        if need_k.any():
            k = top_k[need_k].clamp(min=1, max=vocab)
            k_max = int(k.max().item())
            topk_vals = sampled[need_k].topk(k_max, dim=-1).values
            threshold = topk_vals[torch.arange(len(k), device=sampled.device), k - 1]
            sampled[need_k] = torch.where(sampled[need_k] < threshold.unsqueeze(1), float("-inf"), sampled[need_k])

        # top-p（nucleus）：按概率降序保留累积概率首次达到 p 的最小前缀集
        need_p = top_p < 1.0
        if need_p.any():
            rows = sampled[need_p]
            sorted_idx = rows.argsort(dim=-1, descending=True)
            sorted_probs = torch.softmax(torch.gather(rows, 1, sorted_idx), dim=-1)
            keep = sorted_probs.cumsum(dim=-1) - sorted_probs <= top_p[need_p].unsqueeze(1)
            keep_orig = torch.zeros_like(rows, dtype=torch.bool)
            keep_orig.scatter_(1, sorted_idx, keep)
            sampled[need_p] = torch.where(keep_orig, sampled[need_p], float("-inf"))

        # 温度缩放 + softmax
        return torch.softmax(sampled / temperatures.unsqueeze(1), dim=-1)

    def gumbel_sample(self, probs: torch.Tensor, seq_ids: list[int], seeds: list[int | None]) -> torch.Tensor:
        """Gumbel-max 从每行概率采样一个 token（指数噪声按序列独立；与 forward 原实现一致）。"""
        n, vocab = probs.size(0), probs.size(1)
        device = probs.device
        noise = torch.empty(n, vocab, device=device)
        s_seeds = list(seeds)
        unseeded = [i for i in range(n) if s_seeds[i] is None]
        if unseeded:
            noise[torch.tensor(unseeded, device=device)] = torch.empty(len(unseeded), vocab, device=device).exponential_(1)
        for i in range(n):
            if s_seeds[i] is not None:
                noise[i] = torch.empty(vocab, device=device).exponential_(1, generator=self._generator(seq_ids[i], s_seeds[i]))
        return probs.div(noise.clamp_min(1e-10)).argmax(dim=-1)

    @torch.inference_mode()
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        seq_ids: list[int],
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        seeds: list[int | None],
    ) -> torch.Tensor:
        bs = logits.size(0)
        device = logits.device
        token_ids = torch.empty(bs, dtype=torch.int64, device=device)

        # 贪心：temperature=0 直接 argmax，无噪声
        greedy_mask = temperatures <= 1e-10
        if greedy_mask.any():
            token_ids[greedy_mask] = logits[greedy_mask].argmax(dim=-1)

        sample_mask = ~greedy_mask
        if not sample_mask.any():
            return token_ids
        idx = sample_mask.nonzero(as_tuple=True)[0].tolist()
        probs = self.compute_probs(logits[sample_mask], temperatures[sample_mask],
                                   top_k[sample_mask], top_p[sample_mask])
        token_ids[sample_mask] = self.gumbel_sample(probs, [seq_ids[i] for i in idx], [seeds[i] for i in idx])
        return token_ids
