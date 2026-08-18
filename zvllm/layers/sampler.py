import torch
from torch import nn


class Sampler(nn.Module):
    """采样：贪心（temperature=0）/ top-k / top-p / seed 复现。

    随机流按序列独立（generator 以 seq_id 为键），同一 seed 的序列可跨运行复现。
    """

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
        sampled = logits[sample_mask].float()
        temps = temperatures[sample_mask]
        n, vocab = sampled.size(0), sampled.size(1)

        # top-k：只保留概率最高的 k 个 token（k<=0 不限制）
        need_k = top_k[sample_mask] > 0
        if need_k.any():
            k = top_k[sample_mask][need_k].clamp(min=1, max=vocab)
            k_max = int(k.max().item())
            topk_vals = sampled[need_k].topk(k_max, dim=-1).values
            threshold = topk_vals[torch.arange(len(k), device=device), k - 1]
            sampled[need_k] = torch.where(sampled[need_k] < threshold.unsqueeze(1), float("-inf"), sampled[need_k])

        # top-p（nucleus）：按概率降序保留累积概率首次达到 p 的最小前缀集
        need_p = top_p[sample_mask] < 1.0
        if need_p.any():
            rows = sampled[need_p]
            sorted_idx = rows.argsort(dim=-1, descending=True)
            sorted_probs = torch.softmax(torch.gather(rows, 1, sorted_idx), dim=-1)
            keep = sorted_probs.cumsum(dim=-1) - sorted_probs <= top_p[sample_mask][need_p].unsqueeze(1)
            keep_orig = torch.zeros_like(rows, dtype=torch.bool)
            keep_orig.scatter_(1, sorted_idx, keep)
            sampled[need_p] = torch.where(keep_orig, sampled[need_p], float("-inf"))

        # 温度缩放 + softmax
        probs = torch.softmax(sampled / temps.unsqueeze(1), dim=-1)

        # Gumbel-max：指数噪声按序列独立；带 seed 的序列使用专属 generator
        noise = torch.empty(n, vocab, device=device)
        idx = sample_mask.nonzero(as_tuple=True)[0].tolist()
        s_ids = [seq_ids[i] for i in idx]
        s_seeds = [seeds[i] for i in idx]
        seeded = [i for i, s in enumerate(s_seeds) if s is not None]
        if len(seeded) < n:
            unseeded = torch.tensor([i for i in range(n) if s_seeds[i] is None], device=device)
            noise[unseeded] = torch.empty(len(unseeded), vocab, device=device).exponential_(1)
        for i in seeded:
            noise[i] = torch.empty(vocab, device=device).exponential_(1, generator=self._generator(s_ids[i], s_seeds[i]))

        token_ids[sample_mask] = probs.div_(noise.clamp_min_(1e-10)).argmax(dim=-1)
        return token_ids
