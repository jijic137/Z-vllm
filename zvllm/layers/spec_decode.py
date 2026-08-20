"""投机解码的接受-拒绝（CPU 可单测；GPU 路径由 ModelRunner 委托调用）。

对确定性草稿（n-gram / PLD，点质量分布 q）的标准接受-拒绝：
- 以概率 p(x) 接受草稿 token x（p = 与采样器同一套截断/温度下的目标分布）；
- 首次拒绝时从残差分布 "p 去掉 x 质量后重归一" 采样 bonus token 并停止。
该构造使每步输出分布严格等于目标分布（Leviathan et al. 2023 的确定性草稿特例）。
贪心（temperature=0）退化为 argmax 匹配，输出与非投机贪心严格一致——
这是真机 A/B 一致性测试的基准。
"""
import torch


def accept_drafts(sampler, seq, rows: torch.Tensor) -> list[int]:
    """单序列投机验证。rows = γ+1 行目标 logit（位置 L-1+i，i=0..γ）。

    返回待追加 token 列表 = 接受的草稿前缀 + 1 个 bonus token。
    带 seed 的序列走序列专属 generator（抽取顺序：u_0..u_{γ-1} 后 bonus 噪声），
    同 seed 可跨运行复现。逐序列小 kernel 循环（γ≤8）：nano 规模下同步开销
    可忽略，高并发实测有瓶颈时再向量化。"""
    g = len(seq.draft_tokens)
    device = rows.device
    if seq.temperature <= 1e-10:
        amax = rows.argmax(dim=-1)
        out = []
        for i in range(g):
            a = int(amax[i].item())
            x = seq.draft_tokens[i]
            if x == a:
                out.append(x)
            else:
                out.append(a)
                return out
        out.append(int(amax[g].item()))
        return out
    probs = sampler.compute_probs(
        rows,
        torch.full((g + 1,), seq.temperature, dtype=torch.float32, device=device),
        torch.full((g + 1,), seq.top_k, dtype=torch.int64, device=device),
        torch.full((g + 1,), seq.top_p, dtype=torch.float32, device=device))
    gen = sampler._generator(seq.seq_id, seq.seed) if seq.seed is not None else None

    def draw(shape) -> torch.Tensor:
        t = torch.empty(*shape, device=device)
        return t.exponential_(1, generator=gen) if gen is not None else t.exponential_(1)

    def sample_from(p: torch.Tensor, exclude: int | None) -> int:
        pr = p if exclude is None else p.clone()
        if exclude is not None:
            pr[exclude] = 0.0
            pr = pr / pr.sum()
        return int((pr.div(draw(pr.size()).clamp_min(1e-10))).argmax())

    out = []
    for i in range(g):
        p = probs[i]
        x = seq.draft_tokens[i]
        u = float(torch.rand(1, device=device, generator=gen).item())
        if u < float(p[x].item()):
            out.append(x)
            continue
        out.append(sample_from(p, exclude=x))
        return out
    out.append(sample_from(probs[g], exclude=None))
    return out
