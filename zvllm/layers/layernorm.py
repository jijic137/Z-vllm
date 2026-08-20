import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        # copy=True：输入已是 fp32 时 .float() 恒等返回原张量，后续 mul_ 会原地改掉调用方
        # 的张量（首层 residual 会错拿归一化后的值）；bf16 路径分配不变
        x = x.to(torch.float32, copy=True)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 同上：两个输入都拷贝到 fp32 再相加，避免 fp32 下原地修改调用方张量
        x = x.to(torch.float32, copy=True)
        x.add_(residual.to(torch.float32, copy=True))
        # copy=True：fp32 下 to() 恒等返回原张量，随后的 x.mul_ 会把这个"残差"也归一化掉
        residual = x.to(orig_dtype, copy=True)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
