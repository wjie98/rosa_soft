import torch
from torch import Tensor
from typing import Tuple

__all__ = [
    "rosa_sam_init",
    "rosa_sam_free",
    "rosa_sam_update",
]


def rosa_sam_init(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_soft.rosa_sam_init(ctx)

def rosa_sam_free(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_soft.rosa_sam_free(ctx)

def rosa_sam_update(ctx: Tensor, xq: Tensor, xk: Tensor, xv: Tensor, mq: Tensor, mk: Tensor, u: int) -> Tuple[Tensor, Tensor, Tensor]:
    return torch.ops.rosa_soft.rosa_sam_update(ctx, xq, xk, xv, mq, mk, u)

@torch.library.register_fake("rosa_soft::rosa_sam_init")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_soft::rosa_sam_free")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_soft::rosa_sam_update")
def _(ctx: Tensor, xq: Tensor, xk: Tensor, xv: Tensor, mq: Tensor, mk: Tensor, u: int):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)

    torch._check(xq.dim() == 2)
    torch._check(xq.dtype == torch.long)

    torch._check(xk.dim() == 2)
    torch._check(xk.dtype == torch.long)

    torch._check(xv.dim() == 2)
    torch._check(xv.dtype == torch.long)

    torch._check(mq.dim() == 2)
    torch._check(mq.dtype == torch.long)

    torch._check(mk.dim() == 2)
    torch._check(mk.dtype == torch.long)
    
    out = torch.empty_like(xq, dtype=xv.dtype)
    pos = torch.empty_like(xq, dtype=torch.long)
    len = torch.empty_like(xq, dtype=torch.long)
    return out, pos, len
