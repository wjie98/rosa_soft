import torch
from torch import Tensor
from typing import Tuple

__all__ = [
    "rosa_sam_init",
    "rosa_sam_free",
    "rosa_sam_update",
    "rosa_sam_inspect",

    "rosa_sam_forward",
    "rosa_gss_forward",
]


def rosa_sam_init(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_init(ctx)

def rosa_sam_free(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_free(ctx)

def rosa_sam_update(ctx: Tensor, q: Tensor, k: Tensor, v: Tensor, u: int) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_update(ctx, q, k, v, u)

def rosa_sam_inspect(ctx: Tensor, q: Tensor, k: Tensor, v: Tensor, u: int) -> Tuple[Tensor, Tensor, Tensor]:
    return torch.ops.rosa_cpp.rosa_sam_inspect(ctx, q, k, v, u)

@torch.library.register_fake("rosa_cpp::rosa_sam_init")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_cpp::rosa_sam_free")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_cpp::rosa_sam_update")
def _(ctx: Tensor, q: Tensor, k: Tensor, v: Tensor, u: int):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    
    torch._check(q.dim() == 2)
    torch._check(q.dtype == torch.long)

    torch._check(k.dim() == 2)
    torch._check(k.dtype == torch.long)

    torch._check(v.dim() == 2)
    torch._check(v.dtype == torch.long)
    
    return torch.empty_like(q, dtype=v.dtype)

@torch.library.register_fake("rosa_cpp::rosa_sam_inspect")
def _(q: Tensor, k: Tensor, v: Tensor, u: int):
    torch._check(q.dim() == 2)
    torch._check(q.dtype == torch.long)

    torch._check(k.dim() == 2)
    torch._check(k.dtype == torch.long)

    torch._check(v.dim() == 2)
    torch._check(v.dtype == torch.long)
    
    out = torch.empty_like(q, dtype=v.dtype)
    pos = torch.empty_like(q, dtype=torch.long)
    len = torch.empty_like(q, dtype=torch.long)
    return out, pos, len


def rosa_sam_forward(q: Tensor, k: Tensor, v: Tensor, u: int) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_forward(q, k, v, u)

@torch.library.register_fake("rosa_cpp::rosa_sam_forward")
def _(q: Tensor, k: Tensor, v: Tensor, u: int):
    torch._check(q.dim() == 2)
    torch._check(q.dtype == torch.long)

    torch._check(k.dim() == 2)
    torch._check(k.dtype == torch.long)

    torch._check(v.dim() == 2)
    torch._check(v.dtype == torch.long)
    
    return torch.empty_like(q, dtype=v.dtype)


def rosa_gss_forward(q: Tensor, k: Tensor, v: Tensor, u: int, num_samples: int, tau: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    out, indptr, indices, values = torch.ops.rosa_cpp.rosa_gss_forward(q, k, v, u, num_samples, tau)
    return out, indptr, indices, values

@torch.library.register_fake("rosa_cpp::rosa_gss_forward")
def _(q: Tensor, k: Tensor, v: Tensor, u: int, num_samples: int, tau: float):
    torch._check(q.dim() == 2)
    torch._check(q.dtype == torch.long)

    torch._check(k.dim() == 2)
    torch._check(k.dtype == torch.long)

    torch._check(v.dim() == 2)
    torch._check(v.dtype == torch.long)

    out = torch.empty_like(q, dtype=v.dtype)

    indptr = torch.empty((q.numel() + 1,), dtype=torch.long, device=q.device)

    ctx = torch.library.get_ctx()
    nnz = ctx.new_dynamic_size()

    indices = torch.empty((nnz,), dtype=torch.long, device=q.device)
    quality = torch.empty((nnz,), dtype=torch.float, device=q.device)

    return out, indptr, indices, quality
