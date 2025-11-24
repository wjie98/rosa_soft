import torch
from torch import Tensor

__all__ = [
    "rosa_sam_k8v8_init",
    "rosa_sam_k8v8_free",
    "rosa_sam_k8v8_update",

    "rosa_sam_k8v64_init",
    "rosa_sam_k8v64_free",
    "rosa_sam_k8v64_update",

    "rosa_sam_k64v64_init",
    "rosa_sam_k64v64_free",
    "rosa_sam_k64v64_update",
]

# ================ QKV(8bits, 8bits, 8bits) ================

def rosa_sam_k8v8_init(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k8v8_init(ctx)

def rosa_sam_k8v8_free(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k8v8_free(ctx)

def rosa_sam_k8v8_update(ctx: Tensor, q: Tensor, k: Tensor, v: Tensor, u: int) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k8v8_update(ctx, q, k, v, u)

@torch.library.register_fake("rosa_cpp::rosa_sam_k8v8_init")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_cpp::rosa_sam_k8v8_free")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_cpp::rosa_sam_k8v8_update")
def _(ctx: Tensor, q: Tensor, k: Tensor, v: Tensor, u: int):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    
    torch._check(q.dim() == 2)
    torch._check(q.dtype == torch.uint8)

    torch._check(k.dim() == 2)
    torch._check(k.dtype == torch.uint8)

    torch._check(v.dim() == 2)
    torch._check(v.dtype == torch.uint8)
    return torch.empty_like(q, dtype=v.dtype)

# ================ QKV(8bits, 8bits, 64bits) ================

def rosa_sam_k8v64_init(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k8v64_init(ctx)

def rosa_sam_k8v64_free(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k8v64_free(ctx)

def rosa_sam_k8v64_update(ctx: Tensor, q: Tensor, k: Tensor, v: Tensor, u: int) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k8v64_update(ctx, q, k, v, u)

@torch.library.register_fake("rosa_cpp::rosa_sam_k8v64_init")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_cpp::rosa_sam_k8v64_free")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_cpp::rosa_sam_k8v64_update")
def _(ctx: Tensor, q: Tensor, k: Tensor, v: Tensor, u: int):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    
    torch._check(q.dim() == 2)
    torch._check(q.dtype == torch.uint8)

    torch._check(k.dim() == 2)
    torch._check(k.dtype == torch.uint8)

    torch._check(v.dim() == 2)
    torch._check(v.dtype == torch.long)
    
    return torch.empty_like(q, dtype=v.dtype)

# ================ QKV(64bits, 64bits, 64bits) ================

def rosa_sam_k64v64_init(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k64v64_init(ctx)

def rosa_sam_k64v64_free(ctx: Tensor) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k64v64_free(ctx)

def rosa_sam_k64v64_update(ctx: Tensor, q: Tensor, k: Tensor, v: Tensor, u: int) -> Tensor:
    return torch.ops.rosa_cpp.rosa_sam_k64v64_update(ctx, q, k, v, u)

@torch.library.register_fake("rosa_cpp::rosa_sam_k64v64_init")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_cpp::rosa_sam_k64v64_free")
def _(ctx: Tensor):
    torch._check(ctx.dim() == 1)
    torch._check(ctx.dtype == torch.long)
    return torch.empty_like(ctx)

@torch.library.register_fake("rosa_cpp::rosa_sam_k64v64_update")
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

