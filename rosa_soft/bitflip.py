"""Exact hard ROSA with independent activation-bit output differences."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

__all__ = ["rosa_bitflip"]


@torch.library.register_fake("rosa_soft::bitflip_forward")
def _fake_forward(q: Tensor, k: Tensor, v: Tensor, rows: int):
    torch._check(q.ndim == k.ndim == v.ndim == 4)
    for a, b in zip(q.shape, k.shape):
        torch._check(a == b)
    b, t, h, d = q.shape
    torch._check(b > 0)
    torch._check(h > 0)
    torch._check(1 <= d <= 32)
    torch._check(v.size(0) == b)
    torch._check(v.size(1) == t)
    torch._check(v.size(2) > 0)
    torch._check(v.size(3) > 0)
    torch._check(h % v.size(2) == 0)
    torch._check(q.dtype == k.dtype == v.dtype)
    torch._check(q.device == k.device == v.device)
    torch._check(q.dtype in (torch.float16, torch.bfloat16, torch.float32))
    torch._check(1 <= rows <= 256)
    y = q.new_empty((b, t, h, v.size(3)))
    pq = q.new_empty((b * h, t), dtype=torch.int32)
    pk = torch.empty_like(pq)
    route = q.new_empty((b * h, t), dtype=torch.int64)
    return y, pq, pk, route


@torch.library.register_fake("rosa_soft::bitflip_backward")
def _fake_backward(q, k, v, dy, pq, pk, route, d, rows, mask):
    return tuple(
        x.new_empty(x.shape if mask & (1 << i) else (0,))
        for i, x in enumerate((q, k, v))
    )


def _setup_context(ctx, inputs, output):
    q, k, v, rows = inputs
    mask = sum(int(need) << i for i, need in enumerate(ctx.needs_input_grad[:3]))
    ctx.mask, ctx.d, ctx.rows = mask, q.size(3), rows
    _, pq, pk, route = output
    ctx.save_for_backward(
        q if mask & 1 else q.new_empty(0),
        k if mask & 2 else k.new_empty(0),
        v,
        pq if mask & 3 else pq.new_empty(0),
        pk if mask & 3 else pk.new_empty(0),
        route,
    )
    ctx.set_materialize_grads(False)


@once_differentiable
def _backward(ctx, dy, *ignored):
    if dy is None:
        return None, None, None, None
    q, k, v, pq, pk, route = ctx.saved_tensors
    grads = torch.ops.rosa_soft.bitflip_backward(
        q, k, v, dy, pq, pk, route, ctx.d, ctx.rows, ctx.mask
    )
    return (*(g if ctx.mask & (1 << i) else None for i, g in enumerate(grads)), None)


torch.library.register_autograd(
    "rosa_soft::bitflip_forward", _backward, setup_context=_setup_context
)


def rosa_bitflip(q: Tensor, k: Tensor, v: Tensor, *, rows: int = 256) -> Tensor:
    """Run unlimited hard ROSA with an independent-bit backward estimator.

    Q/K are dense [B,T,H,D], D<=32; V is [B,T,Hv,Dv], H%Hv==0.
    Inputs must be finite CUDA tensors with matching FP16/BF16/FP32 dtype.
    Empty T is supported. Packed documents are not supported.

    rows bounds workspace, not suffix length. Q/K gradients contract each
    independent bit's hard output change with fixed dY, then apply the
    softsign factor. V gradients follow only the hard route.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("rosa_bitflip requires dense [B,T,H,D] inputs")
    if type(rows) is not int or not 1 <= rows <= 256:
        raise ValueError("rows must be an integer in [1,256]")
    return torch.ops.rosa_soft.bitflip_forward(q, k, v, rows)[0]
