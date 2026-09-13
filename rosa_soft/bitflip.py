"""Exact hard ROSA with explicit independent or joint bit differences."""

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


@torch.library.register_fake("rosa_soft::joint_forward")
def _fake_joint_forward(x, v, rows, all):
    torch._check(x.ndim == 4)
    torch._check(x.size(1) < (1 << 20))
    torch._check(x.size(0) * x.size(2) <= 65535)
    if all:
        torch._check(x.shape == v.shape)
    return _fake_forward(x, x, v, rows)


@torch.library.register_fake("rosa_soft::joint_backward")
def _fake_joint_backward(x, v, dy, q, route, rows, all, mask):
    return tuple(z.new_empty(z.shape if mask & (1 << i) else (0,))
                 for i, z in enumerate((x, v)))


def _setup_joint(ctx, inputs, output):
    x, v, rows, all = inputs
    ctx.rows, ctx.all = rows, all
    ctx.mask = int(ctx.needs_input_grad[0]) | (int(ctx.needs_input_grad[1] and not all) << 1)
    _, q, _, route = output
    ctx.save_for_backward(x, v, q, route)
    ctx.set_materialize_grads(False)


@once_differentiable
def _joint_backward(ctx, dy, *ignored):
    if dy is None or not ctx.mask:
        return None, None, None, None
    x, v, q, route = ctx.saved_tensors
    gx, gv = torch.ops.rosa_soft.joint_backward(x, v, dy, q, route, ctx.rows, ctx.all, ctx.mask)
    return gx if ctx.mask & 1 else None, gv if ctx.mask & 2 else None, None, None


torch.library.register_autograd("rosa_soft::joint_forward", _joint_backward, setup_context=_setup_joint)


def rosa_bitflip(q: Tensor, k: Tensor, v: Tensor, *, rows: int = 256,
                 tied: str | None = None) -> Tensor:
    """Run unlimited hard ROSA with exact bit-edit output differences.

    Q/K are dense [B,T,H,D], D<=32; V is [B,T,Hv,Dv], H%Hv==0.
    Inputs must be finite CUDA tensors with matching FP16/BF16/FP32 dtype.
    Empty T is supported. Packed documents are not supported.

    rows bounds workspace, not suffix length. Q/K gradients contract each
    independent bit's hard output change with fixed dY, then apply the
    softsign factor. With independent edits or tied QK, V gradients follow
    only the hard route; tied QKV includes payload changes in its joint edit.

    tied="qk" edits the same bit in Q and K simultaneously; Q and K must be
    the same Tensor object. tied="qkv" additionally edits its V payload and
    requires all three arguments to be the same Tensor. Joint modes require
    T<2**20 and B*H<=65535. No binding is inferred from aliases by default.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("rosa_bitflip requires dense [B,T,H,D] inputs")
    if type(rows) is not int or not 1 <= rows <= 256:
        raise ValueError("rows must be an integer in [1,256]")
    if tied is not None:
        if tied not in ("qk", "qkv"):
            raise ValueError("tied must be None, 'qk', or 'qkv'")
        if q is not k or (tied == "qkv" and q is not v):
            raise ValueError("tied inputs must be the same Tensor object")
        return torch.ops.rosa_soft.joint_forward(q, v, rows, tied == "qkv")[0]
    return torch.ops.rosa_soft.bitflip_forward(q, k, v, rows)[0]
