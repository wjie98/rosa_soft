"""Exact hard ROSA forward with a dense soft CUDA VJP."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

__all__ = ["rosa_soft"]


@torch.library.register_fake("rosa_soft::forward")
def _fake_forward(q: Tensor, k: Tensor, v: Tensor, cu: Tensor):
    del k, cu
    if q.ndim == 4:
        b, t, h, _ = q.shape
        y = q.new_empty((b, t, h, v.size(3)))
        shape = (b, h, t)
    else:
        t, h, _ = q.shape
        y = q.new_empty((t, h, v.size(2)))
        shape = (h, t)
    return y, q.new_empty(shape, dtype=torch.int32), q.new_empty(
        shape, dtype=torch.int32
    )


@torch.library.register_fake("rosa_soft::backward")
def _fake_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dy: Tensor,
    pq: Tensor,
    pk: Tensor,
    seed: Tensor,
    cu: Tensor,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    mask: int,
):
    del dy, pq, pk, seed, cu, scale, dropout_p, mismatch_scale
    return tuple(
        x.new_empty(x.shape, dtype=torch.float32)
        if mask & bit
        else x.new_empty((0,), dtype=torch.float32)
        for x, bit in zip((q, k, v), (1, 2, 4))
    )


def _check(scale: float, dropout_p: float, mismatch_scale: float) -> None:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and > 0")
    if (
        not math.isfinite(dropout_p)
        or not 0 <= dropout_p <= 1 - 2**-24
    ):
        raise ValueError("dropout_p must be in [0, 1 - 2^-24]")
    if (
        not math.isfinite(mismatch_scale)
        or mismatch_scale <= 0
    ):
        raise ValueError("mismatch_scale must be finite and > 0")


class _RosaSoft(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu: Tensor,
        seed: Tensor,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
    ) -> Tensor:
        y, pq, pk = torch.ops.rosa_soft.forward(q, k, v, cu)
        ctx.args = scale, dropout_p, mismatch_scale
        ctx.save_for_backward(q, k, v, pq, pk, seed, cu)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy: Tensor):
        q, k, v, pq, pk, seed, cu = ctx.saved_tensors
        need = ctx.needs_input_grad[:3]
        mask = int(need[0]) | (int(need[1]) << 1) | (int(need[2]) << 2)
        dq, dk, dv = torch.ops.rosa_soft.backward(
            q, k, v, dy, pq, pk, seed, cu, *ctx.args, mask
        )
        grad = tuple(
            g.to(x.dtype) if use else None
            for g, x, use in zip((dq, dk, dv), (q, k, v), need)
        )
        return *grad, None, None, None, None, None


def rosa_soft(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor | None = None,
    *,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    mismatch_scale: float = 3.0,
) -> Tensor:
    """Run unlimited hard ROSA with its unlimited dense surrogate VJP.

    Dense input is ``[B,T,H,D]``. Packed input is ``[N,H,D]`` and requires
    CUDA int32 ``cu_seqlens``. Dropout follows attention semantics and affects
    only the surrogate backward pass.
    """

    if any(isinstance(x, bool) for x in (scale, dropout_p, mismatch_scale)):
        raise ValueError("scale, dropout_p, and mismatch_scale must be numbers")
    scale, dropout_p, mismatch_scale = map(
        float, (scale, dropout_p, mismatch_scale)
    )
    _check(scale, dropout_p, mismatch_scale)
    if q.ndim == 3:
        if cu_seqlens is None:
            raise ValueError("packed input requires cu_seqlens")
        cu = cu_seqlens
    else:
        if cu_seqlens is not None:
            raise ValueError("dense input must not provide cu_seqlens")
        cu = torch.empty(0, dtype=torch.int32, device=q.device)

    backward = torch.is_grad_enabled() and any(
        x.requires_grad for x in (q, k, v)
    )
    if not backward:
        return torch.ops.rosa_soft.forward(q, k, v, cu)[0]
    seed = (
        torch.randint(
            torch.iinfo(torch.int64).max,
            (),
            dtype=torch.int64,
            device=q.device,
        )
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device=q.device)
    )
    return _RosaSoft.apply(
        q, k, v, cu, seed, scale, dropout_p, mismatch_scale
    )
