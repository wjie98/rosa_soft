"""Frozen minimal dense estimator for unlimited hard ROSA.

This module is intentionally independent of the production implementation.
It fixes one deterministic research baseline: exact unlimited hard routing in
the forward pass and the full-horizon dense discovery surrogate in backward.
Do not add optimizations or estimator variants here; create a new version.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable


DEFAULT_SCALE = 1.0
DEFAULT_MISMATCH_SCALE = 3.0
NULL_ROUTE_SCORE = 0.5
_SQRT_UTILITY_SCALE = math.sqrt(2.0) + 1.0


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype in (torch.float32, torch.float64):
        return dtype
    raise ValueError("inputs must use float16, bfloat16, float32, or float64")


def _validate(query: Tensor, key: Tensor, value: Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (B, T, H, D)")
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, and value must share B and T")
    if query.shape != key.shape:
        raise ValueError("query and key must have identical shapes")
    if query.size(0) < 1 or query.size(1) < 1:
        raise ValueError("batch and sequence dimensions must be nonempty")
    if query.size(2) < 1 or value.size(2) < 1:
        raise ValueError("query and value must have at least one head")
    if not 1 <= query.size(-1) <= 32 or value.size(-1) < 1:
        raise ValueError("Q/K width must be in [1, 32] and V width nonempty")
    if query.size(2) % value.size(2) != 0:
        raise ValueError("query heads must be divisible by value heads")
    if query.device != key.device or query.device != value.device:
        raise ValueError("all inputs must share a device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("all inputs must share a dtype")
    _compute_dtype(query.dtype)


def _hard_sign(x: Tensor) -> Tensor:
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))


class _HardSignSoftsignVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor) -> Tensor:
        ctx.save_for_backward(logits)
        return _hard_sign(logits)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (logits,) = ctx.saved_tensors
        return grad_output / (1.0 + logits.abs()).square()


def _soft_vjp_sign(x: Tensor) -> Tensor:
    return _HardSignSoftsignVjp.apply(x)


def _causal_mask(length: int, device: torch.device) -> Tensor:
    query = torch.arange(length, device=device).view(length, 1)
    route = torch.arange(length, device=device).view(1, length)
    return (route == 0) | ((route > 0) & (route <= query))


def _suffix_dp(local_match: Tensor) -> Tensor:
    """Compute S[t,a] = g[t,a] * (1 + S[t-1,a-1])."""

    length = local_match.size(-1)
    previous = local_match.new_zeros(*local_match.shape[:-2], length)
    rows = []
    for row in range(length):
        if row == 0:
            current = previous
        else:
            active = local_match[..., row, 1 : row + 1] * (
                1.0 + previous[..., :row]
            )
            current = F.pad(active, (1, length - row - 1))
        rows.append(current)
        previous = current
    return torch.stack(rows, dim=-2)


def _pairwise_match(query: Tensor, key: Tensor, *, soft: bool) -> Tensor:
    length = query.size(1)
    mask = _causal_mask(length, query.device)
    sign = _soft_vjp_sign if soft else _hard_sign
    q = sign(query.permute(0, 2, 1, 3))
    k = sign(key.permute(0, 2, 1, 3)[..., :-1, :])
    if soft:
        mismatch_rate = 0.5 * (
            1.0 - q.unsqueeze(-2) * k.unsqueeze(-3)
        ).mean(dim=-1)
        local = torch.exp(-DEFAULT_MISMATCH_SCALE * mismatch_rate)
        local = F.pad(local, (1, 0), value=0.0)
    else:
        local = (q.unsqueeze(-2) == k.unsqueeze(-3)).all(dim=-1)
        local = F.pad(local, (1, 0), value=False)
    return local * mask.view(1, 1, length, length)


def _expand_value_heads(value: Tensor, query_heads: int) -> Tensor:
    groups = query_heads // value.size(2)
    return value.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)


def _hard_forward(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    suffix = _suffix_dp(_pairwise_match(query, key, soft=False).to(query.dtype))
    length = query.size(1)
    mask = _causal_mask(length, query.device)
    route = torch.arange(length, device=query.device).view(1, 1, 1, length)
    nonnull = mask.view(1, 1, length, length) & (route > 0)
    longest = suffix.amax(dim=-1, keepdim=True)
    latest = torch.where(nonnull & (suffix == longest), route, 0).amax(dim=-1)
    selected = torch.where(longest.squeeze(-1) > 0, latest, 0)

    route_values = _expand_value_heads(_hard_sign(value), query.size(2))
    route_values = route_values.clone()
    route_values[..., 0, :] = 0.0
    gather = selected.unsqueeze(-1).expand(*selected.shape, value.size(-1))
    return torch.gather(route_values, 2, gather).permute(0, 2, 1, 3)


def _dense_carrier(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    suffix = _suffix_dp(_pairwise_match(query, key, soft=True))
    utility = _SQRT_UTILITY_SCALE * (torch.sqrt(1.0 + suffix) - 1.0)
    length = query.size(1)
    mask = _causal_mask(length, query.device)
    route = torch.arange(length, device=query.device).view(1, 1, 1, length)
    nonnull = mask.view(1, 1, length, length) & (route > 0)
    count = nonnull.sum(dim=-1, keepdim=True).clamp_min(1)

    scores = utility.clone()
    scores[..., 0] = NULL_ROUTE_SCORE
    logits = DEFAULT_SCALE * scores - torch.where(
        nonnull,
        count.to(scores.dtype).log(),
        torch.zeros((), dtype=scores.dtype, device=scores.device),
    )
    logits = logits.masked_fill(~mask.view(1, 1, length, length), -torch.inf)
    probabilities = torch.softmax(
        logits - logits.amax(dim=-1, keepdim=True), -1
    )

    route_values = _expand_value_heads(_soft_vjp_sign(value), query.size(2))
    route_values = torch.where(
        (route > 0).transpose(-2, -1),
        route_values,
        torch.zeros_like(route_values),
    )
    return torch.einsum(
        "bhta,bhad->bhtd", probabilities, route_values
    ).permute(0, 2, 1, 3)


class _ExactForwardDenseBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        dtype = _compute_dtype(query.dtype)
        ctx.save_for_backward(query, key, value)
        return _hard_forward(query.to(dtype), key.to(dtype), value.to(dtype)).to(
            query.dtype
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        saved = ctx.saved_tensors
        needs = ctx.needs_input_grad
        with torch.enable_grad():
            leaves = tuple(
                x.detach().requires_grad_(need)
                for x, need in zip(saved, needs)
            )
            dtype = _compute_dtype(saved[0].dtype)
            carrier = _dense_carrier(*(x.to(dtype) for x in leaves)).to(
                saved[0].dtype
            )
            indices = [index for index, need in enumerate(needs) if need]
            gradients = torch.autograd.grad(
                carrier,
                tuple(leaves[index] for index in indices),
                grad_output,
                allow_unused=True,
            )
        result = [None, None, None]
        for index, gradient in zip(indices, gradients):
            result[index] = (
                torch.zeros_like(leaves[index])
                if gradient is None
                else gradient
            )
        return tuple(result)


def rosa_soft_dense_unbounded_v1(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Tensor:
    """Run the frozen deterministic dense-unbounded RosaSoft baseline."""

    _validate(query, key, value)
    dtype = _compute_dtype(query.dtype)
    if not torch.is_grad_enabled() or not any(
        tensor.requires_grad for tensor in (query, key, value)
    ):
        return _hard_forward(
            query.to(dtype), key.to(dtype), value.to(dtype)
        ).to(query.dtype)
    return _ExactForwardDenseBackward.apply(query, key, value)


__all__ = ["rosa_soft_dense_unbounded_v1"]
