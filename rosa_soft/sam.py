"""Compact CPU suffix automaton for exact ROSA inference and validation."""

from __future__ import annotations

import operator

import torch
from torch import Tensor

from . import _C  # noqa: F401

__all__ = ["RosaSam", "rosa_hard"]


def _positive(name: str, x: int) -> int:
    if isinstance(x, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        x = operator.index(x)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if x <= 0:
        raise ValueError(f"{name} must be positive")
    return x


def _pack(x: Tensor) -> Tensor:
    if not x.is_floating_point() or x.ndim < 1:
        raise TypeError("q and k must be floating-point tensors")
    d = x.size(-1)
    if not 1 <= d <= 32:
        raise ValueError("q/k symbol width must be in [1,32]")
    bit = (1 << torch.arange(d, device=x.device, dtype=torch.int64)).view(
        *([1] * (x.ndim - 1)), d
    )
    return ((x > 0) * bit).sum(-1).to(torch.int32)


def _offsets(cu: Tensor | None, n: int) -> Tensor:
    if cu is None:
        return torch.tensor([0, n], dtype=torch.int64)
    if cu.ndim != 1 or cu.numel() < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")
    if cu.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must be int32 or int64")
    cu = cu.detach().cpu().to(torch.int64).contiguous()
    if cu[0].item() != 0 or cu[-1].item() != n:
        raise ValueError("cu_seqlens must span all packed tokens")
    if bool((cu[1:] < cu[:-1]).any()):
        raise ValueError("cu_seqlens must be nondecreasing")
    return cu


class RosaSam:
    """Stateful exact ROSA routes. One instance must not be updated concurrently."""

    def __init__(self, num_heads: int, symbol_bits: int) -> None:
        self.num_heads = _positive("num_heads", num_heads)
        self.symbol_bits = _positive("symbol_bits", symbol_bits)
        if self.symbol_bits > 32:
            raise ValueError("symbol_bits must be in [1,32]")
        self._sam = torch.classes.rosa_soft.RosaSam(
            self.num_heads, self.symbol_bits
        )

    def reset(self) -> None:
        self._sam.reset()

    def update_packed(
        self,
        q: Tensor,
        k: Tensor,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        if q.shape != k.shape or q.ndim not in (2, 3):
            raise ValueError("packed q/k must have shape [N,H] or [B,T,H]")
        if q.dtype != torch.int32 or k.dtype != torch.int32:
            raise TypeError("packed q/k must be int32")
        if q.device != k.device or q.size(-1) != self.num_heads:
            raise ValueError("packed q/k device or head count mismatch")
        device = q.device
        shape = q.shape
        if q.ndim == 3:
            if cu_seqlens is not None:
                raise ValueError("dense input must not provide cu_seqlens")
            b, t, _ = shape
            if t:
                cu = torch.arange(0, (b + 1) * t, t, dtype=torch.int64)
            else:
                cu = torch.zeros(b + 1, dtype=torch.int64)
            q = q.reshape(b * t, self.num_heads)
            k = k.reshape(b * t, self.num_heads)
        else:
            cu = _offsets(cu_seqlens, q.size(0))
        y = self._sam.update(
            cu,
            q.detach().cpu().contiguous(),
            k.detach().cpu().contiguous(),
        )
        return y.reshape(shape).to(device)

    def update(
        self,
        q: Tensor,
        k: Tensor,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        if q.shape != k.shape or q.ndim not in (3, 4):
            raise ValueError("q/k must have shape [N,H,D] or [B,T,H,D]")
        if q.device != k.device or q.size(-1) != self.symbol_bits:
            raise ValueError("q/k device or symbol width mismatch")
        return self.update_packed(_pack(q), _pack(k), cu_seqlens)


def rosa_hard(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return binary hard output and the matched key end for dense or packed input."""

    if q.shape != k.shape or q.ndim not in (3, 4) or v.ndim != q.ndim:
        raise ValueError("q/k/v must use dense [B,T,H,D] or packed [N,H,D]")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must share a device")
    packed = q.ndim == 3
    if packed:
        if q.size(0) != v.size(0):
            raise ValueError("q/k/v token counts must match")
        h, hv = q.size(1), v.size(1)
    else:
        if q.shape[:2] != v.shape[:2] or cu_seqlens is not None:
            raise ValueError("dense q/k/v shape or cu_seqlens mismatch")
        h, hv = q.size(2), v.size(2)
    if not v.is_floating_point() or hv < 1 or h % hv:
        raise ValueError("v must be floating point and H divisible by value heads")

    route = RosaSam(h, q.size(-1)).update(q, k, cu_seqlens)
    bits = torch.where(v > 0, 1, -1).to(v.dtype)
    if packed:
        cu = _offsets(cu_seqlens, q.size(0))
        lengths = cu[1:] - cu[:-1]
        start = torch.repeat_interleave(cu[:-1], lengths).to(q.device)
        index = start[:, None] + route + 1
        bits = bits.repeat_interleave(h // hv, 1)
        heads = torch.arange(h, device=q.device)[None]
        y = bits[index, heads]
    else:
        index = (route + 1).permute(0, 2, 1)[..., None]
        index = index.expand(-1, -1, -1, v.size(-1))
        bits = bits.repeat_interleave(h // hv, 2).permute(0, 2, 1, 3)
        y = torch.gather(bits, 2, index).permute(0, 2, 1, 3)
    return y.masked_fill(route[..., None] < 0, 0), route
