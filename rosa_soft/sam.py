"""Small synchronous suffix-automaton interface for hard ROSA validation."""

from __future__ import annotations

import operator
from typing import Optional, Tuple

import torch
from torch import Tensor

from . import _C  # noqa: F401 - registers torch.classes.rosa_soft.RosaSam

__all__ = [
    "RosaSam",
    "rosa_hard_reference",
    "rosa_hard_varlen_reference",
]


def _as_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _pack_sign_bits(logits: Tensor) -> Tensor:
    if logits.ndim < 1:
        raise ValueError("query and key logits must have at least one dimension")
    if not logits.is_floating_point():
        raise TypeError("query and key logits must be floating-point")
    bits = logits.size(-1)
    if not 1 <= bits <= 32:
        raise ValueError(f"query/key width must be in [1, 32], got {bits}")
    weights = (
        1
        << torch.arange(bits, dtype=torch.int64, device=logits.device)
    ).view(*([1] * (logits.ndim - 1)), bits)
    return ((logits > 0).to(torch.int64) * weights).sum(dim=-1).to(
        torch.int32
    )


def _normalize_cu_seqlens(cu_seqlens: Tensor, total_tokens: int) -> Tensor:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(
            "cu_seqlens must be a 1D tensor with at least two entries"
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have dtype int32 or int64")
    result = cu_seqlens.detach().to(device="cpu").contiguous()
    if result[0].item() != 0 or result[-1].item() != total_tokens:
        raise ValueError("cu_seqlens must span exactly all packed tokens")
    if bool((result[1:] < result[:-1]).any()):
        raise ValueError("cu_seqlens must be nondecreasing")
    return result


def _flatten_packed_symbols(
    query: Tensor,
    key: Tensor,
    num_heads: int,
    cu_seqlens: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tuple[int, ...]]:
    if query.ndim not in (2, 3) or key.ndim not in (2, 3):
        raise ValueError(
            "packed query and key must have shape [B,T,H] or [N,H]"
        )
    if query.dtype != torch.int32 or key.dtype != torch.int32:
        raise TypeError("packed query and key must have dtype torch.int32")
    if query.shape != key.shape:
        raise ValueError("packed query and key must have the same shape")
    if query.device != key.device:
        raise ValueError("packed query and key must be on the same device")
    if query.size(-1) != num_heads:
        raise ValueError(
            f"expected {num_heads} query/key heads, got {query.size(-1)}"
        )

    if query.ndim == 3:
        if cu_seqlens is not None:
            raise ValueError("dense packed input must not provide cu_seqlens")
        batch, tokens, _ = query.shape
        offsets = torch.arange(
            0,
            (batch + 1) * tokens,
            tokens,
            dtype=torch.int64,
        ) if tokens else torch.zeros(batch + 1, dtype=torch.int64)
        output_shape = tuple(query.shape)
        query = query.reshape(batch * tokens, num_heads)
        key = key.reshape(batch * tokens, num_heads)
    elif query.ndim == 2:
        output_shape = tuple(query.shape)
        offsets = (
            torch.tensor([0, query.size(0)], dtype=torch.int64)
            if cu_seqlens is None
            else _normalize_cu_seqlens(cu_seqlens, query.size(0))
        )
    else:
        raise ValueError(
            "packed query and key must have shape [B,T,H] or [N,H]"
        )

    return (
        query.to(device="cpu").contiguous(),
        key.to(device="cpu").contiguous(),
        offsets,
        output_shape,
    )


class RosaSam:
    """Exact unlimited routes; do not update one instance concurrently."""

    def __init__(self, num_heads: int, symbol_bits: int) -> None:
        self._num_heads = _as_positive_integer("num_heads", num_heads)
        self._symbol_bits = _as_positive_integer("symbol_bits", symbol_bits)
        if self._symbol_bits > 32:
            raise ValueError("symbol_bits must be in [1, 32]")
        self._native = torch.classes.rosa_soft.RosaSam(
            self._num_heads,
            self._symbol_bits,
        )

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def symbol_bits(self) -> int:
        return self._symbol_bits

    def reset(self) -> None:
        self._native.reset()

    def update_packed(
        self,
        query: Tensor,
        key: Tensor,
        *,
        cu_seqlens: Optional[Tensor] = None,
    ) -> Tensor:
        device = query.device
        query_cpu, key_cpu, offsets, output_shape = _flatten_packed_symbols(
            query,
            key,
            self._num_heads,
            cu_seqlens,
        )
        matched_key_end = self._native.update_packed(
            offsets,
            query_cpu,
            key_cpu,
        ).reshape(output_shape)
        return matched_key_end.to(device=device)

    def update(
        self,
        query: Tensor,
        key: Tensor,
        *,
        cu_seqlens: Optional[Tensor] = None,
    ) -> Tensor:
        if query.ndim not in (3, 4) or key.ndim not in (3, 4):
            raise ValueError(
                "query and key must have shape [B,T,H,D] or [N,H,D]"
            )
        if query.shape != key.shape:
            raise ValueError("query and key must have the same shape")
        if query.device != key.device:
            raise ValueError("query and key must be on the same device")
        if query.size(-1) != self._symbol_bits:
            raise ValueError(
                f"expected query/key width {self._symbol_bits}, "
                f"got {query.size(-1)}"
            )
        return self.update_packed(
            _pack_sign_bits(query),
            _pack_sign_bits(key),
            cu_seqlens=cu_seqlens,
        )


def rosa_hard_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return exact hard output and matched key ends for dense Q/K/V input."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B,T,H,D]")
    if query.shape != key.shape:
        raise ValueError("query and key must have the same shape")
    if query.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, and value batch/token shapes must match")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if not value.is_floating_point():
        raise TypeError("value must be floating-point")
    query_heads = query.size(2)
    value_heads = value.size(2)
    if query_heads % value_heads != 0:
        raise ValueError("query heads must be divisible by value heads")

    sam = RosaSam(query_heads, query.size(-1))
    matched_key_end = sam.update(query, key)
    route_index = matched_key_end + 1
    hard_value = torch.where(value > 0, 1.0, -1.0).to(value.dtype)
    hard_value = hard_value.repeat_interleave(
        query_heads // value_heads,
        dim=2,
    ).permute(0, 2, 1, 3)
    gather_index = route_index.permute(0, 2, 1).unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        value.size(-1),
    )
    output = torch.gather(hard_value, 2, gather_index).permute(0, 2, 1, 3)
    output = output.masked_fill(matched_key_end.unsqueeze(-1) < 0, 0.0)
    return output, matched_key_end


def rosa_hard_varlen_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return exact hard output and local matched ends for packed Q/K/V."""

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must have shape [N,H,D]")
    if query.shape != key.shape:
        raise ValueError("query and key must have the same shape")
    if query.size(0) != value.size(0):
        raise ValueError("query, key, and value token counts must match")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if not value.is_floating_point():
        raise TypeError("value must be floating-point")
    query_heads = query.size(1)
    value_heads = value.size(1)
    if query_heads % value_heads != 0:
        raise ValueError("query heads must be divisible by value heads")
    offsets = _normalize_cu_seqlens(cu_seqlens, query.size(0))

    sam = RosaSam(query_heads, query.size(-1))
    matched_key_end = sam.update(
        query,
        key,
        cu_seqlens=offsets,
    )
    lengths = offsets[1:] - offsets[:-1]
    sequence_start = torch.repeat_interleave(offsets[:-1], lengths).to(
        device=query.device,
        dtype=torch.int64,
    )
    route_index = sequence_start.unsqueeze(1) + matched_key_end + 1
    hard_value = torch.where(value > 0, 1.0, -1.0).to(value.dtype)
    hard_value = hard_value.repeat_interleave(
        query_heads // value_heads,
        dim=1,
    )
    head_index = torch.arange(query_heads, device=query.device).view(1, -1)
    output = hard_value[route_index, head_index]
    output = output.masked_fill(matched_key_end.unsqueeze(-1) < 0, 0.0)
    return output, matched_key_end
