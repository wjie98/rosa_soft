"""Shared public contract for RosaSoft implementations."""

from __future__ import annotations

import math
import operator

import torch
from torch import Tensor


ROSA_SOFT_DEFAULT_SCALE = 1.0
ROSA_SOFT_DEFAULT_DROPOUT_P = 0.0
ROSA_SOFT_DEFAULT_MISMATCH_SCALE = 3.0
ROSA_SOFT_NULL_ROUTE_SCORE = 0.5


def make_dropout_seed(
    reference: Tensor,
    dropout_p: float,
    needs_backward: bool,
) -> Tensor:
    if float(dropout_p) == 0.0 or not needs_backward:
        return torch.empty(0, dtype=torch.int64, device=reference.device)
    return torch.randint(
        0,
        torch.iinfo(torch.int64).max,
        (),
        dtype=torch.int64,
        device=reference.device,
    )


def _validate_static_controls(
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    sequence_length_bound: int,
) -> int:
    if isinstance(max_suffix_length, bool):
        raise TypeError("max_suffix_length must be an integer")
    try:
        effective_max_suffix_length = operator.index(max_suffix_length)
    except TypeError as error:
        raise TypeError("max_suffix_length must be an integer") from error
    if effective_max_suffix_length < 1:
        raise ValueError("max_suffix_length must be >= 1")
    if isinstance(scale, bool) or (
        not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ValueError("scale must be finite and > 0")
    if isinstance(dropout_p, bool) or (
        not math.isfinite(float(dropout_p))
        or not 0.0 <= float(dropout_p) <= 1.0 - 2.0**-24
    ):
        raise ValueError(
            "dropout_p must be finite and in [0, 1 - 2^-24]"
        )
    if isinstance(mismatch_scale, bool) or (
        not math.isfinite(float(mismatch_scale))
        or float(mismatch_scale) <= 0.0
    ):
        raise ValueError("mismatch_scale must be finite and > 0")
    return min(effective_max_suffix_length, sequence_length_bound)


def validate_rosa_soft_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
) -> int:
    if (
        query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
    ):
        raise ValueError(
            "query, key, and value must have shape "
            "(B, T, H, D)"
        )
    if (
        query.shape[:2] != key.shape[:2]
        or query.shape[:2] != value.shape[:2]
    ):
        raise ValueError("all inputs must share batch and sequence dimensions")
    if query.size(2) != key.size(2):
        raise ValueError("query and key head counts must match")
    if query.size(2) < 1 or value.size(2) < 1:
        raise ValueError("query and value must have at least one head")
    if query.size(2) % value.size(2) != 0:
        raise ValueError("query heads must be divisible by value heads")
    if query.size(3) != key.size(3):
        raise ValueError("query and key symbol dimensions must match")
    if (
        query.size(0) < 1
        or query.size(1) < 1
        or value.size(3) < 1
    ):
        raise ValueError("batch, sequence, and value dimensions must be >= 1")
    if query.size(3) < 1 or query.size(3) > 32:
        raise ValueError("query/key symbol dimension must be in [1, 32]")
    if (
        query.device != key.device
        or query.device != value.device
    ):
        raise ValueError("all inputs must be on the same device")
    if (
        query.dtype != key.dtype
        or query.dtype != value.dtype
    ):
        raise ValueError("all inputs must have the same dtype")
    if not all(
        tensor.is_floating_point()
        for tensor in (query, key, value)
    ):
        raise ValueError("all inputs must be floating-point tensors")
    return _validate_static_controls(
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
        query.size(1),
    )


def validate_rosa_soft_varlen_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
) -> int:
    if (
        query.ndim != 3
        or key.ndim != 3
        or value.ndim != 3
    ):
        raise ValueError(
            "query, key, and value must have shape "
            "(N, H, D)"
        )
    if (
        query.size(0) != key.size(0)
        or query.size(0) != value.size(0)
    ):
        raise ValueError("all inputs must share the packed token dimension")
    if query.size(0) < 1:
        raise ValueError("packed token dimension must be >= 1")
    if query.size(2) != key.size(2):
        raise ValueError("query and key symbol dimensions must match")
    if query.size(1) != key.size(1):
        raise ValueError("query and key head counts must match")
    if query.size(1) < 1 or value.size(1) < 1:
        raise ValueError("query and value must have at least one head")
    if query.size(1) % value.size(1) != 0:
        raise ValueError("query heads must be divisible by value heads")
    if query.size(2) < 1 or query.size(2) > 32:
        raise ValueError("query/key symbol dimension must be in [1, 32]")
    if value.size(2) < 1:
        raise ValueError("value dimension must be >= 1")
    if (
        query.device != key.device
        or query.device != value.device
        or query.device != cu_seqlens.device
    ):
        raise ValueError("all inputs and cu_seqlens must be on the same device")
    if (
        query.dtype != key.dtype
        or query.dtype != value.dtype
    ):
        raise ValueError("all floating-point inputs must have the same dtype")
    if not all(
        tensor.is_floating_point()
        for tensor in (query, key, value)
    ):
        raise ValueError("query, key, and value must be floating-point tensors")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional int32 tensor")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must have dtype int32")

    if not cu_seqlens.is_cuda:
        offsets = cu_seqlens.tolist()
        if offsets[0] != 0:
            raise ValueError("cu_seqlens must start at zero")
        if offsets[-1] != query.size(0):
            raise ValueError(
                "cu_seqlens must end at the packed token count"
            )
        if any(left > right for left, right in zip(offsets, offsets[1:])):
            raise ValueError("cu_seqlens must be nondecreasing")

    return _validate_static_controls(
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
        query.size(0),
    )


def validate_fp32_surrogate_scalars(
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
) -> None:
    float32 = torch.finfo(torch.float32)
    if not (
        math.isfinite(float(scale))
        and float32.tiny <= float(scale) <= float32.max
    ):
        raise ValueError(
            "scale must be representable as a positive normal float32 value"
        )
    if float(scale) > float32.max / float(max_suffix_length):
        raise ValueError(
            "max_suffix_length * scale must fit in float32"
        )
    keep_probability = 1.0 - float(dropout_p)
    inverse_keep_probability = 1.0 / keep_probability
    if not (
        math.isfinite(inverse_keep_probability)
        and inverse_keep_probability <= float32.max
    ):
        raise ValueError(
            "1 / (1 - dropout_p) must fit in float32"
        )
    if not float32.tiny <= float(mismatch_scale) <= float32.max:
        raise ValueError(
            "mismatch_scale must be representable as a positive normal "
            "float32 value"
        )
    scaled_horizon = float(max_suffix_length) * float(scale)
    if float(mismatch_scale) > float32.max / scaled_horizon:
        raise ValueError(
            "mismatch_scale * max_suffix_length * scale "
            "must fit in float32"
        )
