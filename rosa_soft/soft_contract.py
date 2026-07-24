"""Shared public contract for RosaSoft implementations."""

from __future__ import annotations

import math
import operator

import torch
from torch import Tensor


ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE = 1.0
ROSA_SOFT_DEFAULT_MISMATCH_PENALTY = 3.0
ROSA_SOFT_NULL_ROUTE_SCORE = 0.5


def validate_rosa_soft_inputs(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
) -> int:
    if (
        query_logits.ndim != 4
        or key_logits.ndim != 4
        or payload_logits.ndim != 4
    ):
        raise ValueError(
            "query_logits, key_logits, and payload_logits must have shape "
            "(B, T, H, D)"
        )
    if (
        query_logits.shape[:2] != key_logits.shape[:2]
        or query_logits.shape[:2] != payload_logits.shape[:2]
    ):
        raise ValueError("all inputs must share batch and sequence dimensions")
    if query_logits.size(2) != key_logits.size(2):
        raise ValueError("query and key head counts must match")
    if query_logits.size(2) < 1 or payload_logits.size(2) < 1:
        raise ValueError("query and payload must have at least one head")
    if query_logits.size(2) % payload_logits.size(2) != 0:
        raise ValueError("query heads must be divisible by payload heads")
    if query_logits.size(3) != key_logits.size(3):
        raise ValueError("query and key symbol dimensions must match")
    if (
        query_logits.size(0) < 1
        or query_logits.size(1) < 1
        or payload_logits.size(3) < 1
    ):
        raise ValueError("batch, sequence, and payload dimensions must be >= 1")
    if query_logits.size(3) < 1 or query_logits.size(3) > 32:
        raise ValueError("query/key symbol dimension must be in [1, 32]")
    if (
        query_logits.device != key_logits.device
        or query_logits.device != payload_logits.device
    ):
        raise ValueError("all inputs must be on the same device")
    if (
        query_logits.dtype != key_logits.dtype
        or query_logits.dtype != payload_logits.dtype
    ):
        raise ValueError("all inputs must have the same dtype")
    if not all(
        tensor.is_floating_point()
        for tensor in (query_logits, key_logits, payload_logits)
    ):
        raise ValueError("all inputs must be floating-point tensors")
    if isinstance(max_suffix_length, bool):
        raise TypeError("max_suffix_length must be an integer")
    try:
        effective_max_suffix_length = operator.index(max_suffix_length)
    except TypeError as error:
        raise TypeError("max_suffix_length must be an integer") from error
    if effective_max_suffix_length < 1:
        raise ValueError("max_suffix_length must be >= 1")
    if (
        not math.isfinite(float(route_temperature))
        or float(route_temperature) <= 0.0
    ):
        raise ValueError("route_temperature must be finite and > 0")
    if (
        not math.isfinite(float(mismatch_penalty))
        or float(mismatch_penalty) <= 0.0
    ):
        raise ValueError("mismatch_penalty must be finite and > 0")
    return min(effective_max_suffix_length, query_logits.size(1))


def validate_cuda_surrogate_scalars(
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
) -> None:
    float32 = torch.finfo(torch.float32)
    inverse_temperature = 1.0 / float(route_temperature)
    if not (
        math.isfinite(inverse_temperature)
        and float32.tiny <= inverse_temperature <= float32.max
    ):
        raise ValueError(
            "inverse route_temperature must be representable as a positive "
            "normal float32 value"
        )
    if inverse_temperature > float32.max / float(max_suffix_length):
        raise ValueError(
            "max_suffix_length / route_temperature must fit in float32"
        )
    if not float32.tiny <= float(mismatch_penalty) <= float32.max:
        raise ValueError(
            "mismatch_penalty must be representable as a positive normal "
            "float32 value"
        )
