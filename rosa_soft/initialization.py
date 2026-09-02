"""Reusable model-layer initialization tools for ROSA projections.

The CUDA operator intentionally has no initialization policy.  These helpers
initialize ordinary ``nn.Linear`` projections while keeping Q, K, and V as
independent trainable parameters after initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch import Tensor


_NORMAL_MEDIAN_ABS = 0.6744897501960817


@dataclass(frozen=True, slots=True)
class RosaProjectionInit:
    """Configuration for the retained ROSA projection initialization recipe."""

    num_heads: int
    qk_bits: int
    num_value_heads: int
    value_bits: int
    shared_qk_bits: int | None = None
    qk_correlation: float = 0.97
    qk_logit_median: float = 0.6
    value_logit_median: float = 0.6
    seed: int = 0
    zero_bias: bool = True


def _validate_positive_integer(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_logit_median(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _validate_linear_pair(query: nn.Linear, key: nn.Linear) -> None:
    if not isinstance(query, nn.Linear) or not isinstance(key, nn.Linear):
        raise TypeError("query and key must be torch.nn.Linear modules")
    if query.in_features != key.in_features:
        raise ValueError("query and key must have the same input width")
    if query.weight.device != key.weight.device:
        raise ValueError("query and key weights must be on the same device")
    if query.weight.dtype != key.weight.dtype:
        raise ValueError("query and key weights must have the same dtype")


def _orthogonal_rows(
    rows: int,
    columns: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    if rows > columns:
        raise ValueError(
            "row-orthogonal initialization requires rows <= input width"
        )
    weights = torch.empty(rows, columns, dtype=torch.float32, device="cpu")
    nn.init.orthogonal_(weights, generator=generator)
    return weights


def _zero_bias_(projection: nn.Linear) -> None:
    if projection.bias is not None:
        projection.bias.zero_()


def initialize_partial_shared_qk_(
    query: nn.Linear,
    key: nn.Linear,
    *,
    num_heads: int,
    qk_bits: int,
    shared_qk_bits: int | None = None,
    correlation: float = 0.97,
    logit_median: float = 0.6,
    seed: int = 0,
    zero_bias: bool = True,
) -> None:
    """Initialize correlated shared Q/K rows plus independent private rows.

    Every head receives an independently seeded orthogonal basis.  Shared rows
    have the requested cosine correlation; private Q and K rows are mutually
    orthogonal.  The projections remain separate parameters.
    """

    _validate_linear_pair(query, key)
    num_heads = _validate_positive_integer(num_heads, "num_heads")
    qk_bits = _validate_positive_integer(qk_bits, "qk_bits")
    expected_outputs = num_heads * qk_bits
    if (
        query.out_features != expected_outputs
        or key.out_features != expected_outputs
    ):
        raise ValueError(
            "query and key output widths must equal num_heads * qk_bits"
        )
    if shared_qk_bits is None:
        shared_qk_bits = max(1, qk_bits // 2)
    shared_qk_bits = int(shared_qk_bits)
    if not 0 <= shared_qk_bits <= qk_bits:
        raise ValueError("shared_qk_bits must be in [0, qk_bits]")
    correlation = float(correlation)
    if not math.isfinite(correlation) or not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must be finite and in [0, 1]")
    logit_median = _validate_logit_median(logit_median, "logit_median")

    hidden_size = query.in_features
    if 2 * qk_bits > hidden_size:
        raise ValueError(
            "partial-shared Q/K initialization requires 2 * qk_bits "
            "<= input width"
        )
    gain = logit_median / _NORMAL_MEDIAN_ABS
    independent_scale = math.sqrt(max(0.0, 1.0 - correlation**2))
    query_heads = []
    key_heads = []
    for head in range(num_heads):
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1_000_003 * (head + 1)
        )
        basis = _orthogonal_rows(
            2 * qk_bits,
            hidden_size,
            generator=generator,
        )
        shared = basis[:shared_qk_bits]
        shared_noise = basis[shared_qk_bits : 2 * shared_qk_bits]
        private_start = 2 * shared_qk_bits
        private_bits = qk_bits - shared_qk_bits
        query_private = basis[private_start : private_start + private_bits]
        key_private = basis[
            private_start + private_bits : private_start + 2 * private_bits
        ]
        query_heads.append(torch.cat((shared, query_private), dim=0))
        key_heads.append(
            torch.cat(
                (
                    correlation * shared + independent_scale * shared_noise,
                    key_private,
                ),
                dim=0,
            )
        )

    with torch.no_grad():
        query.weight.copy_(gain * torch.cat(query_heads, dim=0))
        key.weight.copy_(gain * torch.cat(key_heads, dim=0))
        if zero_bias:
            _zero_bias_(query)
            _zero_bias_(key)


def initialize_orthogonal_value_(
    value: nn.Linear,
    *,
    num_value_heads: int,
    value_bits: int,
    logit_median: float = 0.6,
    seed: int = 0,
    zero_bias: bool = True,
) -> None:
    """Initialize each value head with independent row-orthogonal features."""

    if not isinstance(value, nn.Linear):
        raise TypeError("value must be a torch.nn.Linear module")
    num_value_heads = _validate_positive_integer(
        num_value_heads,
        "num_value_heads",
    )
    value_bits = _validate_positive_integer(value_bits, "value_bits")
    if value.out_features != num_value_heads * value_bits:
        raise ValueError(
            "value output width must equal num_value_heads * value_bits"
        )
    if value_bits > value.in_features:
        raise ValueError(
            "orthogonal V initialization requires value_bits <= input width"
        )
    logit_median = _validate_logit_median(logit_median, "logit_median")
    gain = logit_median / _NORMAL_MEDIAN_ABS
    value_heads = []
    for head in range(num_value_heads):
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 2_000_003 * (head + 1)
        )
        value_heads.append(
            _orthogonal_rows(
                value_bits,
                value.in_features,
                generator=generator,
            )
        )

    with torch.no_grad():
        value.weight.copy_(gain * torch.cat(value_heads, dim=0))
        if zero_bias:
            _zero_bias_(value)


def initialize_rosa_projections_(
    query: nn.Linear,
    key: nn.Linear,
    value: nn.Linear,
    config: RosaProjectionInit,
) -> None:
    """Apply the retained Q/K/V recipe without changing parameter ownership."""

    initialize_partial_shared_qk_(
        query,
        key,
        num_heads=config.num_heads,
        qk_bits=config.qk_bits,
        shared_qk_bits=config.shared_qk_bits,
        correlation=config.qk_correlation,
        logit_median=config.qk_logit_median,
        seed=config.seed + 30_000_001,
        zero_bias=config.zero_bias,
    )
    initialize_orthogonal_value_(
        value,
        num_value_heads=config.num_value_heads,
        value_bits=config.value_bits,
        logit_median=config.value_logit_median,
        seed=config.seed + 40_000_003,
        zero_bias=config.zero_bias,
    )


__all__ = [
    "RosaProjectionInit",
    "initialize_orthogonal_value_",
    "initialize_partial_shared_qk_",
    "initialize_rosa_projections_",
]
