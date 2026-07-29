"""Inspection hooks kept outside the RosaSoft training API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from .soft_reference import (
    _hard_route_forward,
    _masked_route_scores,
    _pairwise_soft_match_gates,
    _reference_compute_dtype,
    _route_probabilities,
    _suffix_prefix_product_scores,
    _validate_reference_call,
)


@dataclass(frozen=True)
class RosaSoftInspection:
    query: Tensor
    key: Tensor
    exact_suffix_lengths: Tensor
    soft_suffix_scores: Tensor
    route_probabilities: Tensor
    causal_route_mask: Tensor
    selected_route_indices: Tensor
    effective_max_suffix_length: int
    scale: float
    dropout_p: float
    mismatch_scale: float

    @property
    def route_scores(self) -> Tensor:
        return _masked_route_scores(
            self.soft_suffix_scores,
            self.causal_route_mask,
        )


@torch.no_grad()
def inspect_rosa_soft(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tuple[Tensor, RosaSoftInspection]:
    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    output_dtype = query.dtype
    compute_dtype = _reference_compute_dtype(output_dtype)
    query = query.to(compute_dtype)
    key = key.to(compute_dtype)
    value = value.to(compute_dtype)
    (
        hard_output,
        exact_suffix_lengths,
        selected_route_indices,
        causal_route_mask,
    ) = _hard_route_forward(
        query,
        key,
        value,
        max_suffix_length,
    )
    local_match_gates = _pairwise_soft_match_gates(
        query,
        key,
        causal_route_mask,
        mismatch_scale,
    )
    soft_suffix_scores = _suffix_prefix_product_scores(
        local_match_gates,
        max_suffix_length,
    )
    route_probabilities = _route_probabilities(
        _masked_route_scores(
            soft_suffix_scores,
            causal_route_mask,
        ),
        causal_route_mask,
        scale,
    )
    inspection = RosaSoftInspection(
        query=query.detach(),
        key=key.detach(),
        exact_suffix_lengths=exact_suffix_lengths.detach(),
        soft_suffix_scores=soft_suffix_scores.detach(),
        route_probabilities=route_probabilities.detach(),
        causal_route_mask=causal_route_mask.detach(),
        selected_route_indices=selected_route_indices.detach(),
        effective_max_suffix_length=max_suffix_length,
        scale=float(scale),
        dropout_p=float(dropout_p),
        mismatch_scale=float(mismatch_scale),
    )
    return hard_output.to(output_dtype).detach(), inspection
