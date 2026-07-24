"""Testing and inspection hooks kept outside the RosaSoft training API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    validate_soft_inputs,
)
from .soft_reference import (
    _action_mask,
    _allocation_scores,
    _hard_components,
    _mismatch_shape,
    _pairwise_proxy_local_match,
    _proxy_probabilities,
    _rosa_soft_reference_with_noise,
    _sample_mismatch_noise,
    _working_dtype,
    _diagonal_suffix_sum,
)


@dataclass(frozen=True)
class RosaSoftInspection:
    hard_lengths: Tensor
    proxy_scores: Tensor
    route_scores: Tensor
    route_probabilities: Tensor
    valid_actions: Tensor
    selected_actions: Tensor
    mismatch_noise: Tensor
    route_temperature: float
    mismatch_penalty: float


def rosa_soft_reference_with_noise(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    mismatch_noise: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
) -> Tensor:
    max_suffix_length = validate_soft_inputs(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )
    return _rosa_soft_reference_with_noise(
        query_logits,
        key_logits,
        payload_logits,
        mismatch_noise,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )


def rosa_soft_with_seed(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
    rng_seed: Tensor,
) -> Tensor:
    from .soft import _rosa_soft_with_seed, _validate_cuda_inputs

    max_suffix_length = _validate_cuda_inputs(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )
    return _rosa_soft_with_seed(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
        rng_seed,
    )


@torch.no_grad()
def inspect_rosa_soft(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    max_suffix_length: int = 32,
    route_temperature: float = ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    mismatch_penalty: float = ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    *,
    generator: Optional[torch.Generator] = None,
    mismatch_noise: Optional[Tensor] = None,
) -> Tuple[Tensor, RosaSoftInspection]:
    max_suffix_length = validate_soft_inputs(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )
    work_dtype = _working_dtype(query_logits.dtype)
    query_work = query_logits.to(work_dtype)
    key_work = key_logits.to(work_dtype)
    payload_work = payload_logits.to(work_dtype)
    if mismatch_noise is None:
        mismatch_noise = _sample_mismatch_noise(
            query_logits,
            generator=generator,
        )
    elif tuple(mismatch_noise.shape) != _mismatch_shape(query_logits):
        raise ValueError("mismatch_noise has the wrong shape")
    mismatch_noise = mismatch_noise.to(
        device=query_logits.device,
        dtype=work_dtype,
    )

    hard_output, hard_lengths, selected_actions, valid_actions = (
        _hard_components(
            query_work,
            key_work,
            payload_work,
            max_suffix_length,
        )
    )
    proxy_local = _pairwise_proxy_local_match(
        query_work,
        key_work,
        valid_actions,
        float(mismatch_penalty),
        mismatch_noise,
    )
    proxy_scores = _diagonal_suffix_sum(proxy_local, max_suffix_length)
    route_scores = _allocation_scores(proxy_scores, valid_actions)
    route_probabilities = _proxy_probabilities(
        route_scores,
        float(route_temperature),
    )
    inspection = RosaSoftInspection(
        hard_lengths=hard_lengths,
        proxy_scores=proxy_scores,
        route_scores=route_scores,
        route_probabilities=route_probabilities,
        valid_actions=valid_actions,
        selected_actions=selected_actions,
        mismatch_noise=mismatch_noise.detach(),
        route_temperature=float(route_temperature),
        mismatch_penalty=float(mismatch_penalty),
    )
    return hard_output.to(query_logits.dtype), inspection
