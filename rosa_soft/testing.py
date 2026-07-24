"""Testing and inspection hooks kept outside the RosaSoft training API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    validate_rosa_soft_inputs,
)
from .soft_reference import (
    _causal_route_mask,
    _hard_route_forward,
    _masked_route_scores,
    _mismatch_uniform_shape,
    _pairwise_stochastic_match_gates,
    _reference_dtype,
    _rosa_soft_reference_with_uniforms,
    _route_probabilities,
    _sample_mismatch_uniforms,
    _suffix_prefix_product_scores,
)


@dataclass(frozen=True)
class RosaSoftInspection:
    exact_suffix_lengths: Tensor
    proxy_scores: Tensor
    route_probabilities: Tensor
    valid_routes: Tensor
    selected_routes: Tensor
    route_temperature: float
    mismatch_penalty: float

    @property
    def route_scores(self) -> Tensor:
        return _masked_route_scores(
            self.proxy_scores,
            self.valid_routes,
        )


def rosa_soft_reference_with_uniforms(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    mismatch_uniforms: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
) -> Tensor:
    max_suffix_length = validate_rosa_soft_inputs(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )
    _validate_mismatch_uniforms(query_logits, mismatch_uniforms)
    return _rosa_soft_reference_with_uniforms(
        query_logits,
        key_logits,
        payload_logits,
        mismatch_uniforms,
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
    from .soft import _rosa_soft_cuda_with_seed, _validate_cuda_call

    max_suffix_length = _validate_cuda_call(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )
    if (
        rng_seed.device != query_logits.device
        or rng_seed.dtype != torch.int64
        or rng_seed.numel() != 1
    ):
        raise ValueError(
            "rng_seed must be one int64 value on the input CUDA device"
        )
    return _rosa_soft_cuda_with_seed(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
        rng_seed,
    )


def _validate_mismatch_uniforms(
    query_logits: Tensor,
    mismatch_uniforms: Tensor,
) -> None:
    expected_shape = _mismatch_uniform_shape(query_logits)
    if tuple(mismatch_uniforms.shape) != expected_shape:
        raise ValueError(
            f"mismatch_uniforms must have shape {expected_shape}"
        )
    if (
        mismatch_uniforms.device != query_logits.device
        or not mismatch_uniforms.is_floating_point()
    ):
        raise ValueError(
            "mismatch_uniforms must be floating-point on the input device"
        )
    if not bool(
        torch.isfinite(mismatch_uniforms).all()
        and (mismatch_uniforms >= 0).all()
        and (mismatch_uniforms <= 1).all()
    ):
        raise ValueError(
            "mismatch_uniforms must be finite values in [0, 1]"
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
    mismatch_uniforms: Optional[Tensor] = None,
) -> Tuple[Tensor, RosaSoftInspection]:
    max_suffix_length = validate_rosa_soft_inputs(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )
    reference_dtype = _reference_dtype(query_logits.dtype)
    query_work = query_logits.to(reference_dtype)
    key_work = key_logits.to(reference_dtype)
    payload_work = payload_logits.to(reference_dtype)
    if mismatch_uniforms is None:
        mismatch_uniforms = _sample_mismatch_uniforms(
            query_logits,
            generator=generator,
        )
    else:
        _validate_mismatch_uniforms(
            query_logits,
            mismatch_uniforms,
        )
    mismatch_uniforms = mismatch_uniforms.to(
        device=query_logits.device,
        dtype=reference_dtype,
    )

    (
        hard_output,
        exact_suffix_lengths,
        selected_routes,
        valid_routes,
    ) = (
        _hard_route_forward(
            query_work,
            key_work,
            payload_work,
            max_suffix_length,
        )
    )
    proxy_local = _pairwise_stochastic_match_gates(
        query_work,
        key_work,
        valid_routes,
        float(mismatch_penalty),
        mismatch_uniforms,
    )
    proxy_scores = _suffix_prefix_product_scores(
        proxy_local,
        max_suffix_length,
    )
    route_probabilities = _route_probabilities(
        _masked_route_scores(proxy_scores, valid_routes),
        float(route_temperature),
    )
    inspection = RosaSoftInspection(
        exact_suffix_lengths=exact_suffix_lengths,
        proxy_scores=proxy_scores,
        route_probabilities=route_probabilities,
        valid_routes=valid_routes,
        selected_routes=selected_routes,
        route_temperature=float(route_temperature),
        mismatch_penalty=float(mismatch_penalty),
    )
    return hard_output.to(query_logits.dtype), inspection
