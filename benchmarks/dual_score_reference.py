"""Unbounded discovery/information score reference for hard ROSA.

This module is research-only.  Every mode executes the same exact, unlimited
hard ROSA forward.  The modes differ only in the dense surrogate distribution
used by the custom backward:

``discovery``
    The production normalized-Hamming gate and square-root suffix utility.
``information``
    A tempered, random-collision-normalized suffix Bayes factor.
``dual``
    A fixed convex mixture of the two independently normalized distributions.

The information recurrence is evaluated in log space, so the PyTorch oracle
remains finite even for long exact suffixes.  No continuous score or value
magnitude is visible to the hard forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    ROSA_SOFT_NULL_ROUTE_SCORE,
    make_dropout_seed,
    validate_fp32_surrogate_scalars,
    validate_rosa_soft_inputs,
    validate_rosa_soft_varlen_inputs,
)
from rosa_soft.soft_reference import (
    _apply_attention_dropout,
    _build_vjp_carrier,
    _causal_route_mask,
    _expand_value_heads,
    _gather_routed_values,
    _hard_sign,
    _hard_sign_with_softsign_vjp,
    _pairwise_exact_symbol_match,
    _reference_compute_dtype,
    _select_latest_longest_routes,
    _suffix_score_utility,
)


SCORE_MODES = ("discovery", "information", "dual")
DEFAULT_EVIDENCE_POWER = 0.25
DEFAULT_INFORMATION_WEIGHT = 0.5

_MODE_IDS = {name: index for index, name in enumerate(SCORE_MODES)}
_MODES_BY_ID = {index: name for name, index in _MODE_IDS.items()}


@dataclass(frozen=True)
class DualScoreState:
    """Materialized score tensors for semantic and gradient inspection."""

    causal_route_mask: Tensor
    mismatch_count: Tensor
    discovery_suffix_score: Tensor
    discovery_logits: Tensor
    discovery_probabilities: Tensor
    information_log_evidence: Tensor
    information_logits: Tensor
    information_probabilities: Tensor
    route_probabilities: Tensor
    null_log_gate_mean: float


def _validate_score_controls(
    score_mode: str,
    evidence_power: float,
    information_weight: float,
) -> None:
    if score_mode not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {SCORE_MODES}")
    if isinstance(evidence_power, bool) or not (
        math.isfinite(float(evidence_power))
        and 0.0 < float(evidence_power) <= 1.0
    ):
        raise ValueError("evidence_power must be finite and in (0, 1]")
    if isinstance(information_weight, bool) or not (
        math.isfinite(float(information_weight))
        and 0.0 <= float(information_weight) <= 1.0
    ):
        raise ValueError("information_weight must be finite and in [0, 1]")


def _validate_dense_call(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    score_mode: str,
    evidence_power: float,
    information_weight: float,
) -> None:
    horizon = validate_rosa_soft_inputs(
        query,
        key,
        value,
        query.size(1) if query.ndim == 4 else 1,
        scale,
        dropout_p,
        mismatch_scale,
    )
    _reference_compute_dtype(query.dtype)
    validate_fp32_surrogate_scalars(
        horizon,
        scale,
        dropout_p,
        mismatch_scale,
    )
    _validate_score_controls(
        score_mode,
        evidence_power,
        information_weight,
    )


def _unbounded_suffix_scores(local_match: Tensor) -> Tensor:
    """Evaluate ``S[t,a] = g[t,a] * (1 + S[t-1,a-1])`` in O(T^2)."""

    seq_len = local_match.size(-1)
    previous = local_match.new_zeros(*local_match.shape[:-2], seq_len)
    rows = []
    for row_index in range(seq_len):
        if row_index == 0:
            current = previous
        else:
            active = local_match[..., row_index, 1 : row_index + 1] * (
                1.0 + previous[..., :row_index]
            )
            current = F.pad(active, (1, seq_len - row_index - 1))
        rows.append(current)
        previous = current
    return torch.stack(rows, dim=-2)


def _unbounded_information_log_evidence(
    local_log_gate: Tensor,
    *,
    log_null_gate_mean: float,
    evidence_power: float,
) -> Tensor:
    """Evaluate the tempered suffix Bayes-factor numerator in log space."""

    seq_len = local_log_gate.size(-1)
    negative_infinity = local_log_gate.new_full(
        (*local_log_gate.shape[:-2], seq_len),
        -torch.inf,
    )
    previous = negative_infinity
    rows = []
    log_inverse_background = (
        -float(evidence_power) * float(log_null_gate_mean)
    )
    for row_index in range(seq_len):
        if row_index == 0:
            current = negative_infinity
        else:
            previous_active = previous[..., :row_index]
            active = (
                local_log_gate[..., row_index, 1 : row_index + 1]
                + log_inverse_background
                + torch.logaddexp(
                    torch.zeros_like(previous_active),
                    previous_active,
                )
            )
            current = F.pad(
                active,
                (1, seq_len - row_index - 1),
                value=-torch.inf,
            )
        rows.append(current)
        previous = current
    log_numerator = torch.stack(rows, dim=-2)

    if seq_len == 1:
        return log_numerator
    lengths = torch.arange(
        1,
        seq_len,
        dtype=local_log_gate.dtype,
        device=local_log_gate.device,
    )
    log_background_terms = (
        (1.0 - float(evidence_power))
        * float(log_null_gate_mean)
        * lengths
    )
    log_normalizer = torch.logcumsumexp(log_background_terms, dim=0)
    route_normalizer = F.pad(log_normalizer, (1, 0))
    return log_numerator - route_normalizer.view(
        *((1,) * (log_numerator.ndim - 1)), seq_len
    )


def tempered_diagonal_log_evidence(
    local_log_gates: Tensor,
    *,
    symbol_dim: int,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    evidence_power: float = DEFAULT_EVIDENCE_POWER,
) -> Tensor:
    """Return a calibrated final-route score for independent gate sequences."""

    if local_log_gates.ndim < 1 or local_log_gates.size(-1) < 1:
        raise ValueError("local_log_gates must have a nonempty final dimension")
    if symbol_dim < 1 or symbol_dim > 32:
        raise ValueError("symbol_dim must be in [1, 32]")
    _validate_score_controls("information", evidence_power, 1.0)
    if not math.isfinite(float(mismatch_scale)) or mismatch_scale <= 0.0:
        raise ValueError("mismatch_scale must be finite and positive")
    log_z = null_log_gate_mean(symbol_dim, mismatch_scale)
    lengths = torch.arange(
        1,
        local_log_gates.size(-1) + 1,
        dtype=local_log_gates.dtype,
        device=local_log_gates.device,
    )
    log_numerator = torch.logsumexp(
        local_log_gates.cumsum(dim=-1)
        - float(evidence_power) * log_z * lengths,
        dim=-1,
    )
    log_normalizer = torch.logsumexp(
        (1.0 - float(evidence_power)) * log_z * lengths,
        dim=-1,
    )
    return log_numerator - log_normalizer


def null_log_gate_mean(symbol_dim: int, mismatch_scale: float) -> float:
    """Return ``log E[exp(-lambda H)]`` for ``H ~ Binomial(D, 1/2)``."""

    if symbol_dim < 1 or symbol_dim > 32:
        raise ValueError("symbol_dim must be in [1, 32]")
    mismatch_scale = float(mismatch_scale)
    if not math.isfinite(mismatch_scale) or mismatch_scale <= 0.0:
        raise ValueError("mismatch_scale must be finite and positive")
    one_bit_log_mean = math.log1p(math.exp(-mismatch_scale)) - math.log(2.0)
    return symbol_dim * one_bit_log_mean


def _pairwise_log_gates(
    query: Tensor,
    key: Tensor,
    causal_route_mask: Tensor,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    query_symbols = _hard_sign_with_softsign_vjp(
        query.permute(0, 2, 1, 3)
    )
    key_symbols = _hard_sign_with_softsign_vjp(
        key.permute(0, 2, 1, 3)[..., :-1, :]
    )
    mismatch_count = 0.5 * (
        1.0
        - query_symbols.unsqueeze(-2) * key_symbols.unsqueeze(-3)
    ).sum(dim=-1)
    mismatch_count = F.pad(mismatch_count, (1, 0), value=0.0)
    seq_len = query.size(1)
    route_index = torch.arange(seq_len, device=query.device).view(1, seq_len)
    nonnull = causal_route_mask & route_index.gt(0)
    nonnull = nonnull.view(1, 1, seq_len, seq_len)
    mismatch_count = torch.where(
        nonnull,
        mismatch_count,
        torch.zeros_like(mismatch_count),
    )
    information_log_gate = -float(mismatch_scale) * mismatch_count
    discovery_gate = torch.where(
        nonnull,
        torch.exp(information_log_gate / query.size(-1)),
        torch.zeros_like(information_log_gate),
    )
    return mismatch_count, discovery_gate, information_log_gate


def _route_logits(
    candidate_scores: Tensor,
    causal_route_mask: Tensor,
    *,
    null_logit: float,
) -> Tensor:
    seq_len = candidate_scores.size(-1)
    row_count = torch.arange(
        seq_len,
        dtype=candidate_scores.dtype,
        device=candidate_scores.device,
    ).clamp_min(1)
    logits = candidate_scores - row_count.log().view(1, 1, seq_len, 1)
    logits = logits.clone()
    logits[..., 0] = float(null_logit)
    return logits.masked_fill(
        ~causal_route_mask.view(1, 1, seq_len, seq_len),
        -torch.inf,
    )


def _discovery_route_probabilities(
    discovery_gate: Tensor,
    causal_route_mask: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    suffix_score = _unbounded_suffix_scores(discovery_gate)
    logits = _route_logits(
        float(scale) * _suffix_score_utility(suffix_score),
        causal_route_mask,
        null_logit=float(scale) * ROSA_SOFT_NULL_ROUTE_SCORE,
    )
    return suffix_score, logits, torch.softmax(logits, dim=-1)


def _information_route_probabilities(
    information_log_gate: Tensor,
    causal_route_mask: Tensor,
    *,
    symbol_dim: int,
    mismatch_scale: float,
    evidence_power: float,
) -> tuple[Tensor, Tensor, Tensor, float]:
    log_z = null_log_gate_mean(symbol_dim, mismatch_scale)
    log_evidence = _unbounded_information_log_evidence(
        information_log_gate,
        log_null_gate_mean=log_z,
        evidence_power=evidence_power,
    )
    logits = _route_logits(
        log_evidence,
        causal_route_mask,
        null_logit=0.0,
    )
    return log_evidence, logits, torch.softmax(logits, dim=-1), log_z


def _build_route_probabilities(
    query: Tensor,
    key: Tensor,
    *,
    score_mode: str,
    scale: float,
    mismatch_scale: float,
    evidence_power: float,
    information_weight: float,
) -> Tensor:
    causal_route_mask = _causal_route_mask(query.size(1), query.device)
    _, discovery_gate, information_log_gate = _pairwise_log_gates(
        query,
        key,
        causal_route_mask,
        mismatch_scale,
    )
    if score_mode == "discovery":
        return _discovery_route_probabilities(
            discovery_gate, causal_route_mask, scale
        )[-1]
    if score_mode == "information":
        return _information_route_probabilities(
            information_log_gate,
            causal_route_mask,
            symbol_dim=query.size(-1),
            mismatch_scale=mismatch_scale,
            evidence_power=evidence_power,
        )[2]
    discovery_probabilities = _discovery_route_probabilities(
        discovery_gate, causal_route_mask, scale
    )[-1]
    information_probabilities = _information_route_probabilities(
        information_log_gate,
        causal_route_mask,
        symbol_dim=query.size(-1),
        mismatch_scale=mismatch_scale,
        evidence_power=evidence_power,
    )[2]
    return (
        (1.0 - float(information_weight)) * discovery_probabilities
        + float(information_weight) * information_probabilities
    )


def _build_score_state(
    query: Tensor,
    key: Tensor,
    *,
    scale: float,
    mismatch_scale: float,
    evidence_power: float,
    information_weight: float,
) -> DualScoreState:
    seq_len = query.size(1)
    causal_route_mask = _causal_route_mask(seq_len, query.device)
    mismatch_count, discovery_gate, information_log_gate = (
        _pairwise_log_gates(
            query,
            key,
            causal_route_mask,
            mismatch_scale,
        )
    )
    (
        discovery_suffix_score,
        discovery_logits,
        discovery_probabilities,
    ) = _discovery_route_probabilities(
        discovery_gate,
        causal_route_mask,
        scale,
    )
    (
        information_log_evidence,
        information_logits,
        information_probabilities,
        log_z,
    ) = _information_route_probabilities(
        information_log_gate,
        causal_route_mask,
        symbol_dim=query.size(-1),
        mismatch_scale=mismatch_scale,
        evidence_power=evidence_power,
    )
    route_probabilities = (
        (1.0 - float(information_weight)) * discovery_probabilities
        + float(information_weight) * information_probabilities
    )
    return DualScoreState(
        causal_route_mask=causal_route_mask,
        mismatch_count=mismatch_count,
        discovery_suffix_score=discovery_suffix_score,
        discovery_logits=discovery_logits,
        discovery_probabilities=discovery_probabilities,
        information_log_evidence=information_log_evidence,
        information_logits=information_logits,
        information_probabilities=information_probabilities,
        route_probabilities=route_probabilities,
        null_log_gate_mean=log_z,
    )


def build_dual_score_state(
    query: Tensor,
    key: Tensor,
    *,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    evidence_power: float = DEFAULT_EVIDENCE_POWER,
    information_weight: float = DEFAULT_INFORMATION_WEIGHT,
) -> DualScoreState:
    """Materialize both unbounded route distributions for inspection."""

    if query.ndim != 4 or key.ndim != 4 or query.shape != key.shape:
        raise ValueError("query and key must have identical shape (B, T, H, D)")
    dummy_value = query.new_empty(
        query.size(0), query.size(1), query.size(2), 1
    )
    _validate_dense_call(
        query,
        key,
        dummy_value,
        scale,
        0.0,
        mismatch_scale,
        "dual",
        evidence_power,
        information_weight,
    )
    compute_dtype = _reference_compute_dtype(query.dtype)
    return _build_score_state(
        query.to(compute_dtype),
        key.to(compute_dtype),
        scale=scale,
        mismatch_scale=mismatch_scale,
        evidence_power=evidence_power,
        information_weight=information_weight,
    )


def _hard_route_forward_unbounded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    causal_route_mask = _causal_route_mask(query.size(1), query.device)
    exact_local = _pairwise_exact_symbol_match(
        query,
        key,
        causal_route_mask,
    )
    exact_suffix_lengths = _unbounded_suffix_scores(exact_local.to(query.dtype))
    selected_route_indices = _select_latest_longest_routes(
        exact_suffix_lengths,
        causal_route_mask,
    )
    route_values = _expand_value_heads(_hard_sign(value), query.size(2))
    route_values[..., 0, :] = 0.0
    hard_output = _gather_routed_values(
        route_values,
        selected_route_indices,
    ).permute(0, 2, 1, 3)
    return (
        hard_output,
        exact_suffix_lengths,
        selected_route_indices,
        causal_route_mask,
    )


def _selected_probabilities(state: DualScoreState, score_mode: str) -> Tensor:
    if score_mode == "discovery":
        return state.discovery_probabilities
    if score_mode == "information":
        return state.information_probabilities
    return state.route_probabilities


def dual_score_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    score_mode: str = "dual",
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    evidence_power: float = DEFAULT_EVIDENCE_POWER,
    information_weight: float = DEFAULT_INFORMATION_WEIGHT,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    dropout_seed: Optional[Tensor] = None,
    dropout_batch_offset: int = 0,
) -> Tensor:
    """Return the differentiable carrier used only by the custom backward."""

    _validate_dense_call(
        query,
        key,
        value,
        scale,
        dropout_p,
        mismatch_scale,
        score_mode,
        evidence_power,
        information_weight,
    )
    probabilities = _build_route_probabilities(
        query,
        key,
        score_mode=score_mode,
        scale=scale,
        mismatch_scale=mismatch_scale,
        evidence_power=evidence_power,
        information_weight=information_weight,
    )
    if dropout_seed is None:
        dropout_seed = make_dropout_seed(
            query,
            dropout_p,
            torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in (query, key, value)),
        )
    probabilities = _apply_attention_dropout(
        probabilities,
        dropout_p,
        dropout_seed,
        dropout_batch_offset,
    )
    return _build_vjp_carrier(value, probabilities, query.size(2))


def final_query_score_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    score_mode: str = "dual",
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    evidence_power: float = DEFAULT_EVIDENCE_POWER,
    information_weight: float = DEFAULT_INFORMATION_WEIGHT,
) -> tuple[Tensor, Tensor]:
    """Evaluate one query suffix against every key endpoint in O(P*T*W).

    ``query`` has shape ``[P, W, D]`` and ``key`` has shape ``[P, T, D]``.
    This is the exact final-row specialization of the dense score recurrence;
    it is used only for controlled long-context research.
    """

    _validate_score_controls(score_mode, evidence_power, information_weight)
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must be rank-3 tensors")
    if query.size(0) != key.size(0) or key.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, and value batch/context shapes must agree")
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key symbol dimensions must match")
    if query.size(1) < 1 or query.size(1) > key.size(1):
        raise ValueError("query suffix length must be in [1, context length]")
    if query.size(-1) < 1 or query.size(-1) > 32:
        raise ValueError("query/key symbol dimension must be in [1, 32]")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must share a device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must share a dtype")
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if not math.isfinite(float(mismatch_scale)) or mismatch_scale <= 0.0:
        raise ValueError("mismatch_scale must be finite and positive")

    query_symbols = _hard_sign_with_softsign_vjp(query)
    key_symbols = _hard_sign_with_softsign_vjp(key)
    context_length = key.size(1)
    suffix_length = query.size(1)
    information_levels = []
    valid_levels = []
    for offset in range(suffix_length):
        key_prefix = key_symbols[:, : context_length - offset]
        mismatch_count = 0.5 * (
            1.0
            - query_symbols[:, suffix_length - offset - 1].unsqueeze(1)
            * key_prefix
        ).sum(dim=-1)
        information_levels.append(
            F.pad(-float(mismatch_scale) * mismatch_count, (offset, 0))
        )
        valid_levels.append(
            torch.arange(context_length, device=query.device).ge(offset)
        )
    information_log_gate = torch.stack(information_levels, dim=-1)
    valid = torch.stack(valid_levels, dim=-1).view(
        1, context_length, suffix_length
    )

    candidate_prior = math.log(context_length)
    discovery_probabilities = None
    if score_mode != "information":
        discovery_products = torch.exp(
            information_log_gate / query.size(-1)
        ).cumprod(dim=-1)
        discovery_products = torch.where(
            valid, discovery_products, torch.zeros_like(discovery_products)
        )
        discovery_score = _suffix_score_utility(
            discovery_products.sum(dim=-1)
        )
        discovery_logits = float(scale) * discovery_score - candidate_prior
        discovery_logits = torch.cat(
            (
                discovery_logits.new_full(
                    (query.size(0), 1),
                    float(scale) * ROSA_SOFT_NULL_ROUTE_SCORE,
                ),
                discovery_logits,
            ),
            dim=-1,
        )
        discovery_probabilities = torch.softmax(discovery_logits, dim=-1)

    information_probabilities = None
    if score_mode != "discovery":
        log_z = null_log_gate_mean(query.size(-1), mismatch_scale)
        lengths = torch.arange(
            1,
            suffix_length + 1,
            dtype=query.dtype,
            device=query.device,
        )
        information_terms = (
            information_log_gate.cumsum(dim=-1)
            - float(evidence_power) * log_z * lengths
        ).masked_fill(~valid, -torch.inf)
        log_numerator = torch.logsumexp(information_terms, dim=-1)
        log_normalizers = torch.logcumsumexp(
            (1.0 - float(evidence_power)) * log_z * lengths,
            dim=0,
        )
        available_lengths = torch.arange(
            1, context_length + 1, device=query.device
        ).clamp_max(suffix_length)
        information_score = log_numerator - log_normalizers[
            available_lengths - 1
        ].view(1, -1)
        information_logits = torch.cat(
            (
                information_score.new_zeros(query.size(0), 1),
                information_score - candidate_prior,
            ),
            dim=-1,
        )
        information_probabilities = torch.softmax(information_logits, dim=-1)

    if score_mode == "discovery":
        assert discovery_probabilities is not None
        probabilities = discovery_probabilities
    elif score_mode == "information":
        assert information_probabilities is not None
        probabilities = information_probabilities
    else:
        assert discovery_probabilities is not None
        assert information_probabilities is not None
        probabilities = (
            (1.0 - float(information_weight)) * discovery_probabilities
            + float(information_weight) * information_probabilities
        )
    route_probabilities = probabilities[:, 1:]
    signed_value = _hard_sign_with_softsign_vjp(value)
    carrier = torch.einsum("pt,ptv->pv", route_probabilities, signed_value)
    return carrier, route_probabilities


class _HardForwardDualScoreVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_seed: Tensor,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        evidence_power: float,
        information_weight: float,
        score_mode: int,
        dropout_batch_offset: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward_unbounded(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
        )
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.evidence_power = float(evidence_power)
        ctx.information_weight = float(information_weight)
        ctx.score_mode = _MODES_BY_ID[int(score_mode)]
        ctx.dropout_batch_offset = int(dropout_batch_offset)
        ctx.save_for_backward(query, key, value, dropout_seed)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value, dropout_seed = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            leaves = tuple(
                tensor.detach().requires_grad_(need)
                for tensor, need in zip((query, key, value), needs)
            )
            compute_dtype = _reference_compute_dtype(query.dtype)
            carrier = dual_score_carrier(
                *(tensor.to(compute_dtype) for tensor in leaves),
                score_mode=ctx.score_mode,
                scale=ctx.scale,
                mismatch_scale=ctx.mismatch_scale,
                evidence_power=ctx.evidence_power,
                information_weight=ctx.information_weight,
                dropout_p=ctx.dropout_p,
                dropout_seed=dropout_seed,
                dropout_batch_offset=ctx.dropout_batch_offset,
            ).to(query.dtype)
            required_indices = [
                index for index, need in enumerate(needs) if need
            ]
            required_gradients = torch.autograd.grad(
                carrier,
                tuple(leaves[index] for index in required_indices),
                grad_output,
                create_graph=False,
                allow_unused=True,
            )
            gradients = [None, None, None]
            for index, gradient in zip(required_indices, required_gradients):
                gradients[index] = (
                    torch.zeros_like(leaves[index])
                    if gradient is None
                    else gradient
                )
        return (
            gradients[0],
            gradients[1],
            gradients[2],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _dual_score_reference_with_seed(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_seed: Tensor,
    *,
    score_mode: str,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    evidence_power: float,
    information_weight: float,
    dropout_batch_offset: int,
) -> Tensor:
    if not torch.is_grad_enabled() or not any(
        tensor.requires_grad for tensor in (query, key, value)
    ):
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward_unbounded(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
        )
        return hard_output.to(query.dtype)
    return _HardForwardDualScoreVjp.apply(
        query,
        key,
        value,
        dropout_seed,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        float(evidence_power),
        float(information_weight),
        _MODE_IDS[score_mode],
        int(dropout_batch_offset),
    )


def rosa_dual_score_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    score_mode: str = "dual",
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    evidence_power: float = DEFAULT_EVIDENCE_POWER,
    information_weight: float = DEFAULT_INFORMATION_WEIGHT,
) -> Tensor:
    """Return exact unlimited ROSA values with an unbounded research VJP."""

    _validate_dense_call(
        query,
        key,
        value,
        scale,
        dropout_p,
        mismatch_scale,
        score_mode,
        evidence_power,
        information_weight,
    )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    dropout_seed = make_dropout_seed(query, dropout_p, needs_backward)
    return _dual_score_reference_with_seed(
        query,
        key,
        value,
        dropout_seed,
        score_mode=score_mode,
        scale=scale,
        dropout_p=dropout_p,
        mismatch_scale=mismatch_scale,
        evidence_power=evidence_power,
        information_weight=information_weight,
        dropout_batch_offset=0,
    )


def rosa_dual_score_varlen_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    *,
    score_mode: str = "dual",
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    evidence_power: float = DEFAULT_EVIDENCE_POWER,
    information_weight: float = DEFAULT_INFORMATION_WEIGHT,
) -> Tensor:
    """Apply the unbounded research reference to packed independent segments."""

    horizon = validate_rosa_soft_varlen_inputs(
        query,
        key,
        value,
        cu_seqlens,
        query.size(0),
        scale,
        dropout_p,
        mismatch_scale,
    )
    _reference_compute_dtype(query.dtype)
    validate_fp32_surrogate_scalars(
        horizon,
        scale,
        dropout_p,
        mismatch_scale,
    )
    _validate_score_controls(
        score_mode,
        evidence_power,
        information_weight,
    )
    offsets = cu_seqlens.detach().cpu().tolist()
    if offsets[0] != 0 or offsets[-1] != query.size(0):
        raise ValueError("cu_seqlens must span the packed token dimension")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    dropout_seed = make_dropout_seed(query, dropout_p, needs_backward)
    outputs = []
    for sequence, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if start == end:
            continue
        outputs.append(
            _dual_score_reference_with_seed(
                query[start:end].unsqueeze(0),
                key[start:end].unsqueeze(0),
                value[start:end].unsqueeze(0),
                dropout_seed,
                score_mode=score_mode,
                scale=scale,
                dropout_p=dropout_p,
                mismatch_scale=mismatch_scale,
                evidence_power=evidence_power,
                information_weight=information_weight,
                dropout_batch_offset=sequence,
            ).squeeze(0)
        )
    return torch.cat(outputs, dim=0)
