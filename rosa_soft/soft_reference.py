"""Minimal hard-forward/soft-backward PyTorch oracle for RosaSoft.

Forward executes exact discrete ROSA. Backward uses one dense surrogate:
softsign-STE symbols, exponential Hamming match gates, concave suffix evidence,
candidate-normalized soft routing, and optional attention dropout. It is
deterministic when ``dropout_p=0``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from .soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    ROSA_SOFT_NULL_ROUTE_SCORE,
    make_dropout_seed,
    validate_fp32_surrogate_scalars,
    validate_rosa_soft_inputs,
    validate_rosa_soft_varlen_inputs,
)

__all__ = [
    "ROSA_SOFT_DEFAULT_DROPOUT_P",
    "ROSA_SOFT_DEFAULT_MISMATCH_SCALE",
    "ROSA_SOFT_DEFAULT_SCALE",
    "rosa_soft_reference",
    "rosa_soft_varlen_reference",
]


_REFERENCE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_NORMALIZED_SQRT_SUFFIX_SCALE = math.sqrt(2.0) + 1.0


def _reference_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _REFERENCE_DTYPES:
        raise ValueError(
            "rosa_soft_reference supports float16, bfloat16, float32, "
            "and float64"
        )
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _validate_reference_call(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
) -> int:
    effective_max_suffix_length = validate_rosa_soft_inputs(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    _reference_compute_dtype(query.dtype)
    validate_fp32_surrogate_scalars(
        effective_max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    return effective_max_suffix_length


def _hard_sign(x: Tensor) -> Tensor:
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))


class _HardSignWithSoftsignVjp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor) -> Tensor:
        ctx.save_for_backward(logits)
        return _hard_sign(logits)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (logits,) = ctx.saved_tensors
        denominator = 1.0 + logits.abs()
        return grad_output / denominator.square()


def _hard_sign_with_softsign_vjp(x: Tensor) -> Tensor:
    return _HardSignWithSoftsignVjp.apply(x)


def _causal_route_mask(seq_len: int, device: torch.device) -> Tensor:
    row = torch.arange(seq_len, device=device).view(seq_len, 1)
    route_index = torch.arange(seq_len, device=device).view(1, seq_len)
    return (route_index == 0) | (
        (route_index >= 1) & (route_index <= row)
    )


def _pairwise_exact_symbol_match(
    query: Tensor,
    key: Tensor,
    causal_route_mask: Tensor,
) -> Tensor:
    query_hard = _hard_sign(query.permute(0, 2, 1, 3))
    key_hard = _hard_sign(key.permute(0, 2, 1, 3)[..., :-1, :])
    seq_len = query_hard.size(-2)
    exact = (
        query_hard.unsqueeze(-2) == key_hard.unsqueeze(-3)
    ).all(dim=-1)
    return F.pad(exact, (1, 0), value=False) & causal_route_mask.view(
        1,
        1,
        seq_len,
        seq_len,
    )


def _pairwise_soft_match_gates(
    query: Tensor,
    key: Tensor,
    causal_route_mask: Tensor,
    mismatch_scale: float,
) -> Tensor:
    """Return deterministic local match values with their coherent VJP."""

    query_symbols = _hard_sign_with_softsign_vjp(
        query.permute(0, 2, 1, 3)
    )
    key_symbols = _hard_sign_with_softsign_vjp(
        key.permute(0, 2, 1, 3)[..., :-1, :]
    )
    mismatch_rate = 0.5 * (
        1.0
        - query_symbols.unsqueeze(-2) * key_symbols.unsqueeze(-3)
    ).mean(dim=-1)
    local_match_gates = torch.exp(-float(mismatch_scale) * mismatch_rate)
    seq_len = query.size(1)
    return F.pad(
        local_match_gates,
        (1, 0),
        value=0.0,
    ) * causal_route_mask.view(
        1,
        1,
        seq_len,
        seq_len,
    )


def _suffix_prefix_product_scores(
    local_match: Tensor,
    max_suffix_length: int,
) -> Tensor:
    product = local_match
    score = product
    max_offsets = min(
        int(max_suffix_length),
        local_match.size(-2),
        local_match.size(-1),
    )
    for _ in range(1, max_offsets):
        previous = F.pad(
            product[..., :-1, :-1],
            (1, 0, 1, 0),
            value=0.0,
        )
        product = local_match * previous
        score = score + product
    return score


def _suffix_score_utility(raw_suffix_scores: Tensor) -> Tensor:
    """Compress long evidence while preserving zero and one-match calibration."""

    return _NORMALIZED_SQRT_SUFFIX_SCALE * (
        torch.sqrt(1.0 + raw_suffix_scores) - 1.0
    )


def _select_latest_longest_routes(
    exact_suffix_lengths: Tensor,
    causal_route_mask: Tensor,
) -> Tensor:
    seq_len = exact_suffix_lengths.size(-1)
    route_index = torch.arange(
        seq_len,
        device=exact_suffix_lengths.device,
    ).view(
        1,
        1,
        1,
        seq_len,
    )
    nonnull_routes = causal_route_mask.view(
        1,
        1,
        seq_len,
        seq_len,
    ) & (route_index > 0)
    max_length = exact_suffix_lengths.amax(dim=-1, keepdim=True)
    latest = torch.where(
        nonnull_routes & (exact_suffix_lengths == max_length),
        route_index,
        torch.zeros(
            (),
            dtype=route_index.dtype,
            device=route_index.device,
        ),
    ).amax(dim=-1)
    return torch.where(
        max_length.squeeze(-1) > 0,
        latest,
        torch.zeros_like(latest),
    )


def _masked_route_scores(
    route_scores: Tensor,
    causal_route_mask: Tensor,
) -> Tensor:
    seq_len = route_scores.size(-1)
    scores = route_scores.clone()
    scores[..., 0] = ROSA_SOFT_NULL_ROUTE_SCORE
    return scores.masked_fill(
        ~causal_route_mask.view(1, 1, seq_len, seq_len),
        -torch.inf,
    )


def _route_probabilities(
    route_scores: Tensor,
    causal_route_mask: Tensor,
    scale: float,
) -> Tensor:
    seq_len = route_scores.size(-1)
    route_index = torch.arange(
        seq_len,
        device=route_scores.device,
    ).view(1, 1, 1, seq_len)
    nonnull = causal_route_mask.view(
        1,
        1,
        seq_len,
        seq_len,
    ) & (route_index > 0)
    nonnull_count = nonnull.sum(dim=-1, keepdim=True).clamp_min(1)
    logits = route_scores * float(scale)
    logits = logits - torch.where(
        nonnull,
        nonnull_count.to(route_scores.dtype).log(),
        torch.zeros((), dtype=route_scores.dtype, device=route_scores.device),
    )
    centered = logits - logits.amax(dim=-1, keepdim=True)
    return torch.softmax(centered, dim=-1)


def _expand_value_heads(value: Tensor, query_heads: int) -> Tensor:
    groups = query_heads // value.size(2)
    return value.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)


def _gather_routed_values(
    route_values: Tensor,
    selected_route_indices: Tensor,
) -> Tensor:
    value_dim = route_values.size(-1)
    gather_indices = selected_route_indices.unsqueeze(-1).expand(
        *selected_route_indices.shape,
        value_dim,
    )
    return torch.gather(
        route_values,
        dim=2,
        index=gather_indices,
    )


def _hard_route_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    causal_route_mask = _causal_route_mask(
        query.size(1),
        query.device,
    )
    exact_local = _pairwise_exact_symbol_match(
        query,
        key,
        causal_route_mask,
    )
    exact_suffix_lengths = _suffix_prefix_product_scores(
        exact_local.to(query.dtype),
        max_suffix_length,
    )
    selected_route_indices = _select_latest_longest_routes(
        exact_suffix_lengths,
        causal_route_mask,
    )

    route_values = _expand_value_heads(
        _hard_sign(value),
        query.size(2),
    )
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


def _build_vjp_carrier(
    value: Tensor,
    route_probabilities: Tensor,
    query_heads: int,
) -> Tensor:
    route_values_with_softsign_vjp = _expand_value_heads(
        _hard_sign_with_softsign_vjp(value),
        query_heads,
    )
    nonnull = torch.arange(
        value.size(1),
        device=value.device,
    ).view(1, 1, -1, 1) != 0
    route_values_with_softsign_vjp = torch.where(
        nonnull,
        route_values_with_softsign_vjp,
        torch.zeros_like(route_values_with_softsign_vjp),
    )

    return torch.einsum(
        "bhta,bhad->bhtd",
        route_probabilities,
        route_values_with_softsign_vjp,
    ).permute(0, 2, 1, 3)


_DROPOUT_HASH_MASK = (1 << 32) - 1
_DROPOUT_HASH_MULTIPLIERS = (0x7FEB352D, 0x846CA68B)
_DROPOUT_COORDINATE_SALTS = (
    0xA511E9B3,
    0x63D83595,
    0xB5297A4D,
    0x68E31DA4,
)


def _multiply_u32(state: Tensor, multiplier: int) -> Tensor:
    """Multiply modulo 2^32 without overflowing signed int64 tensors."""

    low = state & 0xFFFF
    high = state >> 16
    return (
        low * multiplier
        + ((high * multiplier & 0xFFFF) << 16)
    ) & _DROPOUT_HASH_MASK


def _hash_dropout_counter(state: Tensor) -> Tensor:
    """Avalanche one uint32 counter while matching CUDA wraparound."""

    state = state & _DROPOUT_HASH_MASK
    state = state ^ (state >> 16)
    state = _multiply_u32(state, _DROPOUT_HASH_MULTIPLIERS[0])
    state = state ^ (state >> 15)
    state = _multiply_u32(state, _DROPOUT_HASH_MULTIPLIERS[1])
    return (state ^ (state >> 16)) & _DROPOUT_HASH_MASK


def _apply_attention_dropout(
    attention_weights: Tensor,
    dropout_p: float,
    dropout_seed: Tensor,
    batch_offset: int,
) -> Tensor:
    if float(dropout_p) == 0.0:
        return attention_weights

    batch, heads, queries, routes = attention_weights.shape
    coordinates = (
        torch.arange(
            batch,
            device=attention_weights.device,
            dtype=torch.int64,
        ).view(batch, 1, 1, 1)
        + int(batch_offset),
        torch.arange(
            heads,
            device=attention_weights.device,
            dtype=torch.int64,
        ).view(1, heads, 1, 1),
        torch.arange(
            queries,
            device=attention_weights.device,
            dtype=torch.int64,
        ).view(1, 1, queries, 1),
        torch.arange(
            routes,
            device=attention_weights.device,
            dtype=torch.int64,
        ).view(1, 1, 1, routes),
    )
    state = _hash_dropout_counter(
        coordinates[-1] ^ _DROPOUT_COORDINATE_SALTS[-1]
    )
    for coordinate, salt in zip(
        reversed(coordinates[:-1]),
        reversed(_DROPOUT_COORDINATE_SALTS[:-1]),
    ):
        state = _hash_dropout_counter(state ^ coordinate ^ salt)
    seed_low = dropout_seed & _DROPOUT_HASH_MASK
    seed_high = (dropout_seed >> 32) & _DROPOUT_HASH_MASK
    state = _hash_dropout_counter(state ^ seed_low)
    state = _hash_dropout_counter(
        state ^ seed_high ^ 0x9E3779B9
    )
    uniform = (state >> 8).to(torch.float32) * (1.0 / (1 << 24))
    keep = uniform >= float(dropout_p)
    return (
        attention_weights
        * keep.to(attention_weights.dtype)
        / (1.0 - float(dropout_p))
    )


class _HardForwardSoftVjpReference(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_seed: Tensor,
        max_suffix_length: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        dropout_batch_offset: int,
    ) -> Tensor:
        reference_compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(reference_compute_dtype),
            key.to(reference_compute_dtype),
            value.to(reference_compute_dtype),
            int(max_suffix_length),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.mismatch_scale = float(mismatch_scale)
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
                for tensor, need in zip(
                    (query, key, value),
                    needs,
                )
            )
            query_leaf, key_leaf, value_leaf = leaves
            reference_compute_dtype = _reference_compute_dtype(
                query.dtype
            )
            query_work = query_leaf.to(reference_compute_dtype)
            key_work = key_leaf.to(reference_compute_dtype)
            value_work = value_leaf.to(reference_compute_dtype)
            causal_route_mask = _causal_route_mask(
                query.size(1),
                query.device,
            )
            local_match_gates = _pairwise_soft_match_gates(
                query_work,
                key_work,
                causal_route_mask,
                ctx.mismatch_scale,
            )
            soft_suffix_scores = _suffix_prefix_product_scores(
                local_match_gates,
                ctx.max_suffix_length,
            )
            route_scores = _masked_route_scores(
                _suffix_score_utility(soft_suffix_scores),
                causal_route_mask,
            )
            attention_weights = _route_probabilities(
                route_scores,
                causal_route_mask,
                ctx.scale,
            )
            attention_weights = _apply_attention_dropout(
                attention_weights,
                ctx.dropout_p,
                dropout_seed,
                ctx.dropout_batch_offset,
            )
            vjp_carrier = _build_vjp_carrier(
                value_work,
                attention_weights,
                query.size(2),
            ).to(query.dtype)
            required_indices = [
                index for index, need in enumerate(needs) if need
            ]
            required_grads = torch.autograd.grad(
                vjp_carrier,
                tuple(leaves[index] for index in required_indices),
                grad_output,
                create_graph=False,
            )
            grads = [None, None, None]
            for index, gradient in zip(required_indices, required_grads):
                grads[index] = gradient

        return (
            grads[0],
            grads[1],
            grads[2],
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _rosa_soft_reference_with_seed(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_seed: Tensor,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    dropout_batch_offset: int,
) -> Tensor:
    if not torch.is_grad_enabled() or not any(
        tensor.requires_grad
        for tensor in (query, key, value)
    ):
        reference_compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(reference_compute_dtype),
            key.to(reference_compute_dtype),
            value.to(reference_compute_dtype),
            max_suffix_length,
        )
        return hard_output.to(query.dtype)

    return _HardForwardSoftVjpReference.apply(
        query,
        key,
        value,
        dropout_seed,
        max_suffix_length,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
        int(dropout_batch_offset),
    )


def rosa_soft_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    """Return exact hard ROSA values with a dense attention-style VJP."""

    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    dropout_seed = make_dropout_seed(
        query,
        dropout_p,
        needs_backward,
    )
    return _rosa_soft_reference_with_seed(
        query,
        key,
        value,
        dropout_seed,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
        0,
    )


def rosa_soft_varlen_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    """Apply the reference operator independently to packed sequences."""

    max_suffix_length = validate_rosa_soft_varlen_inputs(
        query,
        key,
        value,
        cu_seqlens,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    validate_fp32_surrogate_scalars(
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    offsets = cu_seqlens.detach().cpu().tolist()
    if offsets[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if offsets[-1] != query.size(0):
        raise ValueError("cu_seqlens must end at the packed token count")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    dropout_seed = make_dropout_seed(
        query,
        dropout_p,
        needs_backward,
    )
    outputs = []
    for sequence, (start, end) in enumerate(
        zip(offsets[:-1], offsets[1:])
    ):
        if start == end:
            continue
        outputs.append(
            _rosa_soft_reference_with_seed(
                query[start:end].unsqueeze(0),
                key[start:end].unsqueeze(0),
                value[start:end].unsqueeze(0),
                dropout_seed,
                max_suffix_length,
                scale,
                dropout_p,
                mismatch_scale,
                sequence,
            ).squeeze(0)
        )
    return torch.cat(outputs, dim=0)
