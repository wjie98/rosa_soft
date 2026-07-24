"""Hard-forward/soft-backward PyTorch oracle for RosaSoft.

The public training operator exposes only a suffix horizon, route
temperature, and mismatch penalty. Stochastic mismatch exploration, the
hard-Hamming local VJP, and soft payload routing are fixed parts of the
estimator. Payload credit follows the dense route distribution only in
backward.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from .soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    ROSA_SOFT_NULL_ROUTE_SCORE,
    validate_rosa_soft_inputs,
)

__all__ = [
    "ROSA_SOFT_DEFAULT_MISMATCH_PENALTY",
    "ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE",
    "rosa_soft_reference",
]


def _reference_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


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


def _hard_sign_with_softsign_vjp(x: Tensor) -> Tuple[Tensor, Tensor]:
    surrogate = _HardSignWithSoftsignVjp.apply(x)
    return surrogate.detach(), surrogate


def _causal_route_mask(seq_len: int, device: torch.device) -> Tensor:
    row = torch.arange(seq_len, device=device).view(seq_len, 1)
    route_index = torch.arange(seq_len, device=device).view(1, seq_len)
    return (route_index == 0) | (
        (route_index >= 1) & (route_index <= row)
    )


def _pairwise_exact_symbol_match(
    query: Tensor,
    key: Tensor,
    route_mask: Tensor,
) -> Tensor:
    query_hard = _hard_sign(query.permute(0, 2, 1, 3))
    key_hard = _hard_sign(key.permute(0, 2, 1, 3)[..., :-1, :])
    batch, heads, seq_len, bits = query_hard.shape
    exact = torch.ones(
        (batch, heads, seq_len, max(seq_len - 1, 0)),
        dtype=torch.bool,
        device=query.device,
    )
    for bit in range(bits):
        q = query_hard[..., bit].unsqueeze(-1)
        k = key_hard[..., bit].unsqueeze(-2)
        exact &= q == k
    return F.pad(exact, (1, 0), value=False) & route_mask.view(
        1,
        1,
        seq_len,
        seq_len,
    )


def _local_match_gate_with_hard_vjp(
    hard_hamming: Tensor,
    relaxed_hamming: Tensor,
    surrogate_hamming: Tensor,
    mismatch_penalty: float,
) -> Tensor:
    """Return a relaxed gate whose local VJP follows hard Hamming distance."""

    mismatch_penalty = float(mismatch_penalty)
    hard_jacobian_scale = torch.exp(
        -mismatch_penalty * (hard_hamming - relaxed_hamming)
    ).detach()
    proxy = surrogate_hamming * hard_jacobian_scale
    energy = relaxed_hamming.detach() + proxy - proxy.detach()
    return torch.exp(-mismatch_penalty * energy)


def _pairwise_stochastic_match_gates(
    query: Tensor,
    key: Tensor,
    route_mask: Tensor,
    mismatch_penalty: float,
    mismatch_uniforms: Tensor,
) -> Tensor:
    """Build fixed cubic local gates and their hard-Hamming VJP."""

    query_h = query.permute(0, 2, 1, 3)
    key_h = key.permute(0, 2, 1, 3)[..., :-1, :]
    query_hard, query_surrogate = _hard_sign_with_softsign_vjp(query_h)
    key_hard, key_surrogate = _hard_sign_with_softsign_vjp(key_h)
    batch, heads, seq_len, bits = query_h.shape
    pair_shape = (batch, heads, seq_len, max(seq_len - 1, 0))
    hard_hamming = torch.zeros(pair_shape, dtype=query.dtype, device=query.device)
    relaxed_hamming = torch.zeros_like(hard_hamming)
    surrogate_hamming = torch.zeros_like(hard_hamming)

    for bit in range(bits):
        q_hard = query_hard[..., bit].unsqueeze(-1)
        k_hard = key_hard[..., bit].unsqueeze(-2)
        hard_mismatch = 0.5 * (1.0 - q_hard * k_hard)
        uniform = mismatch_uniforms[..., bit]
        mismatch_weight = 1.0 - 0.5 * uniform.pow(3)

        hard_hamming += hard_mismatch
        relaxed_hamming += hard_mismatch * mismatch_weight

        query_bit_surrogate = query_surrogate[..., bit].unsqueeze(-1)
        key_bit_surrogate = key_surrogate[..., bit].unsqueeze(-2)
        surrogate_hamming += 0.5 * (
            1.0 - query_bit_surrogate * key_bit_surrogate
        )

    local_proxy = _local_match_gate_with_hard_vjp(
        hard_hamming,
        relaxed_hamming,
        surrogate_hamming,
        mismatch_penalty,
    )
    return F.pad(local_proxy, (1, 0), value=0.0) * route_mask.view(
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
    max_offsets = min(int(max_suffix_length), local_match.size(-2), local_match.size(-1))
    for _ in range(1, max_offsets):
        previous = F.pad(product[..., :-1, :-1], (1, 0, 1, 0), value=0.0)
        product = local_match * previous
        score = score + product
    return score


def _select_latest_longest_routes(
    exact_suffix_lengths: Tensor,
    route_mask: Tensor,
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
    nonnull_routes = route_mask.view(1, 1, seq_len, seq_len) & (
        route_index > 0
    )
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
    proxy_scores: Tensor,
    route_mask: Tensor,
) -> Tensor:
    seq_len = proxy_scores.size(-1)
    scores = proxy_scores.clone()
    scores[..., 0] = ROSA_SOFT_NULL_ROUTE_SCORE
    return scores.masked_fill(
        ~route_mask.view(1, 1, seq_len, seq_len),
        -torch.inf,
    )


def _route_probabilities(
    route_scores: Tensor,
    route_temperature: float,
) -> Tensor:
    centered = route_scores - route_scores.amax(dim=-1, keepdim=True)
    return torch.softmax(centered / float(route_temperature), dim=-1)


def _expand_payload_heads(payload: Tensor, query_heads: int) -> Tensor:
    groups = query_heads // payload.size(2)
    return payload.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)


def _gather_routed_payloads(
    route_payloads: Tensor,
    route_indices: Tensor,
) -> Tensor:
    payload_dim = route_payloads.size(-1)
    indices = route_indices.unsqueeze(-1).expand(
        *route_indices.shape,
        payload_dim,
    )
    return torch.gather(route_payloads, dim=2, index=indices)


def _hard_route_forward(
    query: Tensor,
    key: Tensor,
    payload: Tensor,
    max_suffix_length: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    route_mask = _causal_route_mask(query.size(1), query.device)
    exact_local = _pairwise_exact_symbol_match(query, key, route_mask)
    exact_suffix_lengths = _suffix_prefix_product_scores(
        exact_local.to(query.dtype),
        max_suffix_length,
    )
    selected_routes = _select_latest_longest_routes(
        exact_suffix_lengths,
        route_mask,
    )

    route_payloads = _expand_payload_heads(
        _hard_sign(payload),
        query.size(2),
    )
    route_payloads[..., 0, :] = 0.0
    hard_output = _gather_routed_payloads(
        route_payloads,
        selected_routes,
    ).permute(0, 2, 1, 3)
    return hard_output, exact_suffix_lengths, selected_routes, route_mask


def _build_vjp_surrogate(
    payload: Tensor,
    probabilities: Tensor,
    query_heads: int,
) -> Tensor:
    payload_hard, payload_surrogate = _hard_sign_with_softsign_vjp(
        payload
    )
    route_payload_hard = _expand_payload_heads(
        payload_hard,
        query_heads,
    )
    route_payload_surrogate = _expand_payload_heads(
        payload_surrogate,
        query_heads,
    )
    nonnull = torch.arange(
        payload.size(1),
        device=payload.device,
    ).view(1, 1, -1, 1) != 0
    route_payload_hard = torch.where(
        nonnull,
        route_payload_hard,
        torch.zeros_like(route_payload_hard),
    )
    route_payload_surrogate = torch.where(
        nonnull,
        route_payload_surrogate,
        torch.zeros_like(route_payload_surrogate),
    )

    route = torch.einsum(
        "bhta,bhad->bhtd",
        probabilities,
        route_payload_hard.detach(),
    )
    payload_path = torch.einsum(
        "bhta,bhad->bhtd",
        probabilities.detach(),
        route_payload_surrogate,
    )
    return (route + payload_path).permute(0, 2, 1, 3)


class _HardForwardSoftVjpReference(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        payload: Tensor,
        mismatch_uniforms: Tensor,
        max_suffix_length: int,
        route_temperature: float,
        mismatch_penalty: float,
    ) -> Tensor:
        reference_dtype = _reference_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(reference_dtype),
            key.to(reference_dtype),
            payload.to(reference_dtype),
            int(max_suffix_length),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.route_temperature = float(route_temperature)
        ctx.mismatch_penalty = float(mismatch_penalty)
        ctx.save_for_backward(query, key, payload, mismatch_uniforms)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, payload, mismatch_uniforms = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            query_leaf = query.detach().requires_grad_(True)
            key_leaf = key.detach().requires_grad_(True)
            payload_leaf = payload.detach().requires_grad_(True)
            reference_dtype = _reference_dtype(query.dtype)
            query_work = query_leaf.to(reference_dtype)
            key_work = key_leaf.to(reference_dtype)
            payload_work = payload_leaf.to(reference_dtype)
            route_mask = _causal_route_mask(
                query.size(1),
                query.device,
            )
            local_proxy = _pairwise_stochastic_match_gates(
                query_work,
                key_work,
                route_mask,
                ctx.mismatch_penalty,
                mismatch_uniforms,
            )
            proxy_scores = _suffix_prefix_product_scores(
                local_proxy,
                ctx.max_suffix_length,
            )
            probabilities = _route_probabilities(
                _masked_route_scores(proxy_scores, route_mask),
                ctx.route_temperature,
            )
            surrogate = _build_vjp_surrogate(
                payload_work,
                probabilities,
                query.size(2),
            ).to(query.dtype)
            grads = torch.autograd.grad(
                surrogate,
                (query_leaf, key_leaf, payload_leaf),
                grad_output,
                create_graph=False,
                allow_unused=True,
            )

        return (
            grads[0] if needs[0] else None,
            grads[1] if needs[1] else None,
            grads[2] if needs[2] else None,
            None,
            None,
            None,
            None,
        )


def _mismatch_uniform_shape(
    query: Tensor,
) -> Tuple[int, int, int, int, int]:
    batch, seq_len, heads, bits = query.shape
    return batch, heads, seq_len, max(seq_len - 1, 0), bits


def _sample_mismatch_uniforms(
    query: Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    return torch.rand(
        _mismatch_uniform_shape(query),
        device=query.device,
        dtype=_reference_dtype(query.dtype),
        generator=generator,
    )


def _rosa_soft_reference_with_uniforms(
    query: Tensor,
    key: Tensor,
    payload: Tensor,
    mismatch_uniforms: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
) -> Tensor:
    """Private deterministic entry point used by CUDA parity tests."""

    expected_shape = _mismatch_uniform_shape(query)
    if tuple(mismatch_uniforms.shape) != expected_shape:
        raise ValueError(
            f"mismatch_uniforms must have shape {expected_shape}, "
            f"got {tuple(mismatch_uniforms.shape)}"
        )
    return _HardForwardSoftVjpReference.apply(
        query,
        key,
        payload,
        mismatch_uniforms,
        int(max_suffix_length),
        float(route_temperature),
        float(mismatch_penalty),
    )


def rosa_soft_reference(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    max_suffix_length: int = 32,
    route_temperature: float = ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    mismatch_penalty: float = ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
) -> Tensor:
    """Return exact hard ROSA values with a fixed stochastic surrogate VJP."""

    max_suffix_length = validate_rosa_soft_inputs(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        float(route_temperature),
        float(mismatch_penalty),
    )
    if not torch.is_grad_enabled() or not any(
        tensor.requires_grad
        for tensor in (query_logits, key_logits, payload_logits)
    ):
        reference_dtype = _reference_dtype(query_logits.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query_logits.to(reference_dtype),
            key_logits.to(reference_dtype),
            payload_logits.to(reference_dtype),
            max_suffix_length,
        )
        return hard_output.to(query_logits.dtype)

    return _rosa_soft_reference_with_uniforms(
        query_logits,
        key_logits,
        payload_logits,
        _sample_mismatch_uniforms(query_logits),
        max_suffix_length,
        float(route_temperature),
        float(mismatch_penalty),
    )
