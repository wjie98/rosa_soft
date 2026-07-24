"""Hard-forward/soft-backward PyTorch oracle for RosaSoft.

The public training operator exposes only a suffix horizon, route
temperature, and mismatch penalty. Stochastic mismatch exploration, Jacobian
anchoring, and soft payload routing are fixed parts of the estimator. Payload
credit follows the dense route distribution only in backward.
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
    validate_soft_inputs,
)

__all__ = [
    "ROSA_SOFT_DEFAULT_MISMATCH_PENALTY",
    "ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE",
    "rosa_soft_reference",
]


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _hard_sign(x: Tensor) -> Tensor:
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))


class _HardSignSoftsignJacobian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor) -> Tensor:
        ctx.save_for_backward(logits)
        return _hard_sign(logits)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (logits,) = ctx.saved_tensors
        denominator = 1.0 + logits.abs()
        return grad_output / denominator.square()


def _sign_with_softsign_jacobian(x: Tensor) -> Tuple[Tensor, Tensor]:
    hard = _hard_sign(x)
    return hard, _HardSignSoftsignJacobian.apply(x)


def _action_mask(seq_len: int, device: torch.device) -> Tensor:
    row = torch.arange(seq_len, device=device).view(seq_len, 1)
    action = torch.arange(seq_len, device=device).view(1, seq_len)
    return (action == 0) | ((action >= 1) & (action <= row))


def _pairwise_exact_local_match(
    query: Tensor,
    key: Tensor,
    action_mask: Tensor,
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
    return F.pad(exact, (1, 0), value=False) & action_mask.view(
        1,
        1,
        seq_len,
        seq_len,
    )


def _relaxed_gate_with_hard_jacobian(
    hard_hamming: Tensor,
    relaxed_hamming: Tensor,
    surrogate_hamming: Tensor,
    mismatch_penalty: float,
) -> Tensor:
    """Return a relaxed value with the hard-Hamming local Jacobian scale."""

    mismatch_penalty = float(mismatch_penalty)
    hard_jacobian_scale = torch.exp(
        -mismatch_penalty * (hard_hamming - relaxed_hamming)
    ).detach()
    proxy = surrogate_hamming * hard_jacobian_scale
    energy = relaxed_hamming.detach() + proxy - proxy.detach()
    return torch.exp(-mismatch_penalty * energy)


def _pairwise_proxy_local_match(
    query: Tensor,
    key: Tensor,
    action_mask: Tensor,
    mismatch_penalty: float,
    mismatch_noise: Tensor,
) -> Tensor:
    """Build fixed cubic local gates and their anchored VJP."""

    query_h = query.permute(0, 2, 1, 3)
    key_h = key.permute(0, 2, 1, 3)[..., :-1, :]
    query_hard, query_ste = _sign_with_softsign_jacobian(query_h)
    key_hard, key_ste = _sign_with_softsign_jacobian(key_h)
    batch, heads, seq_len, bits = query_h.shape
    pair_shape = (batch, heads, seq_len, max(seq_len - 1, 0))
    hard_hamming = torch.zeros(pair_shape, dtype=query.dtype, device=query.device)
    relaxed_a = torch.zeros_like(hard_hamming)
    surrogate_hamming = torch.zeros_like(hard_hamming)

    for bit in range(bits):
        q_hard = query_hard[..., bit].unsqueeze(-1)
        k_hard = key_hard[..., bit].unsqueeze(-2)
        hard_mismatch = 0.5 * (1.0 - q_hard * k_hard)
        uniform = mismatch_noise[..., bit]
        alpha_a = 1.0 - 0.5 * uniform.pow(3)

        hard_hamming += hard_mismatch
        relaxed_a += hard_mismatch * alpha_a

        q_ste = query_ste[..., bit].unsqueeze(-1)
        k_ste = key_ste[..., bit].unsqueeze(-2)
        surrogate_hamming += 0.5 * (1.0 - q_ste * k_ste)

    local_proxy = _relaxed_gate_with_hard_jacobian(
        hard_hamming,
        relaxed_a,
        surrogate_hamming,
        mismatch_penalty,
    )
    return F.pad(local_proxy, (1, 0), value=0.0) * action_mask.view(
        1,
        1,
        seq_len,
        seq_len,
    )


def _diagonal_suffix_sum(local_match: Tensor, max_suffix_length: int) -> Tensor:
    product = local_match
    score = product
    max_offsets = min(int(max_suffix_length), local_match.size(-2), local_match.size(-1))
    for _ in range(1, max_offsets):
        previous = F.pad(product[..., :-1, :-1], (1, 0, 1, 0), value=0.0)
        product = local_match * previous
        score = score + product
    return score


def _select_hard_actions(hard_lengths: Tensor, action_mask: Tensor) -> Tensor:
    seq_len = hard_lengths.size(-1)
    action = torch.arange(seq_len, device=hard_lengths.device).view(
        1,
        1,
        1,
        seq_len,
    )
    candidates = action_mask.view(1, 1, seq_len, seq_len) & (action > 0)
    max_length = hard_lengths.amax(dim=-1, keepdim=True)
    latest = torch.where(
        candidates & (hard_lengths == max_length),
        action,
        torch.zeros((), dtype=action.dtype, device=action.device),
    ).amax(dim=-1)
    return torch.where(
        max_length.squeeze(-1) > 0,
        latest,
        torch.zeros_like(latest),
    )


def _allocation_scores(candidate_scores: Tensor, action_mask: Tensor) -> Tensor:
    seq_len = candidate_scores.size(-1)
    scores = candidate_scores.clone()
    scores[..., 0] = ROSA_SOFT_NULL_ROUTE_SCORE
    return scores.masked_fill(
        ~action_mask.view(1, 1, seq_len, seq_len),
        -torch.inf,
    )


def _proxy_probabilities(allocation_scores: Tensor, route_temperature: float) -> Tensor:
    centered = allocation_scores - allocation_scores.amax(dim=-1, keepdim=True)
    return torch.softmax(centered / float(route_temperature), dim=-1)


def _expand_value_heads(value: Tensor, query_heads: int) -> Tensor:
    groups = query_heads // value.size(2)
    return value.repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)


def _gather_action_values(action_values: Tensor, actions: Tensor) -> Tensor:
    value_dim = action_values.size(-1)
    indices = actions.unsqueeze(-1).expand(*actions.shape, value_dim)
    return torch.gather(action_values, dim=2, index=indices)


def _hard_components(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    action_mask = _action_mask(query.size(1), query.device)
    exact_local = _pairwise_exact_local_match(query, key, action_mask)
    hard_lengths = _diagonal_suffix_sum(exact_local.to(query.dtype), max_suffix_length)
    hard_actions = _select_hard_actions(hard_lengths, action_mask)

    action_values = _expand_value_heads(_hard_sign(value), query.size(2))
    action_values[..., 0, :] = 0.0
    hard_output = _gather_action_values(action_values, hard_actions).permute(0, 2, 1, 3)
    return hard_output, hard_lengths, hard_actions, action_mask


def _surrogate_output(
    value: Tensor,
    probabilities: Tensor,
    query_heads: int,
) -> Tensor:
    value_hard, value_ste = _sign_with_softsign_jacobian(value)
    action_hard = _expand_value_heads(value_hard, query_heads)
    action_ste = _expand_value_heads(value_ste, query_heads)
    nonnull = torch.arange(value.size(1), device=value.device).view(1, 1, -1, 1) != 0
    action_hard = torch.where(nonnull, action_hard, torch.zeros_like(action_hard))
    action_ste = torch.where(nonnull, action_ste, torch.zeros_like(action_ste))

    route = torch.einsum(
        "bhta,bhad->bhtd",
        probabilities,
        action_hard.detach(),
    )
    value_path = torch.einsum(
        "bhta,bhad->bhtd",
        probabilities.detach(),
        action_ste,
    )
    return (route + value_path).permute(0, 2, 1, 3)


class _RosaSoftReferenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mismatch_noise: Tensor,
        max_suffix_length: int,
        route_temperature: float,
        mismatch_penalty: float,
    ) -> Tensor:
        work_dtype = _working_dtype(query.dtype)
        hard_output, _, _, _ = _hard_components(
            query.to(work_dtype),
            key.to(work_dtype),
            value.to(work_dtype),
            int(max_suffix_length),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.route_temperature = float(route_temperature)
        ctx.mismatch_penalty = float(mismatch_penalty)
        ctx.save_for_backward(query, key, value, mismatch_noise)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value, mismatch_noise = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            query_grad = query.detach().requires_grad_(True)
            key_grad = key.detach().requires_grad_(True)
            value_grad = value.detach().requires_grad_(True)
            work_dtype = _working_dtype(query.dtype)
            query_work = query_grad.to(work_dtype)
            key_work = key_grad.to(work_dtype)
            value_work = value_grad.to(work_dtype)
            action_mask = _action_mask(query.size(1), query.device)
            local_proxy = _pairwise_proxy_local_match(
                query_work,
                key_work,
                action_mask,
                ctx.mismatch_penalty,
                mismatch_noise,
            )
            proxy_scores = _diagonal_suffix_sum(local_proxy, ctx.max_suffix_length)
            probabilities = _proxy_probabilities(
                _allocation_scores(proxy_scores, action_mask),
                ctx.route_temperature,
            )
            surrogate = _surrogate_output(
                value_work,
                probabilities,
                query.size(2),
            ).to(query.dtype)
            grads = torch.autograd.grad(
                surrogate,
                (query_grad, key_grad, value_grad),
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


def _mismatch_shape(query: Tensor) -> Tuple[int, int, int, int, int]:
    batch, seq_len, heads, bits = query.shape
    return batch, heads, seq_len, max(seq_len - 1, 0), bits


def _sample_mismatch_noise(
    query: Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    return torch.rand(
        _mismatch_shape(query),
        device=query.device,
        dtype=_working_dtype(query.dtype),
        generator=generator,
    )


def _rosa_soft_reference_with_noise(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mismatch_noise: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
) -> Tensor:
    """Private deterministic entry point used by CUDA parity tests."""

    expected_shape = _mismatch_shape(query)
    if tuple(mismatch_noise.shape) != expected_shape:
        raise ValueError(
            f"mismatch_noise must have shape {expected_shape}, "
            f"got {tuple(mismatch_noise.shape)}"
        )
    return _RosaSoftReferenceFunction.apply(
        query,
        key,
        value,
        mismatch_noise,
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

    max_suffix_length = validate_soft_inputs(
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
        work_dtype = _working_dtype(query_logits.dtype)
        hard_output, _, _, _ = _hard_components(
            query_logits.to(work_dtype),
            key_logits.to(work_dtype),
            payload_logits.to(work_dtype),
            max_suffix_length,
        )
        return hard_output.to(query_logits.dtype)

    return _rosa_soft_reference_with_noise(
        query_logits,
        key_logits,
        payload_logits,
        _sample_mismatch_noise(query_logits),
        max_suffix_length,
        float(route_temperature),
        float(mismatch_penalty),
    )
