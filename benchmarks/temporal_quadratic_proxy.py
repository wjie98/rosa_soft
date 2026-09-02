"""Parameter-free temporal-quadratic VJP for exact hard ROSA.

The numerical forward is the exact longest/latest ROSA route.  Backward first
encodes the last ``W`` hard Q/K symbols with a fixed orthogonal suffix scan,
then applies a complete degree-two polynomial feature map and an additive
fast-weight read.  The proxy has no trainable parameters or auxiliary loss.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.fast_weight_proxy import (  # noqa: E402
    _linear_attention_carrier,
)
from rosa_soft.soft_reference import (  # noqa: E402
    _expand_value_heads,
    _hard_route_forward,
    _hard_sign_with_softsign_vjp,
    _reference_compute_dtype,
    _validate_reference_call,
)


DEFAULT_STATE_DIM = 64
_MAX_STATE_DIM = 128
_SQRT_TWO = math.sqrt(2.0)
_NORMALIZATION_EPS = 1e-6


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _validate_state_dim(state_dim: int, symbol_bits: int) -> int:
    if isinstance(state_dim, bool) or not isinstance(state_dim, int):
        raise TypeError("state_dim must be an integer")
    if not _is_power_of_two(state_dim):
        raise ValueError("state_dim must be a power of two")
    if state_dim < max(2, symbol_bits):
        raise ValueError("state_dim must be at least max(2, Q/K bits)")
    if state_dim > _MAX_STATE_DIM:
        raise ValueError(f"state_dim must not exceed {_MAX_STATE_DIM}")
    return state_dim


def _next_prime(value: int) -> int:
    candidate = max(2, int(value))
    while True:
        limit = math.isqrt(candidate)
        if all(candidate % divisor for divisor in range(2, limit + 1)):
            return candidate
        candidate += 1


def _hadamard(order: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    matrix = torch.ones(1, 1, dtype=dtype, device=device)
    while matrix.size(0) < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(order)


def _fixed_projection(
    symbol_bits: int,
    state_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return deterministic orthonormal symbol directions ``[R, D]``."""

    return _hadamard(state_dim, dtype=dtype, device=device)[:, :symbol_bits]


def _rotation_angles(
    state_dim: int,
    max_suffix_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return distinct fixed frequencies for the orthogonal state pairs."""

    modulus = _next_prime(2 * max_suffix_length + 2 * state_dim + 1)
    frequencies = 2 * torch.arange(
        state_dim // 2,
        dtype=dtype,
        device=device,
    ) + 1
    return frequencies * (2.0 * math.pi / modulus)


def _rotate_pairs(state: Tensor, angles: Tensor) -> Tensor:
    even = state[..., 0::2]
    odd = state[..., 1::2]
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    return torch.stack(
        (
            cosine * even - sine * odd,
            sine * even + cosine * odd,
        ),
        dim=-1,
    ).flatten(-2)


def _temporal_suffix_states(
    logits: Tensor,
    max_suffix_length: int,
    state_dim: int,
) -> Tensor:
    """Encode every causal suffix with a finite orthogonal local scan.

    Input follows the public ``[B, T, H, D]`` layout.  Output is
    ``[B, H, T, R]``.  Numerical states depend only on hard signs, while the
    Q/K VJP uses the production softsign derivative.
    """

    symbol_bits = logits.size(-1)
    state_dim = _validate_state_dim(state_dim, symbol_bits)
    symbols = _hard_sign_with_softsign_vjp(logits.permute(0, 2, 1, 3))
    projection = _fixed_projection(
        symbol_bits,
        state_dim,
        dtype=symbols.dtype,
        device=symbols.device,
    )
    projected = torch.einsum("bhtd,rd->bhtr", symbols, projection)
    angles = _rotation_angles(
        state_dim,
        max_suffix_length,
        dtype=symbols.dtype,
        device=symbols.device,
    )
    states = torch.zeros_like(projected)
    sequence_length = projected.size(2)
    for offset in range(min(max_suffix_length, sequence_length)):
        contribution = _rotate_pairs(
            projected[:, :, : sequence_length - offset],
            angles * offset,
        )
        states[:, :, offset:] = states[:, :, offset:] + contribution
    return states / math.sqrt(max_suffix_length * symbol_bits)


def _quadratic_polynomial_features(state: Tensor) -> Tensor:
    """Map ``u`` so inner products equal ``dot(u, v)**2``."""

    dimension = state.size(-1)
    rows, columns = torch.triu_indices(
        dimension,
        dimension,
        offset=1,
        device=state.device,
    )
    cross = _SQRT_TWO * state[..., rows] * state[..., columns]
    return torch.cat(
        (
            state.square(),
            cross,
        ),
        dim=-1,
    )


def _normalize_temporal_state(state: Tensor) -> Tensor:
    squared_norm = state.square().sum(dim=-1, keepdim=True)
    valid = squared_norm > _NORMALIZATION_EPS**2
    normalized = state * squared_norm.clamp_min(
        _NORMALIZATION_EPS**2
    ).rsqrt()
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def _temporal_quadratic_fingerprints(
    logits: Tensor,
    max_suffix_length: int,
    state_dim: int,
) -> Tensor:
    states = _temporal_suffix_states(
        logits,
        max_suffix_length,
        state_dim,
    )
    return _quadratic_polynomial_features(
        _normalize_temporal_state(states)
    )


def _temporal_quadratic_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int,
    state_dim: int,
) -> Tensor:
    query_features = _temporal_quadratic_fingerprints(
        query,
        max_suffix_length,
        state_dim,
    )
    key_features = _temporal_quadratic_fingerprints(
        key,
        max_suffix_length,
        state_dim,
    )
    values = _expand_value_heads(
        _hard_sign_with_softsign_vjp(value),
        query.size(2),
    )
    carrier = _linear_attention_carrier(
        query_features,
        key_features,
        values,
    )
    dependency = (
        query_features.sum() + key_features.sum() + values.sum()
    ) * 0.0
    return carrier + dependency


class _HardForwardTemporalQuadraticVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        max_suffix_length: int,
        state_dim: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.state_dim = int(state_dim)
        ctx.save_for_backward(query, key, value)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            leaves = tuple(
                tensor.detach().requires_grad_(need)
                for tensor, need in zip((query, key, value), needs)
            )
            compute_dtype = _reference_compute_dtype(query.dtype)
            carrier = _temporal_quadratic_carrier(
                *(tensor.to(compute_dtype) for tensor in leaves),
                max_suffix_length=ctx.max_suffix_length,
                state_dim=ctx.state_dim,
            ).to(query.dtype)
            required_indices = [
                index for index, need in enumerate(needs) if need
            ]
            required_gradients = torch.autograd.grad(
                carrier,
                tuple(leaves[index] for index in required_indices),
                grad_output,
                create_graph=False,
            )
        gradients = [None, None, None]
        for index, gradient in zip(required_indices, required_gradients):
            gradients[index] = gradient
        return gradients[0], gradients[1], gradients[2], None, None


def rosa_temporal_quadratic_proxy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int = 32,
    state_dim: int = DEFAULT_STATE_DIM,
) -> Tensor:
    """Return exact hard ROSA values with a temporal-quadratic VJP."""

    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        1.0,
        0.0,
        1.0,
    )
    state_dim = _validate_state_dim(state_dim, query.size(-1))
    return _HardForwardTemporalQuadraticVjp.apply(
        query,
        key,
        value,
        max_suffix_length,
        state_dim,
    )


__all__ = [
    "DEFAULT_STATE_DIM",
    "rosa_temporal_quadratic_proxy",
]
