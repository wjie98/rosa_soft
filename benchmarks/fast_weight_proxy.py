"""Parameter-free fast-weight VJPs for hard ROSA research.

Every public proxy in this module returns the exact hard ROSA forward.  The
only difference is the backward carrier.  The carriers progress from local
linear attention to an order-sensitive suffix TensorSketch with a normalized
delta-rule memory.  They add runtime state, but no trainable parameters and no
auxiliary loss.
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
)
from rosa_soft.soft_reference import (  # noqa: E402
    _expand_value_heads,
    _hard_route_forward,
    _hard_sign_with_softsign_vjp,
    _reference_compute_dtype,
    _validate_reference_call,
)


PROXIES = (
    "linear_attention",
    "delta_rule",
    "state_linear_delta",
    "state_quadratic_delta",
    "state_cubic_delta",
    "state_linear_attention",
    "state_quadratic_attention",
    "state_cubic_attention",
    "single_suffix_delta",
    "single_suffix_sketch_delta",
    "exact_suffix_delta",
    "sketch_suffix_delta",
)

_PROXY_IDS = {name: index for index, name in enumerate(PROXIES)}
_PROXIES_BY_ID = {index: name for name, index in _PROXY_IDS.items()}
_STATE_PROXY_DEGREES = {
    "state_linear_delta": 1,
    "state_quadratic_delta": 2,
    "state_cubic_delta": 3,
    "state_linear_attention": 1,
    "state_quadratic_attention": 2,
    "state_cubic_attention": 3,
}
_LINEAR_ATTENTION_PROXIES = {
    "linear_attention",
    "state_linear_attention",
    "state_quadratic_attention",
    "state_cubic_attention",
}
_MAX_SYMBOL_BITS = 8
_MAX_EXACT_LEVEL_DIM = 1 << 18
_NORMALIZATION_EPS = 1e-6
_INTERACTION_INDEX_CACHE: dict[tuple[int, int, torch.device], Tensor] = {}


def _symbol_kernel_features(
    logits: Tensor,
    mismatch_scale: float,
) -> Tensor:
    """Map hard D-bit symbols to an exact exponential-Hamming feature map.

    The returned shape is ``[B, H, T, 2**D]``.  Numerical features depend
    only on hard signs; gradients use the production softsign VJP.
    """

    symbols = _hard_sign_with_softsign_vjp(logits.permute(0, 2, 1, 3))
    symbol_bits = symbols.size(-1)
    mismatch_factor = math.exp(-float(mismatch_scale) / symbol_bits)
    constant = math.sqrt(0.5 * (1.0 + mismatch_factor))
    signed = math.sqrt(0.5 * (1.0 - mismatch_factor))

    features = symbols.new_ones(*symbols.shape[:-1], 1)
    for bit in range(symbol_bits):
        bit_values = symbols[..., bit : bit + 1]
        features = torch.cat(
            (features * constant, features * (signed * bit_values)),
            dim=-1,
        )
    return features


def _symbol_interaction_features(
    logits: Tensor,
    mismatch_scale: float,
    max_degree: int,
) -> Tensor:
    """Return the degree-truncated Walsh expansion of one hard symbol.

    Degree ``D`` is the same exact exponential-Hamming feature map as
    :func:`_symbol_kernel_features`, up to a permutation of coordinates.
    Lower degrees retain complete interaction levels, so bit positions remain
    symmetric and there is no random feature selection.
    """

    symbol_bits = logits.size(-1)
    max_degree = min(int(max_degree), symbol_bits)
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    if max_degree == symbol_bits:
        return _symbol_kernel_features(logits, mismatch_scale)

    symbols = _hard_sign_with_softsign_vjp(logits.permute(0, 2, 1, 3))
    mismatch_factor = math.exp(-float(mismatch_scale) / symbol_bits)
    constant = math.sqrt(0.5 * (1.0 + mismatch_factor))
    signed = math.sqrt(0.5 * (1.0 - mismatch_factor))
    levels = [torch.ones_like(symbols[..., :1]) * constant**symbol_bits]
    if max_degree >= 1:
        levels.append(symbols * constant ** (symbol_bits - 1) * signed)
    for degree in range(2, max_degree + 1):
        cache_key = (symbol_bits, degree, symbols.device)
        indices = _INTERACTION_INDEX_CACHE.get(cache_key)
        if indices is None:
            indices = torch.tensor(
                tuple(combinations(range(symbol_bits), degree)),
                dtype=torch.int64,
                device=symbols.device,
            )
            _INTERACTION_INDEX_CACHE[cache_key] = indices
        interactions = symbols[..., indices].prod(dim=-1)
        coefficient = constant ** (symbol_bits - degree) * signed**degree
        levels.append(interactions * coefficient)
    return torch.cat(levels, dim=-1)


def _shift_history(features: Tensor, offset: int) -> Tensor:
    """Return ``features[t - offset]`` with zero-filled invalid positions."""

    if offset == 0:
        return features
    shifted = torch.zeros_like(features)
    shifted[:, :, offset:] = features[:, :, :-offset]
    return shifted


def _exact_suffix_levels(
    local_features: Tensor,
    max_suffix_length: int,
) -> list[Tensor]:
    """Build exact tensor-product features for each suffix length."""

    levels = [local_features]
    for _ in range(1, int(max_suffix_length)):
        previous = _shift_history(levels[-1], 1)
        next_dim = local_features.size(-1) * previous.size(-1)
        if next_dim > _MAX_EXACT_LEVEL_DIM:
            raise ValueError(
                "exact suffix feature level would have dimension "
                f"{next_dim}; use sketch_suffix_delta"
            )
        levels.append(
            (
                local_features.unsqueeze(-1)
                * previous.unsqueeze(-2)
            ).flatten(-2)
        )
    return levels


def _count_sketch_map(
    feature_dim: int,
    sketch_dim: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Construct one fixed hash/sign map without creating model parameters."""

    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) & ((1 << 63) - 1)
    )
    buckets = torch.randint(
        sketch_dim,
        (feature_dim,),
        dtype=torch.int64,
        generator=generator,
    )
    signs = 2 * torch.randint(
        2,
        (feature_dim,),
        dtype=torch.int64,
        generator=generator,
    ) - 1
    return buckets.to(device), signs.to(device)


def _count_sketch(
    features: Tensor,
    sketch_dim: int,
    seed: int,
) -> Tensor:
    buckets, signs = _count_sketch_map(
        features.size(-1),
        sketch_dim,
        seed,
        features.device,
    )
    index = buckets.view(*([1] * (features.ndim - 1)), -1).expand_as(features)
    source = features * signs.to(features.dtype)
    return torch.zeros(
        *features.shape[:-1],
        sketch_dim,
        dtype=features.dtype,
        device=features.device,
    ).scatter_add(-1, index, source)


def _tensor_sketch_suffix_levels(
    local_features: Tensor,
    max_suffix_length: int,
    sketch_dim: int,
    sketch_seed: int,
) -> list[Tensor]:
    """Sketch each exact suffix tensor product with independent factors.

    A length-L level is a TensorSketch of
    ``phi(x_t) tensor ... tensor phi(x_{t-L+1})``.  Independent maps for every
    ``(length, offset)`` make each level a direct CountSketch of the unexpanded
    tensor product, rather than repeatedly sketching an earlier sketch.
    """

    levels = []
    for length in range(1, int(max_suffix_length) + 1):
        product_spectrum = None
        for offset in range(length):
            factor = _shift_history(local_features, offset)
            factor_seed = (
                int(sketch_seed)
                + 1_000_003 * length
                + 97_409 * offset
            )
            factor_sketch = _count_sketch(
                factor,
                sketch_dim,
                factor_seed,
            )
            spectrum = torch.fft.fft(factor_sketch, dim=-1)
            product_spectrum = (
                spectrum
                if product_spectrum is None
                else product_spectrum * spectrum
            )
        levels.append(torch.fft.ifft(product_spectrum, dim=-1).real)
    return levels


def _join_levels(levels: Sequence[Tensor]) -> Tensor:
    if not levels:
        raise ValueError("at least one suffix level is required")
    return torch.cat(tuple(levels), dim=-1)


def _fingerprints(
    query: Tensor,
    key: Tensor,
    *,
    proxy: str,
    fingerprint_length: int,
    mismatch_scale: float,
    sketch_dim: int,
    sketch_seed: int,
) -> tuple[Tensor, Tensor]:
    state_degree = _STATE_PROXY_DEGREES.get(proxy)
    if state_degree is not None:
        return (
            _symbol_interaction_features(
                query,
                mismatch_scale,
                state_degree,
            ),
            _symbol_interaction_features(
                key,
                mismatch_scale,
                state_degree,
            ),
        )

    query_local = _symbol_kernel_features(query, mismatch_scale)
    key_local = _symbol_kernel_features(key, mismatch_scale)

    if proxy in ("linear_attention", "delta_rule"):
        return query_local, key_local
    if proxy == "single_suffix_delta":
        query_levels = _exact_suffix_levels(query_local, fingerprint_length)
        key_levels = _exact_suffix_levels(key_local, fingerprint_length)
        return query_levels[-1], key_levels[-1]
    if proxy == "single_suffix_sketch_delta":
        return (
            _tensor_sketch_suffix_levels(
                query_local,
                fingerprint_length,
                sketch_dim,
                sketch_seed,
            )[-1],
            _tensor_sketch_suffix_levels(
                key_local,
                fingerprint_length,
                sketch_dim,
                sketch_seed,
            )[-1],
        )
    if proxy == "exact_suffix_delta":
        return (
            _join_levels(
                _exact_suffix_levels(query_local, fingerprint_length)
            ),
            _join_levels(
                _exact_suffix_levels(key_local, fingerprint_length)
            ),
        )
    if proxy == "sketch_suffix_delta":
        return (
            _join_levels(
                _tensor_sketch_suffix_levels(
                    query_local,
                    fingerprint_length,
                    sketch_dim,
                    sketch_seed,
                )
            ),
            _join_levels(
                _tensor_sketch_suffix_levels(
                    key_local,
                    fingerprint_length,
                    sketch_dim,
                    sketch_seed,
                )
            ),
        )
    raise ValueError(f"unknown fast-weight proxy: {proxy}")


def _normalize(features: Tensor) -> Tensor:
    squared_norm = features.square().sum(dim=-1, keepdim=True)
    valid = squared_norm > _NORMALIZATION_EPS**2
    normalized = features * squared_norm.clamp_min(
        _NORMALIZATION_EPS**2
    ).rsqrt()
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def _linear_attention_carrier(
    query_features: Tensor,
    key_features: Tensor,
    values: Tensor,
) -> Tensor:
    batch, heads, sequence_length, feature_dim = query_features.shape
    value_dim = values.size(-1)
    memory = query_features.new_zeros(batch, heads, feature_dim, value_dim)
    normalizer = query_features.new_zeros(batch, heads, feature_dim)
    outputs = [values[:, :, 0] * 0.0]

    for position in range(1, sequence_length):
        write_key = key_features[:, :, position - 1]
        write_value = values[:, :, position]
        memory = memory + write_key.unsqueeze(-1) * write_value.unsqueeze(-2)
        normalizer = normalizer + write_key
        read_query = query_features[:, :, position]
        numerator = torch.einsum("bhf,bhfv->bhv", read_query, memory)
        denominator = torch.einsum(
            "bhf,bhf->bh", read_query, normalizer
        ).clamp_min(_NORMALIZATION_EPS)
        outputs.append(numerator / denominator.unsqueeze(-1))
    return torch.stack(outputs, dim=2).permute(0, 2, 1, 3)


def _delta_rule_carrier(
    query_features: Tensor,
    key_features: Tensor,
    values: Tensor,
) -> Tensor:
    query_features = _normalize(query_features)
    key_features = _normalize(key_features)
    batch, heads, sequence_length, feature_dim = query_features.shape
    value_dim = values.size(-1)
    memory = query_features.new_zeros(batch, heads, feature_dim, value_dim)
    outputs = [values[:, :, 0] * 0.0]

    for position in range(1, sequence_length):
        write_key = key_features[:, :, position - 1]
        write_value = values[:, :, position]
        prediction = torch.einsum("bhf,bhfv->bhv", write_key, memory)
        residual = write_value - prediction
        memory = memory + write_key.unsqueeze(-1) * residual.unsqueeze(-2)
        read_query = query_features[:, :, position]
        outputs.append(torch.einsum("bhf,bhfv->bhv", read_query, memory))
    return torch.stack(outputs, dim=2).permute(0, 2, 1, 3)


def _fast_weight_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    proxy: str,
    fingerprint_length: int,
    mismatch_scale: float,
    sketch_dim: int,
    sketch_seed: int,
) -> Tensor:
    query_features, key_features = _fingerprints(
        query,
        key,
        proxy=proxy,
        fingerprint_length=fingerprint_length,
        mismatch_scale=mismatch_scale,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
    )
    values = _expand_value_heads(
        _hard_sign_with_softsign_vjp(value),
        query.size(2),
    )
    if proxy in _LINEAR_ATTENTION_PROXIES:
        carrier = _linear_attention_carrier(
            query_features,
            key_features,
            values,
        )
    else:
        carrier = _delta_rule_carrier(
            query_features,
            key_features,
            values,
        )

    # Preserve a valid zero VJP for sequence_length == 1 and other degenerate
    # all-zero fingerprints without adding any numerical forward signal.
    dependency = (
        query_features.sum() + key_features.sum() + values.sum()
    ) * 0.0
    return carrier + dependency


class _HardForwardFastWeightVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        max_suffix_length: int,
        fingerprint_length: int,
        mismatch_scale: float,
        proxy_id: int,
        sketch_dim: int,
        sketch_seed: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
        )
        ctx.fingerprint_length = int(fingerprint_length)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.proxy = _PROXIES_BY_ID[int(proxy_id)]
        ctx.sketch_dim = int(sketch_dim)
        ctx.sketch_seed = int(sketch_seed)
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
            carrier = _fast_weight_carrier(
                *(tensor.to(compute_dtype) for tensor in leaves),
                proxy=ctx.proxy,
                fingerprint_length=ctx.fingerprint_length,
                mismatch_scale=ctx.mismatch_scale,
                sketch_dim=ctx.sketch_dim,
                sketch_seed=ctx.sketch_seed,
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
        )


def rosa_fast_weight_proxy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    proxy: str,
    max_suffix_length: int = 32,
    fingerprint_length: int | None = None,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    sketch_dim: int = 64,
    sketch_seed: int = 0,
) -> Tensor:
    """Return exact hard ROSA values with a parameter-free fast-weight VJP."""

    if proxy not in PROXIES:
        raise ValueError(f"proxy must be one of {PROXIES}")
    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        1.0,
        0.0,
        mismatch_scale,
    )
    if query.size(-1) > _MAX_SYMBOL_BITS:
        raise ValueError(
            "fast-weight prototype supports at most 8 Q/K bits; "
            "the exact local feature map has dimension 2**D"
        )
    if fingerprint_length is None:
        fingerprint_length = max_suffix_length
    if isinstance(fingerprint_length, bool) or not isinstance(
        fingerprint_length, int
    ):
        raise TypeError("fingerprint_length must be an integer")
    if not 1 <= fingerprint_length <= max_suffix_length:
        raise ValueError(
            "fingerprint_length must be in [1, max_suffix_length]"
        )
    if isinstance(sketch_dim, bool) or not isinstance(sketch_dim, int):
        raise TypeError("sketch_dim must be an integer")
    if sketch_dim < 2 or sketch_dim & (sketch_dim - 1):
        raise ValueError("sketch_dim must be a power of two >= 2")
    if isinstance(sketch_seed, bool) or not isinstance(sketch_seed, int):
        raise TypeError("sketch_seed must be an integer")

    return _HardForwardFastWeightVjp.apply(
        query,
        key,
        value,
        max_suffix_length,
        fingerprint_length,
        float(mismatch_scale),
        _PROXY_IDS[proxy],
        sketch_dim,
        sketch_seed,
    )


__all__ = [
    "PROXIES",
    "rosa_fast_weight_proxy",
]
