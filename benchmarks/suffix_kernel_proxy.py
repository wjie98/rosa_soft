"""Factorized suffix-kernel VJPs for exact hard ROSA research.

The hard forward is unchanged.  Backward first lifts Q/K histories into
aligned multiplicative suffix features, then performs an additive or
delta-rule fast-weight scan.  Exact tensor products are a small-shape oracle;
nested TensorSketch keeps the feature width fixed without deleting any key.
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
    _count_sketch,
    _delta_rule_carrier,
    _exact_suffix_levels,
    _linear_attention_carrier,
    _shift_history,
)
from rosa_soft.soft_reference import (  # noqa: E402
    _expand_value_heads,
    _hard_route_forward,
    _hard_sign_with_softsign_vjp,
    _reference_compute_dtype,
    _validate_reference_call,
)


REPRESENTATIONS = ("exact", "sketch")
READ_RULES = ("additive", "delta")
ROUTE_KERNELS = ("raw", "quadratic", "level_quadratic")

_REPRESENTATION_IDS = {name: index for index, name in enumerate(REPRESENTATIONS)}
_READ_RULE_IDS = {name: index for index, name in enumerate(READ_RULES)}
_ROUTE_KERNEL_IDS = {name: index for index, name in enumerate(ROUTE_KERNELS)}
_REPRESENTATIONS_BY_ID = {index: name for name, index in _REPRESENTATION_IDS.items()}
_READ_RULES_BY_ID = {index: name for name, index in _READ_RULE_IDS.items()}
_ROUTE_KERNELS_BY_ID = {index: name for name, index in _ROUTE_KERNEL_IDS.items()}

_MAX_SYMBOL_BITS = 8
_MAX_ROUTE_FEATURE_DIM = 1 << 18
_SQRT_TWO = math.sqrt(2.0)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


class _ExactSymbolKernelFeatures(torch.autograd.Function):
    """Walsh features with the production exponential-Hamming VJP.

    The finite Walsh map is exact on binary symbols, but its ordinary
    polynomial derivative is not the derivative of production's exponential
    match gate.  At every binary vertex there is a unique linear feature-space
    Jacobian that reproduces that gate derivative against every other binary
    symbol.  Backward applies that Jacobian and then the production softsign
    STE derivative.
    """

    @staticmethod
    def forward(ctx, logits: Tensor, mismatch_scale: float) -> Tensor:
        symbols = torch.where(
            logits.permute(0, 2, 1, 3) > 0,
            torch.ones((), dtype=logits.dtype, device=logits.device),
            -torch.ones((), dtype=logits.dtype, device=logits.device),
        )
        symbol_bits = symbols.size(-1)
        mismatch_factor = math.exp(float(-mismatch_scale) / symbol_bits)
        constant = math.sqrt(0.5 * (1.0 + mismatch_factor))
        signed = math.sqrt(0.5 * (1.0 - mismatch_factor))

        features = symbols.new_ones(*symbols.shape[:-1], 1)
        for bit in range(symbol_bits):
            bit_values = symbols[..., bit : bit + 1]
            features = torch.cat(
                (features * constant, features * (signed * bit_values)),
                dim=-1,
            )
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.constant_squared = constant * constant
        ctx.signed_squared = signed * signed
        ctx.save_for_backward(logits, symbols, features)
        return features

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        logits, symbols, features = ctx.saved_tensors
        symbol_bits = symbols.size(-1)
        coordinate = torch.arange(
            features.size(-1),
            device=features.device,
        )
        low_ratio = ctx.signed_squared / ctx.constant_squared
        high_ratio = ctx.constant_squared / ctx.signed_squared
        alpha = ctx.mismatch_scale / (2.0 * symbol_bits)
        symbol_gradient = symbols.new_empty(symbols.shape)
        weighted = grad_output * features
        for bit in range(symbol_bits):
            contains_bit = ((coordinate >> bit) & 1).to(torch.bool)
            ratio = torch.where(
                contains_bit,
                features.new_tensor(high_ratio),
                features.new_tensor(low_ratio),
            )
            symbol_gradient[..., bit] = (
                alpha
                * symbols[..., bit]
                * (weighted * ratio).sum(dim=-1)
            )
        ste_derivative = (1.0 + logits.abs()).square().reciprocal()
        return symbol_gradient.permute(0, 2, 1, 3) * ste_derivative, None


def _symbol_kernel_features(logits: Tensor, mismatch_scale: float) -> Tensor:
    return _ExactSymbolKernelFeatures.apply(logits, float(mismatch_scale))


def _quadratic_features(features: Tensor) -> Tensor:
    """Return the homogeneous degree-two map with a dot-squared kernel."""

    dimension = features.size(-1)
    output_dim = dimension * (dimension + 1) // 2
    if output_dim > _MAX_ROUTE_FEATURE_DIM:
        raise ValueError(
            "quadratic route feature dimension would be "
            f"{output_dim}; use level_quadratic or a smaller fingerprint"
        )
    rows, columns = torch.triu_indices(
        dimension,
        dimension,
        offset=1,
        device=features.device,
    )
    cross = _SQRT_TWO * features[..., rows] * features[..., columns]
    return torch.cat((features.square(), cross), dim=-1)


def _nested_tensor_sketch_suffix_levels(
    local_features: Tensor,
    max_suffix_length: int,
    sketch_dim: int,
    sketch_seed: int,
) -> list[Tensor]:
    """Sketch all ending suffixes with one independent map per offset.

    The length-L spectrum is the product of the first L shifted factor
    spectra.  This is a direct TensorSketch of the unexpanded tensor product,
    while requiring only one new factor transform per level.
    """

    product_spectrum = None
    levels = []
    for offset in range(int(max_suffix_length)):
        factor = _shift_history(local_features, offset)
        factor_sketch = _count_sketch(
            factor,
            sketch_dim,
            int(sketch_seed) + 97_409 * offset,
        )
        spectrum = torch.fft.fft(factor_sketch, dim=-1)
        product_spectrum = (
            spectrum
            if product_spectrum is None
            else product_spectrum * spectrum
        )
        levels.append(torch.fft.ifft(product_spectrum, dim=-1).real)
    return levels


def _suffix_feature_branches(
    logits: Tensor,
    *,
    representation: str,
    fingerprint_length: int,
    mismatch_scale: float,
    sketch_dim: int,
    sketch_count: int,
    sketch_seed: int,
    local_features: Tensor | None = None,
) -> list[list[Tensor]]:
    if local_features is None:
        local_features = _symbol_kernel_features(logits, mismatch_scale)
    if representation == "exact":
        return [_exact_suffix_levels(local_features, fingerprint_length)]
    return [
        _nested_tensor_sketch_suffix_levels(
            local_features,
            fingerprint_length,
            sketch_dim,
            int(sketch_seed) + 10_000_019 * branch,
        )
        for branch in range(sketch_count)
    ]


def _route_features_from_branches(
    branches: list[list[Tensor]],
    route_kernel: str,
    *,
    route_sketch_dim: int | None = None,
    route_sketch_seed: int = 0,
    shared_raw_level: Tensor | None = None,
) -> Tensor:
    branch_scale = 1.0 / math.sqrt(len(branches))
    outputs = []
    for branch, levels in enumerate(branches):
        if route_kernel == "raw":
            first_level = 1 if shared_raw_level is not None else 0
            outputs.extend(
                level * branch_scale for level in levels[first_level:]
            )
        elif route_kernel == "level_quadratic":
            outputs.extend(
                _quadratic_features(level) * branch_scale
                for level in levels
            )
        else:
            joined = torch.cat(tuple(levels), dim=-1)
            if route_sketch_dim is not None:
                joined = _count_sketch(
                    joined,
                    route_sketch_dim,
                    int(route_sketch_seed) + 3_000_017 * branch,
                )
            outputs.append(_quadratic_features(joined) * branch_scale)
    if shared_raw_level is not None:
        outputs.insert(0, shared_raw_level)
    return torch.cat(tuple(outputs), dim=-1)


def _suffix_kernel_features(
    logits: Tensor,
    *,
    representation: str,
    route_kernel: str,
    fingerprint_length: int,
    mismatch_scale: float,
    sketch_dim: int,
    sketch_count: int,
    sketch_seed: int,
) -> Tensor:
    local_features = _symbol_kernel_features(logits, mismatch_scale)
    branches = _suffix_feature_branches(
        logits,
        representation=representation,
        fingerprint_length=fingerprint_length,
        mismatch_scale=mismatch_scale,
        sketch_dim=sketch_dim,
        sketch_count=sketch_count,
        sketch_seed=sketch_seed,
        local_features=local_features,
    )
    return _route_features_from_branches(
        branches,
        route_kernel,
        route_sketch_dim=(sketch_dim if representation == "sketch" else None),
        route_sketch_seed=int(sketch_seed) + 70_000_027,
        shared_raw_level=(
            local_features
            if representation == "sketch" and route_kernel == "raw"
            else None
        ),
    )


def _suffix_kernel_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    representation: str,
    read_rule: str,
    route_kernel: str,
    fingerprint_length: int,
    mismatch_scale: float,
    sketch_dim: int,
    sketch_count: int,
    sketch_seed: int,
) -> Tensor:
    common = {
        "representation": representation,
        "route_kernel": route_kernel,
        "fingerprint_length": fingerprint_length,
        "mismatch_scale": mismatch_scale,
        "sketch_dim": sketch_dim,
        "sketch_count": sketch_count,
        "sketch_seed": sketch_seed,
    }
    query_features = _suffix_kernel_features(query, **common)
    key_features = _suffix_kernel_features(key, **common)
    values = _expand_value_heads(
        _hard_sign_with_softsign_vjp(value),
        query.size(2),
    )
    if read_rule == "additive":
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
    dependency = (
        query_features.sum() + key_features.sum() + values.sum()
    ) * 0.0
    return carrier + dependency


class _HardForwardSuffixKernelVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        max_suffix_length: int,
        fingerprint_length: int,
        mismatch_scale: float,
        representation_id: int,
        read_rule_id: int,
        route_kernel_id: int,
        sketch_dim: int,
        sketch_count: int,
        sketch_seed: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        hard_output, _, _, _ = _hard_route_forward(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
        )
        ctx.config = (
            int(fingerprint_length),
            float(mismatch_scale),
            int(representation_id),
            int(read_rule_id),
            int(route_kernel_id),
            int(sketch_dim),
            int(sketch_count),
            int(sketch_seed),
        )
        ctx.save_for_backward(query, key, value)
        return hard_output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        (
            fingerprint_length,
            mismatch_scale,
            representation_id,
            read_rule_id,
            route_kernel_id,
            sketch_dim,
            sketch_count,
            sketch_seed,
        ) = ctx.config
        with torch.enable_grad():
            leaves = tuple(
                tensor.detach().requires_grad_(need)
                for tensor, need in zip((query, key, value), needs)
            )
            compute_dtype = _reference_compute_dtype(query.dtype)
            carrier = _suffix_kernel_carrier(
                *(tensor.to(compute_dtype) for tensor in leaves),
                representation=_REPRESENTATIONS_BY_ID[representation_id],
                read_rule=_READ_RULES_BY_ID[read_rule_id],
                route_kernel=_ROUTE_KERNELS_BY_ID[route_kernel_id],
                fingerprint_length=fingerprint_length,
                mismatch_scale=mismatch_scale,
                sketch_dim=sketch_dim,
                sketch_count=sketch_count,
                sketch_seed=sketch_seed,
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
        return (*gradients, *(None for _ in range(9)))


def rosa_suffix_kernel_proxy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    representation: str,
    read_rule: str,
    route_kernel: str,
    max_suffix_length: int = 32,
    fingerprint_length: int | None = None,
    mismatch_scale: float = 3.0,
    sketch_dim: int = 64,
    sketch_count: int = 1,
    sketch_seed: int = 0,
) -> Tensor:
    """Return exact hard ROSA values with a factorized suffix-kernel VJP."""

    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    if read_rule not in READ_RULES:
        raise ValueError(f"read_rule must be one of {READ_RULES}")
    if route_kernel not in ROUTE_KERNELS:
        raise ValueError(f"route_kernel must be one of {ROUTE_KERNELS}")
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
        raise ValueError("suffix-kernel prototype supports at most 8 Q/K bits")
    if fingerprint_length is None:
        fingerprint_length = max_suffix_length
    if isinstance(fingerprint_length, bool) or not isinstance(
        fingerprint_length,
        int,
    ):
        raise TypeError("fingerprint_length must be an integer")
    if not 1 <= fingerprint_length <= max_suffix_length:
        raise ValueError(
            "fingerprint_length must be in [1, max_suffix_length]"
        )
    if isinstance(sketch_dim, bool) or not isinstance(sketch_dim, int):
        raise TypeError("sketch_dim must be an integer")
    if not _is_power_of_two(sketch_dim):
        raise ValueError("sketch_dim must be a positive power of two")
    if isinstance(sketch_count, bool) or not isinstance(sketch_count, int):
        raise TypeError("sketch_count must be an integer")
    if sketch_count < 1:
        raise ValueError("sketch_count must be positive")
    if isinstance(sketch_seed, bool) or not isinstance(sketch_seed, int):
        raise TypeError("sketch_seed must be an integer")

    return _HardForwardSuffixKernelVjp.apply(
        query,
        key,
        value,
        max_suffix_length,
        fingerprint_length,
        float(mismatch_scale),
        _REPRESENTATION_IDS[representation],
        _READ_RULE_IDS[read_rule],
        _ROUTE_KERNEL_IDS[route_kernel],
        sketch_dim,
        sketch_count,
        sketch_seed,
    )


__all__ = [
    "READ_RULES",
    "REPRESENTATIONS",
    "ROUTE_KERNELS",
    "rosa_suffix_kernel_proxy",
]
