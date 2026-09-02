"""Research-only value codecs for the quadratic state-attention VJP."""

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
    _symbol_interaction_features,
)
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
)
from rosa_soft.soft_reference import (  # noqa: E402
    _expand_value_heads,
    _gather_routed_values,
    _hard_route_forward,
    _hard_sign_with_softsign_vjp,
    _reference_compute_dtype,
    _validate_reference_call,
)


VALUE_CODECS = ("binary", "uniform", "exponential", "rms", "float")
VALUE_GRADIENTS = ("proxy", "selected")

_CODEC_IDS = {name: index for index, name in enumerate(VALUE_CODECS)}
_CODECS_BY_ID = {index: name for name, index in _CODEC_IDS.items()}
_GRADIENT_IDS = {name: index for index, name in enumerate(VALUE_GRADIENTS)}
_GRADIENTS_BY_ID = {
    index: name for name, index in _GRADIENT_IDS.items()
}
_NORMALIZATION_EPS = 1e-6


def _validate_codec(
    codec: str,
    quant_bits: int,
    exponent_min: float,
) -> None:
    if codec not in VALUE_CODECS:
        raise ValueError(f"value_codec must be one of {VALUE_CODECS}")
    if isinstance(quant_bits, bool) or not isinstance(quant_bits, int):
        raise TypeError("value_quant_bits must be an integer")
    if codec in ("uniform", "exponential") and not 2 <= quant_bits <= 8:
        raise ValueError(
            "value_quant_bits must be in [2, 8] for quantized codecs"
        )
    if codec == "exponential" and (
        not math.isfinite(exponent_min) or exponent_min >= 0.0
    ):
        raise ValueError("value_exponent_min must be finite and < 0")


def _uniform_values(value: Tensor, quant_bits: int) -> Tensor:
    bounded = value.clamp(-1.0, 1.0)
    intervals = (1 << quant_bits) - 1
    hard = (
        torch.round((bounded + 1.0) * (0.5 * intervals))
        * (2.0 / intervals)
        - 1.0
    )
    return bounded + (hard - bounded).detach()


def _exponential_values(
    value: Tensor,
    quant_bits: int,
    exponent_min: float,
) -> Tensor:
    bounded = value.clamp(-1.0, 1.0)
    magnitude_bins = 1 << (quant_bits - 1)
    exponent_step = -exponent_min / (magnitude_bins - 1)
    magnitude = value.abs().clamp(
        min=2.0**exponent_min,
        max=1.0,
    )
    exponent_index = torch.round(
        (torch.log2(magnitude) - exponent_min) / exponent_step
    ).clamp(0, magnitude_bins - 1)
    hard_magnitude = torch.exp2(
        exponent_min + exponent_index * exponent_step
    )
    hard = torch.where(value > 0.0, hard_magnitude, -hard_magnitude)
    return bounded + (hard - bounded).detach()


def encode_values(
    value: Tensor,
    *,
    codec: str,
    quant_bits: int = 4,
    exponent_min: float = -4.0,
) -> Tensor:
    """Encode V with hard numerical forward and the codec's local VJP."""

    _validate_codec(codec, quant_bits, exponent_min)
    if codec == "binary":
        return _hard_sign_with_softsign_vjp(value)
    if codec == "uniform":
        return _uniform_values(value, quant_bits)
    if codec == "exponential":
        return _exponential_values(value, quant_bits, exponent_min)
    if codec == "rms":
        inverse_rms = value.square().mean(dim=-1, keepdim=True).clamp_min(
            _NORMALIZATION_EPS**2
        ).rsqrt()
        return value * inverse_rms
    return value


def _expanded_nonnull_values(value: Tensor, query_heads: int) -> Tensor:
    expanded = _expand_value_heads(value, query_heads)
    nonnull = torch.arange(
        value.size(1),
        device=value.device,
    ).view(1, 1, -1, 1) != 0
    return torch.where(nonnull, expanded, torch.zeros_like(expanded))


def _gather_encoded_values(
    value: Tensor,
    selected_route_indices: Tensor,
    query_heads: int,
) -> Tensor:
    route_values = _expanded_nonnull_values(value, query_heads)
    return _gather_routed_values(
        route_values,
        selected_route_indices,
    ).permute(0, 2, 1, 3)


def _quadratic_state_carrier(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    mismatch_scale: float,
    codec: str,
    quant_bits: int,
    exponent_min: float,
) -> Tensor:
    query_features = _symbol_interaction_features(
        query,
        mismatch_scale,
        2,
    )
    key_features = _symbol_interaction_features(
        key,
        mismatch_scale,
        2,
    )
    represented = _expand_value_heads(
        encode_values(
            value,
            codec=codec,
            quant_bits=quant_bits,
            exponent_min=exponent_min,
        ),
        query.size(2),
    )
    return _linear_attention_carrier(
        query_features,
        key_features,
        represented,
    )


class _HardForwardValueCodecStateVjp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        max_suffix_length: int,
        mismatch_scale: float,
        codec_id: int,
        quant_bits: int,
        exponent_min: float,
        gradient_id: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        _, _, selected_route_indices, _ = _hard_route_forward(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
        )
        represented = encode_values(
            value.to(compute_dtype),
            codec=_CODECS_BY_ID[int(codec_id)],
            quant_bits=int(quant_bits),
            exponent_min=float(exponent_min),
        )
        output = _gather_encoded_values(
            represented,
            selected_route_indices,
            query.size(2),
        )
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.codec = _CODECS_BY_ID[int(codec_id)]
        ctx.quant_bits = int(quant_bits)
        ctx.exponent_min = float(exponent_min)
        ctx.value_gradient = _GRADIENTS_BY_ID[int(gradient_id)]
        ctx.save_for_backward(query, key, value, selected_route_indices)
        return output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value, selected_route_indices = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        gradients = [None, None, None]
        compute_dtype = _reference_compute_dtype(query.dtype)

        with torch.enable_grad():
            leaves = tuple(
                tensor.detach().requires_grad_(need)
                for tensor, need in zip((query, key, value), needs)
            )
            query_leaf, key_leaf, value_leaf = leaves
            codec_args = {
                "codec": ctx.codec,
                "quant_bits": ctx.quant_bits,
                "exponent_min": ctx.exponent_min,
            }

            if ctx.value_gradient == "proxy":
                carrier = _quadratic_state_carrier(
                    query_leaf.to(compute_dtype),
                    key_leaf.to(compute_dtype),
                    value_leaf.to(compute_dtype),
                    mismatch_scale=ctx.mismatch_scale,
                    **codec_args,
                ).to(query.dtype)
                required = [index for index, need in enumerate(needs) if need]
                required_gradients = torch.autograd.grad(
                    carrier,
                    tuple(leaves[index] for index in required),
                    grad_output,
                    create_graph=False,
                )
                for index, gradient in zip(required, required_gradients):
                    gradients[index] = gradient
            else:
                qk_required = [index for index in (0, 1) if needs[index]]
                if qk_required:
                    carrier = _quadratic_state_carrier(
                        query_leaf.to(compute_dtype),
                        key_leaf.to(compute_dtype),
                        value.detach().to(compute_dtype),
                        mismatch_scale=ctx.mismatch_scale,
                        **codec_args,
                    ).to(query.dtype)
                    qk_gradients = torch.autograd.grad(
                        carrier,
                        tuple(leaves[index] for index in qk_required),
                        grad_output,
                        create_graph=False,
                    )
                    for index, gradient in zip(qk_required, qk_gradients):
                        gradients[index] = gradient
                if needs[2]:
                    represented = encode_values(
                        value_leaf.to(compute_dtype),
                        **codec_args,
                    )
                    selected_carrier = _gather_encoded_values(
                        represented,
                        selected_route_indices,
                        query.size(2),
                    ).to(query.dtype)
                    gradients[2] = torch.autograd.grad(
                        selected_carrier,
                        value_leaf,
                        grad_output,
                        create_graph=False,
                    )[0]

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


def rosa_value_codec_state_proxy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    codec: str,
    quant_bits: int = 4,
    exponent_min: float = -4.0,
    value_gradient: str = "proxy",
    max_suffix_length: int = 32,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    """Return a hard route with a configurable V representation and VJP."""

    _validate_codec(codec, quant_bits, exponent_min)
    if value_gradient not in VALUE_GRADIENTS:
        raise ValueError(f"value_gradient must be one of {VALUE_GRADIENTS}")
    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        1.0,
        0.0,
        mismatch_scale,
    )
    return _HardForwardValueCodecStateVjp.apply(
        query,
        key,
        value,
        max_suffix_length,
        float(mismatch_scale),
        _CODEC_IDS[codec],
        int(quant_bits),
        float(exponent_min),
        _GRADIENT_IDS[value_gradient],
    )


__all__ = [
    "VALUE_CODECS",
    "VALUE_GRADIENTS",
    "encode_values",
    "rosa_value_codec_state_proxy",
]
