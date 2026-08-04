"""Research-only stochastic hard-bit VJPs for batched ROSA training."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft  # noqa: E402
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.soft_reference import (  # noqa: E402
    _expand_value_heads,
    _hard_route_forward,
    _hard_sign,
    _reference_compute_dtype,
    _validate_reference_call,
)


ESTIMATORS = ("arm", "disarm")
BACKENDS = ("reference", "cuda")
_ARM = 0
_DISARM = 1
_REFERENCE = 0
_CUDA = 1


def _backend_code(backend: str, query: Tensor) -> int:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend == "cuda":
        if not query.is_cuda:
            raise ValueError("the CUDA backend requires CUDA tensors")
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("the RosaSoft CUDA extension is unavailable")
        return _CUDA
    return _REFERENCE


def _base_operator(backend_code: int) -> Callable[..., Tensor]:
    return (
        rosa_soft.rosa_soft
        if backend_code == _CUDA
        else rosa_soft.rosa_soft_reference
    )


def _hard_output(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
    backend_code: int,
) -> Tensor:
    if backend_code == _CUDA:
        return rosa_soft.rosa_soft(
            query,
            key,
            value,
            max_suffix_length=max_suffix_length,
            scale=1.0,
            dropout_p=0.0,
            mismatch_scale=1.0,
        )
    output, _, _, _ = _hard_route_forward(
        query,
        key,
        value,
        max_suffix_length,
    )
    return output


def _sample_symbols(
    logits: Tensor,
    uniforms: Tensor,
    bit_temperature: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    probabilities = torch.sigmoid(logits.unsqueeze(0) / bit_temperature)
    direct_bits = uniforms < probabilities
    antithetic_bits = (1.0 - uniforms) < probabilities
    one = torch.ones((), dtype=logits.dtype, device=logits.device)
    direct = torch.where(direct_bits, one, -one)
    antithetic = torch.where(antithetic_bits, one, -one)
    return direct, antithetic, direct_bits, antithetic_bits


def _sampled_qk_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    query_uniforms: Tensor,
    key_uniforms: Tensor,
    *,
    max_suffix_length: int,
    bit_temperature: float,
    estimator_code: int,
    backend_code: int,
) -> tuple[Tensor, Tensor]:
    """Estimate Q/K VJPs using rewards local to each batch and head."""

    pairs, batch, sequence_length, heads, bits = query_uniforms.shape
    direct_query, antithetic_query, direct_q_bits, antithetic_q_bits = (
        _sample_symbols(query, query_uniforms, bit_temperature)
    )
    direct_key, antithetic_key, direct_k_bits, antithetic_k_bits = (
        _sample_symbols(key, key_uniforms, bit_temperature)
    )
    hard_query = _hard_sign(query).unsqueeze(0)
    hard_key = _hard_sign(key).unsqueeze(0)
    direct_query[:, :, 0] = hard_query[:, :, 0]
    antithetic_query[:, :, 0] = hard_query[:, :, 0]
    direct_key[:, :, -1] = hard_key[:, :, -1]
    antithetic_key[:, :, -1] = hard_key[:, :, -1]

    flat_shape = (pairs * batch, sequence_length, heads, bits)
    expanded_value = value.unsqueeze(0).expand(
        pairs,
        *value.shape,
    ).reshape(pairs * batch, *value.shape[1:])
    direct_output = _hard_output(
        direct_query.reshape(flat_shape),
        direct_key.reshape(flat_shape),
        expanded_value,
        max_suffix_length,
        backend_code,
    ).view(pairs, batch, sequence_length, heads, value.size(-1))
    antithetic_output = _hard_output(
        antithetic_query.reshape(flat_shape),
        antithetic_key.reshape(flat_shape),
        expanded_value,
        max_suffix_length,
        backend_code,
    ).view(pairs, batch, sequence_length, heads, value.size(-1))
    direct_reward = (
        direct_output * grad_output.unsqueeze(0)
    ).sum(dim=(2, 4))
    antithetic_reward = (
        antithetic_output * grad_output.unsqueeze(0)
    ).sum(dim=(2, 4))
    reward_shape = (pairs, batch, 1, heads, 1)

    if estimator_code == _ARM:
        reward_delta = (antithetic_reward - direct_reward).view(
            reward_shape
        )
        query_gradient = (
            reward_delta
            * (query_uniforms - 0.5)
            / bit_temperature
        )
        key_gradient = (
            reward_delta
            * (key_uniforms - 0.5)
            / bit_temperature
        )
    elif estimator_code == _DISARM:
        reward_delta = 0.5 * (
            direct_reward - antithetic_reward
        ).view(reward_shape)
        query_phi = query.unsqueeze(0) / bit_temperature
        key_phi = key.unsqueeze(0) / bit_temperature
        query_sign = torch.where(
            antithetic_q_bits,
            -torch.ones_like(query_uniforms),
            torch.ones_like(query_uniforms),
        )
        key_sign = torch.where(
            antithetic_k_bits,
            -torch.ones_like(key_uniforms),
            torch.ones_like(key_uniforms),
        )
        query_gradient = (
            reward_delta
            * query_sign
            * (direct_q_bits != antithetic_q_bits).to(query.dtype)
            * torch.sigmoid(query_phi.abs())
            / bit_temperature
        )
        key_gradient = (
            reward_delta
            * key_sign
            * (direct_k_bits != antithetic_k_bits).to(key.dtype)
            * torch.sigmoid(key_phi.abs())
            / bit_temperature
        )
    else:
        raise RuntimeError(f"unknown estimator code: {estimator_code}")

    query_gradient[:, :, 0] = 0.0
    key_gradient[:, :, -1] = 0.0
    return query_gradient.mean(dim=0), key_gradient.mean(dim=0)


def _production_value_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    max_suffix_length: int,
    scale: float,
    mismatch_scale: float,
    backend_code: int,
) -> Tensor:
    with torch.enable_grad():
        value_leaf = value.detach().requires_grad_()
        output = _base_operator(backend_code)(
            query.detach(),
            key.detach(),
            value_leaf,
            max_suffix_length=max_suffix_length,
            scale=scale,
            dropout_p=0.0,
            mismatch_scale=mismatch_scale,
        )
        (value_gradient,) = torch.autograd.grad(
            output,
            value_leaf,
            grad_output,
            create_graph=False,
        )
    return value_gradient


def _mean_field_latest_probabilities(
    query: Tensor,
    key: Tensor,
    bit_temperature: float,
) -> Tensor:
    """Approximate W=1 latest-match routes with independent candidates."""

    batch, sequence_length, heads, _ = query.shape
    query_probability = torch.sigmoid(
        query.permute(0, 2, 1, 3) / bit_temperature
    )
    key_probability = torch.sigmoid(
        key.permute(0, 2, 1, 3)[..., :-1, :] / bit_temperature
    )
    match_probability = (
        query_probability.unsqueeze(-2) * key_probability.unsqueeze(-3)
        + (1.0 - query_probability.unsqueeze(-2))
        * (1.0 - key_probability.unsqueeze(-3))
    ).prod(dim=-1)
    one = query.new_ones(batch, heads, 1)
    zero = query.new_zeros(batch, heads, 1)
    rows = []
    for query_index in range(sequence_length):
        if query_index == 0:
            rows.append(
                torch.cat(
                    (one, zero.expand(-1, -1, sequence_length - 1)),
                    dim=-1,
                )
            )
            continue
        matches = match_probability[:, :, query_index, :query_index]
        reverse_failure = torch.cumprod(
            (1.0 - matches).flip(-1), dim=-1
        ).flip(-1)
        later_failure = torch.cat(
            (reverse_failure[..., 1:], one), dim=-1
        )
        null_mass = reverse_failure[..., :1]
        route_mass = matches * later_failure
        padding = zero.expand(
            -1,
            -1,
            sequence_length - query_index - 1,
        )
        rows.append(torch.cat((null_mass, route_mass, padding), dim=-1))
    return torch.stack(rows, dim=2)


def _mean_field_qk_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    bit_temperature: float,
) -> tuple[Tensor, Tensor]:
    with torch.enable_grad():
        query_leaf = query.detach().requires_grad_()
        key_leaf = key.detach().requires_grad_()
        probabilities = _mean_field_latest_probabilities(
            query_leaf,
            key_leaf,
            bit_temperature,
        )
        route_values = _expand_value_heads(
            _hard_sign(value.detach()), query.size(2)
        ).clone()
        route_values[:, :, 0] = 0.0
        expected_output = torch.einsum(
            "bhta,bhad->bhtd", probabilities, route_values
        ).permute(0, 2, 1, 3)
        query_gradient, key_gradient = torch.autograd.grad(
            expected_output,
            (query_leaf, key_leaf),
            grad_output,
            create_graph=False,
        )
    return query_gradient, key_gradient


class _HardForwardStochasticQK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_uniforms: Tensor,
        key_uniforms: Tensor,
        max_suffix_length: int,
        scale: float,
        mismatch_scale: float,
        bit_temperature: float,
        estimator_code: int,
        backend_code: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        output = _hard_output(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
            int(max_suffix_length),
            int(backend_code),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.scale = float(scale)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.bit_temperature = float(bit_temperature)
        ctx.estimator_code = int(estimator_code)
        ctx.backend_code = int(backend_code)
        ctx.save_for_backward(
            query,
            key,
            value,
            query_uniforms,
            key_uniforms,
        )
        return output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value, query_uniforms, key_uniforms = ctx.saved_tensors
        needs_query, needs_key, needs_value = ctx.needs_input_grad[:3]
        compute_dtype = _reference_compute_dtype(query.dtype)
        query_gradient = None
        key_gradient = None
        if needs_query or needs_key:
            query_work = query.detach().to(compute_dtype)
            key_work = key.detach().to(compute_dtype)
            value_work = value.detach().to(compute_dtype)
            sampled_query, sampled_key = _sampled_qk_vjp(
                query_work,
                key_work,
                value_work,
                grad_output.to(compute_dtype),
                query_uniforms,
                key_uniforms,
                max_suffix_length=ctx.max_suffix_length,
                bit_temperature=ctx.bit_temperature,
                estimator_code=ctx.estimator_code,
                backend_code=ctx.backend_code,
            )
            if needs_query:
                query_gradient = sampled_query.to(query.dtype)
            if needs_key:
                key_gradient = sampled_key.to(key.dtype)
        value_gradient = None
        if needs_value:
            value_gradient = _production_value_vjp(
                query,
                key,
                value,
                grad_output,
                max_suffix_length=ctx.max_suffix_length,
                scale=ctx.scale,
                mismatch_scale=ctx.mismatch_scale,
                backend_code=ctx.backend_code,
            )
        return (
            query_gradient,
            key_gradient,
            value_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _HardForwardMeanFieldQK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        max_suffix_length: int,
        scale: float,
        mismatch_scale: float,
        bit_temperature: float,
        backend_code: int,
    ) -> Tensor:
        compute_dtype = _reference_compute_dtype(query.dtype)
        output = _hard_output(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
            int(max_suffix_length),
            int(backend_code),
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.scale = float(scale)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.bit_temperature = float(bit_temperature)
        ctx.backend_code = int(backend_code)
        ctx.save_for_backward(query, key, value)
        return output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        query, key, value = ctx.saved_tensors
        needs_query, needs_key, needs_value = ctx.needs_input_grad[:3]
        compute_dtype = _reference_compute_dtype(query.dtype)
        query_gradient = None
        key_gradient = None
        if needs_query or needs_key:
            estimated_query, estimated_key = _mean_field_qk_vjp(
                query.detach().to(compute_dtype),
                key.detach().to(compute_dtype),
                value.detach().to(compute_dtype),
                grad_output.to(compute_dtype),
                bit_temperature=ctx.bit_temperature,
            )
            if needs_query:
                query_gradient = estimated_query.to(query.dtype)
            if needs_key:
                key_gradient = estimated_key.to(key.dtype)
        value_gradient = None
        if needs_value:
            value_gradient = _production_value_vjp(
                query,
                key,
                value,
                grad_output,
                max_suffix_length=ctx.max_suffix_length,
                scale=ctx.scale,
                mismatch_scale=ctx.mismatch_scale,
                backend_code=ctx.backend_code,
            )
        return (
            query_gradient,
            key_gradient,
            value_gradient,
            None,
            None,
            None,
            None,
            None,
        )


def stochastic_hard_rosa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    estimator: str,
    bit_temperature: float = 0.5,
    pairs: int = 1,
    backend: str = "reference",
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = 0.0,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    """Return exact hard values with an ARM or DisARM Q/K VJP."""

    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    bit_temperature = float(bit_temperature)
    if not math.isfinite(bit_temperature) or bit_temperature <= 0.0:
        raise ValueError("bit_temperature must be finite and positive")
    if int(pairs) != pairs or pairs <= 0:
        raise ValueError("pairs must be a positive integer")
    if float(dropout_p) != 0.0:
        raise ValueError("stochastic hard Q/K research requires dropout_p=0")
    backend_code = _backend_code(backend, query)
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not needs_backward or not (query.requires_grad or key.requires_grad):
        return _base_operator(backend_code)(
            query,
            key,
            value,
            max_suffix_length=max_suffix_length,
            scale=scale,
            dropout_p=0.0,
            mismatch_scale=mismatch_scale,
        )

    compute_dtype = _reference_compute_dtype(query.dtype)
    uniform_shape = (int(pairs), *query.shape)
    query_uniforms = torch.rand(
        uniform_shape,
        dtype=compute_dtype,
        device=query.device,
    )
    key_uniforms = torch.rand(
        uniform_shape,
        dtype=compute_dtype,
        device=key.device,
    )
    estimator_code = _ARM if estimator == "arm" else _DISARM
    return _HardForwardStochasticQK.apply(
        query,
        key,
        value,
        query_uniforms,
        key_uniforms,
        max_suffix_length,
        float(scale),
        float(mismatch_scale),
        bit_temperature,
        estimator_code,
        backend_code,
    )


def mean_field_hard_rosa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    bit_temperature: float = 0.5,
    backend: str = "reference",
    max_suffix_length: int = 1,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = 0.0,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    """Return hard ROSA values with a W=1 mean-field latest-match VJP."""

    max_suffix_length = _validate_reference_call(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    if max_suffix_length != 1:
        raise ValueError("mean-field latest-match research supports only W=1")
    bit_temperature = float(bit_temperature)
    if not math.isfinite(bit_temperature) or bit_temperature <= 0.0:
        raise ValueError("bit_temperature must be finite and positive")
    if float(dropout_p) != 0.0:
        raise ValueError("mean-field hard Q/K research requires dropout_p=0")
    backend_code = _backend_code(backend, query)
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not needs_backward or not (query.requires_grad or key.requires_grad):
        return _base_operator(backend_code)(
            query,
            key,
            value,
            max_suffix_length=1,
            scale=scale,
            dropout_p=0.0,
            mismatch_scale=mismatch_scale,
        )
    return _HardForwardMeanFieldQK.apply(
        query,
        key,
        value,
        max_suffix_length,
        float(scale),
        float(mismatch_scale),
        bit_temperature,
        backend_code,
    )
