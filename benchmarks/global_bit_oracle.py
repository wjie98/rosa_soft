"""Tiny globally consistent hard-ROSA gradient oracles.

This module is deliberately research-only.  It enumerates shared Q/K bit
assignments, so its cost is exponential in the number of relevant bits.  The
production operator and its dense backward are not changed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosa_soft.soft_reference import (  # noqa: E402
    _hard_route_forward,
    _hard_sign,
    rosa_soft_reference,
)


__all__ = [
    "ArmDisarmSamples",
    "BitLayout",
    "MarginEditResult",
    "MeanFieldResult",
    "SampledBitflipResult",
    "SharedBitOracleResult",
    "arm_disarm_samples",
    "bitflip_residual_samples",
    "exact_bitflip_vjp",
    "exact_margin_edit_oracle",
    "exact_shared_bit_oracle",
    "mean_field_winner_oracle",
    "production_vjp",
    "sampled_bitflip_residual_vjp",
]


@dataclass(frozen=True)
class BitLayout:
    """Flat locations of the semantically relevant stochastic Q/K bits."""

    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    query_indices: tuple[int, ...]
    key_indices: tuple[int, ...]

    @property
    def query_bit_count(self) -> int:
        return len(self.query_indices)

    @property
    def bit_count(self) -> int:
        return len(self.query_indices) + len(self.key_indices)

    def gather(self, query: Tensor, key: Tensor) -> Tensor:
        query_values = query.reshape(-1)[list(self.query_indices)]
        key_values = key.reshape(-1)[list(self.key_indices)]
        return torch.cat((query_values, key_values))

    def scatter(
        self,
        values: Tensor,
        query_template: Tensor,
        key_template: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if values.shape[-1] != self.bit_count:
            raise ValueError(
                f"expected {self.bit_count} bit values, got {values.shape[-1]}"
            )
        leading = values.shape[:-1]
        query = torch.zeros(
            *leading,
            *self.query_shape,
            dtype=values.dtype,
            device=values.device,
        )
        key = torch.zeros(
            *leading,
            *self.key_shape,
            dtype=values.dtype,
            device=values.device,
        )
        if self.query_indices:
            query.reshape(*leading, -1)[..., list(self.query_indices)] = (
                values[..., : self.query_bit_count]
            )
        if self.key_indices:
            key.reshape(*leading, -1)[..., list(self.key_indices)] = (
                values[..., self.query_bit_count :]
            )
        return query, key


@dataclass(frozen=True)
class SharedBitOracleResult:
    expected_output: Tensor
    expected_scalar: Tensor
    route_probabilities: Tensor
    query_gradient: Tensor
    key_gradient: Tensor
    query_local_expectation_gradient: Tensor
    key_local_expectation_gradient: Tensor
    bit_gradient: Tensor
    local_expectation_bit_gradient: Tensor
    probability_sum: float
    state_count: int
    layout: BitLayout


@dataclass(frozen=True)
class MeanFieldResult:
    expected_output: Tensor
    route_probabilities: Tensor
    query_gradient: Tensor
    key_gradient: Tensor


@dataclass(frozen=True)
class ArmDisarmSamples:
    arm_bit_gradients: Tensor
    disarm_bit_gradients: Tensor
    layout: BitLayout


@dataclass(frozen=True)
class SampledBitflipResult:
    query_gradient: Tensor
    key_gradient: Tensor
    bit_gradient: Tensor
    sampled_indices: Tensor
    layout: BitLayout


@dataclass(frozen=True)
class MarginEditResult:
    query_gradient: Tensor
    key_gradient: Tensor
    target_query_symbols: Tensor
    target_key_symbols: Tensor
    target_output: Tensor
    target_routes: Tensor
    flipped_bits: int
    margin_penalty: float
    linearized_cost: float
    objective_gain: float
    state_count: int
    layout: BitLayout


@dataclass(frozen=True)
class _HardStateTable:
    layout: BitLayout
    assignments: Tensor
    symbols: Tensor
    outputs: Tensor
    routes: Tensor


def _validate_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Optional[Tensor] = None,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 tensors")
    if query.shape != key.shape:
        raise ValueError("query and key must have identical shapes")
    if query.size(0) != 1 or query.size(2) != 1:
        raise ValueError("global-bit oracle requires batch size 1 and one head")
    if value.size(0) != 1 or value.size(1) != query.size(1):
        raise ValueError("value batch and sequence dimensions must match Q/K")
    if value.size(2) != 1:
        raise ValueError("global-bit oracle requires one value head")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must share a device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must share a dtype")
    if not query.dtype.is_floating_point:
        raise ValueError("global-bit oracle requires floating-point tensors")
    if grad_output is not None and grad_output.shape != (
        1,
        query.size(1),
        1,
        value.size(3),
    ):
        raise ValueError(
            "grad_output must have shape [1, T, 1, value_dim]"
        )
    if grad_output is not None and (
        grad_output.device != query.device
        or grad_output.dtype != query.dtype
    ):
        raise ValueError("grad_output must share the Q/K device and dtype")


def _semantic_masks(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    query_mask = torch.zeros_like(query, dtype=torch.bool)
    key_mask = torch.zeros_like(key, dtype=torch.bool)
    if query.size(1) > 1:
        query_mask[:, 1:, :, :] = True
        key_mask[:, :-1, :, :] = True
    return query_mask, key_mask


def _normalize_mask(
    supplied: Optional[Tensor],
    semantic: Tensor,
    name: str,
) -> Tensor:
    if supplied is None:
        return semantic
    if supplied.shape != semantic.shape or supplied.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor matching Q/K")
    supplied = supplied.to(device=semantic.device)
    if bool((supplied & ~semantic).any()):
        raise ValueError(f"{name} selects bits that cannot affect ROSA")
    return supplied


def _make_layout(
    query: Tensor,
    key: Tensor,
    query_stochastic_mask: Optional[Tensor] = None,
    key_stochastic_mask: Optional[Tensor] = None,
) -> BitLayout:
    semantic_query, semantic_key = _semantic_masks(query, key)
    query_mask = _normalize_mask(
        query_stochastic_mask,
        semantic_query,
        "query_stochastic_mask",
    )
    key_mask = _normalize_mask(
        key_stochastic_mask,
        semantic_key,
        "key_stochastic_mask",
    )
    query_indices = tuple(
        torch.nonzero(query_mask.reshape(-1), as_tuple=False)
        .flatten()
        .tolist()
    )
    key_indices = tuple(
        torch.nonzero(key_mask.reshape(-1), as_tuple=False)
        .flatten()
        .tolist()
    )
    return BitLayout(
        tuple(query.shape),
        tuple(key.shape),
        query_indices,
        key_indices,
    )


def _validate_temperature(bit_temperature: float) -> float:
    bit_temperature = float(bit_temperature)
    if not math.isfinite(bit_temperature) or bit_temperature <= 0.0:
        raise ValueError("bit_temperature must be finite and positive")
    return bit_temperature


def _all_assignments(
    bit_count: int,
    max_bits: int,
    device: torch.device,
) -> Tensor:
    if bit_count > int(max_bits):
        raise ValueError(
            f"refusing to enumerate {bit_count} bits; max_bits={max_bits}"
        )
    state_count = 1 << bit_count
    if bit_count == 0:
        return torch.empty(state_count, 0, dtype=torch.bool, device=device)
    state_ids = torch.arange(state_count, dtype=torch.int64, device=device)
    shifts = torch.arange(bit_count, dtype=torch.int64, device=device)
    return ((state_ids[:, None] >> shifts[None, :]) & 1).bool()


def _materialize_assignments(
    query: Tensor,
    key: Tensor,
    layout: BitLayout,
    assignments: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    state_count = assignments.size(0)
    symbols = torch.where(
        assignments,
        torch.ones((), dtype=query.dtype, device=query.device),
        -torch.ones((), dtype=query.dtype, device=query.device),
    )
    query_states = _hard_sign(query.detach()).expand(
        state_count, *query.shape[1:]
    ).clone()
    key_states = _hard_sign(key.detach()).expand(
        state_count, *key.shape[1:]
    ).clone()
    if layout.query_indices:
        query_states.reshape(state_count, -1)[
            :, list(layout.query_indices)
        ] = symbols[:, : layout.query_bit_count]
    if layout.key_indices:
        key_states.reshape(state_count, -1)[
            :, list(layout.key_indices)
        ] = symbols[:, layout.query_bit_count :]
    return query_states, key_states, symbols


def _evaluate_assignments(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout: BitLayout,
    assignments: Tensor,
    max_suffix_length: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    query_states, key_states, symbols = _materialize_assignments(
        query,
        key,
        layout,
        assignments,
    )
    value_states = value.detach().expand(
        assignments.size(0), *value.shape[1:]
    )
    outputs, _, routes, _ = _hard_route_forward(
        query_states,
        key_states,
        value_states,
        int(max_suffix_length),
    )
    return outputs, routes, query_states, key_states, symbols


def _hard_state_table(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
    max_bits: int,
    query_stochastic_mask: Optional[Tensor] = None,
    key_stochastic_mask: Optional[Tensor] = None,
) -> _HardStateTable:
    layout = _make_layout(
        query,
        key,
        query_stochastic_mask,
        key_stochastic_mask,
    )
    assignments = _all_assignments(
        layout.bit_count,
        max_bits,
        query.device,
    )
    outputs, routes, _, _, symbols = _evaluate_assignments(
        query,
        key,
        value,
        layout,
        assignments,
        max_suffix_length,
    )
    return _HardStateTable(
        layout=layout,
        assignments=assignments,
        symbols=symbols,
        outputs=outputs,
        routes=routes,
    )


def _state_scalars(outputs: Tensor, grad_output: Tensor) -> Tensor:
    return (
        outputs * grad_output.detach().expand(outputs.size(0), -1, -1, -1)
    ).flatten(1).sum(dim=1)


def _route_probabilities(
    routes: Tensor,
    weights: Tensor,
    sequence_length: int,
) -> Tensor:
    one_hot = F.one_hot(
        routes.permute(0, 2, 1),
        num_classes=sequence_length,
    ).to(weights.dtype)
    probabilities = torch.einsum("s,stha->tha", weights, one_hot)
    return probabilities.unsqueeze(0)


def exact_shared_bit_oracle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    max_suffix_length: int = 32,
    bit_temperature: float = 1.0,
    max_bits: int = 20,
    query_stochastic_mask: Optional[Tensor] = None,
    key_stochastic_mask: Optional[Tensor] = None,
) -> SharedBitOracleResult:
    """Enumerate the exact fixed-upstream VJP of stochastic hard ROSA."""

    _validate_inputs(query, key, value, grad_output)
    bit_temperature = _validate_temperature(bit_temperature)
    table = _hard_state_table(
        query,
        key,
        value,
        max_suffix_length,
        max_bits,
        query_stochastic_mask,
        key_stochastic_mask,
    )
    logits = table.layout.gather(query, key).detach()
    phi = (logits / bit_temperature).requires_grad_()
    if table.layout.bit_count:
        log_components = torch.where(
            table.assignments,
            F.logsigmoid(phi).unsqueeze(0),
            F.logsigmoid(-phi).unsqueeze(0),
        )
        log_weights = log_components.sum(dim=-1)
    else:
        log_components = torch.empty(
            table.assignments.size(0),
            0,
            dtype=query.dtype,
            device=query.device,
        )
        log_weights = torch.zeros(
            table.assignments.size(0),
            dtype=query.dtype,
            device=query.device,
        )
    weights = log_weights.exp()
    scalars = _state_scalars(table.outputs, grad_output)
    expected_scalar = torch.dot(weights, scalars)
    expected_output = torch.einsum(
        "s,sthd->thd",
        weights,
        table.outputs,
    ).unsqueeze(0)

    if table.layout.bit_count:
        (gradient_phi,) = torch.autograd.grad(expected_scalar, phi)
        bit_gradient = gradient_phi / bit_temperature

        detached_components = log_components.detach()
        log_other = (
            detached_components.sum(dim=-1, keepdim=True)
            - detached_components
        )
        other_weights = log_other.exp()
        scalar_columns = scalars.unsqueeze(-1)
        positive = (
            other_weights
            * scalar_columns
            * table.assignments.to(query.dtype)
        ).sum(dim=0)
        negative = (
            other_weights
            * scalar_columns
            * (~table.assignments).to(query.dtype)
        ).sum(dim=0)
        probabilities = torch.sigmoid(phi.detach())
        local_gradient = (
            probabilities
            * (1.0 - probabilities)
            * (positive - negative)
            / bit_temperature
        )
    else:
        bit_gradient = logits
        local_gradient = logits

    query_gradient, key_gradient = table.layout.scatter(
        bit_gradient.detach(),
        query,
        key,
    )
    local_query_gradient, local_key_gradient = table.layout.scatter(
        local_gradient.detach(),
        query,
        key,
    )
    return SharedBitOracleResult(
        expected_output=expected_output.detach(),
        expected_scalar=expected_scalar.detach(),
        route_probabilities=_route_probabilities(
            table.routes,
            weights.detach(),
            query.size(1),
        ),
        query_gradient=query_gradient,
        key_gradient=key_gradient,
        query_local_expectation_gradient=local_query_gradient,
        key_local_expectation_gradient=local_key_gradient,
        bit_gradient=bit_gradient.detach(),
        local_expectation_bit_gradient=local_gradient.detach(),
        probability_sum=float(weights.detach().sum()),
        state_count=table.assignments.size(0),
        layout=table.layout,
    )


def _product(values: Sequence[Tensor], one: Tensor) -> Tensor:
    result = one
    for value in values:
        result = result * value
    return result


def _mean_field_route_probabilities(
    query: Tensor,
    key: Tensor,
    max_suffix_length: int,
    bit_temperature: float,
    query_stochastic_mask: Tensor,
    key_stochastic_mask: Tensor,
) -> Tensor:
    sequence_length = query.size(1)
    one = query.new_tensor(1.0)
    zero = query.new_tensor(0.0)
    query_soft_probability = torch.sigmoid(
        query[0, :, 0, :] / bit_temperature
    )
    key_soft_probability = torch.sigmoid(
        key[0, :, 0, :] / bit_temperature
    )
    query_hard_probability = (_hard_sign(query)[0, :, 0, :] + 1.0) * 0.5
    key_hard_probability = (_hard_sign(key)[0, :, 0, :] + 1.0) * 0.5
    query_probability = torch.where(
        query_stochastic_mask[0, :, 0, :],
        query_soft_probability,
        query_hard_probability,
    )
    key_probability = torch.where(
        key_stochastic_mask[0, :, 0, :],
        key_soft_probability,
        key_hard_probability,
    )
    rows = []
    for query_index in range(sequence_length):
        if query_index == 0:
            rows.append(torch.stack([one] + [zero] * (sequence_length - 1)))
            continue
        candidate_pmfs: list[list[Tensor]] = []
        candidate_survivals: list[list[Tensor]] = []
        for route_index in range(1, query_index + 1):
            horizon = min(int(max_suffix_length), route_index)
            gates = []
            for offset in range(horizon):
                q_probability = query_probability[query_index - offset]
                k_probability = key_probability[route_index - 1 - offset]
                bit_agreement = (
                    q_probability * k_probability
                    + (1.0 - q_probability) * (1.0 - k_probability)
                )
                gates.append(bit_agreement.prod())
            survival = [one]
            for gate in gates:
                survival.append(survival[-1] * gate)
            pmf = [
                survival[length] * (1.0 - gates[length])
                for length in range(horizon)
            ]
            pmf.append(survival[horizon])
            candidate_survivals.append(survival)
            candidate_pmfs.append(pmf)

        def cdf(candidate: int, length: int) -> Tensor:
            horizon = len(candidate_pmfs[candidate]) - 1
            if length < 0:
                return zero
            if length >= horizon:
                return one
            return 1.0 - candidate_survivals[candidate][length + 1]

        null_mass = _product(
            [pmf[0] for pmf in candidate_pmfs],
            one,
        )
        route_masses = []
        for candidate in range(query_index):
            horizon = len(candidate_pmfs[candidate]) - 1
            terms = []
            for length in range(1, horizon + 1):
                earlier = _product(
                    [cdf(other, length) for other in range(candidate)],
                    one,
                )
                later = _product(
                    [
                        cdf(other, length - 1)
                        for other in range(candidate + 1, query_index)
                    ],
                    one,
                )
                terms.append(
                    candidate_pmfs[candidate][length] * earlier * later
                )
            route_masses.append(sum(terms, zero))
        padding = [zero] * (sequence_length - query_index - 1)
        rows.append(torch.stack([null_mass] + route_masses + padding))
    return torch.stack(rows).view(1, sequence_length, 1, sequence_length)


def mean_field_winner_oracle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    max_suffix_length: int = 32,
    bit_temperature: float = 1.0,
    query_stochastic_mask: Optional[Tensor] = None,
    key_stochastic_mask: Optional[Tensor] = None,
) -> MeanFieldResult:
    """Differentiate independent candidate-length winner marginals."""

    _validate_inputs(query, key, value, grad_output)
    bit_temperature = _validate_temperature(bit_temperature)
    query_leaf = query.detach().requires_grad_()
    key_leaf = key.detach().requires_grad_()
    semantic_query, semantic_key = _semantic_masks(query, key)
    query_mask = _normalize_mask(
        query_stochastic_mask,
        semantic_query,
        "query_stochastic_mask",
    )
    key_mask = _normalize_mask(
        key_stochastic_mask,
        semantic_key,
        "key_stochastic_mask",
    )
    probabilities = _mean_field_route_probabilities(
        query_leaf,
        key_leaf,
        max_suffix_length,
        bit_temperature,
        query_mask,
        key_mask,
    )
    route_values = _hard_sign(value.detach())[0, :, 0, :].clone()
    route_values[0] = 0.0
    expected_output = torch.einsum(
        "btha,ad->bthd",
        probabilities,
        route_values,
    )
    if not bool(query_mask.any()) and not bool(key_mask.any()):
        return MeanFieldResult(
            expected_output=expected_output.detach(),
            route_probabilities=probabilities.detach(),
            query_gradient=torch.zeros_like(query),
            key_gradient=torch.zeros_like(key),
        )
    scalar = (expected_output * grad_output.detach()).sum()
    query_gradient, key_gradient = torch.autograd.grad(
        scalar,
        (query_leaf, key_leaf),
    )
    return MeanFieldResult(
        expected_output=expected_output.detach(),
        route_probabilities=probabilities.detach(),
        query_gradient=query_gradient.detach(),
        key_gradient=key_gradient.detach(),
    )


def _evaluate_scalar_assignments(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    layout: BitLayout,
    assignments: Tensor,
    max_suffix_length: int,
) -> Tensor:
    outputs, _, _, _, _ = _evaluate_assignments(
        query,
        key,
        value,
        layout,
        assignments,
        max_suffix_length,
    )
    return _state_scalars(outputs, grad_output)


def arm_disarm_samples(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    max_suffix_length: int = 32,
    bit_temperature: float = 1.0,
    sample_count: int = 4096,
    seed: int = 0,
    uniforms: Optional[Tensor] = None,
    query_stochastic_mask: Optional[Tensor] = None,
    key_stochastic_mask: Optional[Tensor] = None,
) -> ArmDisarmSamples:
    """Return ARM and DisARM samples for one shared-bit objective."""

    _validate_inputs(query, key, value, grad_output)
    bit_temperature = _validate_temperature(bit_temperature)
    layout = _make_layout(
        query,
        key,
        query_stochastic_mask,
        key_stochastic_mask,
    )
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    logits = layout.gather(query, key).detach()
    phi = logits / bit_temperature
    probabilities = torch.sigmoid(phi)
    if uniforms is None:
        generator = torch.Generator(device=query.device).manual_seed(int(seed))
        uniforms = torch.rand(
            sample_count,
            layout.bit_count,
            dtype=query.dtype,
            device=query.device,
            generator=generator,
        )
    elif uniforms.shape != (sample_count, layout.bit_count):
        raise ValueError(
            f"uniforms must have shape {(sample_count, layout.bit_count)}"
        )
    else:
        uniforms = uniforms.to(dtype=query.dtype, device=query.device)

    if layout.bit_count == 0:
        empty = uniforms.clone()
        return ArmDisarmSamples(empty, empty, layout)

    direct = uniforms < probabilities
    antithetic = (1.0 - uniforms) < probabilities
    direct_scalar = _evaluate_scalar_assignments(
        query,
        key,
        value,
        grad_output,
        layout,
        direct,
        max_suffix_length,
    )
    antithetic_scalar = _evaluate_scalar_assignments(
        query,
        key,
        value,
        grad_output,
        layout,
        antithetic,
        max_suffix_length,
    )
    arm = (
        (antithetic_scalar - direct_scalar).unsqueeze(-1)
        * (uniforms - 0.5)
        / bit_temperature
    )
    disarm_sign = torch.where(
        antithetic,
        -torch.ones((), dtype=query.dtype, device=query.device),
        torch.ones((), dtype=query.dtype, device=query.device),
    )
    disarm = (
        0.5
        * (direct_scalar - antithetic_scalar).unsqueeze(-1)
        * disarm_sign
        * (direct != antithetic).to(query.dtype)
        * torch.sigmoid(phi.abs())
        / bit_temperature
    )
    return ArmDisarmSamples(arm, disarm, layout)


def production_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = 1.0,
    mismatch_scale: float = 3.0,
) -> tuple[Tensor, Tensor]:
    """Return the frozen production Q/K VJP for a fixed upstream tensor."""

    _validate_inputs(query, key, value, grad_output)
    query_leaf = query.detach().requires_grad_()
    key_leaf = key.detach().requires_grad_()
    output = rosa_soft_reference(
        query_leaf,
        key_leaf,
        value.detach(),
        max_suffix_length=max_suffix_length,
        scale=scale,
        dropout_p=0.0,
        mismatch_scale=mismatch_scale,
    )
    gradients = torch.autograd.grad(
        (output * grad_output.detach()).sum(),
        (query_leaf, key_leaf),
    )
    return gradients[0].detach(), gradients[1].detach()


def exact_bitflip_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    max_suffix_length: int = 32,
    query_stochastic_mask: Optional[Tensor] = None,
    key_stochastic_mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, BitLayout]:
    """Return deterministic one-bit hard counterfactual pseudo-gradients."""

    _validate_inputs(query, key, value, grad_output)
    layout = _make_layout(
        query,
        key,
        query_stochastic_mask,
        key_stochastic_mask,
    )
    base_assignments = layout.gather(
        _hard_sign(query),
        _hard_sign(key),
    ) > 0
    if layout.bit_count == 0:
        bit_gradient = query.new_empty(0)
    else:
        assignments = base_assignments.expand(
            layout.bit_count, -1
        ).clone()
        indices = torch.arange(layout.bit_count, device=query.device)
        assignments[indices, indices] = ~assignments[indices, indices]
        base_output, _, _, _ = _hard_route_forward(
            query.detach(),
            key.detach(),
            value.detach(),
            int(max_suffix_length),
        )
        flipped_outputs, _, _, _, _ = _evaluate_assignments(
            query,
            key,
            value,
            layout,
            assignments,
            max_suffix_length,
        )
        deltas = (
            (flipped_outputs - base_output)
            * grad_output.detach().expand(layout.bit_count, -1, -1, -1)
        ).flatten(1).sum(dim=1)
        base_symbols = torch.where(
            base_assignments,
            torch.ones_like(deltas),
            -torch.ones_like(deltas),
        )
        bit_gradient = -base_symbols * deltas
    query_gradient, key_gradient = layout.scatter(
        bit_gradient,
        query,
        key,
    )
    return query_gradient, key_gradient, bit_gradient, layout


def bitflip_residual_samples(
    surrogate_bit_gradient: Tensor,
    bitflip_bit_gradient: Tensor,
    *,
    sample_count: int,
    flips_per_sample: int = 2,
    seed: int = 0,
    sampled_indices: Optional[Tensor] = None,
) -> Tensor:
    """Use a dense baseline plus an unbiased sampled bitflip residual."""

    if surrogate_bit_gradient.ndim != 1:
        raise ValueError("surrogate_bit_gradient must be a vector")
    if bitflip_bit_gradient.shape != surrogate_bit_gradient.shape:
        raise ValueError("bitflip and surrogate vectors must have one shape")
    if sample_count <= 0 or flips_per_sample <= 0:
        raise ValueError("sample_count and flips_per_sample must be positive")
    bit_count = surrogate_bit_gradient.numel()
    if bit_count == 0:
        return surrogate_bit_gradient.expand(sample_count, 0).clone()
    if flips_per_sample > bit_count:
        raise ValueError("flips_per_sample cannot exceed the bit count")
    if sampled_indices is None:
        generator = torch.Generator(
            device=surrogate_bit_gradient.device
        ).manual_seed(int(seed))
        sampled_indices = _sample_unique_indices(
            sample_count,
            bit_count,
            flips_per_sample,
            surrogate_bit_gradient,
            generator,
        )
    else:
        if sampled_indices.shape != (sample_count, flips_per_sample):
            raise ValueError(
                "sampled_indices must have shape "
                f"{(sample_count, flips_per_sample)}"
            )
        if sampled_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("sampled_indices must contain integers")
        sampled_indices = sampled_indices.to(device=surrogate_bit_gradient.device)
        if bool(((sampled_indices < 0) | (sampled_indices >= bit_count)).any()):
            raise ValueError("sampled_indices contains an invalid bit index")
        sorted_indices = sampled_indices.sort(dim=1).values
        if flips_per_sample > 1 and bool(
            (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
        ):
            raise ValueError(
                "sampled_indices must be unique within each sample"
            )
    residual = bitflip_bit_gradient - surrogate_bit_gradient
    sampled_residual = residual[sampled_indices]
    correction = torch.zeros(
        sample_count,
        bit_count,
        dtype=surrogate_bit_gradient.dtype,
        device=surrogate_bit_gradient.device,
    )
    correction.scatter_add_(
        1,
        sampled_indices,
        sampled_residual * (bit_count / flips_per_sample),
    )
    return surrogate_bit_gradient.unsqueeze(0) + correction


def _sample_unique_indices(
    sample_count: int,
    bit_count: int,
    bits_per_sample: int,
    template: Tensor,
    generator: torch.Generator,
) -> Tensor:
    priorities = torch.rand(
        sample_count,
        bit_count,
        dtype=template.dtype,
        generator=generator,
        device=template.device,
    )
    return priorities.topk(
        bits_per_sample,
        dim=1,
        largest=False,
    ).indices


def sampled_bitflip_residual_vjp(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    surrogate_query_gradient: Tensor,
    surrogate_key_gradient: Tensor,
    *,
    max_suffix_length: int = 32,
    flips_per_sample: int = 2,
    seed: int = 0,
    query_stochastic_mask: Optional[Tensor] = None,
    key_stochastic_mask: Optional[Tensor] = None,
) -> SampledBitflipResult:
    """Correct a dense VJP using only sampled exact bit counterfactuals."""

    _validate_inputs(query, key, value, grad_output)
    if surrogate_query_gradient.shape != query.shape:
        raise ValueError("surrogate_query_gradient must match query")
    if surrogate_key_gradient.shape != key.shape:
        raise ValueError("surrogate_key_gradient must match key")
    if (
        surrogate_query_gradient.device != query.device
        or surrogate_key_gradient.device != key.device
        or surrogate_query_gradient.dtype != query.dtype
        or surrogate_key_gradient.dtype != key.dtype
    ):
        raise ValueError(
            "surrogate gradients must share their input device and dtype"
        )
    layout = _make_layout(
        query,
        key,
        query_stochastic_mask,
        key_stochastic_mask,
    )
    if flips_per_sample <= 0 or flips_per_sample > layout.bit_count:
        raise ValueError(
            "flips_per_sample must be positive and at most the bit count"
        )
    surrogate_bits = layout.gather(
        surrogate_query_gradient,
        surrogate_key_gradient,
    ).detach()
    generator = torch.Generator(device=query.device).manual_seed(int(seed))
    sampled_indices = _sample_unique_indices(
        1,
        layout.bit_count,
        flips_per_sample,
        surrogate_bits,
        generator,
    )[0]

    logits = layout.gather(query, key).detach()
    base_assignments = _hard_sign(logits) > 0
    assignments = base_assignments.expand(flips_per_sample, -1).clone()
    sample_rows = torch.arange(flips_per_sample, device=query.device)
    assignments[sample_rows, sampled_indices] = ~assignments[
        sample_rows,
        sampled_indices,
    ]
    base_output, _, _, _ = _hard_route_forward(
        query.detach(),
        key.detach(),
        value.detach(),
        int(max_suffix_length),
    )
    flipped_outputs, _, _, _, _ = _evaluate_assignments(
        query,
        key,
        value,
        layout,
        assignments,
        max_suffix_length,
    )
    loss_deltas = (
        (flipped_outputs - base_output)
        * grad_output.detach().expand(flips_per_sample, -1, -1, -1)
    ).flatten(1).sum(dim=1)
    base_symbols = _hard_sign(logits)
    sampled_exact = -base_symbols[sampled_indices] * loss_deltas
    correction = torch.zeros_like(surrogate_bits)
    correction.scatter_add_(
        0,
        sampled_indices,
        (sampled_exact - surrogate_bits[sampled_indices])
        * (layout.bit_count / flips_per_sample),
    )
    bit_gradient = surrogate_bits + correction
    query_gradient, key_gradient = layout.scatter(
        bit_gradient,
        query,
        key,
    )
    return SampledBitflipResult(
        query_gradient=query_gradient,
        key_gradient=key_gradient,
        bit_gradient=bit_gradient,
        sampled_indices=sampled_indices,
        layout=layout,
    )


def exact_margin_edit_oracle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    max_suffix_length: int = 32,
    eta: float = 1.0,
    max_bits: int = 20,
    query_stochastic_mask: Optional[Tensor] = None,
    key_stochastic_mask: Optional[Tensor] = None,
) -> MarginEditResult:
    """Solve the exact joint bit/route loss-augmented MAP target."""

    _validate_inputs(query, key, value, grad_output)
    eta = float(eta)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive")
    table = _hard_state_table(
        query,
        key,
        value,
        max_suffix_length,
        max_bits,
        query_stochastic_mask,
        key_stochastic_mask,
    )
    logits = table.layout.gather(query, key).detach()
    base_symbols = _hard_sign(logits)
    base_output, _, _, _ = _hard_route_forward(
        query.detach(),
        key.detach(),
        value.detach(),
        int(max_suffix_length),
    )
    state_costs = (
        (table.outputs - base_output)
        * grad_output.detach().expand(table.outputs.size(0), -1, -1, -1)
    ).flatten(1).sum(dim=1)
    state_scores = table.symbols @ logits
    objectives = state_scores - eta * state_costs
    max_objective = objectives.max()
    maximizers = objectives == max_objective
    edit_counts = (table.symbols != base_symbols.unsqueeze(0)).sum(dim=-1)
    minimum_edits = edit_counts.masked_fill(
        ~maximizers,
        table.layout.bit_count + 1,
    ).min()
    target_index = int(
        torch.nonzero(
            maximizers & (edit_counts == minimum_edits),
            as_tuple=False,
        )[0, 0]
    )
    target_symbols = table.symbols[target_index]
    bit_gradient = (base_symbols - target_symbols) / eta
    query_gradient, key_gradient = table.layout.scatter(
        bit_gradient,
        query,
        key,
    )
    target_query, target_key, _ = _materialize_assignments(
        query,
        key,
        table.layout,
        table.assignments[target_index : target_index + 1],
    )
    base_score = torch.dot(base_symbols, logits)
    target_score = state_scores[target_index]
    objective_gain = objectives[target_index] - base_score
    if float(objective_gain) < -1e-10:
        raise RuntimeError("loss-augmented solver returned a worse objective")
    return MarginEditResult(
        query_gradient=query_gradient,
        key_gradient=key_gradient,
        target_query_symbols=target_query,
        target_key_symbols=target_key,
        target_output=table.outputs[target_index : target_index + 1],
        target_routes=table.routes[target_index : target_index + 1],
        flipped_bits=int((target_symbols != base_symbols).sum()),
        margin_penalty=float(base_score - target_score),
        linearized_cost=float(state_costs[target_index]),
        objective_gain=float(objective_gain),
        state_count=table.assignments.size(0),
        layout=table.layout,
    )


def _gradient_metrics(estimate: Tensor, reference: Tensor) -> dict[str, float]:
    estimate = estimate.double().flatten()
    reference = reference.double().flatten()
    reference_norm = reference.norm()
    estimate_norm = estimate.norm()
    if float(reference_norm) == 0.0:
        return {
            "reference_norm": 0.0,
            "estimate_norm": float(estimate_norm),
            "cosine": 0.0,
            "relative_error": float(estimate_norm),
            "scale_aligned_relative_error": float(estimate_norm),
        }
    denominator = estimate_norm * reference_norm
    cosine = (
        float(torch.dot(estimate, reference) / denominator)
        if float(denominator) != 0.0
        else 0.0
    )
    relative_error = float((estimate - reference).norm() / reference_norm)
    if float(estimate.square().sum()) == 0.0:
        aligned_error = 1.0
    else:
        scale = torch.dot(estimate, reference) / estimate.square().sum()
        aligned_error = float((scale * estimate - reference).norm() / reference_norm)
    return {
        "reference_norm": float(reference_norm),
        "estimate_norm": float(estimate_norm),
        "cosine": cosine,
        "relative_error": relative_error,
        "scale_aligned_relative_error": aligned_error,
    }


def _sample_metrics(samples: Tensor, reference: Tensor) -> dict[str, float]:
    samples = samples.double()
    reference = reference.double()
    mean = samples.mean(dim=0)
    metrics = _gradient_metrics(mean, reference)
    reference_norm_sq = float(reference.square().sum())
    centered = samples - mean
    if reference_norm_sq == 0.0:
        normalized_variance = float(centered.square().sum(dim=1).mean())
        normalized_mse = float((samples - reference).square().sum(dim=1).mean())
    else:
        normalized_variance = float(
            centered.square().sum(dim=1).mean() / reference_norm_sq
        )
        normalized_mse = float(
            (samples - reference).square().sum(dim=1).mean()
            / reference_norm_sq
        )
    sample_norms = samples.norm(dim=1)
    reference_norm = reference.norm()
    valid = sample_norms > 0
    if float(reference_norm) == 0.0 or not bool(valid.any()):
        median_cosine = 0.0
        negative_cosine = 0.0
    else:
        cosines = (
            samples[valid] @ reference
            / (sample_norms[valid] * reference_norm)
        )
        median_cosine = float(cosines.median())
        negative_cosine = float((cosines < 0).double().mean())
    metrics.update(
        normalized_variance=normalized_variance,
        normalized_mse=normalized_mse,
        median_sample_cosine=median_cosine,
        negative_sample_cosine_fraction=negative_cosine,
    )
    return metrics


def _make_case(
    seed: int,
    sequence_length: int,
    qk_bits: int,
    value_bits: int,
    repeated_keys: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    shape = (1, sequence_length, 1, qk_bits)
    query = torch.randn(shape, generator=generator, dtype=torch.float64)
    key = torch.randn(shape, generator=generator, dtype=torch.float64)
    value = torch.randn(
        1,
        sequence_length,
        1,
        value_bits,
        generator=generator,
        dtype=torch.float64,
    )
    grad_output = torch.randn(
        1,
        sequence_length,
        1,
        value_bits,
        generator=generator,
        dtype=torch.float64,
    )
    if repeated_keys and sequence_length >= 3:
        key[:, 1, :, :] = key[:, 0, :, :]
        key[:, :2, :, :] = 3.0 * _hard_sign(key[:, :2, :, :])
        query[:, 2:, :, :] *= 0.35
    return query, key, value, grad_output


def _run_case(seed: int, repeated_keys: bool, args) -> dict[str, object]:
    query, key, value, grad_output = _make_case(
        seed,
        args.sequence_length,
        args.qk_bits,
        args.value_bits,
        repeated_keys,
    )
    oracle = exact_shared_bit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=args.max_suffix_length,
        bit_temperature=args.bit_temperature,
        max_bits=args.max_bits,
    )
    reference = oracle.bit_gradient
    production_query, production_key = production_vjp(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=args.max_suffix_length,
        scale=args.scale,
        mismatch_scale=args.mismatch_scale,
    )
    production_bits = oracle.layout.gather(production_query, production_key)
    mean_field = mean_field_winner_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=args.max_suffix_length,
        bit_temperature=args.bit_temperature,
    )
    mean_field_bits = oracle.layout.gather(
        mean_field.query_gradient,
        mean_field.key_gradient,
    )
    stochastic = arm_disarm_samples(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=args.max_suffix_length,
        bit_temperature=args.bit_temperature,
        sample_count=args.sample_count,
        seed=seed + 1009,
    )
    bitflip_query, bitflip_key, bitflip_bits, bitflip_layout = exact_bitflip_vjp(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=args.max_suffix_length,
    )
    if bitflip_layout != oracle.layout:
        raise RuntimeError("bitflip and stochastic layouts differ")
    residual_samples = bitflip_residual_samples(
        production_bits,
        bitflip_bits,
        sample_count=args.sample_count,
        flips_per_sample=args.hard_evaluations,
        seed=seed + 2027,
    )
    margin = exact_margin_edit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=args.max_suffix_length,
        eta=args.margin_eta,
        max_bits=args.max_bits,
    )
    route_difference = (
        oracle.route_probabilities - mean_field.route_probabilities
    ).abs()
    row_total_variation = 0.5 * route_difference.sum(dim=-1)
    return {
        "seed": seed,
        "case": "repeated_keys" if repeated_keys else "random",
        "state_count": oracle.state_count,
        "probability_sum": oracle.probability_sum,
        "local_expectation_max_abs_error": float(
            (oracle.bit_gradient - oracle.local_expectation_bit_gradient)
            .abs()
            .max()
        ),
        "mean_field_route_tv_mean": float(row_total_variation.mean()),
        "mean_field_route_tv_max": float(row_total_variation.max()),
        "mean_field_output_relative_error": float(
            (mean_field.expected_output - oracle.expected_output).norm()
            / oracle.expected_output.norm().clamp_min(1e-30)
        ),
        "estimators_against_stochastic_oracle": {
            "production": _gradient_metrics(production_bits, reference),
            "mean_field": _gradient_metrics(mean_field_bits, reference),
            "arm": _sample_metrics(stochastic.arm_bit_gradients, reference),
            "disarm": _sample_metrics(
                stochastic.disarm_bit_gradients,
                reference,
            ),
            "full_bitflip": _gradient_metrics(bitflip_bits, reference),
            "bitflip_residual_mean": _gradient_metrics(
                residual_samples.mean(dim=0),
                reference,
            ),
        },
        "bitflip_residual_against_bitflip": _sample_metrics(
            residual_samples,
            bitflip_bits,
        ),
        "margin_edit": {
            "eta": args.margin_eta,
            "flipped_bits": margin.flipped_bits,
            "margin_penalty": margin.margin_penalty,
            "linearized_cost": margin.linearized_cost,
            "objective_gain": margin.objective_gain,
            "route_changes": int(
                (margin.target_routes != _hard_route_forward(
                    query,
                    key,
                    value,
                    args.max_suffix_length,
                )[2]).sum()
            ),
        },
        "diagnostic_norms": {
            "bitflip_query": float(bitflip_query.norm()),
            "bitflip_key": float(bitflip_key.norm()),
        },
    }


def _path_values(report: dict, path: Sequence[str]) -> list[float]:
    values = []
    for case in report["cases"]:
        value = case
        for key in path:
            value = value[key]
        values.append(float(value))
    return values


def _summarize(report: dict) -> dict[str, object]:
    estimators = (
        "production",
        "mean_field",
        "arm",
        "disarm",
        "full_bitflip",
        "bitflip_residual_mean",
    )
    summary: dict[str, object] = {
        "case_count": len(report["cases"]),
        "local_expectation_max_abs_error": max(
            _path_values(report, ("local_expectation_max_abs_error",))
        ),
        "mean_field_route_tv_mean": statistics.mean(
            _path_values(report, ("mean_field_route_tv_mean",))
        ),
        "mean_field_route_tv_max": max(
            _path_values(report, ("mean_field_route_tv_max",))
        ),
        "estimators_against_stochastic_oracle": {},
    }
    estimator_summary = summary["estimators_against_stochastic_oracle"]
    assert isinstance(estimator_summary, dict)
    for estimator in estimators:
        estimator_summary[estimator] = {
            metric: statistics.mean(
                _path_values(
                    report,
                    (
                        "estimators_against_stochastic_oracle",
                        estimator,
                        metric,
                    ),
                )
            )
            for metric in (
                "cosine",
                "relative_error",
                "scale_aligned_relative_error",
            )
        }
    for estimator in ("arm", "disarm"):
        estimator_summary[estimator].update(
            {
                metric: statistics.mean(
                    _path_values(
                        report,
                        (
                            "estimators_against_stochastic_oracle",
                            estimator,
                            metric,
                        ),
                    )
                )
                for metric in (
                    "normalized_variance",
                    "normalized_mse",
                    "median_sample_cosine",
                    "negative_sample_cosine_fraction",
                )
            }
        )
    summary["stochastic_oracle_mean_norm"] = statistics.mean(
        _path_values(
            report,
            (
                "estimators_against_stochastic_oracle",
                "production",
                "reference_norm",
            ),
        )
    )
    summary["arm_normalized_variance"] = statistics.mean(
        _path_values(
            report,
            (
                "estimators_against_stochastic_oracle",
                "arm",
                "normalized_variance",
            ),
        )
    )
    summary["disarm_normalized_variance"] = statistics.mean(
        _path_values(
            report,
            (
                "estimators_against_stochastic_oracle",
                "disarm",
                "normalized_variance",
            ),
        )
    )
    summary["bitflip_residual_normalized_variance"] = statistics.mean(
        _path_values(
            report,
            ("bitflip_residual_against_bitflip", "normalized_variance"),
        )
    )
    summary["bitflip_residual_against_bitflip"] = {
        metric: statistics.mean(
            _path_values(
                report,
                ("bitflip_residual_against_bitflip", metric),
            )
        )
        for metric in (
            "cosine",
            "relative_error",
            "scale_aligned_relative_error",
            "normalized_variance",
        )
    }
    summary["margin_edit_changed_fraction"] = statistics.mean(
        float(case["margin_edit"]["flipped_bits"] > 0)
        for case in report["cases"]
    )
    summary["margin_edit_mean_flipped_bits"] = statistics.mean(
        _path_values(report, ("margin_edit", "flipped_bits"))
    )
    summary["margin_edit_mean_linearized_cost"] = statistics.mean(
        _path_values(report, ("margin_edit", "linearized_cost"))
    )
    summary["margin_edit_mean_objective_gain"] = statistics.mean(
        _path_values(report, ("margin_edit", "objective_gain"))
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare tiny globally consistent hard-ROSA estimators",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--qk-bits", type=int, default=2)
    parser.add_argument("--value-bits", type=int, default=3)
    parser.add_argument("--max-suffix-length", type=int, default=3)
    parser.add_argument("--max-bits", type=int, default=20)
    parser.add_argument("--bit-temperature", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument(
        "--hard-evaluations",
        type=int,
        default=2,
        help="sampled bit flips per residual estimate",
    )
    parser.add_argument("--margin-eta", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("random", "repeated_keys"),
        default=["random", "repeated_keys"],
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    report: dict[str, object] = {
        "configuration": vars(args),
        "objective": (
            "fixed-upstream VJP of expected exact hard ROSA under one "
            "globally shared factorial Bernoulli Q/K bit assignment"
        ),
        "hard_evaluation_budget": {
            "arm_per_sample": 2,
            "disarm_per_sample": 2,
            "bitflip_residual_per_sample": args.hard_evaluations,
        },
        "cases": [],
    }
    cases = report["cases"]
    assert isinstance(cases, list)
    for seed in args.seeds:
        for case in args.cases:
            cases.append(_run_case(seed, case == "repeated_keys", args))
    summary = _summarize(report)
    summary["by_case"] = {
        case_name: _summarize(
            {
                "cases": [
                    case for case in cases if case["case"] == case_name
                ]
            }
        )
        for case_name in args.cases
    }
    report["summary"] = summary
    output = report
    if args.summary_only:
        output = {
            "configuration": report["configuration"],
            "hard_evaluation_budget": report["hard_evaluation_budget"],
            "objective": report["objective"],
            "summary": report["summary"],
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
