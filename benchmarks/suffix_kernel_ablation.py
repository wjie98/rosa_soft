"""Diagnostics and fitting gates for factorized suffix-kernel VJPs.

This benchmark keeps three questions separate:

* Does the exact independent Q/K feature state induce a useful route field?
* How much error and seed variance does fixed-width TensorSketch introduce?
* Does either field fit the same hard-forward task as production and bitflip?

All proxy estimators retain exact hard ROSA forward semantics.  The dense
candidate matrices in this file are diagnostics only and are not used by the
linear-memory proxy operator.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft  # noqa: E402
from benchmarks.estimator_fit_ablation import (  # noqa: E402
    _ResearchFitLM,
    rosa_soft_exact_bitflip,
)
from benchmarks.fast_weight_proxy_ablation import (  # noqa: E402
    _gradient_metrics,
    _parameter_gradient_norms,
)
from benchmarks.long_suffix_extrapolation import (  # noqa: E402
    DenseContext,
    dense_hard_route,
    make_gate_spec,
    make_query_logits,
    materialize_context,
)
from benchmarks.suffix_kernel_proxy import (  # noqa: E402
    READ_RULES,
    REPRESENTATIONS,
    ROUTE_KERNELS,
    _suffix_kernel_features,
    rosa_suffix_kernel_proxy,
)
from examples.fit_soft_reference import (  # noqa: E402
    historical_target_mask,
    loss_and_accuracy,
    make_copy_tokens,
)
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    ROSA_SOFT_NULL_ROUTE_SCORE,
)
from rosa_soft.soft_reference import (  # noqa: E402
    _hard_sign_with_softsign_vjp,
    _suffix_score_utility,
    rosa_soft_reference,
)


Estimator = Callable[..., Tensor]
BASELINES = ("production", "bitflip")
PROXY_ESTIMATORS = tuple(
    f"{representation}_{route_kernel}_{read_rule}"
    for representation in REPRESENTATIONS
    for route_kernel in ROUTE_KERNELS
    for read_rule in READ_RULES
)
ESTIMATORS = BASELINES + PROXY_ESTIMATORS
DIAGNOSTIC_ESTIMATORS = (
    "production",
    "bitflip",
    "exact_raw_additive",
    "exact_quadratic_additive",
    "exact_level_quadratic_additive",
)


@dataclass(frozen=True)
class ProxyConfig:
    representation: str
    route_kernel: str
    read_rule: str


def _proxy_config(estimator: str) -> ProxyConfig:
    for representation in REPRESENTATIONS:
        prefix = representation + "_"
        if not estimator.startswith(prefix):
            continue
        for read_rule in READ_RULES:
            suffix = "_" + read_rule
            if not estimator.endswith(suffix):
                continue
            route_kernel = estimator[len(prefix) : -len(suffix)]
            if route_kernel in ROUTE_KERNELS:
                return ProxyConfig(representation, route_kernel, read_rule)
    raise ValueError(f"not a suffix-kernel estimator: {estimator}")


def _production_operator(device: torch.device) -> Estimator:
    if device.type == "cuda":
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("the production CUDA operator is unavailable")
        return rosa_soft.rosa_soft
    return rosa_soft_reference


def _make_operator(
    estimator: str,
    *,
    device: torch.device,
    fingerprint_length: int,
    sketch_dim: int,
    sketch_count: int,
    sketch_seed: int,
) -> Estimator:
    if estimator == "production":
        return _production_operator(device)
    if estimator == "bitflip":
        return rosa_soft_exact_bitflip
    config = _proxy_config(estimator)

    def operator(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        max_suffix_length: int,
        scale: float,
        mismatch_scale: float,
    ) -> Tensor:
        del scale
        return rosa_suffix_kernel_proxy(
            query,
            key,
            value,
            representation=config.representation,
            route_kernel=config.route_kernel,
            read_rule=config.read_rule,
            max_suffix_length=max_suffix_length,
            fingerprint_length=fingerprint_length,
            mismatch_scale=mismatch_scale,
            sketch_dim=sketch_dim,
            sketch_count=sketch_count,
            sketch_seed=sketch_seed,
        )

    return operator


def _candidate_suffix_gates(
    query_logits: Tensor,
    context: DenseContext,
    *,
    window: int,
    mismatch_scale: float,
) -> Tensor:
    query_symbols = _hard_sign_with_softsign_vjp(query_logits)
    probes, context_length, _ = context.key_signs.shape
    local_gates = []
    for local_position in range(window):
        shift = window - 1 - local_position
        valid_keys = context.key_signs[:, : context_length - shift]
        mismatch_rate = 0.5 * (
            1.0
            - query_symbols[:, local_position].unsqueeze(1) * valid_keys
        ).mean(dim=-1)
        gate = torch.exp(-float(mismatch_scale) * mismatch_rate)
        if shift:
            gate = torch.nn.functional.pad(gate, (shift, 0), value=0.0)
        local_gates.append(gate)
    gates = torch.stack(local_gates, dim=-1)
    return torch.flip(gates, dims=(-1,)).cumprod(dim=-1).sum(dim=-1)


def _production_distribution(
    query_logits: Tensor,
    context: DenseContext,
    *,
    window: int,
    scale: float,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    raw_scores = _candidate_suffix_gates(
        query_logits,
        context,
        window=window,
        mismatch_scale=mismatch_scale,
    )
    candidate_logits = (
        _suffix_score_utility(raw_scores) * float(scale)
        - math.log(context.key_signs.size(1))
    )
    null_logits = candidate_logits.new_full(
        (candidate_logits.size(0), 1),
        ROSA_SOFT_NULL_ROUTE_SCORE * float(scale),
    )
    probabilities = torch.softmax(
        torch.cat((null_logits, candidate_logits), dim=-1),
        dim=-1,
    )
    candidate_probabilities = probabilities[:, 1:]
    carrier = (
        candidate_probabilities.unsqueeze(-1) * context.values
    ).sum(dim=1)
    return carrier, candidate_probabilities, probabilities[:, 0]


def _proxy_distribution(
    query_logits: Tensor,
    context: DenseContext,
    *,
    config: ProxyConfig,
    window: int,
    mismatch_scale: float,
    sketch_dim: int,
    sketch_count: int,
    sketch_seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    common = {
        "representation": config.representation,
        "route_kernel": config.route_kernel,
        "fingerprint_length": window,
        "mismatch_scale": mismatch_scale,
        "sketch_dim": sketch_dim,
        "sketch_count": sketch_count,
        "sketch_seed": sketch_seed,
    }
    query_features = _suffix_kernel_features(
        query_logits.unsqueeze(2),
        **common,
    )[:, 0, -1]
    key_features = _suffix_kernel_features(
        context.key_signs.unsqueeze(2),
        **common,
    )[:, 0]
    route_weights = torch.einsum("pf,ptf->pt", query_features, key_features)
    denominator = route_weights.sum(dim=-1).clamp_min(1e-6)
    coefficients = route_weights / denominator.unsqueeze(-1)
    carrier = (coefficients.unsqueeze(-1) * context.values).sum(dim=1)
    null_mass = torch.zeros(
        route_weights.size(0),
        dtype=route_weights.dtype,
        device=route_weights.device,
    )
    return carrier, coefficients, null_mass


def _bitflip_carrier(
    spec,
    context: DenseContext,
    fault_logit: Tensor,
    *,
    logit_margin: float,
) -> Tensor:
    query = make_query_logits(spec, fault_logit, logit_margin=logit_margin)
    hard = dense_hard_route(query, context, spec.window)
    flipped_query = make_query_logits(
        spec,
        -fault_logit.detach(),
        logit_margin=logit_margin,
    )
    flipped = dense_hard_route(flipped_query, context, spec.window)
    fault_sign = torch.where(
        fault_logit.detach() > 0,
        torch.ones_like(fault_logit),
        -torch.ones_like(fault_logit),
    )
    slope = -fault_sign * (flipped.output - hard.output).detach()
    return fault_logit * slope


def _distribution_metrics(
    coefficients: Tensor,
    null_mass: Tensor,
    target_positions: Tensor,
) -> dict[str, float]:
    coefficients = coefficients.detach()
    null_mass = null_mass.detach()
    rows = torch.arange(coefficients.size(0), device=coefficients.device)
    target = coefficients[rows, target_positions]
    distractors = coefficients.clone()
    distractors[rows, target_positions] = 0.0
    absolute = coefficients.abs()
    absolute_total = absolute.sum(dim=-1) + null_mass.abs()
    squared_total = coefficients.square().sum(dim=-1) + null_mass.square()
    effective = absolute_total.square() / squared_total.clamp_min(1e-30)
    return {
        "mean_target_coefficient": float(target.mean()),
        "mean_target_absolute_mass": float(
            (target.abs() / absolute_total.clamp_min(1e-30)).mean()
        ),
        "mean_max_distractor_coefficient": float(
            distractors.amax(dim=-1).mean()
        ),
        "mean_distractor_coefficient_sum": float(
            distractors.sum(dim=-1).mean()
        ),
        "mean_candidate_coefficient_sum": float(coefficients.sum(dim=-1).mean()),
        "mean_null_mass": float(null_mass.mean()),
        "mean_effective_route_count": float(effective.mean()),
        "mean_negative_candidate_fraction": float(
            coefficients.lt(0).float().mean()
        ),
    }


def run_route_diagnostic(
    estimator: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    spec = make_gate_spec(
        seed=seed,
        probes=args.diagnostic_probes,
        window=args.diagnostic_window,
        bits=args.diagnostic_bits,
        value_bits=args.diagnostic_value_bits,
        value_mode=args.diagnostic_value_mode,
        train_context_length=args.diagnostic_context_length,
    ).to(device)
    context = materialize_context(
        spec,
        args.diagnostic_context_length,
        device=device,
    )
    fault_logit = torch.tensor(
        float(args.initial_fault_logit),
        device=device,
        requires_grad=True,
    )
    query_logits = make_query_logits(
        spec,
        fault_logit,
        logit_margin=args.logit_margin,
    )
    distribution = None
    if estimator == "production":
        carrier, coefficients, null_mass = _production_distribution(
            query_logits,
            context,
            window=spec.window,
            scale=args.scale,
            mismatch_scale=args.mismatch_scale,
        )
        distribution = _distribution_metrics(
            coefficients,
            null_mass,
            context.target_positions,
        )
    elif estimator == "bitflip":
        carrier = _bitflip_carrier(
            spec,
            context,
            fault_logit,
            logit_margin=args.logit_margin,
        )
    else:
        config = _proxy_config(estimator)
        if config.read_rule != "additive":
            raise ValueError("route diagnostics support additive reads only")
        carrier, coefficients, null_mass = _proxy_distribution(
            query_logits,
            context,
            config=config,
            window=spec.window,
            mismatch_scale=args.mismatch_scale,
            sketch_dim=args.sketch_dim,
            sketch_count=args.sketch_count,
            sketch_seed=args.sketch_seed + seed,
        )
        distribution = _distribution_metrics(
            coefficients,
            null_mass,
            context.target_positions,
        )
    loss = (carrier - spec.target_values).square().mean()
    (gradient,) = torch.autograd.grad(loss, (fault_logit,))
    return {
        "estimator": estimator,
        "seed": seed,
        "loss": float(loss.detach()),
        "fault_gradient": float(gradient.detach()),
        "gradient_moves_fault_toward_target": bool(float(gradient) < 0.0),
        "distribution": distribution,
    }


def _cosine(left: Tensor, right: Tensor) -> Optional[float]:
    left = left.detach().double().flatten()
    right = right.detach().double().flatten()
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _sketch_case(
    *,
    data_seed: int,
    hash_seed: int,
    route_kernel: str,
    sketch_dim: int,
    sketch_count: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(600_000 + data_seed)
    shape = (
        args.sketch_batch,
        args.sketch_sequence_length,
        1,
        args.sketch_bits,
    )
    query = torch.randn(shape, generator=generator).to(device).requires_grad_()
    key = torch.randn(shape, generator=generator).to(device).requires_grad_()
    coefficients = torch.randn(
        args.sketch_batch,
        args.sketch_sequence_length,
        generator=generator,
    ).to(device)
    common = {
        "route_kernel": route_kernel,
        "fingerprint_length": args.sketch_window,
        "mismatch_scale": args.mismatch_scale,
        "sketch_dim": sketch_dim,
        "sketch_count": sketch_count,
        "sketch_seed": hash_seed,
    }

    exact_q = _suffix_kernel_features(
        query,
        representation="exact",
        **common,
    )[:, 0, -1]
    exact_k = _suffix_kernel_features(
        key,
        representation="exact",
        **common,
    )[:, 0]
    exact_scores = torch.einsum("bf,btf->bt", exact_q, exact_k)
    exact_vjp = torch.autograd.grad(
        (exact_scores * coefficients).sum(),
        (query, key),
        retain_graph=True,
    )

    sketch_q = _suffix_kernel_features(
        query,
        representation="sketch",
        **common,
    )[:, 0, -1]
    sketch_k = _suffix_kernel_features(
        key,
        representation="sketch",
        **common,
    )[:, 0]
    sketch_scores = torch.einsum("bf,btf->bt", sketch_q, sketch_k)
    sketch_vjp = torch.autograd.grad(
        (sketch_scores * coefficients).sum(),
        (query, key),
    )

    exact_flat = exact_scores.detach().double().flatten()
    sketch_flat = sketch_scores.detach().double().flatten()
    relative_l2 = float(
        (sketch_flat - exact_flat).norm() / exact_flat.norm().clamp_min(1e-30)
    )
    latest = torch.arange(
        exact_scores.size(1),
        device=device,
    ).view(1, -1)
    exact_max = exact_scores.amax(dim=-1, keepdim=True)
    sketch_max = sketch_scores.amax(dim=-1, keepdim=True)
    exact_top = torch.where(exact_scores == exact_max, latest, -1).amax(dim=-1)
    sketch_top = torch.where(sketch_scores == sketch_max, latest, -1).amax(dim=-1)
    return {
        "data_seed": data_seed,
        "hash_seed": hash_seed,
        "route_kernel": route_kernel,
        "sketch_dim": sketch_dim,
        "sketch_count": sketch_count,
        "score_cosine": _cosine(sketch_scores, exact_scores),
        "score_relative_l2": relative_l2,
        "top_route_accuracy": float(sketch_top.eq(exact_top).float().mean()),
        "negative_score_fraction": float(sketch_scores.lt(0).float().mean()),
        "query_key_vjp_cosine": _cosine(
            torch.cat([gradient.flatten() for gradient in sketch_vjp]),
            torch.cat([gradient.flatten() for gradient in exact_vjp]),
        ),
    }


def _summarize_sketch_cases(
    cases: Sequence[dict[str, object]],
) -> dict[str, object]:
    summaries = {}
    conditions = sorted(
        {
            (
                str(case["route_kernel"]),
                int(case["sketch_dim"]),
                int(case["sketch_count"]),
            )
            for case in cases
        }
    )
    for route_kernel, sketch_dim, sketch_count in conditions:
        selected = [
            case
            for case in cases
            if case["route_kernel"] == route_kernel
            and case["sketch_dim"] == sketch_dim
            and case["sketch_count"] == sketch_count
        ]
        key = f"{route_kernel}:R{sketch_dim}:C{sketch_count}"
        summaries[key] = {
            "cases": len(selected),
            "mean_score_cosine": statistics.fmean(
                float(case["score_cosine"]) for case in selected
            ),
            "mean_score_relative_l2": statistics.fmean(
                float(case["score_relative_l2"]) for case in selected
            ),
            "mean_top_route_accuracy": statistics.fmean(
                float(case["top_route_accuracy"]) for case in selected
            ),
            "mean_negative_score_fraction": statistics.fmean(
                float(case["negative_score_fraction"]) for case in selected
            ),
            "mean_query_key_vjp_cosine": statistics.fmean(
                float(case["query_key_vjp_cosine"])
                for case in selected
                if case["query_key_vjp_cosine"] is not None
            ),
        }
    return summaries


@torch.no_grad()
def _evaluate(
    model: _ResearchFitLM,
    tokens: Tensor,
    target_mask: Tensor,
) -> tuple[float, float]:
    loss, accuracy = loss_and_accuracy(model(tokens), tokens, target_mask)
    return float(loss), float(accuracy)


def run_fit(
    estimator: str,
    model_seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    tokens_cpu = make_copy_tokens(
        seq_len=args.sequence_length + 1,
        vocab_size=args.vocab_size,
        motif_min=args.motif_min,
        motif_max=args.motif_max,
        seed=100_000 + model_seed,
    )
    target_mask_cpu = historical_target_mask(
        tokens_cpu,
        args.max_suffix_length,
    )
    if not bool(target_mask_cpu.any()):
        raise RuntimeError("generated fitting sequence has no target rows")
    tokens = tokens_cpu.to(device)
    target_mask = target_mask_cpu.to(device)

    torch.manual_seed(300_000 + model_seed)
    model = _ResearchFitLM(
        vocab_size=args.vocab_size,
        num_heads=args.heads,
        qk_bits=args.qk_bits,
        value_heads=args.value_heads,
        value_bits=args.value_bits,
        max_suffix_length=args.max_suffix_length,
        scale=args.scale,
        dropout_p=0.0,
        mismatch_scale=args.mismatch_scale,
        training_operator=_make_operator(
            estimator,
            device=device,
            fingerprint_length=args.fingerprint_length,
            sketch_dim=args.sketch_dim,
            sketch_count=args.sketch_count,
            sketch_seed=args.sketch_seed + model_seed,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    initial_loss, initial_accuracy = _evaluate(model, tokens, target_mask)
    best_loss = initial_loss
    best_step = 0
    first_below = -1
    initial_gradient_norms = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_and_accuracy(model(tokens), tokens, target_mask)
        loss.backward()
        if initial_gradient_norms is None:
            initial_gradient_norms = _parameter_gradient_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        loss_value = float(loss.detach())
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
        if first_below < 0 and loss_value < args.success_threshold:
            first_below = step
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_loss, final_accuracy = _evaluate(model, tokens, target_mask)
    if final_loss < best_loss:
        best_loss = final_loss
        best_step = args.steps
    return {
        "estimator": estimator,
        "model_seed": model_seed,
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "initial_gradient_norms": initial_gradient_norms,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "best_loss": best_loss,
        "best_step": best_step,
        "first_below_threshold": first_below,
        "ever_success": first_below >= 0 or best_loss < args.success_threshold,
        "final_success": math.isfinite(final_loss)
        and final_loss < args.success_threshold,
        "step_ms": elapsed * 1000.0 / max(args.steps, 1),
    }


def _random_nonzero(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    values = torch.randn(shape, generator=generator)
    values = torch.where(values >= 0, values + 0.2, values - 0.2)
    return values.to(device)


def _operator_vjp(
    estimator: str,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    upstream: Tensor,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    leaves = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )
    operator = _make_operator(
        estimator,
        device=query.device,
        fingerprint_length=args.gradient_fingerprint_length,
        sketch_dim=args.sketch_dim,
        sketch_count=args.sketch_count,
        sketch_seed=args.sketch_seed + seed,
    )
    output = operator(
        *leaves,
        max_suffix_length=args.gradient_max_suffix_length,
        scale=args.scale,
        mismatch_scale=args.mismatch_scale,
    )
    gradients = torch.autograd.grad((output * upstream).sum(), leaves)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def run_gradient_case(seed: int, args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(700_000 + seed)
    common = (1, args.gradient_sequence_length, 1)
    query = _random_nonzero(
        (*common, args.gradient_qk_bits),
        generator=generator,
        device=device,
    )
    key = _random_nonzero(
        (*common, args.gradient_qk_bits),
        generator=generator,
        device=device,
    )
    value = _random_nonzero(
        (*common, args.gradient_value_bits),
        generator=generator,
        device=device,
    )
    upstream = _random_nonzero(
        value.shape,
        generator=generator,
        device=device,
    )
    bitflip_output, bitflip_gradient = _operator_vjp(
        "bitflip", query, key, value, upstream, args, seed
    )
    estimators = {}
    for estimator in args.estimators:
        output, gradient = (
            (bitflip_output, bitflip_gradient)
            if estimator == "bitflip"
            else _operator_vjp(
                estimator,
                query,
                key,
                value,
                upstream,
                args,
                seed,
            )
        )
        estimators[estimator] = {
            "hard_forward_equal": bool(torch.equal(output, bitflip_output)),
            **_gradient_metrics(gradient, bitflip_gradient),
        }
    return {"seed": seed, "estimators": estimators}


def _summarize_fits(
    estimator: str,
    fits: Sequence[dict[str, object]],
) -> dict[str, object]:
    selected = [fit for fit in fits if fit["estimator"] == estimator]
    success_steps = [
        int(fit["first_below_threshold"])
        for fit in selected
        if int(fit["first_below_threshold"]) >= 0
    ]
    return {
        "runs": len(selected),
        "ever_successes": sum(bool(fit["ever_success"]) for fit in selected),
        "final_successes": sum(bool(fit["final_success"]) for fit in selected),
        "median_best_loss": statistics.median(
            float(fit["best_loss"]) for fit in selected
        ),
        "median_final_loss": statistics.median(
            float(fit["final_loss"]) for fit in selected
        ),
        "median_final_accuracy": statistics.median(
            float(fit["final_accuracy"]) for fit in selected
        ),
        "median_success_step": (
            statistics.median(success_steps) if success_steps else None
        ),
        "mean_step_ms": statistics.fmean(
            float(fit["step_ms"]) for fit in selected
        ),
    }


def _summarize_diagnostics(
    diagnostics: Sequence[dict[str, object]],
) -> dict[str, object]:
    result = {}
    for estimator in sorted({str(row["estimator"]) for row in diagnostics}):
        selected = [row for row in diagnostics if row["estimator"] == estimator]
        distributions = [
            row["distribution"]
            for row in selected
            if row["distribution"] is not None
        ]
        summary = {
            "runs": len(selected),
            "correct_gradient_fraction": statistics.fmean(
                float(row["gradient_moves_fault_toward_target"])
                for row in selected
            ),
            "mean_fault_gradient": statistics.fmean(
                float(row["fault_gradient"]) for row in selected
            ),
        }
        if distributions:
            for key in distributions[0]:
                summary[key] = statistics.fmean(
                    float(distribution[key]) for distribution in distributions
                )
        result[estimator] = summary
    return result


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _summarize_gradients(
    estimator: str,
    cases: Sequence[dict[str, object]],
) -> dict[str, object]:
    rows = [case["estimators"][estimator] for case in cases]
    return {
        "cases": len(rows),
        "hard_forward_equal_cases": sum(
            bool(row["hard_forward_equal"]) for row in rows
        ),
        "mean_query_key_cosine_to_bitflip": _mean_optional(
            [row["query_key"]["cosine_to_bitflip"] for row in rows]
        ),
        "mean_query_key_sign_agreement_to_bitflip": _mean_optional(
            [
                row["query_key"]["sign_agreement_to_bitflip"]
                for row in rows
            ]
        ),
        "mean_query_key_norm_ratio_to_bitflip": _mean_optional(
            [row["query_key"]["norm_ratio_to_bitflip"] for row in rows]
        ),
        "mean_value_cosine_to_bitflip": _mean_optional(
            [row["value"]["cosine_to_bitflip"] for row in rows]
        ),
        "mean_combined_cosine_to_bitflip": _mean_optional(
            [row["combined"]["cosine_to_bitflip"] for row in rows]
        ),
    }


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    diagnostics = []
    if not args.skip_diagnostics:
        diagnostics = [
            run_route_diagnostic(estimator, seed, args)
            for estimator in args.diagnostic_estimators
            for seed in args.diagnostic_seeds
        ]
    sketch_cases = []
    if not args.skip_sketch_scan:
        sketch_cases = [
            _sketch_case(
                data_seed=data_seed,
                hash_seed=hash_seed,
                route_kernel=route_kernel,
                sketch_dim=sketch_dim,
                sketch_count=sketch_count,
                args=args,
            )
            for route_kernel in args.sketch_route_kernels
            for sketch_dim in args.sketch_dims
            for sketch_count in args.sketch_counts
            for data_seed in args.sketch_data_seeds
            for hash_seed in args.sketch_hash_seeds
        ]
    fits = []
    if not args.skip_fit:
        fits = [
            run_fit(estimator, seed, args)
            for estimator in args.estimators
            for seed in args.model_seeds
        ]
    gradient_cases = []
    if not args.skip_gradient:
        gradient_cases = [
            run_gradient_case(seed, args) for seed in args.gradient_seeds
        ]
    return {
        "schema_version": 1,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "estimators": list(args.estimators),
        "diagnostics": diagnostics,
        "diagnostic_summaries": _summarize_diagnostics(diagnostics),
        "sketch_cases": sketch_cases,
        "sketch_summaries": _summarize_sketch_cases(sketch_cases),
        "fits": fits,
        "fit_summaries": {
            estimator: _summarize_fits(estimator, fits)
            for estimator in args.estimators
        }
        if fits
        else {},
        "gradient_cases": gradient_cases,
        "gradient_summaries": {
            estimator: _summarize_gradients(estimator, gradient_cases)
            for estimator in args.estimators
        }
        if gradient_cases
        else {},
        "gradient_metrics_are_relative_to": "complete bitflip",
        "proxy_has_trainable_parameters": False,
        "proxy_has_auxiliary_loss": False,
        "proxy_deletes_candidates": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=ESTIMATORS,
        default=[
            "production",
            "bitflip",
            "exact_raw_additive",
            "exact_raw_delta",
            "exact_quadratic_additive",
            "exact_level_quadratic_additive",
        ],
    )
    parser.add_argument(
        "--diagnostic-estimators",
        nargs="+",
        choices=ESTIMATORS,
        default=list(DIAGNOSTIC_ESTIMATORS),
    )
    parser.add_argument("--diagnostic-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--diagnostic-probes", type=int, default=8)
    parser.add_argument("--diagnostic-window", type=int, default=3)
    parser.add_argument("--diagnostic-bits", type=int, default=3)
    parser.add_argument("--diagnostic-value-bits", type=int, default=8)
    parser.add_argument(
        "--diagnostic-value-mode",
        choices=("balanced_binary", "coherent_negative"),
        default="balanced_binary",
    )
    parser.add_argument("--diagnostic-context-length", type=int, default=192)
    parser.add_argument(
        "--sketch-route-kernels",
        nargs="+",
        choices=ROUTE_KERNELS,
        default=list(ROUTE_KERNELS),
    )
    parser.add_argument("--sketch-dims", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--sketch-counts", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--sketch-data-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--sketch-hash-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
    )
    parser.add_argument("--sketch-batch", type=int, default=8)
    parser.add_argument("--sketch-sequence-length", type=int, default=32)
    parser.add_argument("--sketch-bits", type=int, default=2)
    parser.add_argument("--sketch-window", type=int, default=3)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--success-threshold", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--motif-min", type=int, default=3)
    parser.add_argument("--motif-max", type=int, default=5)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--qk-bits", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--max-suffix-length", type=int, default=3)
    parser.add_argument("--fingerprint-length", type=int, default=3)
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument("--sketch-count", type=int, default=1)
    parser.add_argument("--sketch-seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-seeds", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--gradient-sequence-length", type=int, default=6)
    parser.add_argument("--gradient-qk-bits", type=int, default=2)
    parser.add_argument("--gradient-value-bits", type=int, default=3)
    parser.add_argument("--gradient-max-suffix-length", type=int, default=3)
    parser.add_argument("--gradient-fingerprint-length", type=int, default=3)
    parser.add_argument("--initial-fault-logit", type=float, default=-0.25)
    parser.add_argument("--logit-margin", type=float, default=1.0)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-sketch-scan", action="store_true")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json-out", default="")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.fingerprint_length > args.max_suffix_length:
        raise ValueError("fingerprint_length cannot exceed max_suffix_length")
    if args.gradient_fingerprint_length > args.gradient_max_suffix_length:
        raise ValueError(
            "gradient_fingerprint_length cannot exceed gradient_max_suffix_length"
        )
    if args.diagnostic_window > args.diagnostic_context_length:
        raise ValueError("diagnostic window cannot exceed context length")
    for estimator in args.diagnostic_estimators:
        if (
            estimator not in BASELINES
            and _proxy_config(estimator).read_rule != "additive"
        ):
            raise ValueError("diagnostic proxy estimators must use additive reads")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    report = run_matrix(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.summary_only:
        print(
            json.dumps(
                {
                    "diagnostics": report["diagnostic_summaries"],
                    "sketch": report["sketch_summaries"],
                    "fits": report["fit_summaries"],
                    "gradients": report["gradient_summaries"],
                },
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
