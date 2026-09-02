"""Fit and gradient gates for parameter-free ROSA fast-weight VJPs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
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
from benchmarks.fast_weight_proxy import (  # noqa: E402
    PROXIES,
    rosa_fast_weight_proxy,
)
from benchmarks.temporal_quadratic_proxy import (  # noqa: E402
    DEFAULT_STATE_DIM as TEMPORAL_DEFAULT_STATE_DIM,
    rosa_temporal_quadratic_proxy,
)
from examples.fit_soft_reference import (  # noqa: E402
    historical_target_mask,
    loss_and_accuracy,
    make_copy_tokens,
)
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.soft_reference import rosa_soft_reference  # noqa: E402


Estimator = Callable[..., Tensor]
ESTIMATORS = ("production", "bitflip", *PROXIES)
TEMPORAL_ESTIMATOR = "temporal_quadratic_attention"
ESTIMATOR_CHOICES = ESTIMATORS + (TEMPORAL_ESTIMATOR,)


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
    sketch_seed: int,
    temporal_state_dim: int,
) -> Estimator:
    if estimator == "production":
        return _production_operator(device)
    if estimator == "bitflip":
        return rosa_soft_exact_bitflip
    if estimator == TEMPORAL_ESTIMATOR:
        def temporal_operator(
            query: Tensor,
            key: Tensor,
            value: Tensor,
            *,
            max_suffix_length: int,
            scale: float,
            mismatch_scale: float,
        ) -> Tensor:
            del scale, mismatch_scale
            return rosa_temporal_quadratic_proxy(
                query,
                key,
                value,
                max_suffix_length=max_suffix_length,
                state_dim=temporal_state_dim,
            )

        return temporal_operator
    if estimator not in PROXIES:
        raise ValueError(f"unknown estimator: {estimator}")

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
        return rosa_fast_weight_proxy(
            query,
            key,
            value,
            proxy=estimator,
            max_suffix_length=max_suffix_length,
            fingerprint_length=fingerprint_length,
            mismatch_scale=mismatch_scale,
            sketch_dim=sketch_dim,
            sketch_seed=sketch_seed,
        )

    return operator


@torch.no_grad()
def _evaluate(
    model: _ResearchFitLM,
    tokens: Tensor,
    target_mask: Tensor,
) -> tuple[float, float]:
    loss, accuracy = loss_and_accuracy(model(tokens), tokens, target_mask)
    return float(loss), float(accuracy)


def _parameter_gradient_norms(model: _ResearchFitLM) -> dict[str, float]:
    def norm(parameters) -> float:
        terms = [
            parameter.grad.detach().float().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not terms:
            return 0.0
        return float(torch.stack(terms).sum().sqrt())

    return {
        "query_key": norm(
            (*model.query.parameters(), *model.key.parameters())
        ),
        "value": norm(model.value.parameters()),
    }


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
            sketch_seed=args.sketch_seed,
            temporal_state_dim=args.temporal_state_dim,
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
        "final_success": (
            math.isfinite(final_loss)
            and final_loss < args.success_threshold
        ),
        "fit_target_count": int(target_mask.sum()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "step_ms": elapsed * 1000.0 / max(args.steps, 1),
    }


def _random_nonzero(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
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
        sketch_seed=args.sketch_seed + seed,
        temporal_state_dim=args.temporal_state_dim,
    )
    output = operator(
        *leaves,
        max_suffix_length=args.gradient_max_suffix_length,
        scale=args.scale,
        mismatch_scale=args.mismatch_scale,
    )
    gradients = torch.autograd.grad((output * upstream).sum(), leaves)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def _gradient_metrics(
    estimate: tuple[Tensor, Tensor, Tensor],
    reference: tuple[Tensor, Tensor, Tensor],
) -> dict[str, dict[str, Optional[float]]]:
    def metrics(
        estimate_tensors: Sequence[Tensor],
        reference_tensors: Sequence[Tensor],
    ) -> dict[str, Optional[float]]:
        estimate_flat = torch.cat(
            [tensor.double().flatten() for tensor in estimate_tensors]
        )
        reference_flat = torch.cat(
            [tensor.double().flatten() for tensor in reference_tensors]
        )
        estimate_norm = estimate_flat.norm()
        reference_norm = reference_flat.norm()
        cosine = None
        if float(estimate_norm) > 0.0 and float(reference_norm) > 0.0:
            cosine = float(
                torch.dot(estimate_flat, reference_flat)
                / (estimate_norm * reference_norm)
            )
        comparable = (estimate_flat != 0) & (reference_flat != 0)
        sign_agreement = None
        if bool(comparable.any()):
            sign_agreement = float(
                (
                    torch.sign(estimate_flat[comparable])
                    == torch.sign(reference_flat[comparable])
                ).double().mean()
            )
        reference_is_zero = float(reference_norm) == 0.0
        estimate_is_zero = float(estimate_norm) == 0.0
        relative_error = (
            0.0
            if reference_is_zero and estimate_is_zero
            else None
            if reference_is_zero
            else float((estimate_flat - reference_flat).norm() / reference_norm)
        )
        norm_ratio = (
            1.0
            if reference_is_zero and estimate_is_zero
            else None
            if reference_is_zero
            else float(estimate_norm / reference_norm)
        )
        return {
            "cosine_to_bitflip": cosine,
            "sign_agreement_to_bitflip": sign_agreement,
            "relative_l2_error_to_bitflip": relative_error,
            "l2_norm": float(estimate_norm),
            "norm_ratio_to_bitflip": norm_ratio,
        }

    return {
        "combined": metrics(estimate, reference),
        "query_key": metrics(estimate[:2], reference[:2]),
        "value": metrics(estimate[2:], reference[2:]),
    }


def run_gradient_case(
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
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
        "bitflip",
        query,
        key,
        value,
        upstream,
        args,
        seed,
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


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return statistics.mean(present) if present else None


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
        "mean_step_ms": statistics.mean(
            float(fit["step_ms"]) for fit in selected
        ),
    }


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
        "mean_combined_relative_l2_error_to_bitflip": _mean_optional(
            [
                row["combined"]["relative_l2_error_to_bitflip"]
                for row in rows
            ]
        ),
    }


def _state_dimensions(args: argparse.Namespace) -> dict[str, int]:
    local = 1 << args.qk_bits
    length = args.fingerprint_length
    exact_single = local**length
    exact_multi = sum(local**level for level in range(1, length + 1))
    sketch_multi = length * args.sketch_dim
    state_linear = 1 + args.qk_bits
    state_quadratic = state_linear + math.comb(args.qk_bits, 2)
    state_cubic = state_quadratic + math.comb(args.qk_bits, 3)
    temporal_quadratic = (
        args.temporal_state_dim * (args.temporal_state_dim + 1) // 2
    )
    return {
        "local_feature": local,
        "state_linear": state_linear,
        "state_quadratic": state_quadratic,
        "state_cubic": state_cubic,
        "temporal_state": args.temporal_state_dim,
        "temporal_quadratic": temporal_quadratic,
        "single_suffix": exact_single,
        "single_suffix_sketch": args.sketch_dim,
        "exact_multi_suffix": exact_multi,
        "sketch_multi_suffix": sketch_multi,
        "exact_single_delta_memory_floats_per_head": (
            exact_single * args.value_bits
        ),
        "single_sketch_delta_memory_floats_per_head": (
            args.sketch_dim * args.value_bits
        ),
        "multi_sketch_delta_memory_floats_per_head": (
            sketch_multi * args.value_bits
        ),
        "state_linear_delta_memory_floats_per_head": (
            state_linear * args.value_bits
        ),
        "state_quadratic_delta_memory_floats_per_head": (
            state_quadratic * args.value_bits
        ),
        "state_cubic_delta_memory_floats_per_head": (
            state_cubic * args.value_bits
        ),
        "state_full_delta_memory_floats_per_head": local * args.value_bits,
        "state_linear_attention_memory_floats_per_head": (
            state_linear * args.value_bits
        ),
        "state_quadratic_attention_memory_floats_per_head": (
            state_quadratic * args.value_bits
        ),
        "state_cubic_attention_memory_floats_per_head": (
            state_cubic * args.value_bits
        ),
        "state_full_linear_attention_memory_floats_per_head": (
            local * args.value_bits
        ),
        "state_linear_attention_total_floats_per_head": (
            state_linear * (args.value_bits + 1)
        ),
        "state_quadratic_attention_total_floats_per_head": (
            state_quadratic * (args.value_bits + 1)
        ),
        "state_cubic_attention_total_floats_per_head": (
            state_cubic * (args.value_bits + 1)
        ),
        "state_full_linear_attention_total_floats_per_head": (
            local * (args.value_bits + 1)
        ),
        "temporal_quadratic_attention_total_floats_per_head": (
            temporal_quadratic * (args.value_bits + 1)
        ),
    }


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    if args.fingerprint_length > args.max_suffix_length:
        raise ValueError("fingerprint_length cannot exceed max_suffix_length")
    if args.gradient_fingerprint_length > args.gradient_max_suffix_length:
        raise ValueError(
            "gradient_fingerprint_length cannot exceed "
            "gradient_max_suffix_length"
        )
    fits = [] if args.gradient_only else [
        run_fit(estimator, seed, args)
        for estimator in args.estimators
        for seed in args.model_seeds
    ]
    gradient_cases = [] if args.fit_only else [
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
        "baselines": ["production", "bitflip"],
        "estimators": list(args.estimators),
        "model_seeds": list(args.model_seeds),
        "gradient_seeds": list(args.gradient_seeds),
        "steps": args.steps,
        "success_threshold": args.success_threshold,
        "sequence_length": args.sequence_length,
        "qk_bits": args.qk_bits,
        "value_bits": args.value_bits,
        "max_suffix_length": args.max_suffix_length,
        "fingerprint_length": args.fingerprint_length,
        "sketch_dim": args.sketch_dim,
        "sketch_seed": args.sketch_seed,
        "temporal_state_dim": args.temporal_state_dim,
        "scale": args.scale,
        "mismatch_scale": args.mismatch_scale,
        "proxy_has_trainable_parameters": False,
        "proxy_has_auxiliary_loss": False,
        "proxy_uses_attention_scale": False,
        "state_dimensions": _state_dimensions(args),
        "fits": fits,
        "fit_summaries": {
            estimator: _summarize_fits(estimator, fits)
            for estimator in args.estimators
        } if fits else {},
        "gradient_cases": gradient_cases,
        "gradient_summaries": {
            estimator: _summarize_gradients(estimator, gradient_cases)
            for estimator in args.estimators
        } if gradient_cases else {},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=ESTIMATOR_CHOICES,
        default=list(ESTIMATORS),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--gradient-seeds",
        nargs="+",
        type=int,
        default=list(range(16)),
    )
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
    parser.add_argument("--sketch-seed", type=int, default=0)
    parser.add_argument(
        "--temporal-state-dim",
        type=int,
        default=TEMPORAL_DEFAULT_STATE_DIM,
    )
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-sequence-length", type=int, default=6)
    parser.add_argument("--gradient-qk-bits", type=int, default=2)
    parser.add_argument("--gradient-value-bits", type=int, default=3)
    parser.add_argument("--gradient-max-suffix-length", type=int, default=3)
    parser.add_argument("--gradient-fingerprint-length", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fit-only", action="store_true")
    mode.add_argument("--gradient-only", action="store_true")
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_matrix(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
