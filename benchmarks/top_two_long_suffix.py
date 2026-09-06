"""Long-suffix gate for winner/rest, top-two, and dense ROSA VJPs.

This research benchmark reuses the exact many-distractor construction and
streaming hard evaluator from ``long_suffix_extrapolation``.  It specializes
the soft estimator to the final query row, so 8K training does not materialize
a quadratic route matrix.  Every estimator has the same hard ROSA output.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dual_score_reference import (  # noqa: E402
    final_query_score_carrier,
)
from benchmarks.long_suffix_extrapolation import (  # noqa: E402
    DenseContext,
    GateSpec,
    _final_query_suffix_products,
    dense_hard_route,
    make_gate_spec,
    make_query_logits,
    materialize_context,
    streaming_hard_evaluate,
)
from benchmarks.top_two_value_reference import (  # noqa: E402
    hard_null_gate_probabilities,
    top2_route_probabilities,
    winner_rest_route_carrier,
)
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    ROSA_SOFT_NULL_ROUTE_SCORE,
)
from rosa_soft.soft_reference import _suffix_score_utility  # noqa: E402


ESTIMATORS = (
    "production_full",
    "winner_rest",
    "hard_winner_rest",
    "no_prior",
    "hard_null_gate",
    "top2_qk",
    "dual",
    "bitflip",
)


def _route_distribution(
    query_logits: Tensor,
    context: DenseContext,
    *,
    window: int,
    scale: float,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    suffix_products = _final_query_suffix_products(
        query_logits,
        context,
        window=window,
        mismatch_scale=mismatch_scale,
    )
    candidate_scores = _suffix_score_utility(suffix_products.sum(dim=-1))
    candidate_logits = candidate_scores * float(scale) - math.log(
        context.key_signs.size(1)
    )
    null_logits = candidate_logits.new_full(
        (candidate_logits.size(0), 1),
        ROSA_SOFT_NULL_ROUTE_SCORE * float(scale),
    )
    logits = torch.cat((null_logits, candidate_logits), dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    route_values = torch.cat(
        (
            context.values.new_zeros(
                context.values.size(0), 1, context.values.size(-1)
            ),
            context.values,
        ),
        dim=1,
    )
    return logits, probabilities, route_values


def _competition_carrier(
    estimator: str,
    query_logits: Tensor,
    context: DenseContext,
    hard_selected_positions: Tensor,
    *,
    window: int,
    scale: float,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor]:
    logits, probabilities, route_values = _route_distribution(
        query_logits,
        context,
        window=window,
        scale=scale,
        mismatch_scale=mismatch_scale,
    )
    if estimator == "no_prior":
        adjusted = logits.clone()
        adjusted[:, 1:] += math.log(context.key_signs.size(1))
        probabilities = torch.softmax(adjusted, dim=-1)
    elif estimator == "hard_null_gate":
        # Every constructed gate has at least one exact non-null match.
        hard_winner = hard_selected_positions + 1
        probabilities = hard_null_gate_probabilities(
            logits[:, None, None], hard_winner[:, None, None]
        )[:, 0, 0]
    if estimator == "production_full":
        carrier = torch.einsum("pa,pav->pv", probabilities, route_values)
    elif estimator == "top2_qk":
        top2 = top2_route_probabilities(logits)
        carrier = torch.einsum("pa,pav->pv", top2, route_values)
    elif estimator in ("winner_rest", "hard_winner_rest"):
        winner = (
            hard_selected_positions + 1
            if estimator == "hard_winner_rest"
            else None
        )
        carrier = winner_rest_route_carrier(
            probabilities[:, None, None],
            route_values[:, None],
            winner=None if winner is None else winner[:, None, None],
        )[:, 0, 0]
    else:
        carrier = torch.einsum("pa,pav->pv", probabilities, route_values)
    target_probability = probabilities.gather(
        1, (context.target_positions + 1).view(-1, 1)
    ).squeeze(1)
    return carrier, target_probability


def hard_forward_with_proxy(
    estimator: str,
    spec: GateSpec,
    context: DenseContext,
    fault_logit: Tensor,
    *,
    logit_margin: float,
    scale: float,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    query_logits = make_query_logits(
        spec, fault_logit, logit_margin=logit_margin
    )
    hard = dense_hard_route(query_logits, context, spec.window)
    if estimator in (
        "production_full",
        "winner_rest",
        "hard_winner_rest",
        "no_prior",
        "hard_null_gate",
        "top2_qk",
    ):
        carrier, target_probability = _competition_carrier(
            estimator,
            query_logits,
            context,
            hard.selected_positions,
            window=spec.window,
            scale=scale,
            mismatch_scale=mismatch_scale,
        )
    elif estimator == "dual":
        carrier, candidate_probabilities = final_query_score_carrier(
            query_logits,
            context.key_signs,
            context.values,
            score_mode="dual",
            scale=scale,
            mismatch_scale=mismatch_scale,
        )
        target_probability = candidate_probabilities.gather(
            1, context.target_positions.view(-1, 1)
        ).squeeze(1)
    else:
        flipped_query = make_query_logits(
            spec, -fault_logit.detach(), logit_margin=logit_margin
        )
        flipped = dense_hard_route(flipped_query, context, spec.window)
        fault_sign = torch.where(
            fault_logit.detach() > 0,
            torch.ones_like(fault_logit),
            -torch.ones_like(fault_logit),
        )
        slope = -fault_sign * (flipped.output - hard.output).detach()
        carrier = fault_logit * slope
        target_probability = fault_logit.new_full(
            (spec.probes,), float("nan")
        )
    output = hard.output + (carrier - carrier.detach())
    return output, hard.selected_positions, target_probability.detach()


def train_condition(
    spec: GateSpec,
    estimator: str,
    *,
    device: torch.device,
    steps: int,
    learning_rate: float,
    initial_fault_logit: float,
    logit_margin: float,
    scale: float,
    mismatch_scale: float,
) -> dict[str, object]:
    context = materialize_context(
        spec, spec.train_context_length, device=device
    )
    fault_logit = torch.nn.Parameter(
        torch.tensor(float(initial_fault_logit), device=device)
    )
    optimizer = torch.optim.Adam((fault_logit,), lr=float(learning_rate))
    first_exact_step = None
    initial_gradient = None
    initial_target_probability = None
    started = time.perf_counter()
    executed_steps = 0
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output, selected_positions, target_probability = hard_forward_with_proxy(
            estimator,
            spec,
            context,
            fault_logit,
            logit_margin=logit_margin,
            scale=scale,
            mismatch_scale=mismatch_scale,
        )
        exact = selected_positions.eq(context.target_positions)
        if bool(exact.all()) and first_exact_step is None:
            first_exact_step = step
        finite = target_probability[torch.isfinite(target_probability)]
        if initial_target_probability is None:
            initial_target_probability = (
                float(finite.mean()) if finite.numel() else None
            )
        if step == steps or bool(exact.all()):
            break
        loss = (output - spec.target_values.to(device)).square().mean()
        loss.backward()
        gradient = float(fault_logit.grad.detach())
        if initial_gradient is None:
            initial_gradient = gradient
        optimizer.step()
        executed_steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    with torch.no_grad():
        final_query = make_query_logits(
            spec, fault_logit, logit_margin=logit_margin
        )
        final_state = dense_hard_route(final_query, context, spec.window)
    return {
        "initial_gradient": initial_gradient,
        "initial_target_probability": initial_target_probability,
        "first_exact_step": first_exact_step,
        "executed_steps": executed_steps,
        "final_fault_logit": float(fault_logit.detach()),
        "train_success": bool(
            final_state.selected_positions.eq(context.target_positions).all()
        ),
        "train_ms": elapsed_ms,
        "ms_per_executed_step": elapsed_ms / max(executed_steps, 1),
    }


def _summarize(
    records: list[dict[str, object]],
    estimators: Sequence[str],
    windows: Sequence[int],
    eval_lengths: Sequence[int],
) -> dict[str, object]:
    summary = {}
    for estimator in estimators:
        by_window = {}
        for window in windows:
            selected = [
                record
                for record in records
                if record["estimator"] == estimator
                and record["window"] == window
            ]
            successes = [
                record for record in selected if record["training"]["train_success"]
            ]
            first_steps = [
                record["training"]["first_exact_step"]
                for record in successes
                if record["training"]["first_exact_step"] is not None
            ]
            by_length = {}
            for length in eval_lengths:
                evaluations = [
                    evaluation
                    for record in selected
                    for evaluation in record["evaluations"]
                    if evaluation["context_length"] == length
                ]
                by_length[str(length)] = {
                    "all_routes_correct": sum(
                        bool(item["all_routes_correct"])
                        for item in evaluations
                    ),
                    "runs": len(evaluations),
                    "mean_route_accuracy": statistics.mean(
                        float(item["route_accuracy"])
                        for item in evaluations
                    ),
                }
            by_window[str(window)] = {
                "train_successes": len(successes),
                "runs": len(selected),
                "median_first_exact_step": (
                    statistics.median(first_steps) if first_steps else None
                ),
                "mean_initial_gradient": statistics.mean(
                    float(record["training"]["initial_gradient"])
                    for record in selected
                    if record["training"]["initial_gradient"] is not None
                ),
                "extrapolation": by_length,
            }
        summary[estimator] = by_window
    return summary


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    records = []
    evaluation_cache = {}
    for window in args.windows:
        for seed in args.seeds:
            spec = make_gate_spec(
                seed=seed,
                probes=args.probes,
                window=window,
                bits=args.bits,
                value_bits=args.value_bits,
                value_mode=args.value_mode,
                train_context_length=args.train_context_length,
            ).to(device)
            for estimator in args.estimators:
                training = train_condition(
                    spec,
                    estimator,
                    device=device,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    initial_fault_logit=args.initial_fault_logit,
                    logit_margin=args.logit_margin,
                    scale=args.scale,
                    mismatch_scale=args.mismatch_scale,
                )
                solved = float(training["final_fault_logit"]) > 0.0
                evaluations = []
                for context_length in args.eval_context_lengths:
                    cache_key = (window, seed, solved, context_length)
                    if cache_key not in evaluation_cache:
                        evaluation_cache[cache_key] = streaming_hard_evaluate(
                            spec,
                            float(training["final_fault_logit"]),
                            context_length=context_length,
                            chunk_size=args.chunk_size,
                            logit_margin=args.logit_margin,
                            device=device,
                        )
                    evaluations.append(dict(evaluation_cache[cache_key]))
                records.append(
                    {
                        "seed": seed,
                        "window": window,
                        "estimator": estimator,
                        "training": training,
                        "evaluations": evaluations,
                    }
                )
    return {
        "schema_version": 1,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "estimators": list(args.estimators),
        "windows": list(args.windows),
        "seeds": list(args.seeds),
        "train_context_length": args.train_context_length,
        "eval_context_lengths": list(args.eval_context_lengths),
        "records": records,
        "summary": _summarize(
            records,
            args.estimators,
            args.windows,
            args.eval_context_lengths,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--estimators", nargs="+", choices=ESTIMATORS, default=list(ESTIMATORS)
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, default=[1, 2, 4, 8, 32]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-bits", type=int, default=8)
    parser.add_argument(
        "--value-mode",
        choices=("balanced_binary", "coherent_negative"),
        default="balanced_binary",
    )
    parser.add_argument("--train-context-length", type=int, default=8192)
    parser.add_argument(
        "--eval-context-lengths",
        nargs="+",
        type=int,
        default=[8192, 65536, 1048576],
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--initial-fault-logit", type=float, default=-0.25)
    parser.add_argument("--logit-margin", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_matrix(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(
        json.dumps(report["summary"], indent=2, allow_nan=False)
        if args.summary_only
        else encoded
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
