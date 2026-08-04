"""Multi-seed fitting gate for tiny globally shared ROSA bits.

The benchmark trains only three key logits.  It intentionally excludes an
embedding, residual path, learned values, and output head, so zero loss means
that the hard ROSA route itself changed to the requested route.
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
from typing import Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.global_bit_oracle import (  # noqa: E402
    arm_disarm_samples,
    exact_bitflip_vjp,
    exact_margin_edit_oracle,
    exact_shared_bit_oracle,
    mean_field_winner_oracle,
    production_vjp,
    sampled_bitflip_residual_vjp,
)
from rosa_soft.soft_reference import _hard_route_forward  # noqa: E402


ESTIMATORS = (
    "production",
    "mean_field",
    "exact_expectation",
    "arm",
    "disarm",
    "exact_bitflip",
    "bitflip_residual",
    "margin_edit",
)
TASKS = ("single_edit", "joint_suffix")


@dataclass(frozen=True)
class TinyRouteTask:
    query: Tensor
    initial_key: Tensor
    value: Tensor
    target_output: Tensor
    query_stochastic_mask: Tensor
    key_stochastic_mask: Tensor
    initial_route: int
    target_route: int
    improving_single_bit_edits: int


def _route_and_output(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
) -> tuple[int, Tensor]:
    output, _, routes, _ = _hard_route_forward(
        query,
        key,
        value,
        max_suffix_length,
    )
    return int(routes[0, 0, -1]), output[:, -1:].clone()


def make_task(
    task_name: str,
    model_seed: int,
    *,
    initial_margin: float = 0.25,
    max_suffix_length: int = 2,
) -> TinyRouteTask:
    """Create a one-edit control or a coordinated suffix-edit task."""

    if task_name not in TASKS:
        raise ValueError(f"task_name must be one of {TASKS}")
    if not math.isfinite(initial_margin) or initial_margin <= 0.0:
        raise ValueError("initial_margin must be finite and positive")
    generator = torch.Generator().manual_seed(100_000 + int(model_seed))
    margins = initial_margin * (
        0.5 + torch.rand(3, generator=generator, dtype=torch.float64)
    )

    query = torch.tensor(
        [[[[1.0]], [[1.0]], [[-1.0]], [[1.0]]]],
        dtype=torch.float64,
    )
    initial_signs = (
        torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)
        if task_name == "single_edit"
        else torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
    )
    initial_key = torch.ones_like(query)
    initial_key[0, :3, 0, 0] = initial_signs * margins
    target_key = torch.ones_like(query)
    target_key[0, :3, 0, 0] = torch.tensor(
        [-1.0, 1.0, 1.0],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[1.0]], [[-1.0]], [[1.0]], [[-1.0]]]],
        dtype=torch.float64,
    )
    initial_route, initial_output = _route_and_output(
        query,
        initial_key,
        value,
        max_suffix_length,
    )
    target_route, target_output = _route_and_output(
        query,
        target_key,
        value,
        max_suffix_length,
    )
    if initial_route != 3 or target_route != 2:
        raise RuntimeError("tiny route task construction is inconsistent")

    improving_edits = 0
    initial_loss = (initial_output - target_output).square().mean()
    for key_index in range(3):
        edited_key = initial_key.clone()
        edited_key[0, key_index, 0, 0].neg_()
        _, edited_output = _route_and_output(
            query,
            edited_key,
            value,
            max_suffix_length,
        )
        if bool(
            (edited_output - target_output).square().mean() < initial_loss
        ):
            improving_edits += 1
    expected_edits = 2 if task_name == "single_edit" else 0
    if improving_edits != expected_edits:
        raise RuntimeError("tiny task has an unexpected one-bit path")

    query_mask = torch.zeros_like(query, dtype=torch.bool)
    key_mask = torch.zeros_like(initial_key, dtype=torch.bool)
    key_mask[:, :3] = True
    return TinyRouteTask(
        query=query,
        initial_key=initial_key,
        value=value,
        target_output=target_output,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
        initial_route=initial_route,
        target_route=target_route,
        improving_single_bit_edits=improving_edits,
    )


def _loss_and_upstream(
    output: Tensor,
    target_output: Tensor,
) -> tuple[float, Tensor]:
    error = output[:, -1:] - target_output
    grad_output = torch.zeros_like(output)
    grad_output[:, -1:] = 2.0 * error / error.numel()
    return float(error.square().mean()), grad_output


def _estimated_key_gradient(
    estimator: str,
    task: TinyRouteTask,
    key: Tensor,
    grad_output: Tensor,
    args: argparse.Namespace,
    sample_seed: int,
) -> Tensor:
    query = task.query
    value = task.value
    masks = {
        "query_stochastic_mask": task.query_stochastic_mask,
        "key_stochastic_mask": task.key_stochastic_mask,
    }
    if estimator == "production":
        _, key_gradient = production_vjp(
            query,
            key,
            value,
            grad_output,
            max_suffix_length=args.max_suffix_length,
            scale=args.scale,
            mismatch_scale=args.mismatch_scale,
        )
    elif estimator == "mean_field":
        key_gradient = mean_field_winner_oracle(
            query,
            key,
            value,
            grad_output,
            max_suffix_length=args.max_suffix_length,
            bit_temperature=args.bit_temperature,
            **masks,
        ).key_gradient
    elif estimator == "exact_expectation":
        key_gradient = exact_shared_bit_oracle(
            query,
            key,
            value,
            grad_output,
            max_suffix_length=args.max_suffix_length,
            bit_temperature=args.bit_temperature,
            max_bits=args.max_bits,
            **masks,
        ).key_gradient
    elif estimator in ("arm", "disarm"):
        samples = arm_disarm_samples(
            query,
            key,
            value,
            grad_output,
            max_suffix_length=args.max_suffix_length,
            bit_temperature=args.bit_temperature,
            sample_count=args.mc_pairs_per_step,
            seed=sample_seed,
            **masks,
        )
        bit_samples = (
            samples.arm_bit_gradients
            if estimator == "arm"
            else samples.disarm_bit_gradients
        )
        _, key_gradient = samples.layout.scatter(
            bit_samples.mean(dim=0),
            query,
            key,
        )
    elif estimator == "exact_bitflip":
        _, key_gradient, _, _ = exact_bitflip_vjp(
            query,
            key,
            value,
            grad_output,
            max_suffix_length=args.max_suffix_length,
            **masks,
        )
    elif estimator == "bitflip_residual":
        production_query, production_key = production_vjp(
            query,
            key,
            value,
            grad_output,
            max_suffix_length=args.max_suffix_length,
            scale=args.scale,
            mismatch_scale=args.mismatch_scale,
        )
        key_gradient = sampled_bitflip_residual_vjp(
            query,
            key,
            value,
            grad_output,
            production_query,
            production_key,
            max_suffix_length=args.max_suffix_length,
            flips_per_sample=args.bitflip_samples_per_step,
            seed=sample_seed,
            **masks,
        ).key_gradient
    elif estimator == "margin_edit":
        key_gradient = exact_margin_edit_oracle(
            query,
            key,
            value,
            grad_output,
            max_suffix_length=args.max_suffix_length,
            eta=args.margin_eta,
            max_bits=args.max_bits,
            **masks,
        ).key_gradient
    else:
        raise ValueError(f"unknown estimator: {estimator}")
    return key_gradient * task.key_stochastic_mask


def run_fit(
    estimator: str,
    task_name: str,
    model_seed: int,
    noise_seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    task = make_task(
        task_name,
        model_seed,
        initial_margin=args.initial_margin,
        max_suffix_length=args.max_suffix_length,
    )
    key = torch.nn.Parameter(task.initial_key.clone())
    optimizer = torch.optim.Adam([key], lr=args.learning_rate)
    best_loss = math.inf
    first_success = -1
    initial_loss = math.nan
    started = time.perf_counter()

    for step in range(args.steps + 1):
        output, _, routes, _ = _hard_route_forward(
            task.query,
            key.detach(),
            task.value,
            args.max_suffix_length,
        )
        loss, grad_output = _loss_and_upstream(output, task.target_output)
        if step == 0:
            initial_loss = loss
        best_loss = min(best_loss, loss)
        if first_success < 0 and loss <= args.success_threshold:
            first_success = step
        if step == args.steps:
            final_route = int(routes[0, 0, -1])
            final_loss = loss
            break

        optimizer.zero_grad(set_to_none=True)
        sample_seed = 1_000_000 + 10_000 * noise_seed + step
        key_gradient = _estimated_key_gradient(
            estimator,
            task,
            key.detach(),
            grad_output,
            args,
            sample_seed,
        )
        key.grad = key_gradient
        torch.nn.utils.clip_grad_norm_([key], args.grad_clip)
        optimizer.step()

    elapsed = time.perf_counter() - started
    return {
        "estimator": estimator,
        "task": task_name,
        "model_seed": model_seed,
        "noise_seed": noise_seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "first_success_step": first_success,
        "ever_success": first_success >= 0,
        "final_success": final_loss <= args.success_threshold,
        "initial_route": task.initial_route,
        "target_route": task.target_route,
        "final_route": final_route,
        "improving_single_bit_edits": task.improving_single_bit_edits,
        "final_key_logits": key.detach()[0, :3, 0, 0].tolist(),
        "step_ms": elapsed * 1000.0 / max(args.steps, 1),
    }


def _summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    successful_steps = [
        int(run["first_success_step"])
        for run in runs
        if int(run["first_success_step"]) >= 0
    ]
    return {
        "runs": len(runs),
        "ever_successes": sum(bool(run["ever_success"]) for run in runs),
        "final_successes": sum(bool(run["final_success"]) for run in runs),
        "ever_success_rate": statistics.mean(
            float(bool(run["ever_success"])) for run in runs
        ),
        "final_success_rate": statistics.mean(
            float(bool(run["final_success"])) for run in runs
        ),
        "median_first_success_step": (
            statistics.median(successful_steps) if successful_steps else None
        ),
        "median_final_loss": statistics.median(
            float(run["final_loss"]) for run in runs
        ),
        "median_best_loss": statistics.median(
            float(run["best_loss"]) for run in runs
        ),
        "mean_step_ms": statistics.mean(
            float(run["step_ms"]) for run in runs
        ),
    }


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    if args.noise_seeds is None:
        seed_pairs = [(seed, seed) for seed in args.model_seeds]
        noise_seeds = list(args.model_seeds)
        pairing = "matched"
    else:
        seed_pairs = [
            (model_seed, noise_seed)
            for model_seed in args.model_seeds
            for noise_seed in args.noise_seeds
        ]
        noise_seeds = list(args.noise_seeds)
        pairing = "cartesian"
    runs = [
        run_fit(estimator, task, model_seed, noise_seed, args)
        for task in args.tasks
        for estimator in args.estimators
        for model_seed, noise_seed in seed_pairs
    ]
    summaries = {
        task: {
            estimator: _summarize_runs(
                [
                    run
                    for run in runs
                    if run["task"] == task
                    and run["estimator"] == estimator
                ]
            )
            for estimator in args.estimators
        }
        for task in args.tasks
    }
    return {
        "objective": "hard route-output squared error with a fixed upstream VJP",
        "model_seeds": list(args.model_seeds),
        "noise_seeds": noise_seeds,
        "seed_pairing": pairing,
        "tasks": list(args.tasks),
        "estimators": list(args.estimators),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "bit_temperature": args.bit_temperature,
        "margin_eta": args.margin_eta,
        "mc_pairs_per_step": args.mc_pairs_per_step,
        "bitflip_samples_per_step": args.bitflip_samples_per_step,
        "hard_gradient_evaluations": {
            "arm": 2 * args.mc_pairs_per_step,
            "disarm": 2 * args.mc_pairs_per_step,
            "bitflip_residual": args.bitflip_samples_per_step,
        },
        "summaries": summaries,
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit tiny hard ROSA routes with global-bit estimators",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=ESTIMATORS,
        default=list(ESTIMATORS),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        default=list(range(16)),
    )
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--success-threshold", type=float, default=1e-12)
    parser.add_argument("--initial-margin", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-suffix-length", type=int, default=2)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--bit-temperature", type=float, default=0.5)
    parser.add_argument("--margin-eta", type=float, default=1.0)
    parser.add_argument("--mc-pairs-per-step", type=int, default=1)
    parser.add_argument("--bitflip-samples-per-step", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_matrix(args)
    output = report
    if args.summary_only:
        output = {key: value for key, value in report.items() if key != "runs"}
    encoded = json.dumps(output, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
