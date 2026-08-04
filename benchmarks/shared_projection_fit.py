"""Shared-parameter fitting gate for tiny hard ROSA route estimators."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.global_bit_fit import (  # noqa: E402
    ESTIMATORS,
    TinyRouteTask,
    _estimated_key_gradient,
    _loss_and_upstream,
    make_task,
)
from rosa_soft.soft_reference import _hard_route_forward  # noqa: E402


@dataclass(frozen=True)
class SharedProjectionTask:
    route_task: TinyRouteTask
    features: Tensor
    initial_parameters: Tensor
    target_parameters: Tensor

    def project_key(self, parameters: Tensor) -> Tensor:
        semantic_key = self.features @ parameters
        return torch.cat(
            (
                semantic_key.view(1, 3, 1, 1),
                parameters.new_ones(1, 1, 1, 1),
            ),
            dim=1,
        )


def make_shared_projection_task(
    model_seed: int,
    *,
    initial_margin: float = 0.25,
    max_suffix_length: int = 2,
) -> SharedProjectionTask:
    """Map two shared projection parameters to three coupled key logits."""

    if max_suffix_length < 2:
        raise ValueError("max_suffix_length must be at least 2")
    if not math.isfinite(initial_margin) or initial_margin <= 0.0:
        raise ValueError("initial_margin must be finite and positive")
    generator = torch.Generator().manual_seed(500_000 + int(model_seed))
    draws = torch.rand(2, generator=generator, dtype=torch.float64)
    positive = initial_margin * (1.0 + 0.25 * draws[0])
    negative_magnitude = initial_margin * (0.2 + 0.25 * draws[1])
    initial_parameters = torch.stack((positive, -negative_magnitude))
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    target_parameters = torch.tensor([-1.0, 2.0], dtype=torch.float64)

    base_task = make_task(
        "joint_suffix",
        model_seed,
        initial_margin=initial_margin,
        max_suffix_length=max_suffix_length,
    )
    projected_key = torch.cat(
        (
            (features @ initial_parameters).view(1, 3, 1, 1),
            torch.ones(1, 1, 1, 1, dtype=torch.float64),
        ),
        dim=1,
    )
    target_key = torch.cat(
        (
            (features @ target_parameters).view(1, 3, 1, 1),
            torch.ones(1, 1, 1, 1, dtype=torch.float64),
        ),
        dim=1,
    )
    _, _, initial_routes, _ = _hard_route_forward(
        base_task.query,
        projected_key,
        base_task.value,
        max_suffix_length,
    )
    target_output, _, target_routes, _ = _hard_route_forward(
        base_task.query,
        target_key,
        base_task.value,
        max_suffix_length,
    )
    if int(initial_routes[0, 0, -1]) != 3:
        raise RuntimeError("shared projection must start on route 3")
    if int(target_routes[0, 0, -1]) != 2:
        raise RuntimeError("shared projection target must select route 2")
    route_task = replace(
        base_task,
        initial_key=projected_key,
        target_output=target_output[:, -1:].clone(),
    )
    return SharedProjectionTask(
        route_task=route_task,
        features=features,
        initial_parameters=initial_parameters,
        target_parameters=target_parameters,
    )


def _hard_loss(
    task: SharedProjectionTask,
    parameters: Tensor,
    max_suffix_length: int,
) -> tuple[float, int, Tensor]:
    key = task.project_key(parameters)
    output, _, routes, _ = _hard_route_forward(
        task.route_task.query,
        key,
        task.route_task.value,
        max_suffix_length,
    )
    loss, grad_output = _loss_and_upstream(
        output,
        task.route_task.target_output,
    )
    return loss, int(routes[0, 0, -1]), grad_output


def _parameter_gradient(
    estimator: str,
    task: SharedProjectionTask,
    parameters: Tensor,
    grad_output: Tensor,
    args: argparse.Namespace,
    sample_seed: int,
) -> Tensor:
    key = task.project_key(parameters).detach()
    key_gradient = _estimated_key_gradient(
        estimator,
        task.route_task,
        key,
        grad_output,
        args,
        sample_seed,
    )
    return task.features.T @ key_gradient[0, :3, 0, 0]


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _hard_response(
    task: SharedProjectionTask,
    parameters: Tensor,
    gradient: Tensor,
    epsilon_grid: Sequence[float],
    max_suffix_length: int,
) -> dict[str, object]:
    base_loss, base_route, _ = _hard_loss(
        task,
        parameters,
        max_suffix_length,
    )
    gradient_norm = gradient.norm()
    if float(gradient_norm) == 0.0:
        return {
            "gradient_norm": 0.0,
            "base_route": base_route,
            "first_improving_epsilon": None,
            "best_loss": base_loss,
            "route_changes": 0,
        }
    direction = gradient / gradient_norm
    best_loss = base_loss
    first_improving = None
    route_changes = 0
    for epsilon in epsilon_grid:
        candidate = parameters - float(epsilon) * direction
        loss, route, _ = _hard_loss(
            task,
            candidate,
            max_suffix_length,
        )
        best_loss = min(best_loss, loss)
        route_changes += int(route != base_route)
        if first_improving is None and loss < base_loss:
            first_improving = float(epsilon)
    return {
        "gradient_norm": float(gradient_norm),
        "base_route": base_route,
        "first_improving_epsilon": first_improving,
        "best_loss": best_loss,
        "route_changes": route_changes,
    }


def run_fit(
    estimator: str,
    model_seed: int,
    noise_seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    task = make_shared_projection_task(
        model_seed,
        initial_margin=args.initial_margin,
        max_suffix_length=args.max_suffix_length,
    )
    parameters = torch.nn.Parameter(task.initial_parameters.clone())
    initial_loss, initial_route, initial_upstream = _hard_loss(
        task,
        parameters.detach(),
        args.max_suffix_length,
    )
    diagnostic_seed = 2_000_000 + 10_000 * noise_seed
    initial_gradient = _parameter_gradient(
        estimator,
        task,
        parameters.detach(),
        initial_upstream,
        args,
        diagnostic_seed,
    )
    exact_gradient = _parameter_gradient(
        "exact_expectation",
        task,
        parameters.detach(),
        initial_upstream,
        args,
        diagnostic_seed,
    )
    margin_gradient = _parameter_gradient(
        "margin_edit",
        task,
        parameters.detach(),
        initial_upstream,
        args,
        diagnostic_seed,
    )
    response = _hard_response(
        task,
        parameters.detach(),
        initial_gradient,
        args.response_epsilons,
        args.max_suffix_length,
    )

    optimizer = torch.optim.Adam([parameters], lr=args.learning_rate)
    best_loss = initial_loss
    first_success = -1
    started = time.perf_counter()
    for step in range(args.steps + 1):
        loss, route, grad_output = _hard_loss(
            task,
            parameters.detach(),
            args.max_suffix_length,
        )
        best_loss = min(best_loss, loss)
        if first_success < 0 and loss <= args.success_threshold:
            first_success = step
        if step == args.steps:
            final_loss = loss
            final_route = route
            break
        optimizer.zero_grad(set_to_none=True)
        sample_seed = 3_000_000 + 10_000 * noise_seed + step
        gradient = _parameter_gradient(
            estimator,
            task,
            parameters.detach(),
            grad_output,
            args,
            sample_seed,
        )
        parameters.grad = gradient
        torch.nn.utils.clip_grad_norm_([parameters], args.grad_clip)
        optimizer.step()
    elapsed = time.perf_counter() - started
    return {
        "estimator": estimator,
        "model_seed": model_seed,
        "noise_seed": noise_seed,
        "initial_parameters": task.initial_parameters.tolist(),
        "final_parameters": parameters.detach().tolist(),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "initial_route": initial_route,
        "target_route": task.route_task.target_route,
        "final_route": final_route,
        "first_success_step": first_success,
        "ever_success": first_success >= 0,
        "final_success": final_loss <= args.success_threshold,
        "initial_parameter_gradient": initial_gradient.tolist(),
        "initial_cosine_to_exact_expectation": _cosine(
            initial_gradient,
            exact_gradient,
        ),
        "initial_cosine_to_margin_edit": _cosine(
            initial_gradient,
            margin_gradient,
        ),
        "initial_hard_response": response,
        "step_ms": elapsed * 1000.0 / max(args.steps, 1),
    }


def _summary(runs: list[dict[str, object]]) -> dict[str, object]:
    success_steps = [
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
        "median_first_success_step": (
            statistics.median(success_steps) if success_steps else None
        ),
        "median_final_loss": statistics.median(
            float(run["final_loss"]) for run in runs
        ),
        "initial_hard_descent_fraction": statistics.mean(
            float(
                run["initial_hard_response"]["first_improving_epsilon"]
                is not None
            )
            for run in runs
        ),
        "mean_initial_cosine_to_exact_expectation": statistics.mean(
            float(run["initial_cosine_to_exact_expectation"])
            for run in runs
        ),
        "mean_initial_cosine_to_margin_edit": statistics.mean(
            float(run["initial_cosine_to_margin_edit"])
            for run in runs
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
        run_fit(estimator, model_seed, noise_seed, args)
        for estimator in args.estimators
        for model_seed, noise_seed in seed_pairs
    ]
    return {
        "parameterization": "key_logits = [[1,0],[0,1],[1,1]] @ theta",
        "objective": "hard route-output squared error after shared projection",
        "model_seeds": list(args.model_seeds),
        "noise_seeds": noise_seeds,
        "seed_pairing": pairing,
        "estimators": list(args.estimators),
        "steps": args.steps,
        "response_epsilons": list(args.response_epsilons),
        "bit_temperature": args.bit_temperature,
        "margin_eta": args.margin_eta,
        "summaries": {
            estimator: _summary(
                [run for run in runs if run["estimator"] == estimator]
            )
            for estimator in args.estimators
        },
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit hard ROSA after a coupled shared key projection",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=ESTIMATORS,
        default=list(ESTIMATORS),
    )
    parser.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        default=list(range(16)),
    )
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
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
    parser.add_argument(
        "--response-epsilons",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.4, 0.8],
    )
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
