"""Compare Q/K estimators on shortcut-free contextual ROSA recall."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.stochastic_hard_vjp import (  # noqa: E402
    mean_field_hard_rosa,
    stochastic_hard_rosa,
)
from benchmarks.suffix_proxy_ablation import (  # noqa: E402
    rosa_soft_suffix_proxy,
)
from examples.contextual_rnn_recall_gate import (  # noqa: E402
    ResetRnnRosaLM,
    _exact_and_bit_accuracy,
    _recall_loss,
    _train,
    _validate_operator,
    evaluate_model,
    make_contextual_recall_batch,
)
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)


ESTIMATORS = (
    "production",
    "production_dropout",
    "collision_lr",
    "mean_field",
    "arm",
    "disarm",
)


class EstimatorResetRnnRosaLM(ResetRnnRosaLM):
    def __init__(
        self,
        *,
        estimator: str,
        bit_temperature: float,
        antithetic_pairs: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        self.estimator = estimator
        self.bit_temperature = float(bit_temperature)
        self.antithetic_pairs = int(antithetic_pairs)

    def _routed_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        route_mode: str,
    ) -> Tensor:
        if route_mode != "rosa" or self.estimator.startswith("production"):
            return super()._routed_values(query, key, value, route_mode)
        if self.estimator == "collision_lr":
            return rosa_soft_suffix_proxy(
                query,
                key,
                value,
                proxy="collision_lr",
                max_suffix_length=1,
                scale=self.scale,
                mismatch_scale=self.mismatch_scale,
            )
        common = {
            "bit_temperature": self.bit_temperature,
            "backend": self.operator,
            "max_suffix_length": 1,
            "scale": self.scale,
            "dropout_p": 0.0,
            "mismatch_scale": self.mismatch_scale,
        }
        if self.estimator == "mean_field":
            return mean_field_hard_rosa(query, key, value, **common)
        return stochastic_hard_rosa(
            query,
            key,
            value,
            estimator=self.estimator,
            pairs=self.antithetic_pairs,
            **common,
        )


def _gradient_norm(parameters) -> float:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().float().square().sum()
        squared = contribution if squared is None else squared + contribution
    return 0.0 if squared is None else float(squared.sqrt())


def _train_estimator(
    model: EstimatorResetRnnRosaLM,
    recall_batch,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    started = time.perf_counter()
    first_exact_step = None
    qk_gradient_norms = []
    value_gradient_norms = []
    final_loss = None
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(recall_batch.tokens, route_mode="rosa")
        loss = _recall_loss(logits, recall_batch)
        exact_accuracy, _ = _exact_and_bit_accuracy(
            logits.detach(), recall_batch
        )
        if first_exact_step is None and exact_accuracy >= 0.99:
            first_exact_step = step
        loss.backward()
        qk_gradient_norms.append(
            _gradient_norm((*model.query.parameters(), *model.key.parameters()))
        )
        value_gradient_norms.append(_gradient_norm(model.value.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        final_loss = float(loss.detach())
    if recall_batch.tokens.is_cuda:
        torch.cuda.synchronize(recall_batch.tokens.device)
    elapsed = time.perf_counter() - started
    return {
        "training_loss": final_loss,
        "first_train_exact_step": first_exact_step,
        "qk_zero_gradient_fraction": (
            sum(norm == 0.0 for norm in qk_gradient_norms) / steps
            if steps
            else 0.0
        ),
        "mean_qk_gradient_norm": (
            statistics.mean(qk_gradient_norms) if qk_gradient_norms else 0.0
        ),
        "mean_value_gradient_norm": (
            statistics.mean(value_gradient_norms)
            if value_gradient_norms
            else 0.0
        ),
        "step_ms": elapsed * 1000.0 / steps if steps else 0.0,
    }


def _candidate_passed(train: Dict[str, float], validation: Dict[str, float]) -> bool:
    return bool(
        train["exact_accuracy"] >= 0.99
        and validation["exact_accuracy"] >= 0.99
        and validation["historical_route_any_head_accuracy"] >= 0.99
        and validation["paired_routed_value_difference_fraction"] >= 0.99
        and validation["zero_route_exact_accuracy"] <= 0.5
        and validation["current_value_exact_accuracy"] <= 0.5
        and validation["query_residual_max_difference"] == 0.0
    )


def _dropout_for(estimator: str, dropout_p: float) -> float:
    return float(dropout_p) if estimator == "production_dropout" else 0.0


def run_seed(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    device = torch.device(args.device)
    train_batch = make_contextual_recall_batch(
        seed=100_000 + seed,
        pairs=args.train_pairs,
        associations=args.associations,
        value_bits=args.value_bits,
    ).to(device)
    validation_batch = make_contextual_recall_batch(
        seed=200_000 + seed,
        pairs=args.validation_pairs,
        associations=args.associations,
        value_bits=args.value_bits,
    ).to(device)
    torch.manual_seed(300_000 + seed)
    initial_model = EstimatorResetRnnRosaLM(
        associations=args.associations,
        hidden_size=args.hidden_size,
        num_heads=args.heads,
        qk_bits=args.qk_bits,
        value_heads=args.value_heads,
        value_bits=args.value_bits,
        context_scale=args.context_scale,
        scale=args.scale,
        dropout_p=0.0,
        mismatch_scale=args.mismatch_scale,
        operator=args.operator,
        estimator="production",
        bit_temperature=args.bit_temperature,
        antithetic_pairs=args.antithetic_pairs,
    ).to(device)

    residual_model = copy.deepcopy(initial_model)
    torch.manual_seed(500_000 + seed)
    residual_training = _train(
        residual_model,
        train_batch,
        route_mode="zero",
        steps=args.baseline_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    if not math.isfinite(float(residual_training["training_loss"])):
        residual_training["training_loss"] = None
    with torch.no_grad():
        residual_logits = residual_model(
            validation_batch.tokens, route_mode="zero"
        )
        residual_exact, residual_bit = _exact_and_bit_accuracy(
            residual_logits, validation_batch
        )
        residual_loss = float(_recall_loss(residual_logits, validation_batch))

    candidates = {}
    for estimator in args.estimators:
        model = copy.deepcopy(initial_model)
        model.estimator = estimator
        model.dropout_p = _dropout_for(estimator, args.dropout_p)
        torch.manual_seed(400_000 + seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(400_000 + seed)
        training = _train_estimator(
            model,
            train_batch,
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
        )
        train_metrics = evaluate_model(model, train_batch)
        validation_metrics = evaluate_model(model, validation_batch)
        candidates[estimator] = {
            "training": training,
            "train": train_metrics,
            "validation": validation_metrics,
            "passed": _candidate_passed(train_metrics, validation_metrics),
        }

    paired_targets_are_complements = bool(
        torch.equal(
            validation_batch.targets[0::2],
            -validation_batch.targets[1::2],
        )
    )
    payload_routes_are_strictly_historical = bool(
        torch.all(
            validation_batch.payload_route_indices
            < validation_batch.query_positions
        )
    )
    return {
        "seed": seed,
        "candidates": candidates,
        "residual_only_baseline": {
            **residual_training,
            "validation_loss": residual_loss,
            "validation_exact_accuracy": residual_exact,
            "validation_bit_accuracy": residual_bit,
            "theoretical_bit_loss_floor": math.log(2.0),
            "exact_accuracy_ceiling": 0.5,
        },
        "paired_targets_are_complements": paired_targets_are_complements,
        "payload_routes_are_strictly_historical": (
            payload_routes_are_strictly_historical
        ),
    }


def _summarize_estimator(runs, estimator: str) -> Dict[str, object]:
    candidates = [run["candidates"][estimator] for run in runs]
    validation = [candidate["validation"] for candidate in candidates]
    first_steps = [
        candidate["training"]["first_train_exact_step"]
        for candidate in candidates
        if candidate["training"]["first_train_exact_step"] is not None
    ]
    return {
        "passed_runs": sum(bool(candidate["passed"]) for candidate in candidates),
        "run_count": len(candidates),
        "mean_validation_exact_accuracy": statistics.mean(
            float(metrics["exact_accuracy"]) for metrics in validation
        ),
        "minimum_validation_exact_accuracy": min(
            float(metrics["exact_accuracy"]) for metrics in validation
        ),
        "mean_validation_bit_accuracy": statistics.mean(
            float(metrics["bit_accuracy"]) for metrics in validation
        ),
        "mean_payload_route_any_head_accuracy": statistics.mean(
            float(metrics["payload_route_any_head_accuracy"])
            for metrics in validation
        ),
        "mean_historical_route_any_head_accuracy": statistics.mean(
            float(metrics["historical_route_any_head_accuracy"])
            for metrics in validation
        ),
        "mean_paired_routed_value_difference_fraction": statistics.mean(
            float(metrics["paired_routed_value_difference_fraction"])
            for metrics in validation
        ),
        "median_first_train_exact_step": (
            statistics.median(first_steps) if first_steps else None
        ),
        "mean_qk_zero_gradient_fraction": statistics.mean(
            float(candidate["training"]["qk_zero_gradient_fraction"])
            for candidate in candidates
        ),
        "mean_training_step_ms": statistics.mean(
            float(candidate["training"]["step_ms"])
            for candidate in candidates
        ),
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    _validate_operator(args.operator, device)
    runs = [run_seed(args, seed) for seed in args.seeds]
    return {
        "schema_version": 1,
        "operator": args.operator,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "estimators": args.estimators,
        "seeds": args.seeds,
        "train_pairs": args.train_pairs,
        "validation_pairs": args.validation_pairs,
        "associations": args.associations,
        "hidden_size": args.hidden_size,
        "heads": args.heads,
        "qk_bits": args.qk_bits,
        "value_heads": args.value_heads,
        "value_bits": args.value_bits,
        "steps": args.steps,
        "bit_temperature": args.bit_temperature,
        "antithetic_pairs": args.antithetic_pairs,
        "dropout_p": args.dropout_p,
        "runs": runs,
        "summary": {
            estimator: _summarize_estimator(runs, estimator)
            for estimator in args.estimators
        },
        "shortcut_checks_passed": all(
            bool(run["paired_targets_are_complements"])
            and bool(run["payload_routes_are_strictly_historical"])
            and run["residual_only_baseline"]["validation_exact_accuracy"] <= 0.5
            for run in runs
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimators", nargs="+", choices=ESTIMATORS, default=list(ESTIMATORS)
    )
    parser.add_argument(
        "--operator", choices=("reference", "cuda"), default="reference"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--train-pairs", type=int, default=64)
    parser.add_argument("--validation-pairs", type=int, default=32)
    parser.add_argument("--associations", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--qk-bits", type=int, default=8)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--context-scale", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--baseline-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bit-temperature", type=float, default=0.5)
    parser.add_argument("--antithetic-pairs", type=int, default=4)
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    encoded = json.dumps(
        report["summary"] if args.summary_only else report,
        indent=2,
        allow_nan=False,
    )
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
