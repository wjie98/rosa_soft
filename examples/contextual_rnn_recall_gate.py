"""Contextual reset-RNN associative-recall gate for RosaSoft.

Each episode stores cue/payload pairs, resets the recurrent state, and then
queries every cue. Paired episodes use complementary payloads but have
identical post-reset inputs. The query residual is therefore exactly
identical across assignments, while the correct answer remains available only
through hard routes into the pre-reset history.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft
from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.soft_reference import _expand_value_heads, _hard_sign
from rosa_soft.testing import inspect_rosa_soft


@dataclass(frozen=True)
class ContextualRecallBatch:
    tokens: Tensor
    targets: Tensor
    query_positions: Tensor
    payload_route_indices: Tensor
    reset_position: int
    reset_token: int

    def to(self, device: torch.device) -> "ContextualRecallBatch":
        return ContextualRecallBatch(
            tokens=self.tokens.to(device),
            targets=self.targets.to(device),
            query_positions=self.query_positions.to(device),
            payload_route_indices=self.payload_route_indices.to(device),
            reset_position=self.reset_position,
            reset_token=self.reset_token,
        )


def _binary_codes(code_ids: Tensor, width: int) -> Tensor:
    shifts = torch.arange(width, dtype=torch.int64)
    bits = ((code_ids.unsqueeze(-1) >> shifts) & 1).bool()
    return torch.where(bits, 1.0, -1.0)


def make_contextual_recall_batch(
    *,
    seed: int,
    pairs: int,
    associations: int,
    value_bits: int,
) -> ContextualRecallBatch:
    if pairs < 1:
        raise ValueError("pairs must be >= 1")
    if associations < 1:
        raise ValueError("associations must be >= 1")
    if value_bits < 1 or value_bits > 16:
        raise ValueError("value_bits must be in [1, 16]")
    code_count = 1 << value_bits
    if associations > code_count:
        raise ValueError("associations cannot exceed the value code count")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = 2 * pairs
    reset_position = 2 * associations
    query_positions = torch.arange(
        reset_position + 1,
        3 * associations + 1,
    )
    payload_route_indices = torch.arange(
        1,
        2 * associations,
        2,
    )
    reset_token = associations + code_count
    tokens = torch.empty(
        batch,
        3 * associations + 1,
        dtype=torch.int64,
    )
    targets = torch.empty(
        batch,
        associations,
        value_bits,
        dtype=torch.float32,
    )
    complement_mask = code_count - 1
    cue_tokens = torch.arange(associations)

    for pair in range(pairs):
        first_codes = torch.randperm(
            code_count,
            generator=generator,
        )[:associations]
        for side in range(2):
            batch_index = 2 * pair + side
            payload_codes = (
                first_codes
                if side == 0
                else first_codes ^ complement_mask
            )
            tokens[batch_index, 0:reset_position:2] = cue_tokens
            tokens[batch_index, 1:reset_position:2] = (
                associations + payload_codes
            )
            tokens[batch_index, reset_position] = reset_token
            tokens[batch_index, query_positions] = cue_tokens
            targets[batch_index] = _binary_codes(
                payload_codes,
                value_bits,
            )

    return ContextualRecallBatch(
        tokens=tokens,
        targets=targets,
        query_positions=query_positions,
        payload_route_indices=payload_route_indices,
        reset_position=reset_position,
        reset_token=reset_token,
    )


class ResetRnnRosaLM(nn.Module):
    def __init__(
        self,
        *,
        associations: int,
        hidden_size: int,
        num_heads: int,
        qk_bits: int,
        value_heads: int,
        value_bits: int,
        context_scale: float,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
        operator: str,
    ) -> None:
        super().__init__()
        if num_heads % value_heads != 0:
            raise ValueError("num_heads must be divisible by value_heads")
        if not math.isfinite(context_scale) or context_scale <= 0.0:
            raise ValueError("context_scale must be finite and > 0")
        self.num_heads = num_heads
        self.qk_bits = qk_bits
        self.value_heads = value_heads
        self.value_bits = value_bits
        self.context_scale = float(context_scale)
        self.scale = float(scale)
        self.dropout_p = float(dropout_p)
        self.mismatch_scale = float(mismatch_scale)
        self.operator = operator
        self.reset_token = associations + (1 << value_bits)
        vocab_size = self.reset_token + 1

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.recurrent = nn.GRUCell(hidden_size, hidden_size)
        self.symbol_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(
            hidden_size,
            num_heads * qk_bits,
            bias=False,
        )
        self.key = nn.Linear(
            hidden_size,
            num_heads * qk_bits,
            bias=False,
        )
        self.value = nn.Linear(
            hidden_size,
            value_heads * value_bits,
            bias=False,
        )
        self.output = nn.Linear(
            num_heads * value_bits,
            hidden_size,
            bias=False,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, value_bits, bias=False)

    def encode_residual(self, tokens: Tensor) -> Tensor:
        inputs = self.embedding(tokens)
        state = torch.zeros(
            tokens.size(0),
            inputs.size(-1),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        residuals = []
        for position in range(tokens.size(1)):
            state = self.recurrent(inputs[:, position], state)
            reset = (tokens[:, position] == self.reset_token).unsqueeze(-1)
            state = torch.where(reset, torch.zeros_like(state), state)
            residuals.append(
                inputs[:, position] + self.context_scale * state
            )
        return torch.stack(residuals, dim=1)

    def project_symbols(
        self,
        residual: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.symbol_norm(residual)
        batch, sequence_length, _ = hidden.shape
        query = self.query(hidden).view(
            batch,
            sequence_length,
            self.num_heads,
            self.qk_bits,
        )
        key = self.key(hidden).view(
            batch,
            sequence_length,
            self.num_heads,
            self.qk_bits,
        )
        value = self.value(hidden).view(
            batch,
            sequence_length,
            self.value_heads,
            self.value_bits,
        )
        return query, key, value

    def _routed_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        route_mode: str,
    ) -> Tensor:
        if route_mode == "zero":
            return torch.zeros(
                query.size(0),
                query.size(1),
                self.num_heads,
                self.value_bits,
                dtype=query.dtype,
                device=query.device,
            )
        if route_mode == "current-value":
            return _expand_value_heads(
                _hard_sign(value),
                self.num_heads,
            ).permute(0, 2, 1, 3)
        if route_mode != "rosa":
            raise ValueError(f"unknown route_mode: {route_mode}")
        training_operator = (
            rosa_soft.rosa_soft
            if self.operator == "cuda"
            else rosa_soft.rosa_soft_reference
        )
        return training_operator(
            query,
            key,
            value,
            max_suffix_length=1,
            scale=self.scale,
            dropout_p=self.dropout_p if self.training else 0.0,
            mismatch_scale=self.mismatch_scale,
        )

    def forward(
        self,
        tokens: Tensor,
        *,
        route_mode: str = "rosa",
    ) -> Tensor:
        residual = self.encode_residual(tokens)
        query, key, value = self.project_symbols(residual)
        routed = self._routed_values(
            query,
            key,
            value,
            route_mode,
        )
        hidden = residual + self.output(routed.flatten(2))
        return self.head(self.output_norm(hidden))


def _recall_loss(
    logits: Tensor,
    recall_batch: ContextualRecallBatch,
) -> Tensor:
    selected = logits[:, recall_batch.query_positions].float()
    return F.softplus(-recall_batch.targets * selected).mean()


def _exact_and_bit_accuracy(
    logits: Tensor,
    recall_batch: ContextualRecallBatch,
) -> tuple[float, float]:
    selected = logits[:, recall_batch.query_positions]
    prediction = torch.where(
        selected > 0,
        torch.ones_like(selected),
        -torch.ones_like(selected),
    )
    correct = prediction == recall_batch.targets
    return (
        float(correct.all(dim=-1).float().mean()),
        float(correct.float().mean()),
    )


@torch.no_grad()
def evaluate_model(
    model: ResetRnnRosaLM,
    recall_batch: ContextualRecallBatch,
) -> Dict[str, float]:
    model.eval()
    residual = model.encode_residual(recall_batch.tokens)
    query, key, value = model.project_symbols(residual)
    routed, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        max_suffix_length=1,
        scale=model.scale,
        dropout_p=0.0,
        mismatch_scale=model.mismatch_scale,
    )
    hidden = residual + model.output(routed.flatten(2))
    logits = model.head(model.output_norm(hidden))
    exact_accuracy, bit_accuracy = _exact_and_bit_accuracy(
        logits,
        recall_batch,
    )
    payload_routes = recall_batch.payload_route_indices.view(1, 1, -1)
    selected_routes = inspection.selected_route_indices[
        :,
        :,
        recall_batch.query_positions,
    ]
    payload_route = selected_routes == payload_routes
    query_positions = recall_batch.query_positions.view(1, 1, -1)
    historical_route = (
        (selected_routes > 0) & (selected_routes < query_positions)
    )
    routed_queries = routed[:, recall_batch.query_positions]
    paired_route_difference = (
        selected_routes[0::2] != selected_routes[1::2]
    ).any(dim=1)
    paired_routed_value_difference = (
        routed_queries[0::2] != routed_queries[1::2]
    ).any(dim=(-1, -2))

    zero_logits = model(recall_batch.tokens, route_mode="zero")
    current_logits = model(
        recall_batch.tokens,
        route_mode="current-value",
    )
    zero_exact, zero_bit = _exact_and_bit_accuracy(
        zero_logits,
        recall_batch,
    )
    current_exact, current_bit = _exact_and_bit_accuracy(
        current_logits,
        recall_batch,
    )
    query_residual = residual[:, recall_batch.query_positions]
    residual_difference = (
        query_residual - query_residual[:1]
    ).abs().max()
    return {
        "loss": float(_recall_loss(logits, recall_batch)),
        "exact_accuracy": exact_accuracy,
        "bit_accuracy": bit_accuracy,
        "payload_route_any_head_accuracy": float(
            payload_route.any(dim=1).float().mean()
        ),
        "payload_route_head_fraction": float(
            payload_route.float().mean()
        ),
        "historical_route_any_head_accuracy": float(
            historical_route.any(dim=1).float().mean()
        ),
        "historical_route_head_fraction": float(
            historical_route.float().mean()
        ),
        "nonnull_route_fraction": float(
            (selected_routes != 0).float().mean()
        ),
        "paired_route_difference_fraction": float(
            paired_route_difference.float().mean()
        ),
        "paired_routed_value_difference_fraction": float(
            paired_routed_value_difference.float().mean()
        ),
        "zero_route_exact_accuracy": zero_exact,
        "zero_route_bit_accuracy": zero_bit,
        "current_value_exact_accuracy": current_exact,
        "current_value_bit_accuracy": current_bit,
        "query_residual_max_difference": float(residual_difference),
    }


def _train(
    model: ResetRnnRosaLM,
    recall_batch: ContextualRecallBatch,
    *,
    route_mode: str,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
) -> Dict[str, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    started = time.perf_counter()
    final_loss = float("nan")
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            recall_batch.tokens,
            route_mode=route_mode,
        )
        loss = _recall_loss(logits, recall_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )
        optimizer.step()
        final_loss = float(loss.detach())
    if recall_batch.tokens.is_cuda:
        torch.cuda.synchronize(recall_batch.tokens.device)
    elapsed = time.perf_counter() - started
    return {
        "training_loss": final_loss,
        "step_ms": (
            elapsed * 1000.0 / steps if steps > 0 else 0.0
        ),
    }


def _validate_operator(
    operator: str,
    device: torch.device,
) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if operator == "cuda":
        if device.type != "cuda":
            raise ValueError("--operator cuda requires --device cuda")
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")


def run_seed(
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, object]:
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
    initial_model = ResetRnnRosaLM(
        associations=args.associations,
        hidden_size=args.hidden_size,
        num_heads=args.heads,
        qk_bits=args.qk_bits,
        value_heads=args.value_heads,
        value_bits=args.value_bits,
        context_scale=args.context_scale,
        scale=args.scale,
        dropout_p=args.dropout_p,
        mismatch_scale=args.mismatch_scale,
        operator=args.operator,
    ).to(device)
    rosa_model = copy.deepcopy(initial_model)
    residual_model = copy.deepcopy(initial_model)

    torch.manual_seed(400_000 + seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(400_000 + seed)
    rosa_training = _train(
        rosa_model,
        train_batch,
        route_mode="rosa",
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    residual_training = _train(
        residual_model,
        train_batch,
        route_mode="zero",
        steps=args.baseline_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    train_metrics = evaluate_model(rosa_model, train_batch)
    validation_metrics = evaluate_model(
        rosa_model,
        validation_batch,
    )
    with torch.no_grad():
        residual_validation_logits = residual_model(
            validation_batch.tokens,
            route_mode="zero",
        )
        residual_exact, residual_bit = _exact_and_bit_accuracy(
            residual_validation_logits,
            validation_batch,
        )
        residual_loss = float(
            _recall_loss(
                residual_validation_logits,
                validation_batch,
            )
        )
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
    passed = bool(
        train_metrics["exact_accuracy"] >= 0.99
        and validation_metrics["exact_accuracy"] >= 0.99
        and validation_metrics["historical_route_any_head_accuracy"] >= 0.99
        and validation_metrics[
            "paired_routed_value_difference_fraction"
        ] >= 0.99
        and validation_metrics["zero_route_exact_accuracy"] <= 0.5
        and validation_metrics["current_value_exact_accuracy"] <= 0.5
        and validation_metrics["query_residual_max_difference"] == 0.0
        and residual_exact <= 0.5
        and paired_targets_are_complements
        and payload_routes_are_strictly_historical
    )
    return {
        "seed": seed,
        "train": train_metrics,
        "validation": validation_metrics,
        "residual_only_baseline": {
            **residual_training,
            "validation_loss": residual_loss,
            "validation_exact_accuracy": residual_exact,
            "validation_bit_accuracy": residual_bit,
            "theoretical_bit_loss_floor": math.log(2.0),
            "exact_accuracy_ceiling": 0.5,
        },
        "rosa_training": rosa_training,
        "paired_targets_are_complements": paired_targets_are_complements,
        "payload_routes_are_strictly_historical": (
            payload_routes_are_strictly_historical
        ),
        "passed": passed,
    }


def run_gate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    _validate_operator(args.operator, device)
    runs = [run_seed(args, seed) for seed in args.seeds]
    validation = [run["validation"] for run in runs]
    return {
        "schema_version": 1,
        "operator": args.operator,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "seeds": args.seeds,
        "train_pairs": args.train_pairs,
        "validation_pairs": args.validation_pairs,
        "associations": args.associations,
        "hidden_size": args.hidden_size,
        "heads": args.heads,
        "qk_bits": args.qk_bits,
        "value_heads": args.value_heads,
        "value_bits": args.value_bits,
        "context_scale": args.context_scale,
        "steps": args.steps,
        "baseline_steps": args.baseline_steps,
        "scale": args.scale,
        "dropout_p": args.dropout_p,
        "mismatch_scale": args.mismatch_scale,
        "runs": runs,
        "summary": {
            "passed_runs": sum(bool(run["passed"]) for run in runs),
            "run_count": len(runs),
            "mean_validation_exact_accuracy": statistics.mean(
                float(metrics["exact_accuracy"]) for metrics in validation
            ),
            "minimum_validation_exact_accuracy": min(
                float(metrics["exact_accuracy"]) for metrics in validation
            ),
            "mean_payload_route_any_head_accuracy": statistics.mean(
                float(metrics["payload_route_any_head_accuracy"])
                for metrics in validation
            ),
            "mean_historical_route_any_head_accuracy": statistics.mean(
                float(metrics["historical_route_any_head_accuracy"])
                for metrics in validation
            ),
            "mean_paired_routed_value_difference_fraction": (
                statistics.mean(
                    float(
                        metrics[
                            "paired_routed_value_difference_fraction"
                        ]
                    )
                    for metrics in validation
                )
            ),
            "mean_zero_route_exact_accuracy": statistics.mean(
                float(metrics["zero_route_exact_accuracy"])
                for metrics in validation
            ),
            "mean_current_value_exact_accuracy": statistics.mean(
                float(metrics["current_value_exact_accuracy"])
                for metrics in validation
            ),
            "mean_residual_only_exact_accuracy": statistics.mean(
                float(
                    run["residual_only_baseline"][
                        "validation_exact_accuracy"
                    ]
                )
                for run in runs
            ),
            "mean_training_step_ms": statistics.mean(
                float(run["rosa_training"]["step_ms"]) for run in runs
            ),
        },
        "passed": all(bool(run["passed"]) for run in runs),
    }


def gate_exit_code(report: Dict[str, object]) -> int:
    return 0 if report.get("passed") is True else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator",
        choices=("reference", "cuda"),
        default="reference",
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
    parser.add_argument(
        "--scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_SCALE,
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=ROSA_SOFT_DEFAULT_DROPOUT_P,
    )
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gate(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return gate_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
