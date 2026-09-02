"""Full next-token gate for the parameter-free state-attention ROSA VJP.

Each episode first presents random cue/payload associations. Every later query
is preceded by a cue-specific reset marker, so structural next tokens remain
predictable while the query residual itself is assignment-independent. The
single training objective is uniform next-token cross entropy over the entire
sequence; recall positions receive no extra loss or weight.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.contextual_estimator_recall import (  # noqa: E402
    ESTIMATORS,
    EstimatorResetRnnRosaLM,
    _gradient_norm,
)
from benchmarks.state_attention_init import (  # noqa: E402
    QK_INITIALIZATIONS,
    VALUE_INITIALIZATIONS,
    initialize_state_attention_model,
)
from examples.contextual_rnn_recall_gate import _validate_operator  # noqa: E402
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.testing import inspect_rosa_soft  # noqa: E402


@dataclass(frozen=True)
class NextTokenRecallBatch:
    tokens: Tensor
    query_positions: Tensor
    storage_payload_positions: Tensor
    associations: int
    payload_bits: int
    reset_start: int
    reset_end: int

    def to(self, device: torch.device) -> "NextTokenRecallBatch":
        return NextTokenRecallBatch(
            tokens=self.tokens.to(device),
            query_positions=self.query_positions.to(device),
            storage_payload_positions=self.storage_payload_positions.to(device),
            associations=self.associations,
            payload_bits=self.payload_bits,
            reset_start=self.reset_start,
            reset_end=self.reset_end,
        )


def make_next_token_recall_batch(
    *,
    seed: int,
    pairs: int,
    associations: int,
    payload_bits: int,
) -> NextTokenRecallBatch:
    if pairs < 1:
        raise ValueError("pairs must be >= 1")
    if associations < 1:
        raise ValueError("associations must be >= 1")
    if payload_bits < 1 or payload_bits > 16:
        raise ValueError("payload_bits must be in [1, 16]")
    code_count = 1 << payload_bits
    if associations > code_count:
        raise ValueError("associations cannot exceed the value code count")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = 2 * pairs
    reset_start = associations + code_count
    reset_end = reset_start + associations
    sequence_length = 5 * associations
    tokens = torch.empty(batch_size, sequence_length, dtype=torch.int64)
    cue_tokens = torch.arange(associations)
    reset_tokens = torch.arange(reset_start, reset_end)
    storage_payload_positions = torch.arange(1, 2 * associations, 2)
    query_positions = 2 * associations + 3 * torch.arange(associations) + 1
    complement_mask = code_count - 1

    random_priority = torch.rand(pairs, code_count, generator=generator)
    first_codes = random_priority.topk(associations, dim=-1).indices
    payload_codes = torch.stack(
        (first_codes, first_codes ^ complement_mask),
        dim=1,
    ).reshape(batch_size, associations)
    payload_tokens = associations + payload_codes

    tokens[:, 0 : 2 * associations : 2] = cue_tokens
    tokens[:, 1 : 2 * associations : 2] = payload_tokens
    query_phase = tokens[:, 2 * associations :].view(
        batch_size,
        associations,
        3,
    )
    query_phase[:, :, 0] = reset_tokens
    query_phase[:, :, 1] = cue_tokens
    query_phase[:, :, 2] = payload_tokens

    return NextTokenRecallBatch(
        tokens=tokens,
        query_positions=query_positions,
        storage_payload_positions=storage_payload_positions,
        associations=associations,
        payload_bits=payload_bits,
        reset_start=reset_start,
        reset_end=reset_end,
    )


class NextTokenEstimatorLM(EstimatorResetRnnRosaLM):
    def __init__(
        self,
        *,
        associations: int,
        hidden_size: int,
        payload_bits: int,
        value_bits: int,
        **kwargs,
    ) -> None:
        super().__init__(
            associations=associations,
            hidden_size=hidden_size,
            value_bits=value_bits,
            **kwargs,
        )
        self.payload_bits = int(payload_bits)
        self.reset_start = associations + (1 << payload_bits)
        self.reset_end = self.reset_start + associations
        self.vocab_size = self.reset_end
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, self.vocab_size, bias=False)

    def encode_residual(self, tokens: Tensor) -> Tensor:
        inputs = self.embedding(tokens)
        cells = (self.recurrent, *self.extra_recurrent)
        states = [torch.zeros_like(inputs[:, 0]) for _ in cells]
        residuals = []
        for position in range(tokens.size(1)):
            hidden = inputs[:, position]
            token = tokens[:, position]
            reset = (
                (token >= self.reset_start) & (token < self.reset_end)
            ).unsqueeze(-1)
            for layer, cell in enumerate(cells):
                state = cell(hidden, states[layer])
                state = torch.where(reset, torch.zeros_like(state), state)
                states[layer] = state
                hidden = hidden + self.context_scale * state
            residuals.append(hidden)
        return torch.stack(residuals, dim=1)


def _next_token_loss(logits: Tensor, tokens: Tensor) -> Tensor:
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )


def _theoretical_limits(associations: int, payload_bits: int) -> dict[str, float]:
    code_count = 1 << payload_bits
    target_count = 5 * associations - 1
    remaining_codes = range(code_count, code_count - associations, -1)
    return {
        "loss_floor": sum(math.log(count) for count in remaining_codes)
        / target_count,
        "top1_accuracy_ceiling": (
            target_count
            - associations
            + sum(1.0 / count for count in remaining_codes)
        )
        / target_count,
    }


def _token_metrics(
    logits: Tensor,
    batch: NextTokenRecallBatch,
) -> dict[str, float]:
    prediction = logits[:, :-1].argmax(dim=-1)
    targets = batch.tokens[:, 1:]
    recall_logits = logits[:, batch.query_positions]
    recall_targets = batch.tokens[:, batch.query_positions + 1]
    recall_correct = recall_logits.argmax(dim=-1) == recall_targets
    return {
        "loss": float(_next_token_loss(logits, batch.tokens)),
        "overall_accuracy": float((prediction == targets).float().mean()),
        "recall_loss": float(
            F.cross_entropy(
                recall_logits.float().reshape(-1, logits.size(-1)),
                recall_targets.reshape(-1),
            )
        ),
        "recall_token_accuracy": float(recall_correct.float().mean()),
        "recall_episode_exact_accuracy": float(
            recall_correct.all(dim=-1).float().mean()
        ),
    }


def _train(
    model: NextTokenEstimatorLM,
    *,
    data_seed: int,
    pairs: int,
    associations: int,
    payload_bits: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
) -> dict[str, object]:
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    first_exact = None
    qk_norms = []
    started = time.perf_counter()
    final_loss = None
    for step in range(1, steps + 1):
        batch = make_next_token_recall_batch(
            seed=data_seed + step,
            pairs=pairs,
            associations=associations,
            payload_bits=payload_bits,
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.tokens, route_mode="rosa")
        loss = _next_token_loss(logits, batch.tokens)
        recall_correct = (
            logits[:, batch.query_positions].argmax(dim=-1)
            == batch.tokens[:, batch.query_positions + 1]
        )
        if first_exact is None and float(
            recall_correct.all(dim=-1).float().mean()
        ) >= 0.99:
            first_exact = step
        loss.backward()
        qk_norms.append(
            _gradient_norm((*model.query.parameters(), *model.key.parameters()))
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        final_loss = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "training_loss": final_loss,
        "first_train_recall_exact_step": first_exact,
        "mean_qk_gradient_norm": statistics.mean(qk_norms) if qk_norms else 0.0,
        "qk_zero_gradient_fraction": (
            sum(norm == 0.0 for norm in qk_norms) / len(qk_norms)
            if qk_norms
            else 0.0
        ),
        "step_ms": elapsed * 1000.0 / steps if steps else 0.0,
    }


@torch.no_grad()
def evaluate_model(
    model: NextTokenEstimatorLM,
    batch: NextTokenRecallBatch,
) -> dict[str, float]:
    model.eval()
    residual = model.encode_residual(batch.tokens)
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
    metrics = _token_metrics(logits, batch)

    selected = inspection.selected_route_indices[:, :, batch.query_positions]
    payload_routes = batch.storage_payload_positions.view(1, 1, -1)
    payload_route = selected == payload_routes
    historical = (
        (selected > 0)
        & (selected < batch.query_positions.view(1, 1, -1))
    )
    zero_metrics = _token_metrics(
        model(batch.tokens, route_mode="zero"),
        batch,
    )
    current_metrics = _token_metrics(
        model(batch.tokens, route_mode="current-value"),
        batch,
    )
    query_residual = residual[:, batch.query_positions]
    pair_difference = (
        query_residual[0::2] - query_residual[1::2]
    ).abs().max()
    extra_metrics = {}
    value_metrics = getattr(model, "value_evaluation_metrics", None)
    if value_metrics is not None:
        extra_metrics = value_metrics(value, batch)
    return {
        **metrics,
        "payload_route_any_head_accuracy": float(
            payload_route.any(dim=1).float().mean()
        ),
        "historical_route_any_head_accuracy": float(
            historical.any(dim=1).float().mean()
        ),
        "zero_route_recall_token_accuracy": zero_metrics[
            "recall_token_accuracy"
        ],
        "current_value_recall_token_accuracy": current_metrics[
            "recall_token_accuracy"
        ],
        "paired_query_residual_max_difference": float(pair_difference),
        **extra_metrics,
    }


@torch.no_grad()
def inspect_initialization(
    model: NextTokenEstimatorLM,
    batch: NextTokenRecallBatch,
) -> dict[str, float]:
    model.eval()
    residual = model.encode_residual(batch.tokens)
    query, key, value = model.project_symbols(residual)
    _, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        max_suffix_length=1,
        scale=model.scale,
        dropout_p=0.0,
        mismatch_scale=model.mismatch_scale,
    )
    selected = inspection.selected_route_indices[:, :, batch.query_positions]
    payload_routes = batch.storage_payload_positions.view(1, 1, -1)
    query_positions = batch.query_positions.view(1, 1, -1)

    query_symbols = query[:, batch.query_positions] > 0
    storage_key_positions = batch.storage_payload_positions - 1
    key_symbols = key[:, storage_key_positions] > 0
    positive_hamming = (query_symbols != key_symbols).float().sum(dim=-1)

    represent_value = getattr(model, "represent_value", None)
    represented_value = (
        represent_value(value)
        if represent_value is not None
        else torch.where(value > 0, 1.0, -1.0)
    )
    storage_values = represented_value[:, batch.storage_payload_positions]
    unique_code_fractions = []
    for batch_index in range(storage_values.size(0)):
        for value_head in range(storage_values.size(2)):
            codes = storage_values[batch_index, :, value_head]
            unique_code_fractions.append(
                codes.unique(dim=0).size(0) / batch.associations
            )
    weight_cosine = F.cosine_similarity(
        model.query.weight,
        model.key.weight,
        dim=-1,
    )
    return {
        "query_key_weight_cosine_mean": float(weight_cosine.mean()),
        "query_logit_abs_median": float(query.abs().median()),
        "key_logit_abs_median": float(key.abs().median()),
        "value_logit_abs_median": float(value.abs().median()),
        "positive_qk_hamming_mean": float(positive_hamming.mean()),
        "positive_qk_exact_any_head_accuracy": float(
            (positive_hamming == 0).any(dim=-1).float().mean()
        ),
        "payload_route_any_head_accuracy": float(
            (selected == payload_routes).any(dim=1).float().mean()
        ),
        "historical_route_any_head_accuracy": float(
            (
                (selected > 0)
                & (selected < query_positions)
            ).any(dim=1).float().mean()
        ),
        "null_route_head_fraction": float((selected == 0).float().mean()),
        "storage_value_unique_code_fraction": statistics.mean(
            unique_code_fractions
        ),
        "storage_value_bit_balance_error": float(
            (
                (storage_values > 0).float().mean(dim=(0, 1)) - 0.5
            ).abs().mean()
        ),
        "storage_value_rms": float(storage_values.float().square().mean().sqrt()),
        "storage_value_max_abs": float(storage_values.float().abs().max()),
    }


def _targets_are_complements(batch: NextTokenRecallBatch) -> bool:
    targets = batch.tokens[:, batch.query_positions + 1] - batch.associations
    complement_mask = (1 << batch.payload_bits) - 1
    return bool(torch.equal(targets[0::2] ^ complement_mask, targets[1::2]))


def _candidate_passed(
    train: dict[str, float],
    validation: dict[str, float],
) -> bool:
    return bool(
        train["recall_episode_exact_accuracy"] >= 0.99
        and validation["recall_episode_exact_accuracy"] >= 0.99
        and validation["payload_route_any_head_accuracy"] >= 0.99
        and validation["zero_route_recall_token_accuracy"] <= 0.5
        and validation["current_value_recall_token_accuracy"] <= 0.5
        and validation["paired_query_residual_max_difference"] == 0.0
    )


def _uses_controlled_module_reset(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "controlled_module_reset", False)
        or args.qk_init != "default"
        or args.value_init != "default"
    )


def run_seed(
    args: argparse.Namespace,
    model_seed: int,
    data_seed: int,
    model_factory=NextTokenEstimatorLM,
) -> dict[str, object]:
    device = torch.device(args.device)
    train_evaluation_batch = make_next_token_recall_batch(
        seed=100_000 + data_seed,
        pairs=args.train_pairs,
        associations=args.associations,
        payload_bits=args.payload_bits,
    ).to(device)
    validation_batch = make_next_token_recall_batch(
        seed=args.validation_seed,
        pairs=args.validation_pairs,
        associations=args.associations,
        payload_bits=args.payload_bits,
    ).to(device)
    torch.manual_seed(300_000 + model_seed)
    initial_model = model_factory(
        associations=args.associations,
        hidden_size=args.hidden_size,
        payload_bits=args.payload_bits,
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
        bit_temperature=0.5,
        antithetic_pairs=1,
        bitflip_gradient_scale=args.bitflip_gradient_scale,
        context_depth=args.context_depth,
    )
    if _uses_controlled_module_reset(args):
        initialize_state_attention_model(
            initial_model,
            seed=300_000 + model_seed,
            qk_init=args.qk_init,
            qk_shared_bits=args.qk_shared_bits,
            qk_correlation=args.qk_correlation,
            qk_logit_median=args.qk_logit_median,
            value_init=args.value_init,
            value_logit_median=args.value_logit_median,
        )
    initial_model = initial_model.to(device)
    initialization = inspect_initialization(
        initial_model,
        train_evaluation_batch,
    )

    candidates = {}
    for estimator in args.estimators:
        model = copy.deepcopy(initial_model)
        model.estimator = estimator
        optimizer_seed = 400_000 + 10_007 * model_seed + data_seed
        torch.manual_seed(optimizer_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(optimizer_seed)
        training = _train(
            model,
            data_seed=1_000_000 + 100_000 * data_seed,
            pairs=args.train_pairs,
            associations=args.associations,
            payload_bits=args.payload_bits,
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
        )
        train_metrics = evaluate_model(model, train_evaluation_batch)
        validation_metrics = evaluate_model(model, validation_batch)
        candidates[estimator] = {
            "training": training,
            "train": train_metrics,
            "validation": validation_metrics,
            "passed": _candidate_passed(train_metrics, validation_metrics),
        }
    return {
        "model_seed": model_seed,
        "data_seed": data_seed,
        "initialization": initialization,
        "candidates": candidates,
        "train_targets_are_complements": _targets_are_complements(
            train_evaluation_batch
        ),
        "validation_targets_are_complements": _targets_are_complements(
            validation_batch
        ),
    }


def _summarize(runs: Sequence[dict[str, object]], estimator: str) -> dict[str, object]:
    candidates = [run["candidates"][estimator] for run in runs]
    validation = [candidate["validation"] for candidate in candidates]
    first_steps = [
        candidate["training"]["first_train_recall_exact_step"]
        for candidate in candidates
        if candidate["training"]["first_train_recall_exact_step"] is not None
    ]
    return {
        "passed_runs": sum(bool(candidate["passed"]) for candidate in candidates),
        "run_count": len(candidates),
        "mean_validation_loss": statistics.mean(
            float(metrics["loss"]) for metrics in validation
        ),
        "mean_validation_recall_token_accuracy": statistics.mean(
            float(metrics["recall_token_accuracy"]) for metrics in validation
        ),
        "minimum_validation_recall_token_accuracy": min(
            float(metrics["recall_token_accuracy"]) for metrics in validation
        ),
        "mean_validation_recall_episode_exact_accuracy": statistics.mean(
            float(metrics["recall_episode_exact_accuracy"])
            for metrics in validation
        ),
        "mean_payload_route_any_head_accuracy": statistics.mean(
            float(metrics["payload_route_any_head_accuracy"])
            for metrics in validation
        ),
        "median_first_train_recall_exact_step": (
            statistics.median(first_steps) if first_steps else None
        ),
        "mean_training_step_ms": statistics.mean(
            float(candidate["training"]["step_ms"])
            for candidate in candidates
        ),
    }


def _seed_pairs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.model_seeds is None and args.data_seeds is None:
        return [(seed, seed) for seed in args.seeds]
    model_seeds = args.model_seeds if args.model_seeds is not None else args.seeds
    data_seeds = args.data_seeds if args.data_seeds is not None else args.seeds
    return list(itertools.product(model_seeds, data_seeds))


def run_benchmark(
    args: argparse.Namespace,
    model_factory=NextTokenEstimatorLM,
) -> dict[str, object]:
    device = torch.device(args.device)
    _validate_operator(args.operator, device)
    if any(estimator not in ESTIMATORS for estimator in args.estimators):
        raise ValueError("unknown estimator")
    seed_pairs = _seed_pairs(args)
    runs = [
        run_seed(
            args,
            model_seed,
            data_seed,
            model_factory=model_factory,
        )
        for model_seed, data_seed in seed_pairs
    ]
    return {
        "schema_version": 2,
        "objective": "uniform full-sequence next-token cross entropy",
        "operator": args.operator,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "estimators": list(args.estimators),
        "seed_pairs": [list(pair) for pair in seed_pairs],
        "associations": args.associations,
        "heads": args.heads,
        "context_depth": args.context_depth,
        "qk_bits": args.qk_bits,
        "payload_bits": args.payload_bits,
        "value_dim": args.value_bits,
        "bitflip_gradient_scale": args.bitflip_gradient_scale,
        "initialization": {
            "controlled_module_reset": _uses_controlled_module_reset(args),
            "qk_init": args.qk_init,
            "qk_shared_bits": args.qk_shared_bits,
            "qk_correlation": args.qk_correlation,
            "qk_logit_median": args.qk_logit_median,
            "value_init": args.value_init,
            "value_logit_median": args.value_logit_median,
        },
        "steps": args.steps,
        "train_pairs": args.train_pairs,
        "validation_pairs": args.validation_pairs,
        "training_data": "fresh complementary mappings each optimizer step",
        "theoretical_full_sequence_limits": _theoretical_limits(
            args.associations,
            args.payload_bits,
        ),
        "runs": runs,
        "summary": {
            estimator: _summarize(runs, estimator)
            for estimator in args.estimators
        },
        "shortcut_checks_passed": all(
            bool(run["train_targets_are_complements"])
            and bool(run["validation_targets_are_complements"])
            and all(
                candidate["validation"][
                    "paired_query_residual_max_difference"
                ]
                == 0.0
                for candidate in run["candidates"].values()
            )
            for run in runs
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=ESTIMATORS,
        default=["production", "state_quadratic_attention"],
    )
    parser.add_argument("--operator", choices=("reference", "cuda"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--model-seeds", nargs="+", type=int)
    parser.add_argument("--data-seeds", nargs="+", type=int)
    parser.add_argument("--validation-seed", type=int, default=200_000)
    parser.add_argument("--train-pairs", type=int, default=32)
    parser.add_argument("--validation-pairs", type=int, default=16)
    parser.add_argument("--associations", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--qk-bits", type=int, default=8)
    parser.add_argument("--value-heads", type=int, default=1)
    parser.add_argument("--payload-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument(
        "--qk-init",
        choices=QK_INITIALIZATIONS,
        default="default",
    )
    parser.add_argument("--qk-shared-bits", type=int, default=4)
    parser.add_argument("--qk-correlation", type=float, default=0.97)
    parser.add_argument("--qk-logit-median", type=float, default=0.6)
    parser.add_argument(
        "--value-init",
        choices=VALUE_INITIALIZATIONS,
        default="default",
    )
    parser.add_argument("--value-logit-median", type=float, default=0.6)
    parser.add_argument("--controlled-module-reset", action="store_true")
    parser.add_argument("--context-scale", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bitflip-gradient-scale", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
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
    output = report["summary"] if args.summary_only else report
    print(json.dumps(output, indent=2, allow_nan=False))
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
