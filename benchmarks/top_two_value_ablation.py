"""Compare unbounded dense, top-two, and value-credit RosaSoft VJPs.

The benchmark is intentionally independent from the production CUDA kernel.
Every estimator executes the same exact unlimited hard ROSA forward.
"""

from __future__ import annotations

import argparse
import copy
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

from benchmarks.contextual_estimator_recall import (  # noqa: E402
    _candidate_passed,
    _train_estimator,
)
from benchmarks.discrete_gradient_alignment import (  # noqa: E402
    discrete_bit_flip_oracle,
    summarize_alignment,
)
from benchmarks.dual_score_reference import (  # noqa: E402
    rosa_dual_score_reference,
)
from benchmarks.estimator_fit_ablation import (  # noqa: E402
    _ResearchFitLM,
    _evaluate,
    rosa_soft_exact_bitflip,
)
from benchmarks.top_two_value_reference import (  # noqa: E402
    competition_state,
    rosa_top_two_value_reference,
)
from examples.contextual_rnn_recall_gate import (  # noqa: E402
    ResetRnnRosaLM,
    evaluate_model,
    make_contextual_recall_batch,
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
from rosa_soft.soft_reference import (  # noqa: E402
    _hard_sign,
    rosa_soft_reference,
)


ESTIMATORS = (
    "production_full",
    "split_dense",
    "selected_value",
    "winner_rest",
    "hard_winner_rest",
    "no_prior",
    "hard_null_gate",
    "top2_qk",
    "top2_selected",
    "dual",
    "bitflip",
)
GRADIENT_ESTIMATORS = tuple(
    estimator for estimator in ESTIMATORS if estimator != "bitflip"
)

Estimator = Callable[..., Tensor]


def _operator(name: str) -> Estimator:
    if name not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")

    def apply(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        max_suffix_length: Optional[int] = None,
        scale: float,
        mismatch_scale: float,
    ) -> Tensor:
        del max_suffix_length
        if name == "production_full":
            return rosa_soft_reference(
                query,
                key,
                value,
                max_suffix_length=query.size(1),
                scale=scale,
                mismatch_scale=mismatch_scale,
            )
        if name == "split_dense":
            return rosa_top_two_value_reference(
                query,
                key,
                value,
                scale=scale,
                mismatch_scale=mismatch_scale,
            )
        if name in (
            "selected_value",
            "winner_rest",
            "hard_winner_rest",
            "no_prior",
            "hard_null_gate",
            "top2_qk",
            "top2_selected",
        ):
            return rosa_top_two_value_reference(
                query,
                key,
                value,
                scale=scale,
                mismatch_scale=mismatch_scale,
                qk_gradient=(
                    "top2"
                    if name.startswith("top2")
                    else name
                    if name in (
                        "winner_rest",
                        "hard_winner_rest",
                        "no_prior",
                        "hard_null_gate",
                    )
                    else "dense"
                ),
                value_gradient=(
                    "selected" if name.endswith("selected") or
                    name == "selected_value" else "dense"
                ),
            )
        if name == "dual":
            return rosa_dual_score_reference(
                query,
                key,
                value,
                score_mode="dual",
                scale=scale,
                dropout_p=0.0,
                mismatch_scale=mismatch_scale,
            )
        return rosa_soft_exact_bitflip(
            query,
            key,
            value,
            max_suffix_length=query.size(1),
            scale=scale,
            mismatch_scale=mismatch_scale,
        )

    return apply


def _nonzero_randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    values = torch.randn(shape, generator=generator, device=device)
    return torch.where(values >= 0.0, 1.0, -1.0) * (values.abs() + 0.2)


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def run_gradient_matrix(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    records = []
    max_value_dim = max(args.gradient_value_bits)
    for seed in args.gradient_seeds:
        for seq_len in args.gradient_lengths:
            for symbol_dim in args.gradient_bits:
                generator = torch.Generator(device=device).manual_seed(
                    700_000
                    + seed * 100_000
                    + seq_len * 1_000
                    + symbol_dim * 100
                )
                query = _nonzero_randn(
                    (1, seq_len, 1, symbol_dim),
                    generator=generator,
                    device=device,
                )
                key = _nonzero_randn(
                    query.shape, generator=generator, device=device
                )
                full_value = _nonzero_randn(
                    (1, seq_len, 1, max_value_dim),
                    generator=generator,
                    device=device,
                )
                full_grad_output = torch.randn(
                    1,
                    seq_len,
                    1,
                    max_value_dim,
                    generator=generator,
                    device=device,
                )
                for value_dim in args.gradient_value_bits:
                    value = full_value[..., :value_dim]
                    grad_output = full_grad_output[..., :value_dim]
                    oracle = discrete_bit_flip_oracle(
                        query,
                        key,
                        value,
                        grad_output,
                        max_suffix_length=seq_len,
                    )
                    state = competition_state(
                        query,
                        key,
                        value,
                        scale=args.scale,
                        mismatch_scale=args.mismatch_scale,
                    )
                    for estimator in args.gradient_estimators:
                        leaves = (
                            query.detach().clone().requires_grad_(),
                            key.detach().clone().requires_grad_(),
                        )
                        output = _operator(estimator)(
                            leaves[0],
                            leaves[1],
                            value,
                            scale=args.scale,
                            mismatch_scale=args.mismatch_scale,
                        )
                        gradients = torch.autograd.grad(
                            output, leaves, grad_output
                        )
                        query_direction = -_hard_sign(query) * gradients[0]
                        key_direction = -_hard_sign(key) * gradients[1]
                        combined = summarize_alignment(
                            torch.cat(
                                (
                                    query_direction.flatten(),
                                    key_direction.flatten(),
                                )
                            ),
                            torch.cat(
                                (
                                    oracle.query_loss_deltas.flatten(),
                                    oracle.key_loss_deltas.flatten(),
                                )
                            ),
                            top_k=args.gradient_top_k,
                        )
                        records.append(
                            {
                                "seed": seed,
                                "sequence_length": seq_len,
                                "symbol_dim": symbol_dim,
                                "value_dim": value_dim,
                                "estimator": estimator,
                                "alignment": combined,
                                "mean_top_two_mass": float(
                                    state.top_two_mass.mean()
                                ),
                                "mean_nonnull_top_two_mass": float(
                                    state.nonnull_top_two_mass.mean()
                                ),
                                "mean_runner_up_share_of_rest": float(
                                    state.runner_up_share_of_rest.mean()
                                ),
                                "hard_soft_winner_agreement": float(
                                    state.hard_soft_winner_agreement.float().mean()
                                ),
                                "soft_winner_is_null": float(
                                    state.soft_winner_is_null.float().mean()
                                ),
                                "top_two_value_collision": float(
                                    state.top_two_value_collision.float().mean()
                                ),
                                "top_two_nonnull_value_collision": float(
                                    state.top_two_nonnull_value_collision[
                                        state.has_two_nonnull_routes
                                    ].float().mean()
                                ),
                            }
                        )

    summaries = {}
    for estimator in args.gradient_estimators:
        selected = [
            record for record in records if record["estimator"] == estimator
        ]
        alignments = [record["alignment"] for record in selected]
        comparable = sum(
            int(alignment["sign_comparable_count"])
            for alignment in alignments
        )
        agreements = sum(
            int(alignment["sign_agreement_count"])
            for alignment in alignments
        )
        useful = sum(
            int(alignment["oracle_useful_count"])
            for alignment in alignments
        )
        missed = sum(
            int(alignment["zero_support"]["missed_oracle_useful_count"])
            for alignment in alignments
        )
        summaries[estimator] = {
            "mean_cosine": _mean_optional(
                [alignment["cosine_similarity"] for alignment in alignments]
            ),
            "sign_agreement": agreements / comparable if comparable else None,
            "missed_oracle_useful_fraction": (
                missed / useful if useful else None
            ),
            "mean_top_two_mass": statistics.mean(
                float(record["mean_top_two_mass"]) for record in selected
            ),
            "mean_nonnull_top_two_mass": statistics.mean(
                float(record["mean_nonnull_top_two_mass"])
                for record in selected
            ),
            "mean_runner_up_share_of_rest": statistics.mean(
                float(record["mean_runner_up_share_of_rest"])
                for record in selected
            ),
            "hard_soft_winner_agreement": statistics.mean(
                float(record["hard_soft_winner_agreement"])
                for record in selected
            ),
            "soft_winner_is_null": statistics.mean(
                float(record["soft_winner_is_null"])
                for record in selected
            ),
            "top_two_value_collision": statistics.mean(
                float(record["top_two_value_collision"])
                for record in selected
            ),
            "top_two_nonnull_value_collision": statistics.mean(
                float(record["top_two_nonnull_value_collision"])
                for record in selected
            ),
            "cells": len(selected),
        }
    return {"records": records, "summary": summaries}


def _train_fit(
    initial_model: _ResearchFitLM,
    estimator: str,
    tokens: Tensor,
    target_mask: Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    model = copy.deepcopy(initial_model)
    model.training_operator = _operator(estimator)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.fit_learning_rate,
        weight_decay=args.weight_decay,
    )
    initial_loss, initial_accuracy = _evaluate(model, tokens, target_mask)
    best_loss = initial_loss
    first_success = None
    started = time.perf_counter()
    for step in range(1, args.fit_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_and_accuracy(model(tokens), tokens, target_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        loss_value = float(loss.detach())
        best_loss = min(best_loss, loss_value)
        if first_success is None and loss_value < args.success_threshold:
            first_success = step
    if tokens.is_cuda:
        torch.cuda.synchronize(tokens.device)
    elapsed = time.perf_counter() - started
    final_loss, final_accuracy = _evaluate(model, tokens, target_mask)
    best_loss = min(best_loss, final_loss)
    return {
        "estimator": estimator,
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "best_loss": best_loss,
        "first_success_step": first_success,
        "ever_success": best_loss < args.success_threshold,
        "final_success": final_loss < args.success_threshold,
        "step_ms": elapsed * 1000.0 / max(args.fit_steps, 1),
    }


def run_fit_matrix(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    records = []
    for seed in args.fit_seeds:
        torch.manual_seed(300_000 + seed)
        tokens = make_copy_tokens(
            seq_len=args.fit_sequence_length + 1,
            vocab_size=args.fit_vocab_size,
            motif_min=args.fit_motif_min,
            motif_max=args.fit_motif_max,
            seed=100_000 + seed,
        ).to(device)
        target_mask = historical_target_mask(
            tokens, args.fit_max_suffix_length
        )
        initial_model = _ResearchFitLM(
            vocab_size=args.fit_vocab_size,
            num_heads=args.heads,
            qk_bits=args.qk_bits,
            value_heads=args.value_heads,
            value_bits=args.value_bits,
            max_suffix_length=args.fit_max_suffix_length,
            scale=args.scale,
            dropout_p=0.0,
            mismatch_scale=args.mismatch_scale,
            training_operator=_operator("split_dense"),
        ).to(device)
        for estimator in args.fit_estimators:
            torch.manual_seed(400_000 + seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(400_000 + seed)
            record = _train_fit(
                initial_model,
                estimator,
                tokens,
                target_mask,
                args,
            )
            record["seed"] = seed
            records.append(record)
    return {
        "records": records,
        "summary": _summarize_training(
            records, args.fit_estimators, "first_success_step"
        ),
    }


class _ContextModel(ResetRnnRosaLM):
    def __init__(self, *, estimator: str, **kwargs) -> None:
        super().__init__(operator="reference", **kwargs)
        self.estimator = estimator

    def _routed_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        route_mode: str,
    ) -> Tensor:
        if route_mode != "rosa":
            return super()._routed_values(query, key, value, route_mode)
        return _operator(self.estimator)(
            query,
            key,
            value,
            scale=self.scale,
            mismatch_scale=self.mismatch_scale,
        )


@torch.no_grad()
def _context_competition_metrics(model: _ContextModel, recall_batch) -> dict:
    residual = model.encode_residual(recall_batch.tokens)
    query, key, value = model.project_symbols(residual)
    state = competition_state(
        query,
        key,
        value,
        scale=model.scale,
        mismatch_scale=model.mismatch_scale,
    )
    rows = recall_batch.query_positions
    return {
        "top_two_mass": float(state.top_two_mass[..., rows].mean()),
        "runner_up_share_of_rest": float(
            state.runner_up_share_of_rest[..., rows].mean()
        ),
        "hard_soft_winner_agreement": float(
            state.hard_soft_winner_agreement[..., rows].float().mean()
        ),
        "top_two_value_collision": float(
            state.top_two_value_collision[..., rows].float().mean()
        ),
    }


def run_context_matrix(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    records = []
    for seed in args.context_seeds:
        train_batch = make_contextual_recall_batch(
            seed=100_000 + seed,
            pairs=args.context_train_pairs,
            associations=args.context_associations,
            value_bits=args.value_bits,
        ).to(device)
        validation_batch = make_contextual_recall_batch(
            seed=200_000 + seed,
            pairs=args.context_validation_pairs,
            associations=args.context_associations,
            value_bits=args.value_bits,
        ).to(device)
        torch.manual_seed(300_000 + seed)
        initial_model = _ContextModel(
            estimator="split_dense",
            associations=args.context_associations,
            hidden_size=args.context_hidden_size,
            num_heads=args.heads,
            qk_bits=args.qk_bits,
            value_heads=args.value_heads,
            value_bits=args.value_bits,
            context_scale=args.context_scale,
            scale=args.scale,
            dropout_p=0.0,
            mismatch_scale=args.mismatch_scale,
        ).to(device)
        initial_competition = _context_competition_metrics(
            initial_model, train_batch
        )
        for estimator in args.context_estimators:
            model = copy.deepcopy(initial_model)
            model.estimator = estimator
            torch.manual_seed(400_000 + seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(400_000 + seed)
            training = _train_estimator(
                model,
                train_batch,
                steps=args.context_steps,
                learning_rate=args.context_learning_rate,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
            )
            train_metrics = evaluate_model(model, train_batch)
            validation_metrics = evaluate_model(model, validation_batch)
            records.append(
                {
                    "seed": seed,
                    "estimator": estimator,
                    "training": training,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "passed": _candidate_passed(
                        train_metrics, validation_metrics
                    ),
                    "initial_competition": initial_competition,
                    "final_competition": _context_competition_metrics(
                        model, validation_batch
                    ),
                }
            )
    return {
        "records": records,
        "summary": _summarize_context(records, args.context_estimators),
    }


def _summarize_training(
    records: Sequence[dict[str, object]],
    estimators: Sequence[str],
    first_step_key: str,
) -> dict[str, object]:
    summary = {}
    for estimator in estimators:
        selected = [
            record for record in records if record["estimator"] == estimator
        ]
        first_steps = [
            int(record[first_step_key])
            for record in selected
            if record[first_step_key] is not None
        ]
        summary[estimator] = {
            "ever_successes": sum(bool(record["ever_success"]) for record in selected),
            "final_successes": sum(bool(record["final_success"]) for record in selected),
            "runs": len(selected),
            "median_final_loss": statistics.median(
                float(record["final_loss"]) for record in selected
            ),
            "median_best_loss": statistics.median(
                float(record["best_loss"]) for record in selected
            ),
            "median_first_success_step": (
                statistics.median(first_steps) if first_steps else None
            ),
            "mean_step_ms": statistics.mean(
                float(record["step_ms"]) for record in selected
            ),
        }
    return summary


def _summarize_context(
    records: Sequence[dict[str, object]], estimators: Sequence[str]
) -> dict[str, object]:
    summary = {}
    for estimator in estimators:
        selected = [
            record for record in records if record["estimator"] == estimator
        ]
        first_steps = [
            int(record["training"]["first_train_exact_step"])
            for record in selected
            if record["training"]["first_train_exact_step"] is not None
        ]
        summary[estimator] = {
            "passed": sum(bool(record["passed"]) for record in selected),
            "runs": len(selected),
            "mean_validation_exact": statistics.mean(
                float(record["validation"]["exact_accuracy"])
                for record in selected
            ),
            "minimum_validation_exact": min(
                float(record["validation"]["exact_accuracy"])
                for record in selected
            ),
            "median_first_train_exact_step": (
                statistics.median(first_steps) if first_steps else None
            ),
            "mean_step_ms": statistics.mean(
                float(record["training"]["step_ms"])
                for record in selected
            ),
            "mean_final_top_two_mass": statistics.mean(
                float(record["final_competition"]["top_two_mass"])
                for record in selected
            ),
            "mean_final_runner_share": statistics.mean(
                float(record["final_competition"]["runner_up_share_of_rest"])
                for record in selected
            ),
            "mean_final_hard_soft_agreement": statistics.mean(
                float(
                    record["final_competition"][
                        "hard_soft_winner_agreement"
                    ]
                )
                for record in selected
            ),
            "mean_final_top_two_value_collision": statistics.mean(
                float(
                    record["final_competition"]["top_two_value_collision"]
                )
                for record in selected
            ),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("gradient", "fit", "context"),
        default=["gradient", "fit", "context"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--qk-bits", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--success-threshold", type=float, default=1e-3)

    parser.add_argument(
        "--gradient-estimators",
        nargs="+",
        choices=GRADIENT_ESTIMATORS,
        default=list(GRADIENT_ESTIMATORS),
    )
    parser.add_argument(
        "--gradient-seeds", type=int, nargs="+", default=[0, 1, 2, 3]
    )
    parser.add_argument(
        "--gradient-lengths", type=int, nargs="+", default=[4, 6]
    )
    parser.add_argument(
        "--gradient-bits", type=int, nargs="+", default=[1, 2, 4, 8]
    )
    parser.add_argument(
        "--gradient-value-bits",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--gradient-top-k", type=int, default=8)

    parser.add_argument(
        "--fit-estimators",
        nargs="+",
        choices=ESTIMATORS,
        default=list(ESTIMATORS),
    )
    parser.add_argument(
        "--fit-seeds", type=int, nargs="+", default=list(range(8))
    )
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--fit-sequence-length", type=int, default=16)
    parser.add_argument("--fit-vocab-size", type=int, default=8)
    parser.add_argument("--fit-motif-min", type=int, default=4)
    parser.add_argument("--fit-motif-max", type=int, default=8)
    parser.add_argument("--fit-max-suffix-length", type=int, default=8)
    parser.add_argument("--fit-learning-rate", type=float, default=0.01)

    parser.add_argument(
        "--context-estimators",
        nargs="+",
        choices=ESTIMATORS,
        default=list(ESTIMATORS),
    )
    parser.add_argument(
        "--context-seeds", type=int, nargs="+", default=list(range(8))
    )
    parser.add_argument("--context-steps", type=int, default=500)
    parser.add_argument("--context-train-pairs", type=int, default=64)
    parser.add_argument("--context-validation-pairs", type=int, default=32)
    parser.add_argument("--context-associations", type=int, default=4)
    parser.add_argument("--context-hidden-size", type=int, default=32)
    parser.add_argument("--context-scale", type=float, default=0.25)
    parser.add_argument("--context-learning-rate", type=float, default=0.005)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    sections = {}
    if "gradient" in args.sections:
        sections["gradient"] = run_gradient_matrix(args)
    if "fit" in args.sections:
        sections["fit"] = run_fit_matrix(args)
    if "context" in args.sections:
        sections["context"] = run_context_matrix(args)
    device = torch.device(args.device)
    return {
        "schema_version": 1,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "scale": args.scale,
        "mismatch_scale": args.mismatch_scale,
        "sections": sections,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    printable = (
        {
            name: section["summary"]
            for name, section in report["sections"].items()
        }
        if args.summary_only
        else report
    )
    encoded = json.dumps(printable, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
