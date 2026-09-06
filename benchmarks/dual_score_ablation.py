"""Matched score and exact-bitflip ablation for unbounded ROSA VJPs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from benchmarks.discrete_gradient_alignment import (
    _nonzero_randn,
    discrete_bit_flip_oracle,
    summarize_alignment,
)
from benchmarks.dual_score_reference import (
    DEFAULT_EVIDENCE_POWER,
    rosa_dual_score_reference,
    tempered_diagonal_log_evidence,
)
from benchmarks.suffix_proxy_ablation import rosa_soft_suffix_proxy


ESTIMATORS = ("linear", "sqrt", "collision_lr", "information", "dual")


def _hard_sign(values: Tensor) -> Tensor:
    return torch.where(values > 0, 1.0, -1.0)


def _run_estimator(
    estimator: str,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
    mismatch_scale: float,
    evidence_power: float,
    information_weight: float,
) -> Tensor:
    if estimator == "linear":
        return rosa_soft_suffix_proxy(
            query,
            key,
            value,
            proxy="baseline",
            max_suffix_length=query.size(1),
            scale=scale,
            mismatch_scale=mismatch_scale,
        )
    if estimator == "collision_lr":
        return rosa_soft_suffix_proxy(
            query,
            key,
            value,
            proxy="collision_lr",
            max_suffix_length=query.size(1),
            scale=scale,
            mismatch_scale=mismatch_scale,
        )
    if estimator in ("sqrt", "information", "dual"):
        score_mode = "discovery" if estimator == "sqrt" else estimator
        return rosa_dual_score_reference(
            query,
            key,
            value,
            score_mode=score_mode,
            scale=scale,
            mismatch_scale=mismatch_scale,
            evidence_power=evidence_power,
            information_weight=information_weight,
        )
    raise ValueError(f"estimator must be one of {ESTIMATORS}")


def _surrogate_flip_directions(
    estimator: str,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    loss_weights: Tensor,
    *,
    scale: float,
    mismatch_scale: float,
    evidence_power: float,
    information_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    query_leaf = query.detach().requires_grad_(True)
    key_leaf = key.detach().requires_grad_(True)
    output = _run_estimator(
        estimator,
        query_leaf,
        key_leaf,
        value.detach(),
        scale=scale,
        mismatch_scale=mismatch_scale,
        evidence_power=evidence_power,
        information_weight=information_weight,
    )
    loss = (output * loss_weights).sum()
    grad_query, grad_key = torch.autograd.grad(loss, (query_leaf, key_leaf))
    return (
        loss.detach(),
        (-_hard_sign(query) * grad_query).detach(),
        (-_hard_sign(key) * grad_key).detach(),
    )


def measure_bitflip_cell(
    *,
    estimator: str,
    seed: int,
    seq_len: int,
    symbol_dim: int,
    value_dim: int,
    scale: float,
    mismatch_scale: float,
    top_k: int,
    evidence_power: float = DEFAULT_EVIDENCE_POWER,
    information_weight: float = 0.5,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    query = _nonzero_randn(
        (1, seq_len, 1, symbol_dim), generator=generator
    )
    key = _nonzero_randn(
        (1, seq_len, 1, symbol_dim), generator=generator
    )
    value = _nonzero_randn((1, seq_len, 1, value_dim), generator=generator)
    loss_weights = _nonzero_randn(
        (1, seq_len, 1, value_dim), generator=generator
    )
    oracle = discrete_bit_flip_oracle(
        query,
        key,
        value,
        loss_weights,
        max_suffix_length=seq_len,
    )
    loss, query_direction, key_direction = _surrogate_flip_directions(
        estimator,
        query,
        key,
        value,
        loss_weights,
        scale=scale,
        mismatch_scale=mismatch_scale,
        evidence_power=evidence_power,
        information_weight=information_weight,
    )
    direction = torch.cat((query_direction.flatten(), key_direction.flatten()))
    oracle_delta = torch.cat(
        (
            oracle.query_loss_deltas.flatten(),
            oracle.key_loss_deltas.flatten(),
        )
    )
    return {
        "estimator": estimator,
        "seed": seed,
        "sequence_length": seq_len,
        "symbol_dim": symbol_dim,
        "evidence_power": evidence_power,
        "information_weight": information_weight,
        "hard_loss_difference": float(loss - oracle.base_loss),
        "query": summarize_alignment(
            query_direction, oracle.query_loss_deltas, top_k=top_k
        ),
        "key": summarize_alignment(
            key_direction, oracle.key_loss_deltas, top_k=top_k
        ),
        "combined": summarize_alignment(direction, oracle_delta, top_k=top_k),
    }


def _mean_optional(values) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _aggregate(records, estimator: str) -> dict[str, object]:
    rows = [
        record["combined"]
        for record in records
        if record["estimator"] == estimator
    ]
    top = [row["top_k"] for row in rows]
    oracle_useful = sum(int(row["oracle_useful_count"]) for row in rows)
    missed_useful = sum(
        int(row["zero_support"]["missed_oracle_useful_count"])
        for row in rows
    )
    top_count = sum(int(row["evaluated_recommendations"]) for row in top)
    top_useful = sum(int(row["oracle_useful_count"]) for row in top)
    return {
        "cells": len(rows),
        "mean_cosine_similarity": _mean_optional(
            row["cosine_similarity"] for row in rows
        ),
        "mean_sign_agreement": _mean_optional(
            row["sign_agreement"] for row in rows
        ),
        "missed_oracle_useful_fraction": (
            missed_useful / oracle_useful if oracle_useful else None
        ),
        "top_k_oracle_useful_fraction": (
            top_useful / top_count if top_count else None
        ),
    }


def exact_score_capacity(
    *,
    symbol_dims: Sequence[int],
    candidate_counts: Sequence[int],
    suffix_lengths: Sequence[int],
    mismatch_scale: float,
    scale: float,
    evidence_power: float = DEFAULT_EVIDENCE_POWER,
) -> list[dict[str, object]]:
    rows = []
    sqrt_scale = math.sqrt(2.0) + 1.0
    for bits, candidates, length in itertools.product(
        symbol_dims, candidate_counts, suffix_lengths
    ):
        sqrt_score = scale * sqrt_scale * (math.sqrt(1.0 + length) - 1.0)
        information_score = float(
            tempered_diagonal_log_evidence(
                torch.zeros(length, dtype=torch.float64),
                symbol_dim=bits,
                mismatch_scale=mismatch_scale,
                evidence_power=evidence_power,
            )
        )
        ideal_hard_information = length * bits * math.log(2.0)
        prior = math.log(candidates)
        rows.append(
            {
                "symbol_dim": bits,
                "candidate_count": candidates,
                "suffix_length": length,
                "evidence_power": evidence_power,
                "sqrt_logit_after_prior": sqrt_score - prior,
                "information_logit_after_prior": information_score - prior,
                "ideal_hard_logit_after_prior": ideal_hard_information - prior,
            }
        )
    return rows


def run_ablation(args: argparse.Namespace) -> dict[str, object]:
    records = []
    for estimator, seed, seq_len, bits in itertools.product(
        args.estimators,
        args.seeds,
        args.sequence_lengths,
        args.symbol_dims,
    ):
        records.append(
            measure_bitflip_cell(
                estimator=estimator,
                seed=seed,
                seq_len=seq_len,
                symbol_dim=bits,
                value_dim=args.value_dim,
                scale=args.scale,
                mismatch_scale=args.mismatch_scale,
                top_k=args.top_k,
                evidence_power=args.evidence_power,
                information_weight=args.information_weight,
            )
        )
    dual_grid = {}
    for evidence_power, information_weight in itertools.product(
        args.evidence_powers,
        args.information_weights,
    ):
        label = f"beta={evidence_power:g},rho={information_weight:g}"
        grid_records = []
        for seed, seq_len, bits in itertools.product(
            args.seeds,
            args.sequence_lengths,
            args.symbol_dims,
        ):
            grid_records.append(
                measure_bitflip_cell(
                    estimator="dual",
                    seed=seed,
                    seq_len=seq_len,
                    symbol_dim=bits,
                    value_dim=args.value_dim,
                    scale=args.scale,
                    mismatch_scale=args.mismatch_scale,
                    top_k=args.top_k,
                    evidence_power=evidence_power,
                    information_weight=information_weight,
                )
            )
        dual_grid[label] = _aggregate(grid_records, "dual")
    return {
        "schema_version": 1,
        "scope": "unbounded dense VJP score geometry; identical hard forward",
        "estimators": list(args.estimators),
        "matrix": {
            "seeds": list(args.seeds),
            "sequence_lengths": list(args.sequence_lengths),
            "symbol_dims": list(args.symbol_dims),
            "scale": args.scale,
            "mismatch_scale": args.mismatch_scale,
            "evidence_power": args.evidence_power,
            "information_weight": args.information_weight,
        },
        "summary": {
            estimator: _aggregate(records, estimator)
            for estimator in args.estimators
        },
        "dual_grid": dual_grid,
        "capacity": exact_score_capacity(
            symbol_dims=args.capacity_symbol_dims,
            candidate_counts=args.candidate_counts,
            suffix_lengths=args.suffix_lengths,
            mismatch_scale=args.mismatch_scale,
            scale=args.scale,
            evidence_power=args.evidence_power,
        ),
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimators", nargs="+", choices=ESTIMATORS, default=ESTIMATORS
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--symbol-dims", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--value-dim", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--evidence-power", type=float, default=DEFAULT_EVIDENCE_POWER
    )
    parser.add_argument("--information-weight", type=float, default=0.5)
    parser.add_argument(
        "--evidence-powers",
        nargs="+",
        type=float,
        default=[DEFAULT_EVIDENCE_POWER],
    )
    parser.add_argument(
        "--information-weights", nargs="+", type=float, default=[0.5]
    )
    parser.add_argument(
        "--capacity-symbol-dims", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument(
        "--candidate-counts", nargs="+", type=int, default=[8192, 1048576, 100000000]
    )
    parser.add_argument(
        "--suffix-lengths", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_ablation(args)
    selected = report["summary"] if args.summary_only else report
    encoded = json.dumps(selected, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
