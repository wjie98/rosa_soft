"""Measure random-background null calibration of unbounded ROSA scores."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.dual_score_reference import (  # noqa: E402
    DEFAULT_EVIDENCE_POWER,
    DEFAULT_INFORMATION_WEIGHT,
    SCORE_MODES,
    final_query_score_carrier,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_condition(
    *,
    candidate_count: int,
    symbol_dim: int,
    suffix_length: int,
    probes: int,
    seed: int,
    score_mode: str,
    evidence_power: float,
    information_weight: float,
    device: torch.device,
) -> dict[str, object]:
    if candidate_count < 1 or suffix_length < 1 or probes < 1:
        raise ValueError("candidate_count, suffix_length, and probes must be positive")
    if symbol_dim < 1 or symbol_dim > 32:
        raise ValueError("symbol_dim must be in [1, 32]")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    query = torch.randn(
        probes,
        suffix_length,
        symbol_dim,
        generator=generator,
        device=device,
    )
    key = torch.randn(
        probes,
        candidate_count,
        symbol_dim,
        generator=generator,
        device=device,
    )
    value = torch.ones(probes, candidate_count, 1, device=device)

    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        _, route_probabilities = final_query_score_carrier(
            query,
            key,
            value,
            score_mode=score_mode,
            evidence_power=evidence_power,
            information_weight=information_weight,
        )
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    nonnull_mass = route_probabilities.sum(dim=-1).cpu()
    return {
        "candidate_count": candidate_count,
        "symbol_dim": symbol_dim,
        "suffix_length": suffix_length,
        "probes": probes,
        "seed": seed,
        "score_mode": score_mode,
        "evidence_power": evidence_power,
        "information_weight": information_weight,
        "mean_nonnull_mass": float(nonnull_mass.mean()),
        "minimum_nonnull_mass": float(nonnull_mass.min()),
        "maximum_nonnull_mass": float(nonnull_mass.max()),
        "elapsed_ms": elapsed_ms,
    }


def _aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, str], list[float]] = {}
    for record in records:
        key = (
            int(record["candidate_count"]),
            int(record["symbol_dim"]),
            str(record["score_mode"]),
        )
        groups.setdefault(key, []).append(float(record["mean_nonnull_mass"]))
    return [
        {
            "candidate_count": candidate_count,
            "symbol_dim": symbol_dim,
            "score_mode": score_mode,
            "runs": len(values),
            "mean_nonnull_mass": statistics.fmean(values),
            "minimum_run_mean": min(values),
            "maximum_run_mean": max(values),
        }
        for (candidate_count, symbol_dim, score_mode), values in sorted(groups.items())
    ]


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not math.isfinite(args.evidence_power) or not 0.0 < args.evidence_power <= 1.0:
        raise ValueError("evidence_power must be finite and in (0, 1]")
    if not math.isfinite(args.information_weight) or not (
        0.0 <= args.information_weight <= 1.0
    ):
        raise ValueError("information_weight must be finite and in [0, 1]")
    device = torch.device(args.device)
    records = [
        measure_condition(
            candidate_count=candidate_count,
            symbol_dim=symbol_dim,
            suffix_length=args.suffix_length,
            probes=args.probes,
            seed=seed,
            score_mode=score_mode,
            evidence_power=args.evidence_power,
            information_weight=args.information_weight,
            device=device,
        )
        for candidate_count in args.candidate_counts
        for symbol_dim in args.symbol_dims
        for seed in args.seeds
        for score_mode in args.score_modes
    ]
    return {
        "schema_version": 1,
        "scope": "random-background final-query null calibration",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "candidate_counts": args.candidate_counts,
        "symbol_dims": args.symbol_dims,
        "suffix_length": args.suffix_length,
        "probes": args.probes,
        "seeds": args.seeds,
        "score_modes": args.score_modes,
        "evidence_power": args.evidence_power,
        "information_weight": args.information_weight,
        "summary": _aggregate(records),
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--candidate-counts", nargs="+", type=int, default=[8192]
    )
    parser.add_argument("--symbol-dims", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--suffix-length", type=int, default=32)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--score-modes", nargs="+", choices=SCORE_MODES, default=list(SCORE_MODES)
    )
    parser.add_argument(
        "--evidence-power", type=float, default=DEFAULT_EVIDENCE_POWER
    )
    parser.add_argument(
        "--information-weight", type=float, default=DEFAULT_INFORMATION_WEIGHT
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
