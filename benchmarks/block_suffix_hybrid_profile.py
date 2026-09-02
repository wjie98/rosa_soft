"""Profile switching a full causal tile to a compact suffix-tail owner."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch

from benchmarks.block_suffix_scan import (
    HYBRID_METHODS,
    block_suffix_hybrid_scores,
    load_block_suffix_scan,
)


def _measure_pair(candidate, baseline, *, warmup: int, repeats: int):
    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()
    candidate_samples = []
    baseline_samples = []

    def measure(operation):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for repeat in range(repeats):
        if repeat % 2:
            candidate_samples.append(measure(candidate))
            baseline_samples.append(measure(baseline))
        else:
            baseline_samples.append(measure(baseline))
            candidate_samples.append(measure(candidate))
    return (
        statistics.median(candidate_samples),
        statistics.median(baseline_samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tail-queries",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--series", type=int, nargs="+", default=[4096])
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--tile-start", type=int, default=128)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument(
        "--methods",
        choices=HYBRID_METHODS,
        nargs="+",
        default=list(HYBRID_METHODS),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    module = load_block_suffix_scan()
    rows = []
    cases = itertools.product(args.bits, args.series, args.tail_queries)
    for bits, series, tail_queries in cases:
        generator = torch.Generator(device="cuda").manual_seed(
            47000 + bits + series + tail_queries
        )
        high = 2**31 - 1 if bits == 32 else 1 << bits
        low = -(2**31) if bits == 32 else 0
        query = torch.randint(
            low,
            high,
            (1, series, args.sequence_length),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
        key = torch.randint(
            low,
            high,
            query.shape,
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )

        def run(method: str):
            return block_suffix_hybrid_scores(
                query,
                key,
                symbol_dim=bits,
                max_suffix_length=args.window,
                tile_start=args.tile_start,
                tail_queries=tail_queries,
                method=method,
                module=module,
            )

        reference = run("full_diagonal_thread")
        case_rows = []
        for method in args.methods:
            actual = run(method)
            error = float((actual - reference).abs().max())
            latency, paired_baseline = _measure_pair(
                lambda method=method: run(method),
                lambda: run("full_diagonal_thread"),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            case_rows.append(
                {
                    "bits": bits,
                    "series": series,
                    "tail_queries": tail_queries,
                    "tail_candidates": tail_queries * (tail_queries + 1) // 2,
                    "window": args.window,
                    "method": method,
                    "latency_ms": latency,
                    "paired_baseline_ms": paired_baseline,
                    "max_abs_error": error,
                }
            )
        for row in case_rows:
            row["full_diagonal_thread_ratio"] = (
                row["latency_ms"] / row["paired_baseline_ms"]
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sequence_length": args.sequence_length,
        "tile_start": args.tile_start,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
