"""Profile exact suffix ownership when few causal queries remain active."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch

from benchmarks.block_suffix_scan import (
    TAIL_METHODS,
    block_suffix_tail_scores,
    load_block_suffix_scan,
)


def _measure_ms(operation, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-queries",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--series", type=int, nargs="+", default=[4, 64, 256])
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--route-start", type=int, default=128)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument(
        "--methods",
        choices=TAIL_METHODS,
        nargs="+",
        default=list(TAIL_METHODS),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    module = load_block_suffix_scan()
    rows = []
    cases = itertools.product(args.bits, args.series, args.active_queries)
    for bits, series, active_queries in cases:
        generator = torch.Generator(device="cuda").manual_seed(
            45000 + bits + series + active_queries
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
            return block_suffix_tail_scores(
                query,
                key,
                symbol_dim=bits,
                max_suffix_length=args.window,
                route_start=args.route_start,
                active_queries=active_queries,
                method=method,
                module=module,
            )

        reference = run("tail_thread")
        timings = {}
        for method in args.methods:
            actual = run(method)
            error = float((actual - reference).abs().max())
            latency = _measure_ms(
                lambda method=method: run(method),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            timings[method] = latency
            row = {
                "bits": bits,
                "series": series,
                "active_queries": active_queries,
                "active_candidates": active_queries * (active_queries + 1) // 2,
                "window": args.window,
                "method": method,
                "latency_ms": latency,
                "max_abs_error": error,
            }
            rows.append(row)
        physical = timings["physical_block64"]
        for row in rows[-len(args.methods):]:
            row["physical_block64_ratio"] = row["latency_ms"] / physical
            print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sequence_length": args.sequence_length,
        "route_start": args.route_start,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
