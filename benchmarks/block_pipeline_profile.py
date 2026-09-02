"""Profile sequential and warp-specialized block route sweeps."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch

from benchmarks.block_pipeline import block_pipeline, load_block_pipeline


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
    return statistics.median(candidate_samples), statistics.median(
        baseline_samples
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--series", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    module = load_block_pipeline()
    rows = []
    for bits, series, seq_len in itertools.product(
        args.bits, args.series, args.lengths
    ):
        generator = torch.Generator(device="cuda").manual_seed(
            54000 + bits + series + seq_len
        )
        high = 2**31 - 1 if bits == 32 else 1 << bits
        low = -(2**31) if bits == 32 else 0
        query = torch.randint(
            low,
            high,
            (series, seq_len),
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
        grad_output = torch.randn(
            series,
            64,
            64,
            device="cuda",
            generator=generator,
        )
        value = torch.randn(
            series,
            seq_len,
            64,
            device="cuda",
            generator=generator,
        )

        def run(method: str):
            return block_pipeline(
                query,
                key,
                grad_output,
                value,
                symbol_dim=bits,
                method=method,
                module=module,
            )

        expected = run("sequential")
        actual = run("warp_specialized")
        candidate_ms, baseline_ms = _measure_pair(
            lambda: run("warp_specialized"),
            lambda: run("sequential"),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        row = {
            "bits": bits,
            "series": series,
            "seq_len": seq_len,
            "route_tiles": seq_len // 64,
            "baseline_ms": baseline_ms,
            "latency_ms": candidate_ms,
            "latency_ratio": candidate_ms / baseline_ms,
            "max_abs_error": float((actual - expected).abs().max()),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
