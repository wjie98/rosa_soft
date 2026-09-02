"""Profile 32x32 physical block-diagonal VJP schedules."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch

import rosa_soft  # noqa: F401 - registers the hard-forward torch operators
from benchmarks.block_diagonal_vjp import (
    block_diagonal_vjp as block64_vjp,
    load_block_diagonal_vjp,
)
from benchmarks.block_diagonal_tile_vjp import (
    THREAD_VARIANTS,
    block_diagonal_tile32_vjp,
    load_block_diagonal_tile32,
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
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--masks", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument(
        "--threads",
        type=int,
        choices=THREAD_VARIANTS,
        nargs="+",
        default=list(THREAD_VARIANTS),
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-dim", type=int, choices=[32, 64], default=32)
    parser.add_argument("--tile-size", type=int, choices=[32, 64], default=32)
    parser.add_argument("--max-registers", type=int, choices=[128])
    parser.add_argument(
        "--baseline-plan",
        choices=["streaming", "block64"],
        default="streaming",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.baseline_plan == "block64" and args.value_dim != 64:
        parser.error("--baseline-plan block64 requires --value-dim 64")
    modules = {
        threads: load_block_diagonal_tile32(
            threads,
            max_registers=args.max_registers,
            value_dim=args.value_dim,
            tile_size=args.tile_size,
        )
        for threads in args.threads
    }
    block64_module = (
        load_block_diagonal_vjp()
        if args.baseline_plan == "block64"
        else None
    )
    rows = []
    for bits, seq_len in itertools.product(args.bits, args.lengths):
        generator = torch.Generator(device="cuda").manual_seed(
            49000 + bits + seq_len
        )
        query = torch.randn(
            1,
            seq_len,
            args.heads,
            bits,
            device="cuda",
            generator=generator,
        )
        key = torch.randn(query.shape, device="cuda", generator=generator)
        value = torch.randn(
            1,
            seq_len,
            args.value_heads,
            args.value_dim,
            device="cuda",
            generator=generator,
        )
        grad_output = torch.randn(
            1,
            seq_len,
            args.heads,
            args.value_dim,
            device="cuda",
            generator=generator,
        )
        _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
            query, key, value
        )
        seed = torch.empty(0, dtype=torch.int64, device="cuda")
        arguments = (
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
        )
        for mask, threads in itertools.product(args.masks, args.threads):
            module = modules[threads]

            def run(plan: str):
                return block_diagonal_tile32_vjp(
                    *arguments,
                    max_suffix_length=32,
                    scale=2.0,
                    dropout_p=0.0,
                    mismatch_scale=3.0,
                    gradient_mask=mask,
                    threads=threads,
                    max_registers=args.max_registers,
                    compiled_value_dim=args.value_dim,
                    compiled_tile_size=args.tile_size,
                    plan=plan,
                    module=module,
                )

            def run_baseline():
                if block64_module is None:
                    return run("baseline")
                return block64_vjp(
                    *arguments,
                    max_suffix_length=32,
                    scale=2.0,
                    dropout_p=0.0,
                    mismatch_scale=3.0,
                    gradient_mask=mask,
                    plan="block_tf32",
                    module=block64_module,
                )

            reference = run("baseline")
            actual = run("block_tf32")
            candidate_ms, baseline_ms = _measure_pair(
                lambda: run("block_tf32"),
                run_baseline,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            errors = [
                0.0
                if expected.numel() == 0
                else float((candidate - expected).abs().max())
                for candidate, expected in zip(actual, reference)
            ]
            row = {
                "seq_len": seq_len,
                "bits": bits,
                "value_dim": args.value_dim,
                "gradient_mask": mask,
                "tile_rows": args.tile_size,
                "tile_routes": args.tile_size,
                "threads": threads,
                "max_registers": args.max_registers,
                "baseline_ms": baseline_ms,
                "baseline_plan": args.baseline_plan,
                "latency_ms": candidate_ms,
                "latency_ratio": candidate_ms / baseline_ms,
                "max_abs_errors": errors,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "heads": args.heads,
        "value_heads": args.value_heads,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
