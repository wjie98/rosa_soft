"""Profile the exact block-diagonal RosaSoft VJP against streaming."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch

import rosa_soft
from benchmarks.block_diagonal_vjp import (
    block_diagonal_vjp,
    load_block_diagonal_vjp,
)


def _measure_pair(candidate, baseline, *, warmup, repeats):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--value-dims", type=int, nargs="+", default=[64])
    parser.add_argument("--masks", type=int, nargs="+", default=[3, 4, 7])
    parser.add_argument(
        "--dtypes",
        choices=["float32", "float16", "bfloat16"],
        nargs="+",
        default=["float32"],
    )
    parser.add_argument("--dropout-ps", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--patterns",
        choices=["random", "all_match"],
        nargs="+",
        default=["random"],
    )
    parser.add_argument(
        "--plan",
        choices=[
            "block_diagonal",
            "block_tf32",
            "block_tf32_pipeline",
        ],
        default="block_tf32",
    )
    parser.add_argument(
        "--baseline-plan",
        choices=["baseline", "block_diagonal", "block_tf32"],
        default="baseline",
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    module = load_block_diagonal_vjp()
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    rows = []
    cases = itertools.product(
        args.bits,
        args.value_dims,
        args.lengths,
        args.dtypes,
        args.dropout_ps,
        args.patterns,
    )
    for bits, value_dim, seq_len, dtype_name, dropout_p, pattern in cases:
        dtype = dtypes[dtype_name]
        generator = torch.Generator(device="cuda").manual_seed(
            16000 + bits + value_dim + seq_len
        )
        query = torch.randn(
            1,
            seq_len,
            args.heads,
            bits,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        key = torch.randn(
            query.shape,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        if pattern == "all_match":
            query.fill_(1)
            key.fill_(1)
        value = torch.randn(
            1,
            seq_len,
            args.value_heads,
            value_dim,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        grad_output = torch.randn(
            1,
            seq_len,
            args.heads,
            value_dim,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
            query, key, value
        )
        seed = (
            torch.tensor(987654321, dtype=torch.int64, device="cuda")
            if dropout_p
            else torch.empty(0, dtype=torch.int64, device="cuda")
        )
        arguments = (
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
        )
        for mask in args.masks:
            def run(plan):
                return block_diagonal_vjp(
                    *arguments,
                    max_suffix_length=32,
                    scale=2.0,
                    dropout_p=dropout_p,
                    mismatch_scale=3.0,
                    gradient_mask=mask,
                    plan=plan,
                    module=module,
                )

            expected = run("baseline")
            actual = run(args.plan)
            candidate_ms, baseline_ms = _measure_pair(
                lambda: run(args.plan),
                lambda: run(args.baseline_plan),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            maximum = [
                0.0
                if reference.numel() == 0
                else float((candidate - reference).abs().max())
                for candidate, reference in zip(actual, expected)
            ]
            row = {
                "seq_len": seq_len,
                "bits": bits,
                "value_dim": value_dim,
                "dtype": dtype_name,
                "dropout_p": dropout_p,
                "pattern": pattern,
                "gradient_mask": mask,
                "plan": args.plan,
                "baseline_plan": args.baseline_plan,
                "baseline_ms": baseline_ms,
                "latency_ms": candidate_ms,
                "latency_ratio": candidate_ms / baseline_ms,
                "max_abs_errors": maximum,
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
