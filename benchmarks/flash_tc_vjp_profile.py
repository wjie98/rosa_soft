"""Validate and profile two-pass FlashROSA Tensor-Core research plans."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

import rosa_soft
from benchmarks.flash_tc_vjp import PLANS, flash_tc_vjp, load_flash_tc_vjp


def _measure_paired_ms(
    candidate: Callable[[], object],
    baseline: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    """Interleave plans so boost-clock drift does not become a speedup."""

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
        if repeat % 2 == 0:
            baseline_samples.append(measure(baseline))
            candidate_samples.append(measure(candidate))
        else:
            candidate_samples.append(measure(candidate))
            baseline_samples.append(measure(baseline))
    return statistics.median(candidate_samples), statistics.median(
        baseline_samples
    )


def _peak_extra_mib(operation: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024 * 1024)


def _make_case(
    *,
    seq_len: int,
    heads: int,
    value_heads: int,
    bits: int,
    value_dim: int,
    window: int,
    dtype: torch.dtype,
    dropout_p: float,
    pattern: str,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(
        21000 + seq_len + 31 * window + bits + value_dim
    )
    query = torch.randn(
        1,
        seq_len,
        heads,
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
        value_heads,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        1,
        seq_len,
        heads,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = (
        torch.tensor(123456789, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    return (
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        window,
        2.0,
        dropout_p,
        3.0,
    )


def _errors(actual, expected) -> tuple[list[float], list[float]]:
    maximum = []
    relative = []
    for candidate, reference in zip(actual, expected):
        if reference.numel() == 0:
            maximum.append(0.0)
            relative.append(0.0)
            continue
        difference = candidate - reference
        maximum.append(float(difference.abs().max()))
        relative.append(
            float(
                difference.norm()
                / reference.norm().clamp_min(torch.finfo(torch.float32).tiny)
            )
        )
    return maximum, relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--windows", type=int, nargs="+", default=[32])
    parser.add_argument("--masks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--plans", nargs="+", choices=PLANS, default=list(PLANS))
    parser.add_argument("--patterns", nargs="+", default=["random"])
    parser.add_argument("--bits", type=int, nargs="+", default=[8])
    parser.add_argument("--value-dims", type=int, nargs="+", default=[64])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    module = load_flash_tc_vjp()
    rows = []

    for pattern in args.patterns:
        for bits in args.bits:
            for value_dim in args.value_dims:
                for window in args.windows:
                    for seq_len in args.lengths:
                        arguments = _make_case(
                            seq_len=seq_len,
                            heads=args.heads,
                            value_heads=args.value_heads,
                            bits=bits,
                            value_dim=value_dim,
                            window=min(window, seq_len),
                            dtype=dtype,
                            dropout_p=args.dropout_p,
                            pattern=pattern,
                        )
                        for mask in args.masks:
                            baseline = lambda: flash_tc_vjp(
                                *arguments[:7],
                                max_suffix_length=arguments[7],
                                scale=arguments[8],
                                dropout_p=arguments[9],
                                mismatch_scale=arguments[10],
                                gradient_mask=mask,
                                plan="baseline",
                                module=module,
                            )
                            expected = baseline()
                            for plan in args.plans:
                                operation = lambda plan=plan: flash_tc_vjp(
                                    *arguments[:7],
                                    max_suffix_length=arguments[7],
                                    scale=arguments[8],
                                    dropout_p=arguments[9],
                                    mismatch_scale=arguments[10],
                                    gradient_mask=mask,
                                    plan=plan,
                                    module=module,
                                )
                                actual = operation()
                                maximum, relative = _errors(actual, expected)
                                latency_ms, baseline_ms = _measure_paired_ms(
                                    operation,
                                    baseline,
                                    warmup=args.warmup,
                                    repeats=args.repeats,
                                )
                                row = {
                                    "pattern": pattern,
                                    "seq_len": seq_len,
                                    "window": min(window, seq_len),
                                    "bits": bits,
                                    "value_dim": value_dim,
                                    "gradient_mask": mask,
                                    "plan": plan,
                                    "baseline_ms": baseline_ms,
                                    "latency_ms": latency_ms,
                                    "latency_ratio": latency_ms / baseline_ms,
                                    "extra_mib": _peak_extra_mib(operation),
                                    "max_abs_errors": maximum,
                                    "relative_l2_errors": relative,
                                }
                                rows.append(row)
                                print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "heads": args.heads,
        "value_heads": args.value_heads,
        "dropout_p": args.dropout_p,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
