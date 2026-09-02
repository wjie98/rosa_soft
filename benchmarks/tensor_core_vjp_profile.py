"""Profile isolated Tensor-Core candidates for the RosaSoft streaming VJP."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.tensor_core_vjp import (
    GATE_METHODS,
    MATMUL_METHODS,
    batched_matmul,
    gate_matrix,
    load_tensor_core_vjp,
    suffix_scores,
)


GATE_SHARED_BYTES = {
    "scalar": 0,
    "fp16": 2048,
    "int8": 1536,
    "b1": 512,
}

MATMUL_SHARED_BYTES = {
    "scalar": 0,
    "fp16": 2048,
    "bfloat16": 2048,
    "tf32": 2048,
    "fp16_scaled": 2048,
    "fp16_hilo": 3072,
}


def _measure_us(
    operation: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> float:
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
        samples.append(1000.0 * start.elapsed_time(end))
    return statistics.median(samples)


def _kernel_us(
    operation: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> float:
    """Measure device execution only, excluding Python and dispatcher gaps."""

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as trace:
        for _ in range(repeats):
            operation()
    torch.cuda.synchronize()
    device_total = sum(
        event.self_device_time_total for event in trace.key_averages()
    )
    return device_total / repeats


def _codes(count: int, bits: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if bits == 32:
        return torch.randint(
            -(2**31),
            2**31,
            (count,),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
    return torch.randint(
        0,
        1 << bits,
        (count,),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual - expected
    denominator = expected.norm().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs_error": float(difference.abs().max()),
        "relative_l2_error": float(difference.norm() / denominator),
    }


def _record(rows: list[dict], row: dict) -> None:
    rows.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--profile-repeats", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    module = load_tensor_core_vjp()
    generator = torch.Generator(device="cuda").manual_seed(17001)
    rows: list[dict] = []

    for bits in [1, 2, 4, 8, 16, 32]:
        for size in [32, 63, 64, 128, 256, 512]:
            query = _codes(size, bits, seed=1000 + bits + size)
            key = _codes(size, bits, seed=2000 + bits + size)
            expected = gate_matrix(
                query,
                key,
                symbol_bits=bits,
                method="scalar",
                module=module,
            )
            baseline_us = None
            baseline_kernel_us = None
            for method in GATE_METHODS:
                operation = lambda method=method: gate_matrix(
                    query,
                    key,
                    symbol_bits=bits,
                    method=method,
                    module=module,
                )
                actual = operation()
                latency_us = _measure_us(
                    operation,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                kernel_us = _kernel_us(
                    operation,
                    warmup=args.warmup,
                    repeats=args.profile_repeats,
                )
                if method == "scalar":
                    baseline_us = latency_us
                    baseline_kernel_us = kernel_us
                _record(
                    rows,
                    {
                        "study": "gate",
                        "method": method,
                        "bits": bits,
                        "rows": size,
                        "columns": size,
                        "latency_us": latency_us,
                        "kernel_us": kernel_us,
                        "latency_ratio": latency_us / baseline_us,
                        "kernel_ratio": kernel_us / baseline_kernel_us,
                        "static_shared_bytes": GATE_SHARED_BYTES[method],
                        **_error(actual, expected),
                    },
                )

    contraction_cases = [
        ("utility_dv64", 256, 32, 64, 32),
        ("utility_dv128", 256, 32, 128, 32),
        ("value_gradient_dv64", 256, 32, 32, 64),
        ("value_gradient_dv128", 256, 32, 32, 128),
        ("gate_credit_d8", 256, 32, 63, 8),
        ("gate_credit_d32", 256, 32, 63, 32),
    ]
    for name, batch, rows_count, inner, columns in contraction_cases:
        left = torch.randn(
            batch,
            rows_count,
            inner,
            generator=generator,
            device="cuda",
        )
        right = torch.randn(
            batch,
            inner,
            columns,
            generator=generator,
            device="cuda",
        )
        expected = torch.bmm(left, right)
        baseline_us = None
        baseline_kernel_us = None
        for method in MATMUL_METHODS:
            operation = lambda method=method: batched_matmul(
                left,
                right,
                method=method,
                module=module,
            )
            actual = operation()
            latency_us = _measure_us(
                operation,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            kernel_us = _kernel_us(
                operation,
                warmup=args.warmup,
                repeats=args.profile_repeats,
            )
            if method == "scalar":
                baseline_us = latency_us
                baseline_kernel_us = kernel_us
            _record(
                rows,
                {
                    "study": "contraction",
                    "case": name,
                    "method": method,
                    "batch": batch,
                    "rows": rows_count,
                    "inner": inner,
                    "columns": columns,
                    "latency_us": latency_us,
                    "kernel_us": kernel_us,
                    "latency_ratio": latency_us / baseline_us,
                    "kernel_ratio": kernel_us / baseline_kernel_us,
                    "static_shared_bytes": MATMUL_SHARED_BYTES[method],
                    **_error(actual, expected),
                },
            )

    for window in [1, 2, 4, 8, 16, 31, 32]:
        for length in [32, 64, 128, 256, 1024]:
            gates = 0.1 + 0.8 * torch.rand(
                4096,
                length,
                generator=generator,
                device="cuda",
            )
            expected = suffix_scores(
                gates,
                max_suffix_length=window,
                method="direct",
                module=module,
            )
            baseline_us = None
            baseline_kernel_us = None
            for method in ["direct", "warp"]:
                operation = lambda method=method: suffix_scores(
                    gates,
                    max_suffix_length=window,
                    method=method,
                    module=module,
                )
                actual = operation()
                latency_us = _measure_us(
                    operation,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                kernel_us = _kernel_us(
                    operation,
                    warmup=args.warmup,
                    repeats=args.profile_repeats,
                )
                if method == "direct":
                    baseline_us = latency_us
                    baseline_kernel_us = kernel_us
                _record(
                    rows,
                    {
                        "study": "suffix",
                        "method": method,
                        "sequences": gates.size(0),
                        "length": length,
                        "window": window,
                        "latency_us": latency_us,
                        "kernel_us": kernel_us,
                        "latency_ratio": latency_us / baseline_us,
                        "kernel_ratio": kernel_us / baseline_kernel_us,
                        **_error(actual, expected),
                    },
                )

    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile_repeats": args.profile_repeats,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
