"""Validate and time the exact tiled-streaming RosaSoft VJP."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft
from benchmarks.streaming_vjp import load_streaming_vjp


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _measure_ms(
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
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _peak_extra_mib(operation: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024 * 1024)


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
        17000 + seq_len + 31 * window
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


def _max_errors(actual, expected) -> list[float]:
    return [
        0.0
        if reference.numel() == 0
        else float((candidate - reference).abs().max())
        for candidate, reference in zip(actual, expected)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--tiles", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--masks", type=int, nargs="+", default=[7])
    parser.add_argument("--patterns", nargs="+", default=["random", "all_match"])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    module = load_streaming_vjp()
    rows = []
    for pattern in args.patterns:
        for seq_len in args.lengths:
            arguments = _make_case(
                seq_len=seq_len,
                heads=args.heads,
                value_heads=args.value_heads,
                bits=args.bits,
                value_dim=args.value_dim,
                window=min(args.window, seq_len),
                dtype=_dtype(args.dtype),
                dropout_p=args.dropout_p,
                pattern=pattern,
            )
            for mask in args.masks:
                production = lambda: torch.ops.rosa_soft.surrogate_vjp_masked(
                    *arguments, mask
                )
                expected = production()
                production_ms = _measure_ms(
                    production, warmup=args.warmup, repeats=args.repeats
                )
                production_memory = _peak_extra_mib(production)
                for tile in args.tiles:
                    candidate = lambda tile=tile: module.streaming_vjp(
                        *arguments, mask, tile
                    )
                    actual = candidate()
                    row = {
                        "pattern": pattern,
                        "seq_len": seq_len,
                        "window": min(args.window, seq_len),
                        "gradient_mask": mask,
                        "query_tile_size": tile,
                        "production_ms": production_ms,
                        "streaming_ms": _measure_ms(
                            candidate,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        ),
                        "production_extra_mib": production_memory,
                        "streaming_extra_mib": _peak_extra_mib(candidate),
                        "max_abs_errors": _max_errors(actual, expected),
                    }
                    row["latency_ratio"] = (
                        row["streaming_ms"] / row["production_ms"]
                    )
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dtype": args.dtype,
        "heads": args.heads,
        "value_heads": args.value_heads,
        "bits": args.bits,
        "value_dim": args.value_dim,
        "dropout_p": args.dropout_p,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
