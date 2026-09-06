"""Profile the exact persistent block-wavefront RosaSoft VJP control."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

import rosa_soft  # noqa: F401 - registers production operators
from benchmarks.persistent_wavefront_vjp import (
    load_persistent_wavefront_vjp,
    persistent_wavefront_stats,
    wavefront_scores,
    wavefront_stats,
    wavefront_vjp,
)


Operation = Callable[[], object]


def _measure_ms(
    operations: dict[str, Operation],
    *,
    warmup: int,
    repeats: int,
    rounds: int,
) -> dict[str, float]:
    samples = {name: [] for name in operations}
    names = list(operations)
    for round_index in range(rounds):
        order = names[round_index % len(names) :] + names[: round_index % len(names)]
        if round_index & 1:
            order.reverse()
        for name in order:
            operation = operations[name]
            for _ in range(warmup):
                operation()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                operation()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / repeats)
    return {name: statistics.median(values) for name, values in samples.items()}


def _peak_mib(operation: Operation) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024 * 1024)


def _max_errors(
    actual: tuple[Tensor, ...],
    expected: tuple[Tensor, ...],
) -> list[float]:
    return [
        0.0
        if reference.numel() == 0
        else float((candidate - reference).abs().max())
        for candidate, reference in zip(actual, expected)
    ]


def _profile_case(
    seq_len: int,
    args: argparse.Namespace,
    module: object,
) -> dict[str, object]:
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + seq_len)
    query = torch.randn(
        args.batch,
        seq_len,
        args.heads,
        args.bits,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(query.shape, dtype=dtype, device="cuda", generator=generator)
    value = torch.randn(
        args.batch,
        seq_len,
        args.value_heads,
        args.value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        args.batch,
        seq_len,
        args.heads,
        args.value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(query, key, value)
    seed = (
        torch.tensor(args.seed, dtype=torch.int64, device="cuda")
        if args.dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    common = dict(
        max_suffix_length=min(args.window, seq_len),
        scale=args.scale,
        dropout_p=args.dropout_p,
        mismatch_scale=args.mismatch_scale,
        gradient_mask=args.gradient_mask,
        module=module,
    )

    def production() -> tuple[Tensor, Tensor, Tensor]:
        return torch.ops.rosa_soft.surrogate_vjp_masked(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            common["max_suffix_length"],
            common["scale"],
            common["dropout_p"],
            common["mismatch_scale"],
            common["gradient_mask"],
        )

    def multilaunch() -> tuple[Tensor, Tensor, Tensor]:
        return wavefront_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            plan="multilaunch",
            **common,
        )

    def persistent() -> tuple[Tensor, Tensor, Tensor]:
        return wavefront_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            plan="persistent",
            **common,
        )

    stats_common = dict(
        symbol_dim=args.bits,
        max_suffix_length=common["max_suffix_length"],
        scale=args.scale,
        dropout_p=args.dropout_p,
        mismatch_scale=args.mismatch_scale,
        module=module,
    )

    def multilaunch_stats() -> Tensor:
        return wavefront_stats(
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            **stats_common,
        )

    def persistent_stats() -> Tensor:
        return persistent_wavefront_stats(
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            **stats_common,
        )

    def multilaunch_stats_scalar() -> Tensor:
        return wavefront_stats(
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            utility_plan="scalar",
            **stats_common,
        )

    def persistent_stats_scalar() -> Tensor:
        return persistent_wavefront_stats(
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            utility_plan="scalar",
            **stats_common,
        )

    operations: dict[str, Operation] = {
        "production_vjp": production,
        "wavefront_multilaunch_vjp": multilaunch,
        "wavefront_persistent_vjp": persistent,
        "wavefront_multilaunch_stats": multilaunch_stats,
        "wavefront_persistent_stats": persistent_stats,
        "wavefront_multilaunch_stats_scalar": multilaunch_stats_scalar,
        "wavefront_persistent_stats_scalar": persistent_stats_scalar,
    }
    expected = production()
    results = {
        "multilaunch_max_abs_errors": _max_errors(multilaunch(), expected),
        "persistent_max_abs_errors": _max_errors(persistent(), expected),
    }
    latency = _measure_ms(
        operations,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    production_ms = latency["production_vjp"]
    results["operators"] = {
        name: {
            "latency_ms": latency[name],
            "production_ratio": latency[name] / production_ms,
            "peak_operator_mib": _peak_mib(operation),
        }
        for name, operation in operations.items()
    }

    if args.score_debug:
        score_operations = {
            plan: (
                lambda plan=plan: wavefront_scores(
                    packed_query,
                    packed_key,
                    symbol_dim=args.bits,
                    max_suffix_length=common["max_suffix_length"],
                    mismatch_scale=args.mismatch_scale,
                    plan=plan,
                    module=module,
                )
            )
            for plan in ("sequential", "affine")
        }
        score_latency = _measure_ms(
            score_operations,
            warmup=args.warmup,
            repeats=args.repeats,
            rounds=args.rounds,
        )
        results["score_debug_ms"] = score_latency
        results["score_affine_ratio"] = (
            score_latency["affine"] / score_latency["sequential"]
        )
    return {"seq_len": seq_len, **results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--gradient-mask", type=int, default=7)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23091)
    parser.add_argument("--score-debug", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    module = load_persistent_wavefront_vjp()
    rows = []
    for seq_len in args.lengths:
        row = _profile_case(seq_len, args, module)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
