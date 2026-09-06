"""Benchmark exact macro-wavefront and folded-diagonal RosaSoft VJPs."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

from benchmarks.macro_wavefront_vjp import (
    EXECUTION_PLANS,
    TILE_SIZES,
    load_macro_wavefront_vjp,
    macro_wavefront_scores,
    macro_wavefront_stats,
    macro_wavefront_vjp,
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


def _pack_symbols(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.size(-1)
    shifts = torch.arange(bits, dtype=torch.int64, device=tensor.device)
    return (
        ((tensor > 0).to(torch.int64).permute(0, 2, 1, 3) << shifts)
        .sum(-1)
        .to(torch.int32)
    )


def _case(
    seq_len: int,
    args: argparse.Namespace,
    module: object,
    replay_module: object | None,
) -> dict:
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
    key = torch.randn_like(query)
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
    packed_query = _pack_symbols(query)
    packed_key = _pack_symbols(key)
    dropout_seed = (
        torch.tensor(args.seed, dtype=torch.int64, device="cuda")
        if args.dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )

    operations: dict[str, Operation] = {}
    for tile_size in args.tile_sizes:
        for plan in args.plans:
            prefix = f"{plan}_l{tile_size}"
            if args.include_scores:
                operations[f"{prefix}_scores"] = lambda t=tile_size, p=plan: (
                    macro_wavefront_scores(
                        packed_query,
                        packed_key,
                        symbol_dim=args.bits,
                        tile_size=t,
                        mismatch_scale=args.mismatch_scale,
                        plan=p,
                        module=module,
                    )
                )
            operations[f"{prefix}_stats"] = lambda t=tile_size, p=plan: (
                macro_wavefront_stats(
                    value,
                    grad_output,
                    packed_query,
                    packed_key,
                    dropout_seed,
                    symbol_dim=args.bits,
                    tile_size=t,
                    scale=args.scale,
                    dropout_p=args.dropout_p,
                    mismatch_scale=args.mismatch_scale,
                    plan=p,
                    module=module,
                )
            )
            operations[f"{prefix}_vjp"] = lambda t=tile_size, p=plan: (
                macro_wavefront_vjp(
                    query,
                    key,
                    value,
                    grad_output,
                    packed_query,
                    packed_key,
                    dropout_seed,
                    tile_size=t,
                    scale=args.scale,
                    dropout_p=args.dropout_p,
                    mismatch_scale=args.mismatch_scale,
                    gradient_mask=args.gradient_mask,
                    plan=p,
                    module=module,
                )
            )

    if replay_module is not None:
        from benchmarks.persistent_wavefront_vjp import unbounded_replay_vjp

        operations[f"unbounded_replay_g{args.replay_group_size}_vjp"] = (
            lambda: unbounded_replay_vjp(
                query,
                key,
                value,
                grad_output,
                packed_query,
                packed_key,
                dropout_seed,
                group_size=args.replay_group_size,
                scale=args.scale,
                dropout_p=args.dropout_p,
                mismatch_scale=args.mismatch_scale,
                gradient_mask=args.gradient_mask,
                module=replay_module,
            )
        )

    latency = _measure_ms(
        operations,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    baseline_name = f"multilaunch_l{args.tile_sizes[0]}_vjp"
    fallback_name = next(name for name in latency if name.endswith("_vjp"))
    baseline_ms = latency.get(baseline_name, latency[fallback_name])
    operator_results = {
        name: {
            "latency_ms": value_ms,
            "baseline_ratio": value_ms / baseline_ms,
            "peak_operator_mib": _peak_mib(operations[name]),
        }
        for name, value_ms in latency.items()
    }
    replay_name = f"unbounded_replay_g{args.replay_group_size}_vjp"
    if replay_name in operations:
        folded_name = f"folded_l{args.tile_sizes[0]}_vjp"
        reference_name = (
            folded_name if folded_name in operations else fallback_name
        )
        reference = operations[reference_name]()
        replay = operations[replay_name]()
        operator_results[replay_name]["reference"] = reference_name
        operator_results[replay_name]["max_abs_errors"] = [
            float((actual - expected).abs().max())
            for actual, expected in zip(replay, reference)
        ]
    return {
        "sequence_length": seq_len,
        "operators": operator_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048]
    )
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument(
        "--plans",
        nargs="+",
        choices=EXECUTION_PLANS,
        default=["multilaunch", "folded", "persistent_rows"],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--gradient-mask", type=int, default=7)
    parser.add_argument(
        "--include-scores",
        action="store_true",
        help="also profile the debug dense-score path",
    )
    parser.add_argument(
        "--include-unbounded-replay",
        action="store_true",
        help="also benchmark the maintained exact linear-workspace replay VJP",
    )
    parser.add_argument("--replay-group-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(tile not in TILE_SIZES or tile == 0 for tile in args.tile_sizes):
        parser.error(f"--tile-sizes must be drawn from {TILE_SIZES[1:]}")
    if args.heads % args.value_heads:
        parser.error("--heads must be divisible by --value-heads")
    if args.replay_group_size <= 0:
        parser.error("--replay-group-size must be positive")

    module = load_macro_wavefront_vjp()
    replay_module = None
    if args.include_unbounded_replay:
        from benchmarks.persistent_wavefront_vjp import (
            load_persistent_wavefront_vjp,
        )

        replay_module = load_persistent_wavefront_vjp()
    report = {
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "cases": [
            _case(length, args, module, replay_module)
            for length in args.lengths
        ],
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
