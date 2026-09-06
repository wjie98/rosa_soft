"""Profile production and fixed-slab exact unbounded RosaSoft VJPs."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

import rosa_soft  # noqa: F401 - registers production operators
from benchmarks.persistent_wavefront_vjp import (
    grouped_checkpoint_reverse,
    grouped_checkpoint_reverse_tensor_symbols,
    grouped_checkpoint_stats_group_size,
    grouped_checkpoint_vjp,
    load_persistent_wavefront_vjp,
    macro_checkpoint_stats,
    unbounded_replay_stats,
    unbounded_replay_vjp,
)


Operation = Callable[[], tuple[Tensor, Tensor, Tensor]]


def _latencies_ms(
    operations: dict[str, Operation],
    *,
    warmup: int,
    repeats: int,
    rounds: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    samples = {name: [] for name in operations}
    names = list(operations)
    for round_index in range(rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
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
    return (
        {name: statistics.median(values) for name, values in samples.items()},
        samples,
    )


def _peak_mib(operation: Operation) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / 2**20


def _max_errors(
    actual: tuple[Tensor, Tensor, Tensor],
    expected: tuple[Tensor, Tensor, Tensor],
) -> list[float]:
    return [
        0.0
        if reference.numel() == 0
        else float((candidate - reference).abs().max())
        for candidate, reference in zip(actual, expected)
    ]


def _profile_length(
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
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = (
        torch.tensor(args.seed, dtype=torch.int64, device="cuda")
        if args.dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    checkpoint_stats_group_size = (
        args.checkpoint_stats_group_size
        if args.checkpoint_stats_group_size is not None
        else grouped_checkpoint_stats_group_size(query)
    )

    def finite_production(window: int) -> tuple[Tensor, Tensor, Tensor]:
        return torch.ops.rosa_soft.surrogate_vjp_masked(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            dropout_seed,
            window,
            args.scale,
            args.dropout_p,
            args.mismatch_scale,
            args.gradient_mask,
        )

    operations: dict[str, Operation] = {
        "production_auto": lambda: torch.ops.rosa_soft.surrogate_vjp_unbounded_masked(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            dropout_seed,
            args.scale,
            args.dropout_p,
            args.mismatch_scale,
            args.gradient_mask,
        ),
        "grouped_checkpoint": lambda: grouped_checkpoint_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            dropout_seed,
            stats_group_size=checkpoint_stats_group_size,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
            gradient_mask=args.gradient_mask,
            module=module,
        ),
    }
    if args.include_components:
        component_stats = unbounded_replay_stats(
            value,
            grad_output,
            packed_query,
            packed_key,
            dropout_seed,
            symbol_dim=args.bits,
            group_size=checkpoint_stats_group_size,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
            module=module,
        )
        operations[
            f"stats_s{checkpoint_stats_group_size}"
        ] = lambda: unbounded_replay_stats(
            value,
            grad_output,
            packed_query,
            packed_key,
            dropout_seed,
            symbol_dim=args.bits,
            group_size=checkpoint_stats_group_size,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
            module=module,
        )
        operations["grouped_reverse"] = lambda: grouped_checkpoint_reverse(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            dropout_seed,
            component_stats,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
            gradient_mask=args.gradient_mask,
            module=module,
        )
        if args.tensor_symbols:
            operations["tensor_symbol_reverse"] = (
                lambda: grouped_checkpoint_reverse_tensor_symbols(
                    query,
                    key,
                    value,
                    grad_output,
                    packed_query,
                    packed_key,
                    dropout_seed,
                    component_stats,
                    scale=args.scale,
                    dropout_p=args.dropout_p,
                    mismatch_scale=args.mismatch_scale,
                    gradient_mask=args.gradient_mask,
                    tensor_symbol_mask=args.tensor_symbol_mask,
                    specialized_replay=args.specialized_replay,
                    module=module,
                )
            )
        for macro_diagonals in args.macro_stats:
            operations[f"macro_stats_m{macro_diagonals}"] = (
                lambda macro_diagonals=macro_diagonals: macro_checkpoint_stats(
                    value,
                    grad_output,
                    packed_query,
                    packed_key,
                    dropout_seed,
                    symbol_dim=args.bits,
                    macro_diagonals=macro_diagonals,
                    slab_size=checkpoint_stats_group_size,
                    scale=args.scale,
                    dropout_p=args.dropout_p,
                    mismatch_scale=args.mismatch_scale,
                    module=module,
                )
            )
    if args.include_finite_controls:
        operations["finite_w32"] = lambda: finite_production(min(32, seq_len))
        operations["finite_full_horizon"] = lambda: finite_production(seq_len)
    for group_size in args.group_sizes:
        operations[f"selected_s{group_size}"] = (
            lambda group_size=group_size: unbounded_replay_vjp(
                query,
                key,
                value,
                grad_output,
                packed_query,
                packed_key,
                dropout_seed,
                group_size=group_size,
                scale=args.scale,
                dropout_p=args.dropout_p,
                mismatch_scale=args.mismatch_scale,
                gradient_mask=args.gradient_mask,
                module=module,
            )
        )

    medians, samples = _latencies_ms(
        operations,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    expected = operations["production_auto"]()
    results: dict[str, object] = {}
    for name, operation in operations.items():
        result = {
            "latency_ms": medians[name],
            "latency_samples_ms": samples[name],
            "peak_operator_mib": _peak_mib(operation),
        }
        if (
            name in ("grouped_checkpoint", "tensor_symbol_reverse")
            or name.startswith("selected_s")
        ):
            result["unbounded_max_abs_errors"] = _max_errors(
                operation(), expected
            )
        if name.startswith("macro_stats_m"):
            result["stats_max_abs_error"] = float(
                (operation() - component_stats).abs().max()
            )
        results[name] = result
    return {"seq_len": seq_len, "operators": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[512, 1024, 2048]
    )
    parser.add_argument(
        "--group-sizes", type=int, nargs="+", default=[32, 128, 256]
    )
    parser.add_argument("--checkpoint-stats-group-size", type=int)
    parser.add_argument("--include-finite-controls", action="store_true")
    parser.add_argument("--include-components", action="store_true")
    parser.add_argument(
        "--macro-stats", type=int, nargs="*", default=[]
    )
    parser.add_argument("--tensor-symbols", action="store_true")
    parser.add_argument(
        "--tensor-symbol-mask", type=int, choices=[1, 2, 3], default=3
    )
    parser.add_argument("--specialized-replay", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--gradient-mask", type=int, default=7)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=58123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    module = load_persistent_wavefront_vjp()
    rows = []
    for seq_len in args.lengths:
        row = _profile_length(seq_len, args, module)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": {
            **vars(args),
            "output": str(args.output) if args.output else None,
        },
        "rows": rows,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
