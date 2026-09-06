"""Profile the exact linear-workspace slabbed RosaSoft VJP."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections.abc import Callable

import torch
from torch import Tensor

from benchmarks.macro_wavefront_vjp import (
    load_macro_wavefront_vjp,
    macro_wavefront_vjp,
)
from benchmarks.persistent_wavefront_vjp import (
    load_persistent_wavefront_vjp,
    unbounded_replay_vjp,
)
from benchmarks.slabbed_checkpoint_replay import (
    load_slabbed_checkpoint_replay,
    slabbed_live_state_elements,
    slabbed_replay_vjp,
)


Result = tuple[Tensor, Tensor, Tensor]
Operation = Callable[[], Result]


def _latencies(
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
    result = operation()
    torch.cuda.synchronize()
    del result
    return (torch.cuda.max_memory_allocated() - baseline) / 2**20


def _case(seq_len: int, args, slabbed_module, replay_module, folded_module):
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
    shifts = torch.arange(args.bits, dtype=torch.int64, device="cuda")
    packed_query = (
        ((query > 0).to(torch.int64).permute(0, 2, 1, 3) << shifts)
        .sum(-1)
        .to(torch.int32)
    )
    packed_key = (
        ((key > 0).to(torch.int64).permute(0, 2, 1, 3) << shifts)
        .sum(-1)
        .to(torch.int32)
    )
    dropout_seed = (
        torch.tensor(args.seed, dtype=torch.int64, device="cuda")
        if args.dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    common = (
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
    )
    operations: dict[str, Operation] = {
        "slabbed": lambda: slabbed_replay_vjp(
            *common,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
            gradient_mask=args.gradient_mask,
            module=slabbed_module,
        ),
        "unbounded_replay_g256": lambda: unbounded_replay_vjp(
            *common,
            group_size=256,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
            gradient_mask=args.gradient_mask,
            module=replay_module,
        ),
        "folded_l32": lambda: macro_wavefront_vjp(
            *common,
            tile_size=32,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
            gradient_mask=args.gradient_mask,
            plan="folded",
            module=folded_module,
        ),
        "persistent_rows_l32": lambda: macro_wavefront_vjp(
            *common,
            tile_size=32,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
            gradient_mask=args.gradient_mask,
            plan="persistent_rows",
            module=folded_module,
        ),
    }
    latency = _latencies(
        operations,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    peak = {name: _peak_mib(operation) for name, operation in operations.items()}
    expected = operations["unbounded_replay_g256"]()
    actual = operations["slabbed"]()
    errors = [
        0.0
        if reference.numel() == 0
        else float((candidate - reference).abs().max())
        for candidate, reference in zip(actual, expected)
    ]
    return {
        "seq_len": seq_len,
        "latency_ms": latency,
        "peak_operator_mib": peak,
        "slabbed_replay_max_abs_errors": errors,
        "slabbed_declared_live_state_mib": (
            4
            * slabbed_live_state_elements(
                args.batch,
                args.heads,
                seq_len,
                needs_symbol_gradients=bool(args.gradient_mask & 3),
            )
            / 2**20
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096]
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--gradient-mask", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=58123)
    args = parser.parse_args()

    slabbed_module = load_slabbed_checkpoint_replay()
    replay_module = load_persistent_wavefront_vjp()
    folded_module = load_macro_wavefront_vjp()
    rows = []
    for seq_len in args.lengths:
        row = _case(seq_len, args, slabbed_module, replay_module, folded_module)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "compute_capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "config": vars(args),
                "rows": rows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
