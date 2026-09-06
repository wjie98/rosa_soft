"""Profile exact unlimited diagonal-DP hard ROSA against production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

import rosa_soft  # noqa: F401 - registers production operators
from benchmarks.indexed_hard_forward import (
    diagonal_hard_forward_from_packed,
    load_indexed_hard_forward,
    pack_sign_bits,
)


def _latency_ms(operation, *, warmup: int, repeats: int, rounds: int) -> float:
    samples = []
    for _ in range(rounds):
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
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)


def _make_case(
    tokens: int,
    pattern: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cuda").manual_seed(
        args.seed + 31 * tokens + args.bits
    )
    shape = (args.batch, tokens, args.heads, args.bits)
    query = torch.randn(shape, dtype=dtype, device="cuda", generator=generator)
    key = torch.randn(shape, dtype=dtype, device="cuda", generator=generator)
    if pattern == "equal":
        query.fill_(1)
        key.fill_(1)
    elif pattern == "mismatch":
        query.fill_(1)
        key.fill_(-1)
    elif pattern == "aligned":
        query[:, 1:].copy_(key[:, :-1])
    elif pattern.startswith("periodic"):
        period = int(pattern.removeprefix("periodic"))
        codes = torch.randint(
            0,
            2,
            (args.batch, period, args.heads, args.bits),
            device="cuda",
            generator=generator,
        ).mul_(2).sub_(1).to(dtype)
        positions = torch.arange(tokens, device="cuda") % period
        query.copy_(codes[:, positions])
        key.copy_(codes[:, positions])
    elif pattern != "random":
        raise ValueError(f"unknown pattern: {pattern}")
    value = torch.randn(
        (args.batch, tokens, args.value_heads, args.value_dim),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    return query, key, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["random", "aligned", "periodic64", "periodic4", "equal", "mismatch"],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9471)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.heads % args.value_heads:
        raise ValueError("--heads must be divisible by --value-heads")
    if not 1 <= args.bits <= 8:
        raise ValueError("--bits must be in [1, 8]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    module = load_indexed_hard_forward()
    rows = []
    for tokens in args.lengths:
        for pattern in args.patterns:
            query, key, value = _make_case(tokens, pattern, args)
            packed_query, packed_key = pack_sign_bits(query, key, module=module)

            def production():
                return torch.ops.rosa_soft.hard_forward(query, key, value)[0]

            def diagonal_from_packed():
                return diagonal_hard_forward_from_packed(
                    packed_query,
                    packed_key,
                    value,
                    max_suffix_length=tokens,
                    module=module,
                )[0]

            def diagonal_full():
                current_query, current_key = pack_sign_bits(
                    query, key, module=module
                )
                return diagonal_hard_forward_from_packed(
                    current_query,
                    current_key,
                    value,
                    max_suffix_length=tokens,
                    module=module,
                )[0]

            expected = production()
            actual = diagonal_from_packed()
            max_abs_error = float((actual - expected).abs().max())
            production_ms = _latency_ms(
                production,
                warmup=args.warmup,
                repeats=args.repeats,
                rounds=args.rounds,
            )
            diagonal_route_ms = _latency_ms(
                diagonal_from_packed,
                warmup=args.warmup,
                repeats=args.repeats,
                rounds=args.rounds,
            )
            diagonal_full_ms = _latency_ms(
                diagonal_full,
                warmup=args.warmup,
                repeats=args.repeats,
                rounds=args.rounds,
            )
            row = {
                "tokens": tokens,
                "pattern": pattern,
                "production_ms": production_ms,
                "diagonal_route_ms": diagonal_route_ms,
                "diagonal_full_ms": diagonal_full_ms,
                "full_ratio": diagonal_full_ms / production_ms,
                "max_abs_error": max_abs_error,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "settings": vars(args) | {"output": str(args.output)},
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
