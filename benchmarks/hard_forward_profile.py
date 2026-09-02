"""Profile exact hard RosaSoft across code-entropy regimes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import rosa_soft  # noqa: F401 - registers torch.ops.rosa_soft


def _measure_ms(operation, *, warmup: int, repeats: int) -> float:
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


def _make_case(
    sequence_length: int,
    pattern: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cuda").manual_seed(
        args.seed + sequence_length
    )
    shape = (args.batch, sequence_length, args.heads, args.bits)
    query = torch.randn(
        shape,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        shape,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    if pattern == "all_match":
        query.fill_(1)
        key.fill_(1)
    elif pattern == "all_mismatch":
        query.fill_(1)
        key.fill_(-1)
    elif pattern == "aligned":
        query[:, 1:] = key[:, :-1]
    elif pattern.startswith("periodic"):
        period = int(pattern.removeprefix("periodic"))
        codes = (
            2
            * torch.randint(
                0,
                2,
                (args.batch, period, args.heads, args.bits),
                device="cuda",
                generator=generator,
            )
            - 1
        ).to(dtype)
        positions = torch.arange(sequence_length, device="cuda") % period
        query.copy_(codes[:, positions])
        key.copy_(codes[:, positions])
    elif pattern != "random":
        raise ValueError(f"unknown pattern: {pattern}")
    value = torch.randn(
        (
            args.batch,
            sequence_length,
            args.value_heads,
            args.value_dim,
        ),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    return query, key, value


def _code_metrics(
    packed_query: torch.Tensor,
    packed_key: torch.Tensor,
    bits: int,
) -> dict[str, float]:
    alphabet = 1 << bits
    key_rows = packed_key.reshape(-1, packed_key.shape[-1]).long()
    entropies = []
    collision_probabilities = []
    maximum_fractions = []
    unique_fractions = []
    for row in key_rows:
        counts = torch.unique(row, return_counts=True)[1].double()
        total = int(row.numel())
        probabilities = counts / total
        entropies.append(float(-(probabilities * probabilities.log2()).sum()))
        collision_probabilities.append(
            float((counts * (counts - 1)).sum() / max(total * (total - 1), 1))
        )
        maximum_fractions.append(float(counts.max() / total))
        unique_fractions.append(
            float(counts.numel() / min(total, alphabet))
        )
    shifted = (
        packed_query[..., 1:] == packed_key[..., :-1]
    ).float().mean()
    return {
        "key_entropy_bits": statistics.mean(entropies),
        "key_collision_probability": statistics.mean(
            collision_probabilities
        ),
        "key_max_code_fraction": statistics.mean(maximum_fractions),
        "key_unique_code_fraction": statistics.mean(unique_fractions),
        "latest_route_symbol_match_fraction": float(shifted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[4096, 8192, 16384]
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=[
            "random",
            "aligned",
            "periodic64",
            "periodic4",
            "all_match",
            "all_mismatch",
        ],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.heads % args.value_heads:
        raise ValueError("--heads must be divisible by --value-heads")
    if not 1 <= args.bits <= 32:
        raise ValueError("--bits must be in [1, 32]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rows = []
    random_latency: dict[int, float] = {}
    for sequence_length in args.sequence_lengths:
        for pattern in args.patterns:
            query, key, value = _make_case(sequence_length, pattern, args)
            _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
                query, key, value
            )
            metrics = _code_metrics(packed_query, packed_key, args.bits)
            operation = lambda: torch.ops.rosa_soft.hard_forward(
                query, key, value
            )
            latency = _measure_ms(
                operation,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            if pattern == "random":
                random_latency[sequence_length] = latency
            random_ms = random_latency.get(sequence_length)
            row = {
                "sequence_length": sequence_length,
                "pattern": pattern,
                "latency_ms": latency,
                "random_latency_ratio": (
                    latency / random_ms if random_ms is not None else None
                ),
                **metrics,
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
