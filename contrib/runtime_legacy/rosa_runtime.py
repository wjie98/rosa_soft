"""Profile exact RosaRuntime work, memory, and chunk invariance."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rosa_soft import RosaRuntime


def _make_case(
    batch: int,
    tokens: int,
    heads: int,
    payload_heads: int,
    bits: int,
    pattern: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + tokens)
    maximum = 1 << bits
    if pattern == "random":
        query = torch.randint(
            maximum,
            (batch, tokens, heads),
            dtype=torch.uint8,
            generator=generator,
        )
        key = torch.randint(
            maximum,
            (batch, tokens, heads),
            dtype=torch.uint8,
            generator=generator,
        )
    elif pattern == "all_match":
        query = torch.zeros(batch, tokens, heads, dtype=torch.uint8)
        key = torch.zeros_like(query)
    elif pattern == "all_mismatch":
        query = torch.zeros(batch, tokens, heads, dtype=torch.uint8)
        key = torch.ones_like(query)
    elif pattern == "skewed":
        query = torch.randint(
            maximum,
            (batch, tokens, heads),
            dtype=torch.uint8,
            generator=generator,
        )
        key = torch.randint(
            maximum,
            (batch, tokens, heads),
            dtype=torch.uint8,
            generator=generator,
        )
        query.masked_fill_(
            torch.rand(query.shape, generator=generator) < 0.9,
            0,
        )
        key.masked_fill_(
            torch.rand(key.shape, generator=generator) < 0.9,
            0,
        )
    elif pattern.startswith("periodic"):
        period = int(pattern.removeprefix("periodic"))
        template = torch.randint(
            maximum,
            (batch, period, heads),
            dtype=torch.uint8,
            generator=generator,
        )
        positions = torch.arange(tokens) % period
        query = template[:, positions].clone()
        key = template[:, positions].clone()
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    payload = torch.randint(
        maximum,
        (batch, tokens, payload_heads),
        dtype=torch.uint8,
        generator=generator,
    )
    return query, key, payload


def _run_once(
    query: torch.Tensor,
    key: torch.Tensor,
    payload: torch.Tensor,
    bits: int,
    chunk_size: int,
) -> tuple[float, dict[str, int], dict[str, int], int]:
    runtime = RosaRuntime(
        num_heads=query.size(-1),
        num_payload_heads=payload.size(-1),
        qk_bits=bits,
        payload_bits=bits,
    )
    checksum = 0
    start = time.perf_counter()
    for begin in range(0, query.size(1), chunk_size):
        end = min(query.size(1), begin + chunk_size)
        output, matched = runtime.update_packed(
            query[:, begin:end],
            key[:, begin:end],
            payload[:, begin:end],
        )
        checksum += int(output.to(torch.int64).sum())
        checksum += 1_000_003 * int(matched.sum())
    elapsed = time.perf_counter() - start
    memory = runtime.memory_stats()
    complexity = runtime.complexity_stats()
    runtime.close()
    return elapsed, memory, complexity, checksum


def _profile_case(
    query: torch.Tensor,
    key: torch.Tensor,
    payload: torch.Tensor,
    bits: int,
    chunk_size: int,
    repeats: int,
) -> dict[str, object]:
    samples = []
    selected = None
    checksums = set()
    for _ in range(repeats):
        elapsed, memory, complexity, checksum = _run_once(
            query,
            key,
            payload,
            bits,
            chunk_size,
        )
        samples.append(elapsed)
        checksums.add(checksum)
        selected = (memory, complexity)
    if len(checksums) != 1 or selected is None:
        raise RuntimeError("RosaRuntime repeats produced different outputs")
    elapsed = statistics.median(samples)
    memory, complexity = selected
    updates = int(complexity["updates"])
    return {
        "latency_ms": elapsed * 1000.0,
        "tokens_per_second": updates / elapsed,
        "logical_bytes_per_update": memory["logical_bytes"] / updates,
        "checksum": next(iter(checksums)),
        "memory": memory,
        "complexity": complexity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[4096])
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["random", "all_match", "periodic64", "skewed"],
    )
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[4096])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--payload-heads", type=int, default=1)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for tokens in args.sequence_lengths:
        for pattern in args.patterns:
            query, key, payload = _make_case(
                args.batch,
                tokens,
                args.heads,
                args.payload_heads,
                args.bits,
                pattern,
                args.seed,
            )
            expected_checksum = None
            for requested_chunk_size in args.chunk_sizes:
                chunk_size = min(tokens, requested_chunk_size)
                result = _profile_case(
                    query,
                    key,
                    payload,
                    args.bits,
                    chunk_size,
                    args.repeats,
                )
                if expected_checksum is None:
                    expected_checksum = result["checksum"]
                elif result["checksum"] != expected_checksum:
                    raise RuntimeError(
                        "chunk size changed exact RosaRuntime output"
                    )
                row = {
                    "sequence_length": tokens,
                    "pattern": pattern,
                    "chunk_size": chunk_size,
                    **result,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "settings": vars(args) | {"output": str(args.output)},
        "torch": torch.__version__,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
