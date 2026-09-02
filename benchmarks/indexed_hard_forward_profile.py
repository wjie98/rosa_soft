"""Profile exact indexed hard-forward stages and adversarial code regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

import rosa_soft  # noqa: F401 - registers torch.ops.rosa_soft
from benchmarks.indexed_hard_forward import (
    INDEXED_METHODS,
    build_occurrence_index,
    diagonal_hard_forward_from_packed,
    indexed_hard_forward_from_packed,
    load_indexed_hard_forward,
    pack_sign_bits,
)


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


def _make_case(sequence_length, pattern, args):
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
    elif pattern in {"collapsed4", "aligned_collapsed4"}:
        query[..., 2:].fill_(-1)
        key[..., 2:].fill_(-1)
        if pattern == "aligned_collapsed4":
            query[:, 1:] = key[:, :-1]
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


def _code_metrics(packed_query, packed_key, bits):
    alphabet = 1 << bits
    entropies = []
    collision_probabilities = []
    maximum_fractions = []
    for row in packed_key.reshape(-1, packed_key.size(-1)).cpu().long():
        counts = torch.bincount(row, minlength=alphabet).double()
        total = row.numel()
        probabilities = counts[counts > 0] / total
        entropies.append(float(-(probabilities * probabilities.log2()).sum()))
        collision_probabilities.append(
            float((counts * (counts - 1)).sum() / max(total * (total - 1), 1))
        )
        maximum_fractions.append(float(counts.max() / total))
    return {
        "key_entropy_bits": statistics.mean(entropies),
        "key_collision_probability": statistics.mean(collision_probabilities),
        "key_max_code_fraction": statistics.mean(maximum_fractions),
        "shifted_symbol_match_fraction": float(
            (packed_query[..., 1:] == packed_key[..., :-1]).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384],
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 128])
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=[
            "random",
            "aligned",
            "collapsed4",
            "aligned_collapsed4",
            "periodic64",
            "periodic4",
            "all_match",
            "all_mismatch",
        ],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, nargs="+", default=[8])
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.heads % args.value_heads:
        raise ValueError("--heads must be divisible by --value-heads")
    if any(not 1 <= bits <= 8 for bits in args.bits):
        raise ValueError("--bits entries must be in [1, 8]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    module = load_indexed_hard_forward()
    rows = []
    for bits in args.bits:
        case_args = argparse.Namespace(**(vars(args) | {"bits": bits}))
        for sequence_length in args.sequence_lengths:
            for pattern in args.patterns:
                query, key, value = _make_case(
                    sequence_length,
                    pattern,
                    case_args,
                )
                packed_query, packed_key = pack_sign_bits(
                    query,
                    key,
                    module=module,
                )
                offsets, occurrences = build_occurrence_index(
                    packed_key,
                    symbol_dim=bits,
                    module=module,
                )
                metrics = _code_metrics(packed_query, packed_key, bits)
                pack_ms = _measure_ms(
                    lambda: pack_sign_bits(query, key, module=module),
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                index_ms = _measure_ms(
                    lambda: build_occurrence_index(
                        packed_key,
                        symbol_dim=bits,
                        module=module,
                    ),
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                for window in args.windows:
                    current_ms = _measure_ms(
                        lambda: torch.ops.rosa_soft.hard_forward(
                            query,
                            key,
                            value,
                        ),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    route_ms = {}
                    for method in INDEXED_METHODS:
                        route_ms[method] = _measure_ms(
                            lambda method=method: (
                                indexed_hard_forward_from_packed(
                                    packed_query,
                                    packed_key,
                                    value,
                                    offsets,
                                    occurrences,
                                    max_suffix_length=window,
                                    method=method,
                                    module=module,
                                )
                            ),
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )

                    full_ms = {}
                    for method in INDEXED_METHODS:

                        def full_indexed(method=method):
                            current_query, current_key = pack_sign_bits(
                                query,
                                key,
                                module=module,
                            )
                            current_offsets, current_occurrences = (
                                build_occurrence_index(
                                    current_key,
                                    symbol_dim=bits,
                                    module=module,
                                )
                            )
                            return indexed_hard_forward_from_packed(
                                current_query,
                                current_key,
                                value,
                                current_offsets,
                                current_occurrences,
                                max_suffix_length=window,
                                method=method,
                                module=module,
                            )

                        full_ms[method] = _measure_ms(
                            full_indexed,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                    diagonal_route_ms = _measure_ms(
                        lambda: diagonal_hard_forward_from_packed(
                            packed_query,
                            packed_key,
                            value,
                            max_suffix_length=window,
                            module=module,
                        ),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )

                    def full_diagonal():
                        current_query, current_key = pack_sign_bits(
                            query,
                            key,
                            module=module,
                        )
                        return diagonal_hard_forward_from_packed(
                            current_query,
                            current_key,
                            value,
                            max_suffix_length=window,
                            module=module,
                        )

                    diagonal_full_ms = _measure_ms(
                        full_diagonal,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    indexed_full_ms = full_ms["certificate"]
                    row = {
                        "sequence_length": sequence_length,
                        "symbol_dim": bits,
                        "window": window,
                        "pattern": pattern,
                        "current_full_ms": current_ms,
                        "pack_ms": pack_ms,
                        "index_build_ms": index_ms,
                        "occurrence_route_ms": route_ms["occurrence"],
                        "certificate_route_ms": route_ms["certificate"],
                        "hybrid_route_ms": route_ms["hybrid"],
                        "occurrence_full_ms": full_ms["occurrence"],
                        "certificate_full_ms": full_ms["certificate"],
                        "hybrid_full_ms": full_ms["hybrid"],
                        "indexed_full_ms": indexed_full_ms,
                        "indexed_current_ratio": indexed_full_ms / current_ms,
                        "diagonal_route_ms": diagonal_route_ms,
                        "diagonal_full_ms": diagonal_full_ms,
                        "diagonal_current_ratio": diagonal_full_ms / current_ms,
                        "index_workspace_bytes": (
                            offsets.numel() * offsets.element_size()
                            + occurrences.numel() * occurrences.element_size()
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
