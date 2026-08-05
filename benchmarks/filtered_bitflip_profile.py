"""Profile exact filtered bit flips across synthetic training trajectories."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.filtered_bitflip import (  # noqa: E402
    CodeOccurrenceIndex,
    generate_influence_events,
    semantic_bit_flips,
    unlimited_hard_routes_diagonal,
    unlimited_hard_routes_direct,
)
from benchmarks.filtered_bitflip_indexes import build_match_index  # noqa: E402
from benchmarks.filtered_bitflip_winner import (  # noqa: E402
    winner_filtered_bitflip,
)


TRAJECTORIES = ("shift_random", "shift_motif", "collapse")
INDEXES = ("direct", "rolling_hash", "dyadic", "suffix_array")
WINNER_BACKENDS = ("rle", "segment")


def make_training_codes(
    sequence_length: int,
    bit_width: int,
    trajectory: str,
    clarity: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Interpolate independent codes toward one structured hard language."""

    if trajectory not in TRAJECTORIES:
        raise ValueError(f"unknown trajectory: {trajectory}")
    if not 0.0 <= clarity <= 1.0:
        raise ValueError("clarity must be in [0, 1]")
    generator = torch.Generator().manual_seed(int(seed))
    alphabet_size = 1 << bit_width
    query = torch.randint(
        0,
        alphabet_size,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    key = torch.randint(
        0,
        alphabet_size,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    if trajectory == "shift_random":
        target_key = torch.randint(
            0,
            alphabet_size,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        )
    elif trajectory == "shift_motif":
        motif_length = min(max(sequence_length // 8, 2), 16)
        motif = torch.randint(
            0,
            alphabet_size,
            (motif_length,),
            dtype=torch.uint8,
            generator=generator,
        )
        target_key = motif.repeat(
            (sequence_length + motif_length - 1) // motif_length
        )[:sequence_length]
    else:
        target_key = torch.zeros(sequence_length, dtype=torch.uint8)
    target_query = target_key.clone()
    if sequence_length > 1:
        target_query[1:] = target_key[:-1]
    query_mask = torch.rand(sequence_length, generator=generator) < clarity
    key_mask = torch.rand(sequence_length, generator=generator) < clarity
    query = torch.where(query_mask, target_query, query)
    key = torch.where(key_mask, target_key, key)
    return query.contiguous(), key.contiguous()


def _entropy(codes: Tensor, bit_width: int) -> float:
    counts = torch.bincount(codes.to(torch.int64), minlength=1 << bit_width)
    probabilities = counts[counts > 0].to(torch.float64)
    probabilities /= probabilities.sum().clamp_min(1)
    entropy = float(-(probabilities * probabilities.log2()).sum())
    return entropy / bit_width


def _median_ms(callable_, repeats: int) -> tuple[object, float]:
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, statistics.median(samples)


def profile_case(
    sequence_length: int,
    bit_width: int,
    trajectory: str,
    clarity: float,
    seed: int,
    repeats: int,
) -> dict[str, object]:
    query, key = make_training_codes(
        sequence_length,
        bit_width,
        trajectory,
        clarity,
        seed,
    )
    direct, direct_ms = _median_ms(
        lambda: unlimited_hard_routes_direct(query, key),
        repeats,
    )
    diagonal, diagonal_ms = _median_ms(
        lambda: unlimited_hard_routes_diagonal(query, key),
        repeats,
    )
    if not torch.equal(direct.routes, diagonal.routes):
        raise RuntimeError("direct and diagonal hard routes disagree")
    occurrence, occurrence_build_ms = _median_ms(
        lambda: CodeOccurrenceIndex.build(query, key),
        repeats,
    )
    flips = semantic_bit_flips(sequence_length, bit_width)
    pair_scan_checks = bit_width * sequence_length * (sequence_length - 1)
    index_results = {}
    built_indexes = {}
    raw_event_count = None
    for name in INDEXES:
        index, build_ms = _median_ms(
            lambda name=name: build_match_index(name, query, key),
            repeats,
        )
        built_indexes[name] = index

        def discover() -> int:
            return sum(
                len(
                    generate_influence_events(
                        query,
                        key,
                        flip,
                        match_index=index,
                        occurrence_index=occurrence,
                    )
                )
                for flip in flips
            )

        event_count, discover_ms = _median_ms(discover, repeats)
        if raw_event_count is None:
            raw_event_count = event_count
        elif raw_event_count != event_count:
            raise RuntimeError("match indexes produced different event counts")
        index_results[name] = {
            "exact": bool(index.exact),
            "build_ms": build_ms,
            "discover_ms": discover_ms,
            "total_ms": build_ms + discover_ms,
            "logical_bytes": int(index.logical_bytes),
        }
    winner_results = {}
    preferred_match = built_indexes[
        "direct" if clarity < 0.5 else "dyadic"
    ]
    reference_routes = None
    for name in WINNER_BACKENDS:
        result, elapsed_ms = _median_ms(
            lambda name=name: winner_filtered_bitflip(
                query,
                key,
                bit_width,
                match_index=preferred_match,
                winner_backend=name,
            ),
            repeats,
        )
        if reference_routes is None:
            reference_routes = result.flipped_routes
        elif not torch.equal(reference_routes, result.flipped_routes):
            raise RuntimeError("winner indexes produced different routes")
        winner_results[name] = {
            "elapsed_ms": elapsed_ms,
            "raw_events": result.raw_events,
            "admitted_event_ranges": result.admitted_event_ranges,
            "admitted_rows": result.admitted_rows,
            "exact_recompute_rows": result.exact_recompute_rows,
            "winner_node_visits": result.winner_node_visits,
            "final_changed_rows": result.final_changed_rows,
            "final_route_change_ranges": result.final_route_change_ranges,
        }
    shift_agreement = (
        float((query[1:] == key[:-1]).to(torch.float64).mean())
        if sequence_length > 1
        else 0.0
    )
    mean_winner_length = (
        float(diagonal.lengths[1:].to(torch.float64).mean())
        if sequence_length > 1
        else 0.0
    )
    return {
        "sequence_length": sequence_length,
        "bit_width": bit_width,
        "trajectory": trajectory,
        "clarity": clarity,
        "seed": seed,
        "normalized_symbol_entropy": _entropy(
            torch.cat((query[1:], key[:-1])),
            bit_width,
        ),
        "shift_agreement": shift_agreement,
        "mean_winner_length": mean_winner_length,
        "max_winner_length": int(diagonal.lengths.max()),
        "pair_scan_checks": pair_scan_checks,
        "raw_local_pair_events": int(raw_event_count or 0),
        "occurrence_pair_fraction": (
            float(raw_event_count or 0) / max(pair_scan_checks, 1)
        ),
        "hard_routes": {
            "direct_ms": direct_ms,
            "diagonal_ms": diagonal_ms,
            "direct_symbol_comparisons": direct.symbol_comparisons,
            "diagonal_symbol_comparisons": diagonal.symbol_comparisons,
        },
        "occurrence_build_ms": occurrence_build_ms,
        "match_indexes": index_results,
        "winner_indexes": winner_results,
    }


def _summary(cases: list[dict[str, object]]) -> dict[str, object]:
    exact_names = ("direct", "dyadic", "suffix_array")
    index_wins = {name: 0 for name in exact_names}
    winner_wins = {name: 0 for name in WINNER_BACKENDS}
    for case in cases:
        exact_winner = min(
            exact_names,
            key=lambda name: case["match_indexes"][name]["total_ms"],
        )
        index_wins[exact_winner] += 1
        winner = min(
            WINNER_BACKENDS,
            key=lambda name: case["winner_indexes"][name]["elapsed_ms"],
        )
        winner_wins[winner] += 1
    return {
        "case_count": len(cases),
        "exact_match_index_wins": index_wins,
        "winner_index_wins": winner_wins,
        "mean_occurrence_pair_fraction": statistics.mean(
            float(case["occurrence_pair_fraction"]) for case in cases
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[64])
    parser.add_argument("--bit-widths", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument(
        "--trajectories",
        nargs="+",
        choices=TRAJECTORIES,
        default=list(TRAJECTORIES),
    )
    parser.add_argument(
        "--clarities",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    cases = []
    for sequence_length in args.sequence_lengths:
        for bit_width in args.bit_widths:
            for trajectory in args.trajectories:
                for clarity in args.clarities:
                    cases.append(
                        profile_case(
                            sequence_length,
                            bit_width,
                            trajectory,
                            clarity,
                            args.seed,
                            args.repeats,
                        )
                    )
    report = {
        "configuration": {
            "sequence_lengths": args.sequence_lengths,
            "bit_widths": args.bit_widths,
            "trajectories": args.trajectories,
            "clarities": args.clarities,
            "seed": args.seed,
            "repeats": args.repeats,
            "torch_version": torch.__version__,
        },
        "cases": cases,
        "summary": _summary(cases),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
