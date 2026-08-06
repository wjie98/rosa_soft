"""Profile the exact suffix-automaton bitflip research route."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from benchmarks.filtered_bitflip import brute_force_bitflip
from benchmarks.filtered_bitflip_compact import compact_filtered_bitflip
from benchmarks.filtered_bitflip_profile import make_training_codes
from benchmarks.sam_bitflip import (
    ExplicitEndPositionIndex,
    ImplicitEndPositionIndex,
    SuffixAutomaton,
    materialize_route_changes,
    sam_bitflip,
)
from benchmarks.sam_bitflip_native import (
    DEFAULT_LIBRARY as DEFAULT_NATIVE_LIBRARY,
    NativeSamBitflip,
    build_native_sam_bitflip,
)


CASES = (
    ("independent_d1", 1, "shift_random", 0.0),
    ("independent_d8", 8, "shift_random", 0.0),
    ("shift_random_d8", 8, "shift_random", 1.0),
    ("shift_motif_d4", 4, "shift_motif", 1.0),
    ("collapse_d8", 8, "collapse", 1.0),
)


CONFIGURATIONS = (
    ("optimized_compressed", "implicit", 0, "bitset", True, False),
    ("optimized_materialized", "implicit", 0, "bitset", True, True),
    ("no_arithmetic", "implicit", 0, "bitset", False, False),
    ("hot8_arithmetic", "implicit", 8, "bitset", True, False),
    ("explicit_arithmetic", "explicit", 0, "bitset", True, False),
)


def _median_ms(function, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, statistics.median(samples)


def _assert_parity(result, expected, name):
    routes = result.flipped_routes
    lengths = result.flipped_lengths
    if routes is None or lengths is None:
        routes, lengths = materialize_route_changes(
            result.base,
            result.route_changes,
        )
    if not torch.equal(routes, expected.flipped_routes):
        raise RuntimeError(f"{name} routes differ from full rerun")
    if not torch.equal(lengths, expected.flipped_lengths):
        raise RuntimeError(f"{name} lengths differ from full rerun")


def profile_case(
    name,
    sequence_length,
    bit_width,
    trajectory,
    clarity,
    seed,
    repeats,
    compare_v1,
    native,
):
    query, key = make_training_codes(
        sequence_length,
        bit_width,
        trajectory,
        clarity,
        seed,
    )
    brute, brute_ms = _median_ms(
        lambda: brute_force_bitflip(query, key, bit_width),
        repeats,
    )
    configurations = {}
    for (
        config_name,
        endpos_backend,
        cache_size,
        center_backend,
        arithmetic_endpos,
        materialize_routes,
    ) in CONFIGURATIONS:
        result, elapsed_ms = _median_ms(
            lambda endpos_backend=endpos_backend,
            cache_size=cache_size,
            center_backend=center_backend,
            arithmetic_endpos=arithmetic_endpos,
            materialize_routes=materialize_routes: sam_bitflip(
                query,
                key,
                bit_width,
                endpos_backend=endpos_backend,
                hot_cache_size=cache_size,
                center_backend=center_backend,
                arithmetic_endpos=arithmetic_endpos,
                materialize_routes=materialize_routes,
            ),
            repeats,
        )
        _assert_parity(result, brute, config_name)
        configurations[config_name] = {
            "elapsed_ms": elapsed_ms,
            "speedup_over_full_rerun": brute_ms / elapsed_ms,
            "profile": asdict(result.profile),
        }

    native_result = None
    if native is not None:
        result, elapsed_ms = _median_ms(
            lambda: native.solve(query, key, bit_width),
            repeats,
        )
        if not torch.equal(result.flipped_routes, brute.flipped_routes):
            raise RuntimeError("native C++ routes differ from full rerun")
        if not torch.equal(result.flipped_lengths, brute.flipped_lengths):
            raise RuntimeError("native C++ lengths differ from full rerun")
        native_result = {
            "elapsed_ms": elapsed_ms,
            "speedup_over_full_rerun": brute_ms / elapsed_ms,
            "profile": result.profile,
        }

    v1 = None
    if compare_v1:
        result, elapsed_ms = _median_ms(
            lambda: compact_filtered_bitflip(
                query,
                key,
                bit_width,
                materialize_routes=True,
            ),
            repeats,
        )
        _assert_parity(result, brute, "frozen v1")
        v1 = {
            "elapsed_ms": elapsed_ms,
            "speedup_over_full_rerun": brute_ms / elapsed_ms,
            "raw_events": result.raw_events,
            "materialized_events": result.materialized_events,
            "processed_event_cells": result.processed_event_cells,
            "sparse_replacement_rows": result.sparse_replacement_rows,
            "route_change_descriptors": result.route_change_descriptors,
        }

    return {
        "case": name,
        "sequence_length": sequence_length,
        "bit_width": bit_width,
        "full_rerun_ms": brute_ms,
        "sam": configurations,
        "native_cpp_materialized": native_result,
        "frozen_v1": v1,
    }


def profile_endpos_scaling(lengths, repeats, seed):
    results = []
    for sequence_length in lengths:
        generator = torch.Generator().manual_seed(seed + sequence_length)
        random_key = torch.randint(
            0,
            256,
            (max(sequence_length - 1, 0),),
            dtype=torch.uint8,
            generator=generator,
        )
        collapse_key = torch.zeros_like(random_key)
        for name, key in (("random_d8", random_key), ("collapse", collapse_key)):
            automaton, sam_ms = _median_ms(
                lambda key=key: SuffixAutomaton(key.tolist()),
                repeats,
            )
            explicit, explicit_ms = _median_ms(
                lambda: ExplicitEndPositionIndex(automaton),
                repeats,
            )
            implicit, implicit_ms = _median_ms(
                lambda: ImplicitEndPositionIndex(automaton),
                repeats,
            )
            results.append(
                {
                    "case": name,
                    "sequence_length": sequence_length,
                    "states": automaton.state_count,
                    "edges": automaton.edge_count,
                    "sam_build_ms": sam_ms,
                    "explicit_build_ms": explicit_ms,
                    "implicit_build_ms": implicit_ms,
                    "explicit_logical_entries": explicit.logical_entries,
                    "implicit_logical_entries": implicit.logical_entries,
                    "entry_ratio_explicit_over_implicit": (
                        explicit.logical_entries / implicit.logical_entries
                    ),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[32, 64])
    parser.add_argument(
        "--index-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compare-v1", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--build-native", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    native = None
    if args.build_native:
        build_native_sam_bitflip()
        args.native = True
    if args.native:
        native = NativeSamBitflip(DEFAULT_NATIVE_LIBRARY)

    results = []
    for sequence_length in args.sequence_lengths:
        for name, bit_width, trajectory, clarity in CASES:
            results.append(
                profile_case(
                    name,
                    sequence_length,
                    bit_width,
                    trajectory,
                    clarity,
                    args.seed,
                    args.repeats,
                    args.compare_v1,
                    native,
                )
            )
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "repeats": args.repeats,
        "results": results,
        "endpos_scaling": profile_endpos_scaling(
            args.index_lengths,
            args.repeats,
            args.seed,
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
