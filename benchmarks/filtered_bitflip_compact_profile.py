"""Profile the frozen filtered-bitflip v1 route and exact baselines."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from benchmarks.filtered_bitflip_certificates import ReplacementCertificateTree
from benchmarks.filtered_bitflip_compact import (
    CompactReplacementCertificateTree,
    CompactSuffixNodeReplacementIndex,
    FILTERED_BITFLIP_INDEX_VERSION,
    compact_filtered_bitflip,
)
from benchmarks.filtered_bitflip_hybrid import hybrid_filtered_bitflip
from benchmarks.filtered_bitflip_indexes import _SuffixArrayPair
from benchmarks.filtered_bitflip_native import DEFAULT_LIBRARY, build_libsais
from benchmarks.filtered_bitflip_profile import make_training_codes
from benchmarks.filtered_bitflip_suffix_nodes import SuffixNodeReplacementIndex


def _median_ms(function, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, statistics.median(samples)


def profile_suffix_builds(
    lengths: list[int],
    repeats: int,
    native_library: Path | None,
) -> list[dict[str, object]]:
    if native_library is None or not native_library.is_file():
        return []
    results = []
    for sequence_length in lengths:
        generator = random.Random(3000 + sequence_length)
        query = tuple(generator.randrange(256) for _ in range(sequence_length))
        key = tuple(generator.randrange(256) for _ in range(sequence_length))
        _, python_ms = _median_ms(
            lambda: _SuffixArrayPair(query, key), repeats
        )
        _, libsais_ms = _median_ms(
            lambda: _SuffixArrayPair(
                query,
                key,
                backend="libsais",
                library_path=native_library,
            ),
            repeats,
        )
        results.append(
            {
                "sequence_length": sequence_length,
                "python_ms": python_ms,
                "libsais_ms": libsais_ms,
                "speedup": python_ms / libsais_ms,
            }
        )
    return results


def profile_structures(
    lengths: list[int], repeats: int, top_k: int
) -> list[dict[str, object]]:
    cases = (
        ("independent_d1", 1, "shift_random", 0.0),
        ("collapse", 8, "collapse", 1.0),
    )
    results = []
    for name, bit_width, trajectory, clarity in cases:
        for sequence_length in lengths:
            query, key = make_training_codes(
                sequence_length,
                bit_width,
                trajectory,
                clarity,
                17,
            )
            old_suffix, old_suffix_ms = _median_ms(
                lambda: SuffixNodeReplacementIndex(query, key), repeats
            )
            compact_suffix, compact_suffix_ms = _median_ms(
                lambda: CompactSuffixNodeReplacementIndex(query, key),
                repeats,
            )
            old_certificate, old_certificate_ms = _median_ms(
                lambda: ReplacementCertificateTree.build(query, key),
                repeats,
            )
            compact_certificate, compact_certificate_ms = _median_ms(
                lambda: CompactReplacementCertificateTree(
                    query,
                    key,
                    compact_suffix,
                    top_k=top_k,
                ),
                repeats,
            )
            results.append(
                {
                    "case": name,
                    "sequence_length": sequence_length,
                    "old_suffix": {
                        "build_ms": old_suffix_ms,
                        "logical_bytes": old_suffix.logical_bytes,
                        "route_postings": old_suffix.route_postings,
                    },
                    "compact_suffix": {
                        "build_ms": compact_suffix_ms,
                        "logical_bytes": compact_suffix.logical_bytes,
                        "route_values": len(compact_suffix.key_ranks),
                        "periodic_nodes": compact_suffix.periodic_nodes,
                    },
                    "old_certificate": {
                        "build_ms": old_certificate_ms,
                        "logical_bytes": old_certificate.logical_bytes,
                        "candidate_lines": len(old_certificate.lines),
                        "postings": old_certificate.certificate_postings,
                    },
                    "compact_certificate": {
                        "build_ms": compact_certificate_ms,
                        "logical_bytes": compact_certificate.logical_bytes,
                        "retained_intervals": (
                            compact_certificate.retained_intervals
                        ),
                        "top_postings": compact_certificate.top_postings,
                    },
                }
            )
    return results


def profile_execution(
    sequence_length: int,
    repeats: int,
    top_ks: list[int],
) -> list[dict[str, object]]:
    cases = (
        ("independent", "shift_random", 0.0),
        ("shift_random", "shift_random", 1.0),
        ("shift_motif", "shift_motif", 1.0),
        ("collapse", "collapse", 1.0),
    )
    results = []
    for name, trajectory, clarity in cases:
        query, key = make_training_codes(
            sequence_length, 8, trajectory, clarity, 17
        )
        old, old_ms = _median_ms(
            lambda: hybrid_filtered_bitflip(query, key, 8), repeats
        )
        compact_results = []
        for top_k in top_ks:
            compact, compact_ms = _median_ms(
                lambda: compact_filtered_bitflip(
                    query, key, 8, top_k=top_k
                ),
                repeats,
            )
            verified = compact_filtered_bitflip(
                query,
                key,
                8,
                top_k=top_k,
                materialize_routes=True,
            )
            if not torch.equal(verified.flipped_routes, old.flipped_routes):
                raise RuntimeError("compact routes differ from exact reference")
            if not torch.equal(verified.flipped_lengths, old.flipped_lengths):
                raise RuntimeError("compact lengths differ from exact reference")
            compact_results.append(
                {
                    "top_k": top_k,
                    "elapsed_ms": compact_ms,
                    "sparse_flips": compact.sparse_flips,
                    "certificate_flips": compact.certificate_flips,
                    "suffix_fallback_rows": (
                        compact.certificate_suffix_fallback_rows
                    ),
                    "periodic_interval_flips": (
                        compact.periodic_interval_flips
                    ),
                    "periodic_interval_rows": compact.periodic_interval_rows,
                    "periodic_winner_segments": (
                        compact.periodic_winner_segments
                    ),
                    "periodic_creation_intervals_expanded": (
                        compact.periodic_creation_intervals_expanded
                    ),
                    "retained_intervals": (
                        compact.certificate_retained_intervals
                    ),
                    "top_postings": compact.top_certificate_postings,
                    "changed_route_rows": compact.changed_route_rows,
                    "periodic_route_ranges": compact.periodic_route_ranges,
                    "route_change_descriptors": (
                        compact.route_change_descriptors
                    ),
                    "affine_equivalent_route_ranges": (
                        compact.affine_equivalent_route_ranges
                    ),
                    "raw_events": compact.raw_events,
                    "materialized_events": compact.materialized_events,
                    "processed_event_cells": (
                        compact.processed_event_cells
                    ),
                    "direct_event_family_flips": (
                        compact.direct_event_family_flips
                    ),
                    "direct_event_families": compact.direct_event_families,
                    "monotone_lce_cell_flips": (
                        compact.monotone_lce_cell_flips
                    ),
                    "monotone_lce_cells": compact.monotone_lce_cells,
                    "monotone_create_occurrences": (
                        compact.monotone_create_occurrences
                    ),
                }
            )
        results.append(
            {
                "case": name,
                "sequence_length": sequence_length,
                "reference_hybrid_ms": old_ms,
                "compact": compact_results,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--execution-length", type=int, default=64)
    parser.add_argument("--top-k", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--build-native", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("validation/filtered_bitflip_compact.json"),
    )
    args = parser.parse_args()
    native_library = build_libsais() if args.build_native else DEFAULT_LIBRARY
    result = {
        "index_route": FILTERED_BITFLIP_INDEX_VERSION,
        "configuration": {
            "sequence_lengths": args.sequence_lengths,
            "execution_length": args.execution_length,
            "top_k": args.top_k,
            "repeats": args.repeats,
            "native_suffix_backend": native_library.is_file(),
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "platform": platform.platform(),
        },
        "suffix_builds": profile_suffix_builds(
            [256, 1024, 4096, 16384],
            args.repeats,
            native_library,
        ),
        "structures": profile_structures(
            args.sequence_lengths, args.repeats, min(args.top_k)
        ),
        "execution": profile_execution(
            args.execution_length, args.repeats, args.top_k
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
