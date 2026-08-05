"""Profile exact replacement-winner structures across training trajectories."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.filtered_bitflip_certificates import (  # noqa: E402
    ReplacementCertificateTree,
    certificate_filtered_bitflip,
)
from benchmarks.filtered_bitflip_profile import (  # noqa: E402
    TRAJECTORIES,
    make_training_codes,
)
from benchmarks.filtered_bitflip_hybrid import hybrid_filtered_bitflip  # noqa: E402
from benchmarks.filtered_bitflip_suffix_nodes import (  # noqa: E402
    SuffixNodeReplacementIndex,
    suffix_node_filtered_bitflip,
)


def _median_ms(callable_, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        samples.append((time.perf_counter() - start) * 1000.0)
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
    certificate_tree, certificate_build_ms = _median_ms(
        lambda: ReplacementCertificateTree.build(query, key),
        repeats,
    )
    suffix_index, suffix_build_ms = _median_ms(
        lambda: SuffixNodeReplacementIndex(query, key),
        repeats,
    )
    certificate, certificate_ms = _median_ms(
        lambda: certificate_filtered_bitflip(query, key, bit_width),
        repeats,
    )
    periodic_certificate, periodic_certificate_ms = _median_ms(
        lambda: certificate_filtered_bitflip(
            query,
            key,
            bit_width,
            overlay_backend="periodic",
        ),
        repeats,
    )
    suffix_flat, suffix_flat_ms = _median_ms(
        lambda: suffix_node_filtered_bitflip(query, key, bit_width),
        repeats,
    )
    suffix_periodic, suffix_periodic_ms = _median_ms(
        lambda: suffix_node_filtered_bitflip(
            query,
            key,
            bit_width,
            posting_backend="periodic",
        ),
        repeats,
    )
    hybrid, hybrid_ms = _median_ms(
        lambda: hybrid_filtered_bitflip(query, key, bit_width),
        repeats,
    )
    routes = (
        certificate.flipped_routes,
        periodic_certificate.flipped_routes,
        suffix_flat.flipped_routes,
        suffix_periodic.flipped_routes,
        hybrid.flipped_routes,
    )
    lengths = (
        certificate.flipped_lengths,
        periodic_certificate.flipped_lengths,
        suffix_flat.flipped_lengths,
        suffix_periodic.flipped_lengths,
        hybrid.flipped_lengths,
    )
    if not all(torch.equal(routes[0], item) for item in routes[1:]):
        raise RuntimeError("replacement structures produced different routes")
    if not all(torch.equal(lengths[0], item) for item in lengths[1:]):
        raise RuntimeError("replacement structures produced different lengths")
    return {
        "sequence_length": sequence_length,
        "bit_width": bit_width,
        "trajectory": trajectory,
        "clarity": clarity,
        "seed": seed,
        "base_structures": {
            "certificate_tree": {
                "build_ms": certificate_build_ms,
                "candidate_lines": len(certificate_tree.lines),
                "canonical_postings": certificate_tree.certificate_postings,
                "logical_bytes": certificate_tree.logical_bytes,
            },
            "suffix_nodes": {
                "build_ms": suffix_build_ms,
                "nodes": len(suffix_index.nodes),
                "route_postings": suffix_index.route_postings,
                "route_runs": suffix_index.route_runs,
                "logical_bytes": suffix_index.logical_bytes,
            },
        },
        "execution": {
            "certificate": {
                "elapsed_ms": certificate_ms,
                "tree_nodes_visited": certificate.tree_nodes_visited,
                "base_candidates_probed": certificate.base_candidates_probed,
                "canonical_forbid_entries": certificate.canonical_forbid_entries,
                "canonical_add_entries": certificate.canonical_add_entries,
            },
            "certificate_periodic": {
                "elapsed_ms": periodic_certificate_ms,
                "event_families": periodic_certificate.event_families,
                "compressed_events": (
                    periodic_certificate.periodic_compressed_events
                ),
                "tree_nodes_visited": periodic_certificate.tree_nodes_visited,
                "base_candidates_probed": (
                    periodic_certificate.base_candidates_probed
                ),
            },
            "suffix_flat": {
                "elapsed_ms": suffix_flat_ms,
                "replacement_rows": suffix_flat.replacement_rows,
                "suffix_nodes_visited": suffix_flat.suffix_nodes_visited,
                "route_candidates_probed": suffix_flat.route_candidates_probed,
            },
            "suffix_periodic": {
                "elapsed_ms": suffix_periodic_ms,
                "replacement_rows": suffix_periodic.replacement_rows,
                "suffix_nodes_visited": suffix_periodic.suffix_nodes_visited,
                "route_candidates_probed": (
                    suffix_periodic.route_candidates_probed
                ),
                "periodic_jumps": suffix_periodic.periodic_jumps,
            },
            "hybrid": {
                "elapsed_ms": hybrid_ms,
                "certificate_flips": hybrid.certificate_flips,
                "suffix_node_flips": hybrid.suffix_node_flips,
                "empty_flips": hybrid.empty_flips,
                "periodic_postings": hybrid.periodic_postings,
                "estimated_overlay_entries": hybrid.estimated_overlay_entries,
                "certificate_tree_nodes_visited": (
                    hybrid.certificate_tree_nodes_visited
                ),
                "suffix_replacement_rows": hybrid.suffix_replacement_rows,
            },
        },
        "shared": {
            "raw_events": certificate.raw_events,
            "event_cells": certificate.event_cells,
            "final_changed_rows": certificate.final_changed_rows,
            "final_route_change_ranges": (
                certificate.final_route_change_ranges
            ),
        },
    }


def profile_structure_case(
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
    certificate_tree, certificate_build_ms = _median_ms(
        lambda: ReplacementCertificateTree.build(query, key),
        repeats,
    )
    suffix_index, suffix_build_ms = _median_ms(
        lambda: SuffixNodeReplacementIndex(query, key),
        repeats,
    )
    return {
        "sequence_length": sequence_length,
        "bit_width": bit_width,
        "trajectory": trajectory,
        "clarity": clarity,
        "seed": seed,
        "certificate_tree": {
            "build_ms": certificate_build_ms,
            "candidate_lines": len(certificate_tree.lines),
            "canonical_postings": certificate_tree.certificate_postings,
            "logical_bytes": certificate_tree.logical_bytes,
        },
        "suffix_nodes": {
            "build_ms": suffix_build_ms,
            "nodes": len(suffix_index.nodes),
            "route_postings": suffix_index.route_postings,
            "route_runs": suffix_index.route_runs,
            "logical_bytes": suffix_index.logical_bytes,
        },
    }


def _summary(cases: list[dict[str, object]]) -> dict[str, object]:
    backend_names = (
        "certificate",
        "certificate_periodic",
        "suffix_flat",
        "suffix_periodic",
        "hybrid",
    )
    wins = {name: 0 for name in backend_names}
    for case in cases:
        winner = min(
            backend_names,
            key=lambda name: case["execution"][name]["elapsed_ms"],
        )
        wins[winner] += 1
    return {
        "case_count": len(cases),
        "execution_wins": wins,
        "all_backends_exactly_agreed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--bit-widths", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--trajectories",
        nargs="+",
        choices=TRAJECTORIES,
        default=list(TRAJECTORIES),
    )
    parser.add_argument("--clarities", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    profiler = profile_structure_case if args.structure_only else profile_case
    cases = [
        profiler(
            sequence_length,
            bit_width,
            trajectory,
            clarity,
            args.seed,
            args.repeats,
        )
        for sequence_length in args.sequence_lengths
        for bit_width in args.bit_widths
        for trajectory in args.trajectories
        for clarity in args.clarities
    ]
    report = {
        "configuration": {
            "sequence_lengths": args.sequence_lengths,
            "bit_widths": args.bit_widths,
            "trajectories": args.trajectories,
            "clarities": args.clarities,
            "seed": args.seed,
            "repeats": args.repeats,
            "structure_only": args.structure_only,
            "torch_version": torch.__version__,
        },
        "cases": cases,
        "summary": (
            {"case_count": len(cases)}
            if args.structure_only
            else _summary(cases)
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
