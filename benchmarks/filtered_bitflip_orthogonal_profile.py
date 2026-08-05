"""Profile exact code-partitioned orthogonal LCE-cell indexes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from benchmarks.filtered_bitflip import (  # noqa: E402
    CodeOccurrenceIndex,
    semantic_bit_flips,
)
from benchmarks.filtered_bitflip_monotone import (  # noqa: E402
    SemanticLceGeometry,
    build_query_create_lce_cells,
)
from benchmarks.filtered_bitflip_monotone_profile import (  # noqa: E402
    make_cases,
)
from benchmarks.filtered_bitflip_orthogonal import (  # noqa: E402
    CausalQueryCreateOrthogonalBuilder,
)
from benchmarks.filtered_bitflip_orthogonal_kd import (  # noqa: E402
    StaticQueryCreateKdBuilder,
)


def _median_scan_ms(function, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, statistics.median(samples)


def _median_builder_ms(factory, flips, repeats: int):
    build_samples = []
    query_samples = []
    builder = None
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        builder = factory()
        build_samples.append((time.perf_counter() - start) * 1e3)
        start = time.perf_counter()
        result = tuple(builder.build(flip) for flip in flips)
        query_samples.append((time.perf_counter() - start) * 1e3)
    return (
        builder,
        result,
        statistics.median(build_samples),
        statistics.median(query_samples),
    )


def _summary(batches) -> tuple[int, int, int]:
    return (
        sum(batch.occurrence_count for batch in batches),
        sum(len(batch.cells) for batch in batches),
        sum(len(batch.dominant_query_create_events()) for batch in batches),
    )


def _cell_signature(batch):
    cells = []
    for cell in batch.cells:
        aggregate = getattr(cell, "aggregate", None)
        if aggregate is None:
            positions = cell.varying_positions
            count = len(positions)
            minimum = min(positions)
            maximum = max(positions)
        else:
            count = aggregate.count
            minimum = aggregate.min_position
            maximum = aggregate.max_position
        cells.append(
            (
                cell.left_matches,
                cell.right_matches,
                count,
                minimum,
                maximum,
            )
        )
    events = tuple(
        sorted(
            batch.dominant_query_create_events(),
            key=lambda event: event.key_position,
        )
    )
    return batch.flip, tuple(sorted(cells)), events


def _require_exact(expected, actual, backend: str) -> None:
    expected_signatures = tuple(_cell_signature(batch) for batch in expected)
    actual_signatures = tuple(_cell_signature(batch) for batch in actual)
    if actual_signatures != expected_signatures:
        raise RuntimeError(f"{backend} changed an exact LCE cell partition")


def profile_case(
    name,
    bit_width,
    query,
    key,
    repeats,
    include_boxes,
):
    sequence_length = query.numel()
    geometry = SemanticLceGeometry(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    flips = tuple(
        flip
        for flip in semantic_bit_flips(sequence_length, bit_width)
        if flip.source == "query"
    )

    def scan():
        return tuple(
            build_query_create_lce_cells(
                query,
                key,
                flip,
                geometry=geometry,
                occurrence_index=occurrences,
            )
            for flip in flips
        )

    scan_batches, scan_ms = _median_scan_ms(scan, repeats)
    dynamic, dynamic_batches, dynamic_build_ms, dynamic_query_ms = (
        _median_builder_ms(
            lambda: CausalQueryCreateOrthogonalBuilder(
                query,
                key,
                geometry=geometry,
                occurrence_index=occurrences,
                enumeration_backend="tree",
            ),
            flips,
            repeats,
        )
    )
    kd, kd_batches, kd_build_ms, kd_query_ms = _median_builder_ms(
        lambda: StaticQueryCreateKdBuilder(
            query,
            key,
            geometry=geometry,
            occurrence_index=occurrences,
        ),
        flips,
        repeats,
    )
    expected = _summary(scan_batches)
    _require_exact(scan_batches, dynamic_batches, "dynamic range tree")
    _require_exact(scan_batches, kd_batches, "static kd")

    result = {
        "case": name,
        "sequence_length": sequence_length,
        "bit_width": bit_width,
        "query_flips": len(flips),
        "create_occurrences": expected[0],
        "lce_cells": expected[1],
        "scan": {
            "elapsed_ms": scan_ms,
            "occurrence_visits": expected[0],
        },
        "dynamic_range_tree": {
            "build_ms": dynamic_build_ms,
            "query_ms": dynamic_query_ms,
            "total_ms": dynamic_build_ms + dynamic_query_ms,
            "speedup_over_scan": scan_ms
            / max(dynamic_build_ms + dynamic_query_ms, 1e-12),
            "logical_bytes": dynamic.logical_bytes,
            "point_activations": dynamic.index.point_activations,
            "tree_nodes_visited": sum(
                batch.tree_nodes_visited for batch in dynamic_batches
            ),
            "rectangle_queries": sum(
                batch.rectangle_queries for batch in dynamic_batches
            ),
        },
        "static_kd": {
            "build_ms": kd_build_ms,
            "query_ms": kd_query_ms,
            "total_ms": kd_build_ms + kd_query_ms,
            "speedup_over_scan": scan_ms
            / max(kd_build_ms + kd_query_ms, 1e-12),
            "logical_bytes": kd.logical_bytes,
            "tree_nodes": kd.index.node_count,
            "tree_nodes_visited": sum(
                batch.tree_nodes_visited for batch in kd_batches
            ),
            "point_probes": sum(
                batch.point_probes for batch in kd_batches
            ),
        },
    }
    if include_boxes:
        boxes, box_batches, box_build_ms, box_query_ms = _median_builder_ms(
            lambda: CausalQueryCreateOrthogonalBuilder(
                query,
                key,
                geometry=geometry,
                occurrence_index=occurrences,
                enumeration_backend="boxes",
            ),
            flips,
            repeats,
        )
        _require_exact(scan_batches, box_batches, "box recursion")
        result["box_recursion"] = {
            "build_ms": box_build_ms,
            "query_ms": box_query_ms,
            "total_ms": box_build_ms + box_query_ms,
            "logical_bytes": boxes.logical_bytes,
            "rectangle_queries": sum(
                batch.rectangle_queries for batch in box_batches
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024],
    )
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--include-boxes", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    results = [
        profile_case(
            name,
            bit_width,
            query,
            key,
            args.repeats,
            args.include_boxes,
        )
        for sequence_length in args.sequence_lengths
        for name, bit_width, query, key in make_cases(
            sequence_length, args.seed
        )
    ]
    payload = {
        "seed": args.seed,
        "repeats": args.repeats,
        "include_boxes": args.include_boxes,
        "validation": "exact cells, extrema, and dominant events",
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
