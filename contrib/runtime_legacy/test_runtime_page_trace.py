from pathlib import Path

import pytest
import torch

import rosa_soft
from benchmarks.runtime_page_trace import (
    RuntimePageTrace,
    collect_runtime_page_trace,
    load_runtime_page_trace,
    make_case,
    replay_cache_strategies,
    summarize_trace,
    validate_trace_against_runtime,
)


ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not rosa_soft.BUILD_CAPABILITIES.rosa_runtime,
    reason="RosaRuntime extension is unavailable",
)


@pytest.fixture(scope="module")
def trace_module():
    return load_runtime_page_trace()


@pytest.mark.parametrize("pattern,bits", [("random", 4), ("skewed", 8), ("copied97", 8)])
def test_cpp_trace_matches_production_routes_and_payload(
    trace_module,
    pattern,
    bits,
):
    query, key, payload = make_case(513, bits, pattern, 17, ROOT / "README.md")
    trace = collect_runtime_page_trace(
        query,
        key,
        payload,
        qk_bits=bits,
        chunk_size=73,
        module=trace_module,
    )
    validate_trace_against_runtime(
        trace,
        query,
        key,
        payload,
        qk_bits=bits,
        payload_bits=8,
        chunk_size=73,
    )


def test_trace_covers_production_transition_indexes(trace_module):
    query, key, payload = make_case(1024, 4, "random", 123, ROOT / "README.md")
    trace = collect_runtime_page_trace(
        query,
        key,
        payload,
        qk_bits=4,
        chunk_size=127,
        module=trace_module,
    )
    regions = set(map(int, trace.records[:, 1].unique().tolist()))

    assert trace.complexity["transition_index_activations"] > 0
    assert {4, 5, 6}.issubset(regions)
    assert int(trace.records[:, 6].sum()) == trace.summary["canonical_page_touches"]


def test_trace_covers_activated_latest_end_lct(trace_module):
    query, key, payload = make_case(
        4096,
        8,
        "periodic64",
        123,
        ROOT / "README.md",
    )
    trace = collect_runtime_page_trace(
        query,
        key,
        payload,
        qk_bits=8,
        chunk_size=1024,
        module=trace_module,
    )
    summary = summarize_trace(trace)

    assert trace.complexity["latest_tree_activations"] == 1
    assert trace.complexity["latest_tree_rotations"] > 0
    assert summary["touches_by_region"]["latest_end_tree"] > 0
    assert summary["unique_pages_by_region"]["latest_end_tree"] > 1


def test_one_trace_replays_multiple_page_sizes_and_policies(trace_module):
    query, key, payload = make_case(768, 4, "random", 9, ROOT / "README.md")
    trace = collect_runtime_page_trace(
        query,
        key,
        payload,
        qk_bits=4,
        chunk_size=128,
        module=trace_module,
    )
    records_before = trace.records.clone()
    reports = []
    for page_bytes in (4096, 16384):
        reports.extend(
            replay_cache_strategies(
                trace,
                page_bytes=page_bytes,
                cache_bytes=65536,
            )
        )

    assert torch.equal(trace.records, records_before)
    assert {row["strategy"] for row in reports} == {
        "lru",
        "slru",
        "region_lru",
        "structural_pin_lru",
        "append_bypass_lru",
    }
    assert all(row["accesses"] == row["hits"] + row["misses"] for row in reports)


def test_lru_replay_has_known_miss_sequence():
    records = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [2, 0, 0, 0, 0, 0, 1],
            [3, 0, 0, 2, 1, 2, 2],
            [4, 0, 0, 1, 0, 0, 1],
            [5, 0, 0, 3, 0, 0, 1],
        ],
        dtype=torch.int64,
    )
    trace = RuntimePageTrace(
        records=records,
        routes=torch.full((6,), -1, dtype=torch.int64),
        output=torch.zeros(6, dtype=torch.uint8),
        summary={
            "states": 0,
            "edges": 0,
            "automaton_logical_bytes": 0,
            "canonical_page_touches": 7,
            "rle_records": 6,
        },
        complexity={},
        canonical_page_bytes=4096,
    )
    [report] = replay_cache_strategies(
        trace,
        page_bytes=4096,
        cache_bytes=8192,
        strategies=["lru"],
    )

    assert report["accesses"] == 7
    assert report["misses"] == 5
    assert report["hits"] == 2
    assert report["dirty_evictions"] == 1


def _synthetic_trace(records: list[list[int]], tokens: int) -> RuntimePageTrace:
    tensor = torch.tensor(records, dtype=torch.int64)
    return RuntimePageTrace(
        records=tensor,
        routes=torch.full((tokens,), -1, dtype=torch.int64),
        output=torch.zeros(tokens, dtype=torch.uint8),
        summary={
            "states": 0,
            "edges": 0,
            "automaton_logical_bytes": 0,
            "canonical_page_touches": int(tensor[:, 6].sum()),
            "rle_records": len(records),
        },
        complexity={},
        canonical_page_bytes=4096,
    )


def test_single_page_slru_never_exceeds_capacity():
    trace = _synthetic_trace(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 2],
            [2, 0, 0, 0, 0, 0, 1],
        ],
        3,
    )
    [report] = replay_cache_strategies(
        trace,
        page_bytes=4096,
        cache_bytes=4096,
        strategies=["slru"],
    )

    assert report["resident_pages_at_end"] == 1
    assert report["misses"] == 3
    assert report["hits"] == 1
    assert report["evictions"] == 2


def test_fully_pinned_cache_counts_unadmitted_repeats_as_misses():
    trace = _synthetic_trace(
        [
            [0, 6, 0, 0, 0, 0, 1],
            [1, 2, 0, 1, 0, 0, 3],
        ],
        2,
    )
    [report] = replay_cache_strategies(
        trace,
        page_bytes=4096,
        cache_bytes=4096,
        strategies=["structural_pin_lru"],
    )

    assert report["preloaded_pages"] == 1
    assert report["misses"] == 3
    assert report["hits"] == 1


def test_zero_capacity_region_counts_every_repeat_as_a_miss():
    trace = _synthetic_trace(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [1, 8, 0, 0, 0, 0, 3],
        ],
        2,
    )
    [report] = replay_cache_strategies(
        trace,
        page_bytes=4096,
        cache_bytes=4096,
        strategies=["region_lru"],
    )

    assert report["misses"] == 4
    assert report["hits"] == 0
