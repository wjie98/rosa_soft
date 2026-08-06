import shutil

import pytest
import torch

from benchmarks.sam_bitflip import sam_bitflip
from benchmarks.sam_bitflip_native import (
    NativeSamBitflip,
    build_native_sam_bitflip,
)


@pytest.fixture(scope="session")
def native_sam_bitflip_library(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the native SAM core")
    output = tmp_path_factory.mktemp("sam_bitflip") / "librosa_sam_bitflip.so"
    return build_native_sam_bitflip(output, compiler=compiler)


def _assert_native_parity(query, key, bit_width, library):
    backend = NativeSamBitflip(library)
    native = backend.solve(query, key, bit_width)
    factorized = backend.solve_factorized(query, key, bit_width)
    factorized_routes, factorized_lengths = factorized.materialize()
    expected = sam_bitflip(query, key, bit_width, materialize_routes=True)
    assert torch.equal(native.flipped_routes, expected.flipped_routes)
    assert torch.equal(native.flipped_lengths, expected.flipped_lengths)
    assert torch.equal(factorized_routes, expected.flipped_routes)
    assert torch.equal(factorized_lengths, expected.flipped_lengths)
    assert torch.equal(
        factorized.base_routes.to(torch.int64), expected.base.routes
    )
    assert torch.equal(
        factorized.base_lengths.to(torch.int64), expected.base.lengths
    )


def _binary_codes(state, sequence_length):
    return torch.tensor(
        [(state >> position) & 1 for position in range(sequence_length)],
        dtype=torch.uint8,
    )


@pytest.mark.parametrize("sequence_length", [4, 5])
def test_native_sam_exhaustive_binary(
    sequence_length,
    native_sam_bitflip_library,
):
    for query_state in range(1 << sequence_length):
        query = _binary_codes(query_state, sequence_length)
        for key_state in range(1 << sequence_length):
            _assert_native_parity(
                query,
                _binary_codes(key_state, sequence_length),
                1,
                native_sam_bitflip_library,
            )


@pytest.mark.parametrize("sequence_length", [0, 1, 2, 3, 8, 17])
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_native_sam_matches_python_random(
    sequence_length,
    bit_width,
    native_sam_bitflip_library,
):
    for seed in range(4):
        generator = torch.Generator().manual_seed(
            1901 + 97 * sequence_length + 11 * bit_width + seed
        )
        query = torch.randint(
            1 << bit_width,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        )
        key = torch.randint(
            1 << bit_width,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        )
        _assert_native_parity(
            query,
            key,
            bit_width,
            native_sam_bitflip_library,
        )


@pytest.mark.parametrize(
    "query,key,bit_width",
    [
        ([0] * 32, [0] * 32, 1),
        ([0, 1] * 16, [1, 0] * 16, 1),
        ([0, 1, 2, 3] * 8, [3, 0, 1, 2] * 8, 2),
        ([0, 7, 0, 7] * 8, [7, 0, 7, 0] * 8, 3),
        (list(range(32)), list(reversed(range(32))), 5),
    ],
)
def test_native_sam_matches_python_structured(
    query,
    key,
    bit_width,
    native_sam_bitflip_library,
):
    _assert_native_parity(
        torch.tensor(query, dtype=torch.uint8),
        torch.tensor(key, dtype=torch.uint8),
        bit_width,
        native_sam_bitflip_library,
    )


def test_native_profile_reports_grouped_query_work(native_sam_bitflip_library):
    query = torch.zeros(64, dtype=torch.uint8)
    key = torch.zeros(64, dtype=torch.uint8)
    result = NativeSamBitflip(native_sam_bitflip_library).solve(query, key, 8)
    ungrouped_steps = 8 * sum(range(1, query.numel()))
    assert result.profile["query_replay_steps"] < ungrouped_steps
    assert result.profile["query_branches_merged"] > 0
    assert result.profile["arithmetic_hits"] > 0
    for field in (
        "forward_sam_build_ns",
        "reverse_sam_build_ns",
        "endpos_build_ns",
        "trace_build_ns",
        "query_solve_ns",
        "key_solve_ns",
        "total_solve_ns",
        "forward_sam_bytes",
        "reverse_sam_bytes",
        "endpos_bytes",
        "solver_working_bytes",
        "materialized_output_bytes",
    ):
        assert result.profile[field] > 0


def test_factorized_k_overrides_store_exact_from_winner(
    native_sam_bitflip_library,
):
    query = torch.tensor([0, 1, 2, 3] * 8, dtype=torch.uint8)
    key = torch.tensor([3, 0, 1, 2] * 8, dtype=torch.uint8)
    result = NativeSamBitflip(native_sam_bitflip_library).solve_factorized(
        query,
        key,
        2,
    )
    routes = result.base_routes.to(torch.int64)
    lengths = result.base_lengths.to(torch.int64)
    for local_flip in range(result.query_flip_count):
        key_position = local_flip // result.bit_width
        delete_start = int(result.key_delete_offsets[key_position])
        delete_stop = int(result.key_delete_offsets[key_position + 1])
        delete_routes = routes.clone()
        delete_lengths = lengths.clone()
        result._apply_changes(
            delete_routes,
            delete_lengths,
            result.key_delete_changes[delete_start:delete_stop],
        )
        start = int(result.key_override_offsets[local_flip])
        stop = int(result.key_override_offsets[local_flip + 1])
        for change in result.key_overrides[start:stop].tolist():
            row_start, row_stop = change[:2]
            for row in range(row_start, row_stop):
                offset = row - row_start
                assert change[2] + offset * change[3] == delete_lengths[row]
                assert change[4] + offset * change[5] == delete_routes[row]
                before = delete_lengths[row], delete_routes[row]
                after = (
                    change[6] + offset * change[7],
                    change[8] + offset * change[9],
                )
                assert after > before


def test_factorized_result_avoids_quadratic_materialization(
    native_sam_bitflip_library,
):
    sequence_length = 128
    query = torch.zeros(sequence_length, dtype=torch.uint8)
    key = torch.zeros_like(query)
    backend = NativeSamBitflip(native_sam_bitflip_library)
    factorized = backend.solve_factorized(query, key, 8)
    materialized_bytes = (
        2
        * 2
        * (sequence_length - 1)
        * 8
        * sequence_length
        * torch.tensor([], dtype=torch.int64).element_size()
    )
    assert factorized.descriptor_bytes < materialized_bytes // 8
    assert factorized.profile["materialized_output_bytes"] == 0
    assert (
        factorized.profile["factorized_result_bytes"]
        == factorized.descriptor_bytes
    )
    assert factorized.profile["query_change_descriptors"] == len(
        factorized.query_changes
    )
    assert factorized.profile["key_delete_descriptors"] == len(
        factorized.key_delete_changes
    )
    assert factorized.profile["key_override_descriptors"] == len(
        factorized.key_overrides
    )
