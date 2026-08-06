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
    native = NativeSamBitflip(library).solve(query, key, bit_width)
    expected = sam_bitflip(query, key, bit_width, materialize_routes=True)
    assert torch.equal(native.flipped_routes, expected.flipped_routes)
    assert torch.equal(native.flipped_lengths, expected.flipped_lengths)


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
