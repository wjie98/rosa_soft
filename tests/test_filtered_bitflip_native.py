import shutil

import pytest
import torch

from benchmarks.filtered_bitflip import brute_force_bitflip
from benchmarks.filtered_bitflip_compact import compact_filtered_bitflip
from benchmarks.filtered_bitflip_hybrid import hybrid_filtered_bitflip
from benchmarks.filtered_bitflip_indexes import (
    DirectMatchIndex,
    LibsaisSuffixArrayMatchIndex,
    _SuffixArrayPair,
)
from benchmarks.filtered_bitflip_native import LibsaisBackend, build_libsais
from benchmarks.filtered_bitflip_suffix_nodes import (
    SuffixNodeReplacementIndex,
    suffix_node_filtered_bitflip,
)


@pytest.fixture(scope="session")
def libsais_library(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the optional libsais backend")
    output = tmp_path_factory.mktemp("libsais") / "librosa_libsais.so"
    return build_libsais(output, compiler=compiler)


@pytest.mark.parametrize(
    "text",
    [
        (),
        (0,),
        (3, 1, 4, 1, 5, 0),
        tuple(range(258)),
        (257, 0, 257, 1, 256, 0),
    ],
)
def test_libsais_matches_python_suffix_array(text, libsais_library):
    native = LibsaisBackend(libsais_library)
    suffix_array, lcp = native.suffix_array_lcp(text)
    expected_suffix_array = tuple(sorted(range(len(text)), key=lambda i: text[i:]))
    inverse = [0] * len(text)
    for rank, position in enumerate(expected_suffix_array):
        inverse[position] = rank
    expected_lcp = _SuffixArrayPair._build_lcp(
        text,
        expected_suffix_array,
        inverse,
    )
    assert suffix_array == expected_suffix_array
    assert lcp == expected_lcp


@pytest.mark.parametrize("bit_width", [1, 4, 8])
def test_native_match_index_matches_direct(bit_width, libsais_library):
    generator = torch.Generator().manual_seed(1200 + bit_width)
    query = torch.randint(
        1 << bit_width, (24,), dtype=torch.uint8, generator=generator
    )
    key = torch.randint(
        1 << bit_width, (24,), dtype=torch.uint8, generator=generator
    )
    if bit_width == 8:
        query[:3] = torch.tensor([253, 254, 255], dtype=torch.uint8)
        key[:3] = torch.tensor([255, 254, 253], dtype=torch.uint8)
    direct = DirectMatchIndex(query, key)
    native = LibsaisSuffixArrayMatchIndex(query, key, library_path=libsais_library)
    for query_position in range(1, query.numel()):
        for key_position in range(query_position):
            assert native.left_matches(
                query_position, key_position
            ) == direct.left_matches(query_position, key_position)
            assert native.right_matches(
                query_position, key_position
            ) == direct.right_matches(query_position, key_position)


def test_native_suffix_node_index_matches_python(libsais_library):
    query = torch.tensor([0, 1, 0, 1] * 4, dtype=torch.uint8)
    key = torch.tensor([1, 0, 1, 0] * 4, dtype=torch.uint8)
    expected = SuffixNodeReplacementIndex(query, key)
    actual = SuffixNodeReplacementIndex(
        query,
        key,
        suffix_backend="libsais",
        library_path=libsais_library,
    )
    assert actual.nodes == expected.nodes
    assert actual.query_nodes == expected.query_nodes
    for query_index in range(query.numel()):
        for excluded in (set(), {1, 3, 5, 7}, set(range(query_index + 1))):
            assert actual.best_excluding(
                query_index, excluded, backend="periodic"
            ) == expected.best_excluding(
                query_index, excluded, backend="periodic"
            )


@pytest.mark.parametrize("solver", ["suffix", "hybrid", "compact"])
def test_native_full_solver_matches_brute(solver, libsais_library):
    generator = torch.Generator().manual_seed(1291)
    query = torch.randint(16, (11,), dtype=torch.uint8, generator=generator)
    key = torch.randint(16, (11,), dtype=torch.uint8, generator=generator)
    expected = brute_force_bitflip(query, key, 4)
    arguments = {
        "suffix_backend": "libsais",
        "library_path": libsais_library,
    }
    if solver == "suffix":
        actual = suffix_node_filtered_bitflip(
            query, key, 4, posting_backend="periodic", **arguments
        )
    elif solver == "hybrid":
        actual = hybrid_filtered_bitflip(query, key, 4, **arguments)
    else:
        actual = compact_filtered_bitflip(
            query, key, 4, materialize_routes=True, **arguments
        )
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)
