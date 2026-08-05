import pytest
import torch

from benchmarks.filtered_bitflip import (
    CodeOccurrenceIndex,
    generate_influence_events,
    semantic_bit_flips,
)
from benchmarks.filtered_bitflip_indexes import (
    DirectMatchIndex,
    DyadicMatchIndex,
    RollingHashMatchIndex,
    SuffixArrayMatchIndex,
    build_match_index,
)


INDEX_TYPES = (
    DirectMatchIndex,
    RollingHashMatchIndex,
    DyadicMatchIndex,
    SuffixArrayMatchIndex,
)


def _case(sequence_length, bit_width, seed):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randint(
            0,
            1 << bit_width,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        ),
        torch.randint(
            0,
            1 << bit_width,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        ),
    )


@pytest.mark.parametrize("sequence_length", range(1, 18))
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_all_match_indexes_equal_direct(sequence_length, bit_width):
    query, key = _case(
        sequence_length,
        bit_width,
        401 + sequence_length + bit_width,
    )
    direct = DirectMatchIndex(query, key)
    indexes = [constructor(query, key) for constructor in INDEX_TYPES[1:]]
    for query_position in range(1, sequence_length):
        for key_position in range(query_position):
            expected_left = direct.left_matches(query_position, key_position)
            expected_right = direct.right_matches(query_position, key_position)
            for index in indexes:
                assert index.left_matches(
                    query_position,
                    key_position,
                ) == expected_left
                assert index.right_matches(
                    query_position,
                    key_position,
                ) == expected_right


@pytest.mark.parametrize(
    "query,key",
    [
        ([0] * 16, [0] * 16),
        ([0, 1] * 8, [1, 0] * 8),
        ([0, 1, 2, 3] * 4, [0, 1, 2, 3] * 4),
        (list(range(16)), list(reversed(range(16)))),
    ],
)
def test_match_indexes_cover_degenerate_patterns(query, key):
    query_codes = torch.tensor(query, dtype=torch.uint8)
    key_codes = torch.tensor(key, dtype=torch.uint8)
    direct = DirectMatchIndex(query_codes, key_codes)
    for constructor in INDEX_TYPES[1:]:
        index = constructor(query_codes, key_codes)
        for query_position in range(1, len(query)):
            for key_position in range(query_position):
                assert index.left_matches(
                    query_position,
                    key_position,
                ) == direct.left_matches(query_position, key_position)
                assert index.right_matches(
                    query_position,
                    key_position,
                ) == direct.right_matches(query_position, key_position)


@pytest.mark.parametrize("constructor", INDEX_TYPES)
def test_indexed_event_generation_matches_pair_scan(constructor):
    query, key = _case(12, 4, 97)
    index = constructor(query, key)
    occurrence = CodeOccurrenceIndex.build(query, key)
    for flip in semantic_bit_flips(12, 4):
        expected = generate_influence_events(query, key, flip)
        indexed = generate_influence_events(
            query,
            key,
            flip,
            match_index=index,
            occurrence_index=occurrence,
        )
        assert indexed == expected


def test_index_factory_and_exactness_labels():
    query, key = _case(8, 3, 23)
    assert build_match_index("direct", query, key).exact
    assert build_match_index("dyadic", query, key).exact
    assert build_match_index("suffix_array", query, key).exact
    assert not build_match_index("rolling_hash", query, key).exact
    with pytest.raises(ValueError, match="unknown match index"):
        build_match_index("missing", query, key)
