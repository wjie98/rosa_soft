import pytest
import torch

from benchmarks.filtered_bitflip import brute_force_bitflip
from benchmarks.filtered_bitflip_indexes import (
    DirectMatchIndex,
    DyadicMatchIndex,
    SuffixArrayMatchIndex,
)
from benchmarks.filtered_bitflip_winner import (
    RleWinnerIndex,
    SegmentWinnerIndex,
    WinnerGeometry,
    winner_filtered_bitflip,
)


def _random_codes(sequence_length, bit_width, seed):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randint(
        0,
        1 << bit_width,
        (sequence_length,),
        generator=generator,
        dtype=torch.uint8,
    )
    key = torch.randint(
        0,
        1 << bit_width,
        (sequence_length,),
        generator=generator,
        dtype=torch.uint8,
    )
    return query, key


def _brute_ranges(codes, start, stop, target, mode):
    selected = []
    run_start = None
    for position in range(start, stop + 1):
        keep = False
        if position < stop:
            keep = target > codes[position] if mode == "greater" else codes[position] == target
        if keep and run_start is None:
            run_start = position
        if not keep and run_start is not None:
            selected.append((run_start, position))
            run_start = None
    return tuple(selected)


def test_winner_indexes_match_elementwise_queries():
    routes = torch.tensor([0, 0, 1, 1, 3, 2, 5, 4, 7], dtype=torch.int64)
    lengths = torch.tensor([0, 0, 1, 1, 2, 1, 3, 2, 1], dtype=torch.int64)
    geometry = WinnerGeometry.build(routes, lengths)
    indexes = (RleWinnerIndex(geometry), SegmentWinnerIndex(geometry))
    for start in range(len(geometry.codes) + 1):
        for stop in range(start, len(geometry.codes) + 1):
            for target in range(min(geometry.codes) - 1, max(geometry.codes) + 2):
                for mode in ("greater", "equal"):
                    expected = _brute_ranges(
                        geometry.codes,
                        start,
                        stop,
                        target,
                        mode,
                    )
                    for index in indexes:
                        result = getattr(index, mode)(start, stop, target)
                        assert result.ranges == expected


@pytest.mark.parametrize("winner_backend", ["rle", "segment"])
@pytest.mark.parametrize(
    "match_constructor",
    [DirectMatchIndex, DyadicMatchIndex, SuffixArrayMatchIndex],
)
@pytest.mark.parametrize("sequence_length", [2, 4, 7, 12])
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_winner_filtered_bitflip_matches_full_rerun(
    winner_backend,
    match_constructor,
    sequence_length,
    bit_width,
):
    query, key = _random_codes(
        sequence_length,
        bit_width,
        701 + sequence_length + bit_width,
    )
    expected = brute_force_bitflip(query, key, bit_width)
    result = winner_filtered_bitflip(
        query,
        key,
        bit_width,
        match_index=match_constructor(query, key),
        winner_backend=winner_backend,
    )
    torch.testing.assert_close(result.flipped_routes, expected.flipped_routes)
    torch.testing.assert_close(result.flipped_lengths, expected.flipped_lengths)


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
@pytest.mark.parametrize("pattern", ["all_match", "alternating", "periodic"])
def test_winner_filter_handles_degenerate_patterns(bit_width, pattern):
    sequence_length = 16
    if pattern == "all_match":
        query = torch.zeros(sequence_length, dtype=torch.uint8)
        key = query.clone()
    elif pattern == "alternating":
        query = (torch.arange(sequence_length) % 2).to(torch.uint8)
        key = query.roll(1)
    else:
        modulus = min(1 << bit_width, 4)
        query = (torch.arange(sequence_length) % modulus).to(torch.uint8)
        key = query.clone()
    expected = brute_force_bitflip(query, key, bit_width)
    result = winner_filtered_bitflip(query, key, bit_width)
    torch.testing.assert_close(result.flipped_routes, expected.flipped_routes)
    torch.testing.assert_close(result.flipped_lengths, expected.flipped_lengths)


def test_winner_filter_preserves_bit_gradient():
    query, key = _random_codes(10, 3, 91)
    generator = torch.Generator().manual_seed(92)
    value = torch.randn(10, 5, generator=generator, dtype=torch.float64)
    grad_output = torch.randn(10, 5, generator=generator, dtype=torch.float64)
    expected = brute_force_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    result = winner_filtered_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    torch.testing.assert_close(result.bit_gradient, expected.bit_gradient)
    assert result.final_changed_rows == int(
        (expected.flipped_routes != expected.base.routes).sum()
    )


def test_winner_filter_rejects_unknown_backend():
    query, key = _random_codes(4, 2, 12)
    with pytest.raises(ValueError, match="unknown winner backend"):
        winner_filtered_bitflip(query, key, 2, winner_backend="missing")
