import itertools

import pytest
import torch

from benchmarks.filtered_bitflip import brute_force_bitflip
from benchmarks.filtered_bitflip_hybrid import hybrid_filtered_bitflip
from benchmarks.filtered_bitflip_profile import make_training_codes


def _random_codes(sequence_length: int, bit_width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    alphabet = 1 << bit_width
    return (
        torch.randint(
            alphabet,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        ),
        torch.randint(
            alphabet,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        ),
    )


def _assert_matches_brute(query: torch.Tensor, key: torch.Tensor, bit_width: int):
    expected = brute_force_bitflip(query, key, bit_width)
    actual = hybrid_filtered_bitflip(query, key, bit_width)
    assert torch.equal(actual.base_routes, expected.base.routes)
    assert torch.equal(actual.base_lengths, expected.base.lengths)
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [7, 37])
def test_hybrid_bitflip_matches_brute_random(bit_width: int, seed: int):
    query, key = _random_codes(9, bit_width, seed)
    _assert_matches_brute(query, key, bit_width)


@pytest.mark.parametrize(
    "query,key,bit_width",
    [
        ([0] * 10, [0] * 10, 4),
        ([0, 1] * 5, [1, 0] * 5, 1),
        ([0, 3, 1, 2] * 3, [2, 0, 3, 1] * 3, 2),
    ],
)
def test_hybrid_bitflip_matches_brute_structured(query, key, bit_width):
    _assert_matches_brute(
        torch.tensor(query, dtype=torch.uint8),
        torch.tensor(key, dtype=torch.uint8),
        bit_width,
    )


def test_hybrid_bitflip_matches_exhaustive_d1_t4():
    states = tuple(itertools.product(range(2), repeat=4))
    for query_values in states:
        query = torch.tensor(query_values, dtype=torch.uint8)
        for key_values in states:
            key = torch.tensor(key_values, dtype=torch.uint8)
            _assert_matches_brute(query, key, 1)


def test_hybrid_dispatch_uses_structure_instead_of_clarity_label():
    early_query, early_key = make_training_codes(64, 8, "shift_random", 0.0, 17)
    clear_query, clear_key = make_training_codes(64, 8, "shift_random", 1.0, 17)
    early = hybrid_filtered_bitflip(early_query, early_key, 8)
    clear = hybrid_filtered_bitflip(clear_query, clear_key, 8)
    assert early.certificate_flips == 0
    assert early.suffix_node_flips > 0
    assert clear.certificate_flips > 0
    assert clear.estimated_overlay_entries * 2 < clear.event_cells


def test_hybrid_bit_gradient_matches_brute():
    query, key = _random_codes(8, 3, 113)
    generator = torch.Generator().manual_seed(127)
    value = torch.randn(8, 5, generator=generator)
    grad_output = torch.randn(8, 5, generator=generator)
    expected = brute_force_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    actual = hybrid_filtered_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.bit_gradient, expected.bit_gradient)
