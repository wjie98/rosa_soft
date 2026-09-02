import itertools

import pytest
import torch

import rosa_soft


pytestmark = pytest.mark.skipif(
    not rosa_soft.BUILD_CAPABILITIES.rosa_sam,
    reason="RosaSam extension is unavailable",
)


def _dp_routes_dense(query: torch.Tensor, key: torch.Tensor, bits: int):
    """Independent O(B*H*T^2) longest/latest suffix landmark."""

    assert query.ndim == 3
    assert query.shape == key.shape
    batch, tokens, heads = query.shape
    mask = (1 << bits) - 1
    routes = torch.full((batch, tokens, heads), -1, dtype=torch.int64)
    for batch_index in range(batch):
        for head in range(heads):
            previous = [0] * tokens
            for query_end in range(tokens):
                current = [0] * tokens
                best_length = 0
                best_key_end = -1
                for key_end in range(query_end + 1):
                    query_symbol = int(query[batch_index, query_end, head]) & mask
                    key_symbol = int(key[batch_index, key_end, head]) & mask
                    if query_symbol == key_symbol:
                        current[key_end] = 1
                        if query_end > 0 and key_end > 0:
                            current[key_end] += previous[key_end - 1]
                    if (
                        key_end < query_end
                        and current[key_end] >= best_length
                        and current[key_end] > 0
                    ):
                        best_length = current[key_end]
                        best_key_end = key_end
                routes[batch_index, query_end, head] = best_key_end
                previous = current
    return routes


def _dp_routes_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor,
    bits: int,
):
    routes = torch.empty_like(query, dtype=torch.int64)
    for begin, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        dense = _dp_routes_dense(
            query[begin:end].unsqueeze(0),
            key[begin:end].unsqueeze(0),
            bits,
        )
        routes[begin:end] = dense.squeeze(0)
    return routes


def _random_codes(shape, bits, seed):
    generator = torch.Generator().manual_seed(seed)
    if bits == 32:
        return torch.randint(
            -(1 << 31),
            1 << 31,
            shape,
            dtype=torch.int32,
            generator=generator,
        )
    return torch.randint(
        0,
        1 << bits,
        shape,
        dtype=torch.int32,
        generator=generator,
    )


def test_exhaustive_binary_length_five_matches_dp_landmark():
    patterns = torch.tensor(
        list(itertools.product((0, 1), repeat=5)),
        dtype=torch.int32,
    ).unsqueeze(-1)
    query = patterns.repeat_interleave(patterns.size(0), dim=0)
    key = patterns.repeat(patterns.size(0), 1, 1)
    expected = _dp_routes_dense(query, key, bits=1)

    actual = rosa_soft.RosaSam(1, 1).update_packed(query, key)

    assert torch.equal(actual, expected)


def test_exhaustive_ternary_length_four_matches_dp_landmark():
    patterns = torch.tensor(
        list(itertools.product((0, 1, 2), repeat=4)),
        dtype=torch.int32,
    ).unsqueeze(-1)
    query = patterns.repeat_interleave(patterns.size(0), dim=0)
    key = patterns.repeat(patterns.size(0), 1, 1)
    expected = _dp_routes_dense(query, key, bits=2)

    actual = rosa_soft.RosaSam(1, 2).update_packed(query, key)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16, 32])
def test_random_symbols_match_dp_landmark(bits):
    query = _random_codes((3, 29, 3), bits, seed=100 + bits)
    key = _random_codes((3, 29, 3), bits, seed=200 + bits)
    expected = _dp_routes_dense(query, key, bits)

    actual = rosa_soft.RosaSam(3, bits).update_packed(query, key)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "key_values",
    [
        [0] * 33,
        [index % 2 for index in range(33)],
        [index % 4 for index in range(33)],
        [0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1] * 3,
    ],
)
def test_structured_clone_and_periodic_cases_match_dp_landmark(key_values):
    key = torch.tensor(key_values, dtype=torch.int32).view(1, -1, 1)
    query = torch.roll(key, shifts=5, dims=1)
    expected = _dp_routes_dense(query, key, bits=3)

    actual = rosa_soft.RosaSam(1, 3).update_packed(query, key)

    assert torch.equal(actual, expected)


def test_equal_length_tie_selects_latest_key_end():
    query = torch.tensor([2, 0, 1, 3], dtype=torch.int32).view(1, -1, 1)
    key = torch.tensor([3, 3, 2, 0], dtype=torch.int32).view(1, -1, 1)
    expected = _dp_routes_dense(query, key, bits=2)
    assert expected[0, 3, 0].item() == 1

    actual = rosa_soft.RosaSam(1, 2).update_packed(query, key)

    assert torch.equal(actual, expected)


def test_null_match_returns_minus_one_and_masks_placeholder_value():
    query = torch.ones(1, 9, 2, 3)
    key = -torch.ones_like(query)
    value = torch.ones(1, 9, 1, 4)

    output, matched_key_end = rosa_soft.rosa_hard_reference(query, key, value)

    assert torch.equal(matched_key_end, torch.full_like(matched_key_end, -1))
    assert torch.equal(output, torch.zeros_like(output))


def test_chunked_updates_equal_one_shot_and_dp_landmark():
    bits = 5
    query = _random_codes((2, 47, 4), bits, seed=7)
    key = _random_codes((2, 47, 4), bits, seed=8)
    expected = _dp_routes_dense(query, key, bits)
    one_shot = rosa_soft.RosaSam(4, bits).update_packed(query, key)

    chunked_sam = rosa_soft.RosaSam(4, bits)
    chunks = []
    for begin, end in zip((0, 1, 9, 10, 31), (1, 9, 10, 31, 47)):
        chunks.append(
            chunked_sam.update_packed(
                query[:, begin:end],
                key[:, begin:end],
            )
        )
    chunked = torch.cat(chunks, dim=1)

    assert torch.equal(one_shot, expected)
    assert torch.equal(chunked, expected)


def test_packed_varlen_with_empty_sequences_matches_dp_landmark():
    bits = 6
    offsets = torch.tensor([0, 0, 7, 7, 18, 23], dtype=torch.int32)
    query = _random_codes((23, 3), bits, seed=9)
    key = _random_codes((23, 3), bits, seed=10)
    expected = _dp_routes_varlen(query, key, offsets, bits)

    actual = rosa_soft.RosaSam(3, bits).update_packed(
        query,
        key,
        cu_seqlens=offsets,
    )

    assert torch.equal(actual, expected)


def test_reset_discards_all_query_and_key_history():
    bits = 4
    query = _random_codes((2, 21, 2), bits, seed=11)
    key = _random_codes((2, 21, 2), bits, seed=12)
    sam = rosa_soft.RosaSam(2, bits)
    first = sam.update_packed(query, key)
    sam.reset()
    second = sam.update_packed(query, key)

    assert torch.equal(first, second)


def test_bits_above_symbol_width_are_ignored():
    query = torch.tensor([1, 2, 3, 0], dtype=torch.int32).view(1, -1, 1)
    key = query | (torch.tensor(0x5A, dtype=torch.int32) << 8)
    expected = _dp_routes_dense(query, key, bits=8)

    actual = rosa_soft.RosaSam(1, 8).update_packed(query, key)

    assert torch.equal(actual, expected)


def test_malformed_symbol_and_logit_ranks_raise_public_value_error():
    sam = rosa_soft.RosaSam(1, 3)
    with pytest.raises(ValueError, match="packed query and key"):
        sam.update_packed(torch.tensor(0, dtype=torch.int32), torch.tensor(0))
    with pytest.raises(ValueError, match="query and key must have shape"):
        sam.update(torch.tensor(0.0), torch.tensor(0.0))


def test_hard_reference_gathers_successor_value_with_grouped_heads():
    generator = torch.Generator().manual_seed(13)
    query = torch.randn(2, 17, 4, 12, generator=generator)
    key = torch.randn(2, 17, 4, 12, generator=generator)
    value = torch.randn(2, 17, 2, 7, generator=generator)
    packed_query = rosa_soft.sam._pack_sign_bits(query)
    packed_key = rosa_soft.sam._pack_sign_bits(key)
    expected_routes = _dp_routes_dense(packed_query, packed_key, bits=12)

    output, routes = rosa_soft.rosa_hard_reference(query, key, value)

    hard_value = torch.where(value > 0, 1.0, -1.0).repeat_interleave(2, dim=2)
    expected_output = torch.zeros_like(output)
    for batch in range(query.size(0)):
        for token in range(query.size(1)):
            for head in range(query.size(2)):
                end = expected_routes[batch, token, head].item()
                if end >= 0:
                    expected_output[batch, token, head] = hard_value[
                        batch,
                        end + 1,
                        head,
                    ]

    assert torch.equal(routes.cpu(), expected_routes)
    assert torch.equal(output, expected_output)


def test_hard_reference_is_independent_of_surrogate_window():
    generator = torch.Generator().manual_seed(14)
    query = torch.randn(1, 15, 2, 3, generator=generator)
    key = torch.randn(1, 15, 2, 3, generator=generator)
    value = torch.randn(1, 15, 1, 5, generator=generator)
    expected, _ = rosa_soft.rosa_hard_reference(query, key, value)

    short = rosa_soft.rosa_soft_reference(
        query,
        key,
        value,
        max_suffix_length=1,
    )
    long = rosa_soft.rosa_soft_reference(
        query,
        key,
        value,
        max_suffix_length=15,
    )

    assert torch.equal(short, expected)
    assert torch.equal(long, expected)


def test_varlen_hard_reference_gathers_local_successor_values():
    generator = torch.Generator().manual_seed(15)
    offsets = torch.tensor([0, 0, 6, 15, 15], dtype=torch.int32)
    query = torch.randn(15, 4, 9, generator=generator)
    key = torch.randn(15, 4, 9, generator=generator)
    value = torch.randn(15, 2, 5, generator=generator)
    packed_query = rosa_soft.sam._pack_sign_bits(query)
    packed_key = rosa_soft.sam._pack_sign_bits(key)
    expected_routes = _dp_routes_varlen(
        packed_query,
        packed_key,
        offsets,
        bits=9,
    )

    output, routes = rosa_soft.rosa_hard_varlen_reference(
        query,
        key,
        value,
        offsets,
    )

    expected_output = torch.zeros_like(output)
    hard_value = torch.where(value > 0, 1.0, -1.0).repeat_interleave(2, dim=1)
    for begin, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        for token in range(begin, end):
            for head in range(query.size(1)):
                route_end = expected_routes[token, head].item()
                if route_end >= 0:
                    expected_output[token, head] = hard_value[
                        begin + route_end + 1,
                        head,
                    ]

    assert torch.equal(routes.cpu(), expected_routes)
    assert torch.equal(output, expected_output)
