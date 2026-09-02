import pytest
import torch

import rosa_soft
from benchmarks.indexed_hard_forward import (
    INDEXED_METHODS,
    build_occurrence_index,
    diagonal_hard_forward_from_packed,
    indexed_hard_forward_from_packed,
    load_indexed_hard_forward,
    pack_sign_bits,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="CUDA RosaSoft extension is unavailable",
)


@pytest.fixture(scope="module")
def indexed_module():
    return load_indexed_hard_forward()


def _unpack_symbols(symbols, bits, *, device=None, dtype=torch.float32):
    unpacked = torch.stack(
        [
            ((symbols.to(torch.int64) >> bit) & 1)
            .to(dtype)
            .mul(2)
            .sub(1)
            for bit in range(bits)
        ],
        dim=-1,
    )
    return unpacked.to(device=device)


def _bounded_reference(query, key, max_suffix_length):
    batch, tokens, heads = query.shape
    routes = torch.zeros(batch, heads, tokens, dtype=torch.int32)
    lengths = torch.zeros_like(routes)
    for batch_index in range(batch):
        for head in range(heads):
            for query_end in range(tokens):
                best_length = 0
                best_route = 0
                for key_end in range(query_end):
                    limit = min(
                        max_suffix_length,
                        query_end + 1,
                        key_end + 1,
                    )
                    length = 0
                    while (
                        length < limit
                        and query[batch_index, query_end - length, head]
                        == key[batch_index, key_end - length, head]
                    ):
                        length += 1
                    route = key_end + 1
                    if length > best_length or (
                        length > 0
                        and length == best_length
                        and route > best_route
                    ):
                        best_length = length
                        best_route = route
                routes[batch_index, head, query_end] = best_route
                lengths[batch_index, head, query_end] = best_length
    return routes, lengths


def _make_packed_case(bits, tokens, *, pattern, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = (2, tokens, 4)
    query = torch.randint(
        0,
        1 << bits,
        shape,
        dtype=torch.uint8,
        generator=generator,
    )
    key = torch.randint(
        0,
        1 << bits,
        shape,
        dtype=torch.uint8,
        generator=generator,
    )
    if pattern == "all_match":
        query.zero_()
        key.zero_()
    elif pattern == "all_mismatch":
        query.zero_()
        key.fill_((1 << bits) - 1)
    elif pattern == "aligned":
        query[:, 1:] = key[:, :-1]
    elif pattern == "periodic":
        period = min(4, tokens)
        codes = torch.randint(
            0,
            1 << bits,
            (2, period, 4),
            dtype=torch.uint8,
            generator=generator,
        )
        positions = torch.arange(tokens) % period
        query = codes[:, positions].clone()
        key = codes[:, positions].clone()
    elif pattern != "random":
        raise ValueError(pattern)
    payload = torch.randint(
        0,
        256,
        (2, tokens, 2),
        dtype=torch.uint8,
        generator=generator,
    )
    return query, key, payload


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("tokens", [1, 2, 7, 31, 32, 33])
@pytest.mark.parametrize("max_suffix_length", [1, 5, 32, 100])
def test_indexed_routes_and_lengths_match_scalar_oracle(
    indexed_module,
    bits,
    tokens,
    max_suffix_length,
):
    query, key, payload = _make_packed_case(
        bits,
        tokens,
        pattern="random",
        seed=bits * 100_000 + tokens * 100 + max_suffix_length,
    )
    query_logits = _unpack_symbols(query, bits, device="cuda")
    key_logits = _unpack_symbols(key, bits, device="cuda")
    value_logits = _unpack_symbols(payload, 8, device="cuda")
    packed_query, packed_key = pack_sign_bits(
        query_logits,
        key_logits,
        module=indexed_module,
    )
    offsets, occurrences = build_occurrence_index(
        packed_key,
        symbol_dim=bits,
        module=indexed_module,
    )
    expected_routes, expected_lengths = _bounded_reference(
        query,
        key,
        max_suffix_length,
    )

    for method in INDEXED_METHODS:
        _, routes, lengths = indexed_hard_forward_from_packed(
            packed_query,
            packed_key,
            value_logits,
            offsets,
            occurrences,
            max_suffix_length=max_suffix_length,
            method=method,
            module=indexed_module,
        )
        assert torch.equal(routes.cpu(), expected_routes)
        assert torch.equal(lengths.cpu(), expected_lengths)


@pytest.mark.parametrize(
    ("pattern", "tokens"),
    [
        ("random", 65),
        ("aligned", 65),
        ("periodic", 65),
        ("all_match", 65),
        ("all_mismatch", 65),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_all_exact_gpu_paths_match_production_and_cpu_runtime(
    indexed_module,
    pattern,
    tokens,
    dtype,
):
    bits = 8
    query, key, payload = _make_packed_case(
        bits,
        tokens,
        pattern=pattern,
        seed=700 + tokens,
    )
    query_logits = _unpack_symbols(
        query,
        bits,
        device="cuda",
        dtype=dtype,
    )
    key_logits = _unpack_symbols(
        key,
        bits,
        device="cuda",
        dtype=dtype,
    )
    value_logits = _unpack_symbols(
        payload,
        8,
        device="cuda",
        dtype=dtype,
    )
    production, expected_packed_query, expected_packed_key = (
        torch.ops.rosa_soft.hard_forward(
            query_logits,
            key_logits,
            value_logits,
        )
    )
    packed_query, packed_key = pack_sign_bits(
        query_logits,
        key_logits,
        module=indexed_module,
    )
    assert torch.equal(packed_query, expected_packed_query)
    assert torch.equal(packed_key, expected_packed_key)
    offsets, occurrences = build_occurrence_index(
        packed_key,
        symbol_dim=bits,
        module=indexed_module,
    )

    matched_key_ends = rosa_soft.RosaSam(4, bits).update_packed(
        query.to(torch.int32),
        key.to(torch.int32),
    )
    local_routes = (matched_key_ends + 1).clamp_min(0)
    expanded_payload = payload.repeat_interleave(2, dim=2).permute(0, 2, 1)
    packed_output = torch.gather(
        expanded_payload,
        2,
        local_routes.permute(0, 2, 1),
    ).permute(0, 2, 1)
    expected_routes = torch.where(
        matched_key_ends >= 0,
        matched_key_ends + 1,
        torch.zeros_like(matched_key_ends),
    ).permute(0, 2, 1).to(torch.int32)
    expected_output = _unpack_symbols(
        packed_output,
        8,
        dtype=dtype,
    )
    expected_output = torch.where(
        matched_key_ends.unsqueeze(-1) >= 0,
        expected_output,
        torch.zeros((), dtype=dtype),
    ).cuda()

    results = []
    for method in INDEXED_METHODS:
        results.append(
            indexed_hard_forward_from_packed(
                packed_query,
                packed_key,
                value_logits,
                offsets,
                occurrences,
                max_suffix_length=tokens,
                method=method,
                module=indexed_module,
            )
        )
    results.append(
        diagonal_hard_forward_from_packed(
            packed_query,
            packed_key,
            value_logits,
            max_suffix_length=tokens,
            module=indexed_module,
        )
    )
    for output, routes, _ in results:
        assert torch.equal(routes, expected_routes.cuda())
        assert torch.equal(output, expected_output)
        assert torch.equal(output, production)


def test_occurrence_index_is_sorted_complete_partition(indexed_module):
    packed_key = torch.tensor(
        [[[3, 1, 3, 0, 1, 2, 3]]],
        dtype=torch.int32,
        device="cuda",
    )
    offsets, occurrences = build_occurrence_index(
        packed_key,
        symbol_dim=2,
        module=indexed_module,
    )
    offsets = offsets.cpu()[0, 0]
    occurrences = occurrences.cpu()[0, 0]

    assert offsets[:5].tolist() == [0, 1, 3, 4, 7]
    assert torch.all(offsets[5:] == 7)
    assert occurrences.tolist() == [3, 1, 4, 5, 0, 2, 6]
