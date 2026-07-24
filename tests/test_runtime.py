import pytest
import torch

import rosa_soft


pytestmark = pytest.mark.skipif(
    not rosa_soft.HAS_ROSA_RUNTIME,
    reason="RosaRuntime extension is unavailable",
)


def _bounded_reference(
    query_symbols,
    key_symbols,
    payload_symbols,
    max_suffix_length,
):
    batch, tokens, heads = query_symbols.shape
    value_heads = payload_symbols.size(2)
    group_size = heads // value_heads
    output = torch.zeros(batch, tokens, heads, dtype=torch.uint8)
    end_positions = torch.full(
        (batch, tokens, heads),
        -1,
        dtype=torch.int64,
    )
    for batch_index in range(batch):
        for head in range(heads):
            value_head = head // group_size
            for query_end in range(tokens):
                best_length = 0
                best_key_end = -1
                for key_end in range(query_end):
                    limit = min(
                        max_suffix_length,
                        query_end + 1,
                        key_end + 1,
                    )
                    length = 0
                    while (
                        length < limit
                        and query_symbols[
                            batch_index,
                            query_end - length,
                            head,
                        ]
                        == key_symbols[
                            batch_index,
                            key_end - length,
                            head,
                        ]
                    ):
                        length += 1
                    if length >= best_length and length > 0:
                        best_length = length
                        best_key_end = key_end
                if best_key_end >= 0:
                    end_positions[
                        batch_index,
                        query_end,
                        head,
                    ] = best_key_end
                    output[
                        batch_index,
                        query_end,
                        head,
                    ] = payload_symbols[
                        batch_index,
                        best_key_end + 1,
                        value_head,
                    ]
    return output, end_positions


def _random_packed(
    shape,
    bits,
    *,
    seed,
):
    return torch.randint(
        0,
        1 << bits,
        shape,
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("max_suffix_length", [1, 2, 5, 64])
def test_packed_dense_matches_bounded_reference(max_suffix_length):
    batch, tokens, heads, value_heads, bits = 2, 19, 4, 2, 4
    query = _random_packed(
        (batch, tokens, heads),
        bits,
        seed=1,
    )
    key = _random_packed(
        (batch, tokens, heads),
        bits,
        seed=2,
    )
    payload = _random_packed(
        (batch, tokens, value_heads),
        bits,
        seed=3,
    )
    expected = _bounded_reference(
        query,
        key,
        payload,
        max_suffix_length,
    )

    with rosa_soft.RosaRuntime(
        heads,
        value_heads,
        bits,
        bits,
        max_suffix_length,
    ) as runtime:
        actual = runtime.update_packed(query, key, payload)
        stats = runtime.stats()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert stats[0] > 0
    assert stats[1] > 0
    assert stats[2] == batch * heads * tokens


@pytest.mark.parametrize("max_suffix_length", [1, 2, 4])
def test_binary_length_four_state_space_matches_oracle(
    max_suffix_length,
):
    patterns = torch.tensor(
        [
            [(code >> shift) & 1 for shift in range(4)]
            for code in range(16)
        ],
        dtype=torch.uint8,
    ).unsqueeze(-1)
    query = patterns.repeat_interleave(16, dim=0)
    key = patterns.repeat(16, 1, 1)
    payload = torch.arange(4, dtype=torch.uint8).view(1, 4, 1)
    payload = payload.expand(256, -1, -1).contiguous()
    expected = _bounded_reference(
        query,
        key,
        payload,
        max_suffix_length,
    )

    with rosa_soft.RosaRuntime(
        1,
        1,
        1,
        2,
        max_suffix_length,
    ) as runtime:
        actual = runtime.update_packed(query, key, payload)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_finite_horizon_changes_route_and_matches_training_reference():
    query = torch.zeros(1, 6, 1, dtype=torch.uint8)
    key = torch.tensor(
        [[[0], [0], [0], [1], [0], [0]]],
        dtype=torch.uint8,
    )
    payload = torch.tensor(
        [[[1], [2], [3], [4], [5], [6]]],
        dtype=torch.uint8,
    )
    results = []
    for horizon in (1, 4):
        with rosa_soft.RosaRuntime(
            1,
            1,
            1,
            3,
            horizon,
        ) as runtime:
            results.append(runtime.update_packed(query, key, payload))
        expected = _bounded_reference(query, key, payload, horizon)
        assert torch.equal(results[-1][0], expected[0])
        assert torch.equal(results[-1][1], expected[1])

    assert not torch.equal(results[0][1], results[1][1])

    query_logits = query.to(torch.float32).mul(2).sub(1).unsqueeze(-1)
    key_logits = key.to(torch.float32).mul(2).sub(1).unsqueeze(-1)
    payload_logits = torch.stack(
        [
            ((payload >> shift) & 1).to(torch.float32).mul(2).sub(1)
            for shift in range(3)
        ],
        dim=-1,
    )
    for horizon, (packed_output, _) in zip((1, 4), results):
        reference = rosa_soft.rosa_soft_reference(
            query_logits,
            key_logits,
            payload_logits,
            max_suffix_length=horizon,
        )
        unpacked = torch.stack(
            [
                ((packed_output >> shift) & 1)
                .to(torch.float32)
                .mul(2)
                .sub(1)
                for shift in range(3)
            ],
            dim=-1,
        )
        unpacked[:, 0] = 0
        assert torch.equal(unpacked, reference)


def test_varlen_and_grouped_value_heads_match_reference():
    lengths = [4, 9, 6]
    offsets = torch.tensor([0, 4, 13, 19], dtype=torch.int32)
    heads, value_heads, bits, horizon = 6, 3, 3, 4
    total = sum(lengths)
    query = _random_packed((total, heads), bits, seed=4)
    key = _random_packed((total, heads), bits, seed=5)
    payload = _random_packed((total, value_heads), bits, seed=6)

    expected_output = torch.empty((total, heads), dtype=torch.uint8)
    expected_end = torch.empty((total, heads), dtype=torch.int64)
    for batch_index, (begin, end) in enumerate(
        zip(offsets[:-1], offsets[1:])
    ):
        del batch_index
        expected = _bounded_reference(
            query[begin:end].unsqueeze(0),
            key[begin:end].unsqueeze(0),
            payload[begin:end].unsqueeze(0),
            horizon,
        )
        expected_output[begin:end] = expected[0].squeeze(0)
        expected_end[begin:end] = expected[1].squeeze(0)

    with rosa_soft.RosaRuntime(
        heads,
        value_heads,
        bits,
        bits,
        horizon,
    ) as runtime:
        output, end_positions = runtime.update_packed(
            query,
            key,
            payload,
            cu_seqlens=offsets,
        )

    assert torch.equal(output, expected_output)
    assert torch.equal(end_positions, expected_end)


def test_chunked_updates_equal_one_shot_update():
    batch, tokens, heads, bits, horizon = 2, 24, 2, 3, 5
    query = _random_packed((batch, tokens, heads), bits, seed=7)
    key = _random_packed((batch, tokens, heads), bits, seed=8)
    payload = _random_packed((batch, tokens, heads), bits, seed=9)

    with rosa_soft.RosaRuntime(
        heads,
        heads,
        bits,
        bits,
        horizon,
    ) as one_shot:
        expected = one_shot.update_packed(query, key, payload)

    chunks = [0, 3, 11, 17, tokens]
    outputs = []
    positions = []
    with rosa_soft.RosaRuntime(
        heads,
        heads,
        bits,
        bits,
        horizon,
    ) as chunked:
        for begin, end in zip(chunks[:-1], chunks[1:]):
            output, end_positions = chunked.update_packed(
                query[:, begin:end],
                key[:, begin:end],
                payload[:, begin:end],
            )
            outputs.append(output)
            positions.append(end_positions)

    assert torch.equal(torch.cat(outputs, dim=1), expected[0])
    assert torch.equal(torch.cat(positions, dim=1), expected[1])


def test_unpacked_null_route_is_exact_zero():
    query = torch.ones(1, 5, 1, 2)
    key = -torch.ones_like(query)
    payload = torch.ones(1, 5, 1, 3)
    with rosa_soft.RosaRuntime(
        1,
        1,
        2,
        3,
        3,
    ) as runtime:
        output, end_positions = runtime.update(
            query,
            key,
            payload,
        )

    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(
        end_positions,
        torch.full_like(end_positions, -1),
    )


def test_packed_inputs_ignore_bits_above_declared_width():
    generator = torch.Generator().manual_seed(19)
    query = torch.randint(
        0,
        2,
        (2, 32, 3),
        dtype=torch.uint8,
        generator=generator,
    )
    key = torch.randint(
        0,
        2,
        query.shape,
        dtype=torch.uint8,
        generator=generator,
    )
    payload = torch.randint(
        0,
        4,
        (2, 32, 1),
        dtype=torch.uint8,
        generator=generator,
    )
    query_high = torch.randint(
        0,
        128,
        query.shape,
        dtype=torch.uint8,
        generator=generator,
    ).mul_(2)
    key_high = torch.randint(
        0,
        128,
        key.shape,
        dtype=torch.uint8,
        generator=generator,
    ).mul_(2)
    payload_high = torch.randint(
        0,
        64,
        payload.shape,
        dtype=torch.uint8,
        generator=generator,
    ).mul_(4)

    with rosa_soft.RosaRuntime(3, 1, 1, 2, 5) as canonical:
        expected = canonical.update_packed(query, key, payload)
    with rosa_soft.RosaRuntime(3, 1, 1, 2, 5) as noisy:
        actual = noisy.update_packed(
            query | query_high,
            key | key_high,
            payload | payload_high,
        )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_cpu_async_updates_are_serialized_in_submission_order():
    query = torch.zeros(1, 7, 1, dtype=torch.uint8)
    key = torch.zeros_like(query)
    first_payload = torch.arange(7, dtype=torch.uint8).view(1, 7, 1)
    second_payload = first_payload.add(10)

    with rosa_soft.RosaRuntime(1, 1, 1, 8, 2) as runtime:
        first = runtime.update_packed(
            query,
            key,
            first_payload,
            async_op=True,
        )
        second = runtime.update_packed(
            query,
            key,
            second_payload,
            async_op=True,
        )
        first_result = first.wait()
        second_result = second.wait()

    assert first_result[1][0, -1, 0] == 5
    assert second_result[1][0, 0, 0] == 6
    assert second_result[0][0, 0, 0] == 10


def test_closed_runtime_rejects_state_access_and_updates():
    runtime = rosa_soft.RosaRuntime(1, 1, 1, 1, 1)
    runtime.close()
    runtime.close()
    packed = torch.zeros(1, 1, 1, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="closed"):
        runtime.stats()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.update_packed(packed, packed, packed)


def test_empty_dense_update_preserves_state_contract():
    packed = torch.empty(2, 0, 1, dtype=torch.uint8)
    with rosa_soft.RosaRuntime(1, 1, 1, 1, 3) as runtime:
        output, end_positions = runtime.update_packed(
            packed,
            packed,
            packed,
        )
        stats = runtime.stats()

    assert output.shape == (2, 0, 1)
    assert end_positions.shape == (2, 0, 1)
    assert stats == (2, 0, 0)


def test_runtime_rejects_empty_batch():
    packed = torch.empty(0, 3, 1, dtype=torch.uint8)
    varlen = torch.empty(0, 1, dtype=torch.uint8)
    with rosa_soft.RosaRuntime(1, 1, 1, 1, 2) as runtime:
        with pytest.raises(ValueError, match="batch"):
            runtime.update_packed(packed, packed, packed)
        with pytest.raises(ValueError, match="at least one sequence"):
            runtime.update_packed(
                varlen,
                varlen,
                varlen,
                cu_seqlens=torch.tensor([0]),
            )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("num_heads", (1.5,)),
        ("qk_bits", (1, None, True)),
        ("max_suffix_length", (1, None, 1, 1, False)),
    ],
)
def test_runtime_rejects_non_integer_constructor_values(name, args):
    with pytest.raises(TypeError, match=name):
        rosa_soft.RosaRuntime(*args)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_stream_async_matches_blocking():
    query = _random_packed((2, 20, 2), 3, seed=10).cuda()
    key = _random_packed((2, 20, 2), 3, seed=11).cuda()
    payload = _random_packed((2, 20, 1), 3, seed=12).cuda()

    with rosa_soft.RosaRuntime(2, 1, 3, 3, 4) as blocking:
        expected = blocking.update_packed(query, key, payload)

    stream = torch.cuda.Stream()
    with rosa_soft.RosaRuntime(2, 1, 3, 3, 4) as asynchronous:
        work = asynchronous.update_packed(
            query,
            key,
            payload,
            stream=stream,
            async_op=True,
        )
        actual = work.wait()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
