import threading
from concurrent.futures import ThreadPoolExecutor

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
    payload_heads = payload_symbols.size(2)
    group_size = heads // payload_heads
    output = torch.zeros(batch, tokens, heads, dtype=torch.uint8)
    end_positions = torch.full(
        (batch, tokens, heads),
        -1,
        dtype=torch.int64,
    )
    for batch_index in range(batch):
        for head in range(heads):
            payload_head = head // group_size
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
                        payload_head,
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
    batch, tokens, heads, payload_heads, bits = 2, 19, 4, 2, 4
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
        (batch, tokens, payload_heads),
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
        payload_heads,
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
    assert stats[2] == batch * payload_heads * tokens


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


def test_varlen_and_grouped_payload_heads_match_reference():
    lengths = [4, 9, 6]
    offsets = torch.tensor([0, 4, 13, 19], dtype=torch.int32)
    heads, payload_heads, bits, horizon = 6, 3, 3, 4
    total = sum(lengths)
    query = _random_packed((total, heads), bits, seed=4)
    key = _random_packed((total, heads), bits, seed=5)
    payload = _random_packed((total, payload_heads), bits, seed=6)

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
        payload_heads,
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
    batch, tokens, heads, payload_heads, bits, horizon = (
        2,
        24,
        4,
        2,
        3,
        5,
    )
    query = _random_packed((batch, tokens, heads), bits, seed=7)
    key = _random_packed((batch, tokens, heads), bits, seed=8)
    payload = _random_packed(
        (batch, tokens, payload_heads),
        bits,
        seed=9,
    )

    with rosa_soft.RosaRuntime(
        heads,
        payload_heads,
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
        payload_heads,
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


def test_cpu_async_submission_snapshots_inputs_and_offsets():
    offsets = torch.tensor([0, 5], dtype=torch.int64)
    query = torch.zeros(5, 1, dtype=torch.uint8)
    key = torch.zeros_like(query)
    payload = torch.arange(5, dtype=torch.uint8).view(5, 1)
    expected = _bounded_reference(
        query.unsqueeze(0),
        key.unsqueeze(0),
        payload.unsqueeze(0),
        2,
    )
    entered = threading.Event()
    release = threading.Event()

    with rosa_soft.RosaRuntime(1, 1, 1, 8, 2) as runtime:
        blocker = runtime._executor.submit(
            lambda: (entered.set(), release.wait())
        )
        assert entered.wait(timeout=5)
        work = runtime.update_packed(
            query,
            key,
            payload,
            cu_seqlens=offsets,
            async_op=True,
        )
        query.fill_(1)
        key.fill_(1)
        payload.fill_(99)
        offsets[1] = 0
        release.set()
        blocker.result(timeout=5)
        actual = work.wait()

    assert torch.equal(actual[0], expected[0].squeeze(0))
    assert torch.equal(actual[1], expected[1].squeeze(0))


def test_work_wait_is_thread_safe_and_repeatable():
    packed = torch.zeros(1, 8, 1, dtype=torch.uint8)
    payload = torch.arange(8, dtype=torch.uint8).view(1, 8, 1)
    with rosa_soft.RosaRuntime(1, 1, 1, 8, 3) as runtime:
        work = runtime.update_packed(
            packed,
            packed,
            payload,
            async_op=True,
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: work.wait(), range(16)))
        repeated = work.wait()

    first = results[0]
    assert all(result[0] is first[0] for result in results)
    assert all(result[1] is first[1] for result in results)
    assert repeated[0] is first[0]
    assert repeated[1] is first[1]


def test_native_update_failure_is_repeatable_and_poison_runtime():
    class FailingNative:
        def __init__(self):
            self.close_calls = 0
            self.update_calls = 0
            self.entered = threading.Event()
            self.release = threading.Event()

        def update_packed(self, *args):
            del args
            self.update_calls += 1
            self.entered.set()
            assert self.release.wait(timeout=5)
            raise RuntimeError("native update failed")

        def close(self):
            self.close_calls += 1

    runtime = rosa_soft.RosaRuntime(1, 1, 1, 1, 1)
    native = FailingNative()
    runtime._native = native
    packed = torch.zeros(1, 1, 1, dtype=torch.uint8)
    work = runtime.update_packed(
        packed,
        packed,
        packed,
        async_op=True,
    )
    assert native.entered.wait(timeout=5)
    queued = runtime.update_packed(
        packed,
        packed,
        packed,
        async_op=True,
    )
    native.release.set()

    for _ in range(2):
        with pytest.raises(RuntimeError, match="native update failed"):
            work.wait()
    with pytest.raises(RuntimeError, match="failed"):
        queued.wait()
    assert native.update_calls == 1
    assert runtime.state == "FAILED"
    with pytest.raises(RuntimeError, match="failed"):
        runtime.update_packed(packed, packed, packed)

    runtime.close()
    assert native.close_calls == 1
    assert runtime.state == "CLOSED"


def test_concurrent_close_has_one_owner_and_exposes_closing_state():
    class BlockingClose:
        def __init__(self):
            self.calls = 0
            self.entered = threading.Event()
            self.release = threading.Event()

        def close(self):
            self.calls += 1
            self.entered.set()
            assert self.release.wait(timeout=5)

    runtime = rosa_soft.RosaRuntime(1, 1, 1, 1, 1)
    native = BlockingClose()
    runtime._native = native
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(runtime.close)
        assert native.entered.wait(timeout=5)
        second = executor.submit(runtime.close)
        assert runtime.state == "CLOSING"
        packed = torch.zeros(1, 1, 1, dtype=torch.uint8)
        with pytest.raises(RuntimeError, match="closing"):
            runtime.update_packed(packed, packed, packed)
        native.release.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert native.calls == 1
    assert runtime.state == "CLOSED"


def test_close_shuts_down_executor_when_native_close_fails():
    class FailingClose:
        def close(self):
            raise RuntimeError("native close failed")

    runtime = rosa_soft.RosaRuntime(1, 1, 1, 1, 1)
    runtime._native = FailingClose()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="native close failed"):
            runtime.close()
    assert runtime.state == "FAILED"
    assert runtime._executor_shutdown


def test_reset_clears_state_stats_and_slot_binding():
    packed = torch.zeros(2, 5, 2, dtype=torch.uint8)
    payload = _random_packed((2, 5, 1), 3, seed=31)
    with rosa_soft.RosaRuntime(2, 1, 1, 3, 2) as runtime:
        runtime.update_packed(
            packed,
            packed,
            payload,
            sequence_ids=[10, 20],
        )
        before = runtime.memory_stats()
        runtime.reset()
        after = runtime.memory_stats()
        reset_result = runtime.update_packed(
            packed[:1],
            packed[:1],
            payload[:1],
            sequence_ids=[20],
        )

    assert before["payload_symbols"] == 2 * 5
    assert before["automata"] == 2 * 2
    assert before["sequences"] == 2
    assert before["logical_bytes"] > before["payload_symbols"]
    assert after == {
        "states": 0,
        "edges": 0,
        "payload_symbols": 0,
        "automata": 0,
        "sequences": 0,
        "logical_bytes": 0,
    }
    assert torch.equal(
        reset_result[1][:, 0],
        torch.full_like(reset_result[1][:, 0], -1),
    )


def test_slot_count_is_fixed_until_reset():
    packed = torch.zeros(2, 2, 1, dtype=torch.uint8)
    with rosa_soft.RosaRuntime(1, 1, 1, 1, 1) as runtime:
        runtime.update_packed(packed, packed, packed)
        with pytest.raises(RuntimeError, match="batch size is fixed"):
            runtime.update_packed(
                packed[:1],
                packed[:1],
                packed[:1],
            )
        assert runtime.state == "FAILED"


def test_grouped_payload_history_is_stored_once_per_payload_head():
    batch, tokens, heads, payload_heads = 3, 7, 6, 2
    query = _random_packed((batch, tokens, heads), 2, seed=32)
    key = _random_packed((batch, tokens, heads), 2, seed=33)
    payload = _random_packed(
        (batch, tokens, payload_heads),
        3,
        seed=34,
    )
    with rosa_soft.RosaRuntime(
        heads,
        payload_heads,
        2,
        3,
        4,
    ) as runtime:
        actual = runtime.update_packed(query, key, payload)
        stats = runtime.stats()
        detailed = runtime.memory_stats()

    expected = _bounded_reference(query, key, payload, 4)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert stats[2] == batch * tokens * payload_heads
    assert detailed["payload_symbols"] == stats[2]
    assert detailed["automata"] == batch * heads


def test_sequence_ids_enforce_fixed_slot_semantics():
    packed = torch.zeros(2, 3, 1, dtype=torch.uint8)
    with rosa_soft.RosaRuntime(1, 1, 1, 1, 2) as runtime:
        runtime.update_packed(
            packed,
            packed,
            packed,
            sequence_ids=torch.tensor([101, 202]),
        )
        with pytest.raises(RuntimeError, match="required"):
            runtime.update_packed(packed, packed, packed)
        with pytest.raises(RuntimeError, match="slots are fixed"):
            runtime.update_packed(
                packed,
                packed,
                packed,
                sequence_ids=[202, 101],
            )
        runtime.update_packed(
            packed,
            packed,
            packed,
            sequence_ids=[101, 202],
        )
        assert runtime.state == "OPEN"


def test_sequence_ids_cannot_be_enabled_after_state_exists():
    packed = torch.zeros(1, 2, 1, dtype=torch.uint8)
    with rosa_soft.RosaRuntime(1, 1, 1, 1, 1) as runtime:
        runtime.update_packed(packed, packed, packed)
        with pytest.raises(RuntimeError, match="first update"):
            runtime.update_packed(
                packed,
                packed,
                packed,
                sequence_ids=[7],
            )
        assert runtime.state == "OPEN"


@pytest.mark.parametrize(
    ("qk_bits", "payload_bits"),
    [(0, 1), (9, 1), (1, 0), (1, 9)],
)
def test_packed_byte_bit_widths_are_limited_to_one_through_eight(
    qk_bits,
    payload_bits,
):
    with pytest.raises(RuntimeError, match=r"in \[1, 8\]"):
        rosa_soft.RosaRuntime(
            1,
            1,
            qk_bits,
            payload_bits,
            1,
        )


def test_checkpoint_boundary_is_explicitly_unsupported():
    runtime = rosa_soft.RosaRuntime(1, 1, 1, 1, 1)
    try:
        with pytest.raises(NotImplementedError, match="unsupported"):
            runtime.state_dict()
        with pytest.raises(NotImplementedError, match="unsupported"):
            runtime.load_state_dict({})
        assert runtime.state == "OPEN"
    finally:
        runtime.close()


def test_payload_names_and_legacy_value_aliases_are_compatible():
    runtime = rosa_soft.RosaRuntime(
        num_heads=4,
        num_payload_heads=2,
        qk_bits=3,
        payload_bits=5,
        max_suffix_length=7,
    )
    try:
        assert runtime.num_payload_heads == 2
        assert runtime.payload_bits == 5
        assert runtime.num_value_heads == runtime.num_payload_heads
        assert runtime.value_bits == runtime.payload_bits
    finally:
        runtime.close()

    legacy = rosa_soft.RosaRuntime(
        num_heads=4,
        num_value_heads=2,
        qk_bits=3,
        value_bits=5,
        max_suffix_length=7,
    )
    legacy.close()


def test_constructor_rejects_conflicting_payload_aliases():
    with pytest.raises(TypeError, match="only one"):
        rosa_soft.RosaRuntime(
            2,
            num_payload_heads=1,
            num_value_heads=1,
        )
    with pytest.raises(TypeError, match="only one"):
        rosa_soft.RosaRuntime(
            2,
            payload_bits=3,
            value_bits=3,
        )


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
