import pytest
import torch

import rosa_soft  # noqa: F401 - register production operators
from benchmarks.persistent_wavefront_vjp import (
    grouped_checkpoint_live_state_elements,
    grouped_checkpoint_stats_group_size,
    grouped_checkpoint_reverse_tensor_symbols,
    grouped_checkpoint_vjp,
    load_persistent_wavefront_vjp,
    unbounded_replay_stats,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="grouped checkpoint VJP requires CUDA",
)


@pytest.fixture(scope="session")
def checkpoint_module():
    try:
        return load_persistent_wavefront_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")


def _case(
    seq_len,
    *,
    bits=8,
    dropout_p=0.0,
    pattern="random",
    batch_size=1,
    num_heads=2,
    num_value_heads=1,
):
    generator = torch.Generator(device="cuda").manual_seed(
        77100 + seq_len * 13 + bits
    )
    query = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        bits,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        query.shape,
        dtype=query.dtype,
        device="cuda",
        generator=generator,
    )
    if pattern == "all_match":
        query.fill_(1.0)
        key.fill_(1.0)
    elif pattern == "alternating":
        signs = torch.where(
            torch.arange(seq_len, device="cuda")[:, None, None] % 2 == 0,
            1.0,
            -1.0,
        )
        query.copy_(signs)
        key.copy_(signs)
    value = torch.randn(
        batch_size,
        seq_len,
        num_value_heads,
        64,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        64,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    seed = (
        torch.tensor(0x31415926, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    return (
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
    )


@pytest.mark.parametrize("seq_len", [1, 2, 17, 33, 65])
@pytest.mark.parametrize("bits", [1, 8, 32])
@pytest.mark.parametrize("gradient_mask", [1, 2, 4, 7])
def test_grouped_checkpoint_matches_unbounded_production(
    seq_len,
    bits,
    gradient_mask,
    checkpoint_module,
):
    arguments = _case(seq_len, bits=bits)
    expected = torch.ops.rosa_soft.surrogate_vjp_unbounded_masked(
        *arguments,
        1.0,
        0.0,
        3.0,
        gradient_mask,
    )
    actual = grouped_checkpoint_vjp(
        *arguments,
        stats_group_size=64,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        gradient_mask=gradient_mask,
        module=checkpoint_module,
    )
    for candidate, reference in zip(actual, expected):
        if reference.numel() == 0:
            assert candidate.numel() == 0
        else:
            torch.testing.assert_close(
                candidate,
                reference,
                rtol=4e-3,
                atol=4e-3,
            )


@pytest.mark.parametrize("pattern", ["random", "all_match", "alternating"])
def test_grouped_checkpoint_dropout_and_patterns(
    pattern,
    checkpoint_module,
):
    arguments = _case(67, dropout_p=0.2, pattern=pattern)
    expected = torch.ops.rosa_soft.surrogate_vjp_unbounded_masked(
        *arguments,
        1.7,
        0.2,
        3.0,
        7,
    )
    actual = grouped_checkpoint_vjp(
        *arguments,
        stats_group_size=64,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=7,
        module=checkpoint_module,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(
            candidate,
            reference,
            rtol=5e-3,
            atol=5e-3,
        )


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "num_value_heads"),
    [(2, 4, 2), (1, 8, 1), (2, 8, 4)],
)
def test_grouped_checkpoint_batch_and_value_head_groups(
    batch_size,
    num_heads,
    num_value_heads,
    checkpoint_module,
):
    arguments = _case(
        65,
        batch_size=batch_size,
        num_heads=num_heads,
        num_value_heads=num_value_heads,
    )
    expected = torch.ops.rosa_soft.surrogate_vjp_unbounded_masked(
        *arguments,
        1.0,
        0.0,
        3.0,
        7,
    )
    actual = grouped_checkpoint_vjp(
        *arguments,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        gradient_mask=7,
        module=checkpoint_module,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(
            candidate,
            reference,
            rtol=5e-3,
            atol=5e-3,
        )


def test_grouped_checkpoint_repeated_execution_is_stable(checkpoint_module):
    arguments = _case(129, dropout_p=0.2, batch_size=2, num_heads=4)
    first = grouped_checkpoint_vjp(
        *arguments,
        stats_group_size=129,
        dropout_p=0.2,
        module=checkpoint_module,
    )
    second = grouped_checkpoint_vjp(
        *arguments,
        stats_group_size=129,
        dropout_p=0.2,
        module=checkpoint_module,
    )
    for candidate, reference in zip(first, second):
        torch.testing.assert_close(candidate, reference, rtol=0.0, atol=2e-6)


@pytest.mark.parametrize("bits", [1, 8, 16, 32])
@pytest.mark.parametrize("gradient_mask", [1, 2, 3, 5, 6, 7])
@pytest.mark.parametrize("tensor_symbol_mask", [1, 2, 3])
def test_tensor_symbol_contraction_matches_scalar_reverse(
    bits,
    gradient_mask,
    tensor_symbol_mask,
    checkpoint_module,
):
    arguments = _case(67, bits=bits, dropout_p=0.2)
    row_stats = unbounded_replay_stats(
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5],
        arguments[6],
        symbol_dim=bits,
        group_size=67,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        module=checkpoint_module,
    )
    expected = grouped_checkpoint_reverse_tensor_symbols(
        *arguments,
        row_stats,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=gradient_mask,
        tensor_symbol_mask=0,
        specialized_replay=False,
        module=checkpoint_module,
    )
    actual = grouped_checkpoint_reverse_tensor_symbols(
        *arguments,
        row_stats,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=gradient_mask,
        tensor_symbol_mask=tensor_symbol_mask,
        module=checkpoint_module,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=3e-4, atol=5e-5)


@pytest.mark.parametrize("bits", [8, 32])
@pytest.mark.parametrize("tensor_symbol_mask", [0, 2, 3])
def test_specialized_replay_matches_scalar_reverse(
    bits,
    tensor_symbol_mask,
    checkpoint_module,
):
    arguments = _case(97, bits=bits, dropout_p=0.2, pattern="alternating")
    row_stats = unbounded_replay_stats(
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5],
        arguments[6],
        symbol_dim=bits,
        group_size=97,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        module=checkpoint_module,
    )
    expected = grouped_checkpoint_reverse_tensor_symbols(
        *arguments,
        row_stats,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=7,
        tensor_symbol_mask=0,
        specialized_replay=False,
        module=checkpoint_module,
    )
    actual = grouped_checkpoint_reverse_tensor_symbols(
        *arguments,
        row_stats,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=7,
        tensor_symbol_mask=tensor_symbol_mask,
        specialized_replay=True,
        module=checkpoint_module,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=3e-4, atol=5e-5)


@pytest.mark.parametrize("bits", [1, 8, 16, 32])
@pytest.mark.parametrize("gradient_mask", [1, 2, 3, 5, 6, 7])
def test_production_dispatch_matches_explicit_scalar_reverse(
    bits,
    gradient_mask,
    checkpoint_module,
):
    arguments = _case(
        2048,
        bits=bits,
        dropout_p=0.2,
        pattern="alternating",
        num_heads=4,
        num_value_heads=2,
    )
    row_stats = unbounded_replay_stats(
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5],
        arguments[6],
        symbol_dim=bits,
        group_size=2048,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        module=checkpoint_module,
    )
    expected = grouped_checkpoint_reverse_tensor_symbols(
        *arguments,
        row_stats,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=gradient_mask,
        tensor_symbol_mask=0,
        specialized_replay=False,
        module=checkpoint_module,
    )
    actual = torch.ops.rosa_soft.surrogate_vjp_unbounded_masked(
        *arguments,
        1.7,
        0.2,
        3.0,
        gradient_mask,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=3e-4, atol=5e-5)


def test_checkpoint_scratch_is_linear_in_sequence_length():
    resident_ctas = 136
    first = grouped_checkpoint_live_state_elements(4096, resident_ctas)
    second = grouped_checkpoint_live_state_elements(8192, resident_ctas)
    assert second == 2 * first


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 4096, 4, 8), 4096),
        ((1, 8192, 4, 8), 4096),
        ((1, 16384, 4, 8), 2720),
        ((2, 8192, 16, 8), 672),
    ],
)
def test_checkpoint_stats_group_size_respects_budget(shape, expected):
    query = torch.empty(shape)
    group_size = grouped_checkpoint_stats_group_size(query)
    assert group_size == expected
    partial_stats_bytes = (
        shape[0] * shape[1] * shape[2] * ((group_size + 31) // 32) * 3 * 4
    )
    assert partial_stats_bytes <= 64 << 20
