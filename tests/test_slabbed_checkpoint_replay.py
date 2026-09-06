import pytest
import torch

from benchmarks.slabbed_checkpoint_replay import (
    SLAB_DIAGONAL_CAPACITY,
    load_slabbed_checkpoint_replay,
    slabbed_live_state_elements,
    slabbed_replay_vjp,
)
from rosa_soft.soft_reference import (
    _apply_attention_dropout,
    _build_vjp_carrier,
    _causal_route_mask,
    _hard_sign,
    _masked_route_scores,
    _pairwise_soft_match_gates,
    _route_probabilities,
    _suffix_prefix_product_scores,
    _suffix_score_utility,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="slabbed replay research requires CUDA",
)


@pytest.fixture(scope="session")
def slabbed_module():
    try:
        return load_slabbed_checkpoint_replay()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")


def _case(
    seq_len: int,
    symbol_dim: int,
    *,
    value_dim: int = 9,
    dtype: torch.dtype = torch.float32,
    dropout_p: float = 0.0,
    pattern: str = "random",
    batch_size: int = 1,
    num_heads: int = 4,
    num_value_heads: int = 2,
):
    generator = torch.Generator(device="cuda").manual_seed(
        62003 + 31 * seq_len + symbol_dim + value_dim
    )
    query_bits = torch.randint(
        0,
        2,
        (batch_size, seq_len, num_heads, symbol_dim),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    key_bits = torch.randint(
        0,
        2,
        query_bits.shape,
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    if pattern == "all_match":
        query_bits.fill_(1)
        key_bits.fill_(1)
    elif pattern == "alternating":
        alternating = (torch.arange(seq_len, device="cuda") % 2).view(
            1, seq_len, 1, 1
        )
        query_bits.copy_(alternating.expand_as(query_bits))
        key_bits.copy_(query_bits)
    elif pattern == "single_mismatch":
        query_bits.fill_(1)
        key_bits.fill_(1)
        key_bits[:, seq_len // 2, :, 0] = 0
    elif pattern != "random":
        raise ValueError(f"unknown pattern: {pattern}")

    query = (query_bits.float() * 2 - 1).to(dtype)
    key = (key_bits.float() * 2 - 1).to(dtype)
    query = query + 0.25 * torch.randn(
        query.shape, device="cuda", dtype=dtype, generator=generator
    )
    key = key + 0.25 * torch.randn(
        key.shape, device="cuda", dtype=dtype, generator=generator
    )
    shifts = torch.arange(symbol_dim, device="cuda", dtype=torch.int64)
    packed_query = (
        ((query > 0).to(torch.int64).permute(0, 2, 1, 3) << shifts)
        .sum(-1)
        .to(torch.int32)
    )
    packed_key = (
        ((key > 0).to(torch.int64).permute(0, 2, 1, 3) << shifts)
        .sum(-1)
        .to(torch.int32)
    )
    value = torch.randn(
        batch_size,
        seq_len,
        num_value_heads,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    seed = (
        torch.tensor(77892311, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )

    query_leaf = query.float().detach().requires_grad_()
    key_leaf = key.float().detach().requires_grad_()
    value_leaf = value.float().detach().requires_grad_()
    mask = _causal_route_mask(seq_len, query.device)
    gates = _pairwise_soft_match_gates(query_leaf, key_leaf, mask, 3.0)
    scores = _suffix_prefix_product_scores(gates, seq_len)
    route_scores = _masked_route_scores(_suffix_score_utility(scores), mask)
    probabilities = _route_probabilities(route_scores, mask, 1.7)
    probabilities = _apply_attention_dropout(
        probabilities, dropout_p, seed, 0
    )
    carrier = _build_vjp_carrier(value_leaf, probabilities, num_heads)
    expected = torch.autograd.grad(
        carrier, (query_leaf, key_leaf, value_leaf), grad_output.float()
    )
    return (
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
        expected,
    )


@pytest.mark.parametrize(
    "seq_len,symbol_dim,dtype,dropout_p,pattern,value_dim",
    [
        (1, 8, torch.float32, 0.0, "random", 9),
        (2, 1, torch.float32, 0.0, "random", 9),
        (31, 32, torch.float32, 0.2, "random", 7),
        (33, 8, torch.float16, 0.0, "all_match", 32),
        (65, 8, torch.float32, 0.2, "alternating", 63),
        (97, 1, torch.float16, 0.0, "single_mismatch", 64),
        (129, 32, torch.bfloat16, 0.2, "random", 65),
        (258, 8, torch.float32, 0.0, "random", 9),
        (259, 8, torch.float32, 0.2, "random", 65),
    ],
)
def test_slabbed_vjp_matches_dense_autograd_oracle(
    seq_len,
    symbol_dim,
    dtype,
    dropout_p,
    pattern,
    value_dim,
    slabbed_module,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    arguments = _case(
        seq_len,
        symbol_dim,
        dtype=dtype,
        dropout_p=dropout_p,
        pattern=pattern,
        value_dim=value_dim,
    )
    actual = slabbed_replay_vjp(
        *arguments[:7],
        scale=1.7,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
        gradient_mask=7,
        module=slabbed_module,
    )
    tolerance = {
        torch.float16: 4e-3,
        torch.bfloat16: 3e-2,
    }.get(dtype, 1.2e-3)
    for candidate, expected in zip(actual, arguments[-1]):
        torch.testing.assert_close(
            candidate, expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("gradient_mask", range(1, 8))
def test_slabbed_vjp_honors_gradient_mask(gradient_mask, slabbed_module):
    arguments = _case(37, 8, value_dim=17)
    actual = slabbed_replay_vjp(
        *arguments[:7],
        scale=1.7,
        mismatch_scale=3.0,
        gradient_mask=gradient_mask,
        module=slabbed_module,
    )
    for bit, candidate, expected in zip((1, 2, 4), actual, arguments[-1]):
        if gradient_mask & bit:
            torch.testing.assert_close(
                candidate, expected, rtol=1.2e-3, atol=1.2e-3
            )
        else:
            assert candidate.numel() == 0


def test_slabbed_vjp_supports_batched_grouped_value_heads(slabbed_module):
    arguments = _case(
        35,
        8,
        value_dim=33,
        batch_size=2,
        num_heads=6,
        num_value_heads=3,
        dropout_p=0.2,
    )
    actual = slabbed_replay_vjp(
        *arguments[:7],
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=7,
        module=slabbed_module,
    )
    for candidate, expected in zip(actual, arguments[-1]):
        torch.testing.assert_close(
            candidate, expected, rtol=1.2e-3, atol=1.2e-3
        )


def test_noncontiguous_inputs_are_repeatable(slabbed_module):
    arguments = list(_case(37, 8, value_dim=17, dropout_p=0.2))
    for index in range(4):
        arguments[index] = (
            arguments[index].transpose(1, 2).contiguous().transpose(1, 2)
        )
        assert not arguments[index].is_contiguous()

    def run():
        return slabbed_replay_vjp(
            *arguments[:7],
            scale=1.7,
            dropout_p=0.2,
            mismatch_scale=3.0,
            gradient_mask=7,
            module=slabbed_module,
        )

    first = run()
    second = run()
    for actual, repeated, expected in zip(first, second, arguments[-1]):
        assert torch.equal(actual, repeated)
        torch.testing.assert_close(
            actual, expected, rtol=1.2e-3, atol=1.2e-3
        )


def test_live_state_is_linear_after_slab_capacity():
    first_tokens = 4 * SLAB_DIAGONAL_CAPACITY
    second_tokens = 2 * first_tokens
    first = slabbed_live_state_elements(2, 3, first_tokens)
    second = slabbed_live_state_elements(2, 3, second_tokens)

    # Row padding disappears for these multiples of 32, so doubling T doubles
    # every live-state term exactly once the internal slab capacity is reached.
    assert second == 2 * first


def test_live_state_drops_utility_slab_for_value_only():
    all_gradients = slabbed_live_state_elements(
        1, 4, 1024, needs_symbol_gradients=True
    )
    value_only = slabbed_live_state_elements(
        1, 4, 1024, needs_symbol_gradients=False
    )
    assert value_only < all_gradients
    assert value_only > all_gradients // 2


def test_input_validation_rejects_invalid_gradient_mask(slabbed_module):
    arguments = _case(2, 1)
    with pytest.raises(ValueError, match="gradient_mask"):
        slabbed_replay_vjp(
            *arguments[:7], gradient_mask=0, module=slabbed_module
        )
