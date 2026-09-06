import math

import pytest
import torch

from benchmarks.macro_wavefront_vjp import (
    load_macro_wavefront_vjp,
    macro_wavefront_rosa_soft,
    macro_wavefront_scores,
    macro_wavefront_stats,
    macro_wavefront_vjp,
)
from rosa_soft.soft_reference import (
    _apply_attention_dropout,
    _build_vjp_carrier,
    _causal_route_mask,
    _expand_value_heads,
    _hard_sign,
    _masked_route_scores,
    _pairwise_soft_match_gates,
    _route_probabilities,
    _suffix_prefix_product_scores,
    _suffix_score_utility,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="macro-wavefront research requires CUDA",
)


@pytest.fixture(scope="session")
def macro_module():
    try:
        return load_macro_wavefront_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")


def _symbols(
    seq_len: int,
    symbol_dim: int,
    *,
    batch: int = 1,
    heads: int = 2,
    pattern: str = "random",
):
    generator = torch.Generator(device="cuda").manual_seed(
        8123 + seq_len * 37 + symbol_dim
    )
    query_bits = torch.randint(
        0,
        2,
        (batch, heads, seq_len, symbol_dim),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    key_bits = torch.randint(
        0,
        2,
        query_bits.shape,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    if pattern == "all_match":
        query_bits.fill_(1)
        key_bits.fill_(1)
    elif pattern == "alternating":
        signs = torch.arange(seq_len, device="cuda") % 2
        query_bits.copy_(
            signs.view(1, 1, seq_len, 1).expand_as(query_bits)
        )
        key_bits.copy_(query_bits)
    elif pattern == "single_mismatch":
        query_bits.fill_(1)
        key_bits.fill_(1)
        key_bits[..., seq_len // 2, 0] = 0
    elif pattern != "random":
        raise ValueError(f"unknown pattern: {pattern}")

    shifts = torch.arange(symbol_dim, device="cuda", dtype=torch.int64)
    packed_query = (query_bits.to(torch.int64) << shifts).sum(-1).to(
        torch.int32
    )
    packed_key = (key_bits.to(torch.int64) << shifts).sum(-1).to(
        torch.int32
    )
    query = (query_bits.float() * 2.0 - 1.0).permute(0, 2, 1, 3)
    key = (key_bits.float() * 2.0 - 1.0).permute(0, 2, 1, 3)
    return query, key, packed_query, packed_key, generator


def _score_oracle(query, key, mismatch_scale=3.0):
    seq_len = query.size(1)
    mask = _causal_route_mask(seq_len, query.device)
    gates = _pairwise_soft_match_gates(query, key, mask, mismatch_scale)
    return _suffix_prefix_product_scores(gates, seq_len)


def _diagonal_dp_score_oracle(query, key, mismatch_scale=3.0):
    """Independent O(T^2) form of the unlimited suffix recurrence."""

    seq_len = query.size(1)
    mask = _causal_route_mask(seq_len, query.device)
    gates = _pairwise_soft_match_gates(query, key, mask, mismatch_scale)
    scores = torch.zeros_like(gates)
    for query_position in range(1, seq_len):
        route_end = query_position + 1
        scores[..., query_position, 1:route_end] = (
            gates[..., query_position, 1:route_end]
            * (1.0 + scores[..., query_position - 1, :query_position])
        )
    return scores


def _stats_oracle(
    query,
    key,
    value,
    grad_output,
    dropout_seed,
    *,
    scale,
    dropout_p,
    mismatch_scale,
):
    seq_len = query.size(1)
    heads = query.size(2)
    mask = _causal_route_mask(seq_len, query.device)
    scores = _score_oracle(query, key, mismatch_scale)
    route_scores = _masked_route_scores(_suffix_score_utility(scores), mask)
    probabilities = _route_probabilities(route_scores, mask, scale)
    dropout_scales = _apply_attention_dropout(
        torch.ones_like(probabilities), dropout_p, dropout_seed, 0
    )
    route_values = _expand_value_heads(_hard_sign(value.float()), heads)
    route_values[..., 0, :] = 0.0
    utilities = torch.einsum(
        "bhtd,bhad->bhta",
        grad_output.float().permute(0, 2, 1, 3),
        route_values,
    )
    expected_utility = (
        probabilities * dropout_scales * utilities
    ).sum(dim=-1)

    route_index = torch.arange(seq_len, device=query.device).view(
        1, 1, 1, seq_len
    )
    nonnull = mask.view(1, 1, seq_len, seq_len) & (route_index > 0)
    candidate_count = nonnull.sum(dim=-1, keepdim=True).clamp_min(1)
    logits = route_scores * scale - torch.where(
        nonnull,
        candidate_count.to(route_scores.dtype).log(),
        torch.zeros((), device=query.device),
    )
    maximum = logits.amax(dim=-1)
    normalizer = torch.exp(logits - maximum.unsqueeze(-1)).masked_fill(
        ~mask.view(1, 1, seq_len, seq_len), 0.0
    ).sum(dim=-1)
    return torch.stack((maximum, normalizer, expected_utility), dim=-1)


@pytest.mark.parametrize("seq_len", [1, 2, 31, 32, 33, 65, 97, 129])
@pytest.mark.parametrize("symbol_dim", [1, 8, 32])
@pytest.mark.parametrize("tile_size", [32, 64, 96, 128])
@pytest.mark.parametrize(
    "plan",
    [
        "multilaunch",
        "persistent",
        "folded",
        "persistent_rows",
    ],
)
def test_scores_match_unlimited_oracle(
    seq_len,
    symbol_dim,
    tile_size,
    plan,
    macro_module,
):
    query, key, packed_query, packed_key, _ = _symbols(
        seq_len, symbol_dim
    )
    expected = _score_oracle(query, key)
    actual, _, _ = macro_wavefront_scores(
        packed_query,
        packed_key,
        symbol_dim=symbol_dim,
        tile_size=tile_size,
        plan=plan,
        module=macro_module,
    )
    torch.testing.assert_close(actual, expected, rtol=8e-4, atol=8e-4)


@pytest.mark.parametrize("seq_len", [255, 256, 257, 511, 513])
@pytest.mark.parametrize("tile_size", [32, 96])
def test_folded_scores_cover_long_partial_boundaries(
    seq_len,
    tile_size,
    macro_module,
):
    query, key, packed_query, packed_key, _ = _symbols(
        seq_len, 8, batch=2, heads=3
    )
    expected = _diagonal_dp_score_oracle(query, key)
    actual, _, _ = macro_wavefront_scores(
        packed_query,
        packed_key,
        symbol_dim=8,
        tile_size=tile_size,
        plan="folded",
        module=macro_module,
    )
    torch.testing.assert_close(actual, expected, rtol=8e-4, atol=8e-4)


@pytest.mark.parametrize("seq_len", [1, 33, 65, 97, 129, 193])
@pytest.mark.parametrize("tile_size", [32, 64, 96, 128])
@pytest.mark.parametrize(
    "plan",
    [
        "persistent",
        "folded",
        "persistent_rows",
    ],
)
def test_checkpoint_is_exact_last_score_of_each_segment(
    seq_len,
    tile_size,
    plan,
    macro_module,
):
    _, _, packed_query, packed_key, _ = _symbols(
        seq_len, 8, batch=2, heads=3
    )
    scores, checkpoints, _ = macro_wavefront_scores(
        packed_query,
        packed_key,
        symbol_dim=8,
        tile_size=tile_size,
        plan=plan,
        module=macro_module,
    )
    macro_tiles = math.ceil(seq_len / tile_size)
    assert checkpoints.shape == (
        2,
        3,
        macro_tiles * (macro_tiles + 1) // 2,
        2 * tile_size,
    )
    for tile_row in range(macro_tiles):
        query_count = min(tile_size, seq_len - tile_row * tile_size)
        for tile_col in range(tile_row + 1):
            key_count = min(tile_size, seq_len - tile_col * tile_size)
            slot = tile_row * (tile_row + 1) // 2 + tile_col
            center_delta = (tile_row - tile_col) * tile_size
            for edge_index in range(2 * tile_size):
                local_delta = edge_index - (tile_size - 1)
                delta = center_delta + local_delta
                query_begin = max(0, local_delta)
                query_end = min(
                    query_count - 1, local_delta + key_count - 1
                )
                valid = (
                    0 < delta < seq_len and query_end >= query_begin
                )
                actual = checkpoints[:, :, slot, edge_index]
                if valid:
                    key_end = query_end - local_delta
                    expected = scores[
                        :,
                        :,
                        tile_row * tile_size + query_end,
                        tile_col * tile_size + key_end + 1,
                    ]
                else:
                    expected = torch.zeros_like(actual)
                torch.testing.assert_close(
                    actual, expected, rtol=2e-5, atol=2e-5
                )


@pytest.mark.parametrize(
    "seq_len,symbol_dim,dtype,dropout_p,pattern",
    [
        (1, 8, torch.float32, 0.0, "random"),
        (31, 32, torch.float32, 0.2, "random"),
        (33, 8, torch.float16, 0.0, "all_match"),
        (65, 8, torch.float32, 0.2, "alternating"),
        (97, 1, torch.float16, 0.0, "single_mismatch"),
        (129, 32, torch.float32, 0.0, "random"),
        (65, 8, torch.bfloat16, 0.2, "random"),
    ],
)
@pytest.mark.parametrize("tile_size", [32, 64, 96, 128])
@pytest.mark.parametrize(
    "plan",
    [
        "multilaunch",
        "persistent",
        "folded",
        "persistent_rows",
    ],
)
def test_online_stats_match_dense_oracle(
    seq_len,
    symbol_dim,
    dtype,
    dropout_p,
    pattern,
    tile_size,
    plan,
    macro_module,
):
    query, key, packed_query, packed_key, generator = _symbols(
        seq_len, symbol_dim, heads=4, pattern=pattern
    )
    value = torch.randn(
        1,
        seq_len,
        2,
        9,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        1,
        seq_len,
        4,
        9,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    seed = (
        torch.tensor(987654321, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    expected = _stats_oracle(
        query,
        key,
        value,
        grad_output,
        seed,
        scale=1.7,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
    )
    actual, _, _ = macro_wavefront_stats(
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
        symbol_dim=symbol_dim,
        tile_size=tile_size,
        scale=1.7,
        dropout_p=dropout_p,
        plan=plan,
        module=macro_module,
    )
    tolerance = {
        torch.float16: 3e-3,
        torch.bfloat16: 2e-2,
    }.get(dtype, 8e-4)
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance
    )


def _vjp_case(
    seq_len,
    symbol_dim,
    *,
    value_dim=9,
    dtype=torch.float32,
    dropout_p=0.0,
    pattern="random",
):
    query_signs, key_signs, packed_query, packed_key, generator = _symbols(
        seq_len, symbol_dim, heads=4, pattern=pattern
    )
    query = (
        query_signs
        + 0.3
        * torch.randn(
            query_signs.shape, device="cuda", generator=generator
        )
    ).to(dtype).requires_grad_()
    key = (
        key_signs
        + 0.3
        * torch.randn(key_signs.shape, device="cuda", generator=generator)
    ).to(dtype).requires_grad_()
    # Repack after the magnitude perturbation so CUDA and the oracle observe
    # exactly the same hard signs.
    shifts = torch.arange(symbol_dim, device="cuda", dtype=torch.int64)
    packed_query = (
        ((query.detach() > 0).to(torch.int64).permute(0, 2, 1, 3))
        << shifts
    ).sum(-1).to(torch.int32)
    packed_key = (
        ((key.detach() > 0).to(torch.int64).permute(0, 2, 1, 3))
        << shifts
    ).sum(-1).to(torch.int32)
    value = torch.randn(
        1,
        seq_len,
        2,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
        requires_grad=True,
    )
    grad_output = torch.randn(
        1,
        seq_len,
        4,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    seed = (
        torch.tensor(87364521, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    mask = _causal_route_mask(seq_len, query.device)
    gates = _pairwise_soft_match_gates(
        query.float(), key.float(), mask, 3.0
    )
    scores = _suffix_prefix_product_scores(gates, seq_len)
    route_scores = _masked_route_scores(
        _suffix_score_utility(scores), mask
    )
    probabilities = _route_probabilities(route_scores, mask, 1.7)
    probabilities = _apply_attention_dropout(
        probabilities, dropout_p, seed, 0
    )
    carrier = _build_vjp_carrier(value.float(), probabilities, 4)
    expected = torch.autograd.grad(
        carrier, (query, key, value), grad_output.float()
    )
    return (
        query.detach(),
        key.detach(),
        value.detach(),
        grad_output,
        packed_query,
        packed_key,
        seed,
        expected,
    )


@pytest.mark.parametrize(
    "seq_len,symbol_dim,dtype,dropout_p,pattern",
    [
        (1, 8, torch.float32, 0.0, "random"),
        (2, 1, torch.float32, 0.0, "random"),
        (31, 32, torch.float32, 0.2, "random"),
        (33, 8, torch.float16, 0.0, "all_match"),
        (65, 8, torch.float32, 0.2, "alternating"),
        (97, 32, torch.float32, 0.0, "single_mismatch"),
        (129, 8, torch.float16, 0.2, "random"),
        (65, 8, torch.bfloat16, 0.2, "random"),
    ],
)
@pytest.mark.parametrize("tile_size", [32, 64, 96, 128])
@pytest.mark.parametrize(
    "plan",
    [
        "multilaunch",
        "persistent",
        "folded",
        "persistent_rows",
    ],
)
def test_full_vjp_matches_unlimited_autograd_oracle(
    seq_len,
    symbol_dim,
    dtype,
    dropout_p,
    pattern,
    tile_size,
    plan,
    macro_module,
):
    arguments = _vjp_case(
        seq_len,
        symbol_dim,
        dtype=dtype,
        dropout_p=dropout_p,
        pattern=pattern,
    )
    expected = arguments[-1]
    actual = macro_wavefront_vjp(
        *arguments[:7],
        tile_size=tile_size,
        scale=1.7,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
        gradient_mask=7,
        plan=plan,
        module=macro_module,
    )
    tolerance = {
        torch.float16: 4e-3,
        torch.bfloat16: 3e-2,
    }.get(dtype, 1.2e-3)
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(
            candidate,
            reference.float(),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.parametrize("value_dim", [32, 63, 64, 65, 127])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_folded_grouped_utility_matches_autograd(
    value_dim,
    dtype,
    macro_module,
):
    arguments = _vjp_case(
        65,
        8,
        value_dim=value_dim,
        dtype=dtype,
        dropout_p=0.2,
    )
    actual = macro_wavefront_vjp(
        *arguments[:7],
        tile_size=32,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=7,
        plan="folded",
        module=macro_module,
    )
    tolerance = {
        torch.float16: 5e-3,
        torch.bfloat16: 3e-2,
    }.get(dtype, 2e-3)
    for candidate, reference in zip(actual, arguments[-1]):
        torch.testing.assert_close(
            candidate,
            reference.float(),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.parametrize("gradient_mask", range(1, 8))
@pytest.mark.parametrize(
    "plan",
    [
        "persistent",
        "folded",
        "persistent_rows",
    ],
)
def test_gradient_masks(gradient_mask, plan, macro_module):
    arguments = _vjp_case(33, 8)
    actual = macro_wavefront_vjp(
        *arguments[:7],
        tile_size=64,
        scale=1.7,
        gradient_mask=gradient_mask,
        plan=plan,
        module=macro_module,
    )
    for bit, candidate, reference in zip(
        (1, 2, 4), actual, arguments[-1]
    ):
        if gradient_mask & bit:
            torch.testing.assert_close(
                candidate, reference.float(), rtol=1.2e-3, atol=1.2e-3
            )
        else:
            assert candidate.numel() == 0


@pytest.mark.parametrize(
    "seq_len,expected_tile", [(1, 32), (64, 32), (65, 64), (256, 64), (257, 128)]
)
def test_auto_tile_and_occupancy_metadata(
    seq_len, expected_tile, macro_module
):
    _, _, packed_query, packed_key, _ = _symbols(seq_len, 8)
    _, checkpoints, info = macro_wavefront_scores(
        packed_query,
        packed_key,
        symbol_dim=8,
        tile_size=0,
        plan="persistent",
        module=macro_module,
    )
    tile_size, grid, resident, sms, shared_bytes, slots = info.tolist()
    assert tile_size == expected_tile
    assert grid > 0 and resident > 0 and sms > 0
    assert grid == resident * sms
    assert shared_bytes == 16 * tile_size
    macro_tiles = math.ceil(seq_len / tile_size)
    assert slots == macro_tiles * (macro_tiles + 1) // 2
    assert checkpoints.shape[-2:] == (slots, 2 * tile_size)


@pytest.mark.parametrize("seq_len", [65, 257, 1024])
def test_folded_auto_tile_uses_warp_width(seq_len, macro_module):
    _, _, packed_query, packed_key, _ = _symbols(seq_len, 8)
    _, _, info = macro_wavefront_scores(
        packed_query,
        packed_key,
        symbol_dim=8,
        tile_size=0,
        plan="folded",
        module=macro_module,
    )
    assert info[0].item() == 32


def test_python_schedule_validation():
    packed = torch.zeros(1, 1, 2, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="tile_size"):
        macro_wavefront_scores(
            packed, packed, symbol_dim=1, tile_size=48
        )
    with pytest.raises(ValueError, match="plan"):
        macro_wavefront_scores(
            packed, packed, symbol_dim=1, plan="unknown"
        )


def test_ready_queue_matches_barrier_control(macro_module):
    arguments = _vjp_case(
        65,
        8,
        dtype=torch.float32,
        dropout_p=0.2,
        pattern="alternating",
    )
    results = [
        macro_wavefront_vjp(
            *arguments[:7],
            tile_size=32,
            scale=1.7,
            dropout_p=0.2,
            mismatch_scale=3.0,
            gradient_mask=7,
            plan=plan,
            module=macro_module,
        )
        for plan in (
            "multilaunch",
            "persistent",
            "persistent_barrier",
            "persistent_rows",
        )
    ]
    for candidate in results[1:]:
        for actual, reference in zip(candidate, results[0]):
            torch.testing.assert_close(
                actual, reference, rtol=1.2e-3, atol=1.2e-3
            )


def test_autograd_wrapper_preserves_exact_hard_forward(macro_module):
    # The wrapper must expose only the exact hard output; tile_size may alter
    # the surrogate schedule but never the observed forward value.
    query = torch.randn(1, 33, 2, 8, device="cuda", requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(1, 33, 1, 9, device="cuda", requires_grad=True)
    expected = torch.ops.rosa_soft.hard_forward(query, key, value)[0]
    outputs = [
        macro_wavefront_rosa_soft(
            query,
            key,
            value,
            tile_size=tile_size,
            plan=plan,
            module=macro_module,
        )
        for tile_size in (32, 64, 128)
        for plan in (
            "multilaunch",
            "persistent",
            "persistent_rows",
        )
    ]
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    outputs[-1].float().sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None

