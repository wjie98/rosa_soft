import pytest
import torch

import rosa_soft
from benchmarks.persistent_wavefront_vjp import (
    load_persistent_wavefront_vjp,
    persistent_wavefront_stats,
    unbounded_group_scores,
    unbounded_replay_stats,
    unbounded_replay_rosa_soft,
    unbounded_replay_vjp,
    wavefront_log_gate_vjp,
    wavefront_rosa_soft,
    wavefront_scores,
    wavefront_stats,
    wavefront_vjp,
)
from benchmarks.diagonal_recurrence import diagonal_suffix_log_gate_vjp
from rosa_soft.soft_reference import (
    _apply_attention_dropout,
    _causal_route_mask,
    _expand_value_heads,
    _hard_sign,
    _masked_route_scores,
    _pairwise_soft_match_gates,
    _route_probabilities,
    _suffix_score_utility,
    _suffix_prefix_product_scores,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="persistent wavefront research requires CUDA",
)


@pytest.fixture(scope="session")
def wavefront_module():
    try:
        return load_persistent_wavefront_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")


def _case(seq_len: int, bits: int, *, all_match: bool = False):
    generator = torch.Generator(device="cuda").manual_seed(
        91000 + seq_len * 17 + bits
    )
    query = torch.randn(
        1,
        seq_len,
        2,
        bits,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        query.shape,
        device="cuda",
        generator=generator,
    )
    if all_match:
        query.fill_(1.0)
        key.fill_(1.0)
    value = torch.ones(1, seq_len, 1, 1, device="cuda")
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query,
        key,
        value,
    )
    return query, key, packed_query, packed_key


def _reference_scores(query, key, window, mismatch_scale):
    mask = _causal_route_mask(query.size(1), query.device)
    gates = _pairwise_soft_match_gates(
        query,
        key,
        mask,
        mismatch_scale,
    )
    return _suffix_prefix_product_scores(gates, window)


@pytest.mark.parametrize("seq_len", [2, 17, 33, 79])
@pytest.mark.parametrize("bits", [1, 8, 32])
@pytest.mark.parametrize("group_width", [1, 7, 32, 64, 128, 256, 1024])
def test_unbounded_group_scores_match_full_suffix_oracle(
    seq_len,
    bits,
    group_width,
    wavefront_module,
):
    query, key, packed_query, packed_key = _case(seq_len, bits)
    expected = _reference_scores(query, key, seq_len, 3.0)
    for group_start in range(1, seq_len, group_width):
        actual = unbounded_group_scores(
            packed_query,
            packed_key,
            symbol_dim=bits,
            group_start=group_start,
            group_width=group_width,
            mismatch_scale=3.0,
            module=wavefront_module,
        )
        actual_width = actual.size(-1)
        for local_delta in range(actual_width):
            delta = group_start + local_delta
            rows = torch.arange(delta, seq_len, device=query.device)
            routes = rows - delta + 1
            torch.testing.assert_close(
                actual[:, :, rows, local_delta],
                expected[:, :, rows, routes],
                rtol=2e-5,
                atol=2e-5,
            )


@pytest.mark.parametrize("seq_len", [1, 17, 65])
@pytest.mark.parametrize("group_size", [1, 7, 32, 64, 128, 256, 1024])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("value_dim", [7, 64])
def test_unbounded_replay_stats_match_full_suffix_oracle(
    seq_len,
    group_size,
    dropout_p,
    value_dim,
    wavefront_module,
):
    bits = 8
    query, key, packed_query, packed_key = _case(seq_len, bits)
    generator = torch.Generator(device="cuda").manual_seed(131000 + seq_len)
    value = torch.randn(
        1, seq_len, 1, value_dim, device="cuda", generator=generator
    )
    grad_output = torch.randn(
        1, seq_len, 2, value_dim, device="cuda", generator=generator
    )
    seed = (
        torch.tensor(987654321, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    scale = 2.0
    mask = _causal_route_mask(seq_len, query.device)
    scores = _reference_scores(query, key, seq_len, 3.0)
    route_scores = _masked_route_scores(_suffix_score_utility(scores), mask)
    probabilities = _route_probabilities(route_scores, mask, scale)
    dropout_scales = _apply_attention_dropout(
        torch.ones_like(probabilities), dropout_p, seed, 0
    )
    route_values = _expand_value_heads(_hard_sign(value), query.size(2))
    route_values[..., 0, :] = 0.0
    utilities = torch.einsum(
        "bhtd,bhad->bhta",
        grad_output.permute(0, 2, 1, 3),
        route_values,
    )
    expected_utility = (
        probabilities * dropout_scales * utilities
    ).sum(dim=-1)
    route_index = torch.arange(seq_len, device=query.device).view(
        1, 1, 1, seq_len
    )
    nonnull = mask.view(1, 1, seq_len, seq_len) & (route_index > 0)
    count = nonnull.sum(dim=-1, keepdim=True).clamp_min(1)
    logits = route_scores * scale - torch.where(
        nonnull,
        count.to(route_scores.dtype).log(),
        torch.zeros((), device=query.device),
    )
    maximum = logits.amax(dim=-1)
    normalizer = torch.exp(logits - maximum.unsqueeze(-1)).masked_fill(
        ~mask.view(1, 1, seq_len, seq_len), 0.0
    ).sum(dim=-1)
    expected = torch.stack(
        (maximum, normalizer.reciprocal(), expected_utility), dim=-1
    )
    actual = unbounded_replay_stats(
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
        symbol_dim=bits,
        group_size=group_size,
        scale=scale,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
        module=wavefront_module,
    )
    tolerance = 7e-5 if value_dim == 7 else 1.2e-3
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    "seq_len,bits,group_size,mask",
    [
        (1, 8, 32, 7),
        (2, 8, 1, 7),
        (65, 8, 1, 7),
        (65, 8, 7, 7),
        (65, 8, 32, 7),
        (65, 8, 64, 7),
        (65, 8, 128, 7),
        (65, 8, 256, 7),
        (65, 8, 1024, 7),
        (37, 1, 32, 7),
        (37, 8, 32, 7),
        (37, 32, 32, 7),
        *((17, 8, 7, mask) for mask in range(1, 8)),
    ],
)
def test_unbounded_replay_vjp_matches_full_suffix_production_oracle(
    seq_len,
    bits,
    group_size,
    mask,
    wavefront_module,
):
    arguments = _vjp_arguments(
        seq_len=seq_len,
        window=seq_len,
        bits=bits,
        value_dim=7,
    )
    expected = torch.ops.rosa_soft.surrogate_vjp_masked(*arguments, mask)
    actual = unbounded_replay_vjp(
        *arguments[:7],
        group_size=group_size,
        scale=arguments[8],
        dropout_p=arguments[9],
        mismatch_scale=arguments[10],
        gradient_mask=mask,
        module=wavefront_module,
    )
    for candidate, reference in zip(actual, expected):
        if reference.numel() == 0:
            assert candidate.numel() == 0
        else:
            torch.testing.assert_close(
                candidate,
                reference,
                rtol=8e-4,
                atol=8e-4,
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("pattern", ["random", "all_match", "alternating"])
def test_unbounded_replay_vjp_dtype_dropout_and_pattern(
    dtype,
    dropout_p,
    pattern,
    wavefront_module,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    arguments = _vjp_arguments(
        seq_len=37,
        window=37,
        dtype=dtype,
        dropout_p=dropout_p,
        pattern=pattern,
        batch=2,
        heads=6,
        value_heads=3,
        bits=32,
        value_dim=65,
    )
    expected = torch.ops.rosa_soft.surrogate_vjp_masked(*arguments, 7)
    actual = unbounded_replay_vjp(
        *arguments[:7],
        group_size=32,
        scale=arguments[8],
        dropout_p=arguments[9],
        mismatch_scale=arguments[10],
        gradient_mask=7,
        module=wavefront_module,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(
            candidate,
            reference,
            rtol=1.2e-3,
            atol=1.2e-3,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("group_size", [32, 128])
def test_unbounded_replay_tensor_tiles_cross_group_boundaries(
    dtype,
    dropout_p,
    group_size,
    wavefront_module,
):
    arguments = _vjp_arguments(
        seq_len=129,
        window=129,
        dtype=dtype,
        dropout_p=dropout_p,
        pattern="random",
        batch=1,
        heads=4,
        value_heads=2,
        bits=32,
        value_dim=65,
    )
    expected = torch.ops.rosa_soft.surrogate_vjp_masked(*arguments, 7)
    actual = unbounded_replay_vjp(
        *arguments[:7],
        group_size=group_size,
        scale=arguments[8],
        dropout_p=arguments[9],
        mismatch_scale=arguments[10],
        gradient_mask=7,
        module=wavefront_module,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(
            candidate,
            reference,
            rtol=1.5e-3,
            atol=1.5e-3,
        )


def test_unbounded_replay_vjp_preserves_hard_route_fitting(wavefront_module):
    seq_len = 256
    generator = torch.Generator(device="cuda").manual_seed(0)
    query = (
        2
        * torch.randint(
            0,
            2,
            (1, seq_len, 1, 8),
            generator=generator,
            device="cuda",
        )
        - 1
    ).float()
    key = (
        2
        * torch.randint(
            0,
            2,
            query.shape,
            generator=generator,
            device="cuda",
        )
        - 1
    ).float()
    target_route = seq_len // 3
    distractor_route = 2 * seq_len // 3
    query_row = seq_len - 1
    for suffix_offset in range(5):
        key[0, target_route - 1 - suffix_offset, 0] = query[
            0, query_row - suffix_offset, 0
        ]
    train_index = (0, target_route - 3, 0, 0)
    desired_sign = float(key[train_index])
    key[train_index] = -desired_sign
    for suffix_offset in range(3):
        key[0, distractor_route - 1 - suffix_offset, 0] = query[
            0, query_row - suffix_offset, 0
        ]
    key[0, distractor_route - 4, 0] = query[0, query_row - 3, 0]
    key[0, distractor_route - 4, 0, 1].neg_()
    value = -torch.ones((1, seq_len, 1, 17), device="cuda")
    value[0, target_route] = 1
    key_base = key.clone()
    key_base[train_index] = 0
    train_mask = torch.zeros_like(key)
    train_mask[train_index] = 1
    logit = torch.nn.Parameter(torch.tensor(-0.2 * desired_sign, device="cuda"))
    optimizer = torch.optim.Adam([logit], lr=0.03)
    for step in range(13):
        output = unbounded_replay_rosa_soft(
            query,
            key_base + logit * train_mask,
            value,
            group_size=128,
            module=wavefront_module,
        )
        loss = (output[:, -1:] - 1).float().square().mean()
        if float(loss.detach()) == 0.0:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) == 0.0
    assert step <= 7


def test_unbounded_replay_autograd_uses_torch_dropout_rng(wavefront_module):
    generator = torch.Generator(device="cuda").manual_seed(17021)
    query = torch.randn(
        1, 65, 2, 8, device="cuda", generator=generator,
        requires_grad=True,
    )
    key = torch.randn(
        query.shape, device="cuda", generator=generator,
        requires_grad=True,
    )
    value = torch.randn(
        1, 65, 1, 17, device="cuda", generator=generator,
        requires_grad=True,
    )

    def gradients(seed):
        torch.manual_seed(seed)
        output = unbounded_replay_rosa_soft(
            query,
            key,
            value,
            group_size=32,
            dropout_p=0.5,
            module=wavefront_module,
        )
        return torch.autograd.grad(output.float().sum(), (query, key, value))

    first = gradients(7001)
    replay = gradients(7001)
    second = gradients(7002)
    for candidate, reference in zip(first, replay):
        torch.testing.assert_close(candidate, reference, rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(candidate, reference)
        for candidate, reference in zip(first, second)
    )


@pytest.mark.parametrize("seq_len", [1, 2, 15, 16, 17, 31, 32, 33, 49])
@pytest.mark.parametrize("bits", [1, 8, 32])
@pytest.mark.parametrize("window", [16, 32, 64])
@pytest.mark.parametrize("all_match", [False, True])
def test_wavefront_scores_match_prefix_product_oracle(
    seq_len,
    bits,
    window,
    all_match,
    wavefront_module,
):
    query, key, packed_query, packed_key = _case(
        seq_len,
        bits,
        all_match=all_match,
    )
    window = min(window, seq_len)
    expected = _reference_scores(query, key, window, 3.0)
    for plan in ("sequential", "affine"):
        actual = wavefront_scores(
            packed_query,
            packed_key,
            symbol_dim=bits,
            max_suffix_length=window,
            mismatch_scale=3.0,
            plan=plan,
            module=wavefront_module,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("seq_len", [1, 17, 33])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("all_match", [False, True])
def test_wavefront_online_stats_match_reference(
    seq_len,
    dropout_p,
    all_match,
    wavefront_module,
):
    bits = 8
    query, key, packed_query, packed_key = _case(
        seq_len,
        bits,
        all_match=all_match,
    )
    generator = torch.Generator(device="cuda").manual_seed(93000 + seq_len)
    value = torch.randn(
        1,
        seq_len,
        1,
        7,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        1,
        seq_len,
        2,
        7,
        device="cuda",
        generator=generator,
    )
    seed = (
        torch.tensor(987654321, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    window = min(32, seq_len)
    scale = 2.0
    mask = _causal_route_mask(seq_len, query.device)
    scores = _reference_scores(query, key, window, 3.0)
    route_scores = _masked_route_scores(
        _suffix_score_utility(scores),
        mask,
    )
    probabilities = _route_probabilities(route_scores, mask, scale)
    dropout_scales = _apply_attention_dropout(
        torch.ones_like(probabilities),
        dropout_p,
        seed,
        0,
    )
    route_values = _expand_value_heads(
        _hard_sign(value),
        query.size(2),
    )
    route_values[..., 0, :] = 0.0
    utilities = torch.einsum(
        "bhtd,bhad->bhta",
        grad_output.permute(0, 2, 1, 3),
        route_values,
    )
    expected_utility = (
        probabilities * dropout_scales * utilities
    ).sum(dim=-1)
    route_index = torch.arange(seq_len, device=query.device).view(
        1, 1, 1, seq_len
    )
    nonnull = mask.view(1, 1, seq_len, seq_len) & (route_index > 0)
    count = nonnull.sum(dim=-1, keepdim=True).clamp_min(1)
    logits = route_scores * scale - torch.where(
        nonnull,
        count.to(route_scores.dtype).log(),
        torch.zeros((), device=query.device),
    )
    maximum = logits.amax(dim=-1)
    normalizer = torch.exp(logits - maximum.unsqueeze(-1)).masked_fill(
        ~mask.view(1, 1, seq_len, seq_len),
        0.0,
    ).sum(dim=-1)
    expected = torch.stack(
        (maximum, normalizer, expected_utility),
        dim=-1,
    )
    actual = wavefront_stats(
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
        symbol_dim=bits,
        max_suffix_length=window,
        scale=scale,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
        module=wavefront_module,
    )
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-5)
    scalar = wavefront_stats(
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
        symbol_dim=bits,
        max_suffix_length=window,
        scale=scale,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
        utility_plan="scalar",
        module=wavefront_module,
    )
    torch.testing.assert_close(scalar, expected, rtol=4e-5, atol=4e-5)
    persistent = persistent_wavefront_stats(
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
        symbol_dim=bits,
        max_suffix_length=window,
        scale=scale,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
        module=wavefront_module,
    )
    torch.testing.assert_close(persistent, expected, rtol=4e-5, atol=4e-5)
    persistent_scalar = persistent_wavefront_stats(
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
        symbol_dim=bits,
        max_suffix_length=window,
        scale=scale,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
        utility_plan="scalar",
        module=wavefront_module,
    )
    torch.testing.assert_close(
        persistent_scalar,
        expected,
        rtol=4e-5,
        atol=4e-5,
    )


@pytest.mark.parametrize("seq_len", [2, 15, 16, 17, 33, 49])
@pytest.mark.parametrize("bits", [1, 8, 32])
@pytest.mark.parametrize("window", [16, 32, 64])
@pytest.mark.parametrize("all_match", [False, True])
def test_reverse_wavefront_matches_log_gate_vjp_oracle(
    seq_len,
    bits,
    window,
    all_match,
    wavefront_module,
):
    query, key, packed_query, packed_key = _case(
        seq_len,
        bits,
        all_match=all_match,
    )
    window = min(window, seq_len)
    mask = _causal_route_mask(seq_len, query.device)
    gates = _pairwise_soft_match_gates(query, key, mask, 3.0)
    scores = _suffix_prefix_product_scores(gates, window)
    generator = torch.Generator(device="cuda").manual_seed(
        97000 + seq_len * 11 + bits + window
    )
    raw_score_vjp = torch.randn(
        scores.shape,
        device="cuda",
        generator=generator,
    ) * mask.view(1, 1, seq_len, seq_len)
    expected = diagonal_suffix_log_gate_vjp(
        gates,
        raw_score_vjp,
        window,
    )
    actual = wavefront_log_gate_vjp(
        packed_query,
        packed_key,
        scores,
        raw_score_vjp,
        symbol_dim=bits,
        max_suffix_length=window,
        mismatch_scale=3.0,
        module=wavefront_module,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)


def test_wave_tiles_have_disjoint_row_column_and_diagonal_writes():
    tile = 16
    for tile_count in range(1, 17):
        for wave in range(2 * tile_count - 1):
            coordinates = [
                (row_tile, wave - row_tile)
                for row_tile in range(tile_count)
                if 0 <= wave - row_tile <= row_tile
            ]
            rows = [row for row, _ in coordinates]
            columns = [column for _, column in coordinates]
            assert len(rows) == len(set(rows))
            assert len(columns) == len(set(columns))
            diagonal_sets = []
            for row_tile, column_tile in coordinates:
                center = (row_tile - column_tile) * tile
                diagonal_sets.append(
                    set(range(center - tile + 1, center + tile))
                )
            for left_index, left in enumerate(diagonal_sets):
                for right in diagonal_sets[left_index + 1 :]:
                    assert left.isdisjoint(right)


@pytest.mark.parametrize("seq_len", [1, 2, 17, 33])
@pytest.mark.parametrize("bits", [1, 8, 32])
@pytest.mark.parametrize("mask", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("grouped_value", [False, True])
def test_full_wavefront_vjp_matches_production(
    seq_len,
    bits,
    mask,
    grouped_value,
    wavefront_module,
):
    query, key, packed_query, packed_key = _case(seq_len, bits)
    generator = torch.Generator(device="cuda").manual_seed(
        99000 + seq_len * 13 + bits + mask
    )
    value_heads = 1 if grouped_value else 2
    value = torch.randn(
        1,
        seq_len,
        value_heads,
        7,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        1,
        seq_len,
        2,
        7,
        device="cuda",
        generator=generator,
    )
    seed = torch.empty(0, dtype=torch.int64, device="cuda")
    window = min(32, seq_len)
    expected = torch.ops.rosa_soft.surrogate_vjp_masked(
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        seed,
        window,
        2.0,
        0.0,
        3.0,
        mask,
    )
    for plan in ("multilaunch", "persistent"):
        actual = wavefront_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            max_suffix_length=window,
            scale=2.0,
            dropout_p=0.0,
            mismatch_scale=3.0,
            gradient_mask=mask,
            plan=plan,
            module=wavefront_module,
        )
        for candidate, reference in zip(actual, expected):
            if candidate.numel() == 0:
                assert reference.numel() == 0
            else:
                torch.testing.assert_close(
                    candidate,
                    reference,
                    rtol=5e-4,
                    atol=5e-4,
                )


def _vjp_arguments(
    *,
    seq_len,
    window,
    dtype=torch.float32,
    dropout_p=0.0,
    pattern="random",
    batch=1,
    heads=2,
    value_heads=1,
    bits=8,
    value_dim=17,
):
    generator = torch.Generator(device="cuda").manual_seed(
        120000 + seq_len + bits + value_dim
    )
    query = torch.randn(
        batch,
        seq_len,
        heads,
        bits,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(query.shape, dtype=dtype, device="cuda", generator=generator)
    if pattern == "all_match":
        query.fill_(1)
        key.fill_(1)
    elif pattern == "one_mismatch":
        query.fill_(1)
        key.fill_(1)
        key[:, seq_len // 2, :, 0] = -1
    elif pattern == "alternating":
        signs = torch.where(
            torch.arange(seq_len, device="cuda") % 2 == 0,
            1,
            -1,
        ).to(dtype)
        query.copy_(signs.view(1, seq_len, 1, 1))
        key.copy_(signs.view(1, seq_len, 1, 1))
    elif pattern != "random":
        raise ValueError(f"unknown pattern: {pattern}")
    value = torch.randn(
        batch,
        seq_len,
        value_heads,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        batch,
        seq_len,
        heads,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query,
        key,
        value,
    )
    seed = (
        torch.tensor(87364521, dtype=torch.int64, device="cuda")
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
        min(window, seq_len),
        1.7,
        dropout_p,
        3.0,
    )


def _assert_full_vjp_parity(arguments, module, *, mask=7, atol=6e-4):
    expected = torch.ops.rosa_soft.surrogate_vjp_masked(*arguments, mask)
    for plan in ("multilaunch", "persistent"):
        actual = wavefront_vjp(
            *arguments[:7],
            max_suffix_length=arguments[7],
            scale=arguments[8],
            dropout_p=arguments[9],
            mismatch_scale=arguments[10],
            gradient_mask=mask,
            plan=plan,
            module=module,
        )
        for candidate, reference in zip(actual, expected):
            if reference.numel() == 0:
                assert candidate.numel() == 0
            else:
                torch.testing.assert_close(
                    candidate,
                    reference,
                    rtol=6e-4,
                    atol=atol,
                )


@pytest.mark.parametrize("window", [1, 15, 16, 17, 31, 32, 33, 65])
def test_full_wavefront_vjp_window_boundaries(window, wavefront_module):
    _assert_full_vjp_parity(
        _vjp_arguments(seq_len=73, window=window),
        wavefront_module,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("pattern", ["random", "all_match"])
def test_full_wavefront_vjp_dtype_dropout_and_pattern(
    dtype,
    dropout_p,
    pattern,
    wavefront_module,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    _assert_full_vjp_parity(
        _vjp_arguments(
            seq_len=97,
            window=32,
            dtype=dtype,
            dropout_p=dropout_p,
            pattern=pattern,
        ),
        wavefront_module,
        atol=8e-4,
    )


@pytest.mark.parametrize("pattern", ["one_mismatch", "alternating"])
def test_full_wavefront_vjp_structured_suffix_patterns(
    pattern,
    wavefront_module,
):
    _assert_full_vjp_parity(
        _vjp_arguments(seq_len=97, window=32, pattern=pattern),
        wavefront_module,
    )


def test_full_wavefront_vjp_irregular_grouped_dimensions(wavefront_module):
    _assert_full_vjp_parity(
        _vjp_arguments(
            seq_len=37,
            window=19,
            dropout_p=0.2,
            batch=2,
            heads=6,
            value_heads=3,
            bits=32,
            value_dim=65,
        ),
        wavefront_module,
        atol=9e-4,
    )


def test_wavefront_vjp_preserves_hard_route_fitting(wavefront_module):
    seq_len = 256
    generator = torch.Generator(device="cuda").manual_seed(0)
    query = (
        2
        * torch.randint(
            0,
            2,
            (1, seq_len, 1, 8),
            generator=generator,
            device="cuda",
        )
        - 1
    ).float()
    key = (
        2
        * torch.randint(
            0,
            2,
            query.shape,
            generator=generator,
            device="cuda",
        )
        - 1
    ).float()
    target_route = seq_len // 3
    distractor_route = 2 * seq_len // 3
    query_row = seq_len - 1
    for suffix_offset in range(5):
        key[0, target_route - 1 - suffix_offset, 0] = query[
            0, query_row - suffix_offset, 0
        ]
    train_index = (0, target_route - 3, 0, 0)
    desired_sign = float(key[train_index])
    key[train_index] = -desired_sign
    for suffix_offset in range(3):
        key[0, distractor_route - 1 - suffix_offset, 0] = query[
            0, query_row - suffix_offset, 0
        ]
    key[0, distractor_route - 4, 0] = query[0, query_row - 3, 0]
    key[0, distractor_route - 4, 0, 1].neg_()
    value = -torch.ones((1, seq_len, 1, 17), device="cuda")
    value[0, target_route] = 1
    key_base = key.clone()
    key_base[train_index] = 0
    train_mask = torch.zeros_like(key)
    train_mask[train_index] = 1

    first_success = {}
    for plan in ("multilaunch", "persistent"):
        logit = torch.nn.Parameter(
            torch.tensor(-0.2 * desired_sign, device="cuda")
        )
        optimizer = torch.optim.Adam([logit], lr=0.03)
        for step in range(13):
            output = wavefront_rosa_soft(
                query,
                key_base + logit * train_mask,
                value,
                max_suffix_length=32,
                plan=plan,
                module=wavefront_module,
            )
            loss = (output[:, -1:] - 1).float().square().mean()
            if float(loss.detach()) == 0.0:
                first_success[plan] = step
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        assert float(loss.detach()) == 0.0

    assert first_success == {"multilaunch": 7, "persistent": 7}
