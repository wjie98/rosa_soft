import pytest
import torch

import rosa_soft
from benchmarks.block_diagonal_vjp import (
    block_diagonal_rosa_soft,
    block_diagonal_vjp,
    load_block_diagonal_vjp,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8,
    reason="the block-diagonal VJP study currently targets Ampere CUDA",
)


@pytest.fixture(scope="session")
def block_diagonal_module():
    try:
        return load_block_diagonal_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")


def _case(
    seq_len,
    bits,
    value_dim,
    *,
    dtype=torch.float32,
    dropout_p=0.0,
    all_match=False,
):
    generator = torch.Generator(device="cuda").manual_seed(
        15000 + seq_len + bits + value_dim
    )
    query = torch.randn(
        1,
        seq_len,
        2,
        bits,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        query.shape,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    if all_match:
        query.fill_(1)
        key.fill_(1)
    value = torch.randn(
        1,
        seq_len,
        1,
        value_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        1,
        seq_len,
        2,
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
        torch.tensor(987654321, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    return query, key, value, grad_output, packed_query, packed_key, seed


def _run(arguments, mask, plan, module, dropout_p=0.0):
    return block_diagonal_vjp(
        *arguments,
        max_suffix_length=32,
        scale=2.0,
        dropout_p=dropout_p,
        mismatch_scale=3.0,
        gradient_mask=mask,
        plan=plan,
        module=module,
    )


@pytest.mark.parametrize("seq_len", [1, 2, 31, 32, 33, 63, 64, 65, 129])
@pytest.mark.parametrize("bits", [1, 8, 32])
@pytest.mark.parametrize("mask", [1, 2, 3, 4, 5, 6, 7])
def test_block_diagonal_vjp_matches_streaming_baseline(
    seq_len,
    bits,
    mask,
    block_diagonal_module,
):
    arguments = _case(seq_len, bits, 17)
    expected = _run(arguments, mask, "baseline", block_diagonal_module)
    actual = _run(arguments, mask, "block_diagonal", block_diagonal_module)
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("all_match", [False, True])
def test_block_diagonal_vjp_dtype_dropout_and_patterns(
    dtype,
    dropout_p,
    all_match,
    block_diagonal_module,
):
    arguments = _case(
        97,
        8,
        33,
        dtype=dtype,
        dropout_p=dropout_p,
        all_match=all_match,
    )
    expected = _run(
        arguments,
        7,
        "baseline",
        block_diagonal_module,
        dropout_p,
    )
    actual = _run(
        arguments,
        7,
        "block_diagonal",
        block_diagonal_module,
        dropout_p,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=4e-4, atol=4e-4)


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("mask", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("plan", ["block_tf32", "block_tf32_pipeline"])
def test_block_tf32_vjp_has_bounded_contraction_error(
    bits,
    mask,
    plan,
    block_diagonal_module,
):
    arguments = _case(97, bits, 64)
    expected = _run(arguments, mask, "baseline", block_diagonal_module)
    actual = _run(arguments, mask, plan, block_diagonal_module)
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=2e-3, atol=2e-3)


def test_block_tf32_falls_back_when_value_dimension_is_unsupported(
    block_diagonal_module,
):
    arguments = _case(97, 8, 33)
    expected = _run(arguments, 7, "baseline", block_diagonal_module)
    actual = _run(arguments, 7, "block_tf32", block_diagonal_module)
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=5e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("all_match", [False, True])
@pytest.mark.parametrize("plan", ["block_tf32", "block_tf32_pipeline"])
def test_block_tf32_vjp_is_robust_to_dtype_dropout_and_pattern(
    dtype,
    dropout_p,
    all_match,
    plan,
    block_diagonal_module,
):
    arguments = _case(
        129,
        8,
        64,
        dtype=dtype,
        dropout_p=dropout_p,
        all_match=all_match,
    )
    expected = _run(
        arguments,
        7,
        "baseline",
        block_diagonal_module,
        dropout_p,
    )
    actual = _run(
        arguments,
        7,
        plan,
        block_diagonal_module,
        dropout_p,
    )
    for candidate, reference in zip(actual, expected):
        difference = candidate - reference
        relative = difference.norm() / reference.norm().clamp_min(1e-20)
        assert float(relative) < 2e-3
        torch.testing.assert_close(candidate, reference, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("all_match", [False, True])
@pytest.mark.parametrize("mask", [4, 7])
def test_production_block_dispatch_matches_streaming_baseline(
    dtype,
    dropout_p,
    all_match,
    mask,
    block_diagonal_module,
):
    arguments = _case(
        4096,
        8,
        64,
        dtype=dtype,
        dropout_p=dropout_p,
        all_match=all_match,
    )
    expected = _run(
        arguments,
        mask,
        "baseline",
        block_diagonal_module,
        dropout_p,
    )
    actual = torch.ops.rosa_soft.surrogate_vjp_masked(
        *arguments,
        32,
        2.0,
        dropout_p,
        3.0,
        mask,
    )
    for candidate, reference in zip(actual, expected):
        if candidate.numel() == 0:
            continue
        relative = (
            (candidate.float() - reference.float()).norm()
            / reference.float().norm().clamp_min(1e-20)
        )
        assert float(relative) < 3e-3


def test_production_block_dispatch_preserves_long_route_fitting(
    block_diagonal_module,
):
    del block_diagonal_module
    seq_len = 4096
    generator = torch.Generator(device="cuda").manual_seed(0)
    query = (
        2 * torch.randint(
            0,
            2,
            (1, seq_len, 1, 8),
            generator=generator,
            device="cuda",
        )
        - 1
    ).float()
    key = (
        2 * torch.randint(
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
    value = -torch.ones((1, seq_len, 1, 64), device="cuda")
    value[0, target_route] = 1
    key_base = key.clone()
    key_base[train_index] = 0
    train_mask = torch.zeros_like(key)
    train_mask[train_index] = 1

    def fit(plan):
        logit = torch.nn.Parameter(
            torch.tensor(-0.2 * desired_sign, device="cuda")
        )
        optimizer = torch.optim.Adam([logit], lr=0.03)
        first_success = None
        for step in range(13):
            fitted_key = key_base + logit * train_mask
            if plan == "production":
                output = rosa_soft.rosa_soft(
                    query,
                    fitted_key,
                    value,
                    max_suffix_length=32,
                )
            else:
                output = block_diagonal_rosa_soft(
                    query,
                    fitted_key,
                    value,
                    max_suffix_length=32,
                    plan="baseline",
                )
            loss = (output[:, -1:] - 1).float().square().mean()
            if float(loss.detach()) == 0.0 and first_success is None:
                first_success = step
            if step == 12:
                return first_success, float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    production_step, production_loss = fit("production")
    streaming_step, streaming_loss = fit("streaming")
    assert production_loss == streaming_loss == 0.0
    assert production_step is not None and streaming_step is not None
    assert abs(production_step - streaming_step) <= 1


@pytest.mark.parametrize(
    ("plan", "tolerance"),
    [
        ("block_diagonal", 5e-4),
        ("block_tf32", 3e-3),
        ("block_tf32_pipeline", 3e-3),
    ],
)
def test_block_diagonal_operator_preserves_hard_forward_and_autograd(
    plan,
    tolerance,
    block_diagonal_module,
):
    del block_diagonal_module
    query, key, value, grad_output, *_ = _case(97, 8, 64)
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )
    candidate_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )
    reference = rosa_soft.rosa_soft(
        *reference_inputs,
        max_suffix_length=32,
        scale=2.0,
        mismatch_scale=3.0,
    )
    candidate = block_diagonal_rosa_soft(
        *candidate_inputs,
        max_suffix_length=32,
        scale=2.0,
        mismatch_scale=3.0,
        plan=plan,
    )
    torch.testing.assert_close(candidate, reference, rtol=0.0, atol=0.0)
    reference.backward(grad_output)
    candidate.backward(grad_output)
    for candidate_input, reference_input in zip(
        candidate_inputs,
        reference_inputs,
    ):
        torch.testing.assert_close(
            candidate_input.grad,
            reference_input.grad,
            rtol=tolerance,
            atol=tolerance,
        )
