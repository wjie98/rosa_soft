import pytest
import torch

import rosa_soft
from benchmarks.flash_tc_vjp import PLANS, flash_tc_vjp, load_flash_tc_vjp


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda
    or torch.cuda.get_device_capability()[0] < 8,
    reason="FlashROSA-TC tests require RosaSoft CUDA on Ampere or newer",
)


@pytest.fixture(scope="session")
def flash_tc_module():
    try:
        return load_flash_tc_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"FlashROSA-TC CUDA toolchain unavailable: {error}")


def _nonzero_randn(shape, *, seed, dtype=torch.float32):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(
        shape,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    return values.sign().masked_fill(values == 0, 1) * (values.abs() + 0.2)


def _arguments(
    *,
    seq_len,
    batch=1,
    heads=2,
    value_heads=1,
    bits=5,
    value_dim=7,
    window=32,
    dropout_p=0.0,
    dtype=torch.float32,
    pattern="random",
):
    query = _nonzero_randn(
        (batch, seq_len, heads, bits),
        seed=11000 + seq_len + bits,
        dtype=dtype,
    )
    key = _nonzero_randn(
        query.shape,
        seed=12000 + seq_len + bits,
        dtype=dtype,
    )
    if pattern == "all_match":
        query.fill_(1)
        key.fill_(1)
    value = _nonzero_randn(
        (batch, seq_len, value_heads, value_dim),
        seed=13000 + seq_len + value_dim,
        dtype=dtype,
    )
    grad_output = _nonzero_randn(
        (batch, seq_len, heads, value_dim),
        seed=14000 + seq_len + value_dim,
        dtype=dtype,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = (
        torch.tensor(987654321, dtype=torch.int64, device="cuda")
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
        dropout_seed,
        window,
        1.7,
        dropout_p,
        3.0,
    )


def _compare(arguments, mask, plan, module, *, atol=8e-5, rtol=3e-4):
    expected = torch.ops.rosa_soft.surrogate_vjp_masked(*arguments, mask)
    actual = flash_tc_vjp(
        *arguments[:7],
        max_suffix_length=arguments[7],
        scale=arguments[8],
        dropout_p=arguments[9],
        mismatch_scale=arguments[10],
        gradient_mask=mask,
        plan=plan,
        module=module,
    )
    for actual_gradient, expected_gradient in zip(actual, expected):
        if expected_gradient.numel() == 0:
            assert actual_gradient.numel() == 0
        else:
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=rtol,
                atol=atol,
            )


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("gradient_mask", range(1, 8))
def test_flash_tc_plans_match_all_gradient_masks(
    plan,
    gradient_mask,
    flash_tc_module,
):
    _compare(
        _arguments(seq_len=37, window=32, dropout_p=0.2),
        gradient_mask,
        plan,
        flash_tc_module,
    )


@pytest.mark.parametrize("window", [1, 2, 4, 8, 16, 31, 32, 33, 65])
@pytest.mark.parametrize("plan", PLANS[1:])
def test_flash_tc_preserves_suffix_boundaries(
    window,
    plan,
    flash_tc_module,
):
    _compare(
        _arguments(seq_len=73, window=window),
        7,
        plan,
        flash_tc_module,
        atol=1.5e-4,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("plan", PLANS[1:])
def test_flash_tc_preserves_supported_dtypes(dtype, plan, flash_tc_module):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    _compare(
        _arguments(
            seq_len=33,
            window=17,
            dtype=dtype,
            dropout_p=0.2,
        ),
        7,
        plan,
        flash_tc_module,
        atol=2e-4,
    )


@pytest.mark.parametrize("bits", [1, 8, 16, 32])
@pytest.mark.parametrize("value_dim", [1, 31, 32, 33, 64, 96, 128])
def test_fused_plan_handles_symbol_and_value_tiles(
    bits,
    value_dim,
    flash_tc_module,
):
    _compare(
        _arguments(
            seq_len=35,
            batch=2,
            heads=6,
            value_heads=3,
            bits=bits,
            value_dim=value_dim,
            window=19,
            dropout_p=0.2,
        ),
        7,
        "fused",
        flash_tc_module,
        atol=3e-4,
    )


@pytest.mark.parametrize(
    "seq_len,window,pattern",
    [(1, 1, "random"), (32, 32, "all_match"), (65, 32, "all_match")],
)
@pytest.mark.parametrize("plan", PLANS[1:])
def test_flash_tc_handles_degenerate_and_exact_match_inputs(
    seq_len,
    window,
    pattern,
    plan,
    flash_tc_module,
):
    _compare(
        _arguments(
            seq_len=seq_len,
            window=window,
            pattern=pattern,
            bits=8,
            value_dim=65,
        ),
        7,
        plan,
        flash_tc_module,
        atol=4e-4,
    )


def test_fused_plan_matches_long_streaming_tile(flash_tc_module):
    _compare(
        _arguments(
            seq_len=1025,
            heads=1,
            bits=8,
            value_dim=64,
            window=32,
            dropout_p=0.2,
        ),
        7,
        "fused",
        flash_tc_module,
        atol=4e-4,
    )


@pytest.mark.parametrize("upstream_scale", [1e-6, 1e6])
def test_tensor_gate_handles_upstream_dynamic_range(
    upstream_scale,
    flash_tc_module,
):
    arguments = list(
        _arguments(
            seq_len=35,
            heads=2,
            bits=32,
            value_dim=33,
            window=32,
        )
    )
    arguments[3] = arguments[3] * upstream_scale
    _compare(
        tuple(arguments),
        3,
        "tc_gate",
        flash_tc_module,
        atol=0.5 if upstream_scale > 1 else 1e-7,
        rtol=4e-4,
    )


@pytest.mark.parametrize("gradient_mask", range(1, 8))
def test_selected_tensor_gate_matches_all_gradient_masks(
    gradient_mask,
    flash_tc_module,
):
    _compare(
        _arguments(
            seq_len=65,
            heads=2,
            bits=32,
            value_dim=33,
            window=32,
            dropout_p=0.2,
        ),
        gradient_mask,
        "tc_gate",
        flash_tc_module,
        atol=3e-4,
    )


@pytest.mark.parametrize(
    "bits,window",
    [(16, 32), (32, 30), (32, 33)],
)
def test_tensor_gate_outside_selected_shape_uses_baseline_semantics(
    bits,
    window,
    flash_tc_module,
):
    arguments = _arguments(
        seq_len=41,
        heads=2,
        bits=bits,
        value_dim=17,
        window=window,
        dropout_p=0.2,
    )
    baseline = flash_tc_vjp(
        *arguments[:7],
        max_suffix_length=arguments[7],
        scale=arguments[8],
        dropout_p=arguments[9],
        mismatch_scale=arguments[10],
        gradient_mask=7,
        plan="baseline",
        module=flash_tc_module,
    )
    actual = flash_tc_vjp(
        *arguments[:7],
        max_suffix_length=arguments[7],
        scale=arguments[8],
        dropout_p=arguments[9],
        mismatch_scale=arguments[10],
        gradient_mask=7,
        plan="tc_gate",
        module=flash_tc_module,
    )
    for candidate, reference in zip(actual, baseline):
        torch.testing.assert_close(candidate, reference, rtol=0.0, atol=5e-7)
