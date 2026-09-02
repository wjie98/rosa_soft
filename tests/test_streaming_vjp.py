import pytest
import torch

import rosa_soft
from benchmarks.streaming_vjp import load_streaming_vjp, streaming_vjp


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA is required for tiled streaming VJP tests",
)


@pytest.fixture(scope="session")
def streaming_module():
    try:
        return load_streaming_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"streaming VJP CUDA toolchain unavailable: {error}")


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
        seed=1000 + seq_len,
        dtype=dtype,
    )
    key = _nonzero_randn(
        query.shape, seed=2000 + seq_len, dtype=dtype
    )
    if pattern == "all_match":
        query = torch.ones_like(query)
        key = torch.ones_like(key)
    value = _nonzero_randn(
        (batch, seq_len, value_heads, value_dim),
        seed=3000 + seq_len,
        dtype=dtype,
    )
    grad_output = _nonzero_randn(
        (batch, seq_len, heads, value_dim),
        seed=4000 + seq_len,
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


def _compare(arguments, mask, tile_size, module, *, atol=3e-5):
    expected = torch.ops.rosa_soft.surrogate_vjp_masked(*arguments, mask)
    actual = streaming_vjp(
        *arguments[:7],
        max_suffix_length=arguments[7],
        scale=arguments[8],
        dropout_p=arguments[9],
        mismatch_scale=arguments[10],
        gradient_mask=mask,
        query_tile_size=tile_size,
        module=module,
    )
    for expected_gradient, actual_gradient in zip(expected, actual):
        if expected_gradient.numel() == 0:
            assert actual_gradient.numel() == 0
        else:
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=3e-4,
                atol=atol,
            )


@pytest.mark.parametrize("gradient_mask", range(1, 8))
@pytest.mark.parametrize("query_tile_size", [16, 32])
def test_streaming_vjp_matches_all_gradient_masks(
    gradient_mask,
    query_tile_size,
    streaming_module,
):
    _compare(
        _arguments(seq_len=37, window=32, dropout_p=0.2),
        gradient_mask,
        query_tile_size,
        streaming_module,
    )


@pytest.mark.parametrize("window", [1, 31, 32, 33, 65])
def test_streaming_vjp_preserves_suffix_tile_boundaries(
    window,
    streaming_module,
):
    _compare(
        _arguments(seq_len=73, window=window),
        7,
        32,
        streaming_module,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_streaming_vjp_preserves_supported_dtypes(dtype, streaming_module):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    _compare(
        _arguments(seq_len=33, window=17, dtype=dtype, dropout_p=0.2),
        7,
        16,
        streaming_module,
        atol=5e-5,
    )


def test_streaming_vjp_handles_full_bit_and_value_tiles(streaming_module):
    _compare(
        _arguments(
            seq_len=29,
            batch=2,
            heads=6,
            value_heads=3,
            bits=32,
            value_dim=65,
            window=19,
            dropout_p=0.2,
        ),
        7,
        16,
        streaming_module,
        atol=7e-5,
    )


@pytest.mark.parametrize("seq_len,window", [(1, 1), (257, 128)])
def test_streaming_vjp_handles_degenerate_and_all_match_inputs(
    seq_len,
    window,
    streaming_module,
):
    _compare(
        _arguments(
            seq_len=seq_len,
            window=window,
            pattern="all_match",
        ),
        7,
        32,
        streaming_module,
        atol=2e-4,
    )


@pytest.mark.parametrize(
    "bits,query_tile_size",
    [(8, 32), (16, 32), (32, 32)],
)
def test_production_long_sequence_dispatch_matches_streaming_vjp(
    bits,
    query_tile_size,
    streaming_module,
):
    _compare(
        _arguments(
            seq_len=4096,
            heads=1,
            bits=bits,
            value_dim=8,
            window=33,
            dropout_p=0.2,
        ),
        7,
        query_tile_size,
        streaming_module,
        atol=8e-5,
    )
