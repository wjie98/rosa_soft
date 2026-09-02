import pytest
import torch

from benchmarks.block_pipeline import block_pipeline, load_block_pipeline


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8,
    reason="the block pipeline study requires Ampere CUDA",
)


@pytest.fixture(scope="session")
def block_pipeline_module():
    try:
        return load_block_pipeline()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")


def _codes(shape, bits, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if bits == 32:
        return torch.randint(
            -(2**31),
            2**31,
            shape,
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
    return torch.randint(
        0,
        1 << bits,
        shape,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )


@pytest.mark.parametrize("seq_len", [128, 256, 512])
@pytest.mark.parametrize("bits", [8, 32])
@pytest.mark.parametrize("all_match", [False, True])
def test_warp_specialized_pipeline_matches_sequential(
    seq_len,
    bits,
    all_match,
    block_pipeline_module,
):
    series = 3
    query = _codes((series, seq_len), bits, 51000 + seq_len + bits)
    key = _codes((series, seq_len), bits, 52000 + seq_len + bits)
    if all_match:
        key.copy_(query)
    generator = torch.Generator(device="cuda").manual_seed(
        53000 + seq_len + bits
    )
    grad_output = torch.randn(
        series,
        64,
        64,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        series,
        seq_len,
        64,
        device="cuda",
        generator=generator,
    )
    arguments = dict(
        symbol_dim=bits,
        module=block_pipeline_module,
    )
    expected = block_pipeline(
        query,
        key,
        grad_output,
        value,
        method="sequential",
        **arguments,
    )
    actual = block_pipeline(
        query,
        key,
        grad_output,
        value,
        method="warp_specialized",
        **arguments,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-4)
