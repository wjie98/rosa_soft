import shutil

import pytest
import torch

from benchmarks.sam_bitflip import sam_bitflip
from benchmarks.sam_bitflip_native import (
    NativeSamBitflip,
    build_native_sam_bitflip,
)
from benchmarks.sam_bitflip_vjp import descriptor_vjp, load_descriptor_vjp


@pytest.fixture(scope="session")
def descriptor_vjp_backends(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for descriptor VJP tests")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for descriptor VJP tests")
    output = tmp_path_factory.mktemp("sam_bitflip_vjp") / "libsam.so"
    native = NativeSamBitflip(
        build_native_sam_bitflip(output, compiler=compiler)
    )
    try:
        module = load_descriptor_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA extension toolchain unavailable: {error}")
    return native, module


@pytest.mark.parametrize("sequence_length", [0, 1, 2, 7, 17])
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_descriptor_vjp_matches_float64_oracle(
    sequence_length,
    bit_width,
    descriptor_vjp_backends,
):
    native, module = descriptor_vjp_backends
    generator = torch.Generator().manual_seed(
        2701 + 31 * sequence_length + bit_width
    )
    query = torch.randint(
        1 << bit_width,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    key = torch.randint(
        1 << bit_width,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    value = torch.randn(
        sequence_length, 13, dtype=torch.float64, generator=generator
    )
    grad_output = torch.randn(
        sequence_length, 13, dtype=torch.float64, generator=generator
    )
    expected = sam_bitflip(
        query,
        key,
        bit_width,
        value=value,
        grad_output=grad_output,
    ).bit_gradient
    factorized = native.solve_factorized(query, key, bit_width)
    actual = descriptor_vjp(
        factorized,
        query,
        key,
        value.cuda(),
        grad_output.cuda(),
        module=module,
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 2e-5, 2e-5),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 2e-2, 3e-1),
    ],
)
def test_descriptor_vjp_matches_structured_mixed_precision(
    dtype,
    rtol,
    atol,
    descriptor_vjp_backends,
):
    native, module = descriptor_vjp_backends
    sequence_length = 32
    query = torch.tensor([0, 1, 2, 3] * 8, dtype=torch.uint8)
    key = torch.tensor([3, 0, 1, 2] * 8, dtype=torch.uint8)
    generator = torch.Generator().manual_seed(2903)
    value = torch.randn(
        sequence_length, 31, dtype=torch.float32, generator=generator
    ).to(dtype)
    grad_output = torch.randn(
        sequence_length, 31, dtype=torch.float32, generator=generator
    ).to(dtype)
    expected = sam_bitflip(
        query,
        key,
        2,
        value=value,
        grad_output=grad_output,
    ).bit_gradient
    actual = descriptor_vjp(
        native.solve_factorized(query, key, 2),
        query,
        key,
        value.cuda(),
        grad_output.cuda(),
        module=module,
    ).cpu()
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=rtol, atol=atol
    )
