import pytest
import torch

from benchmarks.tensor_core_vjp import (
    GATE_METHODS,
    MATMUL_METHODS,
    batched_matmul,
    gate_matrix,
    load_tensor_core_vjp,
    suffix_scores,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8,
    reason="the isolated Tensor-Core study requires an Ampere-or-newer CUDA GPU",
)


@pytest.fixture(scope="session")
def tensor_core_module():
    try:
        return load_tensor_core_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"Tensor-Core CUDA toolchain unavailable: {error}")


def _codes(count, bits, *, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if bits == 32:
        return torch.randint(
            -(2**31),
            2**31,
            (count,),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
    return torch.randint(
        0,
        1 << bits,
        (count,),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16, 31, 32])
@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (19, 23), (32, 33)])
def test_tensor_core_gate_methods_are_exact(
    bits,
    shape,
    tensor_core_module,
):
    query = _codes(shape[0], bits, seed=1000 + bits + shape[0])
    key = _codes(shape[1], bits, seed=2000 + bits + shape[1])
    expected = gate_matrix(
        query,
        key,
        symbol_bits=bits,
        mismatch_scale=3.0,
        method="scalar",
        module=tensor_core_module,
    )
    for method in GATE_METHODS[1:]:
        actual = gate_matrix(
            query,
            key,
            symbol_bits=bits,
            mismatch_scale=3.0,
            method=method,
            module=tensor_core_module,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "shape",
    [(2, 7, 5, 9), (3, 16, 16, 16), (2, 32, 64, 24), (1, 64, 64, 8)],
)
def test_tensor_core_contractions_match_fp32(shape, tensor_core_module):
    batch, rows, inner, columns = shape
    generator = torch.Generator(device="cuda").manual_seed(sum(shape) + 7100)
    left = torch.randn(
        batch,
        rows,
        inner,
        generator=generator,
        device="cuda",
    )
    right = torch.randn(
        batch,
        inner,
        columns,
        generator=generator,
        device="cuda",
    )
    expected = torch.bmm(left, right)
    tolerances = {
        "scalar": (3e-6, 2e-5),
        "fp16": (8e-4, 2e-2),
        "bfloat16": (8e-3, 1e-1),
        "tf32": (8e-4, 2e-2),
        "fp16_scaled": (8e-4, 2e-2),
        "fp16_hilo": (3e-6, 8e-5),
    }
    for method in MATMUL_METHODS:
        actual = batched_matmul(
            left,
            right,
            method=method,
            module=tensor_core_module,
        )
        rtol, atol = tolerances[method]
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_scaled_fp16_handles_large_dynamic_range(tensor_core_module):
    generator = torch.Generator(device="cuda").manual_seed(8100)
    left = 1e5 * torch.randn(4, 32, 64, generator=generator, device="cuda")
    right = 1e-5 * torch.randn(4, 64, 32, generator=generator, device="cuda")
    expected = torch.bmm(left, right)
    actual = batched_matmul(
        left,
        right,
        method="fp16_scaled",
        module=tensor_core_module,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-2)


def test_hilo_reduces_gate_adjoint_error(tensor_core_module):
    generator = torch.Generator(device="cuda").manual_seed(8200)
    gate_adjoint = torch.randn(
        8, 64, 64, generator=generator, device="cuda"
    )
    key_derivative = torch.randn(
        8, 64, 32, generator=generator, device="cuda"
    )
    expected = torch.bmm(gate_adjoint, key_derivative)
    plain = batched_matmul(
        gate_adjoint,
        key_derivative,
        method="fp16",
        module=tensor_core_module,
    )
    hilo = batched_matmul(
        gate_adjoint,
        key_derivative,
        method="fp16_hilo",
        module=tensor_core_module,
    )
    plain_error = (plain - expected).norm()
    hilo_error = (hilo - expected).norm()
    assert hilo_error < plain_error * 0.01


def test_utility_and_value_gradient_contractions(tensor_core_module):
    generator = torch.Generator(device="cuda").manual_seed(8300)
    grad_output = torch.randn(
        6, 32, 96, generator=generator, device="cuda"
    )
    values = torch.randn(6, 32, 96, generator=generator, device="cuda")
    probabilities = torch.softmax(
        torch.randn(6, 32, 32, generator=generator, device="cuda"), dim=-1
    )

    utility_expected = torch.bmm(grad_output, values.transpose(1, 2))
    value_gradient_expected = torch.bmm(
        probabilities.transpose(1, 2), grad_output
    )
    for method in ["fp16", "tf32", "fp16_scaled", "fp16_hilo"]:
        utility = batched_matmul(
            grad_output,
            values.transpose(1, 2).contiguous(),
            method=method,
            module=tensor_core_module,
        )
        value_gradient = batched_matmul(
            probabilities.transpose(1, 2).contiguous(),
            grad_output,
            method=method,
            module=tensor_core_module,
        )
        tolerance = 8e-5 if method == "fp16_hilo" else 3e-2
        torch.testing.assert_close(
            utility, utility_expected, rtol=1e-3, atol=tolerance
        )
        torch.testing.assert_close(
            value_gradient,
            value_gradient_expected,
            rtol=1e-3,
            atol=tolerance,
        )


@pytest.mark.parametrize("window", [1, 2, 4, 8, 16, 31, 32])
@pytest.mark.parametrize("length", [1, 17, 32, 33, 65, 129])
def test_warp_suffix_scan_matches_direct(
    window,
    length,
    tensor_core_module,
):
    generator = torch.Generator(device="cuda").manual_seed(
        9000 + 31 * window + length
    )
    gates = 0.1 + 0.8 * torch.rand(
        11,
        length,
        generator=generator,
        device="cuda",
    )
    expected = suffix_scores(
        gates,
        max_suffix_length=window,
        method="direct",
        module=tensor_core_module,
    )
    actual = suffix_scores(
        gates,
        max_suffix_length=window,
        method="warp",
        module=tensor_core_module,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("window", [1, 7, 16, 31, 32])
def test_warp_suffix_scan_preserves_exact_match_limit(
    window,
    tensor_core_module,
):
    gates = torch.ones(9, 97, device="cuda")
    expected_row = torch.arange(
        1, 98, dtype=torch.float32, device="cuda"
    ).clamp_max(window)
    actual = suffix_scores(
        gates,
        max_suffix_length=window,
        method="warp",
        module=tensor_core_module,
    )
    torch.testing.assert_close(
        actual,
        expected_row.expand_as(actual),
        rtol=0.0,
        atol=0.0,
    )
