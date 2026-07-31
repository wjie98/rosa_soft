import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import rosa_soft


ROOT = Path(__file__).resolve().parents[1]


def _nonzero_randn(shape, *, seed, device="cpu", dtype=torch.float32):
    generator = torch.Generator(device=device).manual_seed(seed)
    values = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    signs = torch.where(
        values >= 0,
        torch.ones_like(values),
        -torch.ones_like(values),
    )
    return signs * (values.abs() + 0.2)


def _offsets(values, *, device="cpu"):
    return torch.tensor(values, dtype=torch.int32, device=device)


def _dropout_seed(dropout_p=0.0, seed=0):
    if dropout_p == 0.0:
        return torch.empty(0, dtype=torch.int64, device="cuda")
    return torch.tensor(seed, dtype=torch.int64, device="cuda")


def test_varlen_reference_matches_independent_dense_sequences():
    query = _nonzero_randn((8, 2, 3), seed=1)
    key = _nonzero_randn((8, 2, 3), seed=2)
    value = _nonzero_randn((8, 1, 2), seed=3)
    cu_seqlens = _offsets([0, 3, 3, 8])

    torch.manual_seed(4)
    actual = rosa_soft.rosa_soft_varlen_reference(
        query,
        key,
        value,
        cu_seqlens,
        max_suffix_length=4,
        scale=1.5,
        mismatch_scale=6.0,
    )
    torch.manual_seed(4)
    expected = torch.cat(
        [
            rosa_soft.rosa_soft_reference(
                query[start:end].unsqueeze(0),
                key[start:end].unsqueeze(0),
                value[start:end].unsqueeze(0),
                max_suffix_length=4,
                scale=1.5,
                mismatch_scale=6.0,
            ).squeeze(0)
            for start, end in ((0, 3), (3, 8))
        ],
        dim=0,
    )

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[[0, 3]]) == 0


def test_varlen_reference_gradients_do_not_cross_sequence_boundaries():
    base = (
        _nonzero_randn((9, 1, 3), seed=10, dtype=torch.float64),
        _nonzero_randn((9, 1, 3), seed=11, dtype=torch.float64),
        _nonzero_randn((9, 1, 2), seed=12, dtype=torch.float64),
    )
    cu_seqlens = _offsets([0, 4, 9])
    grad_output = _nonzero_randn(
        (9, 1, 2),
        seed=13,
        dtype=torch.float64,
    )

    def run(inputs):
        leaves = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in inputs
        )
        torch.manual_seed(14)
        output = rosa_soft.rosa_soft_varlen_reference(
            *leaves,
            cu_seqlens,
            max_suffix_length=5,
            scale=2.0,
            mismatch_scale=9.0,
        )
        gradients = torch.autograd.grad(output, leaves, grad_output)
        return output.detach(), gradients

    baseline_output, baseline_gradients = run(base)
    changed = tuple(tensor.clone() for tensor in base)
    for tensor in changed:
        tensor[:4].mul_(-2.0)
    changed_output, changed_gradients = run(changed)

    assert torch.equal(changed_output[4:], baseline_output[4:])
    for changed_gradient, baseline_gradient in zip(
        changed_gradients,
        baseline_gradients,
    ):
        torch.testing.assert_close(
            changed_gradient[4:],
            baseline_gradient[4:],
            rtol=0.0,
            atol=0.0,
        )


def test_varlen_reference_never_routes_segment_initial_values():
    query = _nonzero_randn((7, 1, 2), seed=20).requires_grad_()
    key = _nonzero_randn((7, 1, 2), seed=21).requires_grad_()
    value = _nonzero_randn((7, 1, 3), seed=22).requires_grad_()
    cu_seqlens = _offsets([0, 1, 4, 7])

    output = rosa_soft.rosa_soft_varlen_reference(
        query,
        key,
        value,
        cu_seqlens,
    )
    output.sum().backward()

    assert torch.count_nonzero(output[[0, 1, 4]]) == 0
    assert torch.count_nonzero(value.grad[[0, 1, 4]]) == 0


@pytest.mark.parametrize(
    ("cu_seqlens", "message"),
    [
        ([1, 4], "start at zero"),
        ([0, 3], "packed token count"),
        ([0, 3, 2, 4], "nondecreasing"),
    ],
)
def test_varlen_reference_rejects_invalid_offsets(cu_seqlens, message):
    values = torch.ones(4, 1, 1)
    with pytest.raises(ValueError, match=message):
        rosa_soft.rosa_soft_varlen_reference(
            values,
            values,
            values,
            _offsets(cu_seqlens),
        )


def test_varlen_reference_rejects_invalid_offset_dtype_and_shape():
    values = torch.ones(4, 1, 1)
    with pytest.raises(ValueError, match="int32"):
        rosa_soft.rosa_soft_varlen_reference(
            values,
            values,
            values,
            torch.tensor([0, 4], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        rosa_soft.rosa_soft_varlen_reference(
            values,
            values,
            values,
            torch.tensor([[0, 4]], dtype=torch.int32),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
@pytest.mark.parametrize(
    "offsets",
    [
        [1, 2, 4],
        [0, 2, 3],
        [0, 3, 2, 4],
    ],
)
def test_varlen_cuda_rejects_invalid_offset_values_in_subprocess(offsets):
    script = """
import ast
import sys

import torch

import rosa_soft

offsets = ast.literal_eval(sys.argv[1])
query = torch.ones(4, 1, 2, device="cuda")
key = torch.ones_like(query)
value = torch.ones(4, 1, 2, device="cuda")
cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")
rosa_soft.rosa_soft_varlen(query, key, value, cu_seqlens)
torch.cuda.synchronize()
"""
    environment = dict(os.environ)
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, repr(offsets)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "assert" in combined_output.lower()
    assert "ModuleNotFoundError" not in combined_output


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
def test_varlen_cuda_rejects_sequence_count_that_cannot_fit_kernel_index():
    query = torch.ones(1, 1, 2, device="cuda")
    key = torch.ones_like(query)
    value = torch.ones(1, 1, 2, device="cuda")
    excessive_offsets = torch.zeros(
        1,
        dtype=torch.int32,
        device="cuda",
    ).expand(2_147_483_521)

    with pytest.raises(
        RuntimeError,
        match="number of packed sequences is too large",
    ):
        torch.ops.rosa_soft.hard_forward_varlen(
            query,
            key,
            value,
            excessive_offsets,
            1,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
def test_varlen_public_and_dispatch_signatures_are_minimal():
    expected_parameters = (
        "query",
        "key",
        "value",
        "cu_seqlens",
        "max_suffix_length",
        "scale",
        "dropout_p",
        "mismatch_scale",
    )
    assert tuple(
        inspect.signature(rosa_soft.rosa_soft_varlen).parameters
    ) == expected_parameters
    assert tuple(
        inspect.signature(
            rosa_soft.rosa_soft_varlen_reference
        ).parameters
    ) == expected_parameters
    assert str(
        torch.ops.rosa_soft.hard_forward_varlen.default._schema
    ) == (
        "rosa_soft::hard_forward_varlen(Tensor query, "
        "Tensor key, Tensor value, Tensor cu_seqlens, "
        "int max_suffix_length) -> "
        "(Tensor output, Tensor packed_query_symbols, "
        "Tensor packed_key_symbols)"
    )
    assert str(
        torch.ops.rosa_soft.surrogate_vjp_varlen_masked.default._schema
    ) == (
        "rosa_soft::surrogate_vjp_varlen_masked(Tensor query, "
        "Tensor key, Tensor value, Tensor cu_seqlens, "
        "Tensor grad_output, Tensor packed_query_symbols, "
        "Tensor packed_key_symbols, Tensor dropout_seed, "
        "int max_suffix_length, float scale, "
        "float dropout_p, "
        "float mismatch_scale, int gradient_mask) -> "
        "(Tensor grad_query, Tensor grad_key, Tensor grad_value)"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
def test_varlen_cuda_equal_lengths_match_dense_batch_forward_and_vjp(
    dropout_p,
):
    batch, length, heads, bits = 3, 6, 2, 4
    query = _nonzero_randn(
        (batch * length, heads, bits),
        seed=30,
        device="cuda",
    )
    key = _nonzero_randn(
        (batch * length, heads, bits),
        seed=31,
        device="cuda",
    )
    value = _nonzero_randn(
        (batch * length, 1, 3),
        seed=32,
        device="cuda",
    )
    cu_seqlens = _offsets(
        [index * length for index in range(batch + 1)],
        device="cuda",
    )
    grad_output = _nonzero_randn(
        (batch * length, heads, 3),
        seed=33,
        device="cuda",
    )
    packed_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )
    dense_inputs = (
        query.reshape(batch, length, heads, bits).requires_grad_(),
        key.reshape(batch, length, heads, bits).requires_grad_(),
        value.reshape(batch, length, 1, 3).requires_grad_(),
    )

    torch.cuda.manual_seed(34)
    packed_output = rosa_soft.rosa_soft_varlen(
        *packed_inputs,
        cu_seqlens,
        max_suffix_length=5,
        scale=1.7,
        dropout_p=dropout_p,
        mismatch_scale=6.0,
    )
    torch.cuda.manual_seed(34)
    dense_output = rosa_soft.rosa_soft(
        *dense_inputs,
        max_suffix_length=5,
        scale=1.7,
        dropout_p=dropout_p,
        mismatch_scale=6.0,
    )
    packed_gradients = torch.autograd.grad(
        packed_output,
        packed_inputs,
        grad_output,
    )
    dense_gradients = torch.autograd.grad(
        dense_output,
        dense_inputs,
        grad_output.reshape(batch, length, heads, 3),
    )

    assert torch.equal(
        packed_output,
        dense_output.reshape_as(packed_output),
    )
    for packed_gradient, dense_gradient in zip(
        packed_gradients,
        dense_gradients,
    ):
        torch.testing.assert_close(
            packed_gradient,
            dense_gradient.reshape_as(packed_gradient),
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
def test_varlen_cuda_variable_lengths_match_reference_vjp(dropout_p):
    lengths = (3, 0, 5, 2)
    total_tokens = sum(lengths)
    heads = 2
    bits = 3
    query = _nonzero_randn(
        (total_tokens, heads, bits),
        seed=40,
        device="cuda",
    )
    key = _nonzero_randn(
        (total_tokens, heads, bits),
        seed=41,
        device="cuda",
    )
    value = _nonzero_randn(
        (total_tokens, 1, 2),
        seed=42,
        device="cuda",
    )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = _offsets(offsets, device="cuda")
    grad_output = _nonzero_randn(
        (total_tokens, heads, 2),
        seed=43,
        device="cuda",
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )
    cuda_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )

    torch.cuda.manual_seed(44)
    reference_output = rosa_soft.rosa_soft_varlen_reference(
        *reference_inputs,
        cu_seqlens,
        max_suffix_length=4,
        scale=1.3,
        dropout_p=dropout_p,
        mismatch_scale=7.0,
    )
    torch.cuda.manual_seed(44)
    cuda_output = rosa_soft.rosa_soft_varlen(
        *cuda_inputs,
        cu_seqlens,
        max_suffix_length=4,
        scale=1.3,
        dropout_p=dropout_p,
        mismatch_scale=7.0,
    )

    assert torch.equal(cuda_output, reference_output)
    reference_gradients = torch.autograd.grad(
        reference_output,
        reference_inputs,
        grad_output,
    )
    cuda_gradients = torch.autograd.grad(
        cuda_output,
        cuda_inputs,
        grad_output,
    )
    for cuda_gradient, reference_gradient in zip(
        cuda_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            cuda_gradient,
            reference_gradient,
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
def test_varlen_cuda_long_key_aggregation_matches_reference(dropout_p):
    lengths = (256,)
    total_tokens = sum(lengths)
    query = _nonzero_randn(
        (total_tokens, 1, 8),
        seed=45,
        device="cuda",
    )
    key = _nonzero_randn(
        (total_tokens, 1, 8),
        seed=46,
        device="cuda",
    )
    value = _nonzero_randn(
        (total_tokens, 1, 3),
        seed=47,
        device="cuda",
    )
    cu_seqlens = _offsets([0, total_tokens], device="cuda")
    grad_output = _nonzero_randn(
        (total_tokens, 1, 3),
        seed=48,
        device="cuda",
    )
    controls = {
        "max_suffix_length": 4,
        "scale": 1.2,
        "dropout_p": dropout_p,
        "mismatch_scale": 3.0,
    }
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )
    cuda_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )

    torch.cuda.manual_seed(49)
    reference_output = rosa_soft.rosa_soft_varlen_reference(
        *reference_inputs,
        cu_seqlens,
        **controls,
    )
    torch.cuda.manual_seed(49)
    cuda_output = rosa_soft.rosa_soft_varlen(
        *cuda_inputs,
        cu_seqlens,
        **controls,
    )
    reference_gradients = torch.autograd.grad(
        reference_output,
        reference_inputs,
        grad_output,
    )
    cuda_gradients = torch.autograd.grad(
        cuda_output,
        cuda_inputs,
        grad_output,
    )

    assert torch.equal(cuda_output, reference_output)
    for cuda_gradient, reference_gradient in zip(
        cuda_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            cuda_gradient,
            reference_gradient,
            rtol=8e-5,
            atol=2e-5,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
def test_varlen_cuda_cooperative_qkv_matches_reference():
    total_tokens = 256
    query = _nonzero_randn(
        (total_tokens, 1, 8),
        seed=55,
        device="cuda",
    )
    key = _nonzero_randn(
        (total_tokens, 1, 8),
        seed=56,
        device="cuda",
    )
    value = _nonzero_randn(
        (total_tokens, 1, 64),
        seed=57,
        device="cuda",
    )
    grad_output = _nonzero_randn(
        (total_tokens, 1, 64),
        seed=58,
        device="cuda",
    )
    cu_seqlens = _offsets([0, total_tokens], device="cuda")
    controls = {
        "max_suffix_length": 4,
        "scale": 1.2,
        "dropout_p": 0.2,
        "mismatch_scale": 3.0,
    }
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )
    cuda_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    )

    torch.cuda.manual_seed(59)
    reference_output = rosa_soft.rosa_soft_varlen_reference(
        *reference_inputs,
        cu_seqlens,
        **controls,
    )
    torch.cuda.manual_seed(59)
    cuda_output = rosa_soft.rosa_soft_varlen(
        *cuda_inputs,
        cu_seqlens,
        **controls,
    )
    reference_gradients = torch.autograd.grad(
        reference_output,
        reference_inputs,
        grad_output,
    )
    cuda_gradients = torch.autograd.grad(
        cuda_output,
        cuda_inputs,
        grad_output,
    )

    assert torch.equal(cuda_output, reference_output)
    for cuda_gradient, reference_gradient in zip(
        cuda_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            cuda_gradient,
            reference_gradient,
            rtol=1e-4,
            atol=1e-6,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
def test_varlen_cuda_query_score_cache_matches_reference():
    total_tokens = 257
    query = _nonzero_randn(
        (total_tokens, 2, 8),
        seed=60,
        device="cuda",
    )
    key = _nonzero_randn(
        (total_tokens, 2, 8),
        seed=61,
        device="cuda",
    )
    value = _nonzero_randn(
        (total_tokens, 1, 3),
        seed=62,
        device="cuda",
    )
    grad_output = _nonzero_randn(
        (total_tokens, 2, 3),
        seed=63,
        device="cuda",
    )
    cu_seqlens = _offsets([0, total_tokens], device="cuda")
    controls = {
        "max_suffix_length": 9,
        "scale": 1.4,
        "dropout_p": 0.2,
        "mismatch_scale": 3.0,
    }
    reference_query = query.detach().clone().requires_grad_()
    cuda_query = query.detach().clone().requires_grad_()

    torch.cuda.manual_seed(64)
    reference_output = rosa_soft.rosa_soft_varlen_reference(
        reference_query,
        key,
        value,
        cu_seqlens,
        **controls,
    )
    torch.cuda.manual_seed(64)
    cuda_output = rosa_soft.rosa_soft_varlen(
        cuda_query,
        key,
        value,
        cu_seqlens,
        **controls,
    )
    reference_gradient = torch.autograd.grad(
        reference_output,
        reference_query,
        grad_output,
    )[0]
    cuda_gradient = torch.autograd.grad(
        cuda_output,
        cuda_query,
        grad_output,
    )[0]

    assert torch.equal(cuda_output, reference_output)
    torch.testing.assert_close(
        cuda_gradient,
        reference_gradient,
        rtol=1e-4,
        atol=2e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
def test_varlen_cuda_query_score_tail_cache_boundary_matches_reference():
    total_tokens = 1027
    query = _nonzero_randn(
        (total_tokens, 1, 4),
        seed=65,
        device="cuda",
    )
    key = _nonzero_randn(
        (total_tokens, 1, 4),
        seed=66,
        device="cuda",
    )
    value = _nonzero_randn(
        (total_tokens, 1, 1),
        seed=67,
        device="cuda",
    )
    grad_output = _nonzero_randn(
        (total_tokens, 1, 1),
        seed=68,
        device="cuda",
    )
    cu_seqlens = _offsets([0, total_tokens], device="cuda")
    controls = {
        "max_suffix_length": 2,
        "scale": 1.4,
        "dropout_p": 0.2,
        "mismatch_scale": 3.0,
    }
    reference_query = query.detach().clone().requires_grad_()
    cuda_query = query.detach().clone().requires_grad_()

    torch.cuda.manual_seed(69)
    reference_output = rosa_soft.rosa_soft_varlen_reference(
        reference_query,
        key,
        value,
        cu_seqlens,
        **controls,
    )
    reference_gradient = torch.autograd.grad(
        reference_output,
        reference_query,
        grad_output,
    )[0]
    torch.cuda.manual_seed(69)
    cuda_output = rosa_soft.rosa_soft_varlen(
        cuda_query,
        key,
        value,
        cu_seqlens,
        **controls,
    )
    cuda_gradient = torch.autograd.grad(
        cuda_output,
        cuda_query,
        grad_output,
    )[0]

    assert torch.equal(cuda_output, reference_output)
    torch.testing.assert_close(
        cuda_gradient,
        reference_gradient,
        rtol=2e-4,
        atol=3e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
def test_varlen_cuda_tiled_value_only_matches_reference(dropout_p):
    total_tokens = 67
    heads = 2
    value_dim = 64
    query = _nonzero_randn(
        (total_tokens, heads, 8),
        seed=50,
        device="cuda",
    )
    key = _nonzero_randn(
        (total_tokens, heads, 8),
        seed=51,
        device="cuda",
    )
    value = _nonzero_randn(
        (total_tokens, 1, value_dim),
        seed=52,
        device="cuda",
    )
    cu_seqlens = _offsets([0, total_tokens], device="cuda")
    grad_output = _nonzero_randn(
        (total_tokens, heads, value_dim),
        seed=53,
        device="cuda",
    )
    controls = {
        "max_suffix_length": 9,
        "scale": 1.2,
        "dropout_p": dropout_p,
        "mismatch_scale": 3.0,
    }
    reference_value = value.detach().clone().requires_grad_()
    cuda_value = value.detach().clone().requires_grad_()

    torch.cuda.manual_seed(54)
    reference_output = rosa_soft.rosa_soft_varlen_reference(
        query,
        key,
        reference_value,
        cu_seqlens,
        **controls,
    )
    torch.cuda.manual_seed(54)
    cuda_output = rosa_soft.rosa_soft_varlen(
        query,
        key,
        cuda_value,
        cu_seqlens,
        **controls,
    )
    reference_gradient = torch.autograd.grad(
        reference_output,
        reference_value,
        grad_output,
    )[0]
    cuda_gradient = torch.autograd.grad(
        cuda_output,
        cuda_value,
        grad_output,
    )[0]

    assert torch.equal(cuda_output, reference_output)
    torch.testing.assert_close(
        cuda_gradient,
        reference_gradient,
        rtol=3e-5,
        atol=3e-6,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_varlen_cuda_long_boundaries_and_dtypes_match_dense_reference(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    lengths = (1, 2, 127, 0, 128, 129)
    total_tokens = sum(lengths)
    query = _nonzero_randn(
        (total_tokens, 1, 4),
        seed=50,
        device="cuda",
        dtype=dtype,
    )
    key = _nonzero_randn(
        (total_tokens, 1, 4),
        seed=51,
        device="cuda",
        dtype=dtype,
    )
    value = _nonzero_randn(
        (total_tokens, 1, 2),
        seed=52,
        device="cuda",
        dtype=dtype,
    )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = _offsets(offsets, device="cuda")

    actual = rosa_soft.rosa_soft_varlen(
        query,
        key,
        value,
        cu_seqlens,
        max_suffix_length=160,
    )
    expected = torch.cat(
        [
            rosa_soft.rosa_soft_reference(
                query[start:end].unsqueeze(0),
                key[start:end].unsqueeze(0),
                value[start:end].unsqueeze(0),
                max_suffix_length=160,
            ).squeeze(0)
            for start, end in zip(offsets[:-1], offsets[1:])
            if end > start
        ],
        dim=0,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
def test_varlen_cuda_grouped_heads_and_noncontiguous_inputs():
    total_tokens = 11
    query = _nonzero_randn(
        (total_tokens, 5, 4),
        seed=60,
        device="cuda",
    ).transpose(1, 2).requires_grad_()
    key = _nonzero_randn(
        (total_tokens, 5, 4),
        seed=61,
        device="cuda",
    ).transpose(1, 2).requires_grad_()
    value = _nonzero_randn(
        (total_tokens, 3, 2),
        seed=62,
        device="cuda",
    ).transpose(1, 2).requires_grad_()
    offset_storage = _offsets([0, -1, 4, -1, 11], device="cuda")
    cu_seqlens = offset_storage[::2]
    assert not cu_seqlens.is_contiguous()

    output = rosa_soft.rosa_soft_varlen(
        query,
        key,
        value,
        cu_seqlens,
        max_suffix_length=6,
    )
    output.sum().backward()

    assert output.shape == (total_tokens, 4, 3)
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (query, key, value)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("gradient_mask", range(1, 8))
def test_varlen_cuda_masked_vjp_matches_enabled_full_vjp(
    gradient_mask,
    dropout_p,
):
    query = _nonzero_randn((9, 2, 4), seed=66, device="cuda")
    key = _nonzero_randn((9, 2, 4), seed=67, device="cuda")
    value = _nonzero_randn((9, 1, 3), seed=68, device="cuda")
    cu_seqlens = _offsets([0, 0, 3, 3, 9, 9], device="cuda")
    output, query_symbols, key_symbols = (
        torch.ops.rosa_soft.hard_forward_varlen(
            query,
            key,
            value,
            cu_seqlens,
            5,
        )
    )
    grad_output = _nonzero_randn(
        output.shape,
        seed=69,
        device="cuda",
    )
    full = torch.ops.rosa_soft.surrogate_vjp_varlen_masked(
        query,
        key,
        value,
        cu_seqlens,
        grad_output,
        query_symbols,
        key_symbols,
        _dropout_seed(dropout_p, seed=69),
        5,
        1.3,
        dropout_p,
        7.0,
        7,
    )
    masked = torch.ops.rosa_soft.surrogate_vjp_varlen_masked(
        query,
        key,
        value,
        cu_seqlens,
        grad_output,
        query_symbols,
        key_symbols,
        _dropout_seed(dropout_p, seed=69),
        5,
        1.3,
        dropout_p,
        7.0,
        gradient_mask,
    )

    for bit, full_gradient, masked_gradient in zip(
        (1, 2, 4),
        full,
        masked,
    ):
        if gradient_mask & bit:
            torch.testing.assert_close(masked_gradient, full_gradient)
        else:
            assert masked_gradient.numel() == 0


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
@pytest.mark.parametrize("required_input", range(3))
def test_varlen_cuda_computes_each_required_gradient(required_input):
    tensors = [
        _nonzero_randn((9, 1, 3), seed=70, device="cuda"),
        _nonzero_randn((9, 1, 3), seed=71, device="cuda"),
        _nonzero_randn((9, 1, 2), seed=72, device="cuda"),
    ]
    tensors[required_input].requires_grad_()
    cu_seqlens = _offsets([0, 3, 9], device="cuda")

    output = rosa_soft.rosa_soft_varlen(
        *tensors,
        cu_seqlens,
        max_suffix_length=5,
    )
    output.sum().backward()

    for index, tensor in enumerate(tensors):
        if index == required_input:
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()
        else:
            assert tensor.grad is None


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)
def test_varlen_cuda_torch_compile_fullgraph_forward_and_backward():
    def operator(query, key, value, cu_seqlens):
        return rosa_soft.rosa_soft_varlen(
            query,
            key,
            value,
            cu_seqlens,
            max_suffix_length=5,
            scale=1.5,
            dropout_p=0.2,
            mismatch_scale=7.0,
        )

    compiled = torch.compile(
        operator,
        backend="aot_eager",
        fullgraph=True,
    )
    inputs = (
        _nonzero_randn(
            (9, 1, 3),
            seed=80,
            device="cuda",
        ).requires_grad_(),
        _nonzero_randn(
            (9, 1, 3),
            seed=81,
            device="cuda",
        ).requires_grad_(),
        _nonzero_randn(
            (9, 1, 2),
            seed=82,
            device="cuda",
        ).requires_grad_(),
    )
    cu_seqlens = _offsets([0, 4, 9], device="cuda")

    output = compiled(*inputs, cu_seqlens)
    gradients = torch.autograd.grad(output.sum(), inputs)

    assert output.shape == (9, 1, 2)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
