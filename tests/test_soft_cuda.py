import inspect

import pytest
import numpy as np
import torch

import rosa_soft
from rosa_soft import rosa_soft_reference
from rosa_soft.testing import (
    rosa_soft_reference_with_noise,
    rosa_soft_with_seed,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not rosa_soft.HAS_ROSA_SOFT_CUDA,
    reason="RosaSoft CUDA extension is unavailable",
)


def _nonzero_randn(shape, *, dtype=torch.float32, seed=0, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    values = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    signs = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
    return signs * (values.abs() + 0.2)


def _counter_uniform(query, seed):
    batch, seq_len, heads, bits = query.shape
    key_positions = max(seq_len - 1, 0)
    prefix = np.arange(
        batch * heads * seq_len,
        dtype=np.uint64,
    ).reshape(batch, heads, seq_len, 1, 1)
    key = np.arange(
        key_positions,
        dtype=np.uint64,
    ).reshape(1, 1, 1, key_positions, 1)
    bit = np.arange(bits, dtype=np.uint64).reshape(1, 1, 1, 1, bits)
    counter = (prefix * np.uint64(seq_len) + key) * np.uint64(bits) + bit
    random_bits = (
        np.uint64(seed)
        + (counter + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    )
    random_bits = (
        (random_bits ^ (random_bits >> np.uint64(30)))
        * np.uint64(0xBF58476D1CE4E5B9)
    )
    random_bits = (
        (random_bits ^ (random_bits >> np.uint64(27)))
        * np.uint64(0x94D049BB133111EB)
    )
    random_bits ^= random_bits >> np.uint64(31)
    mantissa = random_bits >> np.uint64(40)
    samples = (
        mantissa.astype(np.float32) + np.float32(0.5)
    ) * np.float32(1.0 / 16777216.0)
    return torch.from_numpy(samples).to(query.device)


def _seed_tensor(seed):
    return torch.tensor(seed, dtype=torch.int64, device="cuda")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("shape", "value_heads", "max_suffix_length"),
    [
        ((2, 9, 4, 5), 2, 1),
        ((1, 17, 2, 32), 1, 7),
        ((1, 137, 1, 3), 1, 200),
    ],
)
def test_cuda_hard_forward_matches_reference(dtype, shape, value_heads, max_suffix_length):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    batch, seq_len, heads, _ = shape
    query = _nonzero_randn(shape, dtype=dtype, seed=1)
    key = _nonzero_randn(shape, dtype=dtype, seed=2)
    value = _nonzero_randn(
        (batch, seq_len, value_heads, 3),
        dtype=dtype,
        seed=3,
    )

    expected = rosa_soft_reference(query, key, value, max_suffix_length=max_suffix_length)
    actual = rosa_soft.rosa_soft(query, key, value, max_suffix_length=max_suffix_length)

    assert torch.equal(actual, expected)
    assert set(actual.float().unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_cuda_counter_rng_matches_reference_forward_and_vjp():

    shape = (1, 7, 2, 4)
    query = _nonzero_randn(shape, seed=10)
    key = _nonzero_randn(shape, seed=11)
    value = _nonzero_randn((1, 7, 1, 3), seed=12)
    seed = 987654321
    uniform = _counter_uniform(query, seed)
    grad_output = torch.randn(
        1,
        7,
        2,
        3,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(13),
    )

    reference_inputs = tuple(x.detach().clone().requires_grad_() for x in (query, key, value))
    cuda_inputs = tuple(x.detach().clone().requires_grad_() for x in (query, key, value))
    reference = rosa_soft_reference_with_noise(
        *reference_inputs,
        uniform,
        5,
        1.7,
        3.0,
    )
    actual = rosa_soft_with_seed(
        *cuda_inputs,
        5,
        1.7,
        3.0,
        _seed_tensor(seed),
    )

    assert torch.equal(actual, reference)
    reference_grads = torch.autograd.grad(reference, reference_inputs, grad_output)
    cuda_grads = torch.autograd.grad(actual, cuda_inputs, grad_output)
    for actual_grad, reference_grad in zip(cuda_grads, reference_grads):
        torch.testing.assert_close(
            actual_grad,
            reference_grad,
            rtol=2e-5,
            atol=2e-6,
        )


def test_cuda_nonfinite_logits_match_exact_reference_vjp():
    query = torch.tensor(
        [[
            [[float("inf"), -0.5]],
            [[0.4, -float("inf")]],
            [[float("nan"), -1.0]],
        ]],
        device="cuda",
    )
    key = torch.tensor(
        [[
            [[-1.0, float("inf")]],
            [[-float("inf"), -0.7]],
            [[0.2, 0.3]],
        ]],
        device="cuda",
    )
    value = torch.tensor(
        [[
            [[float("inf"), -1.0]],
            [[-float("inf"), 0.2]],
            [[0.5, -0.4]],
        ]],
        device="cuda",
    )
    seed = 1234
    uniform = _counter_uniform(query, seed)
    grad_output = _nonzero_randn((1, 3, 1, 2), seed=1235)
    reference_inputs = tuple(
        tensor.clone().requires_grad_()
        for tensor in (query, key, value)
    )
    cuda_inputs = tuple(
        tensor.clone().requires_grad_()
        for tensor in (query, key, value)
    )

    reference = rosa_soft_reference_with_noise(
        *reference_inputs,
        uniform,
        3,
        1.0,
        3.0,
    )
    actual = rosa_soft_with_seed(
        *cuda_inputs,
        3,
        1.0,
        3.0,
        _seed_tensor(seed),
    )
    reference_grads = torch.autograd.grad(
        reference,
        reference_inputs,
        grad_output,
    )
    cuda_grads = torch.autograd.grad(
        actual,
        cuda_inputs,
        grad_output,
    )

    assert torch.equal(actual, reference)
    for actual_grad, reference_grad in zip(cuda_grads, reference_grads):
        assert torch.equal(
            torch.isnan(actual_grad),
            torch.isnan(reference_grad),
        )
        finite = torch.isfinite(actual_grad) & torch.isfinite(reference_grad)
        torch.testing.assert_close(
            actual_grad[finite],
            reference_grad[finite],
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.parametrize(
    ("seq_len", "bits"),
    [
        (257, 8),  # Multi-tile ScoreCached.
        (640, 1),  # Multi-tile KeyReduced.
        (1152, 1),
    ],
)
def test_cuda_vjp_parity_across_optimized_backward_plans(seq_len, bits):

    query = _nonzero_randn((1, seq_len, 1, bits), seed=seq_len)
    key = _nonzero_randn((1, seq_len, 1, bits), seed=seq_len + 1)
    value = _nonzero_randn((1, seq_len, 1, 1), seed=seq_len + 2)
    grad_output = _nonzero_randn(
        (1, seq_len, 1, 1),
        seed=seq_len + 3,
    )
    seed = 9000 + seq_len
    uniform = _counter_uniform(query, seed)
    reference_inputs = tuple(
        x.detach().clone().requires_grad_() for x in (query, key, value)
    )
    cuda_inputs = tuple(
        x.detach().clone().requires_grad_() for x in (query, key, value)
    )

    reference = rosa_soft_reference_with_noise(
        *reference_inputs,
        uniform,
        8,
        2.0,
        3.0,
    )
    actual = rosa_soft_with_seed(
        *cuda_inputs,
        8,
        2.0,
        3.0,
        _seed_tensor(seed),
    )

    assert torch.equal(actual, reference)
    reference_grads = torch.autograd.grad(
        reference,
        reference_inputs,
        grad_output,
    )
    cuda_grads = torch.autograd.grad(
        actual,
        cuda_inputs,
        grad_output,
    )
    for actual_grad, reference_grad in zip(cuda_grads, reference_grads):
        torch.testing.assert_close(
            actual_grad,
            reference_grad,
            rtol=5e-5,
            atol=1e-5,
        )


def test_cuda_generic_backward_plan_matches_reference():

    query = _nonzero_randn((1, 4, 1, 2), seed=120)
    key = _nonzero_randn((1, 4, 1, 2), seed=121)
    value = _nonzero_randn((1, 4, 1, 12000), seed=122)
    grad_output = _nonzero_randn((1, 4, 1, 12000), seed=123)
    seed = 654321
    uniform = _counter_uniform(query, seed)
    reference_inputs = tuple(
        x.detach().clone().requires_grad_() for x in (query, key, value)
    )
    cuda_inputs = tuple(
        x.detach().clone().requires_grad_() for x in (query, key, value)
    )

    reference = rosa_soft_reference_with_noise(
        *reference_inputs,
        uniform,
        2,
        2.0,
        3.0,
    )
    actual = rosa_soft_with_seed(
        *cuda_inputs,
        2,
        2.0,
        3.0,
        _seed_tensor(seed),
    )
    reference_grads = torch.autograd.grad(
        reference,
        reference_inputs,
        grad_output,
    )
    cuda_grads = torch.autograd.grad(
        actual,
        cuda_inputs,
        grad_output,
    )

    assert torch.equal(actual, reference)
    for actual_grad, reference_grad in zip(cuda_grads, reference_grads):
        torch.testing.assert_close(
            actual_grad,
            reference_grad,
            rtol=5e-5,
            atol=5e-5,
        )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float16, 3e-3, 3e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_cuda_low_precision_vjp_matches_float32_oracle(dtype, rtol, atol):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")

    query = _nonzero_randn((1, 7, 2, 5), dtype=dtype, seed=14)
    key = _nonzero_randn((1, 7, 2, 5), dtype=dtype, seed=15)
    value = _nonzero_randn((1, 7, 1, 3), dtype=dtype, seed=16)
    grad_output = _nonzero_randn((1, 7, 2, 3), dtype=dtype, seed=17)
    seed = 998877
    uniform = _counter_uniform(query, seed)
    reference_inputs = tuple(x.detach().clone().requires_grad_() for x in (query, key, value))
    cuda_inputs = tuple(x.detach().clone().requires_grad_() for x in (query, key, value))

    reference = rosa_soft_reference_with_noise(
        *reference_inputs,
        uniform,
        5,
        1.4,
        2.6,
    )
    actual = rosa_soft_with_seed(
        *cuda_inputs,
        5,
        1.4,
        2.6,
        _seed_tensor(seed),
    )

    assert torch.equal(actual, reference)
    reference_grads = torch.autograd.grad(reference, reference_inputs, grad_output)
    cuda_grads = torch.autograd.grad(actual, cuda_inputs, grad_output)
    for actual_grad, reference_grad in zip(cuda_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("heads", "value_heads", "bits", "value_dim"),
    [(1, 1, 1, 1), (4, 2, 7, 5), (6, 3, 32, 3)],
)
def test_cuda_counter_vjp_parity_for_grouped_heads(
    heads,
    value_heads,
    bits,
    value_dim,
):

    query = _nonzero_randn((1, 6, heads, bits), seed=20 + heads)
    key = _nonzero_randn((1, 6, heads, bits), seed=30 + heads)
    value = _nonzero_randn((1, 6, value_heads, value_dim), seed=40 + heads)
    seed = 1234567 + bits
    uniform = _counter_uniform(query, seed)
    grad_output = _nonzero_randn(
        (1, 6, heads, value_dim),
        seed=50 + heads,
    )

    reference_inputs = tuple(x.detach().clone().requires_grad_() for x in (query, key, value))
    cuda_inputs = tuple(x.detach().clone().requires_grad_() for x in (query, key, value))
    reference = rosa_soft_reference_with_noise(
        *reference_inputs,
        uniform,
        4,
        2.0,
        3.0,
    )
    actual = rosa_soft_with_seed(
        *cuda_inputs,
        4,
        2.0,
        3.0,
        _seed_tensor(seed),
    )

    assert torch.equal(actual, reference)
    reference_grads = torch.autograd.grad(reference, reference_inputs, grad_output)
    cuda_grads = torch.autograd.grad(actual, cuda_inputs, grad_output)
    for actual_grad, reference_grad in zip(cuda_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=3e-5, atol=3e-6)


def test_soft_parameters_and_seed_change_only_backward():

    query = _nonzero_randn((1, 10, 2, 5), seed=60)
    key = _nonzero_randn((1, 10, 2, 5), seed=61)
    value = _nonzero_randn((1, 10, 1, 3), seed=62)
    grad_output = _nonzero_randn((1, 10, 2, 3), seed=63)

    def run(seed, route_temperature, mismatch_penalty):
        inputs = tuple(x.detach().clone().requires_grad_() for x in (query, key, value))
        output = rosa_soft_with_seed(
            *inputs,
            6,
            route_temperature,
            mismatch_penalty,
            _seed_tensor(seed),
        )
        gradients = torch.autograd.grad(output, inputs, grad_output)
        return output.detach(), gradients

    baseline_output, baseline_grads = run(700, 2.0, 3.0)
    cold_output, cold_grads = run(700, 0.7, 3.0)
    high_lambda_output, high_lambda_grads = run(700, 2.0, 7.0)
    other_seed_output, other_seed_grads = run(701, 2.0, 3.0)

    for output in (cold_output, high_lambda_output, other_seed_output):
        assert torch.equal(output, baseline_output)
    for gradients in (cold_grads, high_lambda_grads, other_seed_grads):
        assert any(
            not torch.allclose(actual, baseline)
            for actual, baseline in zip(gradients, baseline_grads)
        )


def test_dense_backward_gives_value_credit_to_every_causal_action():

    seq_len = 64
    query = _nonzero_randn((1, seq_len, 1, 4), seed=64).requires_grad_()
    key = _nonzero_randn((1, seq_len, 1, 4), seed=65).requires_grad_()
    value = _nonzero_randn((1, seq_len, 1, 1), seed=66).requires_grad_()
    output = rosa_soft_with_seed(
        query,
        key,
        value,
        8,
        2.0,
        3.0,
        _seed_tensor(123456),
    )
    grad_output = torch.zeros_like(output)
    grad_output[:, -1] = 1

    grad_value = torch.autograd.grad(output, value, grad_output)[0]

    assert grad_value[:, 0].count_nonzero() == 0
    assert torch.all(grad_value[:, 1:] > 0)


@pytest.mark.parametrize("seed", range(8))
def test_dense_backward_discovers_a_low_rank_useful_route(seed):
    seq_len = 33
    torch.cuda.manual_seed(2000 + seed)
    route_logit = torch.nn.Parameter(torch.tensor(-0.5, device="cuda"))
    optimizer = torch.optim.SGD([route_logit], lr=1.0)

    for step in range(12):
        optimizer.zero_grad(set_to_none=True)
        query = torch.ones(1, seq_len, 1, 1, device="cuda")
        query[0, -1, 0, 0] = route_logit
        key = -torch.ones_like(query)
        key[:, 0] = 1
        value = -torch.ones_like(query)
        value[:, 1] = 1
        output = rosa_soft.rosa_soft(
            query,
            key,
            value,
            max_suffix_length=1,
            route_temperature=2.0,
            mismatch_penalty=3.0,
        )
        loss = (output[0, -1, 0, 0] - 1).square()
        loss.backward()
        optimizer.step()
        if route_logit.item() > 0:
            break

    assert step + 1 <= 6
    assert route_logit.item() > 0
    with torch.no_grad():
        query = torch.ones(1, seq_len, 1, 1, device="cuda")
        query[0, -1, 0, 0] = route_logit
        final_output = rosa_soft.rosa_soft(
            query,
            key,
            value,
            max_suffix_length=1,
        )
    assert final_output[0, -1, 0, 0] == 1


def test_public_rng_is_numerically_reproducible_and_inference_consumes_no_rng():
    query = _nonzero_randn((1, 8, 2, 4), seed=70)
    key = _nonzero_randn((1, 8, 2, 4), seed=71)
    value = _nonzero_randn((1, 8, 1, 3), seed=72)
    grad_output = _nonzero_randn((1, 8, 2, 3), seed=73)

    def run():
        inputs = tuple(x.detach().clone().requires_grad_() for x in (query, key, value))
        torch.cuda.manual_seed(900)
        output = rosa_soft.rosa_soft(*inputs, max_suffix_length=5)
        return output.detach(), torch.autograd.grad(output, inputs, grad_output)

    output_a, gradients_a = run()
    output_b, gradients_b = run()
    assert torch.equal(output_a, output_b)
    for grad_a, grad_b in zip(gradients_a, gradients_b):
        torch.testing.assert_close(grad_a, grad_b, rtol=2e-6, atol=2e-7)

    rng_before = torch.cuda.get_rng_state()
    with torch.no_grad():
        rosa_soft.rosa_soft(query, key, value, max_suffix_length=5)
    assert torch.equal(rng_before, torch.cuda.get_rng_state())


def test_cuda_accepts_noncontiguous_inputs_and_returns_finite_gradients():
    query = _nonzero_randn((2, 3, 9, 4), seed=80).transpose(1, 2).requires_grad_()
    key = _nonzero_randn((2, 3, 9, 4), seed=81).transpose(1, 2).requires_grad_()
    value = _nonzero_randn((2, 1, 9, 5), seed=82).transpose(1, 2).requires_grad_()

    output = rosa_soft.rosa_soft(query, key, value, max_suffix_length=7)
    output.backward(torch.randn_like(output))

    assert output.shape == (2, 9, 3, 5)
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_cuda_singleton_sequence_is_exact_null_with_finite_backward():
    query = torch.ones(2, 1, 2, 3, device="cuda", requires_grad=True)
    key = torch.ones_like(query, requires_grad=True)
    value = torch.ones(2, 1, 1, 4, device="cuda", requires_grad=True)

    output = rosa_soft.rosa_soft(query, key, value)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_cuda_normalizes_large_window_and_rejects_nonintegral_window():
    query = _nonzero_randn((1, 7, 1, 3), seed=90)
    key = _nonzero_randn((1, 7, 1, 3), seed=91)
    value = _nonzero_randn((1, 7, 1, 2), seed=92)

    full = rosa_soft.rosa_soft(query, key, value, max_suffix_length=7)
    large = rosa_soft.rosa_soft(query, key, value, max_suffix_length=10**9)
    assert torch.equal(full, large)
    raw_full = torch.ops.rosa_soft.soft_forward(query, key, value, 7)
    raw_huge = torch.ops.rosa_soft.soft_forward(
        query,
        key,
        value,
        10**18,
    )
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(raw_full, raw_huge)
    )
    for invalid in (True, 1.5):
        with pytest.raises(TypeError, match="integer"):
            rosa_soft.rosa_soft(query, key, value, max_suffix_length=invalid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_suffix_length": 0}, "max_suffix_length"),
        ({"route_temperature": 0.0}, "route_temperature"),
        ({"route_temperature": 1e-300}, "route_temperature"),
        ({"route_temperature": 1e300}, "route_temperature"),
        (
            {
                "max_suffix_length": 5,
                "route_temperature": 1e-38,
            },
            "max_suffix_length",
        ),
        ({"mismatch_penalty": 0.0}, "mismatch_penalty"),
        ({"mismatch_penalty": 1e-300}, "mismatch_penalty"),
        ({"mismatch_penalty": 1e300}, "mismatch_penalty"),
    ],
)
def test_cuda_static_parameter_validation(kwargs, message):
    query = torch.ones(1, 5, 1, 1, device="cuda")
    with pytest.raises(ValueError, match=message):
        rosa_soft.rosa_soft(query, query, query, **kwargs)


def test_raw_cuda_backward_rejects_unrepresentable_soft_parameters():
    query = torch.ones(1, 5, 1, 1, device="cuda")
    output, q_bits, k_bits = torch.ops.rosa_soft.soft_forward(
        query,
        query,
        query,
        2,
    )
    seed = _seed_tensor(123)

    for route_temperature, mismatch_penalty, message in (
        (1e-300, 3.0, "inverse route_temperature"),
        (1e300, 3.0, "inverse route_temperature"),
        (1e-38, 3.0, "max_suffix_length"),
        (2.0, 1e-300, "mismatch_penalty"),
        (2.0, 1e300, "mismatch_penalty"),
    ):
        with pytest.raises(RuntimeError, match=message):
            torch.ops.rosa_soft.soft_backward(
                query,
                query,
                query,
                torch.ones_like(output),
                q_bits,
                k_bits,
                seed,
                5 if route_temperature == 1e-38 else 2,
                route_temperature,
                mismatch_penalty,
            )


def test_cuda_backward_is_explicitly_once_differentiable():
    query = _nonzero_randn((1, 6, 1, 3), seed=130).requires_grad_()
    key = _nonzero_randn((1, 6, 1, 3), seed=131).requires_grad_()
    value = _nonzero_randn((1, 6, 1, 2), seed=132).requires_grad_()

    output = rosa_soft.rosa_soft(query, key, value, max_suffix_length=4)
    gradients = torch.autograd.grad(
        output.sum(),
        (query, key, value),
        create_graph=True,
    )

    assert all(not gradient.requires_grad for gradient in gradients)
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(
            sum(gradient.sum() for gradient in gradients),
            (query, key, value),
        )


def test_cuda_torch_compile_fullgraph_forward_and_backward():

    seed = _seed_tensor(24680)

    def operator(query, key, value):
        return rosa_soft_with_seed(
            query,
            key,
            value,
            4,
            2.0,
            3.0,
            seed,
        )

    compiled = torch.compile(
        operator,
        backend="aot_eager",
        fullgraph=True,
    )
    base_inputs = (
        _nonzero_randn((1, 7, 2, 4), seed=140),
        _nonzero_randn((1, 7, 2, 4), seed=141),
        _nonzero_randn((1, 7, 1, 3), seed=142),
    )
    grad_output = _nonzero_randn((1, 7, 2, 3), seed=143)
    eager_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in base_inputs
    )
    compiled_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in base_inputs
    )

    expected = operator(*eager_inputs)
    actual = compiled(*compiled_inputs)
    expected_grads = torch.autograd.grad(
        expected,
        eager_inputs,
        grad_output,
    )
    actual_grads = torch.autograd.grad(
        actual,
        compiled_inputs,
        grad_output,
    )

    assert torch.equal(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_cuda_uses_the_input_device_and_restores_current_device():
    current_device = torch.cuda.current_device()
    input_device = 1 if current_device == 0 else 0
    device = f"cuda:{input_device}"
    query = _nonzero_randn(
        (1, 9, 2, 4),
        seed=150,
        device=device,
    ).requires_grad_()
    key = _nonzero_randn(
        (1, 9, 2, 4),
        seed=151,
        device=device,
    ).requires_grad_()
    value = _nonzero_randn(
        (1, 9, 1, 3),
        seed=152,
        device=device,
    ).requires_grad_()

    output = rosa_soft.rosa_soft(
        query,
        key,
        value,
        max_suffix_length=5,
    )
    output.sum().backward()
    torch.cuda.synchronize(input_device)

    assert output.device.index == input_device
    assert torch.cuda.current_device() == current_device
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (query, key, value)
    )
    with pytest.raises(RuntimeError, match="same CUDA device"):
        torch.ops.rosa_soft.soft_forward(
            query,
            key.to(f"cuda:{current_device}"),
            value,
            5,
        )


@pytest.mark.parametrize(
    "removed",
    [
        "mismatch_relaxation",
        "value_backward",
        "return_state",
        "return_telemetry",
        "logit_epsilon",
        "qk_damper_strength",
    ],
)
def test_cuda_removed_controls_are_not_public(removed):
    query = torch.ones(1, 2, 1, 1, device="cuda")
    with pytest.raises(TypeError):
        rosa_soft.rosa_soft(query, query, query, **{removed: 1})


def test_cuda_public_and_dispatch_signatures_are_minimal():
    expected_parameters = (
        "query_logits",
        "key_logits",
        "payload_logits",
        "max_suffix_length",
        "route_temperature",
        "mismatch_penalty",
    )
    assert tuple(
        inspect.signature(rosa_soft.rosa_soft).parameters
    ) == expected_parameters
    assert tuple(
        inspect.signature(rosa_soft.rosa_soft_reference).parameters
    ) == expected_parameters
    assert str(
        torch.ops.rosa_soft.soft_forward.default._schema
    ) == (
        "rosa_soft::soft_forward(Tensor query_logits, Tensor key_logits, "
        "Tensor payload_logits, int max_suffix_length) -> Tensor[]"
    )
    assert str(
        torch.ops.rosa_soft.soft_backward.default._schema
    ) == (
        "rosa_soft::soft_backward(Tensor query_logits, Tensor key_logits, "
        "Tensor payload_logits, Tensor grad_output, Tensor query_symbols, "
        "Tensor key_symbols, "
        "Tensor rng_seed, int max_suffix_length, float route_temperature, "
        "float mismatch_penalty) -> Tensor[]"
    )


def test_cuda_rejects_cpu_and_invalid_head_layout():
    cpu = torch.ones(1, 2, 1, 1)
    with pytest.raises(ValueError, match="CUDA"):
        rosa_soft.rosa_soft(cpu, cpu, cpu)

    query = torch.ones(1, 2, 3, 1, device="cuda")
    value = torch.ones(1, 2, 2, 1, device="cuda")
    with pytest.raises(ValueError, match="divisible"):
        rosa_soft.rosa_soft(query, query, value)

    query64 = query[:, :, :1].double()
    with pytest.raises(ValueError, match="float32"):
        rosa_soft.rosa_soft(query64, query64, query64)
