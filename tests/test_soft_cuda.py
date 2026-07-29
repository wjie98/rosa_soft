import inspect
import math

import pytest
import torch

import rosa_soft
from rosa_soft import rosa_soft_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)


def _nonzero_randn(shape, *, dtype=torch.float32, seed=0, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    values = torch.randn(
        shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return values.sign().masked_fill(values == 0, 1) * (values.abs() + 0.2)


def _run_vjp(operator, values, grad_output, **controls):
    inputs = tuple(value.detach().clone().requires_grad_() for value in values)
    output = operator(*inputs, **controls)
    return output, torch.autograd.grad(output, inputs, grad_output)


def _dropout_seed(dropout_p=0.0, seed=0):
    if dropout_p == 0.0:
        return torch.empty(0, dtype=torch.int64, device="cuda")
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
def test_cuda_hard_forward_matches_reference(
    dtype,
    shape,
    value_heads,
    max_suffix_length,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    batch, seq_len, heads, _ = shape
    values = (
        _nonzero_randn(shape, dtype=dtype, seed=1),
        _nonzero_randn(shape, dtype=dtype, seed=2),
        _nonzero_randn(
            (batch, seq_len, value_heads, 3),
            dtype=dtype,
            seed=3,
        ),
    )
    expected = rosa_soft_reference(
        *values,
        max_suffix_length=max_suffix_length,
    )
    actual = rosa_soft.rosa_soft(
        *values,
        max_suffix_length=max_suffix_length,
    )
    assert torch.equal(actual, expected)
    assert set(actual.float().unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_raw_and_fake_packed_symbol_layout_contracts():
    query = _nonzero_randn((2, 5, 3, 4), seed=4)
    key = _nonzero_randn((2, 5, 3, 4), seed=5)
    value = _nonzero_randn((2, 5, 1, 2), seed=6)
    dense = torch.ops.rosa_soft.hard_forward(query, key, value, 4)
    offsets = torch.tensor([0, 5, 10], dtype=torch.int32, device="cuda")
    varlen = torch.ops.rosa_soft.hard_forward_varlen(
        query.flatten(0, 1),
        key.flatten(0, 1),
        value.flatten(0, 1),
        offsets,
        4,
    )
    assert dense[0].shape == (2, 5, 3, 2)
    assert dense[1].shape == dense[2].shape == (2, 3, 5)
    assert dense[1].stride() == dense[2].stride() == (15, 5, 1)
    assert varlen[0].shape == (10, 3, 2)
    assert varlen[1].shape == varlen[2].shape == (10, 3)
    assert varlen[1].stride() == varlen[2].stride() == (3, 1)

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    with fake_mode:
        fake_query = fake_mode.from_tensor(query)
        fake_key = fake_mode.from_tensor(key)
        fake_value = fake_mode.from_tensor(value)
        fake = torch.ops.rosa_soft.hard_forward(
            fake_query,
            fake_key,
            fake_value,
            4,
        )
    assert fake[1].shape == fake[2].shape == (2, 3, 5)


def test_raw_cuda_vjp_rejects_token_major_packed_symbols():
    query = _nonzero_randn((2, 5, 3, 4), seed=7)
    key = _nonzero_randn((2, 5, 3, 4), seed=8)
    value = _nonzero_randn((2, 5, 1, 2), seed=9)
    output, query_symbols, key_symbols = torch.ops.rosa_soft.hard_forward(
        query,
        key,
        value,
        4,
    )
    with pytest.raises(RuntimeError, match="packed_query_symbols head mismatch"):
        torch.ops.rosa_soft.surrogate_vjp_masked(
            query,
            key,
            value,
            torch.ones_like(output),
            query_symbols.transpose(1, 2).contiguous(),
            key_symbols,
            _dropout_seed(),
            4,
            2.0,
            0.0,
            3.0,
            7,
        )


@pytest.mark.parametrize(
    (
        "shape",
        "value_heads",
        "value_dim",
        "window",
        "scale",
        "mismatch_scale",
        "rtol",
        "atol",
    ),
    [
        ((1, 7, 2, 4), 1, 3, 5, 1.7, 3.0, 2e-5, 2e-6),
        ((1, 257, 1, 8), 1, 2, 8, 2.0, 3.0, 6e-5, 1e-5),
        ((1, 160, 1, 1), 1, 2, 160, 1.0, 3.0, 8e-4, 2e-4),
        ((1, 6, 6, 32), 3, 3, 4, 2.0, 3.0, 4e-5, 4e-6),
        ((1, 4, 1, 2), 1, 12000, 2, 2.0, 3.0, 6e-5, 6e-5),
    ],
)
def test_cuda_vjp_matches_minimal_reference(
    shape,
    value_heads,
    value_dim,
    window,
    scale,
    mismatch_scale,
    rtol,
    atol,
):
    batch, seq_len, heads, _ = shape
    values = (
        _nonzero_randn(shape, seed=10 + seq_len),
        _nonzero_randn(shape, seed=11 + seq_len),
        _nonzero_randn(
            (batch, seq_len, value_heads, value_dim),
            seed=12 + seq_len,
        ),
    )
    grad_output = _nonzero_randn(
        (batch, seq_len, heads, value_dim),
        seed=13 + seq_len,
    )
    controls = {
        "max_suffix_length": window,
        "scale": scale,
        "mismatch_scale": mismatch_scale,
    }
    expected, expected_gradients = _run_vjp(
        rosa_soft_reference,
        values,
        grad_output,
        **controls,
    )
    actual, actual_gradients = _run_vjp(
        rosa_soft.rosa_soft,
        values,
        grad_output,
        **controls,
    )
    assert torch.equal(actual, expected)
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            observed,
            reference,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("dropout_p", [0.1, 0.5])
def test_cuda_attention_dropout_vjp_matches_reference(dropout_p):
    values = (
        _nonzero_randn((1, 9, 2, 4), seed=21),
        _nonzero_randn((1, 9, 2, 4), seed=22),
        _nonzero_randn((1, 9, 1, 3), seed=23),
    )
    grad_output = _nonzero_randn((1, 9, 2, 3), seed=24)
    controls = {
        "max_suffix_length": 6,
        "scale": 1.5,
        "dropout_p": dropout_p,
        "mismatch_scale": 4.0,
    }
    torch.cuda.manual_seed(25)
    expected, expected_gradients = _run_vjp(
        rosa_soft_reference,
        values,
        grad_output,
        **controls,
    )
    torch.cuda.manual_seed(25)
    actual, actual_gradients = _run_vjp(
        rosa_soft.rosa_soft,
        values,
        grad_output,
        **controls,
    )

    assert torch.equal(actual, expected)
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            observed,
            reference,
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float16, 3e-3, 3e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_cuda_low_precision_vjp_matches_reference(dtype, rtol, atol):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    values = (
        _nonzero_randn((1, 7, 2, 5), dtype=dtype, seed=30),
        _nonzero_randn((1, 7, 2, 5), dtype=dtype, seed=31),
        _nonzero_randn((1, 7, 1, 3), dtype=dtype, seed=32),
    )
    grad_output = _nonzero_randn(
        (1, 7, 2, 3),
        dtype=dtype,
        seed=33,
    )
    controls = {
        "max_suffix_length": 5,
        "scale": 1.4,
        "mismatch_scale": 2.6,
    }
    expected, expected_gradients = _run_vjp(
        rosa_soft_reference,
        values,
        grad_output,
        **controls,
    )
    actual, actual_gradients = _run_vjp(
        rosa_soft.rosa_soft,
        values,
        grad_output,
        **controls,
    )
    assert torch.equal(actual, expected)
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            observed,
            reference,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
@pytest.mark.parametrize("gradient_mask", range(1, 8))
def test_cuda_masked_vjp_matches_full_vjp(gradient_mask, dropout_p):
    query = _nonzero_randn((1, 9, 2, 4), seed=40)
    key = _nonzero_randn((1, 9, 2, 4), seed=41)
    value = _nonzero_randn((1, 9, 1, 3), seed=42)
    output, query_symbols, key_symbols = torch.ops.rosa_soft.hard_forward(
        query,
        key,
        value,
        6,
    )
    grad_output = _nonzero_randn(output.shape, seed=43)
    arguments = (
        query,
        key,
        value,
        grad_output,
        query_symbols,
        key_symbols,
        _dropout_seed(dropout_p, seed=44),
        6,
        1.3,
        dropout_p,
        7.0,
    )
    full = torch.ops.rosa_soft.surrogate_vjp_masked(*arguments, 7)
    masked = torch.ops.rosa_soft.surrogate_vjp_masked(
        *arguments,
        gradient_mask,
    )
    for bit, expected, observed in zip((1, 2, 4), full, masked):
        if gradient_mask & bit:
            torch.testing.assert_close(observed, expected)
        else:
            assert observed.numel() == 0


@pytest.mark.parametrize("required_input", range(3))
def test_cuda_autograd_computes_only_required_gradients(required_input):
    tensors = [
        _nonzero_randn((1, 7, 1, 3), seed=50),
        _nonzero_randn((1, 7, 1, 3), seed=51),
        _nonzero_randn((1, 7, 1, 2), seed=52),
    ]
    tensors[required_input].requires_grad_()
    rosa_soft.rosa_soft(
        *tensors,
        max_suffix_length=5,
        scale=1.5,
        mismatch_scale=6.0,
    ).sum().backward()
    for index, tensor in enumerate(tensors):
        if index == required_input:
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()
        else:
            assert tensor.grad is None


def test_cuda_nonfinite_policy_matches_reference():
    query = torch.tensor(
        [[[[math.inf, -0.5]], [[0.4, -math.inf]], [[math.nan, -1.0]]]],
        device="cuda",
    )
    key = torch.tensor(
        [[[[-1.0, math.inf]], [[-math.inf, -0.7]], [[0.2, 0.3]]]],
        device="cuda",
    )
    value = torch.tensor(
        [[[[math.inf, -1.0]], [[-math.inf, 0.2]], [[0.5, -0.4]]]],
        device="cuda",
    )
    values = (query, key, value)
    grad_output = _nonzero_randn((1, 3, 1, 2), seed=60)
    controls = {
        "max_suffix_length": 3,
        "scale": 1.0,
        "mismatch_scale": 3.0,
    }
    expected, expected_gradients = _run_vjp(
        rosa_soft_reference,
        values,
        grad_output,
        **controls,
    )
    actual, actual_gradients = _run_vjp(
        rosa_soft.rosa_soft,
        values,
        grad_output,
        **controls,
    )
    assert torch.equal(actual, expected)
    for observed, reference in zip(actual_gradients, expected_gradients):
        assert torch.equal(torch.isnan(observed), torch.isnan(reference))
        finite = torch.isfinite(observed) & torch.isfinite(reference)
        torch.testing.assert_close(
            observed[finite],
            reference[finite],
            rtol=2e-5,
            atol=2e-6,
        )


def test_soft_controls_change_only_backward():
    values = (
        _nonzero_randn((1, 10, 2, 5), seed=70),
        _nonzero_randn((1, 10, 2, 5), seed=71),
        _nonzero_randn((1, 10, 1, 3), seed=72),
    )
    grad_output = _nonzero_randn((1, 10, 2, 3), seed=73)

    def run(scale, mismatch_scale):
        return _run_vjp(
            rosa_soft.rosa_soft,
            values,
            grad_output,
            max_suffix_length=6,
            scale=scale,
            mismatch_scale=mismatch_scale,
        )

    baseline_output, baseline_gradients = run(2.0, 3.0)
    for output, gradients in (run(0.7, 3.0), run(2.0, 7.0)):
        assert torch.equal(output, baseline_output)
        assert any(
            not torch.allclose(observed, baseline)
            for observed, baseline in zip(gradients, baseline_gradients)
        )


def test_dense_backward_credits_every_causal_value():
    seq_len = 64
    query = _nonzero_randn((1, seq_len, 1, 4), seed=80)
    key = _nonzero_randn((1, seq_len, 1, 4), seed=81)
    value = _nonzero_randn(
        (1, seq_len, 1, 1),
        seed=82,
    ).requires_grad_()
    output = rosa_soft.rosa_soft(
        query,
        key,
        value,
        max_suffix_length=8,
        scale=2.0,
        mismatch_scale=3.0,
    )
    grad_output = torch.zeros_like(output)
    grad_output[:, -1] = 1
    gradient = torch.autograd.grad(output, value, grad_output)[0]
    assert gradient[:, 0].count_nonzero() == 0
    assert torch.all(gradient[:, 1:] > 0)


@pytest.mark.parametrize("seed", range(6))
def test_dense_backward_discovers_low_rank_useful_route(seed):
    seq_len = 33
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
            scale=2.0,
            mismatch_scale=3.0,
        )
        (output[0, -1, 0, 0] - 1).square().backward()
        optimizer.step()
        if route_logit.item() > 0:
            break
    assert step + 1 <= 6
    assert route_logit.item() > 0


def test_cuda_is_repeatable_and_consumes_no_rng():
    values = (
        _nonzero_randn((1, 8, 2, 4), seed=90),
        _nonzero_randn((1, 8, 2, 4), seed=91),
        _nonzero_randn((1, 8, 1, 3), seed=92),
    )
    grad_output = _nonzero_randn((1, 8, 2, 3), seed=93)
    state = torch.cuda.get_rng_state()
    first = _run_vjp(
        rosa_soft.rosa_soft,
        values,
        grad_output,
        max_suffix_length=5,
    )
    assert torch.equal(state, torch.cuda.get_rng_state())
    torch.cuda.manual_seed(123456)
    second = _run_vjp(
        rosa_soft.rosa_soft,
        values,
        grad_output,
        max_suffix_length=5,
    )
    assert torch.equal(first[0], second[0])
    for left, right in zip(first[1], second[1]):
        torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-7)


def test_cuda_attention_dropout_is_seeded_backward_only():
    values = (
        _nonzero_randn((1, 8, 2, 4), seed=94),
        _nonzero_randn((1, 8, 2, 4), seed=95),
        _nonzero_randn((1, 8, 1, 3), seed=96),
    )
    grad_output = _nonzero_randn((1, 8, 2, 3), seed=97)

    def run(seed):
        torch.cuda.manual_seed(seed)
        return _run_vjp(
            rosa_soft.rosa_soft,
            values,
            grad_output,
            max_suffix_length=5,
            dropout_p=0.5,
        )

    first = run(98)
    replay = run(98)
    changed = run(99)
    assert torch.equal(first[0], replay[0])
    assert torch.equal(first[0], changed[0])
    for expected, actual in zip(first[1], replay[1]):
        torch.testing.assert_close(
            expected,
            actual,
            rtol=2e-6,
            atol=2e-7,
        )
    assert any(
        not torch.allclose(expected, actual)
        for expected, actual in zip(first[1], changed[1])
    )

    state = torch.cuda.get_rng_state()
    with torch.no_grad():
        rosa_soft.rosa_soft(*values, dropout_p=0.5)
    assert torch.equal(state, torch.cuda.get_rng_state())


def test_cuda_noncontiguous_inputs_and_singleton():
    query = _nonzero_randn((2, 3, 9, 4), seed=100).transpose(1, 2)
    key = _nonzero_randn((2, 3, 9, 4), seed=101).transpose(1, 2)
    value = _nonzero_randn((2, 1, 9, 5), seed=102).transpose(1, 2)
    leaves = tuple(value.requires_grad_() for value in (query, key, value))
    output = rosa_soft.rosa_soft(*leaves, max_suffix_length=7)
    output.backward(torch.randn_like(output))
    assert output.shape == (2, 9, 3, 5)
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in leaves
    )

    singleton = tuple(
        torch.ones(
            2,
            1,
            heads,
            dim,
            device="cuda",
            requires_grad=True,
        )
        for heads, dim in ((2, 3), (2, 3), (1, 4))
    )
    singleton_output = rosa_soft.rosa_soft(*singleton)
    singleton_gradients = torch.autograd.grad(
        singleton_output.sum(),
        singleton,
    )
    assert torch.equal(singleton_output, torch.zeros_like(singleton_output))
    assert all(
        torch.equal(gradient, torch.zeros_like(gradient))
        for gradient in singleton_gradients
    )


def test_cuda_clamps_large_window_and_rejects_nonintegral_window():
    values = (
        _nonzero_randn((1, 7, 1, 3), seed=110),
        _nonzero_randn((1, 7, 1, 3), seed=111),
        _nonzero_randn((1, 7, 1, 2), seed=112),
    )
    full = rosa_soft.rosa_soft(*values, max_suffix_length=7)
    large = rosa_soft.rosa_soft(*values, max_suffix_length=10**9)
    assert torch.equal(full, large)
    for invalid in (True, 1.5):
        with pytest.raises(TypeError, match="integer"):
            rosa_soft.rosa_soft(*values, max_suffix_length=invalid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_suffix_length": 0}, "max_suffix_length"),
        ({"scale": True}, "scale"),
        ({"scale": 0.0}, "scale"),
        ({"scale": 1e-300}, "scale"),
        ({"scale": 1e300}, "scale"),
        ({"dropout_p": True}, "dropout_p"),
        ({"dropout_p": -0.1}, "dropout_p"),
        ({"dropout_p": 1.0}, "dropout_p"),
        ({"dropout_p": math.nan}, "dropout_p"),
        ({"dropout_p": math.inf}, "dropout_p"),
        ({"mismatch_scale": 0.0}, "mismatch_scale"),
        ({"mismatch_scale": True}, "mismatch_scale"),
        ({"mismatch_scale": 1e-300}, "mismatch_scale"),
        ({"mismatch_scale": 1e300}, "mismatch_scale"),
    ],
)
def test_cuda_static_parameter_validation(kwargs, message):
    tensor = torch.ones(1, 5, 1, 1, device="cuda")
    with pytest.raises(ValueError, match=message):
        rosa_soft.rosa_soft(tensor, tensor, tensor, **kwargs)


def test_raw_cuda_vjp_rejects_unrepresentable_scalars():
    tensor = torch.ones(1, 5, 1, 1, device="cuda")
    output, query_symbols, key_symbols = torch.ops.rosa_soft.hard_forward(
        tensor,
        tensor,
        tensor,
        2,
    )
    for window, scale, mismatch_scale, message in (
        (2, 1e-300, 3.0, "scale"),
        (2, 1e300, 3.0, "scale"),
        (5, 1e38, 3.0, "max_suffix_length"),
        (2, 2.0, 1e-300, "mismatch_scale"),
        (2, 2.0, 1e300, "mismatch_scale"),
    ):
        with pytest.raises(RuntimeError, match=message):
            torch.ops.rosa_soft.surrogate_vjp_masked(
                tensor,
                tensor,
                tensor,
                torch.ones_like(output),
                query_symbols,
                key_symbols,
                _dropout_seed(),
                window,
                scale,
                0.0,
                mismatch_scale,
                7,
            )


@pytest.mark.parametrize(
    ("dropout_seed", "dropout_p", "message"),
    [
        (
            lambda: torch.tensor(1, dtype=torch.int64, device="cuda"),
            0.0,
            "scalar exactly",
        ),
        (
            lambda: torch.empty(0, dtype=torch.int64, device="cuda"),
            0.2,
            "scalar exactly",
        ),
        (
            lambda: torch.tensor(1, dtype=torch.int32, device="cuda"),
            0.2,
            "int64",
        ),
        (
            lambda: torch.tensor(1, dtype=torch.int64),
            0.2,
            "CUDA",
        ),
    ],
)
def test_raw_cuda_vjp_validates_dropout_seed(
    dropout_seed,
    dropout_p,
    message,
):
    tensor = torch.ones(1, 3, 1, 1, device="cuda")
    output, query_symbols, key_symbols = (
        torch.ops.rosa_soft.hard_forward(
            tensor,
            tensor,
            tensor,
            2,
        )
    )
    with pytest.raises(RuntimeError, match=message):
        torch.ops.rosa_soft.surrogate_vjp_masked(
            tensor,
            tensor,
            tensor,
            torch.ones_like(output),
            query_symbols,
            key_symbols,
            dropout_seed(),
            2,
            1.0,
            dropout_p,
            9.0,
            7,
        )


def test_cuda_backward_is_once_differentiable():
    inputs = tuple(
        _nonzero_randn(shape, seed=120 + index).requires_grad_()
        for index, shape in enumerate(
            ((1, 6, 1, 3), (1, 6, 1, 3), (1, 6, 1, 2))
        )
    )
    output = rosa_soft.rosa_soft(*inputs, max_suffix_length=4)
    gradients = torch.autograd.grad(
        output.sum(),
        inputs,
        create_graph=True,
    )
    assert all(not gradient.requires_grad for gradient in gradients)


def test_cuda_vjp_respects_deterministic_algorithms():
    inputs = tuple(
        _nonzero_randn(shape, seed=130 + index).requires_grad_()
        for index, shape in enumerate(
            ((1, 5, 1, 2), (1, 5, 1, 2), (1, 5, 1, 2))
        )
    )
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        output = rosa_soft.rosa_soft(*inputs)
        with pytest.raises(RuntimeError, match="deterministic"):
            output.sum().backward()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def test_cuda_torch_compile_fullgraph_forward_and_backward():
    def operator(query, key, value):
        return rosa_soft.rosa_soft(
            query,
            key,
            value,
            max_suffix_length=4,
            scale=2.0,
            dropout_p=0.2,
            mismatch_scale=3.0,
        )

    compiled = torch.compile(operator, backend="aot_eager", fullgraph=True)
    values = (
        _nonzero_randn((1, 7, 2, 4), seed=140),
        _nonzero_randn((1, 7, 2, 4), seed=141),
        _nonzero_randn((1, 7, 1, 3), seed=142),
    )
    grad_output = _nonzero_randn((1, 7, 2, 3), seed=143)
    torch.cuda.manual_seed(144)
    expected, expected_gradients = _run_vjp(
        operator,
        values,
        grad_output,
    )
    torch.cuda.manual_seed(144)
    actual, actual_gradients = _run_vjp(
        compiled,
        values,
        grad_output,
    )
    assert torch.equal(actual, expected)
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            observed,
            reference,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two visible CUDA devices",
)
def test_cuda_uses_input_device_and_restores_current_device():
    current = torch.cuda.current_device()
    input_device = 1 if current == 0 else 0
    device = f"cuda:{input_device}"
    inputs = tuple(
        _nonzero_randn(shape, seed=150 + index, device=device).requires_grad_()
        for index, shape in enumerate(
            ((1, 9, 2, 4), (1, 9, 2, 4), (1, 9, 1, 3))
        )
    )
    output = rosa_soft.rosa_soft(*inputs, max_suffix_length=5)
    output.sum().backward()
    torch.cuda.synchronize(input_device)
    assert output.device.index == input_device
    assert torch.cuda.current_device() == current


def test_cuda_public_and_dispatch_signatures_are_minimal():
    expected_parameters = (
        "query",
        "key",
        "value",
        "max_suffix_length",
        "scale",
        "dropout_p",
        "mismatch_scale",
    )
    assert tuple(inspect.signature(rosa_soft.rosa_soft).parameters) == (
        expected_parameters
    )
    assert tuple(
        inspect.signature(rosa_soft.rosa_soft_reference).parameters
    ) == expected_parameters
    assert str(torch.ops.rosa_soft.hard_forward.default._schema) == (
        "rosa_soft::hard_forward(Tensor query, Tensor key, "
        "Tensor value, int max_suffix_length) -> "
        "(Tensor output, Tensor packed_query_symbols, "
        "Tensor packed_key_symbols)"
    )
    assert str(torch.ops.rosa_soft.surrogate_vjp_masked.default._schema) == (
        "rosa_soft::surrogate_vjp_masked(Tensor query, "
        "Tensor key, Tensor value, Tensor grad_output, "
        "Tensor packed_query_symbols, Tensor packed_key_symbols, "
        "Tensor dropout_seed, int max_suffix_length, float scale, "
        "float dropout_p, "
        "float mismatch_scale, int gradient_mask) -> "
        "(Tensor grad_query, Tensor grad_key, Tensor grad_value)"
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
