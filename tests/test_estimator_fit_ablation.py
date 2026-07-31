import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    name = "_estimator_fit_ablation_benchmark"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "benchmarks/estimator_fit_ablation.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ablation = _load_benchmark()


def _inputs(requires_grad=True):
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(
        1,
        5,
        1,
        2,
        generator=generator,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    key = torch.randn(
        1,
        5,
        1,
        2,
        generator=generator,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    value = torch.randn(
        1,
        5,
        1,
        2,
        generator=generator,
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    return query, key, value


@pytest.mark.parametrize(
    "estimator",
    [
        ablation.rosa_soft_mismatch_random,
        ablation.rosa_soft_exact_bitflip,
        ablation.make_attention_dropout_estimator(0.25),
        ablation.make_suffix_dropout_estimator(0.9),
    ],
)
def test_research_estimators_preserve_hard_forward(estimator):
    inputs = _inputs()
    expected = ablation.rosa_soft_reference(
        *(tensor.detach() for tensor in inputs),
        max_suffix_length=3,
    )
    torch.manual_seed(3)
    actual = estimator(*inputs, max_suffix_length=3)

    assert torch.equal(actual, expected)
    gradients = torch.autograd.grad(actual.square().sum(), inputs)
    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_zero_attention_dropout_matches_deterministic_vjp():
    deterministic_inputs = _inputs()
    dropout_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in deterministic_inputs
    )
    loss_weights = torch.randn_like(deterministic_inputs[2])
    deterministic_output = ablation.rosa_soft_reference(
        *deterministic_inputs,
        max_suffix_length=4,
        mismatch_scale=3.0,
    )
    dropout_output = ablation.make_attention_dropout_estimator(0.0)(
        *dropout_inputs,
        max_suffix_length=4,
        mismatch_scale=3.0,
    )
    deterministic_gradients = torch.autograd.grad(
        (deterministic_output * loss_weights).sum(),
        deterministic_inputs,
    )
    dropout_gradients = torch.autograd.grad(
        (dropout_output * loss_weights).sum(),
        dropout_inputs,
    )

    assert torch.equal(dropout_output, deterministic_output)
    for dropout_gradient, deterministic_gradient in zip(
        dropout_gradients,
        deterministic_gradients,
    ):
        torch.testing.assert_close(
            dropout_gradient,
            deterministic_gradient,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "make_estimator",
    [
        lambda: ablation.rosa_soft_mismatch_random,
        lambda: ablation.make_attention_dropout_estimator(0.5),
        lambda: ablation.make_suffix_dropout_estimator(0.5),
    ],
)
def test_random_estimators_change_vjp_not_hard_output(make_estimator):
    gradients = []
    outputs = []
    for seed in (11, 29):
        inputs = _inputs()
        torch.manual_seed(seed)
        output = make_estimator()(
            *inputs,
            max_suffix_length=3,
            mismatch_scale=3.0,
        )
        gradients.append(
            torch.autograd.grad(output.square().sum(), inputs)
        )
        outputs.append(output)

    assert torch.equal(outputs[0], outputs[1])
    assert any(
        not torch.equal(first, second)
        for first, second in zip(gradients[0], gradients[1])
    )


def _exhaustive_bitflip_gradient(
    role,
    inputs,
    loss_weights,
    max_suffix_length,
):
    selected = inputs[role]
    base_output = ablation.rosa_soft_reference(
        *inputs,
        max_suffix_length=max_suffix_length,
    )
    expected = torch.empty_like(selected)
    for flat_index in range(selected.numel()):
        flipped_inputs = list(inputs)
        flipped = selected.clone()
        sign = 1.0 if selected.view(-1)[flat_index] > 0 else -1.0
        flipped.view(-1)[flat_index] = -sign
        flipped_inputs[role] = flipped
        flipped_output = ablation.rosa_soft_reference(
            *flipped_inputs,
            max_suffix_length=max_suffix_length,
        )
        loss_delta = (
            (flipped_output - base_output) * loss_weights
        ).sum()
        expected.view(-1)[flat_index] = -sign * loss_delta
    return expected


def test_exact_bitflip_vjp_matches_exhaustive_hard_counterfactuals():
    leaves = _inputs()
    detached = tuple(tensor.detach() for tensor in leaves)
    loss_weights = torch.tensor(
        [[[[0.2, -0.3]], [[1.0, 0.4]], [[-0.7, 0.5]],
          [[0.6, -1.2]], [[0.9, 0.1]]]],
        dtype=torch.float64,
    )
    output = ablation.rosa_soft_exact_bitflip(
        *leaves,
        max_suffix_length=3,
    )
    gradients = torch.autograd.grad(
        (output * loss_weights).sum(),
        leaves,
    )

    for role, gradient in enumerate(gradients):
        expected = _exhaustive_bitflip_gradient(
            role,
            detached,
            loss_weights,
            max_suffix_length=3,
        )
        torch.testing.assert_close(
            gradient,
            expected,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("probability", [-0.1, 1.0, float("nan")])
def test_attention_dropout_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match="attention dropout"):
        ablation.make_attention_dropout_estimator(probability)


@pytest.mark.parametrize("probability", [-0.1, 1.0, float("nan")])
def test_suffix_dropout_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match="suffix dropout"):
        ablation.make_suffix_dropout_estimator(probability)


def test_zero_suffix_dropout_matches_deterministic_vjp():
    deterministic_inputs = _inputs()
    dropout_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in deterministic_inputs
    )
    loss_weights = torch.randn_like(deterministic_inputs[2])
    deterministic_output = ablation.rosa_soft_reference(
        *deterministic_inputs,
        max_suffix_length=4,
        mismatch_scale=3.0,
    )
    dropout_output = ablation.make_suffix_dropout_estimator(0.0)(
        *dropout_inputs,
        max_suffix_length=4,
        mismatch_scale=3.0,
    )
    deterministic_gradients = torch.autograd.grad(
        (deterministic_output * loss_weights).sum(),
        deterministic_inputs,
    )
    dropout_gradients = torch.autograd.grad(
        (dropout_output * loss_weights).sum(),
        dropout_inputs,
    )

    assert torch.equal(dropout_output, deterministic_output)
    for actual, expected in zip(
        dropout_gradients,
        deterministic_gradients,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_stratified_row_weights_are_balanced_and_fixed_count():
    sample_count = 80
    row_uniforms = torch.zeros(
        sample_count,
        1,
        8,
        dtype=torch.float64,
    )
    grid = (
        torch.arange(sample_count, dtype=torch.float64) + 0.5
    ) / sample_count
    row_uniforms[:, 0, 0] = grid
    row_uniforms[:, 0, 1] = grid.roll(1)
    weights = ablation._stratified_row_weights(
        sample_count,
        1,
        8,
        0.75,
        row_uniforms,
        torch.float64,
    )

    assert torch.equal(
        (weights > 0).sum(dim=-1),
        torch.full((sample_count, 1), 2),
    )
    torch.testing.assert_close(
        weights.mean(dim=0),
        torch.ones(1, 8, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_suffix_dropout_vjp_is_unbiased_over_all_stratified_samples():
    detached = tuple(tensor.detach() for tensor in _inputs())
    loss_weights = torch.randn(
        detached[2].shape,
        generator=torch.Generator().manual_seed(71),
        dtype=torch.float64,
    )
    dense_inputs = tuple(
        tensor.clone().requires_grad_() for tensor in detached
    )
    dense_output = ablation.rosa_soft_reference(
        *dense_inputs,
        max_suffix_length=3,
        scale=1.0,
        mismatch_scale=3.0,
    )
    expected = torch.autograd.grad(
        (dense_output * loss_weights).sum(),
        dense_inputs,
    )

    sampled_gradients = []
    for first_offset in range(2):
        for second_offset in range(3):
            inputs = tuple(
                tensor.clone().requires_grad_() for tensor in detached
            )
            row_uniforms = torch.zeros(1, 1, 5, dtype=torch.float64)
            row_uniforms[0, 0, 0] = (first_offset + 0.5) / 2
            row_uniforms[0, 0, 1] = (second_offset + 0.5) / 3
            output = ablation._HardForwardSuffixDropout.apply(
                *inputs,
                row_uniforms,
                3,
                1.0,
                0.6,
                3.0,
            )
            sampled_gradients.append(
                torch.autograd.grad(
                    (output * loss_weights).sum(),
                    inputs,
                )
            )

    for role, dense_gradient in enumerate(expected):
        sampled_mean = torch.stack(
            [gradients[role] for gradients in sampled_gradients]
        ).mean(dim=0)
        torch.testing.assert_close(
            sampled_mean,
            dense_gradient,
            rtol=1e-12,
            atol=1e-12,
        )


def test_exact_bitflip_rejects_batched_inputs():
    query, key, value = _inputs()
    with pytest.raises(ValueError, match="batch size 1"):
        ablation.rosa_soft_exact_bitflip(
            query.expand(2, -1, -1, -1),
            key.expand(2, -1, -1, -1),
            value.expand(2, -1, -1, -1),
        )


@pytest.mark.parametrize(
    ("noise_seeds", "expected_pairs", "expected_pairing"),
    [
        (
            None,
            [(2, 2), (5, 5)],
            "matched",
        ),
        (
            [11, 13],
            [(2, 11), (2, 13), (5, 11), (5, 13)],
            "cartesian",
        ),
    ],
)
def test_run_matrix_separates_model_and_noise_seeds(
    monkeypatch,
    noise_seeds,
    expected_pairs,
    expected_pairing,
):
    observed_pairs = []

    def fake_run_fit(estimator, model_seed, noise_seed, _args):
        observed_pairs.append((model_seed, noise_seed))
        return {
            "estimator": estimator,
            "ever_success": False,
            "final_success": False,
            "first_below_threshold": -1,
            "final_loss": 1.0,
            "best_loss": 1.0,
            "step_ms": 0.0,
        }

    monkeypatch.setattr(ablation, "run_fit", fake_run_fit)
    args = SimpleNamespace(
        device="cpu",
        model_seeds=[2, 5],
        noise_seeds=noise_seeds,
        estimators=["attention_dropout"],
        steps=1,
        success_threshold=1e-3,
        dropout_p=0.1,
        sequence_length=16,
        vocab_size=8,
        motif_min=4,
        motif_max=8,
        heads=2,
        qk_bits=4,
        value_heads=2,
        value_bits=4,
        max_suffix_length=8,
        scale=1.0,
        mismatch_scale=9.0,
        learning_rate=0.01,
        weight_decay=0.01,
        grad_clip=1.0,
    )

    report = ablation.run_matrix(args)

    assert observed_pairs == expected_pairs
    assert report["seed_pairing"] == expected_pairing
