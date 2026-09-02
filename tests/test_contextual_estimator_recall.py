import json

import torch

from benchmarks.contextual_estimator_recall import (
    EstimatorResetRnnRosaLM,
    build_parser,
    run_benchmark,
)


def _model(estimator):
    return EstimatorResetRnnRosaLM(
        associations=2,
        hidden_size=8,
        num_heads=1,
        qk_bits=2,
        value_heads=1,
        value_bits=2,
        context_scale=0.25,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
        estimator=estimator,
        bit_temperature=0.5,
        antithetic_pairs=2,
    )


def test_context_depth_one_preserves_the_original_encoder_exactly():
    from examples.contextual_rnn_recall_gate import ResetRnnRosaLM

    estimator_model = _model("production")
    original_model = ResetRnnRosaLM(
        associations=2,
        hidden_size=8,
        num_heads=1,
        qk_bits=2,
        value_heads=1,
        value_bits=2,
        context_scale=0.25,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
    )
    original_model.load_state_dict(estimator_model.state_dict())
    tokens = torch.tensor([[0, 2, 1, 3, 6, 0, 1]])

    torch.testing.assert_close(
        estimator_model.encode_residual(tokens),
        original_model.encode_residual(tokens),
        rtol=0,
        atol=0,
    )


def test_deep_context_reset_keeps_query_residuals_assignment_independent():
    from examples.contextual_rnn_recall_gate import make_contextual_recall_batch

    recall_batch = make_contextual_recall_batch(
        seed=17,
        pairs=2,
        associations=2,
        value_bits=2,
    )
    model = EstimatorResetRnnRosaLM(
        associations=2,
        hidden_size=8,
        num_heads=1,
        qk_bits=2,
        value_heads=1,
        value_bits=2,
        context_scale=0.25,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
        estimator="state_quadratic_attention",
        bit_temperature=0.5,
        antithetic_pairs=2,
        context_depth=3,
    )

    residual = model.encode_residual(recall_batch.tokens)
    query_residual = residual[:, recall_batch.query_positions]

    torch.testing.assert_close(
        query_residual,
        query_residual[:1].expand_as(query_residual),
        rtol=0,
        atol=0,
    )


def test_research_estimators_preserve_one_hard_forward():
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(2, 5, 1, 2, generator=generator)
    key = torch.randn(2, 5, 1, 2, generator=generator)
    value = torch.randn(2, 5, 1, 2, generator=generator)
    expected = _model("production")._routed_values(
        query, key, value, "rosa"
    )

    for estimator in (
        "bitflip",
        "mean_field",
        "arm",
        "disarm",
        "state_linear_delta",
        "state_quadratic_delta",
        "state_cubic_delta",
        "state_full_delta",
        "state_linear_attention",
        "state_quadratic_attention",
        "state_cubic_attention",
        "state_full_linear_attention",
    ):
        inputs = [tensor.detach().clone().requires_grad_() for tensor in (
            query,
            key,
            value,
        )]
        torch.manual_seed(31)
        actual = _model(estimator)._routed_values(*inputs, "rosa")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        assert all(tensor.grad is not None for tensor in inputs)


def test_contextual_estimator_benchmark_smoke_schema():
    args = build_parser().parse_args(
        [
            "--estimators",
            "production",
            "mean_field",
            "arm",
            "disarm",
            "--seeds",
            "0",
            "--train-pairs",
            "2",
            "--validation-pairs",
            "2",
            "--associations",
            "2",
            "--hidden-size",
            "8",
            "--heads",
            "1",
            "--qk-bits",
            "2",
            "--value-heads",
            "1",
            "--value-bits",
            "2",
            "--steps",
            "1",
            "--baseline-steps",
            "0",
            "--antithetic-pairs",
            "2",
        ]
    )

    report = run_benchmark(args)

    assert report["schema_version"] == 1
    assert report["bitflip_gradient_scale"] == 1.0
    json.dumps(report, allow_nan=False)
    assert set(report["summary"]) == set(args.estimators)
    assert report["shortcut_checks_passed"] is True
    run = report["runs"][0]
    assert run["residual_only_baseline"]["training_loss"] is None
    assert set(run["candidates"]) == set(args.estimators)
    assert all(
        candidate["validation"]["query_residual_max_difference"] == 0.0
        for candidate in run["candidates"].values()
    )
