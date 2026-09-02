import math

import torch
import torch.nn.functional as F

from benchmarks.state_attention_init import initialize_state_attention_model
from benchmarks.state_attention_pretraining import (
    NextTokenEstimatorLM,
    _seed_pairs,
    build_parser,
    make_next_token_recall_batch,
)


def _model(*, value_bits=4, estimator="state_quadratic_attention"):
    return NextTokenEstimatorLM(
        associations=4,
        hidden_size=32,
        payload_bits=4,
        num_heads=2,
        qk_bits=8,
        value_heads=1,
        value_bits=value_bits,
        context_scale=0.25,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
        estimator=estimator,
        bit_temperature=0.5,
        antithetic_pairs=1,
        context_depth=2,
    )


def _initialize(model, *, qk_init="default", value_init="default"):
    initialize_state_attention_model(
        model,
        seed=91,
        qk_init=qk_init,
        qk_shared_bits=4,
        qk_correlation=0.97,
        qk_logit_median=0.6,
        value_init=value_init,
        value_logit_median=0.6,
    )


def test_partial_shared_qk_has_requested_geometry():
    model = _model()
    _initialize(model, qk_init="partial_shared_orthogonal")
    query = model.query.weight.view(2, 8, 32)
    key = model.key.weight.view(2, 8, 32)

    shared_cosine = F.cosine_similarity(query[:, :4], key[:, :4], dim=-1)
    private_cosine = F.cosine_similarity(query[:, 4:], key[:, 4:], dim=-1)

    torch.testing.assert_close(
        shared_cosine,
        torch.full_like(shared_cosine, 0.97),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        private_cosine,
        torch.zeros_like(private_cosine),
        rtol=0,
        atol=1e-6,
    )


def test_module_seeds_keep_common_weights_equal_when_value_dim_changes():
    narrow = _model(value_bits=4)
    wide = _model(value_bits=8)
    _initialize(narrow)
    _initialize(wide)

    torch.testing.assert_close(narrow.embedding.weight, wide.embedding.weight)
    torch.testing.assert_close(narrow.recurrent.weight_ih, wide.recurrent.weight_ih)
    torch.testing.assert_close(narrow.query.weight, wide.query.weight)
    torch.testing.assert_close(narrow.key.weight, wide.key.weight)
    torch.testing.assert_close(narrow.head.weight, wide.head.weight)


def test_payload_vocabulary_is_independent_of_hard_value_dimension():
    model = NextTokenEstimatorLM(
        associations=2,
        hidden_size=16,
        payload_bits=2,
        num_heads=1,
        qk_bits=4,
        value_heads=1,
        value_bits=8,
        context_scale=0.25,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
        estimator="state_quadratic_attention",
        bit_temperature=0.5,
        antithetic_pairs=1,
        context_depth=1,
    )
    batch = make_next_token_recall_batch(
        seed=7,
        pairs=1,
        associations=2,
        payload_bits=2,
    )

    assert model.vocab_size == 8
    assert model.value.out_features == 8
    assert model.output.in_features == 8
    assert int(batch.tokens.max()) < model.vocab_size
    assert model(batch.tokens).shape == (2, 10, model.vocab_size)


def test_paired_output_is_scaled_value_transpose():
    model = _model(value_bits=8)
    _initialize(model, value_init="paired_output")
    value = model.value.weight
    expected = value.transpose(0, 1) / math.sqrt(model.num_heads)
    output = model.output.weight.view(32, model.num_heads, 8)

    for head in range(model.num_heads):
        torch.testing.assert_close(output[:, head], expected)


def test_controlled_initialization_preserves_exact_hard_forward():
    model = _model(value_bits=8)
    _initialize(
        model,
        qk_init="partial_shared_orthogonal",
        value_init="paired_output",
    )
    generator = torch.Generator().manual_seed(13)
    query = torch.randn(2, 7, 2, 8, generator=generator)
    key = torch.randn(2, 7, 2, 8, generator=generator)
    value = torch.randn(2, 7, 1, 8, generator=generator)

    actual = model._routed_values(query, key, value, "rosa")
    model.estimator = "production"
    expected = model._routed_values(query, key, value, "rosa")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_model_and_data_seeds_form_a_cross_product_when_requested():
    args = build_parser().parse_args(
        [
            "--model-seeds",
            "1",
            "2",
            "--data-seeds",
            "3",
            "4",
        ]
    )

    assert _seed_pairs(args) == [(1, 3), (1, 4), (2, 3), (2, 4)]
