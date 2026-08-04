import pytest
import torch

from benchmarks.internal_language import StatefulSymbolizer
from benchmarks.latent_grammar_gate import (
    active_content_positions,
    make_grammar_batch,
    oracle_route_accuracy,
    required_candidate_count,
)


def test_required_candidate_counts_cross_each_previous_capacity_boundary():
    assert required_candidate_count(2) == 3
    assert required_candidate_count(4) == 5
    assert required_candidate_count(8) == 17
    assert active_content_positions(2) == (0, 1)
    assert active_content_positions(4) == (0, 2, 3)
    assert active_content_positions(8) == (0, 4, 5, 6, 7)


def test_grammar_batch_has_complementary_payloads_and_identical_queries():
    batch = make_grammar_batch(seed=3, pairs=5, phrase_length=4)
    query_start = batch.query_position - batch.phrase_length + 1

    assert torch.equal(batch.inputs[0::2, query_start:], batch.inputs[1::2, query_start:])
    assert torch.equal(batch.reset_mask[0::2], batch.reset_mask[1::2])
    assert torch.equal(batch.content_mask[0::2], batch.content_mask[1::2])
    assert torch.equal(batch.target[0::2], -batch.target[1::2])
    assert torch.equal(
        batch.value[0::2, batch.candidate_routes],
        -batch.value[1::2, batch.candidate_routes],
    )
    assert torch.all(batch.candidate_routes < batch.query_position)


@pytest.mark.parametrize("phrase_length", [2, 4, 8])
def test_explicit_codebook_needs_the_full_capacity_window(phrase_length):
    candidate_count = required_candidate_count(phrase_length)
    batch = make_grammar_batch(
        seed=4,
        pairs=candidate_count,
        phrase_length=phrase_length,
        candidate_count=candidate_count,
    )

    assert oracle_route_accuracy(batch, phrase_length) == 1.0
    assert oracle_route_accuracy(batch, phrase_length - 1) < 1.0


def test_stateless_blank_tail_cannot_unfold_cue_identity():
    batch = make_grammar_batch(seed=5, pairs=5, phrase_length=4)
    candidate_count = batch.candidate_routes.numel()
    torch.manual_seed(9)
    model = StatefulSymbolizer(
        input_size=candidate_count + 1,
        state_size=12,
        heads=1,
        bits=1,
        stateful=False,
    )
    phrase_inputs = []
    chunk = batch.phrase_length + 1
    for cue in range(candidate_count):
        phrase_inputs.append(batch.inputs[0, cue * chunk : cue * chunk + batch.phrase_length])
    symbols = model(torch.stack(phrase_inputs))

    assert torch.equal(
        symbols.query[0, 1:].expand_as(symbols.query[:, 1:]),
        symbols.query[:, 1:],
    )


def test_stateful_grammar_loss_reaches_qk_and_recurrent_parameters():
    batch = make_grammar_batch(seed=6, pairs=3, phrase_length=2)
    candidate_count = batch.candidate_routes.numel()
    torch.manual_seed(10)
    model = StatefulSymbolizer(
        input_size=candidate_count + 1,
        state_size=8,
        heads=1,
        bits=1,
        stateful=True,
    )
    symbols = model(batch.inputs, reset_mask=batch.reset_mask)
    from rosa_soft import rosa_soft_reference

    output = rosa_soft_reference(
        symbols.query,
        symbols.key,
        batch.value,
        max_suffix_length=2,
    )[:, batch.query_position, 0]
    loss = (output - batch.target).square().mean()
    loss.backward()

    assert model.query_projection.weight.grad is not None
    assert model.key_projection.weight.grad is not None
    assert model.state_projection.weight.grad is not None
