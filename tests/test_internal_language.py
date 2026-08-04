import pytest
import torch

from benchmarks.internal_language import StatefulSymbolizer


def _symbolizer(*, stateful: bool = True, feedback_size: int = 0):
    torch.manual_seed(7)
    return StatefulSymbolizer(
        input_size=4,
        state_size=6,
        heads=2,
        bits=3,
        update_rate=0.25,
        feedback_size=feedback_size,
        stateful=stateful,
    )


def test_stateful_symbolizer_is_deterministic_and_has_separate_heads():
    model = _symbolizer()
    inputs = torch.randn(3, 5, 4)

    first = model(inputs)
    second = model(inputs)

    assert torch.equal(first.latent, second.latent)
    assert torch.equal(first.query, second.query)
    assert torch.equal(first.key, second.key)
    assert first.latent.shape == (3, 5, 6)
    assert first.query.shape == (3, 5, 2, 3)
    assert first.key.shape == (3, 5, 2, 3)
    assert (
        model.query_projection.weight.data_ptr()
        != model.key_projection.weight.data_ptr()
    )


def test_stateful_symbolizer_retains_an_impulse_and_reset_erases_it():
    model = _symbolizer()
    impulse = torch.zeros(2, 5, 4)
    impulse[0, 0, 0] = 1.0
    impulse[1, 0, 1] = 1.0
    no_reset = model(impulse)
    reset_mask = torch.zeros(2, 5, dtype=torch.bool)
    reset_mask[:, 2] = True
    with_reset = model(impulse, reset_mask=reset_mask)

    assert not torch.equal(no_reset.latent[:, 1], no_reset.latent[:, 0])
    assert not torch.equal(no_reset.latent[:, 4], with_reset.latent[:, 4])
    assert torch.equal(with_reset.latent[0, 2:], with_reset.latent[1, 2:])


def test_stateless_symbolizer_does_not_carry_an_impulse():
    model = _symbolizer(stateful=False)
    inputs = torch.zeros(2, 4, 4)
    inputs[0, 0, 0] = 1.0
    inputs[1, 0, 1] = 1.0
    output = model(inputs)

    assert torch.equal(output.latent[0, 1:], output.latent[1, 1:])


def test_feedback_changes_symbols_and_receives_gradients():
    model = _symbolizer(feedback_size=3)
    inputs = torch.randn(2, 4, 4, requires_grad=True)
    feedback = torch.randn(2, 4, 3, requires_grad=True)
    output = model(inputs, feedback=feedback)
    loss = output.query.square().mean() + output.key.square().mean()
    loss.backward()

    assert feedback.grad is not None
    assert torch.count_nonzero(feedback.grad) > 0
    assert inputs.grad is not None
    assert model.state_projection.weight.grad is not None


def test_stateful_symbolizer_rejects_an_empty_sequence():
    model = _symbolizer()
    with pytest.raises(ValueError, match="at least one position"):
        model(torch.empty(2, 0, 4))


@pytest.mark.parametrize("update_rate", [0.0, 1.1])
def test_stateful_symbolizer_rejects_invalid_update_rate(update_rate):
    with pytest.raises(ValueError, match="update_rate"):
        StatefulSymbolizer(
            input_size=4,
            state_size=6,
            heads=1,
            bits=2,
            update_rate=update_rate,
        )
