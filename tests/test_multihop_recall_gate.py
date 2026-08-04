import torch

from benchmarks.multihop_recall_gate import (
    MultihopRouter,
    make_multihop_batch,
    multihop_loss,
)


def test_multihop_batch_changes_only_values_across_complementary_pair():
    batch = make_multihop_batch(
        seed=3,
        pairs=5,
        address_count=4,
        address_bits=6,
    )

    assert torch.equal(batch.inputs[0::2], batch.inputs[1::2])
    assert torch.equal(batch.target[0::2], -batch.target[1::2])
    assert torch.all(
        (batch.expected_feedback[0::2] != batch.expected_feedback[1::2]).any(dim=-1)
    )
    assert torch.all(batch.source_routes < batch.first_query_position)
    assert torch.all(batch.expected_second_routes < batch.first_query_position)


def test_multihop_forward_is_exact_hard_and_feedback_path_is_differentiable():
    batch = make_multihop_batch(
        seed=8,
        pairs=3,
        address_count=3,
        address_bits=5,
    )
    torch.manual_seed(19)
    model = MultihopRouter(
        address_bits=5,
        state_size=12,
        qk_bits=4,
        update_rate=1.0,
        aligned_head_init=False,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
    )
    output = model(batch)
    loss = multihop_loss(output, batch.target)
    loss.backward()

    assert torch.all((output == -1) | (output == 0) | (output == 1))
    assert model.symbolizer.query_projection.weight.grad is not None
    assert model.symbolizer.key_projection.weight.grad is not None
    assert torch.count_nonzero(model.symbolizer.query_projection.weight.grad) > 0


def test_detached_feedback_removes_first_hop_query_gradient_contribution():
    batch = make_multihop_batch(
        seed=11,
        pairs=2,
        address_count=3,
        address_bits=5,
    )
    torch.manual_seed(23)
    model = MultihopRouter(
        address_bits=5,
        state_size=10,
        qk_bits=4,
        update_rate=1.0,
        aligned_head_init=False,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
    )

    routed = model(batch, feedback_mode="routed")
    routed_loss = multihop_loss(routed, batch.target)
    routed_loss.backward()
    routed_grad = model.symbolizer.input_projection.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    detached = model(batch, feedback_mode="detached")
    multihop_loss(detached, batch.target).backward()
    detached_grad = model.symbolizer.input_projection.weight.grad

    assert not torch.equal(routed_grad, detached_grad)
