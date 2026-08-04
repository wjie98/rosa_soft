import pytest
import torch
from torch.utils.checkpoint import checkpoint

import rosa_soft


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)


@pytest.mark.parametrize("varlen", [False, True])
def test_cuda_amp_projection_pipeline_and_grad_scaler(varlen):
    model_dim = 6
    heads = 2
    bits = 3
    value_dim = 2
    tokens = 9
    query_projection = torch.nn.Linear(
        model_dim,
        heads * bits,
        device="cuda",
    )
    key_projection = torch.nn.Linear(
        model_dim,
        heads * bits,
        device="cuda",
    )
    value_projection = torch.nn.Linear(
        model_dim,
        value_dim,
        device="cuda",
    )
    parameters = list(query_projection.parameters())
    parameters += list(key_projection.parameters())
    parameters += list(value_projection.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    source = torch.randn(
        tokens,
        model_dim,
        device="cuda",
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        query = query_projection(source).reshape(tokens, heads, bits)
        key = key_projection(source).reshape(tokens, heads, bits)
        value = value_projection(source).reshape(
            tokens,
            1,
            value_dim,
        )
        if varlen:
            output = rosa_soft.rosa_soft_varlen(
                query,
                key,
                value,
                torch.tensor(
                    [0, 4, tokens],
                    dtype=torch.int32,
                    device="cuda",
                ),
                max_suffix_length=5,
                dropout_p=0.1,
            )
        else:
            output = rosa_soft.rosa_soft(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                max_suffix_length=5,
                dropout_p=0.1,
            )
        loss = output.float().square().mean()

    assert query.dtype == torch.float16
    assert output.dtype == torch.float16
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in parameters
    )
    scaler.step(optimizer)
    scaler.update()


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
def test_cuda_checkpoint_matches_direct_outputs_and_gradients(
    varlen,
    dropout_p,
):
    generator = torch.Generator(device="cuda").manual_seed(100)
    packed_inputs = (
        torch.randn(
            9,
            1,
            3,
            generator=generator,
            device="cuda",
        ),
        torch.randn(
            9,
            1,
            3,
            generator=generator,
            device="cuda",
        ),
        torch.randn(
            9,
            1,
            2,
            generator=generator,
            device="cuda",
        ),
    )
    cu_seqlens = torch.tensor(
        [0, 4, 9],
        dtype=torch.int32,
        device="cuda",
    )

    def operator(*inputs):
        if varlen:
            return rosa_soft.rosa_soft_varlen(
                *inputs,
                cu_seqlens,
                max_suffix_length=5,
                dropout_p=dropout_p,
            )
        return rosa_soft.rosa_soft(
            *(tensor.unsqueeze(0) for tensor in inputs),
            max_suffix_length=5,
            dropout_p=dropout_p,
        ).squeeze(0)

    def run(use_checkpoint):
        torch.cuda.manual_seed(101)
        leaves = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in packed_inputs
        )
        if use_checkpoint:
            output = checkpoint(
                operator,
                *leaves,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            output = operator(*leaves)
        gradients = torch.autograd.grad(output.sum(), leaves)
        return output.detach(), gradients

    direct_output, direct_gradients = run(False)
    checkpoint_output, checkpoint_gradients = run(True)

    assert torch.equal(checkpoint_output, direct_output)
    for checkpoint_gradient, direct_gradient in zip(
        checkpoint_gradients,
        direct_gradients,
    ):
        torch.testing.assert_close(
            checkpoint_gradient,
            direct_gradient,
            rtol=2e-5,
            atol=2e-6,
        )


def _all_match_final_row_value_gradient(dtype, loss_scale=1.0):
    tokens = 40
    query = torch.ones(
        1,
        tokens,
        1,
        4,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    key = torch.ones_like(query, requires_grad=True)
    value = torch.arange(
        tokens,
        device="cuda",
        dtype=dtype,
    ).reshape(1, tokens, 1, 1).requires_grad_()
    output = rosa_soft.rosa_soft(
        query,
        key,
        value,
        max_suffix_length=32,
    )
    (output[0, -1, 0, 0] * loss_scale).backward()
    return value.grad.detach().float().flatten() / loss_scale


def test_cuda_loss_scaling_improves_dense_tail_gradient_accuracy():
    fp16_unscaled = _all_match_final_row_value_gradient(torch.float16)
    fp16_scaled = _all_match_final_row_value_gradient(
        torch.float16,
        loss_scale=1024.0,
    )
    fp32 = _all_match_final_row_value_gradient(torch.float32)

    # Position zero is the null value and never receives a route gradient.
    assert torch.count_nonzero(fp32[1:]) == fp32.numel() - 1
    assert torch.count_nonzero(fp16_scaled[1:]) == fp16_scaled.numel() - 1
    assert torch.linalg.vector_norm(fp16_scaled - fp32) < (
        torch.linalg.vector_norm(fp16_unscaled - fp32)
    )
