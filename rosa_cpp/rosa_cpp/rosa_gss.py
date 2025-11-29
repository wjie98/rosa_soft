import torch
import torch.nn.functional as F

from torch import Tensor
from typing import *

import math

from .ops import *
from .utils import quantize, dequantize, normalize

__all__ = [
    "RosaGSSFunction",
]


class RosaGSSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mismatch: int,
        num_samples: int,
        tau: float,
    ) -> Tensor:
        bsz, num_q_heads, seq_len, num_q_bits = query.size()
        bsz, num_k_heads, seq_len, num_k_bits = key.size()
        bsz, num_v_heads, seq_len, num_v_bits = value.size()

        assert num_k_heads == num_v_heads, f"Key and value must have the same number of heads, got {num_k_heads} and {num_v_heads}."
        assert num_q_bits == num_k_bits, f"Query and key must have the same bit width, got {num_q_bits} and {num_k_bits}."
        assert num_k_bits <= 64, f"Unsupported bit width for key: {num_k_bits}."
        assert num_v_bits <= 64, f"Unsupported bit width for value: {num_v_bits}."

        xq = quantize(query)
        xk = quantize(key)
        xv = quantize(value)

        xq = xq.to("cpu", non_blocking=True)
        xk = xk.to("cpu", non_blocking=True)
        xv = xv.to("cpu", non_blocking=True)

        n_rep = num_q_heads // num_k_heads
        if n_rep > 1:
            xk = xk.view(bsz, num_k_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
            xv = xv.view(bsz, num_v_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
        
        xq = xq.reshape(bsz * num_q_heads, seq_len)
        xk = xk.reshape(bsz * num_q_heads, seq_len)
        xv = xv.reshape(bsz * num_q_heads, seq_len)

        xo = rosa_sam_forward(xq, xk, xv, mismatch)

        xo = xo.to(query.device, non_blocking=True)
        xo = dequantize(xo, value).view(bsz, num_q_heads, seq_len, num_v_bits)

        ctx.save_for_backward(query, key, value)
        ctx.saved_mismatch = mismatch
        ctx.saved_num_samples = num_samples
        ctx.saved_tau = tau

        return xo

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, value = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        mismatch: int = ctx.saved_mismatch
        num_samples: int = ctx.saved_num_samples
        tau: float = ctx.saved_tau

        bsz, num_q_heads, seq_len, num_q_bits = query.size()
        bsz, num_k_heads, seq_len, num_k_bits = key.size()
        bsz, num_v_heads, seq_len, num_v_bits = value.size()

        xq = quantize(query)
        xk = quantize(key)
        xv = quantize(value)

        xq = xq.to("cpu", non_blocking=True)
        xk = xk.to("cpu", non_blocking=True)
        xv = xv.to("cpu", non_blocking=True)

        n_rep = num_q_heads // num_k_heads
        if n_rep > 1:
            xk = xk.view(bsz, num_k_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
            xv = xv.view(bsz, num_v_heads, 1, seq_len).repeat(1, 1, n_rep, 1)

        xq = xq.reshape(bsz * num_q_heads, seq_len)
        xk = xk.reshape(bsz * num_q_heads, seq_len)
        xv = xv.reshape(bsz * num_q_heads, seq_len)
        
        _, indptr, indices, quality = rosa_gss_forward(xq, xk, xv, mismatch, num_samples=num_samples, tau=tau)

        indptr = indptr.to(query.device)
        indices = indices.to(query.device)
        quality = quality.to(query.device)

        num_nodes = bsz * num_q_heads * seq_len

        G = torch.sparse_csr_tensor(indptr, indices, torch.log1p(quality), size=(num_nodes, num_nodes)).to_sparse_coo()
        row, col = G.indices()
        attn_bias = G.values()

        with torch.enable_grad():
            query = query.detach().requires_grad_()
            key = key.detach().requires_grad_()
            value = value.detach().requires_grad_()

            xq = normalize(query)
            xk = normalize(key)
            xv = normalize(value)

            xq = torch.tanh(xq / tau)
            xk = torch.tanh(xk / tau)
            xv = torch.sigmoid(xv / tau)

            xv = F.pad(xv, (0, 0, -1, 1)) # recall next token

            if n_rep > 1:
                xk = xk.view(bsz, num_k_heads, 1, seq_len, num_k_bits).repeat(1, 1, n_rep, 1, 1)
                xv = xv.view(bsz, num_v_heads, 1, seq_len, num_v_bits).repeat(1, 1, n_rep, 1, 1)
            
            xq = xq.reshape(num_nodes, num_q_bits)
            xk = xk.reshape(num_nodes, num_k_bits)
            xv = xv.reshape(num_nodes, num_v_bits)

            alpha = 1.0 / math.sqrt(num_q_bits) / tau
            beta = 1.0 / tau

            xq = xq[row].float()
            xk = xk[col].float()

            logits = (xq * xk).sum(dim=-1) * alpha + attn_bias * beta

            with torch.no_grad():
                max_logits = torch.full((num_nodes,), -torch.inf, dtype=logits.dtype, device=logits.device)
                max_logits.index_reduce_(0, row, logits, reduce="amax", include_self=True)
                max_logits = max_logits[row]
            
            exp_qk = torch.exp(logits - max_logits)

            xo = torch.zeros((num_nodes, num_v_bits), dtype=xv.dtype, device=xv.device)
            xo = xo.index_add_(0, row, xv[col].float() * exp_qk.view(-1, 1))

            xw = torch.zeros((num_nodes,), dtype=exp_qk.dtype, device=exp_qk.device)
            xw = xw.index_add_(0, row, exp_qk)

            output = (xo / xw.view(-1, 1)).type_as(grad_output)

            grad_output = grad_output.reshape(num_nodes, num_v_bits)
            grad_query, grad_key, grad_value = torch.autograd.grad(
                output,
                (query, key, value),
                grad_output
            )
        
        grad_query = grad_query.detach().reshape(bsz, num_q_heads, seq_len, num_q_bits)
        grad_key = grad_key.detach().reshape(bsz, num_k_heads, seq_len, num_k_bits)
        grad_value = grad_value.detach().reshape(bsz, num_v_heads, seq_len, num_v_bits)

        return grad_query, grad_key, grad_value, None, None, None

