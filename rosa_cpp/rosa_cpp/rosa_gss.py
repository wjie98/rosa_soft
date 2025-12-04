import torch
import torch.nn.functional as F

from torch import Tensor
from typing import *

import math
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from .ops import *
from .utils import quantize, dequantize, normalize
from .rosa_sam import RosaStream


__all__ = [
    "rosa_gss_ops",
]


def rosa_gss_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mismatch: int = 0,
        num_samples: int = 8,
        tau: float = 1.0,
        training: bool = False,
        async_op: bool = False,
) -> Union[Tensor, 'RosaGSSWork']:
    trace_tensor, work = RosaGSSDispatchFunction.apply(
        query, key, value, mismatch,
        num_samples, tau, training
    )

    setattr(work, "_trace_tensor", trace_tensor) # saved for autograd

    if async_op:
        return work
    return work.wait()


class RosaGSSWork:
    def __init__(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mismatch: int,
            num_samples: int,
            tau: float,
            training: bool,
    ):
        self.query = query.detach()
        self.key = key.detach()
        self.value = value.detach()

        self.mismatch = mismatch
        self.num_samples = num_samples
        self.tau = tau
        self.training = training

        bsz, num_q_heads, seq_len, num_q_bits = self.query.size()
        bsz, num_k_heads, seq_len, num_k_bits = self.key.size()
        bsz, num_v_heads, seq_len, num_v_bits = self.value.size()

        assert num_k_heads == num_v_heads, f"Key and value must have the same number of heads, got {num_k_heads} and {num_v_heads}."
        assert num_q_bits == num_k_bits, f"Query and key must have the same bit width, got {num_q_bits} and {num_k_bits}."
        assert num_k_bits <= 64, f"Unsupported bit width for key: {num_k_bits}."
        assert num_v_bits <= 64, f"Unsupported bit width for value: {num_v_bits}."

        self._trace_tensor: Optional[Tensor] = None
    
    def wait(self) -> Tensor:
        trace_tensor: Tensor = getattr(self, "_trace_tensor")
        delattr(self, "_trace_tensor")

        output = RosaGSSCombineFunction.apply(trace_tensor, self)
        return output
    
    def _forward_dispatch(self):
        xq = quantize(self.query)
        xk = quantize(self.key)
        xv = quantize(self.value)

        with self.stream(post_wait=False):
            xq = xq.to("cpu", non_blocking=True)
            xk = xk.to("cpu", non_blocking=True)
            xv = xv.to("cpu", non_blocking=True)

        self.saved_tensors = (xq, xk, xv)
    
    def _forward_combine(self):
        bsz, num_q_heads, seq_len, num_q_bits = self.query.size()
        bsz, num_k_heads, seq_len, num_k_bits = self.key.size()
        bsz, num_v_heads, seq_len, num_v_bits = self.value.size()

        xq, xk, xv = self.saved_tensors

        n_rep = num_q_heads // num_k_heads
        if n_rep > 1:
            xk = xk.view(bsz, num_k_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
            xv = xv.view(bsz, num_v_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
        
        xq = xq.reshape(bsz * num_q_heads, seq_len)
        xk = xk.reshape(bsz * num_q_heads, seq_len)
        xv = xv.reshape(bsz * num_q_heads, seq_len)

        if self.training:
            xo, indptr, indices, quality = rosa_gss_forward(
                xq, xk, xv, self.mismatch,
                num_samples=self.num_samples, tau=self.tau,
            )
            self.saved_samples = (indptr, indices, quality)
        else:
            xo = rosa_sam_forward(xq, xk, xv, self.mismatch)

        with self.stream(prev_wait=False):
            xo = xo.to(self.query.device, non_blocking=True)

        xo = dequantize(xo, self.value)
        xo = xo.view(bsz, num_q_heads, seq_len, num_v_bits)
        return xo
    
    def _backward_dispatch(self, grad_output: Tensor):
        indptr, indices, quality = self.saved_samples

        with self.stream(prev_wait=False):
            indptr = indptr.to(self.query.device, non_blocking=True)
            indices = indices.to(self.query.device, non_blocking=True)
            quality = quality.to(self.query.device, non_blocking=True)

        self.saved_samples = (indptr, indices, quality)
        self.saved_grad_output = grad_output.detach()
    
    def _backward_combine(self):
        indptr, indices, quality = self.saved_samples
        grad_output = self.saved_grad_output

        bsz, num_q_heads, seq_len, num_q_bits = self.query.size()
        bsz, num_k_heads, seq_len, num_k_bits = self.key.size()
        bsz, num_v_heads, seq_len, num_v_bits = self.value.size()

        num_nodes = bsz * num_q_heads * seq_len

        G = torch.sparse_csr_tensor(indptr, indices, torch.log1p(quality), size=(num_nodes, num_nodes)).to_sparse_coo()
        row, col = G.indices()
        attn_bias = G.values()

        tau = self.tau
        n_rep = num_q_heads // num_k_heads

        with torch.enable_grad():
            query = self.query.detach().requires_grad_()
            key = self.key.detach().requires_grad_()
            value = self.value.detach().requires_grad_()

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

        return grad_query, grad_key, grad_value

    @property
    def stream(self):
        return self._stream()
    
    @staticmethod
    @lru_cache(maxsize=None)
    def _stream():
        return RosaStream()
    
    @property
    def executor(self):
        return self._executor()
    
    @staticmethod
    @lru_cache(maxsize=None)
    def _executor():
        return ThreadPoolExecutor(max_workers=2)


class RosaGSSDispatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mismatch: int,
        num_samples: int,
        tau: float,
        training: bool,
    ) -> Tensor:
        work = RosaGSSWork(
            query=query,
            key=key,
            value=value,
            mismatch=mismatch,
            num_samples=num_samples,
            tau=tau,
            training=training,
        )

        work._forward_dispatch()
        ctx.saved_rosa_work = work

        trace_tensor = torch.empty(0, dtype=torch.float32, device="cpu")
        return trace_tensor, work

    @staticmethod
    def backward(ctx, grad_trace: Tensor, _: Any):
        work: RosaGSSWork = ctx.saved_rosa_work
        grad_query, grad_key, grad_value = work._backward_combine()
        return grad_query, grad_key, grad_value, None, None, None, None


class RosaGSSCombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        trace_tensor: Tensor,
        work: RosaGSSWork,
    ) -> Tensor:
        ctx.saved_rosa_work = work
        return work._forward_combine()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        work: RosaGSSWork = ctx.saved_rosa_work
        work._backward_dispatch(grad_output)

        grad_trace = torch.empty(0, dtype=torch.float32, device="cpu")
        return grad_trace, None

