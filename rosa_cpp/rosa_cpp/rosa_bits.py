import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext, RosaWork


__all__ = [
    "rosa_bits_ops",
]


def rosa_bits_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        suffix_window: int = 8,
        suffix_factor: Optional[float] = None,
        attention_mask: Optional[Tensor] = None,
        attention_tau: float = 0.1,
        threshold: float | None = None,
        async_op: bool = False,
) -> Union[Tensor, 'RosaBitsWork']:
    """
    Performs the Rapid Online Suffix Automaton (ROSA) attention-like operation.

    This function computes a differentiable, attention-like mechanism based on the
    longest common suffix match between query and key sequences. The inputs are
    expected to be tensors of logits that will be binarized). The operation is designed
    to be efficient on parallel hardware like GPUs.

    Args:
        query (Tensor): (B, H, T, D) Logits for Query bits (pre-tanh).
        key (Tensor): (B, H, T, D) Logits for Key bits (pre-tanh).
        value (Tensor): (B, H, T, D_v) Logits for Value bits.
        suffix_window (int): Size of the lookback window for fingerprinting.
        suffix_factor (Optional[float]): Decay factor for the window.
        async_op (bool): Whether to return a work object for asynchronous execution.

    Returns:
        Tensor: The result of the Hard SAM lookup if async_op is False.
        RosaBitsWork: A work object for asynchronous execution if async_op is True.
    """

    work = RosaBitsWork()
    work._future = RosaContext().update(query, key, value, 0, thresh=threshold, inspect=True, async_op=True)
    work._params = RosaBitsParams(
        suffix_window=suffix_window,
        suffix_factor=suffix_factor,
        attention_mask=attention_mask,
        attention_tau=attention_tau,
    )
    work._query_key_value = (query, key, value)

    if async_op:
        return work
    return work.wait()


class RosaBitsWork:
    def __init__(self):
        self._future: RosaWork
        self._params: RosaBitsParams
        self._query_key_value: Tuple[Tensor, Tensor, Tensor]

    def wait(self):
        if self._future is None:
            raise RuntimeError("wait() called twice")
        
        work = self._future
        params = self._params
        query, key, value = self._query_key_value

        x_hard, info = work.wait()
        params.info["x_hard"] = x_hard
        params.info.update(info)

        self._future = None
        self._params = None
        self._query_key_value = None

        return RosaBitsFunction.apply(query, key, value, params)


class RosaBitsParams:
    def __init__(self,
        suffix_window: int,
        suffix_factor: Optional[float],
        attention_mask: Optional[Tensor],
        attention_tau: float,
    ):
        self.suffix_window = suffix_window
        self.suffix_factor = suffix_factor
        
        if isinstance(attention_mask, Tensor):
            self.attention_mask = attention_mask.detach()
        else:
            self.attention_mask = None
        self.attention_tau = attention_tau

        self.info: Dict[str, Tensor] = {}
        self._ctx: Optional[RosaContext] = None


class RosaBitsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: RosaBitsParams):
        if "x_hard" in params.info:
            x_hard = params.info.pop("x_hard")
        else:
            x_hard, info = RosaContext().update(query, key, value, 0, inspect=True)
            params.info.update(info)

        ctx.save_for_backward(
            query.detach(),
            key.detach(),
            value.detach(),
        )
        ctx.saved_params = params

        return x_hard
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, value = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        params: RosaBitsParams = ctx.saved_params

        length = params.info.pop("length")
        endpos = params.info.pop("endpos")

        with torch.enable_grad():
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

            x_soft = suffix_attention_proxy(
                query, key, value,
                length=length,
                endpos=endpos,
                suffix_window=params.suffix_window,
                suffix_factor=params.suffix_factor,
                attention_mask=params.attention_mask,
                attention_tau=params.attention_tau,
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                outputs=x_soft,
                inputs=(query, key, value),
                grad_outputs=grad_output,
                retain_graph=False,
                only_inputs=True,
            )
        return grad_query, grad_key, grad_value, None


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    bsz, num_heads, seq_len, head_dim = hidden_states.size()
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, -1, seq_len, head_dim)


def unfold_qk(hidden_states: Tensor, win_size: int, offset: int = 0):
    hidden_states = F.pad(hidden_states, (0, 0, win_size - 1 + offset, - offset))
    hidden_states = hidden_states.unfold(-2, win_size, 1).transpose(-2, -1)
    return hidden_states


def decay_qk(xq: Tensor, xk: Tensor, decay_factor: Optional[float] = None):
    bsz, num_heads, seq_len, win_size, num_bits = xq.size()
    
    if win_size > 1:
        inds = torch.arange(win_size, device=xq.device)
        inds = win_size - 1 - inds

        if decay_factor is None: # dynamic
            wd = max(0.1, 1.0 / win_size) ** (1.0 / win_size)
        else:
            wd = max(0.1, min(decay_factor, 0.99))
        
        qk_w = torch.pow(wd, inds)
        qk_w = qk_w / qk_w.sum()
        qk_w_sqrt = torch.sqrt(qk_w) # [win_size]

        xq = xq * qk_w_sqrt.view(-1, 1).type_as(xq)
        xk = xk * qk_w_sqrt.view(-1, 1).type_as(xk)
    return xq, xk


def gather_x(x: Tensor, endpos: Tensor, offset: int = 1):
    bsz, num_heads, seq_len, num_bits = x.size()
    with torch.no_grad():
        epos = endpos.view(bsz, num_heads, seq_len)
    
    mask = (epos >= 0).view(bsz, num_heads, seq_len, 1).type_as(x)
    inds = (epos + offset).clamp(0, seq_len - 1)
    inds = inds.view(bsz, num_heads, seq_len, 1).expand_as(x)

    return x.gather(-2, inds) * mask # mask out the padding tokens


def suffix_attention_proxy(
        query: Tensor, key: Tensor, value: Tensor,
        length: Tensor, endpos: Tensor,
        suffix_window: int,
        suffix_factor: Optional[float],
        attention_mask: Optional[Tensor],
        attention_tau: float,
):
    bsz, num_heads, seq_len, num_q_bits = query.size()
    bsz, num_kv_heads, seq_len, num_k_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    num_qk_bits = num_q_bits
    assert num_q_bits == num_k_bits, "query and key must have the same number of bits"

    MAX_ATTENTION_HEAD_DIM = 256
    assert 0 < num_qk_bits * suffix_window <= MAX_ATTENTION_HEAD_DIM

    # q = torch.linspace(0, 1, 10, device=query.device)
    # qq = torch.quantile(query.flatten(), q, dim=0)
    # qk = torch.quantile(key.flatten(), q, dim=0)
    # qv = torch.quantile(value.flatten(), q, dim=0)
    # print(qq.tolist())
    # print(qk.tolist())
    # print(qv.tolist())

    # xq = torch.tanh(query)
    # xk = torch.tanh(key)
    # xv = torch.tanh(value)

    xq = F.softsign(query)
    xk = F.softsign(key)
    xv = F.softsign(value)

    # xq = QuantizeFunction.apply(query, 1e-3)
    # xk = QuantizeFunction.apply(key, 1e-3)
    # xv = QuantizeFunction.apply(value, 1e-3)

    # xq = xq + (torch.where(query > 0, 1.0, -1.0) - xq).detach()
    # xk = xk + (torch.where(key   > 0, 1.0, -1.0) - xk).detach()
    # xv = xv + (torch.where(value > 0, 1.0, -1.0) - xv).detach() # [-1, 1] is better

    attention_scale = 1.0 / num_qk_bits / attention_tau
    
    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    xq = unfold_qk(xq, win_size=suffix_window, offset=0)
    xk = unfold_qk(xk, win_size=suffix_window, offset=1) # predict next token

    xq, xk = decay_qk(xq, xk, decay_factor=suffix_factor)

    xq = xq.reshape(bsz, num_heads, seq_len, num_q_bits * suffix_window)
    xk = xk.reshape(bsz, num_heads, seq_len, num_k_bits * suffix_window)

    xo = F.scaled_dot_product_attention(
        xq, xk, xv, scale=attention_scale,
        is_causal=True, attn_mask=attention_mask,
    )

    pk = gather_x(xk, endpos)
    pv = gather_x(xv, endpos)

    gg = torch.sum(xq * pk, dim=-1, keepdim=True)
    gg = torch.sigmoid(gg * attention_scale)
    
    xo = xo * (1 - gg) + pv * gg
    return xo


class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, tau: float):
        ctx.save_for_backward(x)
        ctx.saved_tau = tau
        return torch.where(x > 0, 1.0, -1.0)
    
    @staticmethod
    def backward(ctx, g: Tensor):
        x: Tensor = ctx.saved_tensors[0]
        tau: float = ctx.saved_tau

        mag = x.abs().clamp_min_(tau)
        return g / mag, None
    