import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext, RosaWork
from .utils import QuantizeFunction, repeat_kv, unfold_qk, decay_qk, gather_x


__all__ = [
    "rosa_bits_ops",
    "RosaBitsWork",
]


def rosa_bits_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: Optional[float] = None,
        grad_eps: float = 1e-3,
        suffix_window: int = 8,
        suffix_factor: Optional[float] = 0.5,
        attention_mask: Optional[Tensor] = None,
        quantize_mode: str = "soft",
        schmitt_trigger: float = 0.0,
        async_op: bool = False,
) -> Union[Tensor, 'RosaBitsWork']:
    """
    Performs the Rapid Online Suffix Automaton (ROSA) attention-like operation.

    This function computes a differentiable, attention-like mechanism based on the
    longest common suffix match between query and key sequences. The inputs are
    expected to be tensors of logits that will be binarized. The operation is designed
    to be efficient on parallel hardware like GPUs.

    Args:
        query (Tensor): (B, H, T, D) Logits for Query bits.
        key (Tensor): (B, H, T, D) Logits for Key bits.
        value (Tensor): (B, H, T, D_v) Logits for Value bits.
        scale (Optional[float]): Scale factor for the attention scores. If not provided, it will default to `1 / D`. Default: `None`.
        grad_eps (float): Epsilon for gradient scaling in the STE quantization. Default: `1e-3`.
        suffix_window (int): Size of the lookback window for fingerprinting. Default: `8`.
        suffix_factor (Optional[float]): Decay factor for the window. If None, it will be dynamically set based on the window size. Default: `0.5`.
        attention_mask (Optional[Tensor]): Attention mask to be applied on the attention scores. Should be broadcastable to (B, H, T, T). Default: `None`.
        quantize_mode (str): Mode for quantizing the inputs. One of "tanh", "soft", or "ste". Default: "soft".
        schmitt_trigger (float): Threshold for Schmitt trigger to prevent noise. Default: `0.0`.
        async_op (bool): Whether to return a work object for asynchronous execution. Default: `False`.

    Returns:
        Tensor: The result of the Hard SAM lookup if async_op is False.
        RosaBitsWork: A work object for asynchronous execution if async_op is True.
    """

    work = RosaBitsWork()
    work._future = RosaContext().update(
        query=query, key=key, value=value, mismatch=0,
        inspect=True, schmitt_trigger=schmitt_trigger,
        async_op=True,
    )

    work._params = RosaBitsParams(
        scale=scale,
        grad_eps=grad_eps,
        suffix_window=suffix_window,
        suffix_factor=suffix_factor,
        attention_mask=attention_mask,
        quantize_mode=quantize_mode,
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
        scale: Optional[float],
        grad_eps: float,
        suffix_window: int,
        suffix_factor: Optional[float],
        attention_mask: Optional[Tensor],
        quantize_mode: str,
    ):
        self.scale = scale
        self.grad_eps = grad_eps
        self.suffix_window = suffix_window
        self.suffix_factor = suffix_factor
        self.quantize_mode = quantize_mode
        
        if isinstance(attention_mask, Tensor):
            self.attention_mask = attention_mask.detach()
        else:
            self.attention_mask = None

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
                scale=params.scale,
                grad_eps=params.grad_eps,
                suffix_window=params.suffix_window,
                suffix_factor=params.suffix_factor,
                attention_mask=params.attention_mask,
                quantize_mode=params.quantize_mode,
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                outputs=x_soft,
                inputs=(query, key, value),
                grad_outputs=grad_output,
                retain_graph=False,
                only_inputs=True,
            )
        return grad_query, grad_key, grad_value, None


def suffix_attention_proxy(
        query: Tensor, key: Tensor, value: Tensor,
        length: Tensor, endpos: Tensor,
        scale: Optional[float], grad_eps: float,
        suffix_window: int,
        suffix_factor: Optional[float],
        attention_mask: Optional[Tensor],
        quantize_mode: str,
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

    ## quantize query, key, value
    if quantize_mode == "tanh":
        xq = torch.tanh(query)
        xk = torch.tanh(key)
        xv = torch.tanh(value)
    elif quantize_mode == "soft":
        xq = F.softsign(query)
        xk = F.softsign(key)
        xv = F.softsign(value)
    elif quantize_mode == "ste":
        xq = QuantizeFunction.apply(query, grad_eps)
        xk = QuantizeFunction.apply(key, grad_eps)
        xv = QuantizeFunction.apply(value, grad_eps)
    else:
        raise ValueError(f"Unsupported quantize_mode: {quantize_mode}, expected one of 'tanh', 'soft', or 'ste'")
    
    ## repeat key and value for multi-head attention
    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    ## unfold query and key for suffix attention
    xq = unfold_qk(xq, win_size=suffix_window, offset=0)
    xk = unfold_qk(xk, win_size=suffix_window, offset=1) # predict next token

    ## apply geometric decay to the query and key features within the suffix window
    xq, xk = decay_qk(xq, xk, decay_factor=suffix_factor)

    ## compute scaled dot product attention
    xq = xq.reshape(bsz, num_heads, seq_len, num_q_bits * suffix_window)
    xk = xk.reshape(bsz, num_heads, seq_len, num_k_bits * suffix_window)

    if scale is None:
        scale = 1.0 / num_qk_bits
    else:
        scale = float(scale)

    xo = F.scaled_dot_product_attention(
        xq, xk, xv, scale=scale,
        is_causal=True, attn_mask=attention_mask,
    )

    ## apply gating based value detach
    pk = gather_x(xk, endpos)
    pv = gather_x(xv, endpos)

    gg = torch.sum(xq * pk, dim=-1, keepdim=True)
    gg = torch.sigmoid(gg * scale)
    
    xo = xo * (1 - gg) + pv * gg
    return xo

    