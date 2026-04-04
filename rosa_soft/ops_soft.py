import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .sam import RosaContext
from .future import RosaSoftWork

from .utils import quantize


__all__ = [
    "rosa_soft_ops",
]


def rosa_soft_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    quant_mode: str = "cubic",
    quant_scale: Optional[float] = None,
    schmitt_trigger: float = 0.0,
    async_op: bool = False,
) -> Union[Tensor, RosaSoftWork]:
    f"""ROSA Soft Dynamic Programming operator (Baseline).

    Exact softening of suffix matching via recurrent gated accumulation.
    Forward uses discrete ROSA (hard SAM), backward uses soft DP gradients
    via Straight-Through Estimator (STE).

    Complexity: O(T^2) time, O(T^2) space.

    Args:
        query: (B, T, H, D) Query logits.
        key: (B, T, H, D) Key logits.
        value: (B, T, H_v, D_v) Value logits.
        scale: Attention scale factor.
        quant_mode: Quantization mode ("tanh" or "soft").
        quant_scale: Quantization scale factor.
        schmitt_trigger: Threshold for noise filtering.
        async_op: Return work object for async execution.

    Returns:
        Tensor if async_op=False, RosaSoftWork if async_op=True.
    """

    batch_size, _, num_heads, _ = query.size()

    work = RosaSoftWork()

    work._future = RosaContext(
        batch_size=batch_size,
        num_heads=num_heads,
    ).update(
        query=query, key=key, value=value,
        schmitt_trigger=schmitt_trigger,
        async_op=True,
    )

    work._params = RosaSoftParams(
        scale=scale,
        quant_mode=quant_mode,
        quant_scale=quant_scale,
        schmitt_trigger=schmitt_trigger,
    )

    work._function_apply = RosaSoftFunction.apply
    work._query_key_value = (query, key, value)

    if async_op:
        return work
    return work.wait()


class RosaSoftParams:
    def __init__(self,
        scale: Optional[float],
        quant_mode: str,
        quant_scale: Optional[float],
        schmitt_trigger: float,
    ):
        self.scale = scale
        self.quant_mode = quant_mode
        self.quant_scale = quant_scale
        self.schmitt_trigger = schmitt_trigger

        self.info: Dict[str, Tensor] = {}


class RosaSoftFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: RosaSoftParams):
        x_hard = params.info.pop("x_hard")

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
        params: RosaSoftParams = ctx.saved_params
        
        endpos = params.info.pop("endpos")

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        with torch.enable_grad():
            x_soft = rosa_native_proxy(
                query, key, value,
                endpos=endpos,
                scale=params.scale,
                quant_mode=params.quant_mode,
                quant_scale=params.quant_scale,
            )

        grad_query, grad_key, grad_value = torch.autograd.grad(
            outputs=x_soft,
            inputs=(query, key, value),
            grad_outputs=grad_output,
            retain_graph=False,
            only_inputs=True,
        )

        return grad_query, grad_key, grad_value, None


def rosa_native_proxy(
        query: Tensor, key: Tensor, value: Tensor,
        endpos: Tensor, scale: Optional[float],
        quant_mode: str, quant_scale: Optional[float],
):
    bsz, seq_len, num_q_heads, num_q_bits = query.size()
    bsz, seq_len, num_k_heads, num_k_bits = key.size()
    bsz, seq_len, num_v_heads, num_v_bits = value.size()

    assert num_q_heads == num_k_heads, "num_q_heads must be equal to num_k_heads"
    assert num_q_heads % num_v_heads == 0, "num_q_heads must be divisible by num_v_heads"

    query = query.permute(0, 2, 1, 3)
    key   = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    # q = torch.linspace(0, 1, 10, device=query.device)
    # qq = torch.quantile(query.flatten(), q, dim=0)
    # qk = torch.quantile(key.flatten(), q, dim=0)
    # qv = torch.quantile(value.flatten(), q, dim=0)
    # print(qq.tolist())
    # print(qk.tolist())
    # print(qv.tolist())

    ## quantize query, key, value
    xq, xk, xv = quantize(
        query=query, key=key, value=value,
        quant_mode=quant_mode, quant_scale=quant_scale,
    )
    
    ## repeat value for multi-head attention
    n_rep = num_q_heads // num_v_heads
    if n_rep > 1:
        xv = xv[:, :, None, :, :].expand(bsz, num_v_heads, n_rep, seq_len, num_v_bits)
        xv = xv.reshape(bsz, num_q_heads, seq_len, num_v_bits)
    
    ## pad key to predict next token
    xk = F.pad(xk, (0, 0, 1, -1))

    ## dynamic programming attention
    ss = xq @ xk.transpose(-1, -2)
    ss = ss * (10.0 / num_q_bits)
    ss = torch.sigmoid(ss).tril()

    r = torch.arange(seq_len, dtype=torch.long, device=ss.device)

    ### diagonals to columns
    c = (r.view(-1, 1) + r.view(1, -1) + 1) % seq_len
    ss = ss.gather(-1, c.expand(bsz, num_q_heads, seq_len, seq_len))

    ### dynamic programming
    dp = ss.cumsum(dim=-2)
    ss = dp - (dp * (1 - ss)).cummax(dim=-2).values

    ### columns to diagonals
    c = (r.view(1, -1) - r.view(-1, 1) + seq_len - 1) % seq_len
    ss = ss.gather(-1, c.expand(bsz, num_q_heads, seq_len, seq_len))

    ### apply causal mask
    causal_mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=ss.device).tril()
    causal_mask = causal_mask.expand_as(ss)
    ss[~causal_mask] = -torch.inf

    ### softmax attention
    if scale is None:
        scale = 1.0 / math.sqrt(num_q_bits)
    else:
        scale = float(scale)

    ss = torch.softmax(ss * scale, dim=-1)
    xo = ss @ xv

    return xo.permute(0, 2, 1, 3)

