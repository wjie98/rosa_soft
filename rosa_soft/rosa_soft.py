import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext, RosaWork
from .utils import QuantizeFunction, repeat_kv, unfold_qk, decay_qk, gather_x


__all__ = [
    "RosaSoftWork",
    "rosa_soft_ops",
]


class RosaSoftWork:
    def __init__(self):
        self._future: RosaWork
        self._params: Any
        self._function_apply: Callable[[Tensor, Tensor, Tensor, Any], Tensor]
        self._query_key_value: Tuple[Tensor, Tensor, Tensor]

    def wait(self):
        if self._future is None:
            raise RuntimeError("wait() called twice")
        
        work = self._future
        params = self._params
        function_apply = self._function_apply
        query, key, value = self._query_key_value

        x_hard, info = work.wait()
        params.info["x_hard"] = x_hard
        params.info.update(info)

        self._future = None
        self._params = None
        self._function_apply = None
        self._query_key_value = None

        return function_apply(query, key, value, params)


def rosa_soft_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    quant_mode: str = "soft",
    quant_scale: Optional[float] = None,
    schmitt_trigger: float = 0.0,
    async_op: bool = False,
) -> Union[Tensor, RosaSoftWork]:
    """ROSA Soft Dynamic Programming operator (Baseline).

    Exact softening of suffix matching via recurrent gated accumulation.
    Forward uses discrete ROSA (hard SAM), backward uses soft DP gradients
    via Straight-Through Estimator (STE).

    Complexity: O(T^2) time, O(T^2) space.

    Args:
        query: (B, H, T, D) Query logits.
        key: (B, H, T, D) Key logits.
        value: (B, H, T, D_v) Value logits.
        scale: Attention scale factor. Defaults to 1/D.
        quant_mode: Quantization mode ("tanh" or "soft").
        quant_scale: Quantization scale factor.
        schmitt_trigger: Threshold for noise filtering.
        async_op: Return work object for async execution.

    Returns:
        Tensor if async_op=False, RosaSoftWork if async_op=True.
    """

    work = RosaSoftWork()
    work._future = RosaContext().update(
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
        self._ctx: Optional[RosaContext] = None


class RosaSoftFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: RosaSoftParams):
        if "x_hard" in params.info:
            x_hard = params.info.pop("x_hard")
        else:
            x_hard, info = RosaContext().update(
                query=query, key=key, value=value,
                schmitt_trigger=params.schmitt_trigger,
            )
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
        params: RosaSoftParams = ctx.saved_params

        length = params.info.pop("length")
        endpos = params.info.pop("endpos")

        with torch.enable_grad():
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

            x_soft = rosa_native_proxy(
                query, key, value,
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
        scale: Optional[float],
        quant_mode: str, quant_scale: Optional[float],
):
    bsz, num_heads, seq_len, num_q_bits = query.size()
    bsz, num_kv_heads, seq_len, num_k_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    num_qk_bits = num_q_bits
    assert num_q_bits == num_k_bits, "query and key must have the same number of bits"

    # q = torch.linspace(0, 1, 10, device=query.device)
    # qq = torch.quantile(query.flatten(), q, dim=0)
    # qk = torch.quantile(key.flatten(), q, dim=0)
    # qv = torch.quantile(value.flatten(), q, dim=0)
    # print(qq.tolist())
    # print(qk.tolist())
    # print(qv.tolist())

    ## quantize query, key, value

    if quant_scale is not None:
        assert quant_scale >= 1.0, "quant_scale must be >= 1.0"
        query = query * quant_scale
        key = key * quant_scale
        value = value * quant_scale
    else:
        quant_scale = 1.0
        
    if quant_mode == "tanh":
        xq = torch.tanh(query)
        xk = torch.tanh(key)
        xv = torch.tanh(value)
    elif quant_mode == "soft":
        xq = F.softsign(query)
        xk = F.softsign(key)
        xv = F.softsign(value)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}, expected 'tanh' or 'soft'")
    
    ## repeat key and value for multi-head attention
    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    ## dynamic programming attention
    ss = xq @ xk.transpose(-1, -2)
    ss = ss * quant_scale - (num_qk_bits - 1) * (quant_scale - 1)
    ss = torch.sigmoid(ss).tril(-1)
    
    ss = F.pad(ss, (1, -1), value=0.0) # predict next token
    r = torch.arange(seq_len, dtype=torch.long, device=ss.device)

    ### diagonals to columns
    c = (r.view(-1, 1) + r.view(1, -1) + 1) % seq_len
    ss = ss.gather(-1, c.expand(bsz, num_heads, seq_len, seq_len))

    ### dynamic programming
    dp = ss.cumsum(dim=-2)
    ss = dp - (dp * (1 - ss)).cummax(dim=-2).values

    ### columns to diagonals
    c = (r.view(1, -1) - r.view(-1, 1) + seq_len - 1) % seq_len
    ss = ss.gather(-1, c.expand(bsz, num_heads, seq_len, seq_len))

    ### apply causal mask
    causal_mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=ss.device).tril()
    causal_mask = causal_mask.expand_as(ss)
    ss[~causal_mask] = -torch.inf

    ### softmax attention
    if scale is None:
        scale = 1.0 / num_qk_bits
    else:
        scale = float(scale)

    ss = torch.softmax(ss * scale, dim=-1)
    xo = ss @ xv

    return xo

    
