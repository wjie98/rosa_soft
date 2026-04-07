import torch
import torch.nn.functional as F

from torch import Tensor
from typing import *


def quantize(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        quant_mode: str,
        quant_scale: Optional[float] = None,
):
    if quant_scale is None:
        quant_scale = 1.0
    
    assert quant_scale >= 1.0, "quant_scale must be >= 1.0"
    
    if quant_scale != 1.0:
        query = query * quant_scale
        key = key * quant_scale
        value = value * quant_scale
    
    if quant_mode == "ste":
        xq = query
        xk = key
        xv = value
    elif quant_mode == "tanh":
        xq = torch.tanh(query)
        xk = torch.tanh(key)
        xv = torch.tanh(value)
    elif quant_mode == "soft":
        xq = F.softsign(query)
        xk = F.softsign(key)
        xv = F.softsign(value)
    elif quant_mode == "cubic":
        xq = torch.tanh(query.pow(3) + 0.1 * query)
        xk = torch.tanh(key.pow(3) + 0.1 * key)
        xv = torch.tanh(value.pow(3) + 0.1 * value)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}, expected 'ste', 'tanh', 'soft' or 'cubic'")

    xq = (torch.where(xq > 0, 1.0, -1.0) - xq).detach() + xq
    xk = (torch.where(xk > 0, 1.0, -1.0) - xk).detach() + xk
    xv = (torch.where(xv > 0, 1.0, -1.0) - xv).detach() + xv

    return xq, xk, xv
