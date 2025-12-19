import torch

from torch import Tensor
from typing import *


def quantize(x: Tensor) -> Tensor:
    assert x.is_floating_point()
    num_bits = x.size(-1)
    if num_bits <= 8:
        dtype = torch.uint8
    elif num_bits <= 16:
        dtype = torch.int16
    elif num_bits <= 32:
        dtype = torch.int32
    else:
        dtype = torch.int64
    
    r = torch.arange(num_bits, device=x.device)
    x = ((x > 0).to(dtype) << r).sum(dim=-1)
    return x

def dequantize(x: Tensor, v: Tensor | int) -> Tensor:
    assert not x.is_floating_point()
    
    if isinstance(v, Tensor):
        num_bits = v.size(-1)
    else:
        num_bits = int(v)

    r = torch.arange(num_bits, device=x.device)
    x = (x.unsqueeze(-1) >> r) & 1

    if isinstance(v, Tensor):
        return x.type_as(v)
    else:
        return x

