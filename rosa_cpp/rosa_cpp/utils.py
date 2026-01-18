import torch
import torch.nn.functional as F

from torch import Tensor
from typing import *


def quantize(x: Tensor, state: Optional[Tensor] = None, schmitt_trigger: float = 0.0) -> Tensor:
    assert x.is_floating_point()
    *C, seq_len, num_bits = x.size()
    if num_bits <= 8:
        dtype = torch.uint8
    elif num_bits <= 16:
        dtype = torch.int16
    elif num_bits <= 32:
        dtype = torch.int32
    else:
        dtype = torch.int64
    
    if schmitt_trigger > 0.0:
        with torch.no_grad():
            index = torch.arange(seq_len, device=x.device)
            index = index.view(-1, 1).expand_as(x).contiguous()
            index[torch.abs(x) < schmitt_trigger] = -1
            index = torch.cummax(index, dim=-2).values + 1
        
        if state is None:
            x = F.pad(x, (0, 0, 1, 0), value=-(schmitt_trigger + 1.0))
        else:
            x = torch.cat([state, x], dim=-2)

        x = torch.gather(x, dim=-2, index=index)
        state = x.narrow_copy(dim=-2, start=-1, length=1)
    else:
        state = None

    r = torch.arange(num_bits, device=x.device)
    x = ((x > 0).to(dtype) << r).sum(dim=-1)
    return x, state


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

