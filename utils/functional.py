import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

import logging

logger = logging.getLogger(__name__)


__all__ = [
    "cosine_curve_with_warmup",
    "update_optimizer_lr_",
]



def update_optimizer_lr_(optimizer: torch.optim.Optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cosine_curve_with_warmup(step: int, total_steps: int, warmup: int, initial: float = 1e-3, final: float = 0.0) -> float:
    assert 0 <= warmup < total_steps

    step = step % total_steps
    if step < warmup: # linear
        progress = step / warmup
    else:
        progress = (step - warmup) / (total_steps - warmup)
        progress = 0.5 * (1 + math.cos(math.pi * progress))

    value = progress * (initial - final) + final
    value = min(value, max(initial, final))
    value = max(value, min(initial, final))
    return value

