import torch

from torch import Tensor
from typing import *

from .sam import RosaContextWork


__all__ = [
    "RosaSoftWork",
]

class RosaSoftWork:
    def __init__(self):
        self._future: RosaContextWork
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

        x_hard, endpos = work.wait()

        params.info["x_hard"] = x_hard
        params.info["endpos"] = endpos

        self._future = None
        self._params = None
        self._function_apply = None
        self._query_key_value = None

        work.context.destroy()

        return function_apply(query, key, value, params)
