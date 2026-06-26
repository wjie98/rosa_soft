from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rosa_soft import RosaRuntime

try:
    from .rwkv7_export import RWKV_CMix_x070, RWKV_Tmix_x070
except ImportError:
    from rwkv7_export import RWKV_CMix_x070, RWKV_Tmix_x070


class _RosaProjectionWork:
    def __init__(self, work, output: nn.Linear, gain: Tensor, batch: int, tokens: int) -> None:
        self.work = work
        self.output = output
        self.gain = gain
        self.batch = batch
        self.tokens = tokens

    def wait(self) -> Tensor:
        y, _ = self.work.wait() if hasattr(self.work, "wait") else self.work
        y = y.reshape(self.batch, self.tokens, -1) * self.gain
        return self.output(y)


class RWKV7RosaRuntimeMix(nn.Module):
    """Hard ROSA runtime branch for RWKV7 inference.

    This branch computes Q/K/V on GPU, stages packed bits to the CPU runtime,
    and delays the output projection until after the CPU runtime result is back.
    """

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        qk_bits: int = 4,
        value_bits: int = 4,
    ) -> None:
        super().__init__()
        if n_embd % qk_bits != 0:
            raise ValueError("n_embd must be divisible by qk_bits")
        if n_embd % value_bits != 0:
            raise ValueError("n_embd must be divisible by value_bits")
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.qk_bits = qk_bits
        self.value_bits = value_bits
        self.num_q_heads = n_embd // qk_bits
        self.num_value_heads = n_embd // value_bits
        if self.num_q_heads % self.num_value_heads != 0:
            raise ValueError("q-head count must be divisible by value-head count")

        self.x_q = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.x_v = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(self.num_q_heads * value_bits, n_embd, bias=False)
        self.gain = nn.Parameter(torch.ones(1, 1, self.num_q_heads * value_bits))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        c = self.n_embd
        ratio = 1.0 - (self.layer_id / self.n_layer)
        ddd = torch.arange(c, dtype=torch.float32).div_(c).view(1, 1, c)
        self.x_q.data.copy_(1.0 - torch.pow(ddd, 0.9 * ratio))
        self.x_k.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio))
        self.x_v.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio))

        sqrt_c = math.sqrt(c)
        self.query.weight.data.uniform_(-0.5 / sqrt_c, 0.5 / sqrt_c)
        self.key.weight.data.uniform_(-0.05 / sqrt_c, 0.05 / sqrt_c)
        self.value.weight.data.uniform_(-0.5 / sqrt_c, 0.5 / sqrt_c)
        self.output.weight.data.zero_()
        self.gain.data.fill_(1.0)

    @torch.no_grad()
    def start(
        self,
        x: Tensor,
        prev_x: Tensor,
        runtime: RosaRuntime,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> _RosaProjectionWork:
        bsz, tokens, _ = x.shape
        shifted = torch.cat([prev_x[:, None, :], x[:, :-1, :]], dim=1)
        prev_x.copy_(x[:, -1, :])
        delta = shifted - x

        q = self.query(x + delta * self.x_q).reshape(bsz, tokens, self.num_q_heads, self.qk_bits)
        k = self.key(x + delta * self.x_k).reshape(bsz, tokens, self.num_q_heads, self.qk_bits)
        v = self.value(x + delta * self.x_v).reshape(bsz, tokens, self.num_value_heads, self.value_bits)
        work = runtime.update(q, k, v, stream=stream, async_op=stream is not None, return_packed=False)
        return _RosaProjectionWork(work, self.output, self.gain, bsz, tokens)


class RWKV7RosaOverlapBlock(nn.Module):
    """RWKV7 block with a parallel hard-ROSA runtime branch."""

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        head_size: int = 64,
        rosa_qk_bits: int = 4,
        rosa_value_bits: int = 4,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln_rosa = nn.LayerNorm(n_embd)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        self.tmix = RWKV_Tmix_x070(n_embd, n_layer, layer_id, head_size=head_size)
        self.cmix = RWKV_CMix_x070(n_embd, n_layer, layer_id)
        self.rosa = RWKV7RosaRuntimeMix(n_embd, n_layer, layer_id, rosa_qk_bits, rosa_value_bits)

    @torch.no_grad()
    def inference(self, x: Tensor, v_first: Tensor, state: Dict[str, object]) -> Tuple[Tensor, Tensor]:
        if self.layer_id == 0:
            x = self.ln0(x)

        stream = torch.cuda.Stream(device=x.device) if x.is_cuda else None
        rosa_work = self.rosa.start(
            self.ln_rosa(x),
            state["token_shift_rosa"][self.layer_id],
            state["rosa_runtime"][self.layer_id],
            stream=stream,
        )
        tmix_out, v_first = self.tmix.inference(self.ln1(x), v_first, state)
        x = x + tmix_out + rosa_work.wait()
        x = x + self.cmix.inference(self.ln2(x), state)
        return x, v_first


@dataclass
class RWKV7RosaStateSpec:
    n_layer: int
    n_embd: int
    head_size: int = 64
    rosa_qk_bits: int = 4
    rosa_value_bits: int = 4
    backend: str = "compact"


def make_rwkv7_rosa_state(
    spec: RWKV7RosaStateSpec,
    batch_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Dict[str, object]:
    tmix_heads = spec.n_embd // spec.head_size
    rosa_q_heads = spec.n_embd // spec.rosa_qk_bits
    rosa_value_heads = spec.n_embd // spec.rosa_value_bits
    return {
        "token_shift_att": torch.zeros(spec.n_layer, batch_size, spec.n_embd, dtype=dtype, device=device),
        "token_shift_ffn": torch.zeros(spec.n_layer, batch_size, spec.n_embd, dtype=dtype, device=device),
        "token_shift_rosa": torch.zeros(spec.n_layer, batch_size, spec.n_embd, dtype=dtype, device=device),
        "time_mixing_state": torch.zeros(
            spec.n_layer,
            batch_size,
            tmix_heads,
            spec.head_size,
            spec.head_size,
            dtype=torch.float32,
            device=device,
        ),
        "elapsed_tokens": torch.zeros(batch_size, dtype=torch.int32, device=device),
        "rosa_runtime": [
            RosaRuntime(
                rosa_q_heads,
                rosa_value_heads,
                qk_bits=spec.rosa_qk_bits,
                value_bits=spec.rosa_value_bits,
                backend=spec.backend,
            )
            for _ in range(spec.n_layer)
        ],
    }


def close_rwkv7_rosa_state(state: Dict[str, object]) -> None:
    for runtime in state.get("rosa_runtime", []):
        runtime.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = RWKV7RosaStateSpec(n_layer=2, n_embd=128, head_size=64)
    blocks = nn.ModuleList(
        [
            RWKV7RosaOverlapBlock(
                spec.n_embd,
                spec.n_layer,
                layer_id=i,
                head_size=spec.head_size,
                rosa_qk_bits=spec.rosa_qk_bits,
                rosa_value_bits=spec.rosa_value_bits,
            ).to(device)
            for i in range(spec.n_layer)
        ]
    ).eval()
    state = make_rwkv7_rosa_state(spec, batch_size=2, dtype=torch.float32, device=device)
    x = torch.randn(2, 8, spec.n_embd, device=device)
    v_first = torch.empty_like(x)
    with torch.no_grad():
        for block in blocks:
            x, v_first = block.inference(x, v_first, state)
    print(tuple(x.shape))
    close_rwkv7_rosa_state(state)
