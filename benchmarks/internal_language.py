"""Small research components for ROSA internal-symbol experiments.

This module deliberately stays outside the public package.  It provides one
deterministic state transition and separate read/write symbol heads; task
construction, routing, and losses remain in the individual benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class SymbolSequence:
    """Projected symbols and the shared latent state that produced them."""

    latent: Tensor
    query: Tensor
    key: Tensor


class StatefulSymbolizer(nn.Module):
    """Generate read/write symbols from one deterministic slow state.

    ``update_rate`` is a fixed interpolation coefficient, not a schedule or a
    learned gate.  Reset is applied before consuming the current input.  This
    makes phrase-local state explicit and lets causal tests erase all earlier
    state without changing ROSA's external memory.
    """

    def __init__(
        self,
        *,
        input_size: int,
        state_size: int,
        heads: int,
        bits: int,
        update_rate: float = 0.25,
        feedback_size: int = 0,
        stateful: bool = True,
    ) -> None:
        super().__init__()
        if input_size < 1 or state_size < 1 or heads < 1 or bits < 1:
            raise ValueError("symbolizer dimensions must be positive")
        if not 0.0 < update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")
        if feedback_size < 0:
            raise ValueError("feedback_size must be non-negative")

        self.input_size = int(input_size)
        self.state_size = int(state_size)
        self.heads = int(heads)
        self.bits = int(bits)
        self.feedback_size = int(feedback_size)
        self.update_rate = float(update_rate)
        self.stateful = bool(stateful)

        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, state_size, bias=False)
        self.state_projection = nn.Linear(state_size, state_size, bias=False)
        self.feedback_projection = (
            nn.Linear(feedback_size, state_size, bias=False)
            if feedback_size > 0
            else None
        )
        self.latent_norm = nn.LayerNorm(state_size)
        self.query_projection = nn.Linear(
            state_size,
            heads * bits,
            bias=False,
        )
        self.key_projection = nn.Linear(
            state_size,
            heads * bits,
            bias=False,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.zeros(
            batch_size,
            self.state_size,
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        inputs: Tensor,
        state: Tensor,
        *,
        reset: Optional[Tensor] = None,
        feedback: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Consume one position and return state, latent, read Q, and write K."""

        if inputs.ndim != 2 or inputs.size(-1) != self.input_size:
            raise ValueError("inputs must have shape [B, input_size]")
        if state.shape != (inputs.size(0), self.state_size):
            raise ValueError("state must have shape [B, state_size]")
        if reset is not None:
            if reset.shape != (inputs.size(0),):
                raise ValueError("reset must have shape [B]")
            state = torch.where(reset.unsqueeze(-1), torch.zeros_like(state), state)

        drive = self.input_projection(self.input_norm(inputs))
        if self.feedback_projection is not None:
            if feedback is None:
                feedback = inputs.new_zeros(inputs.size(0), self.feedback_size)
            if feedback.shape != (inputs.size(0), self.feedback_size):
                raise ValueError("feedback must have shape [B, feedback_size]")
            drive = drive + self.feedback_projection(feedback)
        elif feedback is not None:
            raise ValueError("feedback was provided but feedback_size is zero")

        if self.stateful:
            proposal = torch.tanh(drive + self.state_projection(state))
            next_state = torch.lerp(state, proposal, self.update_rate)
        else:
            next_state = torch.tanh(drive)
        latent = self.latent_norm(next_state)
        query = self.query_projection(latent).view(
            inputs.size(0),
            self.heads,
            self.bits,
        )
        key = self.key_projection(latent).view(
            inputs.size(0),
            self.heads,
            self.bits,
        )
        return next_state, latent, query, key

    def forward(
        self,
        inputs: Tensor,
        *,
        reset_mask: Optional[Tensor] = None,
        feedback: Optional[Tensor] = None,
    ) -> SymbolSequence:
        if inputs.ndim != 3 or inputs.size(-1) != self.input_size:
            raise ValueError("inputs must have shape [B, T, input_size]")
        batch_size, sequence_length, _ = inputs.shape
        if sequence_length < 1:
            raise ValueError("inputs must contain at least one position")
        if reset_mask is not None and reset_mask.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("reset_mask must have shape [B, T]")
        if feedback is not None and feedback.shape != (
            batch_size,
            sequence_length,
            self.feedback_size,
        ):
            raise ValueError("feedback must have shape [B, T, feedback_size]")

        state = self.initial_state(
            batch_size,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        latent_steps = []
        query_steps = []
        key_steps = []
        for position in range(sequence_length):
            state, latent, query, key = self.step(
                inputs[:, position],
                state,
                reset=(
                    reset_mask[:, position]
                    if reset_mask is not None
                    else None
                ),
                feedback=(
                    feedback[:, position] if feedback is not None else None
                ),
            )
            latent_steps.append(latent)
            query_steps.append(query)
            key_steps.append(key)
        return SymbolSequence(
            latent=torch.stack(latent_steps, dim=1),
            query=torch.stack(query_steps, dim=1),
            key=torch.stack(key_steps, dim=1),
        )
