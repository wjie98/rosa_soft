from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .anchor import AttentionTelemetry, estimate_rosa_anchor_scale

__all__ = [
    "RosaAnchorScaleController",
    "RosaAnchorScaleConfig",
]


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


@dataclass
class RosaAnchorScaleConfig:
    seq_len: int
    qk_bits: int
    window_size: int
    target_top_prob: float = 0.8
    update_interval: int = 100
    adjustment_rate: float = 0.35
    ema_decay: float = 0.9
    min_scale: float = 1.0e-4
    max_scale: float = 1.0e4
    initial_scale: Optional[float] = None


class RosaAnchorScaleController:
    """Lightweight controller for periodic RosaAnchor scale calibration."""

    def __init__(self, config: RosaAnchorScaleConfig) -> None:
        if config.update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        self.config = config
        self.scale = (
            float(config.initial_scale)
            if config.initial_scale is not None
            else estimate_rosa_anchor_scale(
                seq_len=config.seq_len,
                qk_bits=config.qk_bits,
                window_size=config.window_size,
                target_top_prob=config.target_top_prob,
            )
        )
        self.step = 0
        self.top_prob_ema: Optional[float] = None

    def should_probe(self, step: Optional[int] = None) -> bool:
        step = self.step if step is None else int(step)
        return step % self.config.update_interval == 0

    def observe(self, telemetry: AttentionTelemetry) -> float:
        observed = float(telemetry.top_prob.detach().float().cpu())
        if self.top_prob_ema is None:
            self.top_prob_ema = observed
        else:
            decay = min(max(float(self.config.ema_decay), 0.0), 0.999)
            self.top_prob_ema = decay * self.top_prob_ema + (1.0 - decay) * observed

        error = _logit(self.config.target_top_prob) - _logit(self.top_prob_ema)
        factor = math.exp(float(self.config.adjustment_rate) * error)
        self.scale = min(max(self.scale * factor, self.config.min_scale), self.config.max_scale)
        return self.scale

    def advance(self) -> int:
        self.step += 1
        return self.step

    def kwargs(self, step: Optional[int] = None) -> dict:
        return {
            "scale": self.scale,
            "return_telemetry": self.should_probe(step),
        }
