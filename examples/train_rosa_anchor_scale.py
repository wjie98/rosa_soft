from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rosa_soft import RosaAnchorScaleConfig, RosaAnchorScaleController


def train_next_token_epoch(model, dataloader, optimizer, *, window_size: int, qk_bits: int, device):
    """Minimal next-token loop with periodic RosaAnchor scale calibration.

    The model is expected to accept:
        model(input_ids, rosa_scale=..., return_telemetry=...)

    When return_telemetry=True it should return (logits, telemetry), where
    telemetry is the AttentionTelemetry object returned by rosa_anchor_ops.
    """

    controller = RosaAnchorScaleController(
        RosaAnchorScaleConfig(
            seq_len=model.context_length,
            qk_bits=qk_bits,
            window_size=window_size,
            target_top_prob=0.8,
            update_interval=100,
        )
    )
    model.train()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        targets = batch.get("labels", input_ids).to(device=device, non_blocking=True)
        probe = controller.should_probe()

        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids, rosa_scale=controller.scale, return_telemetry=probe)
        if probe:
            logits, telemetry = result
        else:
            logits, telemetry = result, None

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()

        if telemetry is not None:
            controller.observe(telemetry)
        controller.advance()

        yield {
            "loss": float(loss.detach().cpu()),
            "rosa_scale": float(controller.scale),
            "top_prob_ema": controller.top_prob_ema,
        }
