"""Single-sample hard-forward fitting benchmark for RosaSoft."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft
from rosa_soft import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    rosa_soft_reference,
)
from rosa_soft.testing import inspect_rosa_soft


def make_copy_tokens(
    seq_len: int,
    vocab_size: int,
    motif_min: int,
    motif_max: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    motif_len = int(
        torch.randint(motif_min, motif_max + 1, (1,), generator=generator).item()
    )
    motif = torch.randint(0, vocab_size, (motif_len,), generator=generator)
    tokens = motif.repeat(math.ceil(seq_len / motif_len))[:seq_len].clone()

    if seq_len >= motif_len * 4:
        block_count = max(1, seq_len // max(1, motif_len * 2))
        for _ in range(block_count):
            max_span = min(seq_len, motif_len * 4)
            span = int(
                torch.randint(
                    motif_len,
                    max_span + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            src = int(
                torch.randint(
                    0,
                    max(1, seq_len - span + 1),
                    (1,),
                    generator=generator,
                ).item()
            )
            dst = int(
                torch.randint(
                    0,
                    max(1, seq_len - span + 1),
                    (1,),
                    generator=generator,
                ).item()
            )
            tokens[dst : dst + span] = tokens[src : src + span].clone()
    return tokens.unsqueeze(0)


def loss_and_accuracy(logits: Tensor, tokens: Tensor) -> Tuple[Tensor, float]:
    prediction = logits[:, :-1].float()
    targets = tokens[:, 1:]
    loss = F.cross_entropy(
        prediction.reshape(-1, prediction.size(-1)),
        targets.reshape(-1),
    )
    accuracy = float((prediction.argmax(dim=-1) == targets).float().mean().item())
    return loss, accuracy


class TinyRosaFitLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        qk_bits: int,
        value_heads: int,
        value_bits: int,
        max_suffix_length: int,
        route_temperature: float,
        mismatch_penalty: float,
        operator: str,
    ) -> None:
        super().__init__()
        if num_heads % value_heads != 0:
            raise ValueError("num_heads must be divisible by value_heads")
        self.num_heads = num_heads
        self.qk_bits = qk_bits
        self.value_heads = value_heads
        self.value_bits = value_bits
        self.max_suffix_length = max_suffix_length
        self.route_temperature = route_temperature
        self.mismatch_penalty = mismatch_penalty
        self.operator = operator

        hidden_size = num_heads * qk_bits
        value_size = num_heads * value_bits
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, value_heads * value_bits, bias=False)
        self.output = nn.Linear(value_size, hidden_size, bias=False)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def qkv(self, tokens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        residual = self.embedding(tokens)
        hidden = self.input_norm(residual)
        query = self.query(hidden).view(
            tokens.size(0),
            tokens.size(1),
            self.num_heads,
            self.qk_bits,
        )
        key = self.key(hidden).view(
            tokens.size(0),
            tokens.size(1),
            self.num_heads,
            self.qk_bits,
        )
        value = self.value(hidden).view(
            tokens.size(0),
            tokens.size(1),
            self.value_heads,
            self.value_bits,
        )
        return residual, query, key, value

    def forward(self, tokens: Tensor) -> Tensor:
        residual, query, key, value = self.qkv(tokens)
        soft_operator = (
            rosa_soft.rosa_soft
            if self.operator == "cuda"
            else rosa_soft_reference
        )
        routed = soft_operator(
            query,
            key,
            value,
            max_suffix_length=self.max_suffix_length,
            route_temperature=self.route_temperature,
            mismatch_penalty=self.mismatch_penalty,
        )
        hidden = residual + self.output(routed.flatten(2))
        return self.head(self.output_norm(hidden))

    @torch.no_grad()
    def route_stats(self, tokens: Tensor) -> Dict[str, float]:
        _, query, key, value = self.qkv(tokens)
        generator = torch.Generator(device=query.device).manual_seed(12345)
        _, inspection = inspect_rosa_soft(
            query,
            key,
            value,
            max_suffix_length=self.max_suffix_length,
            route_temperature=self.route_temperature,
            mismatch_penalty=self.mismatch_penalty,
            generator=generator,
        )
        probabilities = inspection.route_probabilities[:, :, 1:]
        actions = inspection.selected_actions[:, :, 1:]
        selected = torch.gather(
            probabilities,
            -1,
            actions.unsqueeze(-1),
        ).squeeze(-1)
        effective = probabilities.square().sum(dim=-1).reciprocal()
        return {
            "hard_top_probability": float(selected.mean().item()),
            "effective_actions": float(effective.mean().item()),
            "nonnull_fraction": float((actions != 0).float().mean().item()),
            "observed_max_suffix_length": float(
                inspection.hard_lengths.max().item()
            ),
        }


@torch.no_grad()
def evaluate(model: TinyRosaFitLM, tokens: Tensor) -> Tuple[float, float]:
    loss, accuracy = loss_and_accuracy(model(tokens), tokens)
    return float(loss.item()), accuracy


def fit(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.operator == "cuda":
        if device.type != "cuda":
            raise ValueError("--operator cuda requires --device cuda")
        if not rosa_soft.HAS_ROSA_SOFT_CUDA:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")
    if not math.isfinite(args.route_temperature) or args.route_temperature <= 0.0:
        raise ValueError("route_temperature must be finite and > 0")
    if not math.isfinite(args.mismatch_penalty) or args.mismatch_penalty <= 0.0:
        raise ValueError("mismatch_penalty must be finite and > 0")

    torch.manual_seed(300_000 + args.seed)
    tokens = make_copy_tokens(
        seq_len=args.seq + 1,
        vocab_size=args.vocab_size,
        motif_min=args.motif_min,
        motif_max=args.motif_max,
        seed=100_000 + args.seed,
    ).to(device)
    model = TinyRosaFitLM(
        vocab_size=args.vocab_size,
        num_heads=args.heads,
        qk_bits=args.qk_bits,
        value_heads=args.value_heads,
        value_bits=args.value_bits,
        max_suffix_length=args.max_suffix_length,
        route_temperature=args.route_temperature,
        mismatch_penalty=args.mismatch_penalty,
        operator=args.operator,
    ).to(device)

    perturb_seed = args.seed if args.perturb_seed < 0 else args.perturb_seed
    torch.manual_seed(400_000 + perturb_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(400_000 + perturb_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial_loss, initial_accuracy = evaluate(model, tokens)
    best_loss = initial_loss
    best_accuracy = initial_accuracy
    best_step = 0
    first_below = {"0.1": -1, "0.01": -1, "0.001": -1}
    started = time.perf_counter()
    print(f"tokens={tokens[0].tolist()}", flush=True)
    print(
        f"operator={args.operator} step=0 loss={initial_loss:.9f} "
        f"acc={initial_accuracy:.4f} route_temperature={args.route_temperature:.6f} "
        f"mismatch_penalty={args.mismatch_penalty:.6f}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy = loss_and_accuracy(model(tokens), tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_accuracy = accuracy
            best_step = step
        for threshold in first_below:
            if first_below[threshold] < 0 and loss_value < float(threshold):
                first_below[threshold] = step
        if step % args.log_every == 0 or step == args.steps:
            print(
                f"operator={args.operator} step={step} loss={loss_value:.9f} "
                f"acc={accuracy:.4f} best={best_loss:.9f}@{best_step}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_loss, final_accuracy = evaluate(model, tokens)
    if final_loss < best_loss:
        best_loss = final_loss
        best_accuracy = final_accuracy
        best_step = args.steps
    stats = model.route_stats(tokens)
    result: Dict[str, object] = {
        "operator": args.operator,
        "device": str(device),
        "seed": args.seed,
        "perturb_seed": perturb_seed,
        "steps": args.steps,
        "tokens": tokens[0].tolist(),
        "max_suffix_length": args.max_suffix_length,
        "route_temperature": args.route_temperature,
        "mismatch_penalty": args.mismatch_penalty,
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "best_loss": best_loss,
        "best_accuracy": best_accuracy,
        "best_step": best_step,
        "first_below": first_below,
        "elapsed_seconds": elapsed,
        "step_ms": elapsed * 1000.0 / args.steps,
        **stats,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=["reference", "cuda"], default="reference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturb-seed", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--motif-min", type=int, default=4)
    parser.add_argument("--motif-max", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--qk-bits", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--max-suffix-length", type=int, default=8)
    parser.add_argument(
        "--route-temperature",
        type=float,
        default=ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    )
    parser.add_argument(
        "--mismatch-penalty",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--json-out", default="")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
