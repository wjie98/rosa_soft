"""A residual ROSA block with a nonlinear readout and a minimal training loop."""

import argparse

import torch
from torch import nn

from rosa_soft import rosa_bitflip


class RosaBlock(nn.Module):
    def __init__(self, width=32, heads=2, bits=4, value_size=16):
        super().__init__()
        self.heads, self.bits, self.value_size = heads, bits, value_size
        self.q = nn.Linear(width, heads * bits)
        self.k = nn.Linear(width, heads * bits)
        self.v = nn.Linear(width, heads * value_size)
        self.fuse = nn.Sequential(
            nn.Linear(width + heads * value_size, width), nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        shape = (*x.shape[:2], self.heads)
        q = self.q(x).reshape(*shape, self.bits)
        k = self.k(x).reshape(*shape, self.bits)
        v = self.v(x).reshape(*shape, self.value_size)
        y = rosa_bitflip(q, k, v).flatten(-2)
        return x + self.fuse(torch.cat((x, y), dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()
    torch.manual_seed(17)
    dtype = getattr(torch, args.dtype)
    model = nn.Sequential(RosaBlock(), nn.Linear(32, 8)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    x = torch.randn(2, 33, 32, device="cuda")
    target = torch.zeros(2, 33, 8, device="cuda")
    target[:, 5:] = x[:, :-5, :8]

    def forward(x):
        # Keeping AMP inside the compiled callable makes its dtype explicit.
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            return model(x)

    call = torch.compile(forward, fullgraph=True) if args.compile else forward
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (call(x).float() - target).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
