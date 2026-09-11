import io

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rosa_soft import rosa_bitflip
from tests.oracle import bitflip_vjp, hard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class Reference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        ctx.save_for_backward(q, k, v)
        return hard(*(x.detach().cpu() for x in (q, k, v)))[0].to(v)

    @staticmethod
    def backward(ctx, dy):
        xs = ctx.saved_tensors
        return tuple(g.to(x) for g, x in zip(bitflip_vjp(*xs, dy), xs))


class Model(nn.Module):
    def __init__(self, op, dtype, dv):
        super().__init__()
        self.op, self.dtype, self.dv = op, dtype, dv
        self.trunk = nn.Linear(7, 32)
        self.proj = nn.ModuleList([nn.Linear(32, 12 + 2 * dv) for _ in range(2)])
        self.fuse = nn.ModuleList([
            nn.Sequential(nn.Linear(32 + 2 * dv, 32), nn.SiLU()) for _ in range(2)
        ])
        self.out = nn.Linear(32, 9)

    def forward(self, x):
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            h = self.trunk(x)
            for proj, fuse in zip(self.proj, self.fuse):
                def layer(h, proj=proj, fuse=fuse):
                    z = proj(h)
                    q = z[..., :6].reshape(*z.shape[:2], 2, 3)
                    k = z[..., 6:12].reshape(*z.shape[:2], 2, 3)
                    v = z[..., 12:].reshape(*z.shape[:2], 2, self.dv)
                    y = self.op(q, k, v).flatten(-2)
                    return h + fuse(torch.cat((h, y), -1))
                h = checkpoint(layer, h, use_reentrant=False)
            return self.out(h)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_checkpoint_accumulation_and_resume(dtype, compiled, monkeypatch, dv=5):
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 model requires Ampere")
    torch._dynamo.reset()
    torch.manual_seed(941)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    left = Model(Reference.apply, dtype, dv).cuda()
    right = Model(rosa_bitflip, dtype, dv).cuda()
    right.load_state_dict(left.state_dict())
    opts = [torch.optim.AdamW(m.parameters(), lr=.001) for m in (left, right)]
    scalers = [torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=16) for _ in range(2)]
    x = torch.randn(1, 5, 7, device="cuda")
    target = torch.randint(0, 9, (1, 5), device="cuda")
    call = torch.compile(right, fullgraph=True) if compiled else right
    for step in range(4):
        for model, opt, scaler in zip((left, call), opts, scalers):
            opt.zero_grad(set_to_none=True)
            for _ in range(2):
                y = model(x)
                assert y.dtype == dtype
                loss = nn.functional.cross_entropy(y.float().flatten(0, 1), target.flatten()) / 2
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            scaler.step(opt)
            scaler.update()
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a, b, atol=5e-5 if dtype == torch.float32 else .004, rtol=.004)
        if step == 1:
            data = io.BytesIO()
            torch.save(dict(model=right.state_dict(), opt=opts[1].state_dict(),
                            scaler=scalers[1].state_dict()), data)
            data.seek(0)
            state = torch.load(data, weights_only=True)
            right = Model(rosa_bitflip, dtype, dv).cuda()
            right.load_state_dict(state["model"])
            opts[1] = torch.optim.AdamW(right.parameters(), lr=.001)
            opts[1].load_state_dict(state["opt"])
            scalers[1].load_state_dict(state["scaler"])
            call = torch.compile(right, fullgraph=True) if compiled else right


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_complete_groups(dtype, compiled, monkeypatch):
    test_checkpoint_accumulation_and_resume(dtype, compiled, monkeypatch, dv=32)
