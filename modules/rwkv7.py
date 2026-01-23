import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

import os
import math
from dataclasses import dataclass

from rwkv_cuda.rwkv7 import RWKV7_CLAMPW_CUDA


class RWKV_CMix_x070(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_layer: int,
            layer_id: int,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.key = nn.Linear(self.n_embd, self.n_embd * 4, bias=False)
        self.value = nn.Linear(self.n_embd * 4, self.n_embd, bias=False)

        self.reset_parameters()
    
    def reset_parameters(self):
        ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
        ddd = torch.arange(
            self.n_embd,
            dtype=torch.float32,
        ).div_(self.n_embd).view(1, 1, -1)

        self.x_k.data.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))

        self.key.weight.data.uniform_(
            -0.5 / (self.n_embd ** 0.5),
            +0.5 / (self.n_embd ** 0.5),
        )
        self.value.weight.data.zero_()
    
    def forward(self, x: Tensor):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    

class RWKV_Tmix_x070(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_layer: int,
            layer_id: int,
            head_size: int = 64,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.head_size = head_size

        assert n_embd % head_size == 0
        self.n_head = n_embd // head_size

        H = self.n_head
        N = self.head_size
        C = self.n_embd

        self.x_r = nn.Parameter(torch.zeros(1, 1, C))
        self.x_w = nn.Parameter(torch.zeros(1, 1, C))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.x_v = nn.Parameter(torch.zeros(1, 1, C))
        self.x_a = nn.Parameter(torch.zeros(1, 1, C))
        self.x_g = nn.Parameter(torch.zeros(1, 1, C))

        D_DECAY_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32)) # suggestion
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, C))
        self.w0 = nn.Parameter(torch.zeros(1, 1, C))

        D_AAA_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32)) # suggestion
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.zeros(D_AAA_LORA, C))
        self.a0 = nn.Parameter(torch.zeros(1, 1, C))

        D_MV_LORA = max(32, int(round((1.7 * (C**0.5)) / 32) * 32)) # suggestion
        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.zeros(D_MV_LORA, C))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C))

        # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
        D_GATE_LORA = max(32, int(round((5 * (C**0.5)) / 32) * 32)) # suggestion
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.zeros(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C))
        self.k_a = nn.Parameter(torch.zeros(1, 1, C))
        self.r_k = nn.Parameter(torch.zeros(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

        self.reset_parameters()
    
    def reset_parameters(self):
        H = self.n_head
        N = self.head_size
        C = self.n_embd

        ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
        ddd = torch.arange(C, dtype=torch.float32).div_(C).view(1, 1, C)

        self.x_r.data.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w.data.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a.data.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g.data.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        def ortho_init_(x: Tensor, scale: float):
            if x.dim() == 2:
                gain = max(1.0, math.sqrt(x.size(0) / x.size(1)))
                nn.init.orthogonal_(x, gain=gain * scale)
            elif x.dim() == 3:
                gain = max(1.0, math.sqrt(x.size(1) / x.size(2)))
                for i in range(x.size(0)):
                    nn.init.orthogonal_(x[i], gain=gain * scale)
            else:
                raise ValueError(f"Invalid tensor shape: {x.size()}")
            return x
        
        www = torch.arange(C, dtype=torch.float32).div_(C - 1).pow_(1 + 1 * ratio_0_to_1 ** 0.3)
        www = www.mul_(6).sub_(6).view(1, 1, C)

        zigzag = torch.arange(C) % N
        zigzag = zigzag.to(torch.float32).sub_((N - 1) / 2).div_((N - 1) / 2)
        zigzag = zigzag.mul_(zigzag.abs()).view(1, 1, C)
        
        linear = torch.arange(C, dtype=torch.float32).div_(C - 1).sub_(0.5).view(1, 1, C)

        self.w0.data.copy_(www + 0.5 + zigzag * 2.5) # !!! 0.5 comes from F.softplus !!!
        self.w1.data.zero_()
        ortho_init_(self.w2.data, 0.1)

        self.a0.data.copy_(linear * 0.4 - 0.19 + zigzag * 0.3)
        self.a1.data.zero_()
        ortho_init_(self.a2.data, 0.1)

        self.v0.data.copy_(0.73 - linear * 0.4)
        self.v1.data.zero_()
        ortho_init_(self.v2.data, 0.1)

        self.g1.data.zero_()
        ortho_init_(self.g2.data, 0.1)

        self.k_k.data.copy_(0.71 - linear * 0.1)
        self.k_a.data.fill_(1.02)
        self.r_k.data.fill_(-0.04)

        sqrt_C = C**0.5
        self.receptance.weight.data.uniform_(-0.5 / sqrt_C, +0.5 / sqrt_C)
        self.key.weight.data.uniform_(-0.05 / sqrt_C, +0.05 / sqrt_C)
        self.value.weight.data.uniform_(-0.5 / sqrt_C, +0.5 / sqrt_C)
        self.output.weight.data.zero_()
        self.ln_x.reset_parameters()

    def forward(self, x: Tensor, v_first: Tensor):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RWKV7_CLAMPW_CUDA(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class RWKV_Block_x070(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_layer: int,
            layer_id: int,
            head_size: int = 64,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_Tmix_x070(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            head_size=head_size,
        )

        self.ffn = RWKV_CMix_x070(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
        )
    
    def reset_parameters(self):
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()

        if self.layer_id == 0:
            self.ln0.reset_parameters()
        
        self.att.reset_parameters()
        self.ffn.reset_parameters()
        
    def forward(self, x: Tensor, v_first: Tensor):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss: Tensor, y: Tensor):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        y: Tensor = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.size(0) * y.size(1))
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


@dataclass
class RWKV_Args_x070:
    vocab_size: int
    ctx_len: int
    n_embd: int
    n_layer: int
    head_size: int
    dim_att: int | None = None
    dim_ffn: int | None = None
    weight_decay: float = 0.1
    lr_init: float = 6e-4       # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    lr_final: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.99)
    adam_eps: float = 1e-18


class RWKV_x070(nn.Module):
    def __init__(self, args: RWKV_Args_x070):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList()
        for layer_id in range(args.n_layer):
            self.blocks.append(RWKV_Block_x070(
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                layer_id=layer_id,
                head_size=args.head_size,
            ))

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        self.post_init_weight(module=self, n_layer=args.n_layer)
    
    def forward(self, idx: Tensor):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)
        return x
    
    def compute_loss(self, idx: Tensor, targets: Tensor, ignore_index: int = -100):
        logits: Tensor = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index)
        return L2Wrap.apply(loss, logits)
    
    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for name, p in self.named_parameters():
            if ("att.w0" in name):
                lr_2x.add(name)
            elif (p.squeeze().dim() >= 2) and (args.weight_decay > 0) and (".weight" in name):
                lr_decay.add(name)
            else:
                lr_1x.add(name)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if os.getenv("LOCAL_RANK", "0") == "0":
            print(f"decay {lr_decay}\n")
            print(f"1x {lr_1x}\n")
            print(f"2x {lr_2x}\n")

        param_dict = {n: p for n, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
        return torch.optim.AdamW(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps)
    
    def reset_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, nn.ModuleList):
                for m in module:
                    if not hasattr(m, "reset_parameters"):
                        continue
                    m.reset_parameters()

            if isinstance(module, nn.ModuleDict):
                for m in module.values():
                    if not hasattr(m, "reset_parameters"):
                        continue
                    m.reset_parameters()

            elif hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.post_init_weight(module=self, n_layer=args.n_layer)

    @staticmethod
    def post_init_weight(module: nn.Module, n_layer: int):
        def strs_in_name(strs: List[str], name: str):
            for s in strs:
                if s in name:
                    return True
            return False
        
        def strs_not_in_name(strs: List[str], name: str):
            for s in strs:
                if s not in name:
                    return True
            return False
        
        def name_endswith_strs(name: str, strs: List[str]):
            for s in strs:
                if name.endswith(s):
                    return True
            return False
        
        for name, p in module.named_parameters():
            scale = 1.0
            if strs_in_name(["ln_", ".ln", "time_", "_mask", "pos_emb", ".mask."], name) \
            or strs_not_in_name([".weight"], name) \
            or name_endswith_strs(name, ["_w", "_w1", "_w2", "_bias"]):
                if "ln_x.weight" in name:
                    layer_scale = (1 + int(name.split('.')[1])) / n_layer
                    p.data.fill_(layer_scale ** 0.7)
            elif name == "emb.weight":
                scale = -1e-4
                nn.init.uniform_(p.data, a=scale, b=-scale)
            elif name == "head.weight":
                vocab_size, n_embd = p.size()
                if vocab_size > n_embd:
                    scale = 0.5 * math.sqrt(vocab_size / n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(p.data, gain=scale)
            else:
                assert name.endswith('.weight') # should always be true

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in name:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in name:
                        scale = 0.1

                for kk in [".att.gate."]:
                    if kk in name:
                        scale = 0.1

                if scale == 0:
                    nn.init.zeros_(p)
                elif scale < 0:
                    nn.init.uniform_(p, a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(p, gain=scale)

if __name__ == "__main__":
    args = RWKV_Args_x070(100, 512, 768, 3, head_size=64)
    model = RWKV_x070(args).cuda()
    for name, p in model.named_parameters():
        print(name, p.size())
    
    model.reset_parameters()

    x = torch.randint(0, args.vocab_size, size=(4, 32 * 16), dtype=torch.long).cuda()
    logits = model(x)

    print(x.size(), logits.size())

    optimizer = model.configure_optimizers()
    print(optimizer)
    