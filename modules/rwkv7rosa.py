import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

import os
import math
from dataclasses import dataclass

from rosa_cpp import rosa_bits_ops, RosaContext

try:
    from .rwkv7 import RWKV_Tmix_x070, RWKV_CMix_x070, RWKV_Args_x070, RWKV_x070, L2Wrap
except ImportError:
    from rwkv7 import RWKV_Tmix_x070, RWKV_CMix_x070, RWKV_Args_x070, RWKV_x070, L2Wrap



class RWKV_ROSA_x070(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_layer: int,
            layer_id: int,
            qk_bits: int,
            v_bits: int,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id

        assert n_embd % qk_bits == 0
        self.n_head = n_embd // qk_bits

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_q = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.x_v = nn.Parameter(torch.zeros(1, 1, n_embd))

        self.query = nn.Linear(n_embd, self.n_head * qk_bits, bias=False)
        self.key = nn.Linear(n_embd, self.n_head * qk_bits, bias=False)
        self.value = nn.Linear(self.n_head * v_bits, n_embd, bias=False)

        self.emb0 = nn.Parameter(torch.full((1, 1, self.n_head * v_bits), -1.0))
        self.emb1 = nn.Parameter(torch.full((1, 1, self.n_head * v_bits), +1.0))

        self.schmitt_trigger = 1e-3

        self.reset_parameters()
    
    def reset_parameters(self):
        H = self.n_head
        C = self.n_embd

        ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
        ddd = torch.arange(C, dtype=torch.float32).div_(C).view(1, 1, C)

        self.x_q.data.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))

        sqrt_C = C**0.5
        self.query.weight.data.uniform_(-0.5 / sqrt_C, +0.5 / sqrt_C)
        self.key.weight.data.uniform_(-0.05 / sqrt_C, +0.05 / sqrt_C)
        self.value.weight.data.uniform_(-0.5 / sqrt_C, +0.5 / sqrt_C)

        self.emb0.data.fill_(-1.0)
        self.emb1.data.fill_(+1.0)
    
    def forward(self, x: Tensor):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xq = x + xx * self.x_q
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v

        xq: Tensor = self.query(xq)
        xk: Tensor = self.key(xk)
        xv: Tensor = self.value(xv)

        xq = xq.view(B, T, H, -1).transpose(1, 2)
        xk = xk.view(B, T, H, -1).transpose(1, 2)
        xv = xv.view(B, T, H, -1).transpose(1, 2)

        xo = rosa_bits_ops(xq, xk, xv, schmitt_trigger=self.schmitt_trigger)
        xo = xo.transpose(1, 2).reshape(B, T, -1)
        xo = self.emb1 * xo + self.emb0 * (1.0 - xo)
        return xo
    
    @torch.no_grad()
    def inference(self, x: Tensor, state: Dict[str, Tensor]):
        B, T, C = x.size()
        H = self.n_head

        x0: Tensor = state["token_shift_rosa"][self.layer_id]
        s0: RosaContext = state["rosa_state"][self.layer_id]

        xx = torch.cat([x0[:, None, :], x[:, :-1, :]], dim=1) - x
        x0.copy_(x[:, -1, :])

        xq = x + xx * self.x_q
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v

        xq: Tensor = self.query(xq)
        xk: Tensor = self.key(xk)
        xv: Tensor = self.value(xv)

        xq = xq.view(B, T, H, -1).transpose(1, 2)
        xk = xk.view(B, T, H, -1).transpose(1, 2)
        xv = xv.view(B, T, H, -1).transpose(1, 2)

        xo = s0.update(xq, xk, xv, schmitt_trigger=self.schmitt_trigger)
        xo = xo.transpose(1, 2).reshape(B, T, -1)
        xo = self.emb1 * xo + self.emb0 * (1.0 - xo)
        return xo


class Block(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_layer: int,
            layer_id: int,
            qk_bits: int,
            v_bits: int,
            head_size: int = 64,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

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

        self.rosa = RWKV_ROSA_x070(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            qk_bits=qk_bits,
            v_bits=v_bits,
        )
    
    def reset_parameters(self):
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.ln3.reset_parameters()

        if self.layer_id == 0:
            self.ln0.reset_parameters()
        
        self.att.reset_parameters()
        self.ffn.reset_parameters()
        self.rosa.reset_parameters()
        
    def forward(self, x: Tensor, v_first: Tensor):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_rosa = self.rosa(self.ln3(x))
        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn + x_rosa

        x = x + self.ffn(self.ln2(x))
        return x, v_first
    
    @torch.no_grad()
    def inference(self, x: Tensor, v_first: Tensor, state: Dict[str, Tensor]):
        if self.layer_id == 0:
            x = self.ln0(x)
        
        x_rosa = self.rosa.inference(self.ln3(x), state)
        x_attn, v_first = self.att.inference(self.ln1(x), v_first, state)
        x = x + x_attn + x_rosa

        x = x + self.ffn.inference(self.ln2(x), state)
        return x, v_first


@dataclass
class Model_Args:
    vocab_size: int = 65536
    ctx_len: int = 4096
    n_embd: int = 768
    n_layer: int = 12
    head_size: int = 64
    qk_bits: int = 8
    v_bits: int = 8
    weight_decay: float = 0.1
    lr_init: float = 6e-4       # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    lr_final: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.99)
    adam_eps: float = 1e-18

class Model(nn.Module):
    def __init__(self, args: Model_Args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList()
        for layer_id in range(args.n_layer):
            self.blocks.append(Block(
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                layer_id=layer_id,
                qk_bits=args.qk_bits,
                v_bits=args.v_bits,
                head_size=args.head_size,
            ))

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        RWKV_x070.post_init_weight(module=self, n_layer=args.n_layer)
    
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
    
    @torch.no_grad()
    def generate_zero_state(self, batch_size: int = 1, dtype = None, device = None) -> Dict[str, Tensor | List[Dict[str, Tensor]]]:
        n_embd = self.args.n_embd
        n_layer = self.args.n_layer

        head_size = self.args.head_size
        n_head = n_embd // head_size
        
        return {
            "token_shift_att": torch.zeros(n_layer, batch_size, n_embd, dtype=dtype, device=device),
            "token_shift_ffn": torch.zeros(n_layer, batch_size, n_embd, dtype=dtype, device=device),
            "time_mixing_state": torch.zeros(n_layer, batch_size, n_head, head_size, head_size, dtype=dtype, device=device),
            "elapsed_tokens": torch.zeros(batch_size, dtype=torch.int32, device=device),

            "token_shift_rosa": torch.zeros(n_layer, batch_size, n_embd, dtype=dtype, device=device),
            "rosa_state": [RosaContext() for _ in range(n_layer)],
        }
    
    @torch.no_grad()
    def inference(self, idx: Tensor, state: Dict[str, Tensor]) -> Tensor:
        B, T = idx.size()
        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for layer_id, block in enumerate(self.blocks):
            x, v_first = block.inference(x, v_first, state)
        
        state["elapsed_tokens"] += T

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
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "lr": 1.0 * self.args.lr_init},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "lr": 2.0 * self.args.lr_init},
        ]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "lr": 1.0 * self.args.lr_init}]

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

        RWKV_x070.post_init_weight(module=self, n_layer=self.args.n_layer)

if __name__ == "__main__":
    args = Model_Args() # 0.1b
    print(args)

    model = Model(args).cuda()

    for name, p in model.named_parameters():
        print(name, p.size())
    
    model.reset_parameters()
    model.configure_optimizers()

    x = torch.randint(0, args.vocab_size, (2, 512)).cuda()
    # logits = model(x)
    # print(logits.size())

    state = model.generate_zero_state(batch_size=2, dtype=torch.float32, device="cuda")
    logits = model.inference(x, state=state)
    print(logits.size())

    logits = model.inference(x, state=state)
    print(logits.size())
