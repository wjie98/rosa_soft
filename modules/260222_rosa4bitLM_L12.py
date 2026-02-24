import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

import os
from rosa_soft import *

########################################################################################################
# RWKV Tokenizer (slow version)
########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

tokenizer = RWKV_TOKENIZER(os.path.expanduser("~/rwkv_vocab_v20230424.txt"))

def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                # assert abs(torch.sum(probs).item() - top_p) < 1e-6
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()

########################################################################################################
# RWKV-8 ROSA-4bit
########################################################################################################


class rosa_4bit_layer(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.emb = nn.Parameter(torch.ones((1, 1, C)))
    
    def forward(self, xq: Tensor, xk: Tensor, xv: Tensor) -> Tensor:
        B, T, C = xq.size()

        xq = xq.reshape(B, T, -1, 4).transpose(1, 2)
        xk = xk.reshape(B, T, -1, 4).transpose(1, 2)
        xv = xv.reshape(B, T, -1, 4).transpose(1, 2)

        xo = rosa_sufa_ops(xq, xk, xv)

        # for b in range(xq.size(0)):
        #     for g in range(xq.size(1)):
        #         r = torch.arange(4, device=xq.device)
        #         bq = torch.sum((xq[b, g] > 0).long() << r, dim=-1)
        #         bk = torch.sum((xk[b, g] > 0).long() << r, dim=-1)
        #         bv = torch.sum((xv[b, g] > 0).long() << r, dim=-1)
        #         bo = torch.sum((xo[b, g] > 0).long() << r, dim=-1)
        #         print(f'[{b}, {g}] bq={bq.tolist()}')
        #         print(f'[{b}, {g}] bk={bk.tolist()}')
        #         print(f'[{b}, {g}] bv={bv.tolist()}')
        #         print(f'[{b}, {g}] bo={bo.tolist()}')
        #         print()

        xo = xo.transpose(1, 2).reshape(B, T, C)
        return xo * self.emb
    
    @torch.no_grad()
    def inference(self, xq: Tensor, xk: Tensor, xv: Tensor, state: RosaContext) -> Tensor:
        B, T, C = xq.size()

        xq = xq.reshape(B, T, -1, 4).transpose(1, 2)
        xk = xk.reshape(B, T, -1, 4).transpose(1, 2)
        xv = xv.reshape(B, T, -1, 4).transpose(1, 2)

        xo, _ = state.update(xq, xk, xv)

        xo = xo.transpose(1, 2).reshape(B, T, C)
        return xo * self.emb


class RWKV_ROSA_4bit(nn.Module):
    def __init__(s, C, layer_id):
        super().__init__()
        s.layer_id = layer_id
        s.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        s.x_q = nn.Parameter(torch.zeros(1, 1, C))
        s.x_k = nn.Parameter(torch.zeros(1, 1, C))
        s.x_v = nn.Parameter(torch.zeros(1, 1, C))
        s.q = nn.Linear(C, C, bias=False)
        s.k = nn.Linear(C, C, bias=False)
        s.v = nn.Linear(C, C, bias=False)
        s.rosa_qkv = rosa_4bit_layer(C) # !!! matched 1 => e, matched 0 => -e, unmatched => 0 !!!
        s.o = nn.Linear(C, C, bias=False)
    
    def forward(s, x):
        xx = s.time_shift(x) - x
        q = x + xx * s.x_q
        k = x + xx * s.x_k
        v = x + xx * s.x_v
        y = s.rosa_qkv(s.q(q), s.k(k), s.v(v))
        return s.o(y)
    
    @torch.no_grad()
    def inference(s, x, state: Dict[str, Tensor]):
        x0: Tensor = state["token_shift_rosa"][s.layer_id]
        s0: RosaContext = state["rosa_state"][s.layer_id]

        xx = torch.cat([x0[:, None, :], x[:, :-1, :]], dim=1) - x
        x0.copy_(x[:, -1, :])

        q = x + xx * s.x_q
        k = x + xx * s.x_k
        v = x + xx * s.x_v
        y = s.rosa_qkv.inference(s.q(q), s.k(k), s.v(v), s0)
        return s.o(y)

########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x070(nn.Module):
    def __init__(self, n_embd, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.key = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=False)

    def forward(self, x: Tensor):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)
    
    @torch.no_grad()
    def inference(self, x: Tensor, state: Dict[str, Tensor]):
        x0: Tensor = state["token_shift_ffn"][self.layer_id]

        xx = torch.cat([x0[:, None, :], x[:, :-1, :]], dim=1) - x
        x0.copy_(x[:, -1, :])

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, n_embd, layer_id):
        super().__init__()
        self.layer_id = layer_id

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd) # only used in block 0, should be fused with emb
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

        self.rosa = RWKV_ROSA_4bit(n_embd, layer_id=layer_id)
        self.ffn = RWKV_CMix_x070(n_embd, layer_id=layer_id)
        
    def forward(self, x, v_first):
        # print(f'Block {self.layer_id} start')
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.rosa(self.ln3(x))
        x = x + self.ffn(self.ln2(x))
        # print(f'Block {self.layer_id} end')
        return x, v_first
    
    @torch.no_grad()
    def inference(self, x, v_first, state):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.rosa.inference(self.ln3(x), state=state)
        x = x + self.ffn.inference(self.ln2(x), state=state)

        return x, v_first

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, n_embd=768, n_layer=12, vocab_size=65536):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(self.vocab_size, self.n_embd)

        self.blocks = nn.ModuleList([Block(self.n_embd, i) for i in range(self.n_layer)])

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

    def forward(self, idx):

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)

        return x
    
    @torch.no_grad()
    def inference(self, idx, state: Dict[str, Tensor]):

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block.inference(x, v_first, state)

        x = self.ln_out(x)
        x = self.head(x)

        return x
    
    @torch.no_grad()
    def generate_zero_state(self, batch_size: int = 1) -> Dict[str, Tensor]:
        n_embd = self.n_embd
        n_layer = self.n_layer

        for p in self.parameters():
            dtype = p.dtype
            device = p.device
            break

        return {
            "token_shift_ffn": torch.zeros(n_layer, batch_size, n_embd, dtype=dtype, device=device),
            "token_shift_rosa": torch.zeros(n_layer, batch_size, n_embd, dtype=dtype, device=device),
            "rosa_state": [RosaContext() for _ in range(n_layer)],
        }

########################################################################################################
# RWKV Inference
########################################################################################################

if __name__ == "__main__":
    model_path = os.path.expanduser("~/rwkv-rosa4bit-minipile-loss3dot44-20260221-ctx512.pth")
    state_dict = torch.load(model_path, mmap=True)

    model = RWKV().cuda().half().requires_grad_(False)

    for name, p in model.named_parameters():
        print(name, p.size())
    
    model.load_state_dict(state_dict)

    ########################################################################################################

    prompt = "The apple can be"
    input = tokenizer.encode(prompt)
    print(f'\nInput:\n{input}')

    out = model.forward(torch.tensor(input).reshape(1,-1).cuda())
    print(f'\nOutput:\n{out}')

    # logits of the last token => prediction for the next token    
    out = out[0, -1]
    
    probs = F.softmax(out.float(), dim=-1) # compute softmax in float (more accurate)

    print(f'\n{prompt}')

    _, indices = torch.topk(probs, 10) # print top-10 possibilities
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(token, f'[probability {token_prob:.2%}]')

    ########################################################################################################

    print('\n\nNow testing STUPIDLY SLOW (recompute everything including FFN of full context for every step) decoding, as I am too busy to write correct code...\n\n---\n')

    prompt = "When"
    # prompt = "The"
    # prompt = "I"

    print(prompt, end='', flush=True)
    input = tokenizer.encode(prompt)

    input = torch.tensor(input).reshape(1, -1).cuda()
    state = model.generate_zero_state(batch_size=1)

    for i in range(100):
        out = model.inference(input, state)
        out = out[0, -1]
        token_id = sample_logits(out, temperature=1.0, top_p=0.5, top_k=0)
        if token_id == 0:
            break
        try:
            print(tokenizer.decode([token_id]), end='', flush=True)
        except:
            print(repr(tokenizer.decode([token_id])), end='', flush=True)
        input = torch.tensor([token_id]).reshape(1, -1).cuda()
