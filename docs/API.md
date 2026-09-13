# API and Deployment

[English overview](../README.md) | [中文概览](../README.zh-CN.md) |
[Memory](MEMORY.md) | [Methods](METHODS.md) | [Design](DESIGN.md)

## Layout and Matching

| Symbol | Meaning |
| --- | --- |
| B | Batch size |
| T | Tokens per dense sequence |
| N | Total tokens in packed input |
| H, Hv | Q/K heads and value heads; H must be divisible by Hv |
| D, Dv | Bits per Q/K symbol and coordinates per value; `1 <= D <= 32` |

Dense Q/K are `[B,T,H,D]`, V is `[B,T,Hv,Dv]`, and Y is `[B,T,H,Dv]`.
Packed tensors omit B: Q/K are `[N,H,D]`, V is `[N,Hv,Dv]`, Y is `[N,H,Dv]`.
Query head h reads value head `h // (H/Hv)`.

Q/K signs form discrete symbols. For query i, match against K endpoints
j strictly before i, select the longest suffix and then the latest j, and
return `sign(V[j+1])`. Null output is zero, not `V[0]`. All binary signs
use `x > 0` as +1 and everything else as -1. Use finite inputs.

## Dense Soft Training

```text
rosa_soft(q, k, v, cu_seqlens=None, *,
          scale=1.0, dropout_p=0.0, mismatch_scale=3.0) -> y
```

Q/K must have identical shapes; Q/K/V must be CUDA tensors sharing dtype
and device. Supported dtypes are FP16, BF16 and FP32. Noncontiguous inputs
are accepted through internal contiguous copies.

| Parameter | Contract |
| --- | --- |
| `cu_seqlens` | Omit for dense input; required for packed input |
| `scale` | Finite, positive multiplier of surrogate logits |
| `dropout_p` | Post-softmax surrogate drop probability, in `[0, 1 - 2**-24]`; inverse-keep scaling |
| `mismatch_scale` | Finite, positive mismatch penalty in the soft suffix recurrence |

These parameters are static and affect only backward. There is no internal
schedule or estimator switching. Dropout never removes hard matches and does
not make the candidate scan sparse. Set `dropout_p` explicitly; the functional
operator does not inspect a surrounding module's training flag.

Packed offsets are a nondecreasing CUDA int32 vector, starting at zero and
ending at N. Equal consecutive offsets represent empty sequences. Neither
matching nor gradients cross boundaries. For example, lengths 8, 0, 5 use:

```python
import torch
from rosa_soft import rosa_soft

q = torch.randn(13, 4, 8, device="cuda", requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(13, 2, 16, device="cuda", requires_grad=True)
cu = torch.tensor([0, 8, 8, 13], device="cuda", dtype=torch.int32)
y = rosa_soft(q, k, v, cu_seqlens=cu)  # [13, 4, 16]
```

Packed backward transfers offsets to the host and invokes dense backward
separately for each nonempty segment. Work scales with the sum of squared
segment lengths. This is not a single fused variable-length backward kernel.

## Bitflip Training

```text
rosa_bitflip(q, k, v, *, rows=256, tied=None, chunks=1) -> y
```

Inputs must use the dense CUDA layout with matching FP16/BF16/FP32 dtype.
Noncontiguous inputs, strided dY and T=0 are supported. Packed sequences
are not supported. Do not concatenate independent documents along T.

| Parameter | Contract |
| --- | --- |
| `rows=256` | Python integer in `[1,256]`; maximum live query-band rows |
| `chunks=1` | Positive Python integer dividing both H and Hv; equal head-group count |
| `tied=None` | Independent edits; optional `"qk"` or `"qkv"` for joint activation edits |

Smaller `rows` and larger `chunks` reduce scratch memory. They do not
truncate time history or omit potentially nonzero edits. `chunks=6` with
H=Hv=192 processes 32 heads per group; with H=192, Hv=24 it processes 32
Q/K heads and four complete V heads per group. B is not split. Changing
`chunks` between compiled calls can trigger recompilation.

Binding is explicit:

```python
y = rosa_bitflip(q, q, v, tied="qk")
y = rosa_bitflip(q, q, q, tied="qkv")
```

Bound arguments must be the same Tensor object. The default still performs
independent edits if arguments alias. Joint edits are not the sum of separate
Q, K and V edits; sharing projection weights does not imply activation binding.
Joint modes require `T < 2**20` and `B*H <= 65535`. These are representation
limits, not suffix windows. QKV binding also requires equal value layout.

Q/K credit contracts each complete hard output edit with fixed dY and applies
the softsign factor. Independent and QK-bound V gradients follow only the
original hard route. QKV binding includes payload and routing changes in the
same edit. See [Methods](METHODS.md) for what this estimator does and does not
differentiate, and [Memory](MEMORY.md) for the larger joint repair allocation.

## Autograd and Execution

Both training operators support first-order autograd and `torch.compile`.
Higher-order derivatives are unsupported. FP32 gradient accumulation may use
nondeterministic atomic additions; bitflip raises in strict deterministic
mode when its nondeterministic backward path is required.

For mixed-precision compiled models, put autocast inside the compiled callable:

```python
@torch.compile(fullgraph=True)
def forward(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x)
```

Here `model` is the caller's module. Bitflip independent/joint modes and head
chunks are runtime-tested on SM75 and SM86. Native BF16 model/Inductor tests
require SM80 or newer; BF16 operator arithmetic is also tested on SM75.
This is a tested configuration statement, not qualification of every GPU.

Builds target the installed PyTorch ABI (`torch>=2.11,<2.12`). Rebuild the
extension after changing the PyTorch/CUDA environment. Set
`TORCH_CUDA_ARCH_LIST` at build time when deploying to GPUs other than the
build machine. `_C` provides soft/SAM; `_bitflip` is a separate CUDA extension
compiled without fast math. CPU-only builds expose CUDA-required errors for
the training APIs. There is no import-time JIT build.

## QKV 4-bit Models

Independent `[B,T,C]` projections become `[B,T,C/4,4]`. For C=768, H=192
and D=Dv=4. Use independent bitflip, not a tied mode, for separate projections.
From a source checkout:

```python
from functools import partial
from examples.rosa_4bit import Rosa4Bit
from rosa_soft import rosa_bitflip

layer = Rosa4Bit(768, op=partial(rosa_bitflip, chunks=6)).cuda()
# projected_q, projected_k, projected_v: [B,T,768]
y = layer(projected_q, projected_k, projected_v)
```

The adapter keeps `emb` shaped `[1,1,C]` and multiplies it after binary
retrieval. Install it as the model's `rosa_qkv` submodule to preserve the
parameter path. Do not detach `emb`, put it inside V's sign quantization,
or pre-quantize Q/K/V before passing them to the training operator.

The adapter matches the layer layout in
[RWKV-LM's QKV 4-bit inference example](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/260222_rosa4bitLM_L12.py).
It does not establish that checkpoint's training estimator. Full-model loading
and logit parity remain separate integration checks. Current tests cover
amplitude gradients, two checkpointed residual blocks, AMP/compile and the
192-head, 512-token layout. Examples are source files, not installed public APIs.

## CPU Inference

```python
from rosa_soft import rosa_hard, RosaSam

y, ends = rosa_hard(q, k, v, cu_seqlens=None)
sam = RosaSam(num_heads=4, symbol_bits=8)
ends = sam.update(q_chunk, k_chunk)
ends = sam.update(next_q_chunk, next_k_chunk)
sam.reset()
```

`rosa_hard` creates a fresh SAM, accepts dense or packed floating-point input,
and returns binary values and int64 matched K ends. Packed offsets may be
int32 or int64 on CPU or CUDA. Ends are local to each sequence, not absolute
offsets in the flattened packed batch. No match is -1.

`RosaSam.update` accepts `[B,T,H,D]` or `[N,H,D]` floating-point Q/K.
`update_packed` instead accepts prepacked int32 symbols `[B,T,H]` or `[N,H]`;
its name refers to packed bits, not exclusively variable-length sequences.
For the `[N,H,D]` / `[N,H]` layouts, optional offsets separate the streams;
omitting offsets denotes one sequence. Returned ends refer to each stream's
accumulated history across calls, not just the current chunk. The instance
stores Q/K matching state, not V; the caller retains the V history for reads.

Keep sequence count and ordering fixed, even through empty chunks. One SAM
instance must not be updated concurrently. Empty chunks do not advance state.
Use a fresh instance or `reset()` for a new batch of histories.

These are inference/validation APIs without training gradients or a supported
`torch.compile` path. Matching runs synchronously on CPU; GPU inputs undergo
host staging and returned routes are copied to the input device. They do not
provide asynchronous D2H/H2D overlap. `model.eval()` does not select SAM for you.
