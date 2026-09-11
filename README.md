# rosa_soft

A PyTorch extension for training and running ROSA, a discrete suffix-based
retrieval operator. It provides an exact CUDA forward with a choice of dense
soft or independent-bit backward estimators, and a C++ suffix automaton for
stateful CPU inference.

- Exact, causal, unlimited-length suffix matching.
- Binary forward values with surrogate gradients for Q, K, and V.
- Dense and packed soft training, and dense bitflip training, including
  grouped value heads.
- FP16, BF16, and FP32 inputs, with FP32 gradient accumulation.
- PyTorch autograd and `torch.compile` integration.

`rosa_soft` remains the default dense/packed training operator.
`rosa_bitflip` is an explicit alternative for dense sequences; the library
never switches estimators automatically.

## Installation

Requires Python 3.10+, PyTorch 2.11, and a C++17 compiler. CUDA training also
requires a CUDA toolkit compatible with the installed PyTorch build.

Install PyTorch first, then build the extension against it:

```bash
pip install --no-build-isolation .
```

CUDA is detected automatically. Set `USE_CUDA=1` to require a CUDA build,
or `USE_CUDA=0` for CPU-only inference and validation. Set `CUDA_HOME` when
the toolkit is not on the default search path.

## Quick Start

```python
import torch
from rosa_soft import rosa_soft

q = torch.randn(1, 2048, 4, 8, device="cuda",
                dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(1, 2048, 2, 64, device="cuda",
                dtype=torch.float16, requires_grad=True)

y = rosa_soft(q, k, v)  # [1, 2048, 4, 64]
y.float().square().mean().backward()
```

## Matching Semantics

Each Q/K vector represents a binary symbol: positive elements become `+1`,
and zero or negative elements become `-1`. At query position `i`, ROSA
selects the longest exact match between the Q suffix ending at `i` and a
historical K suffix ending at `j < i`. Ties select the latest `j`.

The output is binary `V[j + 1]`. If no symbol matches, the output is zero.
Matching has no suffix window or candidate limit. The CUDA DP and CPU suffix
automaton implement the same rule.

## Training API

```python
y = rosa_soft(
    q, k, v,
    cu_seqlens=None,
    scale=1.0,
    dropout_p=0.0,
    mismatch_scale=3.0,
)
```

| Argument | Description |
| --- | --- |
| `q`, `k` | Same-shaped CUDA tensors: `[B, T, H, D]` or packed `[N, H, D]`; `1 <= D <= 32`. |
| `v` | `[B, T, Hv, Dv]` or `[N, Hv, Dv]`; `H % Hv == 0`. Same device and dtype as Q/K. |
| `cu_seqlens` | Required for packed input: a nondecreasing CUDA int32 vector starting at zero and ending at `N`. Omit for dense input. |
| `scale` | Positive multiplier of the surrogate logits. |
| `dropout_p` | Probability of dropping a post-softmax surrogate weight, with inverse-keep scaling. Default: zero. |
| `mismatch_scale` | Positive penalty for symbol mismatch in the surrogate recurrence. |

Output shape is `[B, T, H, Dv]` or `[N, H, Dv]`. Empty packed segments
are supported; neither matches nor gradients cross sequence boundaries.

The forward is always hard and binary. Continuous scores and dropout exist
only in backward. The gradient is a dense surrogate over all causal
candidates, not the derivative of the discrete routing decision. Parameters
are static; there is no internal training schedule. Higher-order gradients
are not supported.

CUDA training performs quadratic candidate work. Packed backward runs each
nonempty segment separately and synchronizes offsets to the host; it is not
a single fused variable-length kernel. Use the CPU automaton for stateful
long-context inference.

## Bitflip Training

```python
from rosa_soft import rosa_bitflip

y = rosa_bitflip(q, k, v, rows=256)
```

Q/K are dense `[B,T,H,D]`, with `1 <= D <= 32`; V is `[B,T,Hv,Dv]`,
with `H % Hv == 0`. Inputs must be finite CUDA tensors with matching
FP16/BF16/FP32 dtype. Noncontiguous inputs and empty T are supported.
Packed sequences are not supported: do not concatenate independent
documents into one dense sequence.

The hard output follows the same unlimited matching rule as `rosa_soft`.
For Q/K, backward evaluates every independent activation-bit output edit,
contracts its output difference with fixed upstream dY, and applies the
softsign factor. V gradients follow the hard route only, unlike the dense
V carrier used by `rosa_soft`. This is not an unbiased derivative of an
arbitrary nonlinear task loss or a simultaneous shared-parameter edit.

`rows` is an integer in [1,256] limiting the live work band. It controls
workspace, not suffix length or which edits contribute. Time is quadratic
in sequence length at fixed widths; working space is linear at fixed rows,
but has a larger constant than the soft implementation. Highly repetitive
symbols can be substantially slower than random symbols. Neither estimator
is universally faster or guarantees better training.

Gradients accumulate in FP32 with nondeterministic atomic summation.
Strict deterministic mode raises; higher derivatives are unsupported.
The bitflip extension builds separately without fast math. Its current
optimized kernels are validated on SM75; native BF16 model/Inductor tests
require SM80 or newer. BF16 operator arithmetic is tested on SM75 as well.

See [the residual-block training example](examples/train_bitflip.py).
For mixed-precision compiled models, put autocast inside the compiled
callable so the dtype is explicit in the graph:

```python
@torch.compile(fullgraph=True)
def forward(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x)
```

## CPU Inference

For a complete sequence, `rosa_hard` returns both values and matched K ends:

```python
from rosa_soft import rosa_hard

y, matched_key_end = rosa_hard(q, k, v)
```

It accepts dense or packed inputs, using the same optional `cu_seqlens`
argument. No-match ends are `-1`; packed ends are local to each sequence.

For incremental routing, retain a `RosaSam` instance:

```python
from rosa_soft import RosaSam

sam = RosaSam(num_heads=4, symbol_bits=8)
ends = sam.update(q_chunk, k_chunk)
# Later updates continue the same sequence histories.
ends = sam.update(next_q_chunk, next_k_chunk)
sam.reset()
```

`RosaSam` stores Q/K matching state, not V. Ends refer to the accumulated
sequence history. Keep the sequence count and ordering fixed between updates,
including empty chunks, and do not update one instance concurrently. Empty
chunks return empty results without advancing history. `update_packed` accepts
prepacked int32 symbols of shape `[B, T, H]` or `[N, H]`.

CPU matching is synchronous. GPU inputs are staged through the host; these
interfaces do not provide asynchronous transfer overlap or training gradients.

## Development

```bash
pip install --no-build-isolation ".[test]"
python -m pytest -q
```

Tests compare hard routing with an independent dynamic-programming oracle
and soft gradients with an independent PyTorch definition. Bitflip tests
enumerate independent hard output edits, including cancellation-sensitive
values, and exercise checkpointed residual-block training. Tests cover
causality, latest-match ties, unbounded suffixes, chunked inference, packed
sequences, dtypes, gradient masks, dropout, and compilation.

See [Architecture and Gradient Definition](docs/DESIGN.md) for the equations
and implementation structure.

## Acknowledgements

ROSA was introduced by **Peng Bo (BlinkDL)**. This project builds on his ROSA
work in [RWKV-LM / RWKV-v8](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8).
We thank Peng Bo and the [RWKV-LM project](https://github.com/BlinkDL/RWKV-LM)
for the original design and implementation.
