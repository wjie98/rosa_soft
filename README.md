# rosa_soft

A PyTorch extension for training and running ROSA, a discrete suffix-based
retrieval operator. It combines an exact CUDA forward with a dense surrogate
backward, and provides a C++ suffix automaton for stateful CPU inference.

- Exact, causal, unlimited-length suffix matching.
- Binary forward values with surrogate gradients for Q, K, and V.
- A single training API for dense and packed sequences, including grouped
  value heads.
- FP16, BF16, and FP32 inputs, with FP32 gradient accumulation.
- PyTorch autograd and `torch.compile` integration.

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
and do not update one instance concurrently. `update_packed` accepts
prepacked int32 symbols of shape `[B, T, H]` or `[N, H]`.

CPU matching is synchronous. GPU inputs are staged through the host; these
interfaces do not provide asynchronous transfer overlap or training gradients.

## Development

```bash
pip install --no-build-isolation ".[test]"
python -m pytest -q
```

Tests compare hard routing with an independent dynamic-programming oracle
and surrogate gradients with an independent PyTorch definition. They cover
causality, latest-match ties, unbounded suffixes, chunked inference, packed
sequences, dtypes, gradient masks, dropout, and compilation.

See [Architecture and Gradient Definition](docs/DESIGN.md) for the equations
and implementation structure.

## Acknowledgements

ROSA was introduced by **Peng Bo (BlinkDL)**. This project builds on his ROSA
work in [RWKV-LM / RWKV-v8](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8).
We thank Peng Bo and the [RWKV-LM project](https://github.com/BlinkDL/RWKV-LM)
for the original design and implementation.
