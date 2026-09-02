# rosa-soft

`rosa-soft` trains exact discrete ROSA routing with a dense surrogate VJP.
Its maintained scope is deliberately small:

- exact unlimited hard forward for training;
- dense CUDA surrogate backward;
- a readable PyTorch reference;
- a compact C++ suffix automaton for hard-route validation.

Long-context inference scheduling, paging, compression, and external-memory
state are not part of this package.

## Semantics

For query position `t`, ROSA compares the suffix ending at `Q[t]` with every
causal key suffix ending before `t`. It selects the globally longest exact
match and resolves equal lengths with the latest key end. A match ending at
key position `p` returns the hard value at `p + 1`; no match returns zero.

Hard matching is unlimited. `max_suffix_length` bounds only the surrogate
backward recurrence and can never affect the forward result.

Forward uses signs only. No continuous score, probability, value magnitude,
or dropout mask is visible to the model.

## Installation

Install a matching PyTorch first, then build against that ABI:

```bash
pip install --no-build-isolation .
```

Build controls:

```bash
USE_CUDA=1 pip install --no-build-isolation .
USE_CUDA=0 pip install --no-build-isolation .
ROSA_BUILD_EXTENSION=0 pip install --no-build-isolation .
```

The three resulting capability sets are `cuda`, `cpu`, and `reference`.
Inspect the active build through `rosa_soft.BUILD_CAPABILITIES`.

## Training API

```python
from rosa_soft import rosa_soft

output = rosa_soft(
    query,                    # [B, T, Hq, D], D in 1..32
    key,                      # [B, T, Hq, D]
    value,                    # [B, T, Hv, Dv], Hq % Hv == 0
    max_suffix_length=32,     # surrogate backward only
    scale=1.0,
    dropout_p=0.0,
    mismatch_scale=3.0,
)
```

`rosa_soft_varlen` accepts packed `[N,H,D]` tensors and int32
`cu_seqlens`. Empty segments are valid and no state crosses a segment
boundary.

`dropout_p` has the same meaning as PyTorch attention dropout: it is the
probability of dropping a post-softmax route weight. It applies only to the
backward carrier.

The backward estimator visits every causal candidate. Candidate top-k,
sampling, hard-neighborhood pruning, and suffix-index pruning are not part of
this operator.

## References

The PyTorch implementations are executable semantic references:

```python
from rosa_soft import rosa_soft_reference, rosa_soft_varlen_reference
```

They materialize quadratic state and are intended for small correctness and
gradient tests, not training throughput.

The C++ SAM independently validates unlimited hard routes:

```python
from rosa_soft import RosaSam, rosa_hard_reference

sam = RosaSam(num_heads=4, symbol_bits=8)
matched_key_end = sam.update(query, key)

hard_output, matched_key_end = rosa_hard_reference(query, key, value)
```

`RosaSam` is synchronous, route-only, and supports symbols from 1 to 32 bits.
It has no value storage, async worker, stream ownership, sequence IDs, paging,
or checkpoint state. Sequence count is fixed by the first update; call
`reset()` before changing it. Chunked updates preserve accumulated Q/K state.
Do not update the same instance concurrently.

Packed symbols use `torch.int32`:

```python
matched_key_end = sam.update_packed(packed_query, packed_key)
```

Dense packed shape is `[B,T,H]`. Variable-length packed shape is `[N,H]`
with `cu_seqlens`. `rosa_hard_varlen_reference` also returns exact packed hard
output and sequence-local matched key ends.

## Surrogate

For one Q/K symbol pair, the normalized Hamming mismatch and local gate are

```text
m = popcount(q xor k) / D
g = exp(-mismatch_scale * m)
```

For each route, products of consecutive gates are accumulated up to
`max_suffix_length` into raw suffix evidence `S`. The route score is

```text
U(S) = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
```

The null route score is fixed at `0.5`. Non-null logits receive
`-log(candidate_count)` before the dense softmax. Q/K use a softsign VJP at
their hard signs; Q/K/value accumulation is FP32 inside CUDA.

## CUDA Execution

The production extension contains three backward schedules with identical
candidate support and equations:

- row-owned cache/recompute for short and packed-varlen input;
- exact tiled streaming for long fixed-length input;
- one gated SM80+ block/Tensor-Core path for `W=32,Dv=64`.

The block and streaming implementations are separate translation units and
share only the small primitives in `rosa_soft_vjp_common.cuh`. Execution-plan
selection is private and has no public tuning switch.

Training hard forward uses one exact unlimited CUDA scan. The former
occurrence index was inference-oriented and is archived outside the package.

## Validation

Core checks:

```bash
python -m pytest -q \
  tests/test_sam.py \
  tests/test_soft_reference.py \
  tests/test_soft_cuda.py \
  tests/test_soft_varlen.py \
  tests/test_streaming_vjp.py
```

`tests/test_sam.py` uses an independent dynamic-programming landmark:

```text
L[i,j] = Q[i] == K[j] ? L[i-1,j-1] + 1 : 0
```

It covers exhaustive binary strings, random widths through D=32, clones,
periodic input, latest ties, reset, chunking, grouped values, varlen input,
and empty segments.

## Source Layout

```text
rosa_soft/
  soft.py                       CUDA autograd wrapper
  soft_reference.py             PyTorch semantic reference
  sam.py                        synchronous validation API
  csrc/rosa_sam_core.h          pure C++ suffix automaton
  csrc/rosa_sam.cpp             PyTorch batch binding
  csrc/rosa_soft.cpp            CUDA validation and dispatch
  csrc/cuda/rosa_soft_kernels.cu
  csrc/cuda/rosa_soft_streaming_kernels.cu
  csrc/cuda/rosa_soft_block_diagonal_kernels.cu
```

Historical inference runtime work is preserved under
`contrib/runtime_legacy/` and is excluded from the package, extension build,
and default test suite.
