# rosa_soft

`rosa_soft` is a small PyTorch extension for ROSA's discrete suffix router.
It has one training operator and one exact CPU inference implementation.

```python
from rosa_soft import RosaSam, rosa_hard, rosa_soft
```

## Semantics

Q, K, and V are quantized by sign (`x > 0` is `+1`; otherwise `-1`). At query
position `i`, ROSA searches all historical K suffixes ending at `j < i` and
selects the longest suffix equal to the Q suffix ending at `i`. Equal lengths
select the latest `j`. A match ending at `j` returns binary `V[j + 1]`; no
match returns zero.

The search is unlimited. There is no suffix window, candidate pruning, top-k,
or approximate hard route.

## Training

```python
y = rosa_soft(
    q,                       # [B,T,H,D]
    k,                       # [B,T,H,D]
    v,                       # [B,T,Hv,Dv], H % Hv == 0
    scale=1.0,
    dropout_p=0.0,
    mismatch_scale=3.0,
)
```

The CUDA forward is the exact hard operation. The backward is a dense
surrogate over every causal candidate:

```text
m[i,a] = mean_d (1 - q[i,d] k[a-1,d]) / 2
g[i,a] = exp(-mismatch_scale * m[i,a])
S[i,a] = g[i,a] * (1 + S[i-1,a-1])
U(S)    = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
z[i,a] = scale * U(S[i,a]) - log(i)    for 1 <= a <= i
z[i,0] = scale * 0.5
p[i,:] = softmax(z[i,:])
```

The sign operation uses the softsign derivative
`1 / (1 + abs(x))^2` only in backward. `dropout_p` is the probability of
dropping a post-softmax route and uses inverse-keep scaling, matching PyTorch
attention terminology. Forward never sees a soft value or dropout mask.

Packed input uses `[N,H,D]` and CUDA int32 cumulative offsets:

```python
y = rosa_soft(q, k, v, cu_seqlens)
```

The public function is the same for dense and packed layouts. Packed backward
reuses the dense kernel per segment, so its work is `O(sum(length**2))` rather
than scanning across sequence boundaries.

## Inference And Validation

`RosaSam` is a stateful synchronous CPU suffix automaton. It returns the local
matched K end and can retain history across calls:

```python
sam = RosaSam(num_heads=8, symbol_bits=8)
matched_key_end = sam.update(q, k)
sam.reset()
```

`rosa_hard(q, k, v, cu_seqlens=None)` creates a temporary SAM, gathers binary
successor V, and returns `(output, matched_key_end)`. Q/K may originate on a
GPU, but matching is staged through CPU and is therefore intended for exact
inference or validation, not training.

## Install

Install PyTorch first, then build against that exact ABI:

```bash
pip install --no-build-isolation .
```

Set `USE_CUDA=0` for a SAM-only CPU build or `USE_CUDA=1` to require CUDA.
The default is automatic. The CUDA implementation supports FP16, BF16, and
FP32 with Q/K symbol width `1..32`.

The pre-cleanup research tree is preserved at tag
`rosa-soft-research-archive-v1` (commit `582fe45`). Frozen dense estimator
semantics remain available at tag `rosa-soft-dense-unbounded-v1`.

See [docs/DESIGN.md](docs/DESIGN.md) for implementation boundaries.
