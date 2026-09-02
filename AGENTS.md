# ROSA Development Constraints

These rules apply to the whole repository. Production code is the package and
native extension under `rosa_soft/`; experiments belong in `benchmarks/`,
`docs/research/`, and `validation/`.

## Hard ROSA Semantics

The hard route is exact and unlimited. For each query position `t` and head:

1. Match the suffix of `Q[:t+1]` against suffixes of `K[:t]`.
2. Select the globally longest exact symbol match.
3. Break equal-length ties with the latest key end.
4. Return the signed value immediately after that key end.
5. Return zero when no positive-length match exists.

Query matching happens before the key at the same position is inserted. A
token must never retrieve through its own key. Dense batches, packed segments,
and incremental chunks must preserve the same order.

`max_suffix_length` is only the finite recurrence horizon of the surrogate
backward. It must never cap hard matching or affect hard output. Hard operator
schemas must not acquire a suffix-window argument. Keep a regression witness
where a match longer than `max_suffix_length` beats a newer shorter match.

Forward observes signs only. Continuous logits, surrogate probabilities,
dropout masks, and value magnitudes must not leak into hard output.

## Dense Training Contract

`rosa_soft` trains the discrete hard route with dense credit assignment:

- forward is exact hard ROSA;
- backward visits null and every valid causal non-null route;
- Q, K, and value gradients come from one coherent route distribution.

Do not replace the default VJP with top-k candidates, ANN/LSH retrieval,
sampled negatives, hard-winner neighborhoods, score thresholds, suffix-index
pruning, or early exit. Those mechanisms remove discovery gradients. They may
exist only as separately named research operators.

The retained estimator is fixed:

```text
m = popcount(q xor k) / D
g = exp(-mismatch_scale * m)
S = sum of consecutive suffix products of g, up to max_suffix_length
U(S) = (sqrt(2) + 1) * (sqrt(1 + S) - 1)
```

The null score is `0.5`. Every non-null logit receives the candidate prior
correction `-log(candidate_count)`, then `scale` is applied. Q/K signs use the
softsign VJP. Values receive the same probability-weighted credit as Q/K.

The public controls are exactly:

- `max_suffix_length=32`;
- `scale=1.0`;
- `dropout_p=0.0`;
- `mismatch_scale=3.0`.

`dropout_p` follows PyTorch attention semantics: it is the probability of
dropping a post-softmax route weight, with inverted scaling on retained
weights. It affects backward only. With `dropout_p=0`, no RNG state is drawn.
With dropout enabled, save one seed and reconstruct masks by counter; never
save a quadratic random tensor.

Do not add dynamic schedules, window-derived scales, confidence weights,
winner margins, recency bias, configurable utility functions, value-gradient
modes, or estimator switches without a new explicit design decision and
evidence across the full validation gate.

## CUDA Implementation

Every execution schedule must evaluate the same candidate set and equations.
Allowed optimizations include:

- online softmax statistics;
- FlashAttention-style route tiling and exact recomputation;
- exact finite-window diagonal recurrences and adjoints;
- shared-memory/register caching;
- packed signs, warp reductions, and coalesced FP32 accumulation;
- `needs_input_grad` specialization;
- internal schedule selection with a complete exact fallback.

Do not materialize an `O(BHT^2D)` comparison tensor or an `O(BHT^2)` CUDA
score/VJP workspace. The PyTorch reference may materialize quadratic tensors
because it is an oracle. A kernel may underflow a route numerically, but it may
not structurally omit that route.

Training hard forward remains an exact unlimited CUDA implementation. It is
allowed to be quadratic because this package does not claim to provide the
long-context inference runtime. Do not add an approximate inference index to
the training build graph.

The fixed-length production VJP currently has row-owned, tiled-streaming, and
one gated SM80+ block/Tensor-Core schedule. Their selection is private. Do not
expose plan knobs or restore rejected kernels merely to improve one isolated
shape. Promotion requires reference parity and a representative matrix over
dtype, length, dimensions, gradient masks, dropout, and input patterns.

All score, softmax, and gradient accumulation arithmetic is FP32. Public
gradients return in input dtype. Global FP32 atomics are not bitwise
deterministic; deterministic-algorithm checks must remain explicit.

## Packed Variable Length

`rosa_soft_varlen` and its reference consume packed `[N,H,D]` tensors plus
`cu_seqlens`. Segment boundaries isolate hard routes, suffix products, null
normalization, dropout counters, and gradients. Empty segments are valid.
Local value position zero is never a non-null route target.

Do not infer competition sets from adjacent packed storage or substitute total
`N` for local sequence length. Offset validation must remain outside unsafe
CUDA indexing.

## Compact SAM Reference

`RosaSam` is a synchronous, route-only validation interface. Its maintained
core is `csrc/rosa_sam_core.h`, a plain C++ suffix automaton with only:

- state length, suffix link, latest end, and first edge;
- sparse symbol transitions;
- one persistent query cursor and one key-construction cursor;
- match-before-append update order.

It accepts 1 to 32 symbol bits and returns sequence-local matched key ends. It
does not own values, devices, streams, workers, sequence IDs, paging,
serialization, LCT state, compression policy, or external storage. Those are
inference-framework concerns, not reasons to expand this validation library.

The direct latest-end suffix-link update is intentionally simple. Like the
compact reference algorithm, it can be quadratic on adversarial repetitive
input. Do not hide a complex runtime data structure behind `RosaSam`; add a
separate backend only when inference requirements are explicitly in scope.

SAM tests must derive landmarks independently with the direct dynamic program

```text
L[i,j] = Q[i] == K[j] ? L[i-1,j-1] + 1 : 0
```

and select only `j < i` by longest length then latest `j`. Tests must not use
another SAM, cached production routes, historical documentation, or old
Runtime output as their expected result. Retain exhaustive small alphabets,
random D1..D32, periodic/clone-heavy strings, latest ties, chunking, reset,
grouped value gather, packed segments, and empty segments.

## Package Boundary

Supported public names are declared in `rosa_soft.__all__`. Build variants are:

- `reference`: PyTorch references only;
- `cpu`: references plus `RosaSam`;
- `cuda`: `RosaSam` plus CUDA training operators.

The native build contains `export.cpp`, `rosa_sam.cpp`, `rosa_soft.cpp`, and
the maintained CUDA translation units listed in `setup.py`. Historical
Runtime, paging, compressed-index, bitflip, and rejected-kernel work must stay
outside this graph. `contrib/runtime_legacy/` is an archive, not a dependency.

New estimator families must use a separate module/operator. The tag
`rosa-soft-dense-reference-v1` remains the frozen comparison baseline.

## Validation Gate

Before merging a semantic or kernel change, verify:

- SAM versus independent DP, including exhaustive and adversarial cases;
- bit-exact hard output, null behavior, successor gather, and latest ties;
- hard-output independence from every surrogate control;
- PyTorch/CUDA VJP parity for every nonempty Q/K/value gradient mask;
- zero and fixed nonzero dropout, FP32/FP16/BF16, grouped heads, D32,
  singleton, and non-contiguous input;
- packed segment isolation, empty segments, and invalid offsets;
- `torch.compile(..., backend="aot_eager", fullgraph=True)`, autocast,
  GradScaler, and checkpoint replay;
- multi-seed fitting and route-discovery regressions;
- latency, peak memory, registers, and spills for kernel changes.

The PyTorch reference and independent DP landmark take precedence over old
benchmarks and historical tests. Speed never justifies changing semantics.
