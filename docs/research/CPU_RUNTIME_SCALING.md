# Exact CPU Runtime Scaling

Status: production runtime optimization and research audit, 2026-08-22.

## 1. Scope and invariant

This work separates two execution environments:

- fixed-length training uses the exact CUDA hard forward and dense CUDA VJP;
- stateful deployment, offline evaluation, prefill, and extrapolated long
  contexts use `RosaRuntime` on CPU.

Both return the globally longest causal suffix, with the latest key end on a
tie. Neither receives the surrogate `max_suffix_length`. No candidate cap,
sampled hard value, or finite hard window was promoted.

The CPU route is necessary because a dense or quadratic CUDA scan does not
scale to the intended 1M-to-100M deployment regime. The current CPU automaton
is online and exact, but general high-entropy histories still consume linear
memory; the measurements below do not claim that arbitrary 100M histories are
solved.

## 2. Baseline diagnosis

`AutomatonComplexityStats` was added before changing the algorithms. It counts
transition probes, query and extension suffix-link steps, physical latest-end
writes, clones, copied edges, dynamic-path operations, and compressed-run work.
`benchmarks/rosa_runtime.py` checks output checksums across chunk sizes while
reporting these counters and logical bytes.

The frozen baseline at T=8192, one 8-bit head, showed two independent costs:

| Pattern | Time | Dominant work |
| --- | ---: | --- |
| random | 5.408 ms | 2.28M linked transition probes |
| all-match | 57.669 ms | 33.57M latest-end ancestor writes |
| periodic64 | 2.398 ms | 537,979 latest-end writes |
| skewed | 5.071 ms | 2.00M transition probes plus longer suffix chains |

Raw evidence: `validation/runtime_complexity_baseline.json`.

## 3. Retained exact mechanisms

### 3.1 Lazy single-symbol run

While the complete key history is one repeated symbol, the runtime stores only
the symbol, run length, and current query run length. A matching query symbol
routes to the last prior key end; a mismatch resets the query run. This is the
exact suffix language of a unary history.

If the key changes, the unary SAM is materialized once in O(T): state `i`
represents the suffix of length `i`, links to `i-1`, and all states receive the
same latest end. The compressed query run maps directly to state `i`. There is
no approximate transition at the materialization boundary.

At 8K all-match this reduces 8193 states to one and eliminates all 33.57M
latest-end writes. At 1M the automaton still has one state; the remaining
1,048,576 bytes are payload history.

### 3.2 Adaptive latest-end path assignment

Appending a key makes its terminal state and every suffix-link ancestor end at
the new position. Directly writing that path is quadratic on periodic input.
After observing a path of at least 64 states and cumulative writes of at least
`16 * state_count`, the runtime builds an exact link-cut tree:

- ordinary SAM suffix links become represented-tree links;
- clone insertion reparents the cloned child exactly;
- append performs one root-path assignment;
- query performs one point lookup.

The `16x` gate matters. The earlier `8x` gate incorrectly promoted skewed
32K input, raising time to 12.28 ms and logical memory to 2.88 MB. The retained
gate leaves that case on direct writes at 4.40 ms and 1.81 MB, while periodic64
still promotes and stays near 4.0 ms at 32K.

### 3.3 Compact transition lookup

Three representations coexist behind one exact lookup:

1. low-degree states keep the linked edge list;
2. at 16 edges, a state receives a compact power-of-two open-address index;
3. at 65 edges, its 256 slots become a direct byte-symbol table.

The root has a lazy 256-entry direct table after 16 symbols. The state stores
an encoded index ID in its existing `first_edge` field, so unindexed states do
not grow. Clones copy their outgoing edges and build an independent index.

Edges use separate symbol, destination, and successor arrays. Their logical
width is 9 bytes rather than the padded 12-byte AoS width. This improved 32K
random and skewed latency by roughly 13% and 9%, in addition to saving 3 bytes
per edge.

### 3.4 Packed asynchronous staging

CUDA-input runtime calls still snapshot Q/K/payload before returning. For an
explicit stream, native matching now writes into preallocated pinned outputs,
removing a pageable output plus pinning copy. This improved the smallest
asynchronous microbenchmark, but asynchronous throughput alone remains worse
than blocking for 256/1024-token chunks because events, allocation, and Python
submission dominate. It is useful only when the caller overlaps independent
GPU work; it is not the default performance recommendation.

## 4. Retained performance

One-head T=8192 medians after all retained CPU changes:

| Pattern | Baseline | Final | Speedup |
| --- | ---: | ---: | ---: |
| random | 5.408 ms | 1.078 ms | 5.02x |
| all-match | 57.669 ms | 0.397 ms | 145.3x |
| periodic64 | 2.398 ms | 1.140 ms | 2.10x |
| skewed | 5.071 ms | 1.033 ms | 4.91x |

At 1,048,576 tokens, one 8-bit head:

| Pattern | Time | Logical bytes | States |
| --- | ---: | ---: | ---: |
| random | 0.626 s | 46.97 MB | 1,145,741 |
| all-match | 9.86 ms | 1.05 MB | 1 |
| periodic64 | 0.105 s | 48.24 MB | 1,048,582 |
| skewed | 0.512 s | 46.29 MB | 1,298,206 |

Chunk sizes 65,536 and 1,048,576 produced identical checksums. Raw evidence is
in `validation/runtime_final_phase5_8k.json`,
`validation/runtime_long_phase5_1m.json`, and
`validation/runtime_final_phase5_1m_chunk_parity.json`. Five-repeat timing for
the final direct-transition layout is in
`validation/runtime_direct_transition_phase5_1m.json`.

## 5. Large-symbol state compression prototype

`benchmarks/runtime_periodic_state.py` treats one repeated primitive motif as
a macro symbol without changing token-level matching. It stores one active
suffix length per motif phase. For phase `e`, the latest finite-history end is

```text
e + period * floor((history_length - 1 - e) / period).
```

The prototype matched the exact diagonal DP across periods 1, 2, 3, 8, and 16
and random queries. A period-64 history needs about 576 logical state bytes
instead of O(T) SAM state; at 65K the current SAM used about 3.15 MB excluding
payload.

It was not promoted. Automatic motif discovery needs a probation state, and a
later non-periodic key requires O(T) SAM materialization plus exact query-state
reconstruction. That state-machine complexity is not justified by natural
stream evidence yet. The unary case is retained because its detection and
state mapping are trivial and exact.

## 6. Training fusion experiment

`benchmarks/training_hard_fusion.py` asks whether the finite-W surrogate scan
can also produce exact hard output. Candidates matching all W symbols and able
to extend are continued exactly; all other candidate lengths are already
certified.

At T=2048:

- random D8/W4 had no continuation candidates;
- random D4/W4 had 42 continuation candidates out of 2.10M;
- skewed D8/W32 had 1,292 candidates and only 0.064% of direct comparison
  work, but a capped winner still made 10 route errors;
- periodic64/W32 had only 1.47% continuation candidates, but their long runs
  retained 87% to 91% of direct comparison work;
- all-match retained nearly all fallback work.

The exact one-pass softmax VJP identity was also validated:

```text
gradient = A - E[utility] * B
A = sum_j p_j utility_j d(score_j)/dx
B = sum_j p_j d(score_j)/dx
```

It matches autograd, but requires a second full Q/K accumulator. Moving row
softmax statistics into forward does not by itself remove the backward utility
pass, and exact hard forward is already a small fraction of training cost.
Neither bounded hard nor fused-state VJP was promoted.

## 7. Validation

- 64K random, periodic64, all-match, and all-mismatch outputs matched the exact
  CUDA hard operator element by element on GPU0.
- Runtime output matched the independent diagonal oracle through dynamic-tree
  activation and unary materialization.
- Monolithic and chunked 8K, 64K, and 1M runs had identical checksums.
- The periodic macro state passed 20 period/seed cells.
- The dual-accumulator identity passed nine shape cells against autograd.

Reproduction entry points:

```bash
python -m benchmarks.rosa_runtime ...
python -m benchmarks.runtime_periodic_state ...
python -m benchmarks.runtime_async_pipeline ...
python -m benchmarks.training_hard_fusion ...
python -m pytest tests/test_runtime.py \
  tests/test_runtime_periodic_state.py \
  tests/test_training_hard_fusion.py
```

## 8. Remaining limits

General random input still costs about 45 logical bytes per token per head,
plus one payload byte per token per payload head. Extrapolating that unchanged
layout to 100M tokens is multiple gigabytes per head. The next useful research
target is an exact compressed-text automaton or externally paged state, not a
hard suffix window. Periodic macro states are evidence that sublinear state is
possible for structured languages, but they are not yet a general solution.
The operation contract, candidate-index audit, and initial page-locality screen
are continued in `EXACT_COMPRESSED_RUNTIME.md`.
