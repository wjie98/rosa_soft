# Exact Tiled-Streaming VJP

## Status and Contract

The tiled-streaming VJP is the production fixed-length CUDA schedule for
`T >= 4096`. It changes only execution order. Hard forward, every causal
candidate, suffix evidence, square-root score transform, null candidate,
candidate-count correction, dropout counter, softmax VJP, and softsign
Jacobians are unchanged.

The implementation is in
`rosa_soft/csrc/cuda/rosa_soft_streaming_kernels.cu`. It does not materialize a
`[B,H,T,T]` score, probability, random, or adjoint tensor. Short fixed-length
inputs retain the calibrated row kernel, and packed variable-length inputs
retain their separate kernel.

## Tile Ownership

One CTA owns one `(batch, head, query tile)`. One warp owns one query row and
its 32 lanes own 32 adjacent route positions. Production has one layout:

| Symbol width | Query rows | Route columns | Suffix chunk | Value chunk | Direct suffix capacity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `D=1..32` | 32 | 32 | 32 | 32 | 4 |

The older 16-row template remains available only in the direct benchmark
wrapper as a negative control. Shared-memory lifetime reuse made the 32-row
layout legal and faster overall at `D=32`, so production has no symbol-width
dispatch ladder.

`W` is not truncated to one suffix chunk. For `W > 32`, the same tile advances
in 32-offset chunks while each lane carries the exact prefix product, raw
score, derivative prefix, and remaining suffix tail across chunks.

Packed Q/K words are staged once per tile. Local mismatch gates are cached by
`(query-key diagonal, query coordinate)`, which shares one gate across route
endpoints without evaluating unused cells in a bounding rectangle. Value
signs use a padded transposed `[feature, route + 1]` layout to avoid shared
memory bank conflicts.

## First Route Sweep

Each row starts with the exact null item and walks every causal non-null route.
A lane reconstructs raw suffix evidence in production order and applies:

```text
route_logit = scale * (sqrt(2) + 1) * (sqrt(1 + raw_score) - 1)
              - log(candidate_count)
```

The warp maintains an online triple `(maximum, normalizer,
utility_numerator)`. Merge operations rescale both partial normalizers and
utility numerators to the larger maximum. The resulting row state is exactly
the softmax maximum, denominator, and expected dropped route utility. Only
these three scalars per query row survive the first sweep.

## Second Route Sweep

The CTA recomputes the same complete score tiles and immediately consumes
them. It reconstructs each probability from the saved online statistics and
recreates post-softmax dropout from the unchanged counter-based RNG.

The value tile then performs both required reductions:

1. dot signed route values with `grad_output` for Q/K route utility;
2. sum probability-weighted `grad_output` across query rows before one value
   atomic per `(route, feature)`.

For Q/K, the exact route-score VJP is:

```text
d_route = scale * probability
          * (dropout_scale * utility - expected_utility)
d_raw = d_route * (sqrt(2) + 1) / (2 * sqrt(1 + raw_score))
```

The suffix adjoint advances in the same 32-offset chunks. Every route-suffix
term first produces the exact scalar credit for one local match gate. Most
chunks sum those scalars into a shared `(diagonal, query_coordinate)` gate
adjoint; Q and K then contract each unique gate adjoint with the opposite sign
word once. This is valid because all terms reaching one local gate have the
same Q/K Jacobian. It removes duplicate bit contractions without merging or
dropping routes.

Clearing and atomically filling the aggregate costs more than direct
contraction for very small chunks. The kernel therefore uses a fixed direct
microtile when `suffix_count == 1`, or when `suffix_count <= 4` and
`suffix_count * D <= 16`. Partial aggregate chunks contract only their
reachable Q/K words and diagonal band; a full 32-suffix chunk keeps the fully
unrolled contraction. No candidate or low-probability route is omitted. For
`W <= 32`, the second score sweep's gate tile remains resident and is reused
by the suffix adjoint.

Q, K, and value adjoints accumulate in FP32. A final kernel applies each
input's original softsign Jacobian once and transposes the private head-major
K accumulator back to `[B,T,H,D]`.

## Memory and Resources

Shared storage is independent of `T`, `D`, and the number of suffix chunks.
The production 32-row layout uses 10,367 32-bit words, or `40.50 KiB`. The
route-probability buffer is dead before Q/K suffix contraction and is reused
as either the direct-credit microtile or the aggregated local-gate adjoint.
No dedicated K-credit tile remains.

On `sm_75`, the FP32/FP16/BF16 32-row main kernels use 64 registers per thread,
one barrier, zero stack frame, and zero register spill. The benchmark-only
16-row control uses 96 registers and zero spill. Operator-owned global state
remains linear in the inputs and output gradients; quadratic scratch is zero.

## Production Dispatch

`csrc/rosa_soft.cpp` uses this schedule only for fixed-length `T >= 4096`:

```text
if T < 4096: calibrated row-owned VJP
else: 32-row tiled-streaming VJP
```

The threshold is an SM75-calibrated performance boundary, not model
semantics. The smallest tested workload at `T=4096` remained faster than the
row kernel. No model-facing parameter, startup autotuner, candidate heuristic,
or environment switch is added.

## Validation

Focused tests cover all seven nonempty gradient masks, dropout on/off,
FP32/FP16/BF16, grouped value heads, `D=32`, value dimensions crossing a tile,
random and all-match symbols, singleton and partial tiles, and
`W=1/31/32/33/65/128`. Production-threshold tests exercise the 32-row layout
through `D=32` at `T=4096`; the 16-row control remains covered separately.
The largest recorded VJP error against the PyTorch oracle is `9.918e-5`, below
the `3e-4` promotion gate. The optimized-vs-pre-aggregation maximum is
`4.77e-7` in the 4K-16K matrix. The complete suite reports `1204 passed,
2 skipped`. Compute Sanitizer reports zero memcheck errors, zero synccheck
errors, and zero racecheck hazards or warnings for direct microtiles,
aggregate chunks, dropout, `D=32`, and suffix boundaries.

Idle RTX 2080 Ti, FP32, `B=1,H=4,Hv=2,D=8,Dv=64,W=32`, QKV:

| `T` | Frozen row | First streaming | Current | Current / first | Speedup vs row | Fixed SUFA | Current / SUFA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `100.82 ms` | `35.44 ms` | `17.56 ms` | `0.496` | `5.74x` | `17.14 ms` | `1.025` |
| 8192 | `391.72 ms` | `134.30 ms` | `67.17 ms` | `0.500` | `5.83x` | `65.54 ms` | `1.025` |
| 16384 | `1553.83 ms` | `528.01 ms` | `265.46 ms` | `0.503` | `5.85x` | `254.30 ms` | `1.044` |

The corresponding production training-step peak increments were
`7.63/15.25/30.50 MiB`; historical fixed SUFA used
`143.14/286.28/572.56 MiB`. Gate-space aggregation removes the former roughly
2x latency gap while retaining the richer exact dense estimator and linear
workspace. The remaining measured gap to fixed SUFA is about `2.5%..4.4%`.

## Rejected Variants

- Four-row CTAs did not expose enough cross-row reuse.
- Multiple rows per warp increased register pressure and slowed the kernel.
- A rectangular gate cache evaluated many cells that no suffix consumed.
- Two-thread K reductions added shuffle overhead.
- Keeping Q accumulators across all route tiles reduced occupancy.
- The former 24-row compromise and 16-row high-dimensional fallback were
  superseded by the spill-free 32-row layout through `D=32`.
- An exact rolling diagonal score recurrence passed parity but serialized each
  diagonal onto only 63 active threads. It slowed QKV by `15%..32%` for
  `W <= 32` and value-only by `66%..82%`, so the parallel suffix loop remains.
- Carrying a complete value-gradient numerator through the first route sweep
  won only `2%..5%` at `Dv=128`, was neutral at `Dv=32/64`, and regressed
  `Dv=8` by `6%..9%`; the second-sweep value reduction remains.
- Three compile-time gradient-mask kernel classes improved Q/K-only by
  `2%..4%` but changed full QKV by at most `1%` and grew the extension by
  about `81%`; the single runtime mask stays.
- Precomputing the row candidate correction produced no measurable gain, and
  a shared mismatch-gate LUT moved paired medians by only `0.4%..1.5%` with
  sign changes across shapes. Both were removed.
