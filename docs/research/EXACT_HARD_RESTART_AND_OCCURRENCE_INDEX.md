# Exact Hard Restart And Occurrence Index

## 1. Scope

This note records two production optimizations for fixed-shape CUDA hard
forward:

1. parallel construction of the stable K-code occurrence index;
2. exact batch processing of a frequent candidate-trajectory restart delta.

They are execution optimizations only. Hard suffix length remains unlimited,
the latest route still wins an equal-length tie, and every route not eliminated
by an exact bound remains available. Neither structure is used by the dense
surrogate VJP.

## 2. Original Bottlenecks

The occurrence index partitions K endpoints by the deepest exact suffix code
that fits in eight bits. The list for each code must remain sorted by endpoint,
because hard forward binary-searches its causal prefix and scans routes from
newest to oldest. The original builder assigned one CTA to one `(batch, head)`
series and made that CTA scan the complete sequence twice. Work was linear,
but long sequences and small `BH` exposed too little CTA-level parallelism.

The exact latest-candidate trajectory removes repeated suffix scans while the
newest candidate endpoint advances by one. A `restart` is a row where that
condition stops and a new base certificate is required. Periodic codebooks can
produce many restarts with the same

```text
delta = query_row - latest_candidate_end.
```

Those base certificates describe one aligned Q/K diagonal. Recomputing them
independently discards structure that can be shared exactly.

## 3. Parallel Stable Occurrence Builder

Let `G = min(32, ceil(T / 2048))`. At `T >= 8192`, each series is partitioned
into `G` contiguous endpoint intervals. The alphabet of packed suffix codes is
fixed at 256.

### 3.1 Count

One CTA owns one `(series, chunk)` pair. It counts every suffix code in the
chunk and, when narrow symbols need the null proof, atomically minimizes the
first position of each symbol. Warp `match_any` groups equal codes before the
shared-memory atomic, reducing contention without changing counts.

### 3.2 Prefix

One 256-thread CTA per series sums each code across chunks. A two-level warp
scan computes the global code offsets. The same pass rewrites each
`chunk_offsets[chunk, code]` count into the absolute beginning of that chunk's
range for that code.

### 3.3 Scatter

The original `(series, chunk)` CTAs run again. Each initializes 256 local
cursors from the absolute chunk bases and emits endpoints in increasing local
position order.

The result is stable for three independent reasons:

1. chunks cover contiguous positions and are ordered by position;
2. prefix assigns every earlier chunk a lower disjoint range for the same code;
3. within a chunk, tiles, warps, and lanes are committed in position order.

Therefore the parallel list is identical to the serial list, rather than just
a permutation with the same histogram. That identity preserves upper-bound
search, newest-first traversal, and latest-route ties.

### 3.4 Workspace

The temporary builder workspace is

```text
BH * G * 256 * sizeof(int32)
G <= 32.
```

At the 8K gate it is at most `T/8` int32 entries per series; beyond 64K it is
capped at 8192 entries per series. Its C++ scope ends before trajectory arrays
are allocated. The persistent operation state still contains the existing
`BHT` occurrence list, not a second full occurrence list.

## 4. Exact Heavy-Restart Diagonal

### 4.1 Performance-Only Selection

One CTA per series takes 64 stratified row samples. Each stratum uses a
co-prime, series-dependent offset rather than its first row, avoiding phase
locking on periodic inputs. Sampled restart deltas vote for one mode.

The proposed delta is then counted over every actual restart. It must occur at
least

```text
max(64, ceil(T / 64))
```

times. If it accounts for less than one eighth of all restarts, the selector
also requires at least seven eighths symbol agreement over the complete
diagonal. These thresholds only decide whether building the auxiliary index is
likely worthwhile. They are not semantic filters: rejection leaves the old
exact path unchanged.

### 4.2 Mismatch Prefix

For accepted `delta`, one warp handles each group of 32 rows and records the
last mismatch in that group for

```text
Q[row] == K[row - delta].
```

Rows before `delta` are treated as mismatches, which enforces the causal route
boundary. A one-warp per-series inclusive prefix-max carries the most recent
mismatch across groups. To query row `t`, a warp checks only the current group:

- if the group contains a mismatch at or before `t`, the nearest one gives the
  exact suffix length;
- otherwise the previous group's prefix gives the nearest earlier mismatch.

Thus the aligned route and its exact suffix length are reconstructed without
walking back through the entire match. The route is merged into every row's
winner, not only rows where the latest-candidate trajectory restarted.

### 4.3 Workspace

The additional live state is

```text
heavy_delta:       BH int32
mismatch_prefix:  BH * ceil(T / 32) int32.
```

The prefix costs 0.125 bytes per row. Relative to the two existing trajectory
arrays, which cost 8 bytes per row, this is 1.5625% extra. No `BHT^2` state and
no host synchronization are introduced.

## 5. Why The Result Is Exact

The selector may miss a useful delta. In that case performance falls back, but
the occurrence list and exact candidate scan are unchanged.

The selector may accept an unhelpful delta. Its suffix is still computed from
the complete mismatch history, so it is merely one extra exact candidate. A
zero-length candidate cannot replace the null route, and a positive-length
candidate participates in the normal length-then-latest comparison.

Occurrence routes are visited from newest to oldest. For a route index `r`,
its suffix length cannot exceed `r`. Once

```text
r <= current_best_length,
```

that route cannot have a longer suffix. At equality, its route is no newer
than the current winner because every valid winner route is at least its suffix
length. All later occurrence entries have still smaller routes. Stopping there
therefore preserves both the maximum suffix length and latest-route tie rule.

## 6. Measured Results

Measurements used an idle RTX 3070, PyTorch `2.11.0+cu128`, CUDA 12.8, and
same-process medians.

The compile-time serial-versus-parallel builder A/B used random `D=8,Dv=8`
inputs and kept the same final route path:

| `T` | Heads | Serial builder | Parallel builder | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16K | 1 | 0.475 ms | 0.204 ms | 2.33x |
| 16K | 4 | 0.617 ms | 0.341 ms | 1.81x |
| 16K | 16 | 1.274 ms | 0.968 ms | 1.32x |
| 16K | 32 | 2.226 ms | 1.964 ms | 1.13x |
| 64K | 1 | 1.593 ms | 0.481 ms | 3.31x |
| 64K | 4 | 2.252 ms | 1.096 ms | 2.06x |
| 64K | 16 | 5.848 ms | 4.099 ms | 1.43x |
| 64K | 32 | 9.882 ms | 9.206 ms | 1.07x |

The integrated profile used `B=1,H=4,Hv=2,D=8,Dv=64,FP16,T=65536`:

| Pattern | Prior exact index | Final | Speedup |
| --- | ---: | ---: | ---: |
| random | 2.166 ms | 1.273 ms | 1.70x |
| aligned | 1.778 ms | 0.788 ms | 2.25x |
| periodic64 | 9.045 ms | 1.076 ms | 8.41x |
| periodic4 | 1.255 ms | 0.847 ms | 1.48x |
| all match | 1.242 ms | 0.846 ms | 1.47x |
| all mismatch | 0.877 ms | 0.483 ms | 1.82x |

The final 64K period sweep measured:

| Pattern | random | p16 | p32 | p64 | p128 | p256 | p512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Latency | 1.255 | 0.942 | 1.064 | 1.083 | 0.999 | 1.011 | 1.094 ms |

Raw retained evidence:

- `validation/unlimited_hard_parallel_builder_sm86.json`
- `validation/unlimited_hard_restart_builder_final_sm86.json`
- `validation/unlimited_hard_period_sweep_final_v2_sm86.json`

## 7. Rejected Variants And Remaining Limit

- A lane-0 Misra-Gries scan serialized dependent global loads and made 64K
  random and periodic64 runs about 10 ms. It was removed.
- Sampling the first restart in each range phase-locked with periodic data.
  Stratified co-prime offsets fixed that bias.
- Accelerating only restart rows did not help later rows on the same diagonal.
  The exact route is now merged at every eligible row.
- Selecting the right periodic256 diagonal was insufficient while fallback
  still scanned routes that could not win. The exact route upper bound reduced
  that case from 235.55 ms to 1.15 ms.

A single 64-sample mode does not reliably nominate very low-frequency deltas,
such as some period-1024 or multi-period mixtures. This is a latency limitation
only. A future top-K selector must justify its extra per-row candidate work and
`K * ceil(T/32)` prefix storage; it may not cap or replace the exact fallback.
For contexts far beyond the measured range, the one-warp carry over mismatch
groups may also need an exact hierarchical prefix scan.

## 8. Validation

- 32 focused hard-index production tests pass, including the 8191/8192 builder
  boundary, random `D=1` and `D=4` at 8K, and periodic64.
- Direct CPU parity passes periods 16, 32, 64, 128, 256, and 512 at 8K.
- Direct CPU parity passes `D=1..8` on periodic64 at 8K.
- Compute Sanitizer periodic64 at 8K reports zero memcheck errors, zero
  synccheck errors, and zero racecheck hazards or warnings.
