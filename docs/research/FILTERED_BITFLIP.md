# Exact Filtered Bitflip Prototype

> Historical record of the first seven stages. The frozen implementation map,
> retained mechanisms, and current entry point are in
> `docs/research/FILTERED_BITFLIP_V1.md`.

This document records the first seven stages of the unlimited-suffix exact
bitflip investigation.  The implementation is research-only under
`benchmarks/`; it does not change the frozen `rosa_soft` package or CUDA
schemas.

`filtered_bitflip` means that every semantically active Q/K bit still receives
its full one-bit hard counterfactual.  Filtering removes provably irrelevant
work.  It does not sample bits, truncate suffixes, or approximate the hard
route.

## 1. Semantic contract

For query row `t` and non-null route `r`, where `1 <= r <= t`, the local
endpoint compares `Q[t]` with `K[r-1]`.  Exact suffix length follows

```text
L[t,r] = Q[t] == K[r-1] ? 1 + L[t-1,r-1] : 0.
```

The winner has maximum `(length, route)`, so ties select the latest route.
Route zero is the exact null value.  There is no semantic suffix horizon.
`Q[0]` and `K[T-1]` cannot affect any causal shifted route and are excluded
from bitflip coordinates.

The full oracle reruns this hard operator after every active bit flip.  Its
fixed-upstream pseudo-gradient uses the same sign convention as the existing
`exact_bitflip_vjp`:

```text
g_i = -base_symbol_i * <grad_output, output(flip_i) - output(base)>.
```

## 2. Exact influence event

Flipping bit `d` changes a local pair `(p,j)` only when its packed-code XOR is
zero or `1 << d`:

- XOR zero breaks an exact match;
- XOR `1 << d` creates an exact match;
- every other pair is unchanged.

Let `A` be the exact run immediately left of `(p,j)` and `R` the exact run
immediately right.  For every `s=0..R`, the affected cell is

```text
query = p + s
route = j + s + 1
long_length = A + 1 + s
short_length = s.
```

A break changes long to short; a creation changes short to long.  Along the
whole interval, both winner coordinates normalize to constants:

```text
long_z  = (A + 1 - p, j + 1 - p)
short_z = (-p,        j + 1 - p).
```

The implementation checks every edited matrix cell against its expected old
length before applying the event.  Random cases, structured cases, the frozen
full-horizon hard forward, the existing full bitflip oracle, and exhaustive
`D=1,T=4` Q/K states all agree.  The focused suite currently has 262 tests.

## 3. Raw event limit

For independent D-bit codes, one causal Q/K pair contributes an expected
`4D/2^D` changes across all Q-side and K-side flips.  Therefore

```text
E[raw events] ~= (2D / 2^D) T^2.
```

Occurrence buckets improve the constant but not this asymptotic bound.  At
`T=64` the fraction of all counterpart checks retained by exact code buckets
was approximately:

| State | D=1 | D=4 | D=8 |
| --- | ---: | ---: | ---: |
| Independent initialization | 1.000 | 0.124 | 0.0095 |
| Fully collapsed | 1.000 | 1.000 | 1.000 |

The bucket is highly useful for wide early codes and useless for D=1 or a
collapsed codebook.  It is a filter, not the final scaling algorithm.

## 4. Training-clarity trajectory

The benchmark interpolates independent codes toward three hard languages:

- `shift_random`: high-entropy K and `Q[t]=K[t-1]` at full clarity;
- `shift_motif`: a repeated grammar with the same shifted alignment;
- `collapse`: one repeated symbol.

This matters because marginal symbol entropy does not identify long suffixes.
`shift_random` remains high entropy while its winning suffix reaches mean
length 32 at `T=64`.

The D=8 occurrence filter remains selective as healthy languages become
clear.  The retained fraction of all Q/K bit-pair checks evolves as follows:

| Trajectory | Clarity 0 | 0.25 | 0.50 | 0.75 | 1.00 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shifted random | 0.95% | 1.05% | 1.53% | 2.33% | 3.89% |
| Shifted motif | 0.95% | 1.63% | 3.96% | 7.84% | 13.89% |
| Collapse | 0.95% | 6.01% | 23.87% | 50.85% | 100.00% |

Therefore the occurrence index must stay active throughout training; switching
it off once suffixes become long would discard useful filtering.  Its failure
condition is code collapse, not pattern clarity itself.  This is also why
entropy alone cannot select an execution path: a clear shifted-random language
has high marginal entropy, long matches, and a still-selective occurrence
index, while collapse has the same long-match statistic but no selectivity.

The D=8 exact work census at `T=64` was:

| State | Raw events | Winner-admitted rows | Final changed rows | Final affine ranges |
| --- | ---: | ---: | ---: | ---: |
| Independent initialization | 308 | 272 | 272 | 272 |
| Clear shifted random | 1,254 | 32,256 | 1,008 | 1,008 |
| Clear shifted motif | 4,480 | 32,256 | 7,608 | 1,560 |
| Full collapse | 32,256 | 32,256 | 8,448 | 8,448 |

Measured `T=32 -> 64` slopes for final affine ranges were about `2.28`,
`1.02`, `1.04`, and `1.96`, respectively.  Healthy clear languages therefore
have useful final interval structure, even when their changed-row count is
large.  Independent initialization and collapse still expose a quadratic
explicit-range boundary.

The current winner predicate has a separate limitation: at full clarity it
admits every row for all three trajectories, including the two healthy ones.
RLE still compresses and queries the base winner geometry efficiently, and
the final changed routes remain compact for shifted random and motif, but the
present necessary-condition predicate does not expose that compactness early
enough.  A replacement-winner index is needed; occurrence filtering and winner
filtering should not be treated as one mechanism.

## 5. Packed CPU hard-scan baseline

`filtered_bitflip_cpu_scan.cpp` compares two exact hard controls:

- direct backward scan for every candidate;
- one streaming diagonal recurrence over every causal pair.

The standalone benchmark was compiled with GCC 11.4, `-O3 -march=native`,
pinned to one CPU, and run under WSL2 on the currently exposed AVX2 host.

| D=8 state | T | Direct | Diagonal | Direct / diagonal |
| --- | ---: | ---: | ---: | ---: |
| Random | 1,024 | 0.579 ms | 0.405 ms | 1.43x |
| Clear shifted random | 2,048 | 3.881 ms | 1.604 ms | 2.42x |
| Clear motif | 1,024 | 9.003 ms | 0.401 ms | 22.45x |
| Period four | 1,024 | 33.512 ms | 0.487 ms | 68.81x |
| Collapse | 1,024 | 131.189 ms | 0.479 ms | 273.9x |

The diagonal path sustains roughly `1.1..1.35` billion pair comparisons per
second in D=8 cases and has regular contiguous accesses.  It is the better CPU
quadratic oracle.  It does not solve the unlimited-length scaling target.

## 6. Match-length indexes

Four LCE backends were tested behind the same exact event formula:

| Backend | Exact | Logical space | Decision |
| --- | --- | --- | --- |
| Direct packed scan | Yes | `O(T)` with a tiny constant | Keep as early/high-entropy path. |
| Rolling hash | No | `O(T)` | Reject for final decisions; also lost measured timings. |
| Dyadic canonical ranks | Yes | `O(T log T)` | Useful clear-pattern control, reject as 100M-scale storage. |
| SA + LCP + flat RMQ | Yes | `O(T)` | Retain as the main exact-index candidate. |

At `T=64,D=8`, including index build, event discovery took:

| State | Direct | Dyadic | SA/LCP/RMQ |
| --- | ---: | ---: | ---: |
| Independent initialization | 10.61 ms | 11.09 ms | 11.67 ms |
| Clear shifted random | 17.84 ms | 14.55 ms | 15.75 ms |
| Clear shifted motif | 31.32 ms | 21.03 ms | 24.96 ms |
| Collapse | 149.78 ms | 79.95 ms | 126.34 ms |

Across 45 repeated trajectory cases, direct won 39 and dyadic won six; all
six dyadic wins were fully clear states.  The prototype suffix array won
none, but it uses Python prefix doubling and a log-time flat RMQ, so this is
not a production verdict on an induced-sorting C++ implementation.

A temporary one-point bit flip does not require a dynamic string index.  An
edited LCE can be composed from the immutable base LCE, the edited-symbol
comparison, and at most one further base LCE.

## 7. Winner indexes

Base winner geometry is stored as

```text
Z[t] = (winner_length[t] - t, winner_route[t] - t).
```

For a creation event, rows where `event_long_z > Z[t]` may change.  For a
break, only rows where `event_long_z == Z[t]` can lose the base winner.  Both
conditions are exact.

Run-length indexing and a flat min/max segment tree returned identical row
ranges and exact full-bitflip results.  RLE won 42 of 45 repeated cases.  The
segment tree won three isolated timings but usually visited four to seven
times as many nodes and uses much more memory.  Keep RLE; retain the segment
tree only as an ablation.

Winner filtering alone is insufficient.  At full clarity, admitted rows were
on average `10.25x` the rows whose final route actually changed.  A break can
invalidate the base score while leaving the same route as the winner.  The
next exact structure must find the replacement winner in compressed form,
rather than recomputing every admitted row.

## 8. Decisions

Keep for the next prototype:

1. packed `uint8` codes and exact causal semantics;
2. the create/break interval algebra as the semantic oracle;
3. contiguous code-occurrence lists;
4. direct LCE for short contexts;
5. an exact `O(T)` suffix index candidate for long contexts;
6. normalized-winner RLE;
7. final affine-route ranges as the VJP interface.

Do not promote:

1. explicit raw-event or affected-row materialization;
2. rolling hash as an exact decision;
3. persistent dyadic `O(T log T)` ranks at very long context;
4. the winner segment tree;
5. a full candidate-length matrix;
6. an entropy-only dispatch rule.

The unresolved problem is now narrower: discover final changed affine ranges,
or aggregate their VJP equivalently, without enumerating raw Hamming-0/1 pairs
or all base-winner break rows.  Clear shifted and motif trajectories show that
such a compressed answer often exists.  Early random and collapsed states show
that a universal explicit-range algorithm still needs additional structural
batching, such as suffix-node or periodic-run certificates.

The follow-up exact prototypes and their hybrid design are documented in
`docs/research/FILTERED_BITFLIP_REPLACEMENT.md`.  They solve replacement-winner
batching for measured long-overlap cases, but do not claim a universal
`O(T log T)` bound for adversarial unstructured events.

## 9. Reproduction

```bash
python -m pytest -q \
  tests/test_filtered_bitflip.py \
  tests/test_filtered_bitflip_indexes.py \
  tests/test_filtered_bitflip_winner.py \
  tests/test_filtered_bitflip_profile.py

python benchmarks/filtered_bitflip_profile.py \
  --sequence-lengths 64 --bit-widths 1 4 8 \
  --trajectories shift_random shift_motif collapse \
  --clarities 0 0.25 0.5 0.75 1 --repeats 3 \
  --json-out validation/filtered_bitflip_profile.json

g++ -std=c++17 -O3 -march=native -DNDEBUG \
  -Wall -Wextra -Wpedantic \
  benchmarks/csrc/filtered_bitflip_cpu_scan.cpp \
  -o /tmp/filtered_bitflip_cpu_scan
```

Raw records are in `validation/filtered_bitflip_profile.json`,
`validation/filtered_bitflip_scaling.json`, and
`validation/filtered_bitflip_cpu_scan.json`.
