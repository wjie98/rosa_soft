# Exact Replacement-Winner Structures

> Chronological follow-up and proof log. The canonical frozen-v1 architecture
> and mechanism decisions are summarized in
> `docs/research/FILTERED_BITFLIP_V1.md`.

This document records the suffix-node, periodic-run, and compressed
replacement-certificate investigation built on the exact unlimited-suffix
filtered-bitflip semantics.  All implementations remain CPU-only research
prototypes under `benchmarks/`; the frozen `rosa_soft` operator is unchanged.

## 1. Problem reduction

For query row `t` and route `r`, define the diagonal offset

```text
o = r - t.
```

On one maximal equal-code run `[a,b)` of that diagonal, the candidate is

```text
length(t) = t - a + 1
route(t)  = t + o
priority  = (length(t) - t, route(t) - t) = (1 - a, o).
```

Thus every positive candidate matrix cell belongs to one interval carrying a
constant normalized lexicographic priority.  A bit flip makes only two exact
overlay operations:

- creation: add the event's long-priority interval;
- break: remove the old long-priority interval and add the shorter interval
  from the next row, where its length becomes positive.

For one flip `f`, the new winner is exactly

```text
W_f(t) = max((base intervals at t) - forbidden_f(t), added_f(t)).
```

No soft score, candidate truncation, sampling, or approximate hash decision is
used.  This formulation isolates replacement-winner work from event discovery.

## 2. Compressed envelope certificate

`ReplacementCertificateTree` puts every base candidate interval into the
canonical nodes of a row segment tree.  Each node stores its covering
candidates in descending normalized priority and the maximum possible priority
in its subtree.

For one temporary bit flip, break and add intervals are also decomposed into
canonical nodes.  During recursive traversal, a whole row interval is certified
without visiting its rows when all of the following hold:

1. the current candidate covers the whole interval;
2. it dominates the maximum base priority in every descendant;
3. it dominates every added interval that intersects the subtree;
4. its base line is not forbidden in a proper subrange.

Otherwise the traversal descends.  The result is an exact sequence of affine
winner segments.  The prototype materializes rows only to compare with the
existing oracles; a future VJP can consume the segments directly.

This removes event-cell expansion effectively when intervals are long.  At
`T=64,D=8`:

| State | Event cells | Certificate tree nodes | Cells / node |
| --- | ---: | ---: | ---: |
| Independent | 308 | 16,598 | 0.02 |
| Clear shifted random | 32,502 | 23,184 | 1.40 |
| Clear shifted motif | 102,144 | 27,344 | 3.74 |
| Collapse | 698,880 | 37,600 | 18.59 |

The independent case is the important counterexample: starting a tree walk for
many isolated one-cell edits costs more than expanding those cells.  The
certificate tree is a long-overlap backend, not a universal replacement.

Its other limit is base construction.  If `R` is the number of maximal
matching diagonal runs, the current tree stores `O(R log T)` canonical
postings.  `R` is linear for collapse and clear motifs but can be quadratic for
random low-bit codes.

The Python proof implementation also materializes temporary forbidden-id sets
for every segment-tree subtree.  This makes correctness auditing simple but is
not the intended memory layout.  A production traversal must keep rollbackable
path state or persistent top-certificate cursors instead of copying those sets.

## 3. Suffix-node replacement

Reverse Q and K.  Query row `t` becomes one suffix of reversed Q, and route `r`
becomes one suffix of reversed K.  Their hard candidate length is their LCP,
bounded automatically by the key-prefix sentinel.  Every positive LCP interval
of the generalized suffix array is one suffix node.

For an edited row, the best unchanged candidate is found exactly by walking the
query leaf's suffix-node ancestors from deepest to shallowest.  At each node,
the query asks for the latest route `<= t` that is not touched by the current
flip.  The first successful node gives maximum length, and the latest route in
that node gives the exact tie break.  Changed candidates are then compared using
their exact event-provided new lengths.

This avoids a full candidate scan and gives a compact fallback in random,
low-bit states.  However the current prototype still expands affected event
rows; suffix nodes solve replacement selection, not batch output by themselves.

For convenient repeated tests, the prototype also precomputes every query
leaf's ancestor list and every node's explicit route list.  Either can be
quadratic on repetitive inputs.  The linear-size production interpretation is
the compact tree topology with parent links plus a succinct range-predecessor
index; the Python containers are not a 100M-context storage proposal.

Flat route postings also have a bad repetitive case.  In collapse, suffix-node
postings are quadratic because every deeper node contains a long route prefix.
That motivates periodic postings.

## 4. Periodic runs

### Route postings

Every sorted route posting list is partitioned into exact arithmetic
progressions.  The excluded routes for one row are represented the same way.
If a candidate progression is aligned with an excluded progression, the query
jumps below the entire excluded block instead of probing its routes one by one.

At `T=64` collapse, suffix-node storage contains 2,016 flat route postings but
only 63 arithmetic runs.  Across all bit flips, periodic predecessor queries
reduce route probes from 12,229,392 to 761,376.  The route output remains
identical.

### Event families

An event is represented by five integer fields:

```text
(start, stop, long_normalized_length, short_normalized_length, route_offset).
```

Consecutive events with one constant five-dimensional delta form an exact
periodic family.  Coverage and intersection with a segment-tree node reduce to
linear inequalities over the family index.  Maximum added priority and old-line
membership are then computed without expanding family members.

At `T=64,D=8`:

| State | Raw events | Event families | Canonical entries / family |
| --- | ---: | ---: | ---: |
| Independent | 308 | 308 | 1.00 |
| Clear shifted random | 1,254 | 1,254 | 5.02 |
| Clear shifted motif | 4,480 | 1,136 | 20.68 |
| Collapse | 32,256 | 1,024 | 160.50 |

The symbolic representation is valuable, but the Python periodic solver scans
all families at each visited tree node and is slower than canonical overlays.
Keep the representation; reject this particular query implementation.  A C++
version needs families indexed by start/stop and phase so a node sees only
families that cover or intersect it.

## 5. Structural hybrid

`hybrid_filtered_bitflip` chooses a backend independently for every flip.  It
does not use trajectory labels, entropy, training step, or a learned dispatch:

```text
event_cells >= 2 * canonical_overlay_entries
    -> compressed envelope certificate
otherwise
    -> suffix-node row replacement
```

Periodic suffix postings are selected when their run count is less than half
the flat posting count.  Every branch is exact, so dispatch changes only cost.

Across the `T=64,D=8` clarity trajectories, the overlay compression ratio stays
near one through clarity `0.75`, then rises to `4.25..5.16` when long structure
is fully established.  The gate therefore keeps the occurrence and suffix
filters active while patterns are still forming and moves only genuinely
overlapping flips to the interval backend.

Representative single-run Python timings in milliseconds were:

| State | Certificate | Suffix flat | Suffix periodic | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| Independent | 152.6 | 31.8 | 31.7 | 33.6 |
| Clear shifted random | 273.9 | 438.0 | 463.7 | 235.4 |
| Clear shifted motif | 300.8 | 530.9 | 610.5 | 344.3 |
| Collapse | 780.7 | 1,705.0 | 1,306.8 | 782.2 |

These timings compare Python prototypes, not production constants.  The more
stable result is structural: the hybrid selected the fastest implementation in
eight of nine `T=32` bit-width/trajectory cases, while all five exact backends
returned identical routes and lengths.

## 6. Scaling observations

No single materialization has acceptable scaling on every language:

- D=1 independent candidate lines at `T=64/128/256` were
  `525/2,069/8,239`, approximately quadratic.
- The corresponding suffix-node route postings were `372/881/2,075`, much
  slower growth in this range.
- Collapse candidate lines were `63/127/255`, linear.
- Collapse flat suffix postings were `2,016/8,128/32,640`, quadratic, while
  periodic route runs were `63/127/255`, linear.

This is the concrete reason the final design must compose the structures.

## 7. Production candidate

The next implementation should use the following ownership boundaries:

1. Build an exact generalized suffix array/tree over reversed packed Q/K with
   induced sorting and O(T) storage.
2. Store irregular node routes in a succinct range-predecessor structure and
   repetitive routes as arithmetic or periodic runs.
3. Discover local create/break events through the existing occurrence filter
   and exact LCE index.
4. Group events by suffix node and periodic phase before row expansion.
5. Estimate canonical overlay compression from interval endpoints.
6. Use direct suffix-node replacement for sparse edits and an output-sensitive
   interval envelope for long overlapping edits.
7. Store only a small top-certificate at envelope nodes; fall back to the
   suffix-node predecessor when temporary deletions exhaust it.
8. Emit affine changed-route ranges directly into the VJP accumulator.

This removes both problematic global materializations: all candidate lines in
random low-bit states and all flat suffix postings in repetitive states.

The current work does **not** prove a universal `O(T log T)` bound.  An
adversarial low-bit language can still contain quadratically many unrelated
Hamming-0/1 influence events that do not form periodic families.  The exact
worst-case question remains open; the new result is an output-sensitive hybrid
with verified useful compression on the measured training trajectories.

## 8. Verification and data

The new focused tests cover random D=1/2/4/8 states, structured shift/motif and
collapse states, exact bit-gradient equality, and exhaustive `D=1,T=4` Q/K
states for the certificate, suffix-node, periodic, and hybrid paths.

Reproduce the execution and structure profiles with:

```bash
python benchmarks/filtered_bitflip_certificate_profile.py \
  --sequence-lengths 16 32 64 --bit-widths 8 \
  --trajectories shift_random shift_motif collapse \
  --clarities 0 1 \
  --json-out validation/filtered_bitflip_certificate_profile.json

python benchmarks/filtered_bitflip_certificate_profile.py \
  --structure-only --sequence-lengths 64 128 256 \
  --bit-widths 1 4 8 \
  --trajectories shift_random shift_motif collapse \
  --clarities 0 1 \
  --json-out validation/filtered_bitflip_certificate_structure.json
```

Additional bit-width and clarity sweeps are stored in
`validation/filtered_bitflip_certificate_bits.json` and
`validation/filtered_bitflip_certificate_clarity.json`.

## 9. Compact 1-8 experiment

`benchmarks/filtered_bitflip_compact.py` now implements and tests the eight
production-candidate directions from Section 7.  One proposed indexing
dimension was rejected as ill-defined, as described below.  This is still a
research path; the frozen `rosa_soft` package and its dense training gradient
are unchanged.

### 9.1 Implemented structures

1. **Induced suffix sorting.**  The optional native backend vendors `libsais`
   at commit `b6e52ef33fe14f9d5c14c580d162b6fd2c27f2a8`.  It uses the integer SA and
   PLCP APIs because 256 packed codes plus two ordered sentinels require 258
   symbols.  Native and Python SA/LCP results are compared exactly in tests.
2. **Succinct predecessor prototype.**  One global wavelet matrix stores key
   routes in suffix-rank order.  Its rank bitvectors are packed into 64-bit
   words, so an LCP node stores only a suffix-rank interval rather than a copy
   of every route below that node.
3. **Periodic predecessor.**  A node whose route set is proven contiguous by
   `(maximum - minimum + 1 == cardinality)` uses an arithmetic-run predecessor.
   Excluded event-family phases are also passed as exact arithmetic runs;
   contiguous exclusions are skipped in one jump.
4. **Indexed event families.**  The retained part is a start/stop envelope
   interval tree followed by the existing exact affine phase-range calculation.
   This rejects unrelated families without changing event semantics.  The
   proposed base-suffix-node bucket was not retained: a create event introduces
   a match absent from the base suffix tree, so it has no valid base node.
   Assigning one would require an edited suffix structure or a synthetic bucket
   that adds no filtering power.  Break events are already identified exactly
   by their old normalized priority.  Family construction also still consumes
   the raw event list; this experiment accelerates family queries but does not
   yet generate periodic families directly from occurrence runs.
5. **Sparse path.**  Low-compression flips expand only their influence-event
   cells.  Every touched row asks the compact suffix index for the best
   unchanged route, excludes the changed candidates, and compares their exact
   new lengths.
6. **Long-overlap path.**  High-compression flips use an interval winner
   certificate.  The same static `event_cells / canonical_entries` gate is
   retained; there is no clarity, entropy, or training-step heuristic.
7. **Bounded top certificate.**  Construction obtains the exact top `k+1`
   candidates per row through suffix predecessor queries and merges retained
   priorities into intervals.  The `(k+1)`-th priority is an exact omitted
   upper bound.  If temporary deletions exhaust the retained evidence, the
   traversal reaches a leaf and calls the suffix predecessor, preserving exact
   output for every `k >= 1`.
8. **Range VJP.**  Counterfactual winners are compared with the base winner
   segments and emitted as `AffineRouteRange` objects.  The VJP gathers only
   changed rows; it never allocates the `O(number_of_flips * T)` route matrix
   unless `materialize_routes=True` is explicitly requested for verification.

The certificate is built lazily on the first long-overlap flip.  Independent
states therefore pay for the compact suffix index but not for any candidate
certificate.

### 9.2 Exactness checks

The following checks passed on the recorded version:

- native versus Python SA and LCP, including packed values 253, 254, and 255;
- wavelet count, order statistic, and predecessor against direct sorting;
- compact versus posting-based suffix replacement on random and degenerate
  patterns;
- `top_k=1` certificate against every bit flip for all `D=1,T=4` Q/K states;
- 100 randomized end-to-end route, length, and VJP comparisons;
- complete repository suite: 980 tests passed and 2 skipped.

Direct range-VJP and matrix-VJP reductions differ only by floating-point sum
grouping; the observed float32 maximum error was `9.54e-7`.

### 9.3 Measured results

Single-run CPU measurements are stored in
`validation/filtered_bitflip_compact.json`.  They are Python prototype timings,
not a native-kernel forecast.

`libsais` generalized SA/LCP construction speedup over the Python doubling
backend was `1.20x/1.98x/2.40x/2.34x` at sequence lengths
`256/1,024/4,096/16,384`.  Python tuple conversion and RMQ construction remain
in both measurements.

Representative logical storage at `T=256` was:

| State and structure | Old bytes | Compact bytes | Ratio |
| --- | ---: | ---: | ---: |
| Independent D=1 suffix index | 36,556 | 16,408 | 0.45 |
| Independent D=1 certificate | 211,664 | 13,472 | 0.06 |
| Collapse suffix index | 269,280 | 9,660 | 0.04 |
| Collapse certificate | 9,196 | 4,384 | 0.48 |

The compact top-4 builder retains 481 intervals instead of enumerating 8,239
base match lines in the independent D=1 case.  In collapse it retains four
intervals at every measured length.  From `T=64` to `T=512`, collapse top-4
build time grew from about `2.4 ms` to `15.7 ms`; its retained interval count
stayed four.

End-to-end `T=64,D=8` results were mixed:

| State | Old hybrid ms | Compact top-k | Compact ms | Suffix fallback rows |
| --- | ---: | ---: | ---: | ---: |
| Independent | 32.4 | 4 | 23.6 | 0 |
| Clear shifted random | 258.4 | 4 | 323.1 | 0 |
| Clear shifted motif | 311.5 | 8 | 605.6 | 0 |
| Collapse | 822.0 | 8 | 2,325.0 | 21,280 |

### 9.4 Decision

Steps 1-3, the start/stop/phase part of 4, steps 5-6, and step 8 are useful and
should be retained as the compact exact candidate.  Base-suffix-node event
bucketing should be dropped unless a future edited-tree design gives create
events a real node identity.  Step 7 is correct but not yet universally fast.
A small fixed
`top_k` works for independent and shifted-random states; motif needed eight in
this test.  Collapse can delete a number of leading alternatives proportional
to context length, so no small constant `top_k` prevents repeated leaf
fallback.  Increasing `top_k` trades memory for work but does not solve that
scaling limit.

Consequently this version is a better sparse-state and memory reference, not a
replacement for the frozen dense-gradient implementation or the first full
certificate prototype.  Section 10 evaluates the proposed interval-level
suffix fallback.  It remains exact and output-sensitive; sampling and bounded
candidate truncation are still disallowed.

This work also does not establish a universal `O(T log T)` bound.  Exact event
discovery can itself emit quadratically many unrelated Hamming-0/1 changes, and
the collapse result shows that replacement queries can repeat over many rows
even when the events are highly compressible.

## 10. Whole-interval periodic suffix fallback

`benchmarks/filtered_bitflip_periodic.py` replaces repeated leaf replacement
queries with closed-form row envelopes when the base symbols provide a proof,
not merely a periodicity heuristic.  Unsupported and near-periodic inputs keep
the original exact suffix/certificate path.

### 10.1 Priority convention

At query row `t`, candidate route `r` with suffix length `L` is ordered by the
normalized lexicographic priority

```text
(L - t, r - t).
```

The first field implements longest suffix.  The second implements the latest
route tie break.  Expressing certificates in this coordinate system is useful
because a diagonal match run has constant priority over its complete row
interval.

### 10.2 Exact supported languages

The fallback currently proves three cases.

1. **Shift-aligned query edit.**  If `Q[1:] == K[:-1]`, route offset zero has
   base length `t`.  After flipping `Q[p]`, every unchanged candidate that
   avoids `p` has length at most `t-p`.  The shortened offset-zero candidate
   attains that bound and wins the latest-route tie.  A candidate longer than
   this bound must cross the edited equality, so it is represented by an exact
   create event.  The complete suffix baseline is therefore null at row `p`
   and priority `(-p, 0)` after it.
2. **Shift-aligned finite-period key edit.**  Let `K` have exact minimal finite
   period `P`, found in `O(T)` by a KMP prefix function, and again require
   `Q[1:] == K[:-1]`.  For a key edit at `j >= 2P-1`, the best route no greater
   than `j` is the latest route with the query row's phase.  Its priority is
   `(-kP, -kP)` for a whole block of `P` rows.  Other phases mismatch within
   one complete period.  This staircase competes with the shortened
   offset-zero priority `(-(j+1), 0)` and exact create events.
3. **Uniform period one.**  If Q and K contain one equal symbol, row `t` has
   base offsets `[1-t, 0]` and each priority is `(o, o)`.  A key edit at `j`
   has one affine prefix whose winner route is the constant `j`, followed by
   the shortened offset-zero route.  It is emitted with a constant number of
   descriptors.  The general arithmetic-predecessor implementation remains an
   exact reference for arbitrary periodic exclusion runs.

The `j >= 2P-1` guard is intentional.  Before two complete periods exist, the
phase candidate need not dominate a competing partial suffix.  Those early
key edits are handled by the general exact certificate rather than extending
the proof with a fragile boundary case.

### 10.3 Interval algorithm

The implementation has four stages per supported flip.

1. Build the closed-form baseline envelope for the entire affected suffix.
2. Ignore break-event short candidates: the proof above already supplies a
   candidate at least as long, with a no-earlier route on a tie.
3. For create events, compare each event family's maximum priority with the
   minimum baseline priority over the range.  A dominated family is rejected
   exactly.  Remaining constant-priority intervals are combined by an endpoint
   sweep and max heap.
4. Take the lexicographic maximum of the baseline and create envelope and emit
   only winner segments.

`AffineWinnerSegment.rows_per_step` encodes both ordinary affine priorities
and periodic staircases.  For a period-`P` key suffix, one descriptor stores
the whole phase-aligned range and advances its priority by `(-P, -P)` every
`P` rows.  Materialization is vectorized.  The existing VJP interface expands
the staircase only when converting it to affine route ranges; no suffix query
is performed during that conversion.

For `E_c` non-dominated create intervals and `S` emitted winner segments, the
replacement stage is `O(E_c log E_c + S)` after family construction.  It no
longer performs `O(rows * event_families)` work.  This is an output-sensitive
bound, not a universal bound on filtered bitflip: raw equality-change event
discovery remains separate.

### 10.4 Verification and measurements

The focused tests cover every supported flip in uniform, random shifted, and
period-2/3/4 inputs; finite-period boundary rejection; staircase
materialization; affine route-range reconstruction; VJP equality; and a fixed
case where a create addition really wins.  An additional differential fuzz
run passed 200 periodic or deliberately near-periodic route/length/range cases
and 16 VJP cases. The complete repository suite passes 980 tests with 2
environment-dependent skips.

An isolated Python control prebuilt identical event families and compared the
old row-by-family loop with the interval solver on a period-four motif.  At
`T=32/64/128`, replacement time changed from
`18.99/169.12/1374.69 ms` to `1.91/6.05/21.49 ms`, or
`9.96x/27.97x/63.98x`.  Winner descriptors changed from
`705/1,893/5,421` to `606/1,326/2,766`; all materialized winners were
identical.

The reproducible end-to-end `T=64,D=8` profile is stored in
`validation/filtered_bitflip_compact.json`:

| State | Exact hybrid ms | Compact ms | Logical periodic rows | Winner segments | Leaf queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shifted random | 243.8 | 167.9 | 16,128 | 1,992 | 0 |
| Shifted motif | 310.1 | 110.3 | 25,536 | 3,280 | 0 |
| Collapse | 795.1 | 17.3 | 32,256 | 3,752 | 0 |

No retained create interval was expanded in those three profile inputs; the
family-level bound proved all of them unable to beat the baseline.  This is a
measured property, not an assumption: the nonempty create-sweep path has a
separate exact regression case.

### 10.5 Decision and remaining bottleneck

Retain the whole-interval fallback as an exact third backend ahead of sparse
row updates and bounded certificates.  It removes the collapse/motif leaf
fallback failure and needs no learned threshold, candidate cap, sampling, or
approximation.

The initial interval version did not solve raw event enumeration.  In that
version, collapse events were `7,936/32,256/130,048` at `T=32/64/128` and
approximately quadrupled when T doubled.  Section 11 implements direct
occurrence-phase family generation and a periodic route-range VJP while
preserving conceptual event counts for comparison.

## 11. Periodic follow-up: items 1-3

### 11.1 Direct occurrence phases to event families

For exact shifted periodic inputs, a changed pair is identified by the motif
phase of its key-side and query-side symbols.  Positions in one phase form an
arithmetic progression with step `P`.  Within that progression:

- query/key positions are affine;
- left and right context lengths are constant or affine;
- `start`, `stop`, normalized long/short length, and route offset are therefore
  affine;
- at most one occurrence at a finite sequence boundary needs a singleton
  family.

`ShiftPeriodicEventFamilyBuilder` constructs these families directly.  It
computes neighboring phase compatibility within one primitive motif period,
then emits `PeriodicEventFamily` records without constructing changed-pair or
`InfluenceEvent` objects.  The builder requires exact `Q[1:] == K[:-1]` and the
exact finite minimum period.  Compact dispatch uses it only for flips already
accepted by the interval proof.  Nonperiodic query flips and early periodic key
flips still call the general event generator.

The direct and materialized paths were compared field by field over 12,656
periodic flips.  Expanded families had identical create/break flags,
`start/stop`, normalized long/short lengths, route offsets, event counts, and
event-cell counts.

At `T=64,D=8`:

| State | Conceptual events | Materialized events | Direct-family flips | Monotone cells | Compact ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shifted random | 1,254 | 691 | 0 | 64 | 167.9 |
| Shifted motif | 4,480 | 896 | 888 | 0 | 110.3 |
| Collapse | 32,256 | 0 | 1,008 | 0 | 17.3 |

The shifted-random row now includes the nonperiodic monotone-LCE path from
Section 12. Its 67 relevant create occurrences collapse to 64 cells; the
remaining materialized events are the general key-side path. The periodic
motif and collapse rows continue to use direct event families.

Collapse now scales close to the number of semantic flips in this Python
prototype: measured times at `T=32/64/128/256` were approximately
`7.9/15.8/33.8/67.2 ms`, despite conceptual event counts reaching 522,240 at
`T=256`.

### 11.2 Periodic route ranges

The first interval implementation compressed winner priorities but expanded a
staircase back into one affine route range per period block before VJP.
`PeriodicRouteRange` now stores the exact route formula

```text
r(t) = t + o0 + delta_o * floor((t - t0) / P).
```

It is emitted only when every overlapping base segment has route `t` and the
periodic offset cannot cross zero.  Otherwise conversion conservatively keeps
the previous affine expansion.  VJP constructs all rows and routes with one
vectorized floor division per descriptor and retains the same exact hard
output difference.

For shifted motif at `T=64/128/256`, descriptor counts changed from the affine
equivalent `1,560/3,168/10,296` to `1,008/2,040/4,080`.  An isolated `Dv=32`
VJP comparison gave `1.09x/1.15x/1.41x` speedups.  Compact and expanded VJPs
were numerically identical within the existing float32 reduction tolerance.

### 11.3 Why early key edits still use the general path

The finite-period key proof still requires `j >= 2P-1`.  A sweep over primitive
motifs with alphabet size four and periods one through five forced the phase
formula below this guard.  It found real failures, not only missing proof:

- period 3 had 20-40% failures at several early positions;
- period 4 reached 55% failures around `j=3,4`;
- period 5 reached 68.2% failures around `j=4,5`.

A minimal concrete counterexample is period-three motif `[0,1,0]`.  Flipping
`key[2]` makes the phase formula produce invalid route zero at row three, while
the exact winner is unchanged partial-suffix route one.  Create events cannot
repair this because the missing candidate does not cross the edited cell.

Some positions, including `j=0` and `j=2P-2` in this finite sweep, happened to
be exact.  They are not promoted: adding motif-dependent boundary rules would
increase complexity without removing the constant-size early prefix.  Every
`j < 2P-1` key flip therefore continues through the general sparse/certificate
backend with materialized exact events.

### 11.4 Current decision

Retain both new compressed structures under the existing proof gate.  They
remove the measured periodic bottlenecks while leaving arbitrary and early
states on the established exact implementation.  The next scaling question is
not another periodic special case; it is whether nonperiodic occurrence lists
can be grouped into exact monotone LCE runs without making construction as
expensive as enumerating their events.

## 12. Nonperiodic monotone LCE runs

`benchmarks/filtered_bitflip_monotone.py` answers the next question without
adding another periodic language case. Its input may be arbitrary. The only
structure it assumes is the exact suffix-array/LCP identity.

### 12.1 The one-sided theorem

Let suffix `x` have suffix-array rank `r`, and let `h[i]` be the LCP of suffix
ranks `i-1` and `i`. For any `i < r`,

```text
LCE(i, r) = min(h[i + 1], ..., h[r]).
```

As `i` moves toward `r`, this RMQ interval only shrinks. Its minimum therefore
cannot decrease. On ranks above `r`, the symmetric LCE cannot increase while
moving away from `r`. Filtering either arm by symbol code, causality, or any
other predicate takes a subsequence and cannot violate the monotonicity.

Equal minima form maximal rank plateaus. A monotone stack precomputes the
previous and next strictly smaller LCP entry in `O(T)`. Following those links
from one target rank emits one complete plateau per step, so construction is
`O(number_of_runs)` after the suffix array and LCP are available. It neither
scans a suffix character by character nor invokes a periodic fallback.

The semantic boundary is important. ROSA route candidates end at `K[T-2]`;
`K[T-1]` is never a usable route symbol. Capping an ordinary LCP afterward by
`T-k-2` introduces a candidate-position-dependent value and is not covered by
the rank theorem. The new forward generalized suffix array indexes `Q` and
`K[:-1]`, with distinct sentinels, so the raw LCP itself has the exact ROSA
right boundary. The reversed index handles left context. The common
`SuffixArrayMatchIndex` now uses the same exact forward boundary.

The implementation was checked in two ways:

- every emitted plateau covers each non-target rank exactly once and agrees
  with RMQ at every rank;
- all binary `T=4` Q/K states, semantic flips, and 3,072 influence events had
  zero forward/reverse monotonicity violations and exact event reconstruction.

### 12.2 Why one order is insufficient

Right LCE is monotone in the forward suffix order. Left LCE is monotone in the
reversed suffix order. The latest-route tie break is monotone in original
position order. These three permutations are unrelated on an arbitrary
string. Therefore a single occurrence list cannot, in general, be split into
runs that make all three quantities monotone. Position-order oscillations are
real; sorting by forward rank fixes right LCE but does not fix left LCE or route
priority.

The exact representation is instead a set of occurrence points

```text
(forward_suffix_rank, reverse_suffix_rank, original_position).
```

A forward LCE plateau and a reverse LCE plateau define one rectangle. Its
points have identical exact `(left_matches, right_matches)` and form one
`InfluenceLceCell`. The Python oracle currently intersects runs by visiting
the code-filtered occurrences. This deliberately measures the maximum useful
compression before committing to a heavier orthogonal range index.

### 12.3 Exact dominance inside a cell

For a query flip at fixed position `p`, all members of one cell have the same
row interval and the same normalized length priority. Only key position `k`
differs. The largest `k < p` is therefore the latest route on every active
row. One representative exactly preserves a create-cell candidate envelope.

Break cells need more care. Smaller routes are dominated before and after the
edit, but a generic replacement certificate must still know that all of them
were edited; otherwise it can resurrect a stale smaller route after deleting
the representative. A future generic solver must carry the cell rectangle as
an exclusion descriptor, not silently discard its points.

For a key flip at fixed `j`, cell members have different query positions `q`.
Their equal-width intervals are `[q, q+R+1)`, and both normalized priority
fields decrease as `q` increases. At output row `t`, the exact cell winner is
the earliest occurrence in `[t-R, t]`. Every occurrence can own a final tail
row, so one global representative is invalid. The prototype implements and
tests this successor rule; compression requires a shared occurrence index or
an envelope descriptor rather than event deletion.

### 12.4 Measured compression

`validation/filtered_bitflip_monotone.json` contains three-repeat Python
measurements at `T=64/128`. Every listed key lacks a period no greater than
half its length. Representative `T=128` query-side results are:

| State | Events / dual cell | Create representatives / occurrences | Singleton cells |
| --- | ---: | ---: | ---: |
| Independent byte | 1.20x | 81.3% | 80.3% |
| Independent alphabet-4 | 3.77x | 25.3% | 45.6% |
| Shifted motif plus noise | 1.90x | 18.4% | 81.2% |
| Shifted Thue-Morse | 4.10x | 18.2% | 33.1% |
| Shifted ruler sequence | 5.51x | 11.4% | 41.4% |

The byte result is the required negative control: sparse high-entropy
occurrences are mostly singletons, so a 2D/3D cell index cannot create useful
compression. Low alphabets and hierarchical nonperiodic strings do expose
large exact cells. Individual one-sided runs compress more strongly than the
dual cells because intersecting the independent forward/reverse partitions
necessarily fragments them.

### 12.5 Retained compact path

The existing shifted-query proof already supplies the complete post-edit
baseline and proves break-event short candidates irrelevant. For a shifted
query with a nonperiodic key, compact execution now:

1. visits only occurrences of the newly created center code;
2. partitions them into exact dual LCE cells;
3. keeps the largest key position in each cell;
4. sends those representatives to the existing interval envelope;
5. keeps the conceptual raw event count separate from materialized events and
   reports represented row work as `processed_event_cells`.

When a flip has zero or one create occurrence, it performs zero run
construction and directly evaluates at most one exact LCE. This parameter-free
case is not a heuristic: no merge is possible.

Same-process alternating A/B tests compared the retained path with an exact
copy in which only this branch was disabled. At `T=64/128/256`, nonperiodic
motif-noise, Thue-Morse, and ruler inputs generally improved by about 5-17%.
A 31-round `T=64` shifted-random-byte control improved from median
`177.20 ms` to `170.47 ms` (`1.039x`). The standard `T=64,D=8` profile reduced
materialized shifted-random events from 1,254 to 691 while preserving all hard
routes and lengths.

### 12.6 What is not proved

This result does not establish `O(T log T)` total bitflip evaluation. The
current cell oracle still visits every selected occurrence, and an adversarial
input can have linearly many LCE plateaus or occupied dual cells for one edit.
Nor does suffix-rank monotonicity justify sorting by original position or using
one query-cell representative for key edits.

The next data-structure candidate is a code-partitioned orthogonal index over
the three occurrence coordinates. It must support rectangle count, causal
position predecessor/successor, and rectangle exclusion references. A useful
implementation should enumerate only nonempty dual cells by recursively
pruning groups of forward/reverse plateaus. It should be promoted only if its
shared storage and query time beat the direct occurrence scan on low-alphabet
cases without regressing the singleton byte control. No additional periodic
special case is needed for that experiment.

## 13. Code-partitioned 3D orthogonal index

`benchmarks/filtered_bitflip_orthogonal.py` and
`benchmarks/filtered_bitflip_orthogonal_kd.py` implement the proposed index as
two exact research backends. Neither changes the compact solver or the frozen
dense-gradient operator.

### 13.1 Exact geometric reduction

For every usable key position `k`, store the point

```text
(code(k), forward_rank(k), reverse_rank(k), k).
```

The packed center code is categorical, so selecting the code created by a bit
flip first chooses one independent bucket. The remaining point has three
ordered coordinates. For a query flip at `p`, one dual LCE cell is exactly the
half-open box

```text
forward_rank in F_a
reverse_rank in R_b
position in [0, p).
```

`F_a` and `R_b` are exact suffix-rank plateaus from Section 12. Every point in
the box has the same `(left_matches, right_matches)`. Its aggregate stores
`count`, `min_position`, and `max_position`. `max_position` is the exact
latest-route representative for a query-create cell; no member is visited to
choose it. Code partitioning is important: treating code as another ordered
coordinate would add an unnecessary fourth tree dimension and would obscure
the exact XOR-created-code lookup.

The final key symbol remains excluded before point construction, so the index
uses the same `K[:-1]` semantic boundary as the LCE oracle.

### 13.2 Causal sweep as a dynamic 2D index

Semantic query flips are already ordered by `p`. Before processing position
`p`, the dynamic backend activates each key point with `k < p` exactly once.
The position prefix is then represented by membership in the active set, and
the 3D box becomes a 2D forward/reverse-rank rectangle.

Each code bucket has an outer segment tree over forward rank. Every outer node
has an inner aggregate tree over the reverse ranks present below that node.
One activation updates `O(log T)` outer nodes and `O(log T)` inner state per
node. A standalone rectangle query is `O(log^2 T)`, and logical storage is
`O(T log T)`.

Issuing one rectangle query for every product of forward and reverse plateaus
was exact but unsuccessful. Even recursive empty-box pruning issued thousands
of queries at `T=128` on low alphabets, often five to ten queries per occupied
cell. The retained `tree` research backend instead traverses the outer and
inner trees in synchrony with both LCE band partitions. A fully contained
tree node contributes its aggregate directly to one cell. It therefore visits
tree nodes, not occurrences, and never constructs the Cartesian product of all
plateau pairs.

The Python topology builder independently sorts every inner list, so its
current construction bound is `O(T log^2 T)`. A native range-tree build can
merge child rank lists bottom-up in `O(T log T)` without changing the query
semantics.

### 13.3 Linear-space static 3D alternative

The static backend builds one balanced 3D kd tree per code. Every node stores
its forward-rank, reverse-rank, and position bounding box plus the same
aggregate. Splits cycle position, reverse rank, then forward rank; leaves hold
at most four points by default.

It exposes an exact arbitrary half-open 3D box query, not only a causal prefix.
A node outside the box is pruned, a node fully inside contributes in constant
time, and only a boundary leaf probes points. Consequently the aggregate can
answer rectangle count, position predecessor, and position successor queries.
The LCE grouping path similarly accepts an arbitrary position interval.

Logical storage is `O(T)`, but an adversarial query can still visit every kd
node. Bounding boxes are also fragmented simultaneously by causality and two
unrelated suffix orders. The structure is therefore a useful low-memory exact
reference and a future key-side primitive, not a guaranteed faster cell
enumerator. The Python median-split builder is `O(T log^2 T)` because it sorts
at each level; a native presorted build can reduce construction cost.

### 13.4 Exactness checks

The focused test matrix validates:

- every query flip for all binary `D=1,T=4` Q/K states;
- random and boundary-heavy inputs for `D=1,2,4,8`;
- dynamic rectangles against direct active-point filtering;
- static arbitrary 3D boxes against direct filtering for leaf sizes 1, 4,
  and 8;
- exact LCE cell keys, counts, position extrema, and dominant events;
- static LCE grouping for leaf sizes 1, 4, 8, and 16;
- rejection of a causal sweep that moves backward.

The profiler repeats the exact cell/extrema/event comparison outside its timed
regions. Thus equal aggregate counts alone cannot hide a changed partition or
latest-route result.

### 13.5 Measured behavior

`validation/filtered_bitflip_orthogonal.json` and the separate `T=2048` run
contain three-repeat Python medians. Geometry and the ordinary occurrence
index are shared by all paths; each indexed total includes its own index build.
Representative `T=2048` results are:

| State | Occurrences / cell | Scan ms | Dynamic ms | Dynamic speedup | Dynamic KiB | Static kd ms | Static kd KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent byte | 4.42 | 531.5 | 765.6 | 0.69x | 312 | 572.4 | 74 |
| Independent alphabet-4 | 21.14 | 1287.2 | 1963.7 | 0.66x | 656 | 2438.8 | 74 |
| Shifted random byte | 4.42 | 552.5 | 741.4 | 0.75x | 313 | 653.5 | 75 |
| Shifted motif plus noise | 71.00 | 1105.9 | 981.3 | 1.13x | 576 | 1380.8 | 75 |
| Shifted Thue-Morse | 77.77 | 1024.1 | 693.4 | 1.48x | 656 | 1152.2 | 72 |
| Shifted ruler | 122.12 | 1097.2 | 588.2 | 1.87x | 723 | 879.7 | 79 |

The reported memory is logical packed-array storage, not Python object RSS.
On the three structured rows, dynamic tree-node visits were only 12.7-20.0%
of occurrence visits. On independent alphabet-4 they were 77.2%, and on the
two sparse byte controls they were about 194-197%. This explains both the
useful crossover and the regressions: an index eliminates direct occurrence
iteration only when many points can be accepted as complete tree regions.
Context length, alphabet size, and raw occurrence count do not establish that
property by themselves.

### 13.6 Decision and remaining work

Retain both indexes as exact research references, but do not dispatch the
compact solver to either one yet. There is no universal win:

- the dynamic range tree is the speed candidate for long structured inputs,
  but its `O(T log T)` words are not a plausible 100M-token representation;
- the static kd tree uses linear logical storage, but its boundary traversal
  usually loses to the dynamic tree and can lose to the occurrence scan;
- an arbitrary string can still have linearly many occupied dual cells for
  one flip, so neither backend proves subquadratic total bitflip evaluation.

A production experiment should first flatten all code-bucket topology into
native arrays, use bottom-up merged construction, and batch flips at one query
position. It should retain the direct occurrence path for fragmented cases.
A sampled prefix may estimate `tree_nodes_visited / occurrence_count` or
`occurrence_count / occupied_cells`, but that gate must pay for itself and be
validated across training trajectories; a threshold based only on `T`, code
density, or alphabet width is unsupported by these results.

For key-side edits, `min_position` over an arbitrary position box supplies the
required local successor primitive. It does not yet supply the complete
winner envelope across all output rows. Break edits additionally require a
rectangle exclusion descriptor so replacement queries cannot resurrect an
edited nonrepresentative. Those two algorithms remain prerequisites for a
full indexed filtered-bitflip path.
