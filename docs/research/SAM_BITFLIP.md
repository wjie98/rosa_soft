# Exact SAM Bitflip Route

This document describes the second exact unlimited-suffix bitflip research
route. Its entry point is `sam_bitflip` in `benchmarks/sam_bitflip.py`. It is
independent from the frozen `filtered-bitflip-index-v1` route and does not
change the production RosaSoft operator.

The semantic oracle remains `brute_force_bitflip` in
`benchmarks/filtered_bitflip.py`.

## 1. Exact contract

For query row `t`, candidate route `r` uses key end `e=r-1`, with
`0 <= e <= t-1`. Its match length is the longest equal suffix of `Q[:t+1]`
and `K[:e+1]`. The winner maximizes `(length, route)` lexicographically.
Length zero selects null route zero. Consequently:

- usable key history is `K[0:T-1]`; `K[T-1]` is inactive;
- active query positions are `Q[1:T]`; `Q[0]` is inactive;
- equal positive lengths select the latest key end;
- no configured suffix window is used.

Every execution backend in this route preserves that result exactly. The
hot-cache size and center-enumeration backend affect work only.

## 2. Why full endpos helps, and what it does not contain

A suffix automaton state represents one end-position equivalence class. For
state `s`, all substring lengths in

```text
max_length(link(s)) < length <= max_length(s)
```

share `endpos(s)`. A predecessor query

```text
latest end in endpos(s) with end <= causal_bound
```

therefore recovers the latest legal route for a represented query suffix.
Full endpos sets are sufficient to reconstruct every required Q/K match, but
they are not the literal `Q x K` dynamic-programming table. Query trajectories
are still needed to identify which state and represented length are active.

Explicitly storing every `endpos(s)` is an exact `M=infinity` oracle. Its total
number of entries can be quadratic. For example, a unary key has one nested
suffix state per length and each state contains many end positions.

## 3. Shared static index

The solver builds a SAM over `K[0:T-1]` and records the terminal state after
every key prefix. It then builds the suffix-link tree, iterative Euler order,
binary-lifting ancestors, two exact endpos backends, and an optional exact
arithmetic certificate in front of either backend.

### 3.1 Explicit backend

`ExplicitEndPositionIndex` propagates each prefix terminal's end position up
the suffix-link tree and stores a sorted vector per state. Queries use binary
search. This is the simple correctness oracle and can be fast on high-entropy
short inputs, but it has quadratic worst-case storage.

### 3.2 Implicit backend

For key end `e`, let `terminal[e]` be its prefix terminal. The defining
identity is

```text
e is in endpos(s)
iff
s is an ancestor of terminal[e] in the suffix-link tree.
```

Euler numbering maps the subtree of `s` to `[tin(s), tout(s))`. The endpos
predecessor becomes:

```text
rightmost e <= bound
such that tin(terminal[e]) is in [tin(s), tout(s)).
```

`ImplicitEndPositionIndex` answers this exact two-dimensional query with a
wavelet range-predecessor tree. It stores no state-to-occurrence expansion.
The Python prototype uses integer lists at every wavelet level; a native
version should pack rank/select bitvectors before treating these logical
entry counts as physical memory estimates.

### 3.3 Exact arithmetic-progression certificate

For every suffix-link subtree, the solver aggregates occurrence count,
minimum, maximum, and the GCD of all position gaps. The endpos set is proven
to be one complete arithmetic progression exactly when

```text
count == 1
or
gcd > 0 and (maximum - minimum) / gcd + 1 == count.
```

Its predecessor is then computed in `O(1)` by rounding the bound down to the
progression. States that fail the equality use the exact backing index. This
is a certificate rather than a periodicity guess: it never approximates or
truncates end positions. It is enabled by default because structured inputs
produce very high certificate hit rates, while random sparse states fall back
without changing semantics.

### 3.4 Causal static-SAM matching

The SAM is built from the complete usable key, so a transition may exist only
in the future relative to row `t`. A transition is accepted only if its
canonical endpos state has a predecessor at `end <= t-1`. Otherwise matching
follows suffix links and retries. This makes the static SAM exactly equivalent
to incrementally exposing key prefixes.

Each base query row records:

```text
(canonical_state, matched_length, latest_end)
```

These traces are both the base hard result and checkpoints for Q edits.

## 4. Dual SAM and anchored context lengths

A second SAM is built over `reverse(K[0:T-1])`. Suffix-link-tree LCA gives the
common-suffix length of two represented prefixes. The forward SAM computes

```text
A = LCS(Q before p, K before j)
```

and the reverse SAM computes

```text
R = LCP(Q after p, K after j).
```

The result is capped by the represented query length and exact prefix
boundaries. Tests compare every returned `A/R` with direct character scans.

## 5. Query-bit edits

Flipping `Q[p]` cannot change rows before `p`. The solver restarts from the
base checkpoint at `p-1`, consumes the changed symbol, and then consumes the
unchanged suffix `Q[p+1:]` with the normal causal bounds.

Replay stops when the counterfactual trace equals the base trace at the same
row. At that point the future symbols, future bounds, DFA state, represented
length, and latest end are all equal, so every later transition is identical.
This is an exact coalescence condition, not a recovery-distance heuristic.

All bit alternatives at one query position are advanced together. Alternatives
with the same complete trace `(state, length, latest_end)` are merged into one
DFA branch and share every later transition until they coalesce with the base.
This changes work, not the estimator: each bit still receives its own exact
counterfactual rows.

Random high-alphabet traces often coalesce quickly. A periodic or collapsed
trace can remain different to the end, so the worst case is still linear per
query bit.

## 6. Key-bit edits

For `K[j]: a -> b`, every substring of the edited key belongs to exactly one
of two disjoint languages:

```text
U_j     original-key substrings whose interval excludes j
C_j,b   new substrings whose interval crosses j and uses symbol b
```

The edited result is the longest/latest winner over `U_j union C_j,b`.

### 6.1 Unaffected language

For state `s`, represented length `l`, row bound `bound`, and occurrence end
`e`, an occurrence avoids `j` iff either:

```text
e < j
```

or

```text
e - l + 1 > j, equivalently e >= j + l.
```

The solver asks for the latest occurrence before `j` and the latest occurrence
before `bound`. If the latter is right of `j`, it supports at most `e-j`
symbols. Feasibility is monotone in candidate length: once a length is
feasible, every shorter suffix is feasible because its endpos set can only
grow and its right-of-edit threshold can only weaken. The Python reference
therefore binary-searches the maximum feasible length. Binary lifting maps a
length to its canonical suffix-link ancestor, reducing a linear suffix-link
walk to `O(log L)` exact feasibility probes. The final two predecessors choose
the latest legal route at that length.

This replacement work is needed only when the base winner interval contains
`j`. If the base winner excludes `j`, removing matches that cross `j` cannot
change or improve it. `_WinnerIntervalIndex` stores base intervals in a static
segment tree and reports exactly those rows stabbed by a key position.

Replacement rows are shared by every bit at the same key position because
deleting the old symbol is independent of which new bit value is inserted.

The native core batches this problem in the other direction. For one query
row with base length `L`, it walks the suffix-link chain once and records, for
every suffix length `l`, its canonical state, minimum occurrence end
`minimum[l]`, and latest causal occurrence end `latest[l]`. Every occurrence
of that suffix crosses key position `j` exactly on the interval

```text
[latest[l] - l + 1, minimum[l]].
```

These invalid-position intervals are nested as `l` increases. An ascending
length sweep assigns only newly covered key positions the replacement length
`l-1`. One final predecessor query is needed only when the selected occurrence
must come from the left side of the edit. This computes all K-deletion
replacements for the row together and removes the per-`(row,j)` binary search.
The native path therefore does not need a winner-interval segment tree.

### 6.2 Anchored virtual runs

An ordinary transition must not be added to a SAM state for a K edit. A SAM
state merges many occurrences; a global edge would incorrectly enable the
edited transition at unrelated positions.

Instead, every new matching center `(j,p)` with `p>j` and `Q[p]=b` creates an
anchored virtual run. With the dual-SAM context lengths `A` and `R`, it covers
rows

```text
p <= t <= p + R
```

and has

```text
length(t) = A + 1 + t - p
route(t)  = j + 1 + t - p.
```

Subtracting the common row coordinate gives constant priority:

```text
(length(t)-t, route(t)-t) = (A+1-p, j+1-p).
```

The implementation sweeps run starts and expirations with a priority heap.
This evaluates one affine interval instead of simulating an NFA path for every
token. It compares the best active virtual run with the exact `U_j` result.

## 7. Last-M cache

`hot_cache_size=M` stores the latest `M` end positions per suffix-link
subtree. If a cached predecessor exists, it is necessarily the exact latest
predecessor. If a state has at most `M` occurrences, a negative lookup is also
exact. Every other miss falls back to the implicit or explicit exact index.

A fixed last-M-only implementation is not exact. In periodic data, all `M`
cached occurrences can lie after an early causal bound or inside an excluded
region while an older valid occurrence remains. In this route, `M` is only a
performance parameter and defaults to zero.

The measurements do not support unconditional caching once arithmetic
certificates are enabled. At `T=64`, `M=8` improved independent D1 from
`22.33 ms` to `19.24 ms`, was neutral on independent D8, and ranged from a
small improvement to a regression on structured cases. The cache therefore
remains optional and default-off.

## 8. Batched centers and bitsets

Query positions are partitioned by packed code. For all bit alternatives at
one key position, only the occurrence bucket for `K[j] xor (1<<bit)` can form
new centers. Runs are built once per `(j,bit)` group and reused by the solve.

The prototype provides two exact enumerators:

- sorted occurrence lists with `bisect_right`;
- one Python integer bitset per packed code.

Neither is a universal timing winner at `T=32/64`; the differences were small
and pattern dependent. Both remain explicit execution backends. A native CPU
implementation can use word-level bitset intersection only after profiles
show enough selected centers to amortize scanning the words.

## 9. Factorized result and direct VJP

The default Python result does not allocate the full `[2(T-1)D, T]` route and
length matrices. Each flip stores only affine row intervals:

```text
row in [start, stop)
length(row) = length_start + (row - start) * length_step
route(row)  = route_start  + (row - start) * route_step.
```

Validation can reconstruct the matrices with `materialize_route_changes`.
The Python bitflip VJP consumes the intervals directly, so unchanged rows are
not stored. It still visits each changed row and remains a reference path.

The native representation further factorizes K edits. Let
`F=(T-1)D` be the number of Q flips and also the number of K flips. It stores:

1. one base route/length vector of length `T`;
2. one affine change list for each Q flip, relative to the base;
3. one shared K-deletion change list for each key position, relative to the
   base;
4. one K-insertion override list for each K flip, relative to that position's
   shared deletion result.

An affine change is six `int32` values:

```text
[start, stop, length_start, length_step, route_start, route_step]
```

A K override is ten `int32` values and records both affine winners:

```text
[start, stop,
 from_length_start, from_length_step, from_route_start, from_route_step,
 to_length_start,   to_length_step,   to_route_start,   to_route_step]
```

The `from` winner is required for a direct VJP: a virtual insertion may
replace a non-base deletion winner. Keeping only the final route would force a
materialization or recomputation. CSR-style `int64` offsets delimit each list.

For an upstream row gradient `g`, define `delta(a,b)` as the contraction of
`g` with the hard value change from route `a` to route `b`. The exact work is
then

```text
Q flip: delta(base, q_counterfactual)
K flip: delta(base, shared_delete) + delta(shared_delete, bit_override).
```

This identity is why K deletion can be evaluated once per key position rather
than once per bit. Route matrices are needed only for validation.

## 10. Native C++ and CUDA path

`benchmarks/csrc/sam_bitflip_cpu.cpp` is an independent C++17 implementation
of the dual-SAM algorithm. It includes causal endpos queries, exact arithmetic
certificates, grouped Q branches, row-batched K deletion, and anchored virtual
runs. `benchmarks/sam_bitflip_native.py` exposes two APIs:

- `solve_factorized()` returns the base and three descriptor families;
- `solve()` reconstructs the legacy full matrices from that same factorized
  result for parity tests.

There is no second materialized solver. The opaque C ABI uses
`factorized_create`, `factorized_sizes`, `factorized_copy`, and
`factorized_destroy`, so allocation ownership and descriptor sizes are
explicit. The legacy `rosa_sam_bitflip_routes` symbol remains a validation
adapter.

The native profiler exports 31 counters covering SAM construction, endpos and
trace construction, Q/K solve time, materialization time, allocated capacity,
predecessor work, virtual-run work, descriptor counts, and factorized versus
materialized bytes. The original first 13 counters retain their ABI order.

`benchmarks/csrc/sam_bitflip_vjp_cuda.cu` consumes the descriptors without
building route matrices. It launches:

- one Q block per Q flip;
- one shared-delete block per key position;
- one override-correction block per K flip.

Each 256-thread block walks its affine ranges, distributes value features
across threads, and performs one block reduction. FP16 and BF16 accumulate in
FP32; FP32 and FP64 use their PyTorch accumulation types. The extension keeps
value and upstream-gradient tensors on the GPU, supports the current CUDA
stream/device, and intentionally does not use `--use_fast_math`.

This remains a benchmark-only research backend. It is not wired into
`RosaSoftFunction`, does not alter hard forward semantics, and does not replace
the frozen dense-gradient production reference.

## 11. Complexity and remaining bottleneck

Let `S <= 2T-1` be the SAM state count and `C` be the number of changed rows
represented by the emitted descriptors.

- SAM, suffix-link tree, traces, and terminal arrays use `O(T)` storage.
- The implicit endpos index uses `O(T log S)` rank/select entries.
- One ordinary endpos predecessor is logarithmic; certified arithmetic states
  answer directly.
- One Q flip replays only until exact trace coalescence, but remains `O(T)` in
  the worst case.
- Native K deletion walks the suffix lengths of each query row once and then
  writes affected key positions. It removes a binary-search factor but remains
  `O(T^2)` in the worst case over all rows.
- Anchored insertion centers and active virtual rows can also be quadratic on
  low-alphabet structured input.
- Factorized output uses `O(T + descriptor_count)` storage; its worst case is
  still `O(D T^2)` because exact counterfactuals need not compress.
- CUDA VJP work is `O(C * Dv)` for value width `Dv`, with `O(DT)` output and
  no persistent `[2(T-1)D,T]` route/length tensors.

There is no suffix window `W`: all matches are exact and unbounded. The route
does not establish an `O(T log T)` worst-case bitflip algorithm. It removes
avoidable repeated work and quadratic materialization in the common path,
not the inherent adversarial number of changed counterfactual rows.

## 12. Validation and measurements

`tests/test_sam_bitflip.py` covers exhaustive binary inputs at `T=4,5`, random
`D=1,2,4,8` inputs, periodic/collapsed/shifted/tie cases, dual-SAM LCE,
explicit versus implicit endpos, Q replay, K replacement, virtual runs,
compressed reconstruction, direct VJP, and arithmetic certificates.

`tests/test_sam_bitflip_native.py` independently checks materialized and
factorized native results against Python, including exhaustive binary inputs.
`tests/test_sam_bitflip_vjp.py` checks the CUDA descriptor VJP against the
Python bitflip oracle in FP64, FP32, FP16, and BF16. FP64 parity is constrained
to `rtol=atol=1e-12`. The 2026-08-06 validation run also passed strict C++
warnings, a 176-case UBSan corpus, and the complete repository suite:

```text
1176 passed, 2 skipped
```

Reproduce CPU timing, compression, and work counters with:

```bash
python benchmarks/sam_bitflip_profile.py \
  --build-native --sequence-lengths 32 64 --repeats 5 --compare-v1 \
  --json-out validation/sam_bitflip.json
```

Reproduce factorized CPU plus CUDA VJP timing on GPU 1 with:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/sam_bitflip_vjp_profile.py \
  --sequence-lengths 64 128 256 512 --feature-size 128 --repeats 5 \
  --validate-max-length 128 \
  --json-out validation/sam_bitflip_vjp.json
```

The VJP profiler repeats the expensive Python full-bitflip oracle only through
the configured validation length. Longer records set `oracle_validated=false`;
the standalone exhaustive and FP64 tests remain the correctness gate.

The five-repeat `T=64` native measurements below include oracle parity checks.
Times are milliseconds; bytes include base arrays, offsets, and descriptors:

| Case | Materialized C++ | Factorized C++ | Descriptor bytes | Matrix / descriptor |
| --- | ---: | ---: | ---: | ---: |
| Independent D1 | 0.357 | 0.362 | 15,616 | 8.3x |
| Independent D8 | 0.238 | 0.186 | 14,632 | 70.5x |
| Shift D8 | 0.540 | 0.465 | 35,888 | 28.8x |
| Motif D4 | 0.408 | 0.406 | 20,912 | 24.7x |
| Collapse D8 | 0.962 | 0.798 | 35,192 | 29.3x |

At short lengths, factorized timing can be similar to materialization because
handle creation and descriptor copies dominate. Its important property is
that persistent result bytes track actual changes rather than forcing
`Theta(D T^2)` matrices.

## 13. Outcome of optimization pass 1-6

1. **Retain native phase and capacity profiling.** It localized work to Q
   replay, K deletion, virtual overrides, descriptor copying, or optional
   materialization instead of attributing all time to SAM construction.
2. **Retain the compressed C++ ABI.** Both APIs now share one factorized solve;
   full matrices are a reconstruction target, not the native algorithm's
   internal contract.
3. **Retain factorized K delete plus override.** The bit-independent deletion
   term is stored and contracted once per key position. Exact `from/to`
   overrides preserve the inserted-bit correction without hidden state.
4. **Retain CUDA descriptor VJP as research infrastructure.** It passes the
   exact oracle and removes host-side value contraction and route-matrix
   storage. Integration into training requires a separate architecture and
   scaling decision.
5. **Delete the winner-index and event-envelope experiments from the native
   implementation.** A winner interval index improved the old point-query
   path by about 25% on one random D8 `T=512` case but was neutral on structured
   cases; row batching superseded it. The event envelope slowed random D1-D4
   cases by roughly 20-33% because construction cost exceeded saved scans.
6. **Retain row-batched K replacement.** The nested invalid-interval sweep
   replaced per-cell binary search. On structured `T=2048` probes fell from
   about 21.66 million to 2.096 million. The earlier point path's `T=1024`
   collapse case fell from about 254 ms to 76 ms before later cleanup; final
   behavior remains exact on exhaustive, random, and structured corpora. A
   final five-repeat CPU run at `T=2048` measured 204.9 ms for shift D8,
   203.9 ms for motif D4, and 246.9 ms for collapse D8.

The final native implementation deliberately contains no heuristic candidate
limit, suffix window, approximate occurrence cache, winner event special case,
or duplicate materialized solver.
