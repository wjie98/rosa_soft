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
grow and its right-of-edit threshold can only weaken. The implementation
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

## 9. Compressed result and direct VJP

The default Python result does not allocate the full `[2(T-1)D, T]` route and
length matrices. Each flip stores only affine row intervals:

```text
row in [start, stop)
length(row) = length_start + (row - start) * length_step
route(row)  = route_start  + (row - start) * route_step.
```

Validation can reconstruct the matrices with `materialize_route_changes`.
The bitflip VJP consumes the intervals directly, so unchanged rows are never
stored or visited. `materialize_routes=False` is the default; setting it true
is a test/debug operation. The current direct VJP still visits every changed
row inside an interval and is not yet a closed-form range reduction.

## 10. Native C++ core

`benchmarks/csrc/sam_bitflip_cpu.cpp` is an independent C++17 implementation
of the same dual-SAM algorithm. It includes the arithmetic certificate,
grouped Q branches, binary K replacement search, and anchored virtual runs.
`benchmarks/sam_bitflip_native.py` builds and calls it through a small ctypes
ABI. It currently materializes complete route/length matrices on purpose so
its timing can be compared with the Python materialized path and every cell
can be checked exactly.

This core is not a production operator yet. Its output alone is
`Theta(D T^2)` and dominates memory at long lengths: `T=512,D=8` writes about
`64 MiB` for routes plus lengths. The next native interface should consume
compressed changes or accumulate the VJP directly before this path is used in
training.

## 11. Complexity and remaining bottleneck

Let `S <= 2T-1` be the SAM state count.

- SAM, link tree, and prefix terminals: `O(T)` logical structure.
- explicit endpos: up to `Theta(T^2)` entries.
- implicit wavelet prototype: `O(T log S)` rank/select entries.
- one endpos predecessor: `O(log S)` in the implicit backend.
- one Q flip: actual replay recovery distance, worst-case `O(T)`.
- one key position's base replacement: stabbed rows times `O(log L)` length
  probes and predecessor cost.
- one virtual-run group: selected centers plus active envelope rows, with heap
  logarithms.

The route does not prove subquadratic work across all K flips. At low alphabet
width, the number of created `(j,p)` centers can be quadratic. Materializing
all counterfactual route matrices is itself `Theta(D T^2)`. The Python path
has separated compressed VJP consumption from validation materialization; the
C++ ABI has not. Neither path proves subquadratic total work for all possible
low-alphabet inputs.

## 12. Validation and measurements

`tests/test_sam_bitflip.py` covers:

- every binary Q/K pair at `T=4` and `T=5`;
- random `D=1,2,4,8` sequences;
- periodic, collapsed, shifted, and latest-route tie cases;
- direct LCE versus dual-SAM LCA;
- every-state/every-bound explicit versus implicit endpos queries;
- query replay, K replacement, virtual runs, cache cold fallback, and VJP;
- occurrence-list and bitset center enumeration;
- compressed descriptor reconstruction and direct VJP parity;
- arithmetic certificates against explicit predecessor answers.

`tests/test_sam_bitflip_native.py` additionally compiles the C++ core and
checks empty and singleton boundaries, random and structured cases, and every
binary Q/K pair at `T=4,5`. A separate UBSan corpus covers
`T=0..32,64,128`.

Reproduce timing and work counters with:

```bash
python benchmarks/sam_bitflip_profile.py \
  --build-native --sequence-lengths 32 64 --repeats 5 --compare-v1 \
  --json-out validation/sam_bitflip.json
```

The profiler checks every configuration against the full-rerun oracle before
recording it. The frozen v1 comparison is a route comparison, not a claim that
either architecture dominates all input structures.

The WSL2/Python 3.12 profile recorded these logical endpos entry counts. They
are structure counts, not Python heap bytes:

| Key | T | Explicit entries | Implicit entries | Explicit / implicit |
| --- | ---: | ---: | ---: | ---: |
| Random D8 | 64 | 136 | 1,081 | 0.13x |
| Random D8 | 1,024 | 3,052 | 25,814 | 0.12x |
| Collapse | 64 | 2,079 | 945 | 2.20x |
| Collapse | 256 | 32,895 | 4,845 | 6.79x |
| Collapse | 1,024 | 524,799 | 23,529 | 22.30x |

This motivated the implemented exact hybrid: retain small explicit posting
lists and use the implicit predecessor for large endpos classes. The current
default still uses implicit endpos plus arithmetic certificates to preserve a
simple nonquadratic storage bound.

The five-repeat `T=64` measurements below include complete correctness checks
against rerunning every bit flip. Times are milliseconds on the recorded WSL2
CPU environment:

| Case | Python compressed | Python materialized | C++ materialized | Frozen v1 materialized |
| --- | ---: | ---: | ---: | ---: |
| Independent D1 | 22.33 | 30.87 | 0.35 | 152.09 |
| Independent D8 | 7.05 | 14.00 | 0.21 | 39.97 |
| Shift D8 | 165.33 | 393.74 | 0.42 | 328.84 |
| Motif D4 | 100.81 | 211.21 | 0.52 | 159.97 |
| Collapse D8 | 167.26 | 386.94 | 0.56 | 200.63 |

The C++ result shows that implementation language and data layout dominate
the short-sequence timings. Libsais in frozen v1 accelerates SA/LCP
construction only and does not remove Python solve-loop overhead. It also
does not remove the quadratic materialized-output bound.

## 13. Outcome of optimization pass 1-6

1. **Retain:** compressed affine route changes and default-off
   materialization. Direct VJP reads the compressed form.
2. **Retain:** monotone binary search for K replacement length. Exhaustive
   binary tests prove parity with full reruns.
3. **Retain:** grouped Q counterfactual branches. It is exact and strongly
   reduces replay on low-alphabet and collapsed trajectories.
4. **Retain as optional:** exact last-M plus cold fallback. It helps only some
   sparse cases, so `hot_cache_size=0` remains the default.
5. **Retain and enable by default:** exact arithmetic-progression endpos
   certificates. They improve structured cases without semantic risk.
6. **Retain as a research backend:** the independent C++ core. Its large
   speedup justifies a native direct-VJP follow-up, but the current
   materialized ABI is not a production training interface.
