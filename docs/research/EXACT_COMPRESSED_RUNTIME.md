# Exact Compressed and Pageable Runtime

Status: architecture constraint and experiment plan, 2026-08-22.

## 1. Non-negotiable semantics

For token `t`, one ROSA runtime head must execute in this order:

1. query the half-open key history `K[0:t)`, excluding `K[t]`;
2. find the longest non-empty suffix of `Q[0:t+1)` occurring in that history;
3. among equal-length occurrences, select the greatest key end position;
4. return the payload at the selected end plus one, or exact null when no
   non-empty suffix occurs;
5. append `K[t]` to the index without changing the result for token `t`.

The history and suffix length are unlimited. Memory capacity is not a semantic
parameter. In particular, no compressed or paged implementation may turn a
cache size, page size, compressed-block size, rebuild epoch, or I/O budget into
a hard matching window or candidate limit.

The replacement must preserve the following observable state across arbitrary
chunk boundaries:

- active query state and represented suffix length;
- every exact transition and suffix fallback;
- the latest causal occurrence for every represented substring class;
- latest-occurrence tie-breaking and null behavior;
- payload successor addressing;
- state evolution after clones, transition redirection, and suffix-link
  reparenting.

## 2. Why the current representation does not reach 100M

The current online suffix automaton is exact and fast, but random one-head D8
input at 1M tokens used about 47 MB, roughly 45 automaton bytes per token plus
one payload byte. Keeping the same layout at 100M therefore requires several
gigabytes per head. The unary shortcut proves that structured histories can be
represented sublinearly, but the period-64 experiment also shows that a
standard finite-text SAM may remain linear even when the source has a compact
generator.

There are two distinct optimization targets:

- **compressed automaton:** reduce total bytes as a function of text
  repetitiveness, such as runs, maximal repeats, or grammar size;
- **pageable state:** retain exact total state but reduce resident RAM by moving
  cold blocks to a lower storage tier.

Paging is not compression. Compression is not automatically page-friendly.
Both require stable logical IDs and exact mutation handling.

## 3. Required abstract operations

Candidate indexes should be judged against ROSA operations, not generic
substring-search throughput:

| Operation | Required behavior |
| --- | --- |
| `append_key(c, t)` | Append one symbol online and make it visible only to later queries. |
| `extend_query(state, c, t)` | Extend the current query suffix; on failure, contract from its old end until the globally longest causal suffix is found. |
| `latest_end(state, t)` | Return the greatest occurrence end strictly below `t`. |
| `read_successor(end)` | Return payload `end + 1` exactly, including across storage pages. |

An index that supports only static `count(pattern)` or `locate(pattern)` is not
sufficient. ROSA additionally needs online text growth, right contraction of
the active pattern after a failed extension, and a latest-position aggregate.

## 4. Candidate structures

### 4.1 Paged suffix automaton

This is the lowest semantic-risk first implementation because it preserves the
existing state machine and changes storage rather than matching logic.

Use stable 32-bit logical IDs and split storage into:

- a resident hot arena for the root, high-degree transition indexes, active
  query states, newest states, and mutable construction frontier;
- sealed cold pages for state and edge records;
- a sparse exact overlay for fields of sealed records changed by clone
  creation, transition redirection, suffix-link reparenting, or new edges;
- a bounded decompressed-page cache with explicit hit, miss, dirty, and write
  amplification counters;
- separately paged payload history, because a correct automaton page cache does
  not prevent a random payload-successor fault.

Cold pages should be fixed-size blocks, not individually variable-length
records. Within a block, frame-of-reference and bit packing can compress
`max_length`, `suffix_link`, transition destinations, and edge successors while
retaining an offset table for bounded random access. Hot and dirty pages remain
uncompressed. Recompression occurs only after a page is sealed or its overlay
crosses a measured threshold.

The current link-cut latest-end tree was the main suspected paging hazard:
splay rotations mutate paths spanning unrelated state IDs. The production-core
trace in Section 8 shows strong locality on periodic64 despite many rotations,
so replacement is no longer assumed necessary. Large or adversarial trees
still require measured working-set and dirty-page behavior before promotion.

### 4.2 Online CDAWG or weighted compact automaton

A CDAWG compacts paths associated with maximal repeats and is a closer match to
the desired repetition-sensitive state bound than a standard SAM. The cited
online algorithm incrementally maintains an implicit CDAWG with suffix links,
implicit final-edge extensions, node cloning, edge splitting, and redirection.
For a constant alphabet it proves linear construction time. ROSA's 1--8 bit
alphabet satisfies that premise, but the result does not imply a sublinear
worst-case graph size.

A ROSA implementation still needs:

- an active implicit position `(node, edge, offset)` rather than only a node;
- exact suffix contraction from implicit positions;
- latest occurrence metadata for the represented substring class;
- update rules for edge splitting and metadata after every appended key;
- stable logical handles if pages are evicted.

An online symmetric CDAWG is also known and indexes the text and reverse text
together. It is the preferred second contraction experiment if a one-way
CDAWG cannot canonicalize the independent query locus efficiently. It still
does not supply latest-causal occurrence metadata.

The period-macro prototype is a weighted compact automaton of this kind: a
small control state plus an unbounded length counter. It demonstrates the right
kind of compression, but only for a known periodic language. The general task
is to discover and maintain equivalent compact paths without losing exact
fallback behavior when the stream later diverges.

The detailed operation audit and staged prototype plan are in
`docs/research/CDAWG_RUNTIME_ROUTE.md`.

### 4.3 Dynamic bidirectional RLBWT/r-index

Reversing the key and query turns a new query symbol into a backward-search
extension, which is attractive for a run-length BWT. A normal FM/r-index is
still insufficient: when extension fails, ROSA must delete the oldest symbol
from the active reversed pattern. That requires bidirectional contraction or
compressed-suffix-tree navigation. Latest-tie additionally requires a dynamic
range maximum over occurrence positions, and appending the original key means
prepending its reversal to the indexed text.

This route can provide `r`-sensitive total space on repetitive histories, but
its dynamic rank/select, contraction, and latest-position aggregate make it a
higher-risk implementation than a paged SAM. Published dynamic r-index
character insertion is also bounded by an LCP term, which is potentially
unfavorable on exactly the long-repeat histories ROSA targets. It should be
prototyped only after the four required operations above have standalone
oracle tests.

As of arXiv v4 (2025-10-29), the published bounds are `O(r)` words, count in
`O(m log r / log log r)`, locate in
`O(m log r / log log r + occ log r)`, and substring update in
`O((m + L_max) log n)`. None of these operations is the required dynamic
latest-position range maximum.

### 4.4 Grammar/LZ index

Grammar and LZ indexes can compress very repetitive text strongly, but most are
optimized for a static pattern supplied in full. Per-token online append,
active-pattern contraction, and latest occurrence are the difficult parts for
ROSA. They are candidates for immutable old storage only when a global exact
index also covers matches crossing arbitrarily many storage boundaries; they
are not the first all-online replacement.

Relevant starting points include the online CDAWG construction literature,
the repetition-aware CDAWG representation, and dynamic r-index work:

- https://doi.org/10.1016/j.dam.2004.04.012
- https://arxiv.org/abs/1705.08640
- https://doi.org/10.1109/SPIRE.2001.989743
- https://arxiv.org/abs/2504.19482
- https://arxiv.org/abs/2407.08826

These establish useful index primitives, not a ready-made ROSA runtime. The
latest-causal-occurrence and streaming query-state composition remain project
work.

## 5. Recommended architecture

Do not replace the production runtime yet. First preserve one logical global
automaton and make only its physical storage tiered:

```text
query symbol
    -> stable logical state ID
    -> resident hot record or exact cold-page decode
    -> global transition/suffix/latest operation across arbitrary pages
    -> exact payload-successor read
```

The hot arena and cold pages are not separate search indexes. They are physical
backends for the same stable-ID graph, so transitions and suffix links may cross
any number of pages without a boundary case. This is preferable to independent
epoch indexes: an epoch design needs a global exact bridge for matches crossing
arbitrarily many epochs, and a fixed overlap would silently reintroduce a hard
window. Independent compressed epochs remain a later option only if that
global bridge is proved exact.

The practical order is:

1. use an exact ideal-transition model to screen page sizes and expose
   state/edge/latest/payload ownership separately;
2. instrument the current C++ state/edge/latest/payload accesses once, record
   compact page IDs, and replay that trace offline for alternative caches;
3. measure cache misses and working sets on random, skewed, periodic, copied,
   and naturally colliding symbol streams at increasing lengths;
4. prototype stable-ID cold pages plus exact mutable overlays, initially with
   no compression;
5. add block packing and measure decompression bytes, overlay growth, and write
   amplification;
6. separately prototype the four ROSA operations on an online CDAWG or
   bidirectional RLBWT;
7. promote only if exact route parity, chunk invariance, and end-to-end latency
   beat the resident SAM in the intended memory regime.

## 6. Acceptance gates

Correctness gates:

- elementwise route and payload parity with the independent unlimited oracle;
- adversarial latest-tie, null, clone, suffix fallback, and page-boundary tests;
- matches that begin before one or many page/epoch boundaries;
- one-shot, random chunking, eviction/reload, and checkpoint/restore identity;
- exact behavior after a long compressed region diverges at one symbol.

Scaling gates:

- resident bytes, total bytes, and bytes per token per head;
- state, edge, latest metadata, and payload cache miss rates separately;
- p50/p99 storage reads per token and bytes decompressed per useful record;
- dirty-overlay size and page write amplification;
- latency under RAM-cache sizes substantially below total index size;
- per-head misses and cross-head/batch I/O coalescing, because one-head
  locality does not predict a many-head deployment;
- compression ratio against run count, maximal-repeat/CDAWG size, and an
  empirical grammar/LZ size proxy.

A candidate fails even with perfect output if it performs near-random storage
I/O per token. It also fails if good performance depends on never crossing a
fixed history boundary. The target is exact unlimited matching with a smaller
total or resident state, not bounded matching under a different name.

## 7. Initial page-locality screen

`benchmarks/runtime_pageability.py` implements the same unlimited online route
and latest-tie semantics as `RosaRuntime`. Fourteen focused tests cover random,
skewed, periodic, chunked, long-match, and unary cases; every route matched the
native runtime. The model uses ideal O(1) transition lookup and direct exact
suffix-path latest-end writes. Consequently, transition accesses are a useful
lower bound, while periodic latest-end traffic intentionally describes the
uncompressed path-update baseline rather than the production link-cut tree.

At 64K tokens with one head, 4 KiB pages, and one shared 1 MiB LRU:

| Pattern | Total logical bytes | Miss rate | Misses/token | Transfer upper bound | Cold writes |
| --- | ---: | ---: | ---: | ---: | ---: |
| random D8 | 2.61 MiB | 1.70% | 0.528 | 135.2 MiB | 31,810 |
| skewed D8 | 2.93 MiB | 1.27% | 0.565 | 144.6 MiB | 280,329 |
| periodic64 | 1.63 MiB | 0.39% | 2.070 | 529.9 MiB | 9 |
| copied, lag 1024 | 2.61 MiB | 0.88% | 0.256 | 65.7 MiB | 31,684 |
| repository text bytes | 1.89 MiB | 0.044% | 0.012 | 3.2 MiB | 28,579 |

Raw evidence is in `validation/runtime_pageability_8k.json` and
`validation/runtime_pageability_64k.json`.

The experiment rules out a naive `mmap` conversion. Even the ideal transition
model causes roughly one page miss every two tokens on random and skewed data
when resident memory is materially below total state. Large 64 KiB pages
amplify this into multi-gigabyte transfers because the cache holds too few
independent state, edge, and payload regions. Natural and copied streams have
substantially better locality, so paging remains viable as a structured-data
optimization.

The periodic result isolated the latest-end problem. Direct path propagation
performs 33.7M state writes at 64K; the production link-cut tree avoids those
logical path writes but replaces them with rotations. Section 8 now measures
the production tree directly and supersedes this lower-bound estimate.

These observations refine the pageable layout:

- use 4 KiB-class independently cached state, edge, latest, and payload blocks;
- test a charged sequential write buffer so append traffic does not evict
  query pages without discarding newly written hot pages;
- keep root/high-frequency transition data and the live construction frontier
  resident;
- store sealed-page mutations in an exact sparse overlay;
- give latest-occurrence maintenance its own representation and cache policy;
- record one compact C++ page-ID trace, then compare policies offline instead
  of maintaining many LRUs in the production hot loop.

No production paging mechanism is promoted by this screen. It establishes the
next implementation boundary without introducing a history horizon.

## 8. Production-core C++ page trace

`benchmarks/runtime_page_trace.py` and
`benchmarks/csrc/rosa_runtime_page_trace.cpp` now instantiate the exact same
C++ automaton core as production in a benchmark-only extension. Normal builds
compile every access marker to an empty statement and expose no new operator,
class method, schema, or runtime branch.

The trace covers:

- the 16-byte state array;
- all three physical edge SoA arrays;
- transition-index metadata;
- every independent hash/direct slots vector;
- the root direct table;
- LCT node construction, access, splay rotation, path assignment, and point
  query;
- payload append and successor reads.

It records canonical 4 KiB logical page IDs with token, region, owner,
read/write, operation kind, and run length. Larger pages and all cache policies
are mapped from the same immutable trace. Cache replay cannot change routes.
Seven focused tests prove native route/payload parity, transition-index and LCT
coverage, schema accounting, and multi-policy replay.

The logical layout deliberately gives every independent transition slots
vector its own owner namespace. This models a conservative stable-ID pageable
layout, not allocator virtual pages. Allocator relocation/copy traffic,
allocator metadata, input tensors, and output tensors are excluded. The
`std::vector` control object is included in transition metadata, while its
separately allocated slots use the owner namespace. Small slots vectors should
therefore be slab-packed in the next layout prototype rather than interpreted
as necessarily occupying one physical OS page today.

### 8.1 Cache-policy result

At 64K tokens, canonical 4 KiB pages, one head, and a shared 1 MiB cache:

| Pattern | Logical automaton | LRU misses/token | SLRU | Region LRU | Structural pin | Append bypass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random D8 | 2.80 MiB | 2.507 | 2.521 | 2.879 | 2.505 | 4.730 |
| skewed D8 | 2.92 MiB | 1.166 | 1.194 | 1.473 | 1.166 | 2.203 |
| periodic64 | 2.81 MiB | 0.011 | 0.011 | 0.011 | 0.011 | 1.100 |
| copied, lag 1024 | 2.80 MiB | 1.224 | 1.239 | 1.455 | 1.222 | 3.570 |
| repository text bytes | 1.85 MiB | 0.012 | 0.012 | 0.014 | 0.012 | 2.550 |

Raw evidence is in `validation/runtime_page_trace_8k.json` and
`validation/runtime_page_trace_64k.json`.

No replacement policy wins generally:

- SLRU saves some dirty evictions but has slightly more misses on the 64K
  random, skewed, and copied streams;
- fixed `55/30/15` core/LCT/payload partitions are consistently unsafe because
  an inactive or temporarily small region strands capacity;
- pinning six structural pages changes results only at the per-mille level;
- append no-admit is a failed negative control because newly appended state,
  edge, LCT, and payload pages are read again before becoming cold;
- ordinary write-allocate LRU remains the reference policy.

The dominant random-stream problem is layout. At 64K D8 the trace contains
256 transition-index owners, and transition slots account for 69,800 of the
164,306 LRU misses. Small independent slots allocations should be packed into
page-sized slabs or inlined into transition metadata. A cache algorithm cannot
recover the capacity lost to one-page-per-owner fragmentation.

The LCT result reverses the pessimistic conclusion from the direct-path model.
Periodic64 at 64K performs 8.08M LCT page touches and 342,749 rotations, but
the 1 MiB LRU has zero read misses: its 742 total misses are first writes while
the structure grows. High operation count does not imply a large active page
working set. LCT paging still needs longer and adversarial tests, but it is no
longer justified to replace the tree solely from the old 33.7M direct-write
estimate.

At 8K and 1 MiB, 4 KiB pages remain the robust default. Moving random D8 from
4 KiB to 16 KiB raises the LRU transfer upper bound from 10.9 MiB to 158.5 MiB;
at 64 KiB it reaches 736.7 MiB. Periodic data tolerates large pages, but a
global page-size choice must survive random and copied histories.

### 8.2 Next storage prototype

1. retain write-allocate LRU as the control;
2. slab-pack small transition slots and replay the same logical accesses;
3. add a charged sequential write buffer, rather than no-admit bypass;
4. allow dynamic borrowing between state/edge, latest, and payload pools;
5. model stable cold-page overlays for state/edge mutations;
6. rerun at capacities below total state and with adversarial LCT access before
   implementing actual page eviction.
