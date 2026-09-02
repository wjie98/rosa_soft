# CDAWG and Dynamic r-index for Exact ROSA Runtime

Status: literature audit and implementation boundary, 2026-08-22.

## 1. Decision

CDAWG is the highest-value compressed-automaton route to investigate in
parallel with the paged suffix automaton. It is not yet a production
replacement.

The reason is specific to ROSA rather than generic full-text indexing:

- an online CDAWG can append the key stream without first materializing a
  suffix tree or DAWG;
- its explicit nodes correspond to maximal repeats and its arcs to their
  extensions, so its total size can track repetition better than the current
  finite-text SAM;
- suffix links and implicit extensions already appear in the online
  construction algorithm;
- a CDAWG plus its reverse can represent suffix-tree navigation and matching
  statistics in repetition-sensitive space.

None of those results directly supplies ROSA's dynamic latest-causal
occurrence. Nor do the static suffix-tree representations directly supply the
online contraction of an arbitrary implicit query locus while the indexed text
is changing. Those are the two research problems that decide viability.

The dynamic r-index remains lower priority than both paged SAM and CDAWG. Its
`O(r)`-word space is attractive, but the current dynamic result has LCP-bounded
updates and exposes backward search/count/locate, not ROSA's right contraction
and latest-position aggregate.

## 2. What the cited results establish

### 2.1 Online CDAWG

Inenaga et al. construct the implicit CDAWG left to right, one character per
phase. The algorithm uses:

- variable-length edge labels represented as positions in the text;
- suffix links between strict factor classes;
- implicit extensions into the final node;
- node creation and cloning;
- edge splitting and redirection;
- one active endpoint carried from one phase to the next.

For a finite alphabet, the paper proves `O(n)` construction time and
`O(n * |Sigma|)` space with a transition matrix, or `O(n * |Sigma|)` time and
`O(n)` space with adjacency lists. Thus "online linear construction" assumes
constant-alphabet transition access. This is compatible with ROSA's 1--8 bit
symbols, but it does not prove a repetition-sensitive worst-case state bound;
the general upper bound remains linear.

The construction paper builds an *implicit* CDAWG during streaming. Converting
the final prefix to the conventional CDAWG uses an extra unique terminator
phase. A ROSA runtime cannot terminate and rebuild after every token, so its
production candidate must operate directly on the implicit online form.

Source:

- https://doi.org/10.1016/j.dam.2004.04.012
- open conference version: https://www.iss.is.tohoku.ac.jp/~ayumi/papers/CPM2001_cdawg.pdf

### 2.2 Repetition-aware CDAWG representation

Belazzougui and Cunial characterize the CDAWG as the minimal compact automaton
recognizing all suffixes. Except for the sink, explicit nodes correspond to
maximal repeats. A maximal repeat and a relative suffix length identify the
right-maximal strings compacted into the same class.

Their suffix-tree representation uses `O(e_T + e_reverse(T))` words, where
`e_T` is the number of CDAWG arcs. It supports important suffix-tree operations
between `O(1)` and `O(log log n)` and additional operations up to `O(log n)`.
The result is empirically sublinear on highly repetitive corpora, not a
universal sublinear bound. There are string families where CDAWG size and the
smallest grammar differ by `O(n / log n)`.

Later work obtains `O(m)` matching statistics and `O(m + occ)` locate time in
CDAWG-sized static structures. This proves that compact edge labels and
suffix-tree navigation need not force a scan proportional to uncompressed
text length. It does not make those structures dynamically updatable.

Sources:

- https://arxiv.org/abs/1705.08640
- https://arxiv.org/abs/1502.05937
- https://arxiv.org/abs/1707.08197

### 2.3 Symmetric CDAWG

An online SCDAWG jointly indexes a string and its reverse. This is relevant
because ROSA needs extension in one direction and contraction at the opposite
end after a failed extension. SCDAWG is therefore a better second prototype
than bolting ad-hoc reverse links onto a one-way CDAWG.

It is not free: both transition directions, edge-label addressing, and dynamic
metadata must be maintained. The source establishes online construction of the
dual index, not ROSA's latest-causal occurrence query.

Source: https://doi.org/10.1109/SPIRE.2001.989743

### 2.4 Dynamic r-index

The current arXiv version is v4, revised 2025-10-29 and titled *Dynamic
r-index: An Updatable Self-Index in LCP-bounded Time*. It provides:

- `O(r)` words of space;
- count in `O(m log r / log log r)`;
- locate in `O(m log r / log log r + occ log r)`;
- one-character update in `O((1 + L_max) log n)`;
- length-`m` insertion/deletion in `O((m + L_max) log n)`, with corresponding
  average bounds using `L_avg`.

The paper's update structure maintains RLBWT and sampled suffix-array
information. It does not claim a bidirectional active-pattern API or a dynamic
range maximum of occurrence positions.

Source: https://arxiv.org/abs/2504.19482

## 3. ROSA operation matrix

| ROSA operation | Online CDAWG | CDAWG suffix-tree representation | Dynamic r-index |
| --- | --- | --- | --- |
| append one key symbol | Directly supported by implicit online construction, amortized linear for constant alphabet | Static result; not an update algorithm | General insertion supported, but with LCP-bounded overhead |
| extend query by one symbol | Follow compact arc using first-symbol transition plus edge-label comparison | Static matching statistics support this | Backward search supports one extension direction |
| contract to longest suffix after failure | Suffix links exist, but an arbitrary query may be inside an arc; exact canonicalization is project work | Static suffix-tree suffix-link/navigation primitives help | Removing the opposite end needs bidirectional contraction or compressed suffix-tree support |
| return latest causal occurrence | Not supplied; must augment explicit and implicit loci | Static locate can enumerate occurrences, but enumeration is too expensive | Locate reports occurrences; no dynamic interval maximum is supplied |
| payload successor | Separate exact position-addressed store | Separate | Separate |
| stable pageable identity | Feasible with append-only IDs plus mutable edge overlay | Static IDs only | Dynamic balanced/RLE structures require their own persistence scheme |

The table is the reason CDAWG is promising but not ready. Three of the four
ROSA operations have a close published primitive. `latest_end` remains a new
dynamic aggregate, and contraction must be made explicit for an online
implicit locus.

## 4. Exact CDAWG state needed by ROSA

One head needs at least:

```text
text store:
    append-only key symbols used by compact edge labels

topology:
    stable node IDs
    stable edge IDs
    edge source/destination
    edge label begin and open/fixed end
    suffix link
    transition lookup by first symbol

construction cursor:
    active node/edge/offset
    current online phase and implicit-final-edge end

query cursor:
    node/edge/offset
    represented suffix length

occurrence metadata:
    latest end for every query-distinguishable locus
    exact update rules for clone, split, redirect, and suffix contraction

payload store:
    exact read at latest_end + 1
```

The construction and query cursors are distinct. Reusing the online builder's
active endpoint as the ROSA query state would conflate key topology updates
with an independent Q stream.

### 4.1 Implicit loci

A node ID alone is insufficient. A query often stops in the middle of a
compacted arc, so the state is `(edge, offset, matched_length)`. On mismatch,
the implementation must find the suffix of that implicit string and
canonicalize it to another explicit or implicit locus without decompressing
the whole label.

The online paper's suffix-link walk is evidence that this can be amortized for
its construction extensions. It is not a proof for ROSA's independent query
stream. The first prototype must instrument total edge-label characters and
suffix-link/canonicalization steps per query token.

### 4.2 Latest occurrence

This is the highest-risk component.

Every appended key creates a new occurrence of every suffix of the current
text. In the SAM, production implements the resulting suffix-link-tree path
assignment with a link-cut tree. A CDAWG reduces or merges topology, but it
does not automatically remove this aggregate update.

Three exact candidates should be tested:

1. store the rightmost suffix-tree leaf/rank represented by each CDAWG locus
   and update an online compressed suffix path;
2. maintain path assignment/point query on the CDAWG/SCDAWG class tree, using
   the same semantic oracle as the current LCT;
3. derive latest position from a dynamic lexicographic interval plus range
   maximum, provided interval updates and implicit-locus mapping stay
   repetition-sensitive.

Any claim that one `latest_end` per explicit CDAWG node is sufficient needs a
proof for all implicit positions and after every online edge split. Equal
occurrence counts within a compacted class do not by themselves establish the
required online update rule. Strict class equivalence may permit one aggregate
per explicit node, but every edge-interior locus and split still needs an exact
latest-causal-end proof.

## 5. Why dynamic r-index remains lower priority

Reverse the active query suffix. Appending a new query symbol then prepends a
symbol to the reversed pattern, which fits FM backward search. But when this
extension fails, ROSA drops the oldest original query symbol. In the reversed
pattern this removes the *last* symbol, which ordinary backward search cannot
do. A bidirectional index or suffix-tree contraction is still required.

Even after solving contraction, ROSA needs the greatest occurrence end in the
current BWT interval. Calling `locate` and taking the maximum is `Omega(occ)`
and fails on repeated patterns. The required augmentation is a dynamic range
maximum keyed by original positions, synchronized with RLBWT updates.

Finally, long repeated ROSA histories intentionally create large LCP values.
The dynamic r-index's LCP-bounded update is therefore not a harmless term for
this workload. This is a workload inference, not a contradiction of the
paper's empirical results.

## 6. Prototype order

1. Implement a small implicit online CDAWG over an 8-bit alphabet with stable
   IDs and no occurrence optimization.
2. Prove topology after every prefix against an offline suffix-set oracle,
   including edge labels, accepted suffixes, clones, splits, and redirects.
3. Add an independent Q cursor and exact longest-suffix route; compare every
   token to `RosaRuntime` while using exhaustive short strings.
4. Add latest occurrence first by a deliberately slow exact occurrence oracle.
5. Replace that oracle with each exact aggregate candidate and measure update
   traffic separately from transition traffic.
6. Add SCDAWG only if one-way implicit contraction is not both simple and
   amortized.
7. Measure nodes/arcs/label bytes against SAM states/edges, BWT run count, and
   grammar/LZ proxies on random, skewed, periodic, copied, and natural streams.
8. Add stable cold pages only after exact chunk invariance and route parity.

Do not start with a fully compressed label grammar, dynamic range structures,
and paging simultaneously. The first decisive experiment is whether an online
CDAWG can implement the four ROSA operations exactly with less topology and
bounded per-token canonicalization work.

## 7. Promotion gates

- exact route, latest-tie, null, payload-successor, and chunk parity;
- no history or suffix window;
- online updates; no terminator/rebuild after each token;
- explicit accounting for edge-label text storage;
- p50/p99 transition, canonicalization, and latest-aggregate work per token;
- total bytes and resident bytes materially below paged SAM on repetitive
  streams;
- no asymptotic regression on random streams beyond an explicitly accepted
  fallback boundary.

Until these gates pass, paged SAM remains the production-compatible route and
CDAWG remains a parallel compressed-topology experiment.
