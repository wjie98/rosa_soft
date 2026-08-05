# Filtered Bitflip Index Route v1

This document is the canonical map of the frozen first exact indexing route.
The development chronology and proofs remain in `FILTERED_BITFLIP.md` and
`FILTERED_BITFLIP_REPLACEMENT.md`; they are not descriptions of parallel
production paths.

The snapshot is CPU research code. It computes exact one-bit hard
counterfactuals for unlimited suffix matching and does not replace the frozen
dense-gradient `rosa_soft` operator. The canonical entry point is
`compact_filtered_bitflip` in `benchmarks/filtered_bitflip_compact.py`. The
semantic oracle is `brute_force_bitflip` in `benchmarks/filtered_bitflip.py`.

## 1. Exact contract

For query row `t` and route `r`, `1 <= r <= t`, the candidate endpoint is
`(Q[t], K[r-1])`. The winner maximizes `(suffix_length, route)`, so equal
positive lengths choose the latest route. Route zero is null. `Q[0]` and
`K[T-1]` are semantically inactive and are never flipped.

Every optimization in v1 preserves the complete hard result of flipping one
active Q/K bit. It may compress equivalent events, candidate exclusions,
winner intervals, or VJP accumulation. It may not sample bit coordinates,
truncate suffixes, or discard a candidate without a proof or an exact
fallback.

## 2. Current data flow

### Shared construction

1. Pack sign bits into byte codes outside this solver.
2. Build the compact suffix-node replacement index.
3. Build the exact suffix-array/LCP match index, optionally with pinned
   `libsais` construction.
4. Build contiguous occurrence lists for every center code.
5. Detect the exact finite languages supported by the uniform and shifted
   periodic solvers.
6. When shifted but nonperiodic, share the suffix-array pairs with the
   monotone-LCE geometry.
7. Compute the unedited longest/latest route once per query row.

### Per-bit state machine

Event preparation has one exclusive path:

1. Direct periodic event families, when the exact language proof applies.
2. Monotone query-create LCE cells, only for the proven shifted nonperiodic
   query path.
3. General exact create/break influence events for every other flip.

The prepared flip then has one exclusive solve path:

1. `empty`: no changed local match exists.
2. `periodic`: solve the complete affected interval with the supported
   uniform/shifted formula.
3. `certificate`: use bounded top candidates plus exact suffix fallback when
   interval work is sufficiently compressible.
4. `sparse`: update the affected rows and query the exact replacement suffix
   winner after excluding changed routes.

All paths emit affine or periodic route-change descriptors. The VJP consumes
those descriptors directly; a full `[number_of_flips, T]` counterfactual route
matrix is optional validation output, not required state. Tests assert that
the four dispatch counts partition every semantic flip exactly once.

## 3. Mechanism inventory

| Mechanism | Role and measured benefit | Cost or limitation | v1 status |
| --- | --- | --- | --- |
| Diagonal hard recurrence | Regular exact quadratic oracle. At `D=8`, it was `1.43x` faster than direct scan on random `T=1024`, `22.45x` on motif `T=1024`, and `273.9x` on collapse `T=1024`. | Still `O(T^2)` for one hard route evaluation and is not the bitflip scaling solution. | Retained oracle/control. |
| Code occurrence buckets | Enumerate only Hamming-0/1 center-code pairs. At `T=64,D=8`, retained work was about `0.95%` initially, `3.89%` for clear shifted-random, `13.89%` for clear motif, and `100%` for collapse. | No benefit at one-bit alphabets or code collapse; expected event count remains quadratic. | Always retained. |
| Exact create/break interval algebra | Converts one changed local pair into an exact diagonal influence interval and constant normalized priorities. | Materializing all events or rows can still be quadratic or worse in aggregate. | Semantic foundation. |
| SA/LCP/RMQ match index | Gives exact left/right LCE and enables suffix-node, periodic, and monotone proofs. In the frozen three-repeat profile, vendored `libsais` reduced suffix construction by `1.48x/1.87x/2.23x/3.19x` at `T=256/1024/4096/16384`. | Python suffix construction was not a timing winner in the early `T=64` matrix; native build is optional. | Retained shared index. |
| Compact suffix nodes | Exact `best_excluding` without storing every query-ancestor path. At `T=256`, logical storage fell from `36,556` to `16,408` bytes on independent D1 and from `269,280` to `9,660` bytes on collapse. | Individual exclusion queries can still become expensive; no universal subquadratic bound. | Retained replacement primitive. |
| Wavelet predecessor and symbolic arithmetic exclusions | Finds the latest allowed route in suffix-rank intervals and excludes periodic route sets without expanding them. | Adds static index complexity and is useful only behind suffix-node queries. | Retained internal primitive. |
| Sparse row update | Avoids certificate construction when exact event-cell work is already small. On independent `T=64,D=8`, compact took `27.58 ms` versus `33.21 ms` for the old hybrid (`1.20x`). | Explicitly visits affected rows; unsuitable for long overlapping events. | Retained fallback. |
| Bounded top-k certificate with exact suffix fallback | Keeps certificate storage bounded while preserving exactness for candidates outside top-k. | Full certificates were not universal winners: across the 18-case reference matrix, certificate variants won only 2 cases. Compact certificate construction is also expensive on fragmented independent data. | Retained only behind the structural work gate and built lazily. |
| Structural dispatch | Compares represented event-cell work with exact overlay-entry work. It changes execution only, never semantics. | `interval_compression` is an empirical cost ratio, not a complexity proof. | Retained with default `2`. |
| Exact periodic recognizers | Detect uniform, shifted-query, and sufficiently interior shifted-periodic-key cases. Entire row intervals are solved as priority envelopes. | Strict finite-boundary proof gates are required; early periodic key edits stay on the general path. | Retained. |
| Direct periodic event families | Builds phase-affine event families without materializing changed pairs. Collapse `T=64,D=8` materialized `0` of `32,256` conceptual events. Motif materialized `896` of `4,480`. | Applies only to exact finite periodic languages. | Retained. |
| Affine and periodic route ranges | Compress final changed routes and let VJP accumulate directly. On motif `T=64`, 1,560 affine-equivalent ranges were represented by 1,008 descriptors, including 384 periodic ranges. | Adversarial outputs can still require linearly many descriptors per flip. | Retained output contract. |
| Monotone suffix-rank LCE cells | Groups equal `(left_lce,right_lce)` query-create occurrences. At `T=128`, one representative retained `25.3%` of alphabet-4 occurrences, `18.4%` for motif-noise, `18.2%` for Thue-Morse, and `11.4%` for ruler. | Byte controls are mostly singleton cells. Key-side and break cells cannot use one global representative. The v1 Python path still scans selected occurrences. | Retained only under the shifted-query proof. |
| Dynamic orthogonal range tree | Removes explicit occurrence iteration by realizing the causal 3D prefix as an active 2D set. At `T=2048`, it sped up motif-noise/Thue-Morse/ruler by `1.13x/1.48x/1.87x`. | It slowed independent/random controls to `0.66x..0.75x` and uses `O(T log T)` logical storage. | Post-v1 experiment; not dispatched. |
| Static 3D kd index | Supplies exact arbitrary box count and position extrema in linear logical storage. It reached `1.25x` on ruler `T=2048`. | Usually slower than scan or dynamic range tree; worst-case query is linear. | Post-v1 reference; not dispatched. |

The compact execution comparison at `T=64,D=8` was:

| State | Old hybrid ms | v1 compact ms | Speedup | Materialized / conceptual events |
| --- | ---: | ---: | ---: | ---: |
| Independent | 33.21 | 27.58 | 1.20x | 308 / 308 |
| Shifted random | 245.41 | 170.03 | 1.44x | 691 / 1,254 |
| Shifted motif | 304.96 | 101.61 | 3.00x | 896 / 4,480 |
| Collapse | 807.86 | 23.86 | 33.85x | 0 / 32,256 |

These are three-repeat Python medians, not production throughput claims.
The hard-scan and occurrence measurements come from
`filtered_bitflip_cpu_scan.json` and `filtered_bitflip_profile.json`; compact
dispatch and storage come from `filtered_bitflip_compact.json`; monotone and
orthogonal measurements come from their same-named validation JSON files.

## 4. Exact baselines retained outside v1

The following modules remain because they independently verify semantics and
expose failure modes. They are not alternative branches inside the current
solver:

| Module | Purpose |
| --- | --- |
| `filtered_bitflip.py` | Full rerun oracle, event oracle, diagonal hard route, and route-range reconstruction. |
| `filtered_bitflip_winner.py` | First-generation RLE and segment-tree base-winner filters. |
| `filtered_bitflip_certificates.py` | Full replacement-certificate solver and shared periodic-family algebra. |
| `filtered_bitflip_suffix_nodes.py` | Posting-based suffix-node solver and arithmetic-run reference. |
| `filtered_bitflip_hybrid.py` | Pre-v1 structural dispatcher used for timing comparisons. |

The retained ablations are intentionally not hidden inside
`compact_filtered_bitflip`. A new index route should compare with them through
profiles rather than adding another switch to the frozen v1 dispatcher.

## 5. Rejected or noncanonical mechanisms

- Rolling hash: not collision-free and lost the measured event-discovery
  timings.
- Dyadic canonical ranks: exact but `O(T log T)` persistent storage; useful
  only as a short-context ablation.
- Winner segment tree: exact but RLE won 42 of 45 early cases and used less
  traversal/storage.
- Winner filtering alone: admitted about `10.25x` more rows than final route
  changes at full clarity.
- Full event matrices, candidate-length matrices, and counterfactual route
  matrices: exact but defeat the scaling objective.
- Universal full certificates: no stable timing winner and expensive to build
  on fragmented inputs.
- Recursive orthogonal box enumeration: exact but issued thousands of
  rectangle queries before reaching occupied cells.
- Unconditional dynamic range-tree dispatch: structured wins do not offset
  random-data regressions or `O(T log T)` storage.
- One representative for key-side or break cells: semantically invalid.
- Entropy-, context-length-, or alphabet-only dispatch: these statistics do
  not predict LCE-cell fragmentation.

## 6. Module map

Current v1 implementation:

- `filtered_bitflip.py`: semantics and oracle;
- `filtered_bitflip_indexes.py`: exact suffix-array/LCP support;
- `filtered_bitflip_compact.py`: compact index, state machine, route ranges,
  and VJP;
- `filtered_bitflip_periodic.py`: periodic proofs, direct families, and
  interval solvers;
- `filtered_bitflip_monotone.py`: exact nonperiodic LCE cells;
- `filtered_bitflip_native.py` and `third_party/libsais`: optional native
  suffix construction.

Post-v1 index study:

- `filtered_bitflip_orthogonal.py`;
- `filtered_bitflip_orthogonal_kd.py`;
- `filtered_bitflip_orthogonal_profile.py`.

Files ending in `_profile.py`, tests under `tests/test_filtered_bitflip*`, and
JSON under `validation/filtered_bitflip*` are reproducibility surfaces, not
solver dependencies.

The dependency direction is acyclic:

```text
oracle
  -> exact indexes
  -> legacy baselines
  -> periodic + monotone structures
  -> compact v1 solver
  -> orthogonal post-v1 study
```

## 7. Frozen controls

The v1 solver exposes two execution-tuning controls:

- `top_k=4`: retained candidates per compact certificate node;
- `interval_compression=2`: required event-cell/overlay-work ratio before
  certificate dispatch.

Both preserve exactness because omitted top-k candidates use the suffix
fallback and the dispatch alternatives are exact. `suffix_backend` selects
Python or pinned `libsais` suffix construction. `match_index` is dependency
injection for tests. `materialize_routes` is validation-only output. `value`
and `grad_output` request the compressed exact bitflip VJP.

Do not add another index backend or data-dependent heuristic to this function
after the `filtered-bitflip-index-v1` tag. A new route should have a separate
entry point and must compare exact routes, lengths, VJP, memory, and dispatch
work against this snapshot.

## 8. Validation and reproduction

The test matrix includes exhaustive binary `D=1,T=4` states, random
`D=1,2,4,8` inputs, degenerate and periodic languages, exact finite-boundary
counterexamples, compact VJP parity, optional native `libsais`, monotone cells,
and arbitrary orthogonal boxes.

The freeze run with `CUDA_VISIBLE_DEVICES=1` passed `997` tests. Two unrelated
multi-GPU tests were skipped because only one CUDA device was exposed; the
optional native `libsais` tests compiled and ran.

```bash
CUDA_VISIBLE_DEVICES=1 python -m pytest -q

python benchmarks/filtered_bitflip_compact_profile.py \
  --build-native --sequence-lengths 64 128 256 \
  --execution-length 64 --top-k 4 8 \
  --json-out validation/filtered_bitflip_compact.json

python benchmarks/filtered_bitflip_monotone_profile.py \
  --sequence-lengths 64 128 --repeats 3 \
  --json-out validation/filtered_bitflip_monotone.json

python benchmarks/filtered_bitflip_orthogonal_profile.py \
  --sequence-lengths 128 256 512 1024 --repeats 3 \
  --json-out validation/filtered_bitflip_orthogonal.json
```

The annotated Git tag `filtered-bitflip-index-v1` identifies the frozen code
and validation snapshot.
