# Production V2, Initialization, and Hard Forward

This note records the 2026-08-21 production VJP and exact-hard-index
promotions and separates supported model initialization from research-only
ideas. Neither promotion changes hard semantics or dense backward support.

## 1. Production VJP Outcome

The public API and estimator are unchanged. Fixed-length CUDA backward now
selects internally:

| Gate | Backend |
| --- | --- |
| SM75 and `T >= 4096` | 32-row exact tiled streaming |
| SM80+, Q or K requested, `T >= 512` | 32-row exact tiled streaming |
| SM80+, value only, `T >= 2048` | 32-row exact tiled streaming |
| SM80+, `T >= 4096`, `W=32`, `Dv=64` | `64 x 64` block-diagonal TF32 |
| Otherwise | row-owned cache/recompute |

The block backend has no public selector. It uses the exact finite-window
suffix recurrence and complete causal route set; TF32 is limited to matrix
contractions. Scores, online softmax statistics, dropout reconstruction, and
suffix reverse state stay FP32. Streaming is the universal fallback.

Validation:

- 310 focused source, block, dispatch, dtype, dropout, mask, and pattern tests;
- 209 production/runtime/varlen/streaming tests passed, with one unsupported
  environment case skipped;
- 144 production-boundary gradient outputs had worst relative error
  `5.01e-4` against exact streaming;
- a `T=4096` one-bit hard-route repair reached zero loss at step 7 under both
  production TF32 and exact streaming;
- the real `T=4096` production gate reports zero memcheck errors, zero
  racecheck hazards, and zero synccheck errors under CUDA 12.8;
- SM75 retained its 4096 streaming boundary and never selected TF32;
- no quadratic persistent state was added; the 16K benchmark allocated
  26.5 MiB at the operator boundary.

End-to-end FP16 training latency for
`B=1,H=4,Hv=2,D=8,Dv=64,W=32`:

| T | Old production | Production v2 | Fixed SUFA | Flash d64 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2.43 ms | 0.84 ms | 3.48 ms | 0.46 ms |
| 1,024 | 8.71 ms | 2.30 ms | 3.51 ms | 0.46 ms |
| 2,048 | 33.12 ms | 7.04 ms | 7.61 ms | 0.45 ms |
| 4,096 | 24.16 ms | 20.20 ms | 22.74 ms | 1.13 ms |
| 8,192 | 96.51 ms | 72.48 ms | 80.68 ms | 3.79 ms |
| 16,384 | 372.83 ms | 280.89 ms | 309.13 ms | 13.44 ms |

Production v2 is faster than fixed SUFA at every measured length. Flash still
has a large long-sequence advantage because its core is a highly optimized
matrix attention kernel, while RosaSoft computes exact suffix structure and a
dense surrogate VJP.

## 2. Initialization Decisions

Initialization belongs to the model layer, not `rosa_soft`. The operator sees
only Q/K/V logits and must not silently initialize, normalize, or coordinate
their projections.

### 2.1 Retain

**Partial-shared orthogonal Q/K bases.** For each head, initialize half of the
Q/K rows from correlated orthogonal directions and the remaining rows from
independent directions. The current controlled point is `D=8`, four shared
rows, correlation `0.97`. This gives Q and K a common symbol convention while
leaving private bits available for specialization. Parameters are not tied
after initialization.

This is the strongest repeatable initialization result. In the cross-order
quadratic-proxy screen, default Q/K achieved mean validation token accuracy
`0.549` with minimum `0.148`; partial-shared Q/K plus orthogonal V8 achieved
`0.969` with minimum `0.828`. Under the production estimator, the same V8
configuration improved mean token accuracy from `0.248` to `0.832` at 1K
steps, and the two difficult tails reached mean `0.979` at 2K steps.

**Independent head rotations.** Each head must receive its own seeded
orthogonal basis. Sharing the correlation recipe is useful; copying complete
Q/K matrices across heads is not. Head diversity gives several independent
routes out of a poor discrete basin.

**Orthogonal binary V rows, normally V8.** Equal-norm, decorrelated V rows
make route utility more identifiable and reduce credit aliasing. V8 produced
about `0.992` unique-code fraction in the controlled task and was more robust
than V4. V16 did not clearly improve the seed tail enough to pay for its extra
model width. Keep binary V8 plus dense quadratic value credit as the default
research recipe unless the downstream payload requires another width.

**A static sign margin.** With LayerNorm-like inputs, scaling orthogonal rows
to median absolute logits around `0.6` gave stable signs without saturating the
softsign Jacobian. Retain the principle and use `0.6` as a starting value, not
as an operator constant. There is not yet a sufficient correlation-by-margin
factorial sweep to declare `0.6` or `0.97` universal.

**Multiple heads before more initializer machinery.** Raising the controlled
pretraining model from two to four heads improved production mean validation
token accuracy from `0.406` to `0.908`. This is architectural redundancy, not
an estimator trick, and was more reliable than adding specialized proxy
losses.

### 2.2 Do Not Retain as Defaults

**Paired V/output weights.** Pairing the output projection to V created a
shortcut and a severe tail: minimum validation token accuracy fell to `0.082`
in the controlled screen. Keep it only as a negative control.

**Fully aligned Q=K projection weights.** Exact alignment can make tiny
one-bit grammar probes start well, but it seeds synchronous fingerprint
matching, reduces head diversity, and encourages code collapse. Partial
alignment is the safer general recipe.

**Independent Q/K with no shared coordinates.** Setting shared-row
correlation to zero reduced mean validation accuracy from `0.728` to `0.600`
in the matched V4 control and substantially worsened the seed tail.

**Hand-planted recent/remote periodic routes.** They may make a diagnostic
task easier, but they also directly seed the low-entropy periodic structure
that makes hard forward expensive and can lock heads into simple fingerprints.
No shortcut-free pretraining evidence currently justifies them.

**Conflict-driven symbol growth.** Frozen conflict splitting is a useful
capacity diagnostic, but it uses externally identified conflicts and changes
model structure. It is not a from-scratch, uniform training initializer yet.

### 2.3 Minimal Recipe

For a new pretraining model:

1. Apply LayerNorm before Q/K/V projection.
2. Give each head an independently seeded orthogonal basis.
3. For `D=8`, correlate four Q/K rows at `rho=0.97`; leave four rows private.
4. Initialize V8 with independent orthogonal rows per value head.
5. Scale Q/K/V rows to median absolute logits near `0.6`; zero projection
   biases unless the surrounding architecture has a tested bias convention.
6. Do not tie Q to K or V to the output projection after initialization.
7. Log bit balance, unique-code fraction, per-head Q/K Hamming distance, null
   route fraction, and inter-head code correlation during early training.

These diagnostics are acceptance checks, not an adaptive scheduler. If an
architecture changes hidden normalization or residual scale, remeasure the
static gain rather than adding a runtime feedback controller.

The retained recipe is available as a model-layer utility:

```python
from rosa_soft.initialization import (
    RosaProjectionInit,
    initialize_rosa_projections_,
)

initialize_rosa_projections_(
    query_projection,
    key_projection,
    value_projection,
    RosaProjectionInit(
        num_heads=4,
        qk_bits=8,
        num_value_heads=2,
        value_bits=8,
        seed=17,
    ),
)
```

The helper writes ordinary `nn.Linear` parameters once. It adds no operator
state, parameter sharing, auxiliary loss, or runtime scheduler.

## 3. Unlimited Hard-Forward Bottleneck

The hard contract has no suffix window. For each query row it must compare
against every causally possible key ending until an exact proof identifies the
global longest suffix and latest tie. Independent D8 symbols usually mismatch
immediately, but aligned, periodic, or collapsed symbols can keep certificates
alive for a distance proportional to T. A one-warp-per-row implementation also
hits the resident-block limit before filling the SM with warps.

The exact byte-index baseline on RTX 3070 FP16 at
`B=1,H=4,Hv=2,D=8,Dv=64` showed both costs:

| Pattern | 16K | 64K |
| --- | ---: | ---: |
| random | 0.657 ms | 3.007 ms |
| shifted aligned | 1.369 ms | 15.398 ms |
| period 64 | 1.257 ms | 15.335 ms |
| period 4 | 1.115 ms | 15.045 ms |
| all match | 1.104 ms | 14.685 ms |
| all mismatch | 0.246 ms | 0.934 ms |

Entropy is not a sufficient dispatch statistic: all-match and all-mismatch
both have zero entropy but opposite suffix work. The production route therefore
uses fixed shape gates and exact certificates, not host-side content sampling.

## 4. Production Exact GPU Route

### 4.1 Deepest byte-sized suffix index

Sign packing writes the public int32 symbols and private uint8 Q/K codes in one
pass. For symbol width D, the index code contains
`G = max(1, floor(8 / D))` complete trailing symbols. Thus D1 indexes an exact
8-symbol suffix, D2 a 4-symbol suffix, D3/D4 a 2-symbol suffix, and D5-D8 one
symbol. Every code fits the same 256-bin table:

```text
offsets[B,H,257]          # occurrence-slice boundaries
occurrences[B,H,T]        # K end positions, ascending per code
first_position[B,H,256]   # narrow-code final-symbol null proof
```

One CTA per `(batch, head)` builds the stable occurrence lists in O(T) work.
For each query row, binary search returns only causal occurrences of its exact
suffix code. If no multi-symbol code exists, the final-symbol table either
proves null or sends the row to the complete causal fallback. This distinction
is required: a missing long code does not prove that the exact suffix length is
zero when its last symbol appeared earlier.

The fixed index gate is `D <= 4, T >= 1024` or `5 <= D <= 8, T >= 512`.
`D > 8` and packed-varlen hard forward retain the brute exact scan.

### 4.2 Warp certificate and ordered proof

Four independent warps share each 128-thread route CTA, and each warp owns one
query row. The newest code-equal candidate is tested first. The first 32 suffix
symbols use one symbol per lane; later iterations assign eight consecutive
symbols to each lane, so one ballot covers 256 ordered suffix positions.

An older route `a` cannot have suffix length greater than `a`. Therefore, once
the next older route is no larger than the best exact suffix length, the winner
is proved. Dense collision slices may test three more newest candidates before
that proof. If it still fails, lanes partition every remaining occurrence and
reduce `(length, latest_route)` exactly. These probes change only execution
order; there is no survivor cap, approximate pruning, or finite horizon.

### 4.3 Exact latest-candidate trajectory

At `T >= 4096`, let `p_i` be the newest causal K end matching row i's indexed
suffix code. Whenever `p_i = p_(i-1) + 1`, both compared endpoints advance by
one and the exact suffix certificate follows the recurrence

```text
L_i = min(L_s + i - s, i + 1, p_i + 1)
```

where s is the start of the current consecutive-candidate run and `L_s` is
computed once by the unlimited warp certificate. This is an exact DP reuse,
not a predicted length.

The implementation fuses latest-candidate lookup with a 256-row local
segmented prefix scan. A warp scans only the `T/256` tile carries, and the main
kernel resolves cross-tile starts lazily. Persistent base-certificate warps
then process only segment starts. The trajectory uses two `BHT` int32 work
arrays and does not synchronize with the host.

### 4.4 Workspace and invariants

The indexed path adds two `BHT` uint8 Q/K buffers, one `BHT` int32 occurrence
array, 257 int32 offsets per series, and the narrow-code first-position table.
The 4K trajectory adds two `BHT` int32 arrays. Returned packed int32 Q/K symbols
are part of the existing operator contract, not index-only workspace.

No hard kernel receives `max_suffix_length`. Every unresolved route can still
reach the full query and key boundaries, latest ties are exact, and the index
never changes the complete dense backward route set.

## 5. Implementation Outcome

Final RTX 3070 FP16 latency at `B=1,H=4,Hv=2,D=8,Dv=64`, compared with the
exact one-warp byte-index baseline:

| Pattern | 16K old | 16K new | 64K old | 64K new | 64K speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.657 | 0.594 | 3.007 | 2.166 | 1.39x |
| shifted aligned | 1.369 | 0.524 | 15.398 | 1.778 | 8.66x |
| period 64 | 1.257 | 1.165 | 15.335 | 9.045 | 1.70x |
| period 4 | 1.115 | 0.369 | 15.045 | 1.255 | 11.99x |
| all match | 1.104 | 0.362 | 14.685 | 1.242 | 11.83x |
| all mismatch | 0.246 | 0.245 | 0.934 | 0.877 | 1.06x |

Parity covers D1-D8, both index thresholds, the 4095/4096 trajectory boundary,
FP16/BF16/FP32, random/aligned/periodic/all-match/all-mismatch symbols,
multi-batch latest ties, and comparison with packed-varlen brute hard forward.
The focused hard/build suite passed 43 tests; CUDA/varlen/reference coverage
passed 195 tests with one environment skip. The final SM75 full suite passed
1816 tests and skipped 589 architecture-gated cases. Compute Sanitizer reported
zero memcheck errors, zero synccheck errors, and zero racecheck hazards.

The remaining hard case is discontinuous medium-period structure, represented
by period 64. Its trajectory repeatedly restarts at long base certificates.
Any future optimization must batch those exact restarts or improve the stable
index builder without imposing a candidate or suffix bound.

The canonical report is
`validation/unlimited_hard_optimized_final_sm86.json`.
