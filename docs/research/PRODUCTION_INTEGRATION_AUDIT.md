# RosaSoft Production Integration Audit

Status: implementation and evidence audit, 2026-08-21.

Historical note (2026-09-05): the absolute latency tables below describe the
earlier finite-window production implementation, not the subsequently
promoted unbounded slab-replay VJP. Some intervening GPU1 runs were affected
by unrelated load. Current unbounded decisions use only the idle-card,
interleaved recheck recorded in `PERSISTENT_MACRO_TILE_VJP.md`.

2026-09-06 update: eligible fixed-length FP16/Dv64 unbounded workloads now use
the exact occupancy-bounded grouped-checkpoint specialization documented in
`GROUPED_CHECKPOINT_VJP.md`. The estimator is unchanged; other shapes and
packed varlen retain exact slab replay. Sections below remain the estimator
and finite-kernel decision record rather than a current dispatch inventory.

Post-audit outcome: the architecture-aware streaming crossover and ordinary
`64 x 64` block-TF32 plan were promoted later the same day. The estimator and
public API did not change. Treat the candidate rankings below as the evidence
at decision time; current dispatch and post-promotion measurements are in
[Production V2, Initialization, and Hard Forward](PRODUCTION_V2_INITIALIZATION_AND_HARD_FORWARD.md).

This document separates four things that must not be conflated:

1. the estimator and execution mechanisms already shipped by `rosa_soft`;
2. research results that are strong enough to justify a production gate;
3. useful diagnostics and model-level ideas that should remain outside the
   operator;
4. measured negative results that should not be reintroduced without new
   evidence.

The current production contract remains exact hard forward plus a complete
dense surrogate VJP. No recommendation below removes a causal candidate,
samples the gradient support, or stores a persistent `T x T` tensor.

## 1. Current Production Semantics

| Mechanism | Purpose | Evidence and decision |
| --- | --- | --- |
| Binary hard Q/K/V forward | Prevents a model from exploiting soft values or score magnitude during training | Required. Hard forward is the deployment operation. |
| Latest-longest suffix winner, latest tie | Implements the discrete ROSA route | Required and bit-exact across dtypes. |
| Softsign VJP at hard signs | Gives continuous logits a bounded local pseudo-derivative | Retain. This is a declared surrogate, not the mathematical derivative of the hard route. |
| Normalized Hamming mismatch `h / D` | Makes local mismatch geometry stable across symbol width | Retain. `h / sqrt(D)` collapsed Q/K gradient scale at larger D. |
| Deterministic exponential local gate | Gives every local mismatch a nonzero dense discovery path | Retain. It removed stochastic mismatch variance and is GPU-friendly. |
| Complete suffix prefix-product sum `S` | Credits every suffix length up to W without selecting a soft suffix tier | Retain. No early exit or candidate pruning is allowed in backward. |
| `U(S)=(sqrt(2)+1)(sqrt(1+S)-1)` | Compresses the score advantage of already-long matches while remaining increasing and unbounded | Retain. It passed all directed long/short cells where raw S failed at long distractor lengths. |
| Fixed `null_score=0.5` | Provides one differentiable no-retrieval competitor | Retain. Dynamic null policies did not establish a joint route-creation and null-retention gain. |
| `-log(nonnull_count)` | Makes non-null competition an average candidate weight instead of a context-length-growing sum | Required for long candidate spaces. |
| Static `scale` | Standard inverse attention-temperature control | Retain as an explicit static argument. Do not infer it from D, T, W, or training step. |
| Static `mismatch_scale` | Controls near-match leakage and local Q/K credit | Retain as an explicit static argument. No universal D/T/W schedule was found. |
| Optional post-softmax inverted dropout | Adds standard attention-style route exploration without leaking a soft forward | Retain, default zero. It was the only estimator addition with a small contextual convergence gain. |
| Dense value carrier in backward | Lets losing routes receive task-directed value and route credit | Retain. Selected-only V credit was less stable. |

The semantic surface is intentionally small: `max_suffix_length`, `scale`,
`dropout_p`, and `mismatch_scale`. The fixed utility, null score, candidate
prior, quantizer VJP, and tie rule are not public tuning axes.

## 2. Current Production Execution

The following are execution optimizations of the same estimator and are worth
keeping:

| Mechanism | Current role |
| --- | --- |
| Packed head-major Q/K sign words | One XOR and popcount evaluates a complete symbol up to D=32. |
| Saved packed symbols | Avoids quantizing Q/K again in backward with only linear saved state. |
| Three-bit requested-gradient mask | Skips allocations and work for unused Q, K, or V gradients. |
| FP32 internal accumulation and one final softsign pass | Keeps long reductions stable and removes repeated Jacobian work from hot loops. |
| Row cache/recompute planner | Keeps short and packed-varlen shapes viable when a full tiled CTA is not profitable. |
| Head-major K adjoints and local K aggregation | Coalesces global updates and reduces atomics on gated low-D shapes. |
| Cooperative utility and value tiles | Removes avoidable Dv serialization without changing candidate support. |
| Counter-based dropout replay | Saves one scalar seed instead of a quadratic mask. |
| Two-pass tiled streaming VJP | Computes exact online softmax statistics, then fuses probability, utility, Q/K credit, and V credit with no quadratic workspace. |
| Local match-gate adjoint aggregation | Shares Q/K contractions among suffix terms that reach the same exact local gate. |
| Direct tiny-suffix microtile | Avoids aggregation setup when a suffix chunk contains too little reusable work. |

Fixed-length production currently dispatches the streaming VJP only for
`T >= 4096`. Shorter inputs use the row-owned VJP. Packed variable-length
inputs still use their separate row schedule.

## 3. Estimator Experiments

### 3.1 Retained or useful

| Experiment | Result | Production action |
| --- | --- | --- |
| Deterministic mismatch reduction | Lower variance and no held-out fitting loss versus mismatch jitter | Already production. |
| Standard route dropout | Primary contextual gate improved median first-exact step from 92.5 to 85 while preserving 4/4 success | Already optional production control; keep default at zero. |
| Static temperature ladder | Lower temperature accelerated finite-budget repair but did not change eventual reachable suffix length | Keep only the static `scale` argument; recipe tuning remains external. |
| Static mismatch-scale calibration | Distance to the current match, not D alone, determined the useful scale | Keep one static argument; do not schedule from shape. |
| Square-root suffix utility | Passed 18/18 directed aggregate cells versus 12/18 for raw S and did not lose broader fitting cells | Already production. |
| Candidate-count correction and fixed null | Prevented random-background non-null mass from growing mechanically with N | Already production. |
| Collision-likelihood ratio | Good theoretical diagnostic and 35/48 broad fitting cells | Research only because it failed the directed long-route gradient gate. |
| Mean-field winner model | Strong tiny-oracle cosine and deterministic | Keep as semantic diagnostic; repeated-key correlation assigns mass to impossible routes. |
| Exact global-bit oracle | Defines a coherent stochastic hard objective on tiny problems | Keep as a judge, not a scalable estimator. |
| Exact margin edit | Solves coordinated tiny edits and is a strong structured diagnostic | Keep as an oracle; exhaustive state search and loss-scale-dependent eta block production. |

### 3.2 Rejected as production estimators

| Experiment family | Why it is not promoted |
| --- | --- |
| Independent per-mismatch perturbation | Added estimator variance and RNG work without a held-out fitting advantage. |
| Antithetic mismatch branch | Reduced already-small local variance but doubled branch work and did not pass the paired fitting gate. |
| Linear/quadratic/cubic random mismatch shapes | Differences were tied to the removed perturbation mechanism; no stable general winner justified an API axis. |
| Dynamic temperature, mismatch scale, or proxy lambda | No robust law over D/T/W/current distance was found; shape-only schedules can suppress the very long-match discovery they try to help. |
| Hard-tier wrapper and separate numerical/derivative gates | Produced sharply decaying or internally inconsistent credit and duplicated state. |
| Residual suffix dropout and high-dropout sampling variants | Did not provide a stable quality gain. Dropout occurs after normalization, so it cannot skip the score pass without changing the estimator. |
| Inverse-probability or low-probability-biased sampling | Raises variance on rare routes and changes support when the sampled branch is detached. It is especially poorly matched to discontinuous suffix winners. |
| ARM and DisARM | Correct for the declared stochastic-bit objective, but single-sample normalized variance was 46.0 and 34.7; extra hard evaluations did not beat production contextual convergence. |
| Complete bitflip as the training target | Can be exactly zero when a useful suffix requires coordinated edits. On the joint-suffix gate it passed 0/16 while production passed 16/16. |
| Sampled bitflip residual correction | More accurate sampling converged toward the zero complete-bitflip field and made the coordinated task worse. |
| Fixed-window historical suffix attention | Fast, but it is a different decayed `D*W` fingerprint estimator and omits current null, per-length geometry, Hamming gates, and candidate calibration. Keep only as a baseline. |

No alternative estimator clears the production replacement gate. The present
dense deterministic VJP remains the best default combination of discovery
support, variance, fitting reliability, and implementation regularity.

## 4. Bitflip Index Routes

Two exact unlimited-suffix lines were implemented and should remain research
infrastructure:

| Route | Useful mechanisms | Limitation and decision |
| --- | --- | --- |
| Filtered bitflip v1 | Packed code buckets, exact create/break interval algebra, SA/LCP and suffix-node indexes, periodic certificates, compressed winner envelopes, affine-range VJP | Excellent filtering for healthy D=8 languages and exact common-case batching, but no subquadratic adversarial bound; collapse can recover quadratic work. Keep as an exact reference. |
| Full-endpos SAM bitflip | Dual suffix automata, explicit/implicit endpos predecessor, exact Q replay coalescence, anchored virtual K runs, arithmetic-progression certificates, factorized descriptors and CUDA contraction | Removes repeated reruns and supports unbounded suffixes, but full occurrences and active edit families can still be quadratic. Keep as an independent oracle and data-structure testbed. |

These indexes are useful for offline diagnostics, tiny exact-gradient studies,
and possible future CPU research. They should not replace the dense discovery
VJP. Exact hard-forward indexing may be reconsidered separately because it
does not determine backward support.

## 5. Learned and Linear-State Proxy Routes

| Route | Positive evidence | Blocking evidence | Decision |
| --- | --- | --- | --- |
| Exact fixed suffix feature | Supported the one-head/one-short-pattern hypothesis and reached 3/4 fitting seeds | Feature rank grows as `2^(D*L)` | Oracle only. |
| TensorSketch suffix feature | Reduced one useful state by 16x and passed 3/4 on one setting | Non-monotone width curve, hash-seed sensitivity, and long-product gradient failures | Demote. |
| Current-symbol quadratic state attention | Small 37-feature state, 7/8 contextual passes, useful two-head next-token result | Advantage disappeared at four heads; oldest decisive suffix bit gets exactly zero gradient for W>1; prototype is slower | Retain as a model research candidate, not an operator path. |
| Temporal quadratic proxy | Linear-time two-pass construction and no trainable proxy parameters | Failed W>1 long-suffix credit and extrapolation gates | Reject as production. |
| Exact aligned suffix state | Reproduces raw suffix evidence and local VJP | Exact rank is exponential; fixed-rank sketches lose long balanced-value credit | Mathematical oracle only. |
| Delta-rule memory | Exact overwrite behavior for repeated unit keys | Overwrite hurt coexistence of several associations | Reject; additive normalized memory is the useful research control. |
| Float, uniform k-bit, exponent, or RMS V proxy | Tests whether richer V gives finer credit | Binary V8 with dense credit was more accurate and seed-stable | Keep binary dense-credit research default. |

The controlled initializer with partially shared Q/K geometry and orthogonal V
is a useful opt-in model recipe. It is not an operator parameter and did not
establish a universal pretraining default.

Stateful internal-language experiments also stay at the model layer. A slow
state made length-2 grammars trainable, length-4 seed-sensitive, and length-8
unsuccessful. Conflict-driven dormant-bit growth proved that a codebook can be
expanded while frozen, but simultaneous language-model gradients destroyed
Q/K alignment. These results motivate codebook diagnostics and staged model
training, not more RosaSoft kernel controls.

## 6. Kernel Experiments

### 6.1 Strong production candidates

| Candidate | Current sm_86 evidence | Integration decision |
| --- | --- | --- |
| Lower fixed-length streaming crossover | Existing exact kernel is already faster than row VJP at much shorter T for almost every tested mask/shape | Highest priority. Add an internal architecture, mask, and shape dispatch; no ABI or estimator change. |
| `64 x 64` block-diagonal exact recurrence plus TF32 contractions | FP16 all-mask matrix: 28/28 wins, mean ratio 0.776, worst 0.886; QKV ratio 0.809/0.765 for D=8 at 4K/8K and 0.700/0.658 for D=32 | Promote next behind an sm80+, `W=32,Dv=64`, long-sequence gate after model-fitting parity and cross-architecture review. Keep streaming fallback. |
| D=32 gate-adjoint HMMA | About 0.927 mean ratio; QK/QKV about 0.88..0.90 | Retain only as a fallback candidate for block-unsupported Dv/W shapes. Do not add beside the block path without a dispatch win. |

### 6.2 Useful but not ready

| Candidate | Decision |
| --- | --- |
| Named-barrier block pipeline | The newest FP16 matrix improves mean production ratio from 0.776 to 0.768, only about one additional percentage point, while shared memory rises from 69.1 to 85.1 KiB. Do not bundle into first promotion. |
| Exact unlimited hard diagonal DP | Promoted for T>=512. Random T=4096 adds about 0.08 ms, while periodic/all-equal inputs improve by over 160x; `O(BHT)` winner state and exact SAM parity are retained. |
| Warp/diagonal prefix scan primitive | Useful in isolated diagonal ownership and for a selected hard route. Keep as a building block, not dense-route ownership. |

### 6.3 Measured negative kernel variants

| Variant | Result |
| --- | --- |
| Warp per route candidate | Roughly 5x..16x slower because 32 lanes service one candidate instead of 32 candidates. |
| Sparse-tail warp handoff | Isolated tiny tails sometimes won, but the integrated VJP ratio was 0.9996 and register count increased. |
| `32 x 32` block tile | All 28 long Dv=64 cases were about 4%..25% slower than `64 x 64`. |
| Scalar block recurrence without matrix contractions | Mean ratio about 4.05 and worst about 5.93; recurrence reuse alone is not enough. |
| Tensor Core local Hamming gate | Packed scalar XOR/POPC plus EXP remained faster than FP16, INT8, and B1 Tensor Core paths. |
| Tensor Core utility or dV in the existing 32-row layout | Isolated GEMMs won, but conversion, barriers, and ownership changes made integrated latency neutral or worse. |
| Full Tensor Core plus warp-scan plan | Numerically correct but averaged 1.217x production latency. |
| Rolling diagonal recurrence in the row-owned layout | Serialized diagonals and regressed QKV by 15%..32% for W<=32. |
| First-pass value-gradient carrier | Won only 2%..5% at Dv=128, was neutral at Dv=32/64, and regressed Dv=8. |
| Compile-time mask classes | Small Q/K-only gain, no QKV gain, and about 81% binary growth. |
| Candidate-correction precompute and mismatch LUT | At most low-single-percent noise with sign changes across shapes. |
| Sequential TF32 high/low fragments | NVCC already reused registers; aggregate change stayed below one percent with regressions. |

## 7. Current Speed Comparison

### 7.1 Measurement contract

The current same-process report is
`validation/attention_speed_production_sm86.json`:

- idle RTX 3070, `sm_86`, PyTorch 2.11 + CUDA 12.8;
- FP16, `B=1,Hq=4,Hv=2,D=8,Dv=64,W=32`, random inputs;
- complete forward plus Q/K/V backward, excluding learned projections;
- rotating benchmark order, five rounds, three warmups, seven repeated calls;
- production RosaSoft as currently dispatched;
- historical fixed SUFA with the same exact hard ROSA forward and precomputed
  end positions;
- PyTorch SDPA controls explicitly forced to `FLASH_ATTENTION`.

FlashAttention requires equal Q/K/V head width. Therefore no single control
exactly matches RosaSoft's `D=8,Dv=64` shape:

- Flash d8 matches Q/K width but returns only eight value features;
- Flash d64 matches output/value width but uses 64-wide Q/K.

Fixed SUFA unfolds Q/K to `D*W=256`, keeps V at 64, and selected PyTorch's
memory-efficient attention backend rather than FlashAttention. It is the
closest historical suffix estimator, but not the same gradient.

### 7.2 Complete training step

Ratios are `RosaSoft / comparison`; lower than one means RosaSoft is faster.

| T | RosaSoft ms | Fixed SUFA ms | ROSA/SUFA | Flash d8 ms | ROSA/Flash d8 | Flash d64 ms | ROSA/Flash d64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2.431 | 3.430 | 0.709 | 0.391 | 6.22 | 0.398 | 6.11 |
| 1,024 | 8.707 | 3.544 | 2.457 | 0.421 | 20.66 | 0.398 | 21.89 |
| 2,048 | 33.118 | 7.552 | 4.386 | 0.463 | 71.61 | 0.449 | 73.69 |
| 4,096 | 24.160 | 22.382 | 1.079 | 0.631 | 38.29 | 1.137 | 21.26 |
| 8,192 | 96.511 | 82.167 | 1.175 | 2.141 | 45.07 | 3.795 | 25.43 |
| 16,384 | 372.828 | 312.854 | 1.192 | 7.521 | 49.57 | 13.614 | 27.39 |

The discontinuity between 2K and 4K is a dispatch artifact: 2K still uses the
row VJP, while 4K uses the faster streaming VJP. Current long-sequence
RosaSoft is 8%..19% slower than fixed SUFA on this GPU, not 5x..6x slower as
the old row implementation was. On the earlier RTX 2080 Ti FP32 calibration,
the same production streaming design was only 2.5%..4.4% slower than fixed
SUFA at 4K..16K. The exact gap is backend, dtype, and architecture dependent.

Against output-width-matched Flash d64, current long training is about
21x..27x slower. Against narrow Flash d8 it is about 38x..50x slower at
4K..16K, but that control computes and returns one eighth as many value
features. This gap is algorithmic: both scan all causal candidates, while
RosaSoft additionally reconstructs a length-W nonlinear suffix state and its
dense Q/K adjoint for each candidate.

### 7.3 Memory and hard forward

| T | ROSA MiB | Fixed SUFA MiB | Flash d64 MiB | ROSA/SUFA | ROSA/Flash d64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.83 | 17.96 | 2.27 | 0.046 | 0.365 |
| 1,024 | 1.66 | 35.91 | 4.58 | 0.046 | 0.362 |
| 2,048 | 3.31 | 71.82 | 9.06 | 0.046 | 0.366 |
| 4,096 | 6.62 | 143.64 | 18.13 | 0.046 | 0.366 |
| 8,192 | 13.25 | 287.28 | 36.25 | 0.046 | 0.366 |
| 16,384 | 26.50 | 574.56 | 72.50 | 0.046 | 0.366 |

RosaSoft uses about 21.7x less incremental memory than fixed SUFA. It also
uses about 2.7x less than Flash d64 in this benchmark, but part of that result
comes from eight-wide ROSA Q/K versus 64-wide Flash Q/K.

For random codes, hard RosaSoft forward takes 0.198/0.647/2.587 ms at
4K/8K/16K. It is faster than Flash d64 forward by about 1.6x..1.8x and near
the narrow Flash d8 control. Collapsed all-match codes are a separate worst
case: hard forward measured 3.015/10.444/41.572 ms. This does not justify
changing the dense backward, but it keeps a pattern-gated exact hard index on
the long-term optimization list.

## 8. Dispatch Opportunity Already Available

The direct crossover benchmark forces the already compiled 32-row streaming
kernel below the current 4096 threshold and compares it with public production
VJP. Values below are means over random and all-match inputs:

| Shape | Gradient mask | T=512 | T=1,024 | T=2,048 |
| --- | --- | ---: | ---: | ---: |
| `D=1,Dv=8,W=1` | QKV | 0.897 | 0.799 | 0.673 |
| `D=8,Dv=64,W=32` | QKV | 0.362 | 0.230 | 0.199 |
| `D=8,Dv=8,W=32` | QKV | 0.227 | 0.197 | 0.168 |
| `D=8,Dv=128,W=128` | QKV | 0.245 | 0.199 | 0.185 |
| `D=32,Dv=64,W=32` | QKV | 0.153 | 0.126 | 0.108 |
| `D=8,Dv=64,W=32` | V only | 1.024 | 0.906 | 0.766 |
| `D=8,Dv=128,W=128` | V only | 1.326 | 1.133 | 0.927 |
| `D=32,Dv=64,W=32` | V only | 1.149 | 0.911 | 0.763 |

This is the highest-confidence immediate integration:

1. on sm80+, start with streaming at `T>=512` whenever Q or K gradients are
   requested;
2. use a conservative `T>=2048` threshold for V-only until a wider Dv/W
   matrix supports a lower boundary;
3. retain the current sm75 threshold until the same mask/shape matrix is run;
4. keep packed variable-length dispatch unchanged until it has its own tiled
   implementation;
5. encode the result as a small internal capability/mask table, not a public
   parameter or runtime autotuner.

For the default QKV shape, combining the measured forced-streaming VJP with
the measured hard forward gives an estimated complete step of roughly
0.73/2.03/6.59 ms at 512/1K/2K, versus current 2.43/8.71/33.12 ms. This is an
estimate across two matched kernel harnesses; the final dispatch change needs
one direct end-to-end confirmation.

## 9. Recommended Integration Order

1. **Recalibrate fixed-length streaming dispatch.** This uses shipped code,
   preserves exact arithmetic structure, and fixes the largest current
   performance anomaly with almost no maintenance cost.
2. **Promote block-diagonal TF32 behind a strict internal gate.** Start with
   sm80+, `W=32,Dv=64,T>=4096`; keep current streaming as the universal
   fallback. Require long fitting parity, Ada/Hopper measurements, and binary
   size/resource review first.
3. **Do not initially promote the named-barrier pipeline.** Its incremental
   FP16 gain over the ordinary block plan is too small for the extra 16 KiB of
   shared memory and synchronization complexity.
4. **Retain D32 gate-only HMMA as a fallback experiment.** Re-evaluate only on
   shapes that cannot use the block plan.
5. **Profile collapsed-code hard forward in real checkpoints.** Build a
   code-density dispatch only if collapsed or low-entropy heads consume a
   material fraction of full training time.
6. **Leave estimator alternatives, bitflip, state attention, and symbol
   growth outside production.** Their useful role is diagnosis and model
   research; none currently beats the dense deterministic VJP across the
   required gates.

The production optimization target is therefore narrow: improve scheduling
of the exact dense estimator. There is no current evidence for adding another
estimator mode, random branch, adaptive scalar, sampled support, or proxy
model to the public operator.

## 10. Reproduction Artifacts

- `validation/attention_speed_production_sm86.json`: current production,
  fixed SUFA, and forced FlashAttention comparison.
- `validation/streaming_vjp_crossover_sm86_fp16.json`: default all-mask
  fixed-length crossover.
- `validation/streaming_vjp_crossover_sm86_minimal_fp16.json`: minimal
  `D=1,Dv=8,W=1` control.
- `validation/streaming_vjp_crossover_sm86_d8_dv8_w32_fp16.json`: narrow V.
- `validation/streaming_vjp_crossover_sm86_d8_dv128_w128_fp16.json`: wide V
  and long suffix horizon.
- `validation/streaming_vjp_crossover_sm86_d32_dv64_w32_fp16.json`: widest
  supported Q/K symbol.
- `validation/block_diagonal_vjp_tf32_fp16_all_masks_sm86.json`: ordinary
  block plan against production, all seven masks.
- `validation/block_diagonal_vjp_pipeline_fp16_all_masks_sm86.json`: named
  barrier plan against production, all seven masks.
