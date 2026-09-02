# Benchmarks

These scripts are manual probes, not correctness tests.

The stateful Runtime, paging/LCT, and promoted hard-index sections below are
historical records. Their sources and tests now live in
`contrib/runtime_legacy/` and are excluded from the package, build graph, and
default pytest collection. Commands using `benchmarks.rosa_runtime` or
`benchmarks.runtime_*` therefore describe the archived tree, not the current
production library. The maintained hard reference is `rosa_soft.RosaSam` and
its correctness landmark is the independent DP in `tests/test_sam.py`.

The frozen first filtered-bitflip index route is summarized in
`docs/research/FILTERED_BITFLIP_V1.md`. Its current entry point is
`filtered_bitflip_compact.py`; earlier solvers are exact baselines and the
orthogonal indexes are post-v1 experiments.

- `rosa_soft.py` measures RosaSoft forward or training-step latency and peak
  CUDA allocator usage. It can independently select dense or packed-varlen
  layout and any required subset of Q/K/value gradients.
- `historical_suffix_attention_speed.py` reproduces the fixed-window SUFA
  backward from commit `c776101` while sharing the current exact hard forward.
  `attention_speed_comparison.py` adds forced PyTorch FlashAttention controls
  at Q/K-width-matched and output-width-matched head dimensions. The current
  production integration decision is recorded in
  `docs/research/PRODUCTION_INTEGRATION_AUDIT.md`.
- `discrete_gradient_alignment.py` exhaustively compares the surrogate VJP
  direction with exact hard Q/K sign-bit flips on small CPU problems.
- `suffix_proxy_ablation.py` keeps exact hard forward while comparing Hamming
  normalization, normalized power/log suffix utilities, and a research-only
  random-collision likelihood-ratio route. It keeps model width fixed while
  changing Q/K bit width. Its directed competition gate tests whether a
  shorter target can repair itself against a frozen longer distractor.
- `fast_weight_proxy.py` keeps exact hard forward while comparing
  parameter-free local linear attention, normalized delta memory, exact
  fixed/multi-length suffix fingerprints, and fixed TensorSketch compression.
  `fast_weight_proxy_ablation.py` compares only those proxies, production, and
  complete bitflip on matched fitting and gradient gates. The evidence and
  current non-promotion decision are in
  `docs/research/FAST_WEIGHT_PROXY.md`.
- `suffix_kernel_proxy.py` is the aligned independent-Q/K suffix-state oracle.
  Exact tensor levels reproduce the dense raw suffix score and its production
  local-gate VJP; nested TensorSketch, exact-local hybrid raw, bounded global
  quadratic, and per-level quadratic are fixed-width controls.
  `suffix_kernel_ablation.py` reports candidate mass, effective-route count,
  sketch score/VJP error, complete-bitflip alignment, and matched fitting.
- `long_suffix_extrapolation.py` trains one shared oldest-suffix fault at 8K
  against every context position, with strict `W-1` aligned distractors for
  `W=1/2/4/8/32`. It compares the exact production final-row VJP, the current
  quadratic carrier, exact and sketched factorized suffix kernels, and an exact
  relevant-bit counterfactual, then freezes the symbols and performs packed
  exact scans at 64K and 1M. Balanced binary V8 is the main condition;
  coherent negative V is an adversarial control.
- `null_calibration_ablation.py` estimates random-background route partitions
  and planted-route capacity across D/W/candidate count. It compares the
  production candidate prior and fixed null score with no correction,
  moment calibration, and collision-likelihood-ratio controls.
- `global_bit_oracle.py` is an exponential, float64-scale research oracle for
  one globally shared assignment of every relevant Q/K bit. It implements
  exact local expectation, mean-field winner marginals, ARM/DisARM, exact and
  sampled-residual bitflip VJPs, and exact loss-augmented margin edits.
- `filtered_bitflip.py` defines a separate unlimited-suffix full-bitflip
  oracle and exact create/break influence intervals.  The companion
  `filtered_bitflip_indexes.py`, `filtered_bitflip_winner.py`, and
  `filtered_bitflip_profile.py` compare static LCE indexes, code-occurrence
  discovery, normalized-winner filtering, and synthetic training-clarity
  trajectories.  They remain research-only and do not modify frozen
  RosaSoft semantics.  `csrc/filtered_bitflip_cpu_scan.cpp` is the standalone
  packed CPU hard-scan microbenchmark.
- `filtered_bitflip_certificates.py` treats each maximal matching diagonal run
  as a constant normalized-priority interval and computes exact temporary
  delete/add winner envelopes.  `filtered_bitflip_suffix_nodes.py` provides
  exact LCP-node replacement queries with flat or arithmetic-run postings.
  `filtered_bitflip_hybrid.py` dispatches each flip between those exact paths
  using measured interval compression only.  Their common trajectory profiler
  is `filtered_bitflip_certificate_profile.py`.
- `filtered_bitflip_compact.py` is the frozen exact v1 research solver. It
  combines a pinned optional `libsais` backend, packed wavelet
  range predecessor, parent-linked LCP nodes, periodic route exclusions,
  indexed event families, bounded top-k certificates with exact suffix
  fallback, and direct affine-range VJP accumulation.
  `filtered_bitflip_periodic.py` adds conservative whole-interval proofs for
  shifted query edits, finite-period shifted key edits, and the uniform
  period-one case.  It also generates exact event families directly from
  periodic occurrence phases, without materializing every changed pair.
  Staircase-affine winners and `PeriodicRouteRange` represent complete
  periodic row ranges through replacement and VJP without leaf queries or
  per-period Python objects.  Early key edits outside the proof retain the
  general exact path.  These files deliberately do not modify or replace the
  frozen dense-gradient operator.  Reproduce their measurements with
  `filtered_bitflip_compact_profile.py`.
- `filtered_bitflip_monotone.py` studies arbitrary, nonperiodic occurrence
  lists through exact suffix-rank LCE plateaus.  It uses semantic sentinels,
  previous/next-smaller LCP links, and intersections of forward and reversed
  runs to form exact `(left_lce, right_lce)` cells.  The compact solver uses
  one latest-route representative per query-create cell only where its
  existing shifted-query baseline proof makes all break events redundant.
  `filtered_bitflip_monotone_profile.py` measures both singleton failure cases
  and useful low-alphabet compression; it is not evidence of a universal
  subquadratic bitflip bound.
- `filtered_bitflip_orthogonal.py` replaces query-create occurrence scans with
  one code-partitioned dynamic range tree.  A causal position sweep realizes
  the static `(forward rank, reverse rank, position)` query as an active 2D
  aggregate, and synchronized band traversal enumerates exact nonempty cells
  without probing their members.  `filtered_bitflip_orthogonal_kd.py` is the
  linear-space static 3D reference with arbitrary position-box count and
  predecessor/successor extrema.  Both remain research-only because range
  trees regress on fragmented inputs and use `O(T log T)` logical storage,
  while the kd tree has linear worst-case query work.  Compare them with the
  occurrence scan using `filtered_bitflip_orthogonal_profile.py`.
- `sam_bitflip.py` is a second independent exact unlimited-suffix route. It
  uses a dual suffix automaton, suffix-link-tree endpos predecessor queries,
  exact checkpoint replay for Q edits, and an exclude-plus-anchored-run
  decomposition for K edits. The default result uses compressed affine route
  changes, grouped Q branches, binary K replacement search, and exact
  arithmetic-progression endpos certificates. `sam_bitflip_native.py` builds
  the independent C++17 backend, whose primary ABI emits base winners,
  affine Q changes, shared K-deletion changes, and bit-specific K overrides;
  its materialized API reconstructs that factorized result for validation.
  `sam_bitflip_vjp.py` JIT-builds the benchmark-only CUDA descriptor VJP.
  `sam_bitflip_profile.py` compares CPU routes, compression, optional exact
  last-M caching, the full-rerun oracle, and frozen v1.
  `sam_bitflip_vjp_profile.py` separates CPU indexing, descriptor transfer,
  and cached CUDA contraction. The design and proof map is
  `docs/research/SAM_BITFLIP.md`.
- `global_bit_fit.py` trains only three key logits on one-edit and coordinated
  two-edit hard-route tasks. It is the multi-seed optimization gate for the
  global-bit estimators and deliberately has no residual or readout shortcut.
- `shared_projection_fit.py` maps two trainable parameters into three coupled
  key logits. It checks whether activation-space estimators remain useful
  after gradients are aggregated through a shared projection Jacobian.
- `stochastic_hard_vjp.py` contains research-only batched ARM, DisARM, and
  `W=1` mean-field Q/K VJPs. All variants preserve exact hard forward and use
  the production dense V VJP; they are not public RosaSoft operators.
- `contextual_estimator_recall.py` compares those research VJPs with the
  production estimator on the reset-RNN shortcut-free recall gate. Its
  `--context-depth` control stacks reset-GRU context layers while preserving
  the original depth-one path exactly.
- `state_attention_scaling.py` drives a deduplicated one-variable matrix over
  association count, Q/K head count, and reset-GRU depth for production and
  state-attention estimators.
- `state_attention_pretraining.py` trains with one uniform full-sequence
  next-token cross entropy. It samples fresh complementary cue/payload maps on
  every optimizer step and reports hard-route, zero-route, current-value, and
  assignment-independence controls plus the data-distribution loss floor.
  Payload vocabulary width and hard V width are independent controls. Its
  research estimator list also accepts complete bitflip; bitflip variants are
  evaluated in memory-bounded batches without sampling any coordinate.
  `--bitflip-gradient-scale` changes only the exact Q/K/V VJP magnitude; it
  does not change the hard forward or omit any counterfactual.
- `state_attention_init.py` provides research-only deterministic Q/K/V
  initializers. The partial-shared Q/K initializer gives repeated contexts an
  initial distance-independent match bias without tying any parameters after
  initialization. `state_attention_init_ablation.py` crosses model and data
  seeds and separates Q/K correlation, orthogonal V geometry, V width, and a
  paired-output negative control. These paths do not change the production
  operator or its default initialization.
- `value_codec_proxy.py` keeps exact hard ROSA routing while comparing binary,
  fixed uniform k-bit, fixed signed-exponent k-bit, RMS-normalized, and raw
  float V representations. It can give V either the existing dense state
  credit or only the current hard-selected value gradient.
  `state_attention_value_ablation.py` runs the controlled V4/V8 next-token
  matrix. Both files are research-only and add no production operator option.
- `state_attention_estimator_vjp.py` profiles matched hard-forward VJPs for
  production RosaSoft, complete bitflip, and quadratic state attention. It
  reports latency, CUDA allocator peak, and exact forward parity for explicit
  `batch:sequence-length` cases.
- `internal_language.py` provides the research-only deterministic slow-state
  trunk and separate read-Q/write-K heads shared by the internal-language
  gates. It is not a public model layer or operator.
- `latent_grammar_gate.py` gives each phrase one learned content bit plus a
  fixed phase bit and outline mask. Its 3/5/17-candidate tasks strictly need
  complete 2/4/8-position trajectories and have no trainable value/readout
  shortcut.
- `multihop_recall_gate.py` runs two exact-hard RosaSoft calls. The first hard
  value is the only input to the second query; zero, shuffled, detached, and
  oracle feedback are explicit causal interventions.
- `symbol_growth.py` groups hard key states by continuation label and
  initializes one dormant Q/K bit with a deterministic conflict split. It
  reports key collisions separately from query/key self-alignment.
- `estimator_fit_ablation.py` compares exact-hard-forward training VJPs:
  deterministic, mismatch-random, exact bitflip, standard post-softmax
  attention dropout, and research-only long-suffix residual dropout.
  Model/data and stochastic-estimator seeds can be crossed independently.
- `trained_fit_alignment.py` trains the repeated-motif model first, then
  compares its production surrogate with exact hard bit flips at the trained
  checkpoint. It reports tensor and shared-parameter alignment together with
  the hard-feature entropy floor.
- `examples/fit_soft_reference.py` is the production repeated-motif fitting
  probe. Its JSON output reports the empirical conditional-entropy lower
  bound induced by `(current token, hard routed values)`, the loss above that
  bound, and route-value versus quantized-value collision counts.
  `--target-mode any-candidate` is a combinatorial fitting stress test;
  `strict-longest-latest` retains only targets selected by the raw
  longest/latest token route.
- `examples/contextual_rnn_recall_gate.py` stores cue/payload associations,
  explicitly resets a GRU, and queries the associations with post-reset
  residuals that are exactly identical across complementary assignments.
  Hard-history, zero-route, current-value, and residual-only ablations make
  it a shortcut-free learned-recall gate.
- `fast_weight_proxy.py` keeps exact hard ROSA forward values while supplying
  parameter-free fast-weight VJPs. Its current-symbol ladder compares complete
  first-, second-, third-, and full-order bit interactions under additive
  normalized linear attention and delta-rule state updates.
- `temporal_quadratic_proxy.py` is the two-pass long-suffix research control.
  A fixed orthogonal finite-window scan exposes all W query symbols, then a
  complete homogeneous quadratic feature map drives an additive normalized
  fast-weight read. It has exact hard ROSA forward values, no trainable proxy
  parameters, no auxiliary loss, and no configurable decay.
- `suffix_kernel_proxy.py` is the second two-pass control. It builds aligned
  multiplicative suffix features independently for Q and K, then performs an
  additive or delta fast-weight scan. The exact expansion is a mathematical
  oracle; the fixed-width sketches are non-promoted scaling experiments.
- `rosa_runtime.py` measures exact hard-runtime latency, logical state,
  operation counters, and chunk invariance across random, unary, periodic,
  and skewed streams.
- `runtime_periodic_state.py` validates the exact repeated-motif macro-state
  prototype against the native runtime.
- `runtime_pageability.py` preserves exact unlimited online SAM routes while
  replaying ideal indexed state, edge, latest, and payload accesses through
  exact LRU caches. Ideal transition lookup is a lower bound, while its direct
  exact latest-end writes deliberately do not model the production link-cut
  tree. Page and cache sizes affect counters only and never matching semantics.
- `runtime_page_trace.py` compiles the production C++ automaton core into a
  benchmark-only trace build. It records stable logical pages for the real
  transition indexes, root table, edge SoA, LCT, state, and payload paths once,
  then replays LRU, SLRU, fixed-region, structural-pin, and append-bypass
  policies offline. The package build and runtime schemas remain unchanged.
- `runtime_async_pipeline.py` compares blocking, explicit-stream, and depth-2
  packed CPU/GPU staging.
- `training_hard_fusion.py` measures finite-W hard ambiguity fallback and
  validates the dual-accumulator softmax VJP identity.
- `diagonal_recurrence.py` validates the exact finite-window diagonal score
  recurrence, log-gate adjoint, and complete Q/K symbol VJP against explicit
  prefix products and full-matrix autograd. Its bounded local-contribution
  VJP is the cancellation-resistant oracle for long exact matches.
- `streaming_vjp.py` builds a direct benchmark wrapper around the production
  tiled-streaming CUDA source. It can force the production 32-row layout or
  the benchmark-only 16-row negative control for parity and calibration
  without adding a public operator control.
- `streaming_vjp_profile.py` compares that direct kernel with the public
  production dispatch and reports latency, allocation, and parity by sequence
  length, gradient mask, pattern, and query-tile size.
- `block_suffix_scan.py` isolates candidate-thread, diagonal-block, and
  warp-per-candidate suffix ownership. `block_suffix_scan_profile.py` records
  the tile/window matrix; the warp path is retained as a measured negative
  control.
- `block_diagonal_vjp.py` exposes the exact hard-forward research operator and
  the scalar or TF32 `64 x 64` block-diagonal dense VJP.
  `block_diagonal_vjp_profile.py` compares every gradient mask, dtype, dropout
  setting, and code pattern with the frozen streaming baseline. The design and
  current sm_86 evidence are in
  `docs/research/BLOCK_DIAGONAL_TENSOR_CORE_VJP.md`.
- `hard_forward_profile.py` measures the production exact hard forward across
  random, shifted-aligned, periodic, all-match, and all-mismatch codebooks. It
  records entropy, collision mass, latest-route alignment, and latency for
  calibrating and auditing the unlimited production hard-index dispatch across
  learned code regimes.
- `indexed_hard_forward.py` exposes the exact D1..D8 occurrence-index,
  latest-candidate certificate, and diagonal research controls without adding
  schemas to the production extension. `indexed_hard_forward_profile.py`
  separates index construction and route latency and records the production
  bit/length gate evidence. Survivor handoff and approximate diagonal
  execution remain negative controls. Production additionally uses an exact
  parallel stable builder at 8K+ and one exactly reconstructed heavy-restart
  diagonal; both preserve the complete occurrence fallback. Their design and
  ablation are in
  `docs/research/EXACT_HARD_RESTART_AND_OCCURRENCE_INDEX.md`.
- `production_hard_index_sanitizer_smoke.py` is the minimal fixed-length target
  for memcheck, racecheck, and synccheck of the promoted hard index.
- `pretraining_codebook.py` consumes matched Q/K trajectories and
  continuation labels from a checkpoint. It reports Q/K alignment, hard-key
  continuation conflicts, conditional entropy, and route quality at every
  suffix horizon; controlled aligned, corrupted, role-drift, and collapsed
  snapshots are included.

Historical non-ROSA probes live under `contrib/` and are not packaged or run
in CI.

Run the scripts from the repository root. Results depend on the PyTorch build,
CUDA toolkit, GPU, clocks, host CPU, thread count, input shape, and data
pattern; record those fields with any published measurement.

CUDA builds pass `-res-usage`; the build log reports registers, stack, and
spills for every mangled backward instance, including plan, layout,
gradient-mask, aggregation, cooperative-utility, and packed-score-cache
template axes. Record that table with timing results rather than adding
telemetry to the training operator.

```bash
python benchmarks/rosa_soft.py \
  --operator cuda --sequence-lengths 64 128 256 \
  --max-suffix-length 32 --scale 1 --dropout-p 0.1 \
  --mismatch-scale 3

python benchmarks/rosa_soft.py \
  --operator cuda --layout varlen --segment-length 64 \
  --sequence-lengths 128 256 512 --gradients qkv

python benchmarks/rosa_soft.py \
  --operator cuda --sequence-lengths 128 256 512 \
  --gradients v

CUDA_VISIBLE_DEVICES=1 python \
  benchmarks/historical_suffix_attention_speed.py \
  --dtype float32 --sequence-lengths 64 128 256 512 1024 2048 \
  --suffix-windows 32 --bits 8 --value-dim 64

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.attention_speed_comparison \
  --sequence-lengths 512 1024 2048 4096 8192 16384 \
  --dtype float16 --bits 8 --value-dim 64 --suffix-window 32 \
  --flash-head-dims 8 64 \
  --output validation/attention_speed_production_sm86.json

python benchmarks/discrete_gradient_alignment.py

python benchmarks/suffix_proxy_ablation.py \
  --device cpu --qk-bits 2 4 8 --windows 2 8 32 \
  --mismatch-scales 1.5 3 --model-seeds 0 1 2 3 \
  --shell-seeds 0 1 2 3 --run-competition

CUDA_VISIBLE_DEVICES=0 python benchmarks/fast_weight_proxy_ablation.py \
  --device cuda --estimators production bitflip \
    single_suffix_sketch_delta \
  --model-seeds 0 1 2 3 --steps 1000 \
  --qk-bits 4 --value-bits 4 --max-suffix-length 8 \
  --fingerprint-length 3 --sketch-dim 256 \
  --json-out validation/fast_weight_proxy_d4w8_l3_r256_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/suffix_kernel_ablation.py \
  --device cuda --model-seeds 0 1 2 3 --steps 500 \
  --sketch-dims 16 32 64 --sketch-counts 1 2 4 \
  --json-out validation/suffix_kernel_ablation_sm86.json --summary-only

CUDA_VISIBLE_DEVICES=0 python benchmarks/long_suffix_extrapolation.py \
  --device cuda --estimators exact_suffix_raw_attention \
    suffix_sketch_raw_attention \
  --windows 1 2 4 8 32 --seeds 0 1 2 3 4 5 6 7 \
  --probes 32 --bits 8 --value-bits 8 --value-mode balanced_binary \
  --train-context-length 8192 \
  --eval-context-lengths 8192 65536 1048576 \
  --steps 50 --suffix-sketch-dim 64 --suffix-sketch-count 4 \
  --chunk-size 65536 --summary-only \
  --json-out validation/suffix_kernel_hybrid_raw_balanced_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/long_suffix_extrapolation.py \
  --device cuda --windows 1 2 4 8 32 --seeds 0 1 2 3 4 5 6 7 \
  --probes 32 --bits 8 --value-bits 8 --value-mode balanced_binary \
  --train-context-length 8192 \
  --eval-context-lengths 8192 65536 1048576 \
  --steps 50 --chunk-size 65536 --summary-only \
  --json-out validation/long_suffix_extrapolation_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/long_suffix_extrapolation.py \
  --device cuda --estimators temporal_quadratic_attention \
  --windows 1 2 4 8 32 --seeds 0 1 2 3 4 5 6 7 \
  --probes 32 --bits 8 --value-bits 8 --value-mode balanced_binary \
  --train-context-length 8192 \
  --eval-context-lengths 8192 65536 1048576 \
  --steps 50 --temporal-state-dim 64 --summary-only \
  --json-out validation/temporal_quadratic_extrapolation_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/fast_weight_proxy_ablation.py \
  --device cuda \
  --estimators production bitflip temporal_quadratic_attention \
  --model-seeds 0 1 2 3 --gradient-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --steps 500 --temporal-state-dim 64 \
  --json-out validation/temporal_quadratic_fit_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_scaling.py \
  --operator cuda --device cuda --seeds 0 1 2 3 --steps 400 \
  --association-values 2 4 8 16 --head-values 1 2 4 \
  --depth-values 1 2 4 --json-out validation/state_attention_scaling_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_pretraining.py \
  --operator cuda --device cuda --seeds 0 1 2 3 --steps 2000 \
  --associations 4 --context-depth 2 --heads 4 \
  --json-out validation/state_attention_pretraining_h4_sm86.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/state_attention_init_ablation.py \
  --operator cuda --device cuda \
  --estimators state_quadratic_attention \
  --model-seeds 0 1 2 3 --data-seeds 0 1 --steps 1000 \
  --train-pairs 32 --validation-pairs 64 \
  --experiments E0_default_dv4 E1_shared_dv4 \
    E6_shared_dv4_orthogonal E3_shared_dv8 \
    E7_independent_qk_dv4 \
  --json-out validation/state_attention_init_ablation_sm86.json \
  --summary-only

CUDA_VISIBLE_DEVICES=1 python benchmarks/null_calibration_ablation.py \
  --device cuda --json-out validation/null_calibration_ablation.json

python benchmarks/global_bit_oracle.py \
  --seeds 0 1 2 3 4 5 6 7 --sample-count 4096 --summary-only

python benchmarks/global_bit_fit.py \
  --model-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --steps 100 --summary-only

python benchmarks/shared_projection_fit.py \
  --model-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --steps 100 --summary-only

CUDA_VISIBLE_DEVICES=1 python benchmarks/contextual_estimator_recall.py \
  --operator cuda --device cuda --seeds 0 1 2 3 \
  --bit-temperature 0.5 --antithetic-pairs 4 \
  --dropout-p 0.1 --steps 1000 --summary-only

CUDA_VISIBLE_DEVICES=1 python benchmarks/latent_grammar_gate.py \
  --operator cuda --device cuda:0 --seeds 0 1 2 3 \
  --phrase-lengths 2 --windows 2 --mode both \
  --pairs 3 --steps 500 --learning-rate 0.001

CUDA_VISIBLE_DEVICES=1 python benchmarks/multihop_recall_gate.py \
  --operator cuda --device cuda:0 --seeds 0 1 2 3 \
  --training-feedback-modes routed zero --head-init aligned \
  --pairs 24 --steps 500 --learning-rate 0.001

CUDA_VISIBLE_DEVICES=1 python benchmarks/symbol_growth.py \
  --operator cuda --device cuda:0 --seeds 0 1 2 3 \
  --strategies fixed growth --candidate-count 32 \
  --initial-bits 2 --max-bits 8 --steps 300 --growth-interval 50

python benchmarks/diagonal_recurrence.py \
  --device cpu --dtype float64

python benchmarks/pretraining_codebook.py \
  --concepts 64 --trajectory-length 8 --bits 4 \
  --json-out validation/pretraining_codebook.json

python benchmarks/filtered_bitflip_compact_profile.py \
  --build-native --sequence-lengths 64 128 256 \
  --execution-length 64 --top-k 4 8 --repeats 3 \
  --json-out validation/filtered_bitflip_compact.json

python benchmarks/filtered_bitflip_monotone_profile.py \
  --sequence-lengths 64 128 --repeats 3 \
  --json-out validation/filtered_bitflip_monotone.json

python benchmarks/filtered_bitflip_orthogonal_profile.py \
  --sequence-lengths 128 256 512 1024 --repeats 3 \
  --json-out validation/filtered_bitflip_orthogonal.json

python benchmarks/sam_bitflip_profile.py \
  --build-native --sequence-lengths 32 64 --repeats 5 --compare-v1 \
  --json-out validation/sam_bitflip.json

CUDA_VISIBLE_DEVICES=1 python benchmarks/sam_bitflip_vjp_profile.py \
  --sequence-lengths 64 128 256 512 --feature-size 128 --repeats 5 \
  --validate-max-length 128 \
  --json-out validation/sam_bitflip_vjp.json

CUDA_VISIBLE_DEVICES=1 python benchmarks/streaming_vjp_profile.py \
  --lengths 128 256 512 1024 2048 4096 8192 \
  --tiles 16 32 --masks 1 2 3 4 5 6 7 \
  --patterns random all_match \
  --output validation/streaming_vjp.json

# Isolated Tensor-Core arithmetic and integrated two-pass execution plans.
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.tensor_core_vjp_profile \
  --output validation/tensor_core_primitives_sm86.json

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.flash_tc_vjp_profile \
  --lengths 512 1024 2048 4096 8192 --windows 32 \
  --masks 1 2 3 4 5 6 7 --bits 32 --value-dims 64 \
  --plans baseline tc_gate \
  --output validation/flash_tc_gate_lengths_sm86.json

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.block_suffix_scan_profile \
  --lengths 4096 --windows 8 16 32 64 128 --bits 8 --heads 4 \
  --methods thread diagonal32 diagonal64 diagonal128 warp_suffix \
  --output validation/block_suffix_tile_matrix_sm86.json

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.block_diagonal_vjp_profile \
  --lengths 4096 8192 --bits 8 32 --value-dims 64 \
  --masks 1 2 3 4 5 6 7 --plan block_tf32 \
  --output validation/block_diagonal_vjp_tf32_hilo_all_masks_sm86.json

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.hard_forward_profile \
  --sequence-lengths 4096 8192 16384 65536 \
  --patterns random aligned periodic64 periodic4 all_match all_mismatch \
  --batch 1 --heads 4 --value-heads 2 --bits 8 --value-dim 64 \
  --seed 7000 --warmup 7 --repeats 21 \
  --output validation/unlimited_hard_restart_builder_final_sm86.json

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.hard_forward_profile \
  --sequence-lengths 65536 \
  --patterns random periodic16 periodic32 periodic64 periodic128 periodic256 periodic512 \
  --batch 1 --heads 4 --value-heads 2 --bits 8 --value-dim 64 \
  --seed 7000 --warmup 7 --repeats 21 \
  --output validation/unlimited_hard_period_sweep_final_v2_sm86.json

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.indexed_hard_forward_profile \
  --sequence-lengths 4096 8192 16384 --windows 32 128 \
  --bits 1 2 4 8 \
  --output validation/indexed_hard_forward_long_sm86.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. compute-sanitizer --tool memcheck \
  python -m benchmarks.production_hard_index_sanitizer_smoke \
  --pattern periodic64 --sequence-length 8192

# Exact sparse-tail ownership and full-tile hand-off controls.
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.block_suffix_tail_profile \
  --series 4096 \
  --output validation/block_suffix_tail_ownership_saturated_sm86.json

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.block_suffix_hybrid_profile \
  --output validation/block_suffix_hybrid_tail_paired_sm86.json

# Named-barrier producer/consumer overlap against the unchanged TF32 plan.
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.block_diagonal_vjp_profile \
  --lengths 4096 8192 --bits 8 32 --value-dims 64 \
  --masks 1 2 3 4 5 6 7 --plan block_tf32_pipeline \
  --baseline-plan block_tf32 \
  --output validation/block_diagonal_vjp_pipeline_sm86.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. compute-sanitizer --tool racecheck \
  python -m benchmarks.block_diagonal_vjp_sanitizer_smoke

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. compute-sanitizer --tool memcheck \
  python -m benchmarks.production_block_sanitizer_smoke
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. compute-sanitizer --tool racecheck \
  python -m benchmarks.production_block_sanitizer_smoke
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. compute-sanitizer --tool synccheck \
  python -m benchmarks.production_block_sanitizer_smoke

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. compute-sanitizer --tool racecheck \
  python -m benchmarks.block_suffix_scan_sanitizer_smoke

CUDA_VISIBLE_DEVICES=1 python benchmarks/estimator_fit_ablation.py \
  --device cuda --model-seeds 0 1 2 3 4 5 6 7 \
  --estimators deterministic mismatch_random bitflip attention_dropout \
  --dropout-p 0.1 --steps 1000

CUDA_VISIBLE_DEVICES=1 python benchmarks/estimator_fit_ablation.py \
  --device cuda --model-seeds 0 1 2 3 4 5 6 7 \
  --estimators suffix_dropout \
  --dropout-p 0.9 --mismatch-scale 9 --steps 1000

CUDA_VISIBLE_DEVICES=1 python benchmarks/trained_fit_alignment.py \
  --operator cuda --device cuda:0 --model-seeds 0 2 10 27 29 \
  --target-mode strict-longest-latest --steps 1000

CUDA_VISIBLE_DEVICES=1 python examples/contextual_rnn_recall_gate.py \
  --operator cuda --device cuda:0 --seeds 0 1 2 3 \
  --dropout-p 0.1 --steps 1000

python -m benchmarks.rosa_runtime \
  --sequence-lengths 8192 65536 --patterns random all_match periodic64 skewed \
  --chunk-sizes 8192 65536 --batch 1 --heads 1 --payload-heads 1 --bits 8

python -m benchmarks.runtime_periodic_state \
  --tokens 8192 65536 --periods 1 2 8 64 --bits 8

python -m benchmarks.runtime_pageability \
  --tokens 8192 65536 \
  --patterns random skewed periodic64 copied1024 natural \
  --page-kib 4 16 64 --cache-mib 0.25 1 4

python -m benchmarks.runtime_page_trace \
  --tokens 8192 65536 \
  --patterns random skewed periodic64 copied1024 natural \
  --page-kib 4 16 64 --cache-mib 0.25 1

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.training_hard_fusion \
  --tokens 512 2048 --bits 1 4 8 --windows 1 2 4 8 32 \
  --patterns random skewed periodic64 all_match --device cuda:0
```
