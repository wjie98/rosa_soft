# Benchmarks

These scripts are manual probes, not correctness tests.

- `rosa_soft.py` measures RosaSoft forward or training-step latency and peak
  CUDA allocator usage. It can independently select dense or packed-varlen
  layout and any required subset of Q/K/value gradients.
- `discrete_gradient_alignment.py` exhaustively compares the surrogate VJP
  direction with exact hard Q/K sign-bit flips on small CPU problems.
- `suffix_proxy_ablation.py` keeps exact hard forward while comparing Hamming
  normalization, normalized power/log suffix utilities, and a research-only
  random-collision likelihood-ratio route. It keeps model width fixed while
  changing Q/K bit width. Its directed competition gate tests whether a
  shorter target can repair itself against a frozen longer distractor.
- `null_calibration_ablation.py` estimates random-background route partitions
  and planted-route capacity across D/W/candidate count. It compares the
  production candidate prior and fixed null score with no correction,
  moment calibration, and collision-likelihood-ratio controls.
- `global_bit_oracle.py` is an exponential, float64-scale research oracle for
  one globally shared assignment of every relevant Q/K bit. It implements
  exact local expectation, mean-field winner marginals, ARM/DisARM, exact and
  sampled-residual bitflip VJPs, and exact loss-augmented margin edits.
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
  production estimator on the reset-RNN shortcut-free recall gate.
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
- `rosa_runtime.py` measures hard-runtime update latency and reports runtime
  statistics.
- `diagonal_recurrence.py` validates the exact finite-window diagonal score
  recurrence, log-gate adjoint, and complete Q/K symbol VJP against explicit
  prefix products and full-matrix autograd. Its bounded local-contribution
  VJP is the cancellation-resistant oracle for long exact matches.
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

python benchmarks/discrete_gradient_alignment.py

python benchmarks/suffix_proxy_ablation.py \
  --device cpu --qk-bits 2 4 8 --windows 2 8 32 \
  --mismatch-scales 1.5 3 --model-seeds 0 1 2 3 \
  --shell-seeds 0 1 2 3 --run-competition

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

python benchmarks/rosa_runtime.py \
  --B 4 --T 256 --H 8 --Hv 2 --bits 4 \
  --max-suffix-length 32
```
