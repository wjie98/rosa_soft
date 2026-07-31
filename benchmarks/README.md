# Benchmarks

These scripts are manual probes, not correctness tests.

- `rosa_soft.py` measures RosaSoft forward or training-step latency and peak
  CUDA allocator usage. It can independently select dense or packed-varlen
  layout and any required subset of Q/K/value gradients.
- `discrete_gradient_alignment.py` exhaustively compares the surrogate VJP
  direction with exact hard Q/K sign-bit flips on small CPU problems.
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
  prefix products and full-matrix autograd. It is a research oracle, not a
  production operator benchmark. Random-gate FP32 parity does not cover the
  known long exact-match adjoint cancellation; see
  `docs/research/KERNEL_OPTIMIZATION.md`.

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

python benchmarks/diagonal_recurrence.py \
  --device cpu --dtype float64

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
