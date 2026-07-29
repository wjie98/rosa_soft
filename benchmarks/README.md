# Benchmarks

These scripts are manual probes, not correctness tests.

- `rosa_soft.py` measures RosaSoft forward or training-step latency and peak
  CUDA allocator usage. It can independently select dense or packed-varlen
  layout and any required subset of Q/K/value gradients.
- `discrete_gradient_alignment.py` exhaustively compares the surrogate VJP
  direction with exact hard Q/K sign-bit flips on small CPU problems.
- `estimator_fit_ablation.py` compares four exact-hard-forward training VJPs:
  deterministic, mismatch-random, exact bitflip, and standard post-softmax
  attention dropout. Model/data and stochastic-estimator seeds can be crossed
  independently.
- `rosa_runtime.py` measures hard-runtime update latency and reports runtime
  statistics.

Historical non-ROSA probes live under `contrib/` and are not packaged or run
in CI.

Run the scripts from the repository root. Results depend on the PyTorch build,
CUDA toolkit, GPU, clocks, host CPU, thread count, input shape, and data
pattern; record those fields with any published measurement.

CUDA builds pass `-res-usage`; the build log reports registers, stack, and
spills separately for both mangled `BackwardPlan` template instances.
Record that table with timing results rather than adding telemetry to the
training operator.

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

CUDA_VISIBLE_DEVICES=1 python benchmarks/estimator_fit_ablation.py \
  --device cuda --model-seeds 0 1 2 3 4 5 6 7 \
  --estimators deterministic mismatch_random bitflip attention_dropout \
  --dropout-p 0.1 --steps 1000

python benchmarks/rosa_runtime.py \
  --B 4 --T 256 --H 8 --Hv 2 --bits 4 \
  --max-suffix-length 32
```
