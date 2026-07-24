# Benchmarks

These scripts are manual probes, not correctness tests.

- `rosa_soft.py` measures RosaSoft forward or training-step latency and peak
  CUDA allocator usage.
- `rosa_runtime.py` measures hard-runtime update latency and reports runtime
  statistics.
- `legacy/` contains optional RWKV7 kernel probes retained for historical
  comparison. They require `ROSA_BUILD_RWKV7=1` and are not part of CI.

Run the scripts from the repository root. Results depend on the PyTorch build,
CUDA toolkit, GPU, clocks, host CPU, thread count, input shape, and data
pattern; record those fields with any published measurement.

```bash
python benchmarks/rosa_soft.py \
  --operator cuda --sequence-lengths 64 128 256 \
  --max-suffix-length 32 --route-temperature 1 \
  --mismatch-penalty 3

python benchmarks/rosa_runtime.py \
  --B 4 --T 256 --H 8 --Hv 2 --bits 4 \
  --max-suffix-length 32
```
