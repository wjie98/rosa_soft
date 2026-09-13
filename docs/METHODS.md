# Methods and Related Work

[English overview](../README.md) | [中文概览](../README.zh-CN.md) |
[API](API.md) | [Memory](MEMORY.md) | [Design](DESIGN.md)

## Separate the Contracts

ROSA implementations should be compared along independent axes:

- **Forward:** symbol encoding, suffix limits, longest/latest ordering,
  successor values, null behavior and sequence boundaries.
- **Credit:** full bit-edit output differences, local counterfactuals, or a
  continuous/sparse surrogate. Independent and joint edits are different.
- **Logit mapping:** how discrete credit becomes a continuous input gradient.
- **Execution:** CPU/GPU implementation, live memory, time, dtype and API support.

An exact suffix automaton does not by itself imply an exact backward or an
unchanged hard winner when learned ranking is added. Likewise, exact bit-edit
credit does not imply an ordinary derivative of the hard threshold function.

## The Two Production Estimators

Both operators use the same unlimited hard forward. Their backward contracts
are deliberately different and are never switched automatically.

| Property | `rosa_soft` | `rosa_bitflip` |
| --- | --- | --- |
| Routing credit | Dense soft suffix recurrence over all causal candidates | Complete hard output differences for each requested activation-bit edit |
| Symbol mapping | Softsign STE | Oriented difference followed by softsign scaling |
| Independent V credit | Dense probability-weighted carrier | Original hard-route scatter |
| Joint QK/QKV edits | Not a separate estimator mode | Explicit `tied` modes |
| Randomness | Optional backward-only dropout | No edit sampling |
| Layout | Dense and packed | Dense only |
| Main scratch | Bounded slabs or checkpoint replay | Reusable row bands, optionally head-chunked |

For binary signs b, hard output F, and fixed upstream gradient g, bitflip
defines the complete activation-edit credit:

```text
c[e] = <g, F(flip(b,e)) - F(b)>
dX[e] = -sign(X[e]) * c[e] / (2*(1+abs(X[e]))^2)
```

This is exact for the output counterfactual in real arithmetic. It is a finite
difference of the linearized downstream objective, not generally the actual
nonlinear loss difference. No downstream network is rerun for each bit.
Flipping one activation bit is also not flipping a shared projection weight.
Floating-point accumulation is subject to rounding and may be nondeterministic.

For an independent V bit, the route cannot change. Its complete oriented
difference with the same scaling equals hard-route scatter followed by the
softsign derivative, so no explicit V-edit simulation is required. Joint QKV
edits must instead evaluate routing and payload changes together.

The soft estimator can assign credit to candidates that no single bit flip
would make win, but this signal is a surrogate. Bitflip measures each local
discrete move directly, but returns zero credit when the move does not change
the output. Neither property establishes universal training superiority.
The dense recurrence, utility, null prior and derivative are in [Design](DESIGN.md).

## Related Implementations

Public sources reviewed on 2026-09-14. These are methodological descriptions,
not common-hardware benchmarks or claims of independent reproduction.

| Project / reference | Method and comparison boundary |
| --- | --- |
| [RWKV-LM single-stream training](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251014_rosa_1bit_layer.py) | Original SAM-based training reference: reruns each bit edit, contracts fixed dY, and uses margin-based scaling. This is not an independent QKV operator. |
| [RWKV-LM QKV 4-bit model](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/260222_rosa4bitLM_L12.py) | Independent Q/K/V symbols, binary successor values and an external amplitude. The public inference file does not disclose its training estimator. |
| [xiaoiecc/qkv-rosa-fast-exact-backward](https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward) | Unlimited single-bit QKV credit using suffix indexes and compressed deletion/repair regions, with Python/C++ implementations. Its optional learned null payload and sigmoid/identity logit maps must be aligned before comparison. |
| [johanwind/wind_rosa](https://github.com/johanwind/wind_rosa) | CUDA bitflip for a bounded match length; linear in T at fixed truncation parameters. Its reference allows empty-suffix fallback to current V and returns 0/1 values, unlike this package's zero null and signed values. |
| [ROSA-Tuning](https://arxiv.org/html/2602.02499v2) | Model integration with run-level folding, query-bit counterfactuals, a run-level K surrogate, and CPU/GPU pipelining. This is not complete independent-bit replay over raw Q/K streams. |
| [aabbdev/rosa](https://github.com/aabbdev/rosa) | Exact SAM backbone with bounded candidate histories, differentiable verification, learned ranking and straight-through selection. Additional ranking/value branches change the architecture; not just the gradient of the same hard operator. |

The xiaoiecc implementation's stated cost includes the number of compiled
regions, Lambda: `O(T*log(T)^2 + Lambda*log(T) + T*D)` time and
`O(T*log(T) + Lambda)` space. Do not transfer that index complexity to this
package's GPU row-band implementation, or omit Lambda when comparing costs.
See its [algorithm description](https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward/blob/main/ALGORITHM.md).

Related work is not the same as code provenance. The README acknowledges
ROSA's original author; this list describes alternative designs without
claiming that each was incorporated into production.

## Fair Evaluation

First compare matched lengths, latest endpoints, nulls and values with a
simple independent oracle. Next compare raw fixed-dY credit under the same
edit convention, then evaluate the logit mapping separately. Bitwise equality
of integer routes, floating tolerance, and bitwise floating reproducibility
are distinct checks.

For training, hold the trunk, head/value widths, initialization, amplitude,
data, loss, optimizer and compute budget fixed. Include failure seeds and
held-out retrieval tasks, not only direct symbol fitting. For speed, report
forward and backward separately as well as end-to-end, including transfers
and index construction when required. Use the [memory measurement scope](MEMORY.md)
and record exact project revisions; this page does not rank unmeasured kernels.
