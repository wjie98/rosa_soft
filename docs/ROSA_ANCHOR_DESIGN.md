# RosaAnchor Operator Report

This document focuses on the current public operator, `rosa_anchor_ops`.
The broader history from bitflip perturbation to soft-DP, suffix attention, and
RosaAnchor is now merged into [CONCEPT.md](CONCEPT.md).

## 1. What RosaAnchor Is

RosaAnchor is a differentiable training proxy for hard ROSA. It is not the hard
inference runtime itself. The intended contract is:

- train with `rosa_anchor_ops`;
- infer with `RosaRuntime`;
- keep the proxy close enough to hard ROSA that learned Q/K/V projections
  transfer to hard suffix routing.

The simplest training call is:

```python
y = rosa_anchor_ops(q, k, v, window_size=128)
```

The corresponding hard runtime call is:

```python
with RosaRuntime(num_heads, num_value_heads, qk_bits=4, value_bits=4) as rt:
    out, endpos = rt.update(q, k, v, return_packed=False)
```

RosaAnchor exists because the earlier routes each solved only part of the
problem:

| Previous step | Useful idea kept | Problem RosaAnchor fixes |
| --- | --- | --- |
| Bitflip perturbation | Hard signs and hard-forward route sensitivity matter. | Avoids per-bit hard probes by using one dense soft route distribution. |
| Soft-DP | Candidate routes should receive dense gradients. | Avoids full `O(T^2)` DP state and a soft forward that drifts from hard inference. |
| Suffix attention | A fixed suffix window is GPU-friendly. | Replaces additive suffix evidence with multiplicative mismatch decay. |

The result is a finite-scale proxy that still has a hard ROSA limit as scale
increases.

## 2. Tensor Layout

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `query` | `[B, T, H, D]` | Route logits for Q. |
| `key` | `[B, T, H, D]` | Route logits for K. |
| `value` | `[B, T, H_v, D_v]` | Value logits. |

`H % H_v == 0`, so value heads are grouped-query style. The training CUDA
operator supports `1 <= D <= 32`. The hard runtime packs Q/K/V symbols and
supports `1..8` bits.

## 3. Hard Signs, Soft Gradients

Q/K route bits are hard:

```math
b(x)=\mathbf{1}[x>0]
```

V is also represented as signed values in the proxy:

```math
s_v(x)=
\begin{cases}
+1, & x>0 \\
-1, & x\le 0
\end{cases}
```

Backward uses a fixed softsign-style straight-through derivative:

```math
\frac{\partial s_v(x)}{\partial x}
\approx
\frac{1}{(1+|x|)^2}
```

This is deliberately not exposed as a user knob.

## 4. Candidate Window

For query position `i`, candidate `j` means "use the historical value attached
to the key prefix ending at `j - 1`". Only a suffix window is compared:

```math
L(i,j)=\min(W,i+1,j)
```

where `W = window_size`. Offset `l = 0` compares `q_i` with `k_{j-1}`; offset
`l = 1` compares `q_{i-1}` with `k_{j-2}`.

## 5. Weighted Mismatch

The hard bit mismatch at offset `l` is:

```math
\delta_{l,d}(i,j)
=
b(q_{i-l,d}) \oplus b(k_{j-1-l,d})
```

RosaAnchor uses activation magnitude only as mismatch reliability:

```math
c(x)=\tanh(|x|)
```

```math
w_{l,d}(i,j)
=
0.25+0.75\min(c(q_{i-l,d}),c(k_{j-1-l,d}))
```

```math
h_l(i,j)
=
\sum_d w_{l,d}(i,j)\delta_{l,d}(i,j)
```

Exact sign matches remain exact. Mismatches with high confidence are penalized
more strongly, while low-confidence mismatches still have a positive minimum
penalty.

## 6. Multiplicative Suffix Score

The local match is:

```math
\mu_l(i,j)=\exp(-\lambda h_l(i,j))
```

The suffix score is:

```math
p_0(i,j)=1
```

```math
p_{l+1}(i,j)=p_l(i,j)\mu_l(i,j)
```

```math
R(i,j)=\sum_{l=0}^{L(i,j)-1}p_{l+1}(i,j)
```

This multiplicative recurrence is the central design choice. Once a strong
mismatch appears, later suffix positions are discounted. That makes the proxy
closer to hard longest-suffix routing than an additive suffix-attention score.

The match sharpness follows the route scale:

```math
\lambda=\log W+1.5\log(1+|\alpha|)
```

where `alpha = scale`.

## 7. Route Distribution

RosaAnchor includes a null candidate with zero value contribution. The non-null
candidate logit is:

```math
z(i,j)
=
\alpha
\left(
R(i,j)-0.5+\rho(i,j)
\right)
```

The weak recency tie-break is:

```math
\rho(i,j)=0.25\frac{j}{\max(i,1)}
```

The candidate probability is:

```math
P(j|i)
=
\frac{\exp(z(i,j))}
{1+\sum_{1\le r\le i}\exp(z(i,r))}
```

The null probability is:

```math
P(\varnothing|i)
=
\frac{1}
{1+\sum_{1\le r\le i}\exp(z(i,r))}
```

The output is:

```math
y_i
=
\sum_{j=1}^{i}P(j|i)s_v(v_j)
```

As `scale` grows, the distribution becomes sharper. With a large enough
`window_size` and a unique best route, the proxy approaches hard ROSA.

## 8. Parameters

| Parameter | Default | Recommendation |
| --- | ---: | --- |
| `window_size` | `32` | Main semantic knob. Increase for longer suffix behavior. CUDA backward currently supports up to `512`. |
| `scale` | `None` | Recommended. Uses a built-in shape estimate. Use the scale controller instead of manual tuning. |
| `return_telemetry` | `False` | Leave off in the hot path. Probe periodically for scale control. |
| `logit_epsilon` | `0.0` | Exact inside the selected window. Try `1e-3` only after validating task metrics. |
| `qk_damper_strength` | `0.0` | No damping. Try `0.1..0.3` only if Q/K saturation collapses route entropy. |

## 9. Scale Control

For longer training, use `RosaAnchorScaleController`. It periodically asks the
operator for telemetry and adjusts scale toward a target top probability. The
recommended target is `0.8`.

If `p_t` is the observed top probability and `bar_p_t` is its EMA:

```math
\bar{p}_t
=
\beta\bar{p}_{t-1}+(1-\beta)p_t
```

then the controller updates scale as:

```math
\alpha
\leftarrow
\alpha
\exp\left(
  \eta
  \left[
    \mathrm{logit}(p_{\mathrm{target}})
    -
    \mathrm{logit}(\bar{p}_t)
  \right]
\right)
```

This keeps tuning tied to a distribution-level signal instead of raw sequence
length or activation magnitude.

## 10. Validation Snapshot

Hard-limit probe with `window_size=12`:

| Scale | MAE vs hard output | Max error |
| ---: | ---: | ---: |
| `4` | `0.24676017` | `0.75562179` |
| `16` | `0.02811864` | `0.15165848` |
| `64` | `0.00000115` | `0.00005531` |
| `256` | `0.00000000` | `0.00000000` |

Single-sample hard-forward next-token fit:

| Method | Train hard CE | Train hard acc |
| --- | ---: | ---: |
| bitflip perturbation | `0.000161` | `1.000` |
| current weighted RosaAnchor CUDA | `0.000271` | `1.000` |

These operator-level checks show that the current weighted RosaAnchor path
preserves the hard limit and closes the local fitting gap to the bitflip oracle.
