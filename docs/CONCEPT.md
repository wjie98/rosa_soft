# ROSA and RosaAnchor Technical Report

This document explains the training proxy used by `rosa_soft`. It is written
for users who want to understand why the current operator looks the way it
does, not for kernel developers who need every CUDA detail.

The short version:

- `RosaRuntime` is the hard ROSA inference path.
- `rosa_anchor_ops` is the differentiable training proxy.
- The current proxy is the result of several failed or partial routes:
  bitflip perturbation, soft dynamic programming, suffix attention, and finally
  RosaAnchor.

## Design Ladder

The methods below are not independent alternatives. They form a sequence of
repairs. Each step keeps the useful property of the previous step and tries to
remove the bottleneck that made it unsuitable as the main training path.

| Step | Main idea | What it fixed | What still failed |
| --- | --- | --- | --- |
| Hard ROSA | Exact discrete suffix route. | Gives the desired inference behavior. | Almost no useful gradient for Q/K. |
| Bitflip perturbation | Probe hard-route sensitivity by flipping bits. | Keeps hard forward and gives meaningful route gradients. | Too many hard probes; expensive and discontinuous. |
| Soft-DP | Replace exact suffix equality with a differentiable recurrence. | Gives dense gradients to all candidate routes in one soft model. | `O(T^2)`, recurrence-heavy, and too soft compared with hard inference. |
| Suffix attention | Flatten recent suffixes into attention-style fingerprints. | Makes the proxy much more GPU-friendly. | Additive scoring can hide early mismatches and drifts from longest-suffix semantics. |
| RosaAnchor | Use weighted mismatches plus multiplicative suffix decay. | Keeps finite-scale gradients while preserving the hard ROSA limit. | Current production proxy. |

## 1. Hard ROSA

ROSA routes by discrete suffix matching. Q/K activations are converted into
bits:

```math
b(x)=\mathbf{1}[x>0]
```

For a query suffix ending at position `i` and a key suffix ending at position
`j`, hard ROSA prefers the longest exact suffix match. A simple way to write
the hard suffix length is:

```math
L_{i,j}^{\mathrm{hard}}
=
\begin{cases}
1 + L_{i-1,j-1}^{\mathrm{hard}}, & b(q_i)=b(k_j) \\
0, & b(q_i)\ne b(k_j)
\end{cases}
```

The route selects a historical value from the best matching suffix:

```math
y_i = v_{r(i)}
```

where `r(i)` is the selected hard route. This is attractive for inference
because the state can be maintained by a suffix automaton. The training problem
is that bit comparisons and route selection are step functions, so ordinary
backpropagation has little useful signal for learning Q/K.

This is the starting point: the forward semantics are exactly what we want, but
there is no practical gradient. The first repair is therefore not to soften the
forward, but to ask how the hard forward would change if one bit changed.

## 2. Attempt One: Bitflip Perturbation

The most direct proxy is to keep the hard forward path and estimate the effect
of changing one bit at a time.

Let `F(B,V)` be the hard ROSA output from a packed bit state `B` and value
state `V`. Given upstream gradient `g_i` at output position `i`, flipping one
route bit `u` gives a directional score:

```math
\Delta_u
=
\left\langle
  g,\,
  F(\mathrm{flip}_u(B),V)-F(B,V)
\right\rangle
```

That score is then used as a surrogate gradient for the activation that
produced bit `u`, usually with a straight-through estimator around the sign
boundary.

This method is conceptually clean:

- the forward path is exactly hard ROSA;
- it directly asks whether a bit decision matters to the current upstream
  gradient;
- it has strong local exploration because every bit can be probed.

But it is expensive. A useful bitflip gradient needs many hard-operator probes,
and each probe changes a discontinuous route. In small next-token fitting tests
it is a strong oracle, but it is not a practical production training operator.

What bitflip improved over plain hard ROSA:

- it keeps the true hard forward path;
- it turns an upstream activation gradient into a route-bit sensitivity signal;
- it can escape some local route mistakes because it actively probes nearby
  bit decisions.

What still blocks it:

- cost scales with the number of probed bits and positions;
- probes are hard-route executions, not one fused dense GPU pass;
- the perturbation signal is useful as an oracle but awkward as a stable,
  scalable training primitive.

The next repair is soft-DP: instead of probing one changed route at a time, make
all candidate routes differentiable at once.

## 3. Attempt Two: Soft Dynamic Programming

Soft-DP relaxes exact equality into a differentiable match score. Let
`m(i,j)` be a soft match between the current Q/K symbols:

```math
0 \le m(i,j) \le 1
```

A natural soft version of suffix length is:

```math
S_{i,j}
=
m(i,j)\left(1 + S_{i-1,j-1}\right)
```

Then candidate routes can be selected with a softmax:

```math
a_{i,j}
=
\frac{\exp(\alpha S_{i,j})}
{\sum_r \exp(\alpha S_{i,r})}
```

and the output is:

```math
y_i=\sum_j a_{i,j}v_j
```

This is the most faithful differentiable relaxation of longest suffix matching.
Its problem is practical:

- it needs an `O(T^2)` score table;
- the diagonal recurrence is awkward for GPU parallelism;
- the soft forward path can leak information through probability tails;
- training can optimize the soft objective without learning routes that work
  well under hard inference.

Soft-DP remains a useful reference idea, but it is too slow and too soft for
the main path.

What soft-DP improved over bitflip:

- it gives every candidate route a gradient in one forward/backward pass;
- it replaces discrete bit probes with a smooth route distribution;
- it directly models suffix length instead of relying on finite-difference
  perturbations.

What still blocks it:

- the full candidate table is quadratic in sequence length;
- the diagonal recurrence is hard to schedule efficiently on modern GPUs;
- the soft forward objective can be optimized in ways that do not survive the
  switch back to hard ROSA.

The next repair is suffix attention: keep the idea of comparing suffixes, but
remove the recurrence and express the comparison in a GPU-friendly attention
shape.

## 4. Attempt Three: Suffix Attention

Suffix attention tries to keep the "compare recent suffixes" idea while using
an attention-like layout. For a suffix window `W`, define windowed features:

```math
\phi_q(i)=
\left[
q_i,\,
\eta q_{i-1},\,
\eta^2 q_{i-2},\,
\ldots,\,
\eta^{W-1}q_{i-W+1}
\right]
```

```math
\phi_k(j)=
\left[
k_j,\,
\eta k_{j-1},\,
\eta^2 k_{j-2},\,
\ldots,\,
\eta^{W-1}k_{j-W+1}
\right]
```

The route score is an attention score over suffix fingerprints:

```math
s_{i,j}^{\mathrm{sufa}}
=
\frac{\left\langle\phi_q(i),\phi_k(j)\right\rangle}
{\sqrt{WD}}
```

and the output is again a soft weighted sum:

```math
y_i=\sum_j \mathrm{softmax}_j(\alpha s_{i,j}^{\mathrm{sufa}})v_j
```

This is much more GPU-friendly than soft-DP and resembles FlashAttention in
shape. The drawback is semantic: the score is mostly additive. A strong
mismatch at the newest position can still be compensated by other offsets, so
the proxy does not naturally behave like hard longest-suffix matching. In our
small hard-forward training checks, this family was easier to optimize as a
soft model than as a hard ROSA proxy.

What suffix attention improved over soft-DP:

- it removes the diagonal DP dependency;
- it can be implemented with attention-like tiled kernels;
- the suffix window `W` gives a clear cost/semantic knob.

What still blocks it:

- the score is additive across offsets;
- a mismatch near the current token can be compensated by older offsets;
- therefore the proxy can assign high confidence to routes that are not the
  hard longest-suffix route.

The next repair is RosaAnchor: keep the windowed suffix comparison, but replace
the additive score with a multiplicative recurrence so early mismatches suppress
the rest of the suffix.

## 5. Current Path: RosaAnchor

RosaAnchor keeps the useful parts:

- hard sign bits in the route definition;
- a finite-scale soft distribution for gradients;
- multiplicative suffix decay, so an early mismatch suppresses the rest of the
  suffix;
- V gradients kept enabled;
- one main semantic knob, `window_size`.

RosaAnchor is therefore not "soft-DP plus optimizations" or "attention with a
different score." It is the smallest form that kept the useful properties we
needed:

- from bitflip: stay aligned with hard signs and hard-route sensitivity;
- from soft-DP: expose dense gradients over candidate routes;
- from suffix attention: use a fixed suffix window and GPU-friendly candidate
  parallelism;
- new in RosaAnchor: make mismatch effects multiplicative so the hard ROSA
  limit is recovered as scale grows.

### Route Bits and Values

Q/K route bits use hard signs:

```math
b(x)=\mathbf{1}[x>0]
```

V uses signed values in the proxy:

```math
s_v(x)=
\begin{cases}
+1, & x>0 \\
-1, & x\le 0
\end{cases}
```

The backward path uses a fixed softsign-style straight-through derivative:

```math
\frac{\partial s_v(x)}{\partial x}
\approx
\frac{1}{(1+|x|)^2}
```

### Weighted Mismatch

For candidate `j` and suffix offset `l`, the hard bit mismatch is:

```math
\delta_{l,d}(i,j)
=
b(q_{i-l,d}) \oplus b(k_{j-1-l,d})
```

RosaAnchor uses magnitude only as reliability for mismatched bits. The
confidence of a scalar is:

```math
c(x)=\tanh(|x|)
```

The per-bit mismatch weight is:

```math
w_{l,d}(i,j)
=
0.25 + 0.75\min(c(q_{i-l,d}),c(k_{j-1-l,d}))
```

The weighted mismatch cost is:

```math
h_l(i,j)
=
\sum_d w_{l,d}(i,j)\delta_{l,d}(i,j)
```

Exact sign matches still contribute no mismatch. Because the minimum mismatch
weight is positive, any sign mismatch disappears as scale grows.

### Multiplicative Suffix Score

The local match term is:

```math
\mu_l(i,j)=\exp(-\lambda h_l(i,j))
```

The suffix score is a product recurrence:

```math
p_0(i,j)=1
```

```math
p_{l+1}(i,j)=p_l(i,j)\mu_l(i,j)
```

```math
R(i,j)=\sum_{l=0}^{L(i,j)-1}p_{l+1}(i,j)
```

where:

```math
L(i,j)=\min(W,i+1,j)
```

This is the main difference from suffix attention. In suffix attention, offsets
mostly add evidence. In RosaAnchor, the product term means one strong mismatch
reduces the influence of all later offsets.

This fixes the core suffix-attention failure. A route can no longer hide a
nearby mismatch by accumulating enough weak matches farther back in the window.
That is the behavior needed to approach hard longest-suffix routing.

The mismatch sharpness is tied to selector scale:

```math
\lambda=\log W+1.5\log(1+|\alpha|)
```

where `alpha` is the route scale.

### Candidate Distribution

RosaAnchor includes a null candidate with zero value contribution. This is
similar in spirit to an attention sink: it gives the model a way to say that no
history route is useful.

The non-null candidate logit is:

```math
z(i,j)
=
\alpha
\left(
R(i,j)-0.5+\rho(i,j)
\right)
```

The recency term is intentionally weak:

```math
\rho(i,j)=0.25\frac{j}{\max(i,1)}
```

It only breaks near ties in favor of recent matches. It should not dominate the
suffix score.

The distribution over candidates is:

```math
P(j|i)
=
\frac{\exp(z(i,j))}
{1+\sum_{1\le r\le i}\exp(z(i,r))}
```

The null candidate probability is:

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

As `scale` grows, the distribution becomes sharper. With a large enough window
and a unique best route, RosaAnchor approaches hard ROSA.

This fixes the soft-DP failure mode at the other end of training: the proxy can
be soft at finite scale for gradients, but it has a clear hard limit. The model
does not need to learn a separate behavior for inference.

## 6. Scale and Telemetry

`scale=None` is the recommended starting point. It uses a shape-based estimate.
For longer training runs, use `RosaAnchorScaleController`, which periodically
measures route confidence and adjusts scale toward a target top probability.
The default target is `0.8`.

If `p_t` is the observed top probability and `bar_p_t` is its EMA:

```math
\bar{p}_t
=
\beta\bar{p}_{t-1}+(1-\beta)p_t
```

the controller updates scale in logit space:

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

Telemetry is not meant for every call. The ordinary user path should look like:

```python
y = rosa_anchor_ops(q, k, v, window_size=128)
```

The controller enables telemetry only at probe steps.

## 7. Parameter Guide

| Parameter | Default | Recommendation |
| --- | ---: | --- |
| `window_size` | `32` | Main semantic knob. Increase for longer suffix behavior. CUDA backward currently supports up to `512`. |
| `scale` | `None` | Keep `None` first. Use the controller for calibration instead of manual tuning. |
| `return_telemetry` | `False` | Keep off in the hot path. Sample periodically for scale control. |
| `logit_epsilon` | `0.0` | Exact within the selected window. Try `1e-3` only after validating task metrics. |
| `qk_damper_strength` | `0.0` | No damping by default. Try `0.1..0.3` only if Q/K saturation collapses route entropy. |

## 8. Validation Snapshot

These checks focus on the operator contract: hard ROSA forward with different
surrogate backward rules.

### Hard-Limit Check

On a random Q/K/V probe with `window_size=12`, increasing scale made RosaAnchor
approach hard ROSA:

| Scale | MAE vs hard output | Max error |
| ---: | ---: | ---: |
| `4` | `0.24676017` | `0.75562179` |
| `16` | `0.02811864` | `0.15165848` |
| `64` | `0.00000115` | `0.00005531` |
| `256` | `0.00000000` | `0.00000000` |
| `1024` | `0.00000000` | `0.00000000` |

### Single-Sample Next-Token Fit

The fitting test used a tiny repeated-motif next-token task:

```text
tokens -> embedding -> Q/K/V projections -> hard ROSA forward
       -> output projection -> LM head -> cross entropy
```

The forward path was hard ROSA for every method. Only the backward proxy
changed.

| Method | Train hard CE | Train hard acc |
| --- | ---: | ---: |
| bitflip perturbation | `0.000161` | `1.000` |
| current weighted RosaAnchor CUDA | `0.000271` | `1.000` |

Readout: the current weighted RosaAnchor path gets close to the bitflip oracle
on this fitting problem while remaining a single CUDA training operator.

## 9. Why This Is the Mainline

| Method | What it taught us | Why it is not the current public path |
| --- | --- | --- |
| Bitflip perturbation | Hard-forward gradients can solve small route/value tasks. | Too many hard probes for production training. |
| Soft-DP | The closest differentiable suffix formulation. | `O(T^2)`, recurrence-heavy, and too soft under hard inference. |
| Suffix attention | A GPU-friendly suffix fingerprint view. | Additive scores do not enforce hard longest-suffix behavior well enough. |
| RosaAnchor | Multiplicative mismatch decay preserves the hard limit and trains efficiently. | Current mainline. |

The design goal is not to make the smoothest possible soft model. The goal is
to train parameters that will still work when inference uses hard ROSA.

## 10. Runtime Contract

Training uses `rosa_anchor_ops`:

```python
y = rosa_anchor_ops(q, k, v, window_size=128)
```

Inference uses `RosaRuntime`:

```python
with RosaRuntime(num_heads, num_value_heads, qk_bits=4, value_bits=4) as rt:
    out, endpos = rt.update(q, k, v, return_packed=False)
```

`RosaRuntime` stores the hard suffix state explicitly. It supports packed
Q/K/V symbols and an optional CUDA-to-CPU staging path for overlap with GPU
work.
