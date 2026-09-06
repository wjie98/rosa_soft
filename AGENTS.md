# ROSA Production Constraints

These rules are part of the operator contract.

1. Hard forward is exact, causal, latest-longest, and unlimited. Never add a
   suffix window, candidate cap, pruning, sampling, or approximate winner to
   the production path.
2. Training forward must contain only hard binary ROSA values. Soft state may
   exist only in the custom backward; otherwise the model can exploit numeric
   leakage and diverge from inference.
3. Backward includes every causal candidate and uses the frozen dense suffix
   recurrence. Do not replace it with sparse gradients without a new operator
   and independent training evidence.
4. Public API is only `RosaSam`, `rosa_hard`, and `rosa_soft`. Dense and packed
   layouts share `rosa_soft`; do not add `varlen`, `unbounded`, `anchor`, or
   schedule-specific aliases.
5. `scale`, `dropout_p`, and `mismatch_scale` are static user parameters.
   Do not reintroduce automatic temperature, lambda, or context-length
   schedules.
6. CPU SAM is the exact inference and validation implementation. CUDA hard DP
   is the exact training forward. They must stay bitwise aligned with a naive
   definition-level oracle, including no-match `-1`, successor `V[end + 1]`,
   GQA head mapping, and latest-position ties.
7. Dense backward may stream, tile, checkpoint, or recompute, but optimization
   must not change its mathematical VJP. Full `T x T` candidate state must not
   become an API requirement.
8. Prefer a small native ABI and generic internal names. Experimental kernels,
   benchmark-only operators, compatibility branches, and generated validation
   data belong on a research branch or tagged commit, not in production.
9. Tests must use simple independent DP/math oracles. Do not establish
   correctness by comparing two descendants of the same optimized kernel.
10. Before changing kernels, verify dense and packed hard semantics, all seven
    Q/K/V gradient masks, FP16/BF16/FP32, dropout, GQA, empty packed segments,
    and `torch.compile`.

Archive points:

- `rosa-soft-research-archive-v1` (`582fe45`): complete kernel and estimator
  research tree before cleanup.
- `rosa-soft-dense-unbounded-v1`: frozen minimal dense estimator.
