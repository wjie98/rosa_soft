# RosaSoft Documentation

The documentation is divided by stability and audience. Production behavior
is described separately from research evidence so experimental mechanisms do
not accidentally become part of the supported operator contract.

## Production Documentation

| Document | Purpose |
| --- | --- |
| [Production Guide](PRODUCTION_GUIDE.md) | Public API, build variants, code ownership, runtime lifecycle, validation, and release workflow. Start here when maintaining or integrating the package. |
| [Concept](CONCEPT.md) | Mathematical definition of exact hard forward and dense surrogate backward. |
| [Design](ROSA_SOFT_DESIGN.md) | Python, C++, CUDA, autograd, execution-plan, and packed-layout implementation boundaries. |
| [Reference Guide](ROSA_SOFT_REFERENCE.md) | Equation-level PyTorch oracle and inspection tensor conventions. |
| [Dense Reference Freeze](DENSE_REFERENCE_FREEZE.md) | Frozen baseline, reproduction snapshot, tag policy, and known limits. |

The supported package behavior is the intersection of the public Python API,
the shared validation contract, and the frozen semantic documents. Internal
`torch.ops` schemas, testing helpers, and diagnostics are not additional
training APIs.

## Research Documentation

Files under [`research/`](research/) record rejected mechanisms, ablations,
kernel experiments, and possible future estimator families. They are evidence,
not production configuration. A research result changes the supported surface
only after an explicit implementation decision, production parity, and an
update to the documents above.

## Reading Paths

For model integration:

1. Read [Production Guide](PRODUCTION_GUIDE.md), sections 2 through 6.
2. Use the dense or packed example in the root [README](../README.md).
3. Check the numerical and integration boundaries before choosing dtype or
   dropout settings.

For operator maintenance:

1. Read [Production Guide](PRODUCTION_GUIDE.md) completely.
2. Read [Design](ROSA_SOFT_DESIGN.md) for execution-plan ownership.
3. Use [Reference Guide](ROSA_SOFT_REFERENCE.md) as the parity oracle.
4. Apply the repository constraints in [`AGENTS.md`](../AGENTS.md).

For estimator research:

1. Treat the tag `rosa-soft-dense-reference-v1` as the control.
2. Keep new operators outside the frozen package surface.
3. Record hypotheses and results under `docs/research/`, `benchmarks/`, and
   `validation/`.
