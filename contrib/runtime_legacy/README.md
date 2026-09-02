# Legacy Inference Runtime

This directory preserves the former asynchronous `RosaRuntime`, its optimized
suffix automaton, and its deployment-oriented tests. It is not part of the
`rosa_soft` package, extension build, or default pytest suite.

The maintained training library uses the route-only `RosaSam` implementation
in `rosa_soft/csrc/rosa_sam_core.h`. Paging, compression, scheduling, and
long-context inference state belong in a separate inference framework.
