# Vendored libsais

This research backend vendors the C99 implementation of
[`libsais`](https://github.com/IlyaGrebnov/libsais) at commit
`b6e52ef33fe14f9d5c14c580d162b6fd2c27f2a8`.

Included upstream files:

- `include/libsais.h`
- `src/libsais.c`
- `LICENSE`

The upstream project is Apache-2.0 licensed.  Files are kept unmodified.  The
ROSA build helper compiles them into an optional shared library used only by
the exact filtered-bitflip research path.  The bridge calls `libsais_int` and
`libsais_plcp_int`: packed ROSA bytes plus two distinct sentinels require up to
258 ordered symbols and therefore do not fit the byte-only API.
