#pragma once
#include <cuda_runtime.h>

namespace rosa::cuda {
// Dense codes are [BH,T]. Priorities encode length:32 | (K end + 1):32.
// Each warp combines four adjacent unlimited diagonal chains before writing.
static __global__ void match(const int* q, const int* k, unsigned long long* out,
                             int s, int t) {
  int lane = threadIdx.x & 31, id = blockIdx.x * 4 + threadIdx.x / 32;
  int groups = (t - 1 + 3) / 4;
  if (id >= s * groups) return;
  int a = id / groups, first = (id % groups) * 4 + 1;
  int last[4] = {-1, -1, -1, -1};
  for (int start = first; start < t; start += 32) {
    int i = start + lane;
    int code = i < t ? q[a * t + i] : 0;
    unsigned long long best = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      int delta = first + g, j = i - delta;
      bool valid = i < t && j >= 0;
      bool same = valid && code == k[a * t + j];
      unsigned mask = __ballot_sync(0xffffffffu, valid && !same);
      unsigned prefix = mask & (0xffffffffu >> (31 - lane));
      int m = prefix ? start - delta + 31 - __clz(prefix) : last[g];
      if (same) best = max(best, (static_cast<unsigned long long>(j - m) << 32) |
                                  static_cast<unsigned>(j + 1));
      if (mask) last[g] = start - delta + 31 - __clz(mask);
    }
    if (best) atomicMax(out + a * t + i, best);
  }
}
}  // namespace rosa::cuda
