#pragma once
#include <torch/extension.h>

#include <cstdint>
#include <tuple>

namespace rosa::bitflip {

using U = unsigned long long;
using Tensor = torch::Tensor;
using Output = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using Grads = std::tuple<Tensor, Tensor, Tensor>;

// Private FP32 credit: Q/K are [BH,D,T], common is [BH,2,T].
struct Credit {
  Tensor q, k, common;
};
#ifdef __CUDACC__
// Route: length:32 | (K end + 1):32. Zero denotes null.
__host__ __device__ inline U priority(int n, int j) {
  return n > 0 ? (static_cast<U>(n) << 32) | static_cast<unsigned>(j + 1) : 0;
}
// Record: exact length:28 | repaired length:28 | (mismatch bit + 1):6.
__device__ inline U record(int n, int r, int bit) {
  return (static_cast<U>(n) << 34) | (static_cast<U>(r) << 6) | (bit + 1);
}
__device__ inline U stamp(U epoch, int r, int j, int shift) {
  return (epoch << (2 * shift)) | (static_cast<U>(r) << shift) | (j + 1);
}
#endif

void check_inputs(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
torch::Tensor hard(const torch::Tensor&, const torch::Tensor&);
torch::Tensor pack_values(const torch::Tensor&);
Output forward(const Tensor&, const Tensor&, const Tensor&);
Credit credit(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
              const Tensor&, int d, int rows);
Grads backward(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
               const Tensor&, const Tensor&, const Tensor&, int d, int rows, int mask);
}  // namespace rosa::bitflip
