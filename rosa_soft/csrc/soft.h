#pragma once
#include <torch/extension.h>
#include <tuple>

namespace rosa::soft {
using Tensor = torch::Tensor;
using Grads = std::tuple<Tensor, Tensor, Tensor>;

// Host-only arguments. Device kernels retain their explicit pointer parameters.
struct Input {
  Tensor q, k, v, dy, pq, pk, seed;
};
struct Args {
  float scale, dropout, mismatch;
};

inline Grads gradients(const Input& x, int mask) {
  auto allocate = [mask](const Tensor& v, int bit) {
    const auto options = v.options().dtype(torch::kFloat32);
    return mask & bit ? torch::zeros(v.sizes(), options) : torch::empty({0}, options);
  };
  return {allocate(x.q, 1), allocate(x.k, 2), allocate(x.v, 4)};
}

Grads backward(const Input&, const Args&, int mask);
Grads backward_fp16(const Input&, const Args&, int mask, const Tensor& prior);
Tensor stats_fp16(const Input&, const Args&, const Tensor& prior);
void finish(const Input&, Grads&);
}  // namespace rosa::soft
