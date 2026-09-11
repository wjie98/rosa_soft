#include <ATen/Context.h>
#include "soft.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

using Tensors = rosa::soft::Grads;
using rosa::soft::Args;

Tensors rosa_hard_cuda(const torch::Tensor&, const torch::Tensor&,
                       const torch::Tensor&, const torch::Tensor&);

namespace {

constexpr int64_t kPad = 128;

bool dtype_ok(c10::ScalarType t) {
  return t == torch::kFloat || t == torch::kHalf ||
      t == torch::kBFloat16;
}

float positive_float(double x, const char* name) {
  const float y = static_cast<float>(x);
  TORCH_CHECK(
      std::isfinite(x) && y > 0.0f && std::isnormal(y),
      name, " must be a positive normal float32 value");
  return y;
}

Args check_args(double scale, double dropout, double mismatch, int64_t n) {
  TORCH_CHECK(
      std::isfinite(dropout) && dropout >= 0.0 &&
          dropout <= 1.0 - std::ldexp(1.0, -24),
      "dropout_p must be in [0, 1 - 2^-24]");
  const float s = positive_float(scale, "scale");
  const float a = positive_float(mismatch, "mismatch_scale");
  const float p = static_cast<float>(dropout);
  const double fmax = std::numeric_limits<float>::max();
  TORCH_CHECK(scale <= fmax / n, "sequence length * scale overflows float32");
  TORCH_CHECK(
      mismatch <= fmax / (n * scale),
      "sequence length * scale * mismatch_scale overflows float32");
  return {s, p, a};
}

bool packed(const torch::Tensor& q) {
  return q.dim() == 3;
}

void check_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && cu.is_cuda(),
              "q, k, v, and cu_seqlens must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == cu.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(dtype_ok(q.scalar_type()),
              "q, k, and v must be float32, float16, or bfloat16");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == v.scalar_type(),
              "q, k, and v must have the same dtype");
  TORCH_CHECK(q.dim() == k.dim() && (q.dim() == 3 || q.dim() == 4),
              "q and k must have shape [B,T,H,D] or [N,H,D]");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");

  const bool p = packed(q);
  const int64_t n = p ? q.size(0) : q.size(1);
  const int64_t h = p ? q.size(1) : q.size(2);
  const int64_t d = p ? q.size(2) : q.size(3);
  TORCH_CHECK(p ? v.dim() == 3 : v.dim() == 4,
              "v rank must match q/k layout");
  const int64_t hv = p ? v.size(1) : v.size(2);
  TORCH_CHECK(p || q.size(0) > 0, "batch size must be positive");
  TORCH_CHECK(n > 0 && h > 0 && d > 0 && d <= 32,
              "sequence/head dimensions must be positive and D must be in [1,32]");
  TORCH_CHECK((p ? v.size(0) == q.size(0)
                 : v.size(0) == q.size(0) && v.size(1) == q.size(1)),
              "q, k, and v token dimensions must match");
  TORCH_CHECK(hv > 0 && h % hv == 0 && v.size(-1) > 0,
              "H must be divisible by value heads and value width must be positive");
  TORCH_CHECK(q.numel() / d <= std::numeric_limits<int>::max(),
              "token-head count exceeds int32 indexing");
  TORCH_CHECK(n <= std::numeric_limits<int>::max() - kPad,
              "sequence length exceeds int32 indexing");
  TORCH_CHECK(v.size(-1) <= std::numeric_limits<int>::max() - kPad,
              "value width exceeds int32 indexing");

  TORCH_CHECK(cu.scalar_type() == torch::kInt32 && cu.dim() == 1,
              "cu_seqlens must be a one-dimensional int32 tensor");
  if (p) {
    TORCH_CHECK(cu.numel() >= 2,
                "packed input requires at least two sequence offsets");
    TORCH_CHECK(cu.numel() - 1 <= std::numeric_limits<int>::max(),
                "sequence count exceeds int32 indexing");
  } else {
    TORCH_CHECK(cu.numel() == 0,
                "dense input must not provide sequence offsets");
  }
}

void check_backward(
    const torch::Tensor& q,
    const torch::Tensor& v,
    const torch::Tensor& dy,
    const torch::Tensor& pq,
    const torch::Tensor& pk,
    const torch::Tensor& seed,
    double dropout,
    int64_t mask) {
  TORCH_CHECK(dy.is_cuda() && dy.device() == q.device() &&
                  dy.scalar_type() == q.scalar_type(),
              "dy must match q device and dtype");
  if (packed(q)) {
    TORCH_CHECK(dy.sizes() == torch::IntArrayRef({q.size(0), q.size(1), v.size(2)}),
                "dy shape mismatch");
    TORCH_CHECK(pq.sizes() == torch::IntArrayRef({q.size(1), q.size(0)}),
                "packed q symbol shape mismatch");
  } else {
    TORCH_CHECK(dy.sizes() == torch::IntArrayRef(
                    {q.size(0), q.size(1), q.size(2), v.size(3)}),
                "dy shape mismatch");
    TORCH_CHECK(pq.sizes() == torch::IntArrayRef({q.size(0), q.size(2), q.size(1)}),
                "packed q symbol shape mismatch");
  }
  TORCH_CHECK(pk.sizes() == pq.sizes() && pq.scalar_type() == torch::kInt32 &&
                  pk.scalar_type() == torch::kInt32 && pq.device() == q.device() &&
                  pk.device() == q.device(),
              "packed q/k symbols are invalid");
  TORCH_CHECK(seed.is_cuda() && seed.device() == q.device() &&
                  seed.scalar_type() == torch::kInt64 &&
                  seed.numel() == (dropout > 0.0 ? 1 : 0),
              "dropout seed is invalid");
  TORCH_CHECK(mask >= 1 && mask <= 7, "gradient mask must be in [1,7]");
}

Tensors backward_packed(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& dy,
    const torch::Tensor& pq,
    const torch::Tensor& pk,
    const torch::Tensor& seed,
    const torch::Tensor& cu,
    const Args& a,
    int mask) {
  const auto f32 = q.options().dtype(torch::kFloat32);
  auto dq = mask & 1 ? torch::zeros(q.sizes(), f32) : torch::empty({0}, f32);
  auto dk = mask & 2 ? torch::zeros(k.sizes(), f32) : torch::empty({0}, f32);
  auto dv = mask & 4 ? torch::zeros(v.sizes(), f32) : torch::empty({0}, f32);
  const auto host = cu.cpu();
  const auto* off = host.data_ptr<int32_t>();
  const int64_t batch = host.numel() - 1;
  TORCH_CHECK(off[0] == 0 && off[batch] == q.size(0),
              "cu_seqlens must span all packed tokens");

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t start = off[b];
    const int64_t length = off[b + 1] - start;
    TORCH_CHECK(length >= 0, "cu_seqlens must be nondecreasing");
    if (length == 0) continue;
    const auto slice = [start, length](const torch::Tensor& x) {
      return x.narrow(0, start, length).unsqueeze(0);
    };
    const auto bits = [start, length](const torch::Tensor& x) {
      return x.narrow(1, start, length).unsqueeze(0).contiguous();
    };
    const auto local_seed = a.dropout > 0.0f ? seed.add(b) : seed;
    auto grad = rosa::soft::backward(
        {slice(q), slice(k), slice(v), slice(dy), bits(pq), bits(pk), local_seed}, a, mask);
    if (mask & 1) dq.narrow(0, start, length).copy_(std::get<0>(grad).squeeze(0));
    if (mask & 2) dk.narrow(0, start, length).copy_(std::get<1>(grad).squeeze(0));
    if (mask & 4) dv.narrow(0, start, length).copy_(std::get<2>(grad).squeeze(0));
  }
  return {dq, dk, dv};
}

}  // namespace

Tensors rosa_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu) {
  check_inputs(q, k, v, cu);
  return rosa_hard_cuda(q.contiguous(), k.contiguous(), v.contiguous(),
                        cu.contiguous());
}

Tensors rosa_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor dy,
    torch::Tensor pq,
    torch::Tensor pk,
    torch::Tensor seed,
    torch::Tensor cu,
    double scale,
    double dropout,
    double mismatch,
    int64_t mask) {
  check_inputs(q, k, v, cu);
  check_backward(q, v, dy, pq, pk, seed, dropout, mask);
  const int64_t n = packed(q) ? q.size(0) : q.size(1);
  const Args a = check_args(scale, dropout, mismatch, n);
  at::globalContext().alertNotDeterministic("rosa_soft::backward");
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  dy = dy.contiguous();
  pq = pq.contiguous();
  pk = pk.contiguous();
  seed = seed.contiguous();
  cu = cu.contiguous();

  if (packed(q)) {
    return backward_packed(
        q, k, v, dy, pq, pk, seed, cu, a, static_cast<int>(mask));
  }
  return rosa::soft::backward({q, k, v, dy, pq, pk, seed}, a, static_cast<int>(mask));
}

TORCH_LIBRARY_IMPL(rosa_soft, CUDA, m) {
  m.impl("forward", &rosa_forward);
  m.impl("backward", &rosa_backward);
}
