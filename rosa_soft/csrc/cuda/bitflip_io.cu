#include <ATen/Context.h>
#include "../dispatch.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "bitflip.cuh"
#include "hard.cuh"

namespace rosa::bitflip {
namespace {

template <class F>
__global__ void pack(const F* q, const F* k, int* pq, int* pk, int total, int t, int h,
                     int d) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= total)
    return;
  int a = row % h, i = (row / h) % t, b = row / (h * t), out = (b * h + a) * t + i;
  unsigned x = 0, y = 0;
  for (int z = 0; z < d; ++z) {
    x |= static_cast<unsigned>(
             static_cast<float>(q[static_cast<int64_t>(row) * d + z]) > 0)
         << z;
    y |= static_cast<unsigned>(
             static_cast<float>(k[static_cast<int64_t>(row) * d + z]) > 0)
         << z;
  }
  pq[out] = x;
  pk[out] = y;
}

template <class F>
__global__ void gather(const F* v, const U* route, F* out, int64_t size, int t, int h,
                       int hv, int d) {
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= size)
    return;
  int z = p % d, a = (p / d) % h, i = (p / (d * h)) % t,
      b = p / (static_cast<int64_t>(d) * h * t);
  int j = static_cast<unsigned>(route[(b * h + a) * t + i]);
  float value = 0.f;
  if (j)
    value = static_cast<float>(
                v[((static_cast<int64_t>(b) * t + j) * hv + a / (h / hv)) * d + z]) > 0
                ? 1.f
                : -1.f;
  out[p] = static_cast<F>(value);
}

template <class F>
__global__ void pack_v(const F* v, int* out, int64_t size, int t, int hv, int d) {
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= size)
    return;
  int words = (d + 31) / 32, word = p % words, i = (p / words) % t,
      a = (p / (words * t)) % hv, b = p / (static_cast<int64_t>(words) * t * hv);
  unsigned bits = 0;
  for (int z = 0; z < 32 && word * 32 + z < d; ++z)
    bits |=
        static_cast<unsigned>(
            static_cast<float>(
                v[((static_cast<int64_t>(b) * t + i) * hv + a) * d + word * 32 + z]) >
            0)
        << z;
  out[p] = bits;
}

template <class F>
__global__ void qk_grad(const F* q, const F* k, const float* dq, const float* dk,
                        const float* common, F* oq, F* ok, int64_t size, int t, int h,
                        int d, bool need_q,
                        bool need_k) {
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= size)
    return;
  int z = p % d, a = (p / d) % h, i = (p / (d * h)) % t,
      b = p / (static_cast<int64_t>(d) * h * t), s = b * h + a;
  int64_t raw = (static_cast<int64_t>(s) * d + z) * t + i;
  if (need_q) {
    float x = static_cast<float>(q[p]), z = 1 + fabsf(x);
    float g = dq[raw] + common[static_cast<int64_t>(s) * 2 * t + i];
    oq[p] = static_cast<F>((x > 0 ? -.5f : .5f) * g / (z * z));
  }
  if (need_k) {
    float x = static_cast<float>(k[p]), z = 1 + fabsf(x);
    float g = dk[raw] + common[(static_cast<int64_t>(s) * 2 + 1) * t + i];
    ok[p] = static_cast<F>((x > 0 ? -.5f : .5f) * g / (z * z));
  }
}

template <class F>
__global__ void scatter(const F* dy, const U* route, float* dv, int64_t size, int t,
                        int h, int hv, int d) {
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= size)
    return;
  int z = p % d, a = (p / d) % h, i = (p / (d * h)) % t,
      b = p / (static_cast<int64_t>(d) * h * t);
  int j = static_cast<unsigned>(route[(b * h + a) * t + i]);
  if (j)
    atomicAdd(dv + ((static_cast<int64_t>(b) * t + j) * hv + a / (h / hv)) * d + z,
              static_cast<float>(dy[p]));
}

template <class F>
__global__ void v_grad(const F* v, const float* dv, F* out, int64_t size) {
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p < size) {
    float x = 1 + fabsf(static_cast<float>(v[p]));
    out[p] = static_cast<F>(dv[p] / (x * x));
  }
}

void check(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 4,
              "contiguous CUDA BTHD input required");
}
}  // namespace

void check_inputs(const torch::Tensor& q, const torch::Tensor& k,
                  const torch::Tensor& v) {
  check(q);
  check(k);
  check(v);
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "device mismatch");
  TORCH_CHECK(
      q.sizes() == k.sizes() && q.size(0) == v.size(0) && q.size(1) == v.size(1),
      "shape mismatch");
  TORCH_CHECK(q.size(0) > 0 && q.size(2) > 0 && q.size(3) > 0 && q.size(3) <= 32 &&
                  v.size(2) > 0 && q.size(2) % v.size(2) == 0 && v.size(3) > 0,
              "invalid dimensions");
  TORCH_CHECK(
      (q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16 ||
       q.scalar_type() == torch::kFloat32) &&
          k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
      "matching FP16/BF16/FP32 required");
  TORCH_CHECK(q.size(1) < (1 << 28) && q.numel() < (1LL << 30) &&
                  q.size(0) * q.size(2) < (1LL << 30),
              "index limit exceeded");
  TORCH_CHECK(v.numel() < (1LL << 30) && q.size(2) * v.size(3) < (1LL << 30) &&
                  q.size(0) * q.size(1) * q.size(2) * v.size(3) < (1LL << 30),
              "value index limit exceeded");
}

Output forward(const Tensor& q, const Tensor& k, const Tensor& v) {
  check_inputs(q, k, v);
  c10::cuda::CUDAGuard guard(q.device());
  int b = q.size(0), t = q.size(1), h = q.size(2), d = q.size(3), hv = v.size(2),
      dv = v.size(3);
  auto pq = torch::empty({b * h, t}, q.options().dtype(torch::kInt32)),
       pk = torch::empty_like(pq);
  auto out = torch::empty({b, t, h, dv}, v.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_ROSA_FLOAT_TYPES(q.scalar_type(), "bitflip_pack", [&] {
                                    if (t)
                                      pack<<<(b * t * h + 255) / 256, 256, 0, stream>>>(
                                          q.data_ptr<scalar_t>(),
                                          k.data_ptr<scalar_t>(), pq.data_ptr<int>(),
                                          pk.data_ptr<int>(), b * t * h, t, h, d);
                                  });
  auto route = hard(pq, pk);
  DISPATCH_ROSA_FLOAT_TYPES(v.scalar_type(), "bitflip_gather",
      [&] {
        if (t)
          gather<<<(out.numel() + 255) / 256, 256, 0, stream>>>(
              v.data_ptr<scalar_t>(),
              reinterpret_cast<const U*>(route.data_ptr<int64_t>()),
              out.data_ptr<scalar_t>(), out.numel(), t, h, hv, dv);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, pq, pk, route};
}

torch::Tensor pack_values(const torch::Tensor& v) {
  check(v);
  c10::cuda::CUDAGuard guard(v.device());
  int b = v.size(0), t = v.size(1), hv = v.size(2), d = v.size(3);
  auto out = torch::empty({b * hv, t, (d + 31) / 32}, v.options().dtype(torch::kInt32));
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_ROSA_FLOAT_TYPES(v.scalar_type(), "packed_values",
      [&] {
        if (out.numel())
          pack_v<<<(out.numel() + 255) / 256, 256, 0, stream>>>(
              v.data_ptr<scalar_t>(), out.data_ptr<int>(), out.numel(), t, hv, d);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Grads backward(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& dy,
               const Tensor& pq, const Tensor& pk, const Tensor& route,
               int d, int rows, int mask) {
  check(v);
  check(dy);
  c10::cuda::CUDAGuard guard(v.device());
  const bool need_q = mask & 1, need_k = mask & 2, need_v = mask & 4;
  const int b = dy.size(0), t = dy.size(1), h = dy.size(2), hv = v.size(2), dv = v.size(3);
  TORCH_CHECK(b > 0 && h > 0 && hv > 0 && dv > 0 && h % hv == 0, "invalid dimensions");
  TORCH_CHECK(dy.device() == v.device() && dy.scalar_type() == v.scalar_type() &&
                  dy.size(0) == v.size(0) && dy.size(1) == v.size(1) &&
                  dy.size(3) == v.size(3), "invalid dY");
  TORCH_CHECK((v.scalar_type() == torch::kFloat16 || v.scalar_type() == torch::kBFloat16 ||
               v.scalar_type() == torch::kFloat32) &&
                  t < (1 << 28) && v.numel() < (1LL << 30) && dy.numel() < (1LL << 30) &&
                  int64_t(b) * h < (1LL << 30) && int64_t(h) * dv < (1LL << 30),
              "invalid value type or index limit");
  TORCH_CHECK(route.device() == v.device() && route.scalar_type() == torch::kInt64 &&
                  route.is_contiguous() && route.sizes() == torch::IntArrayRef({int64_t(b) * h, t}),
              "invalid route");
  if (need_q || need_k) {
    TORCH_CHECK(pq.device() == v.device() && pk.device() == v.device() &&
                    pq.scalar_type() == torch::kInt32 && pk.scalar_type() == torch::kInt32 &&
                    pq.is_contiguous() && pk.is_contiguous() &&
                    pq.sizes() == route.sizes() && pk.sizes() == route.sizes() &&
                    pq.numel() < (1LL << 30), "invalid Q/K codes");
    auto input = [&](const Tensor& x) {
      check(x);
      TORCH_CHECK(x.device() == v.device() && x.scalar_type() == v.scalar_type() &&
                      x.sizes() == torch::IntArrayRef({b, t, h, d}) && x.numel() < (1LL << 30),
                  "invalid saved input");
    };
    if (need_q) input(q);
    if (need_k) input(k);
  }
  auto raw = mask & 3 ? credit(pq, pk, v, dy, route, d, rows) : Credit{};
  if (t > 1 && need_v)
    at::globalContext().alertNotDeterministic("rosa_bitflip V backward");
  auto oq = need_q ? torch::empty({b, t, h, d}, v.options()) : torch::empty({0}, v.options());
  auto ok = need_k ? torch::empty({b, t, h, d}, v.options()) : torch::empty({0}, v.options());
  auto ov = need_v ? torch::empty_like(v) : torch::empty({0}, v.options());
  auto rawv = need_v ? torch::zeros_like(v, v.options().dtype(torch::kFloat32))
                     : torch::empty({0}, v.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_ROSA_FLOAT_TYPES(v.scalar_type(), "bitflip_finish", [&] {
    if (!t) return;
    const int64_t size = int64_t(b) * t * h * d;
    if (need_q || need_k)
      qk_grad<<<(size + 255) / 256, 256, 0, stream>>>(
          need_q ? q.data_ptr<scalar_t>() : nullptr, need_k ? k.data_ptr<scalar_t>() : nullptr,
          raw.q.data_ptr<float>(), raw.k.data_ptr<float>(), raw.common.data_ptr<float>(),
          oq.data_ptr<scalar_t>(), ok.data_ptr<scalar_t>(), size, t, h, d, need_q, need_k);
    if (need_v) {
      scatter<<<(dy.numel() + 255) / 256, 256, 0, stream>>>(
          dy.data_ptr<scalar_t>(), reinterpret_cast<const U*>(route.data_ptr<int64_t>()),
          rawv.data_ptr<float>(), dy.numel(), t, h, hv, dv);
      v_grad<<<(v.numel() + 255) / 256, 256, 0, stream>>>(
          v.data_ptr<scalar_t>(), rawv.data_ptr<float>(), ov.data_ptr<scalar_t>(), v.numel());
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {oq, ok, ov};
}

torch::Tensor hard(const torch::Tensor& q, const torch::Tensor& k) {
  TORCH_CHECK(q.is_cuda() && q.device() == k.device() && q.dim() == 2 &&
                  q.sizes() == k.sizes() && q.scalar_type() == torch::kInt32 &&
                  k.scalar_type() == torch::kInt32 && q.is_contiguous() &&
                  k.is_contiguous() && q.size(0) > 0 && q.numel() < (1LL << 30) &&
                  q.size(1) < (1 << 28),
              "invalid CUDA codes");
  c10::cuda::CUDAGuard guard(q.device());
  int s = q.size(0), t = q.size(1);
  auto out = torch::zeros(q.sizes(), q.options().dtype(torch::kInt64));
  if (t > 1)
    rosa::cuda::match<<<(s * ((t - 1 + 3) / 4) + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<int>(), k.data_ptr<int>(),
        reinterpret_cast<U*>(out.data_ptr<int64_t>()), s, t);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace rosa::bitflip
