#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

namespace {

constexpr int W = 32;
constexpr int N = 128;
constexpr int NW = N / W;

#define ROSA_DISPATCH(TYPE, NAME, ...)                         \
  AT_DISPATCH_SWITCH(                                          \
      TYPE,                                                    \
      NAME,                                                    \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)     \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)      \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

template <typename T>
__device__ __forceinline__ float load(const T* x, int64_t i) {
  return static_cast<float>(x[i]);
}

template <typename T>
__global__ void pack(
    const T* __restrict__ q,
    const T* __restrict__ k,
    int32_t* __restrict__ pq,
    int32_t* __restrict__ pk,
    int64_t n,
    int t,
    int h,
    int d) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (i >= n) return;

  const int64_t bh = static_cast<int64_t>(t) * h;
  const int64_t b = i / bh;
  const int64_t x = i - b * bh;
  const int j = static_cast<int>(x / h);
  const int a = static_cast<int>(x - static_cast<int64_t>(j) * h);
  const int64_t o = (b * h + a) * t + j;
  const int64_t p = i * d;
  uint32_t qw = 0;
  uint32_t kw = 0;
  for (int z = 0; z < d; ++z) {
    qw |= static_cast<uint32_t>(load(q, p + z) > 0.0f) << z;
    kw |= static_cast<uint32_t>(load(k, p + z) > 0.0f) << z;
  }
  pq[o] = static_cast<int32_t>(qw);
  pk[o] = static_cast<int32_t>(kw);
}

__device__ __forceinline__ int segment(
    const int32_t* __restrict__ cu,
    int b,
    int x) {
  int l = 0;
  int r = b;
  while (l < r) {
    const int m = l + (r - l) / 2;
    if (cu[m + 1] <= x) l = m + 1;
    else r = m;
  }
  return l;
}

__global__ void check_cu(
    const int32_t* __restrict__ cu,
    int b,
    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) CUDA_KERNEL_ASSERT(cu[0] == 0);
  if (i < b) CUDA_KERNEL_ASSERT(cu[i] <= cu[i + 1]);
  if (i == b) CUDA_KERNEL_ASSERT(cu[b] == n);
}

template <bool Packed>
__global__ void route(
    const int32_t* __restrict__ q,
    const int32_t* __restrict__ k,
    int64_t* __restrict__ out,
    int b,
    int t,
    int h,
    const int32_t* __restrict__ cu,
    int ns,
    int nt) {
  const int w = threadIdx.x / W;
  const int l = threadIdx.x & (W - 1);
  const int64_t id = static_cast<int64_t>(blockIdx.x) * NW + w;

  int delta;
  int n;
  const int32_t* qs;
  const int32_t* ks;
  int64_t* y;
  if constexpr (Packed) {
    if (id >= static_cast<int64_t>(h) * nt) return;
    const int a = static_cast<int>(id / nt);
    const int x = static_cast<int>(id - static_cast<int64_t>(a) * nt);
    int s = 0;
    int p = 0;
    int e = 0;
    if (l == 0) {
      s = segment(cu, ns, x);
      if (s < ns) {
        p = cu[s];
        e = cu[s + 1];
      }
    }
    s = __shfl_sync(0xffffffffu, s, 0);
    p = __shfl_sync(0xffffffffu, p, 0);
    e = __shfl_sync(0xffffffffu, e, 0);
    if (s >= ns || e <= p) return;
    delta = x - p;
    if (delta <= 0) return;
    n = e - x;
    qs = q + static_cast<int64_t>(a) * nt + p;
    ks = k + static_cast<int64_t>(a) * nt + p;
    y = out + static_cast<int64_t>(a) * nt + p;
  } else {
    const int nd = t - 1;
    if (id >= static_cast<int64_t>(b) * h * nd) return;
    const int s = static_cast<int>(id / nd);
    delta = static_cast<int>(id - static_cast<int64_t>(s) * nd) + 1;
    n = t - delta;
    qs = q + static_cast<int64_t>(s) * t;
    ks = k + static_cast<int64_t>(s) * t;
    y = out + static_cast<int64_t>(s) * t;
  }

  int last = -1;
  for (int x = 0; x < n; x += W) {
    const int j = x + l;
    const bool active = j < n;
    const bool same = active && qs[j + delta] == ks[j];
    int m = active && !same ? j : -1;
#pragma unroll
    for (int s = 1; s < W; s <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, m, s);
      if (l >= s) m = max(m, v);
    }
    if (same) {
      const int len = j - max(last, m);
      const uint64_t p = (static_cast<uint64_t>(len) << 32) |
          static_cast<uint32_t>(j + 1);
      atomicMax(
          reinterpret_cast<unsigned long long*>(y + j + delta),
          static_cast<unsigned long long>(p));
    }
    last = max(last, __shfl_sync(0xffffffffu, m, W - 1));
  }
}

template <typename T, bool Packed>
__global__ void gather(
    const int64_t* __restrict__ route,
    const T* __restrict__ v,
    T* __restrict__ y,
    int b,
    int t,
    int h,
    int hv,
    int d,
    const int32_t* __restrict__ cu,
    int ns,
    int nt) {
  __shared__ int start;
  const int x = blockIdx.x;
  int i;
  int a;
  int z;
  int base = 0;
  int64_t ri;
  if constexpr (Packed) {
    i = x / h;
    a = x - i * h;
    if (threadIdx.x == 0) {
      const int s = segment(cu, ns, i);
      start = s < ns ? cu[s] : -1;
    }
    __syncthreads();
    z = start;
    if (z < 0) return;
    base = z;
    ri = static_cast<int64_t>(a) * nt + i;
  } else {
    a = x % h;
    i = (x / h) % t;
    z = x / (t * h);
    ri = (static_cast<int64_t>(z) * h + a) * t + i;
  }
  const int r = static_cast<int>(static_cast<uint64_t>(route[ri]));
  const int av = a / (h / hv);
  for (int j = threadIdx.x; j < d; j += blockDim.x) {
    float u = 0.0f;
    if (r > 0) {
      const int64_t vi = Packed
          ? (static_cast<int64_t>(base + r) * hv + av) * d + j
          : ((static_cast<int64_t>(z) * t + r) * hv + av) * d + j;
      u = load(v, vi) > 0.0f ? 1.0f : -1.0f;
    }
    y[static_cast<int64_t>(x) * d + j] = static_cast<T>(u);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rosa_hard_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu) {
  const c10::cuda::CUDAGuard guard(q.device());
  const bool packed = q.dim() == 3;
  const int b = packed ? 1 : static_cast<int>(q.size(0));
  const int t = packed ? static_cast<int>(q.size(0))
                       : static_cast<int>(q.size(1));
  const int h = packed ? static_cast<int>(q.size(1))
                       : static_cast<int>(q.size(2));
  const int hv = packed ? static_cast<int>(v.size(1))
                        : static_cast<int>(v.size(2));
  const int dv = packed ? static_cast<int>(v.size(2))
                        : static_cast<int>(v.size(3));
  const int ns = packed ? static_cast<int>(cu.numel() - 1) : 0;
  const int64_t rows = static_cast<int64_t>(b) * t * h;
  auto i32 = q.options().dtype(torch::kInt32);
  auto pq = packed ? torch::empty({h, t}, i32)
                   : torch::empty({b, h, t}, i32);
  auto pk = torch::empty_like(pq);
  auto y = packed ? torch::empty({t, h, dv}, q.options())
                  : torch::empty({b, t, h, dv}, q.options());
  auto best = packed
      ? torch::zeros({h, t}, q.options().dtype(torch::kInt64))
      : torch::zeros({b, h, t}, q.options().dtype(torch::kInt64));
  const auto stream = at::cuda::getCurrentCUDAStream();

  ROSA_DISPATCH(q.scalar_type(), "rosa_soft_forward", [&] {
    const int d = packed ? static_cast<int>(q.size(2))
                         : static_cast<int>(q.size(3));
    pack<scalar_t><<<(rows + 255) / 256, 256, 0, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
        pq.data_ptr<int32_t>(), pk.data_ptr<int32_t>(), rows, t, h, d);
    if (packed) {
      check_cu<<<(ns + 256) / 256, 256, 0, stream>>>(
          cu.data_ptr<int32_t>(), ns, t);
    }
    const int64_t tasks = packed
        ? static_cast<int64_t>(h) * t
        : static_cast<int64_t>(b) * h * (t - 1);
    if (tasks > 0) {
      if (packed) {
        route<true><<<(tasks + NW - 1) / NW, N, 0, stream>>>(
            pq.data_ptr<int32_t>(), pk.data_ptr<int32_t>(),
            best.data_ptr<int64_t>(), b, t, h, cu.data_ptr<int32_t>(), ns, t);
      } else {
        route<false><<<(tasks + NW - 1) / NW, N, 0, stream>>>(
            pq.data_ptr<int32_t>(), pk.data_ptr<int32_t>(),
            best.data_ptr<int64_t>(), b, t, h, nullptr, 0, t);
      }
    }
    if (packed) {
      gather<scalar_t, true><<<rows, N, 0, stream>>>(
          best.data_ptr<int64_t>(), v.data_ptr<scalar_t>(),
          y.data_ptr<scalar_t>(), b, t, h, hv, dv,
          cu.data_ptr<int32_t>(), ns, t);
    } else {
      gather<scalar_t, false><<<rows, N, 0, stream>>>(
          best.data_ptr<int64_t>(), v.data_ptr<scalar_t>(),
          y.data_ptr<scalar_t>(), b, t, h, hv, dv, nullptr, 0, t);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, pq, pk};
}
