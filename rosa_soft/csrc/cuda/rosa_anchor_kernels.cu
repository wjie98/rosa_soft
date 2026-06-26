#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

constexpr int kBlockThreads = 128;
constexpr float kRouteWeightMin = 0.25f;
constexpr int kWarpSize = 32;
constexpr int kBackwardScanMaxOffsets = 2;

__host__ __device__ inline int k_tile_capacity_for(int suffix_window) {
  int capacity = 1;
  const int needed = kBlockThreads + suffix_window;
  while (capacity < needed) {
    capacity <<= 1;
  }
  return capacity;
}

enum TelemetryStat : int {
  kTelemetryTopProb = 0,
  kTelemetryProbGap = 1,
  kTelemetryCandidateTopProb = 2,
  kTelemetryCandidateProbGap = 3,
  kTelemetryCandidateRows = 4,
  kTelemetryCandidateMass = 5,
  kTelemetryNullProb = 6,
  kTelemetryEntropyNorm = 7,
  kTelemetryCandidateEntropyNorm = 8,
  kTelemetryRows = 9,
  kTelemetryEffectiveOffsets = 10,
  kTelemetryFullOffsets = 11,
  kTelemetryTruncatedCandidates = 12,
  kTelemetryCandidates = 13,
  kTelemetryCount = 14,
};

struct RosaAnchorScore {
  float score;
  int effective_offsets;
  int full_offsets;
  int truncated;
};

#define DISPATCH_ROSA_FLOAT_TYPES(TYPE, NAME, ...)              \
  AT_DISPATCH_SWITCH(                                           \
      TYPE,                                                     \
      NAME,                                                     \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

template <typename scalar_t>
__device__ __forceinline__ float read_float(const scalar_t* __restrict__ ptr, int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t write_scalar(float value) {
  return static_cast<scalar_t>(value);
}

__device__ __forceinline__ uint32_t dim_mask(int dim) {
  return dim == 32 ? 0xffffffffu : ((1u << dim) - 1u);
}

__device__ __forceinline__ int sign_from_bit(uint32_t bits, int bit) {
  return ((bits >> bit) & 1u) ? 1 : -1;
}

__device__ __forceinline__ float route_weight_from_conf(float conf) {
  return kRouteWeightMin + (1.0f - kRouteWeightMin) * conf;
}

template <typename scalar_t>
__device__ __forceinline__ float quant_derivative(
    const scalar_t* __restrict__ x,
    int64_t idx) {
  const float z = read_float(x, idx);
  const float denom = 1.0f + fabsf(z);
  return 1.0f / (denom * denom);
}

template <typename scalar_t>
__global__ void qk_directional_damper_kernel(
    const scalar_t* __restrict__ values,
    float* __restrict__ grads,
    int64_t numel,
    float strength) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  const float s = fminf(fmaxf(strength, 0.0f), 1.0f);
  if (s <= 0.0f) {
    return;
  }
  const float value = read_float(values, idx);
  const float grad = grads[idx];
  if (value * grad >= 0.0f) {
    return;
  }
  // Fixed ramp from |x|=2 to |x|=4. It is intentionally a single-parameter
  // guardrail: qk_damper_strength controls the maximum damping.
  float t = (fabsf(value) - 2.0f) * 0.5f;
  t = fminf(fmaxf(t, 0.0f), 1.0f);
  const float saturation = t * t * (3.0f - 2.0f * t);
  grads[idx] = grad * (1.0f - s * saturation);
}

template <typename scalar_t>
__global__ void pack_sign_bits_kernel(
    const scalar_t* __restrict__ x,
    int32_t* __restrict__ bits,
    int64_t total_tokens,
    int dim) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_tokens) {
    return;
  }

  const int64_t base = idx * dim;
  uint32_t word = 0;
  for (int d = 0; d < dim; ++d) {
    word |= (read_float(x, base + d) > 0.0f ? 1u : 0u) << d;
  }
  bits[idx] = static_cast<int32_t>(word);
}

__device__ __forceinline__ uint32_t load_q_word(
    const int32_t* __restrict__ q_bits,
    int b,
    int h,
    int t,
    int seq_len,
    int num_heads) {
  const int64_t idx = (static_cast<int64_t>(b) * seq_len + t) * num_heads + h;
  return static_cast<uint32_t>(q_bits[idx]);
}

__device__ __forceinline__ uint32_t load_k_word(
    const int32_t* __restrict__ k_bits,
    int b,
    int h,
    int t,
    int seq_len,
    int num_heads) {
  const int64_t idx = (static_cast<int64_t>(b) * seq_len + t) * num_heads + h;
  return static_cast<uint32_t>(k_bits[idx]);
}

__device__ __forceinline__ int64_t token_index(
    int b,
    int h,
    int t,
    int seq_len,
    int num_heads) {
  return (static_cast<int64_t>(b) * seq_len + t) * num_heads + h;
}

__device__ __forceinline__ int64_t qk_base_index(
    int b,
    int h,
    int t,
    int seq_len,
    int num_heads,
    int dim) {
  return token_index(b, h, t, seq_len, num_heads) * dim;
}

__device__ __forceinline__ float route_confidence(float value) {
  return tanhf(fabsf(value));
}

__device__ __forceinline__ float route_confidence_derivative(float value) {
  const float sign = value >= 0.0f ? 1.0f : -1.0f;
  const float conf = route_confidence(value);
  return sign * (1.0f - conf * conf);
}

__device__ __forceinline__ float exact_pair_route_weight(float q_value, float k_value) {
  return route_weight_from_conf(fminf(route_confidence(q_value), route_confidence(k_value)));
}

template <typename scalar_t>
__device__ float rosa_anchor_match_weighted(
    uint32_t q_word,
    uint32_t k_word,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    int b,
    int h,
    int q_pos,
    int k_pos,
    int seq_len,
    int num_heads,
    int dim,
    float match_lambda) {
  const uint32_t mismatch_mask = (q_word ^ k_word) & dim_mask(dim);
  if (mismatch_mask == 0u) {
    return 1.0f;
  }
  float hamming = 0.0f;
  uint32_t mask = mismatch_mask;
  const int64_t q_base = qk_base_index(b, h, q_pos, seq_len, num_heads, dim);
  const int64_t k_base = qk_base_index(b, h, k_pos, seq_len, num_heads, dim);
  while (mask != 0u) {
    const int bit = __ffs(mask) - 1;
    hamming += exact_pair_route_weight(
        read_float(query, q_base + bit),
        read_float(key, k_base + bit));
    mask &= mask - 1u;
  }
  return __expf(-match_lambda * hamming);
}

template <typename scalar_t, bool kUseKCircular = false>
__device__ RosaAnchorScore rosa_anchor_score_pair_shared(
    const uint32_t* __restrict__ q_suffix,
    const uint32_t* __restrict__ k_tile,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    int b,
    int h,
    int tile_base,
    int k_tile_capacity,
    int i,
    int j,
    int seq_len,
    int num_heads,
    int dim,
    int suffix_window,
    float match_lambda,
    float scale_abs,
    float logit_epsilon) {
  if (j == 0) {
    return {0.0f, 0, 0, 0};
  }

  const int valid = min(suffix_window, min(i + 1, j));
  float prod = 1.0f;
  float score = 0.0f;
  int effective = 0;
  int truncated = 0;
  const bool use_early_stop = logit_epsilon > 0.0f;

  for (int offset = 0; offset < valid; ++offset) {
    const int k_pos = j - 1 - offset;
    const int q_pos = i - offset;
    const uint32_t q_word = q_suffix[offset];
    const uint32_t k_word = kUseKCircular
        ? k_tile[k_pos & (k_tile_capacity - 1)]
        : k_tile[k_pos - tile_base];
    const float match = rosa_anchor_match_weighted<scalar_t>(
        q_word,
        k_word,
        query,
        key,
        b,
        h,
        q_pos,
        k_pos,
        seq_len,
        num_heads,
        dim,
        match_lambda);
    prod *= match;
    score += prod;
    ++effective;

    const int remaining = valid - offset - 1;
    if (use_early_stop && remaining > 0) {
      const float tail_logit_bound = scale_abs * static_cast<float>(remaining) * prod;
      if (tail_logit_bound <= logit_epsilon) {
        truncated = 1;
        break;
      }
    }
  }

  return {score, effective, valid, truncated};
}

template <typename scalar_t, bool kUseKCircular = false>
__device__ RosaAnchorScore rosa_anchor_score_pair_scan_shared(
    const uint32_t* __restrict__ q_suffix,
    const uint32_t* __restrict__ k_tile,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float* __restrict__ prefix_products,
    bool* __restrict__ scan_complete,
    int b,
    int h,
    int tile_base,
    int k_tile_capacity,
    int i,
    int j,
    int seq_len,
    int num_heads,
    int dim,
    int suffix_window,
    float match_lambda,
    float scale_abs,
    float logit_epsilon) {
  if (j == 0) {
    *scan_complete = true;
    return {0.0f, 0, 0, 0};
  }

  const int valid = min(suffix_window, min(i + 1, j));
  float prod = 1.0f;
  float score = 0.0f;
  int effective = 0;
  int truncated = 0;
  bool complete = true;
  const bool use_early_stop = logit_epsilon > 0.0f;

  for (int offset = 0; offset < valid; ++offset) {
    const int k_pos = j - 1 - offset;
    const int q_pos = i - offset;
    const uint32_t q_word = q_suffix[offset];
    const uint32_t k_word = kUseKCircular
        ? k_tile[k_pos & (k_tile_capacity - 1)]
        : k_tile[k_pos - tile_base];
    const float match = rosa_anchor_match_weighted<scalar_t>(
        q_word,
        k_word,
        query,
        key,
        b,
        h,
        q_pos,
        k_pos,
        seq_len,
        num_heads,
        dim,
        match_lambda);
    prod *= match;
    score += prod;
    if (offset < kBackwardScanMaxOffsets) {
      prefix_products[offset] = prod;
    } else {
      complete = false;
    }
    ++effective;

    const int remaining = valid - offset - 1;
    if (use_early_stop && remaining > 0) {
      const float tail_logit_bound = scale_abs * static_cast<float>(remaining) * prod;
      if (tail_logit_bound <= logit_epsilon) {
        truncated = 1;
        break;
      }
    }
  }

  *scan_complete = complete;
  return {score, effective, valid, truncated};
}

__device__ __forceinline__ float rosa_anchor_tie_break(
    int i,
    int j,
    float tie_strength) {
  const float denom = fmaxf(1.0f, static_cast<float>(i));
  return tie_strength * static_cast<float>(j) / denom;
}

__device__ __forceinline__ float rosa_anchor_candidate_logit(
    float score,
    int i,
    int j,
    float scale,
    float sink_threshold,
    float tie_strength) {
  return scale * (score - sink_threshold + rosa_anchor_tie_break(i, j, tie_strength));
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  unsigned mask = 0xffffffffu;
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  unsigned mask = 0xffffffffu;
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(mask, value, offset));
  }
  return value;
}

__device__ __forceinline__ void add_grad_query_value(
    float* __restrict__ grad_query,
    int64_t grad_idx,
    float value) {
  atomicAdd(&grad_query[grad_idx], value);
}

template <typename scalar_t>
__device__ __forceinline__ void accumulate_pair_grad(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    int b,
    int h,
    int q_pos,
    int k_pos,
    int seq_len,
    int num_heads,
    int dim,
    uint32_t q_word,
    uint32_t k_word,
    float common_base) {
  const int64_t q_base =
      (((static_cast<int64_t>(b) * seq_len + q_pos) * num_heads + h) * dim);
  const int64_t k_base =
      (((static_cast<int64_t>(b) * seq_len + k_pos) * num_heads + h) * dim);

  for (int d = 0; d < dim; ++d) {
    const float q_sign = static_cast<float>(sign_from_bit(q_word, d));
    const float k_sign = static_cast<float>(sign_from_bit(k_word, d));
    const float bit_weight = exact_pair_route_weight(
        read_float(query, q_base + d),
        read_float(key, k_base + d));
    const float common = 0.5f * common_base * bit_weight;
    const float dq_proxy = common * k_sign;
    const float dk_proxy = common * q_sign;
    const float dq = dq_proxy * quant_derivative(query, q_base + d);
    const float dk = dk_proxy * quant_derivative(key, k_base + d);
    add_grad_query_value(
        grad_query,
        q_base + d,
        dq);
    atomicAdd(&grad_key[k_base + d], dk);

    const float mismatch = 0.5f * (1.0f - q_sign * k_sign);
    if (mismatch > 0.0f) {
      const float q_value = read_float(query, q_base + d);
      const float k_value = read_float(key, k_base + d);
      const float q_conf = route_confidence(q_value);
      const float k_conf = route_confidence(k_value);
      if (q_conf <= k_conf) {
        add_grad_query_value(
            grad_query,
            q_base + d,
            -common_base * mismatch * (1.0f - kRouteWeightMin) *
                route_confidence_derivative(q_value));
      }
      if (k_conf <= q_conf) {
        atomicAdd(
            &grad_key[k_base + d],
            -common_base * mismatch * (1.0f - kRouteWeightMin) *
                route_confidence_derivative(k_value));
      }
    }
  }
}

__device__ float block_reduce_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  value = tid < num_warps ? scratch[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__device__ float block_reduce_max(float value, float* scratch) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_max(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  value = tid < num_warps ? scratch[lane] : -FLT_MAX;
  if (warp == 0) {
    value = warp_reduce_max(value);
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__device__ void load_q_suffix_shared(
    const int32_t* __restrict__ q_bits,
    uint32_t* __restrict__ q_suffix,
    int b,
    int h,
    int i,
    int seq_len,
    int num_heads,
    int suffix_window) {
  for (int offset = threadIdx.x; offset < suffix_window; offset += blockDim.x) {
    const int q_pos = i - offset;
    q_suffix[offset] = q_pos >= 0 ? load_q_word(q_bits, b, h, q_pos, seq_len, num_heads) : 0u;
  }
  __syncthreads();
}

__device__ void load_k_range_circular_shared(
    const int32_t* __restrict__ k_bits,
    uint32_t* __restrict__ k_tile,
    int b,
    int h,
    int seq_len,
    int num_heads,
    int k_tile_capacity,
    int load_start,
    int load_end) {
  for (int k_pos = load_start + threadIdx.x; k_pos <= load_end; k_pos += blockDim.x) {
    k_tile[k_pos & (k_tile_capacity - 1)] = (k_pos >= 0 && k_pos < seq_len)
        ? load_k_word(k_bits, b, h, k_pos, seq_len, num_heads)
        : 0u;
  }
  __syncthreads();
}

__device__ int load_k_tile_linear_shared(
    const int32_t* __restrict__ k_bits,
    uint32_t* __restrict__ k_tile,
    int b,
    int h,
    int seq_len,
    int num_heads,
    int suffix_window,
    int tile_start,
    int tile_len) {
  const int tile_base = tile_start - suffix_window;
  const int tile_count = tile_len + suffix_window;

  for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
    const int k_pos = tile_base + t;
    k_tile[t] = (k_pos >= 0 && k_pos < seq_len)
        ? load_k_word(k_bits, b, h, k_pos, seq_len, num_heads)
        : 0u;
  }
  __syncthreads();
  return tile_base;
}

__device__ int load_k_tile_circular_shared(
    const int32_t* __restrict__ k_bits,
    uint32_t* __restrict__ k_tile,
    int b,
    int h,
    int seq_len,
    int num_heads,
    int suffix_window,
    int tile_start,
    int tile_len,
    int k_tile_capacity,
    int loaded_k_end,
    int* tile_base_out) {
  const int tile_base = tile_start - suffix_window;
  const int tile_end = tile_start + tile_len - 1;
  int load_start = loaded_k_end + 1;
  if (load_start < tile_base) {
    load_start = tile_base;
  }
  if (load_start < 0) {
    load_start = 0;
  }
  if (load_start <= tile_end) {
    load_k_range_circular_shared(
        k_bits, k_tile, b, h, seq_len, num_heads, k_tile_capacity, load_start, tile_end);
  } else {
    __syncthreads();
  }
  *tile_base_out = tile_base;
  return tile_end > loaded_k_end ? tile_end : loaded_k_end;
}

template <typename scalar_t, bool kUseKCircular = false>
__global__ void rosa_anchor_forward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const int32_t* __restrict__ q_bits,
    const int32_t* __restrict__ k_bits,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    float* __restrict__ telemetry_stats,
    float* __restrict__ row_max_out,
    float* __restrict__ row_denom_out,
    int batch_size,
    int seq_len,
    int num_heads,
    int dim,
    int num_value_heads,
    int value_dim,
    float scale,
    int suffix_window,
    float match_lambda,
    float sink_threshold,
    float tie_strength,
    float logit_epsilon) {
  extern __shared__ unsigned char shared_raw[];
  float* scratch = reinterpret_cast<float*>(shared_raw);
  uint32_t* q_suffix = reinterpret_cast<uint32_t*>(scratch + blockDim.x);
  const int k_tile_capacity = kUseKCircular
      ? k_tile_capacity_for(suffix_window)
      : kBlockThreads + suffix_window;
  uint32_t* k_tile = q_suffix + suffix_window;
  float* out_sums = reinterpret_cast<float*>(k_tile + k_tile_capacity);

  const int row = blockIdx.x;
  const int i = row % seq_len;
  const int h = (row / seq_len) % num_heads;
  const int b = row / (seq_len * num_heads);
  const int value_head = h / (num_heads / num_value_heads);
  const float scale_abs = fabsf(scale);

  load_q_suffix_shared(q_bits, q_suffix, b, h, i, seq_len, num_heads, suffix_window);

  float local_max = -FLT_MAX;
  float local_effective_offsets = 0.0f;
  float local_full_offsets = 0.0f;
  float local_truncated_candidates = 0.0f;
  float local_candidates = 0.0f;
  int loaded_k_end = -1;
  for (int tile_start = 1; tile_start <= i; tile_start += blockDim.x) {
    const int tile_len = min(blockDim.x, i - tile_start + 1);
    int tile_base = 0;
    if constexpr (kUseKCircular) {
      loaded_k_end = load_k_tile_circular_shared(
          k_bits,
          k_tile,
          b,
          h,
          seq_len,
          num_heads,
          suffix_window,
          tile_start,
          tile_len,
          k_tile_capacity,
          loaded_k_end,
          &tile_base);
    } else {
      tile_base = load_k_tile_linear_shared(
          k_bits, k_tile, b, h, seq_len, num_heads, suffix_window, tile_start, tile_len);
    }

    const int j = tile_start + threadIdx.x;
    if (threadIdx.x < tile_len) {
      const RosaAnchorScore score = rosa_anchor_score_pair_shared<scalar_t, kUseKCircular>(
          q_suffix,
          k_tile,
          query,
          key,
          b,
          h,
          tile_base,
          k_tile_capacity,
          i,
          j,
          seq_len,
          num_heads,
          dim,
          suffix_window,
          match_lambda,
          scale_abs,
          logit_epsilon);
      const float logit =
          rosa_anchor_candidate_logit(score.score, i, j, scale, sink_threshold, tie_strength);
      local_max = fmaxf(local_max, logit);
      local_effective_offsets += static_cast<float>(score.effective_offsets);
      local_full_offsets += static_cast<float>(score.full_offsets);
      local_truncated_candidates += static_cast<float>(score.truncated);
      local_candidates += 1.0f;
    }
    __syncthreads();
  }

  const float candidate_max = block_reduce_max(local_max, scratch);
  const float row_max = fmaxf(candidate_max, 0.0f);

  for (int d = threadIdx.x; d < value_dim; d += blockDim.x) {
    out_sums[d] = 0.0f;
  }
  __syncthreads();

  float local_denom = 0.0f;
  float local_weight_logw = 0.0f;
  float local_top1 = -FLT_MAX;
  float local_top2 = -FLT_MAX;
  loaded_k_end = -1;
  for (int tile_start = 1; tile_start <= i; tile_start += blockDim.x) {
    const int tile_len = min(blockDim.x, i - tile_start + 1);
    int tile_base = 0;
    if constexpr (kUseKCircular) {
      loaded_k_end = load_k_tile_circular_shared(
          k_bits,
          k_tile,
          b,
          h,
          seq_len,
          num_heads,
          suffix_window,
          tile_start,
          tile_len,
          k_tile_capacity,
          loaded_k_end,
          &tile_base);
    } else {
      tile_base = load_k_tile_linear_shared(
          k_bits, k_tile, b, h, seq_len, num_heads, suffix_window, tile_start, tile_len);
    }

    const int j = tile_start + threadIdx.x;
    float weight = 0.0f;
    if (threadIdx.x < tile_len) {
      const RosaAnchorScore score = rosa_anchor_score_pair_shared<scalar_t, kUseKCircular>(
          q_suffix,
          k_tile,
          query,
          key,
          b,
          h,
          tile_base,
          k_tile_capacity,
          i,
          j,
          seq_len,
          num_heads,
          dim,
          suffix_window,
          match_lambda,
          scale_abs,
          logit_epsilon);
      const float logit =
          rosa_anchor_candidate_logit(score.score, i, j, scale, sink_threshold, tie_strength);
      const float logw = logit - row_max;
      weight = __expf(logit - row_max);
      local_denom += weight;
      local_weight_logw += weight * logw;
      if (weight >= local_top1) {
        local_top2 = local_top1;
        local_top1 = weight;
      } else if (weight > local_top2) {
        local_top2 = weight;
      }
    }
    for (int d = 0; d < value_dim; ++d) {
      float contribution = 0.0f;
      if (threadIdx.x < tile_len) {
        const int64_t v_idx =
            (((static_cast<int64_t>(b) * seq_len + j) * num_value_heads + value_head) * value_dim + d);
        const float xv = read_float(value, v_idx) > 0.0f ? 1.0f : -1.0f;
        contribution = weight * xv;
      }
      const float warp_sum = warp_reduce_sum(contribution);
      if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        atomicAdd(&out_sums[d], warp_sum);
      }
    }
    __syncthreads();
  }

  const float sink_weight = __expf(-row_max);
  const float candidate_denom_sum = block_reduce_sum(local_denom, scratch);
  const float denom = sink_weight + candidate_denom_sum;
  if (threadIdx.x == 0) {
    if (row_max_out != nullptr) {
      row_max_out[row] = row_max;
    }
    if (row_denom_out != nullptr) {
      row_denom_out[row] = denom;
    }
  }

  if (telemetry_stats != nullptr) {
    const float candidate_weight_logw = block_reduce_sum(local_weight_logw, scratch);
    const float effective_offsets = block_reduce_sum(local_effective_offsets, scratch);
    const float full_offsets = block_reduce_sum(local_full_offsets, scratch);
    const float truncated_candidates = block_reduce_sum(local_truncated_candidates, scratch);
    const float candidates = block_reduce_sum(local_candidates, scratch);
    const float candidate_top = block_reduce_max(local_top1, scratch);
    const float local_top_count =
        (candidate_top > 0.0f && local_top1 == candidate_top) ? 1.0f : 0.0f;
    const float top_count = block_reduce_sum(local_top_count, scratch);
    const float second_source =
        (candidate_top > 0.0f && local_top1 == candidate_top && top_count < 1.5f) ? local_top2 : local_top1;
    const float candidate_second_raw = block_reduce_max(second_source, scratch);
    const float candidate_second =
        (top_count >= 1.5f) ? candidate_top : fmaxf(candidate_second_raw, 0.0f);
    const float safe_denom = fmaxf(denom, 1.0e-20f);
    const float candidate_denom = fmaxf(candidate_denom_sum, 1.0e-20f);

    float top_weight = sink_weight;
    float second_weight = fmaxf(candidate_top, candidate_second);
    if (candidate_top >= sink_weight) {
      top_weight = candidate_top;
      second_weight = fmaxf(candidate_second, sink_weight);
    }

    const float top_prob = top_weight / safe_denom;
    const float prob_gap = fmaxf(top_weight - second_weight, 0.0f) / safe_denom;
    const float candidate_mass = candidate_denom_sum / safe_denom;
    const float null_prob = sink_weight / safe_denom;
    const float total_weight_logw = candidate_weight_logw + sink_weight * (-row_max);
    const float entropy = logf(safe_denom) - total_weight_logw / safe_denom;
    const float entropy_norm = entropy / logf(fmaxf(static_cast<float>(i + 1), 2.0f));

    if (threadIdx.x == 0) {
      atomicAdd(&telemetry_stats[kTelemetryTopProb], top_prob);
      atomicAdd(&telemetry_stats[kTelemetryProbGap], prob_gap);
      atomicAdd(&telemetry_stats[kTelemetryCandidateMass], candidate_mass);
      atomicAdd(&telemetry_stats[kTelemetryNullProb], null_prob);
      atomicAdd(&telemetry_stats[kTelemetryEntropyNorm], entropy_norm);
      atomicAdd(&telemetry_stats[kTelemetryRows], 1.0f);
      atomicAdd(&telemetry_stats[kTelemetryEffectiveOffsets], effective_offsets);
      atomicAdd(&telemetry_stats[kTelemetryFullOffsets], full_offsets);
      atomicAdd(&telemetry_stats[kTelemetryTruncatedCandidates], truncated_candidates);
      atomicAdd(&telemetry_stats[kTelemetryCandidates], candidates);
      if (i >= 2 && candidate_denom_sum > 0.0f) {
        const float candidate_top_prob = candidate_top / candidate_denom;
        const float candidate_prob_gap = fmaxf(candidate_top - candidate_second, 0.0f) / candidate_denom;
        const float candidate_entropy =
            logf(candidate_denom) - candidate_weight_logw / candidate_denom;
        const float candidate_entropy_norm = candidate_entropy / logf(static_cast<float>(i));
        atomicAdd(&telemetry_stats[kTelemetryCandidateTopProb], candidate_top_prob);
        atomicAdd(&telemetry_stats[kTelemetryCandidateProbGap], candidate_prob_gap);
        atomicAdd(&telemetry_stats[kTelemetryCandidateRows], 1.0f);
        atomicAdd(&telemetry_stats[kTelemetryCandidateEntropyNorm], candidate_entropy_norm);
      }
    }
  }

  __syncthreads();
  for (int d = threadIdx.x; d < value_dim; d += blockDim.x) {
    const int64_t out_idx =
        (((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) * value_dim + d);
    output[out_idx] = write_scalar<scalar_t>(out_sums[d] / denom);
  }
}

template <typename scalar_t, bool kUseKCircular = false>
__global__ void rosa_anchor_backward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ q_bits,
    const int32_t* __restrict__ k_bits,
    const float* __restrict__ saved_row_max,
    const float* __restrict__ saved_row_denom,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int batch_size,
    int seq_len,
    int num_heads,
    int dim,
    int num_value_heads,
    int value_dim,
    float scale,
    int suffix_window,
    float match_lambda,
    float sink_threshold,
    float tie_strength,
    float logit_epsilon) {
  extern __shared__ unsigned char shared_raw[];
  float* scratch = reinterpret_cast<float*>(shared_raw);
  uint32_t* q_suffix = reinterpret_cast<uint32_t*>(scratch + blockDim.x);
  const int k_tile_capacity = kUseKCircular
      ? k_tile_capacity_for(suffix_window)
      : kBlockThreads + suffix_window;
  uint32_t* k_tile = q_suffix + suffix_window;
  const bool use_grad_output_shared = logit_epsilon > 0.0f;
  float* grad_output_row = use_grad_output_shared
      ? reinterpret_cast<float*>(k_tile + k_tile_capacity)
      : nullptr;

  const int row = blockIdx.x;
  const int i = row % seq_len;
  const int h = (row / seq_len) % num_heads;
  const int b = row / (seq_len * num_heads);
  const int value_head = h / (num_heads / num_value_heads);
  const float scale_abs = fabsf(scale);

  load_q_suffix_shared(q_bits, q_suffix, b, h, i, seq_len, num_heads, suffix_window);
  if (use_grad_output_shared) {
    for (int d = threadIdx.x; d < value_dim; d += blockDim.x) {
      const int64_t go_idx =
          (((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) * value_dim + d);
      grad_output_row[d] = read_float(grad_output, go_idx);
    }
  }
  __syncthreads();

  const bool has_saved_row_stats = saved_row_max != nullptr && saved_row_denom != nullptr;
  float row_max = has_saved_row_stats ? saved_row_max[row] : -FLT_MAX;
  int loaded_k_end = -1;
  if (!has_saved_row_stats) {
    float local_max = -FLT_MAX;
    for (int tile_start = 1; tile_start <= i; tile_start += blockDim.x) {
      const int tile_len = min(blockDim.x, i - tile_start + 1);
      int tile_base = 0;
      if constexpr (kUseKCircular) {
        loaded_k_end = load_k_tile_circular_shared(
            k_bits,
            k_tile,
            b,
            h,
            seq_len,
            num_heads,
            suffix_window,
            tile_start,
            tile_len,
            k_tile_capacity,
            loaded_k_end,
            &tile_base);
      } else {
        tile_base = load_k_tile_linear_shared(
            k_bits, k_tile, b, h, seq_len, num_heads, suffix_window, tile_start, tile_len);
      }

      const int j = tile_start + threadIdx.x;
      if (threadIdx.x < tile_len) {
        const RosaAnchorScore score = rosa_anchor_score_pair_shared<scalar_t, kUseKCircular>(
            q_suffix,
            k_tile,
            query,
            key,
            b,
            h,
            tile_base,
            k_tile_capacity,
            i,
            j,
            seq_len,
            num_heads,
            dim,
            suffix_window,
            match_lambda,
            scale_abs,
            logit_epsilon);
        const float logit =
            rosa_anchor_candidate_logit(score.score, i, j, scale, sink_threshold, tie_strength);
        local_max = fmaxf(local_max, logit);
      }
      __syncthreads();
    }

    const float candidate_max = block_reduce_max(local_max, scratch);
    row_max = fmaxf(candidate_max, 0.0f);
  }

  float local_denom = 0.0f;
  float local_delta_num = 0.0f;
  loaded_k_end = -1;
  for (int tile_start = 1; tile_start <= i; tile_start += blockDim.x) {
    const int tile_len = min(blockDim.x, i - tile_start + 1);
    int tile_base = 0;
    if constexpr (kUseKCircular) {
      loaded_k_end = load_k_tile_circular_shared(
          k_bits,
          k_tile,
          b,
          h,
          seq_len,
          num_heads,
          suffix_window,
          tile_start,
          tile_len,
          k_tile_capacity,
          loaded_k_end,
          &tile_base);
    } else {
      tile_base = load_k_tile_linear_shared(
          k_bits, k_tile, b, h, seq_len, num_heads, suffix_window, tile_start, tile_len);
    }

    const int j = tile_start + threadIdx.x;
    if (threadIdx.x < tile_len) {
      const RosaAnchorScore score = rosa_anchor_score_pair_shared<scalar_t, kUseKCircular>(
          q_suffix,
          k_tile,
          query,
          key,
          b,
          h,
          tile_base,
          k_tile_capacity,
          i,
          j,
          seq_len,
          num_heads,
          dim,
          suffix_window,
          match_lambda,
          scale_abs,
          logit_epsilon);
      const float logit =
          rosa_anchor_candidate_logit(score.score, i, j, scale, sink_threshold, tie_strength);
      const float weight = __expf(logit - row_max);

      float dot = 0.0f;
      for (int d = 0; d < value_dim; ++d) {
        const int64_t go_idx =
            (((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) * value_dim + d);
        const int64_t v_idx =
            (((static_cast<int64_t>(b) * seq_len + j) * num_value_heads + value_head) * value_dim + d);
        const float xv = read_float(value, v_idx) > 0.0f ? 1.0f : -1.0f;
        const float go = use_grad_output_shared ? grad_output_row[d] : read_float(grad_output, go_idx);
        dot += go * xv;
      }

      if (!has_saved_row_stats) {
        local_denom += weight;
      }
      local_delta_num += weight * dot;
    }
    __syncthreads();
  }

  const float sink_weight = __expf(-row_max);
  const float denom = has_saved_row_stats
      ? saved_row_denom[row]
      : sink_weight + block_reduce_sum(local_denom, scratch);
  const float delta_num = block_reduce_sum(local_delta_num, scratch);
  const float delta = delta_num / denom;

  loaded_k_end = -1;
  for (int tile_start = 1; tile_start <= i; tile_start += blockDim.x) {
    const int tile_len = min(blockDim.x, i - tile_start + 1);
    int tile_base = 0;
    if constexpr (kUseKCircular) {
      loaded_k_end = load_k_tile_circular_shared(
          k_bits,
          k_tile,
          b,
          h,
          seq_len,
          num_heads,
          suffix_window,
          tile_start,
          tile_len,
          k_tile_capacity,
          loaded_k_end,
          &tile_base);
    } else {
      tile_base = load_k_tile_linear_shared(
          k_bits, k_tile, b, h, seq_len, num_heads, suffix_window, tile_start, tile_len);
    }

    const int j = tile_start + threadIdx.x;
    if (threadIdx.x < tile_len) {
      const bool use_backward_scan = logit_epsilon > 0.0f && suffix_window >= 64;
      float thread_scan_products[kBackwardScanMaxOffsets];
      bool scan_complete = false;
      const RosaAnchorScore score = use_backward_scan
          ? rosa_anchor_score_pair_scan_shared<scalar_t, kUseKCircular>(
                q_suffix,
                k_tile,
                query,
                key,
                thread_scan_products,
                &scan_complete,
                b,
                h,
                tile_base,
                k_tile_capacity,
                i,
                j,
                seq_len,
                num_heads,
                dim,
                suffix_window,
                match_lambda,
                scale_abs,
                logit_epsilon)
          : rosa_anchor_score_pair_shared<scalar_t, kUseKCircular>(
                q_suffix,
                k_tile,
                query,
                key,
                b,
                h,
                tile_base,
                k_tile_capacity,
                i,
                j,
                seq_len,
                num_heads,
                dim,
                suffix_window,
                match_lambda,
                scale_abs,
                logit_epsilon);
      const float logit =
          rosa_anchor_candidate_logit(score.score, i, j, scale, sink_threshold, tie_strength);
      const float prob = __expf(logit - row_max) / denom;

      float dot = 0.0f;
      for (int d = 0; d < value_dim; ++d) {
        const int64_t go_idx =
            (((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) * value_dim + d);
        const int64_t v_idx =
            (((static_cast<int64_t>(b) * seq_len + j) * num_value_heads + value_head) * value_dim + d);
        const float go = use_grad_output_shared ? grad_output_row[d] : read_float(grad_output, go_idx);
        const float xv = read_float(value, v_idx) > 0.0f ? 1.0f : -1.0f;
        dot += go * xv;

        const float dv_proxy = prob * go;
        const float dv_scale = quant_derivative(value, v_idx);
        atomicAdd(&grad_value[v_idx], dv_proxy * dv_scale);
      }

      const float dscore = scale * prob * (dot - delta);

      if (scan_complete) {
        if (score.effective_offsets == 1) {
          const int offset = 0;
          const int k_pos = j - 1;
          const int q_pos = i;
          const uint32_t q_word = q_suffix[offset];
          const uint32_t k_word = kUseKCircular
              ? k_tile[k_pos & (k_tile_capacity - 1)]
              : k_tile[k_pos - tile_base];
          const float common_base = match_lambda * dscore * thread_scan_products[offset];
          accumulate_pair_grad<scalar_t>(
              query,
              key,
              grad_query,
              grad_key,
              b,
              h,
              q_pos,
              k_pos,
              seq_len,
              num_heads,
              dim,
              q_word,
              k_word,
              common_base);
        } else {
          float suffix_prod = 0.0f;
          for (int offset = score.effective_offsets - 1; offset >= 0; --offset) {
            suffix_prod += thread_scan_products[offset];
            const int k_pos = j - 1 - offset;
            const int q_pos = i - offset;
            const uint32_t q_word = q_suffix[offset];
            const uint32_t k_word = kUseKCircular
                ? k_tile[k_pos & (k_tile_capacity - 1)]
                : k_tile[k_pos - tile_base];
            const float common_base = match_lambda * dscore * suffix_prod;
            accumulate_pair_grad<scalar_t>(
                query,
                key,
                grad_query,
                grad_key,
                b,
                h,
                q_pos,
                k_pos,
                seq_len,
                num_heads,
                dim,
                q_word,
                k_word,
                common_base);
          }
        }
      } else {
        float prefix_prod = 1.0f;
        float suffix_prod = score.score;
        for (int offset = 0; offset < score.effective_offsets; ++offset) {
          const int k_pos = j - 1 - offset;
          const int q_pos = i - offset;
          const uint32_t q_word = q_suffix[offset];
          const uint32_t k_word = kUseKCircular
              ? k_tile[k_pos & (k_tile_capacity - 1)]
              : k_tile[k_pos - tile_base];
          const float match = rosa_anchor_match_weighted<scalar_t>(
              q_word,
              k_word,
              query,
              key,
              b,
              h,
              q_pos,
              k_pos,
              seq_len,
              num_heads,
              dim,
              match_lambda);
          prefix_prod *= match;
          const float common_base = match_lambda * dscore * suffix_prod;
          accumulate_pair_grad<scalar_t>(
              query,
              key,
              grad_query,
              grad_key,
              b,
              h,
              q_pos,
              k_pos,
              seq_len,
              num_heads,
              dim,
              q_word,
              k_word,
              common_base);
          suffix_prod -= prefix_prod;
        }
      }
    }
    __syncthreads();
  }

}

template <typename scalar_t>
void launch_pack(
    const torch::Tensor& x,
    torch::Tensor& bits,
    int64_t total_tokens,
    int dim,
    cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((total_tokens + threads - 1) / threads);
  pack_sign_bits_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      x.data_ptr<scalar_t>(),
      bits.data_ptr<int32_t>(),
      total_tokens,
      dim);
}

bool use_k_circular_buffer(int suffix_window, double logit_epsilon) {
  return logit_epsilon > 0.0 && suffix_window >= kBlockThreads;
}

size_t shared_bytes_for(int suffix_window, bool use_k_circular) {
  return kBlockThreads * sizeof(float) +
      static_cast<size_t>(suffix_window) * sizeof(uint32_t) +
      static_cast<size_t>(
          use_k_circular ? k_tile_capacity_for(suffix_window) : kBlockThreads + suffix_window) *
      sizeof(uint32_t);
}

size_t shared_bytes_for_forward(int suffix_window, int value_dim, bool use_k_circular) {
  return shared_bytes_for(suffix_window, use_k_circular) + static_cast<size_t>(value_dim) * sizeof(float);
}

size_t shared_bytes_for_backward(
    int suffix_window,
    int value_dim,
    bool use_grad_output_shared,
    bool use_k_circular) {
  return shared_bytes_for(suffix_window, use_k_circular) +
      (use_grad_output_shared ? static_cast<size_t>(value_dim) * sizeof(float) : 0);
}

}  // namespace

torch::Tensor rosa_anchor_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon) {
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int dim = static_cast<int>(query.size(3));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));

  auto bits_options = query.options().dtype(torch::kInt32);
  auto q_bits = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto k_bits = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto output = torch::empty({batch_size, seq_len, num_heads, value_dim}, query.options());

  const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len * num_heads;
  const int64_t total_rows = total_tokens;
  const bool use_k_circular =
      use_k_circular_buffer(static_cast<int>(suffix_window), logit_epsilon);
  const size_t shared_bytes =
      shared_bytes_for_forward(static_cast<int>(suffix_window), value_dim, use_k_circular);
  const auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(query.scalar_type(), "rosa_anchor_pack_forward", [&] {
        launch_pack<scalar_t>(query, q_bits, total_tokens, dim, stream);
        launch_pack<scalar_t>(key, k_bits, total_tokens, dim, stream);
        auto launch_forward = [&](auto circular_tag) {
          constexpr bool kLaunchCircular = decltype(circular_tag)::value;
          rosa_anchor_forward_kernel<scalar_t, kLaunchCircular><<<
              total_rows,
              kBlockThreads,
              shared_bytes,
              stream>>>(
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              q_bits.data_ptr<int32_t>(),
              k_bits.data_ptr<int32_t>(),
              value.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              nullptr,
              nullptr,
              nullptr,
              batch_size,
              seq_len,
              num_heads,
              dim,
              num_value_heads,
              value_dim,
              static_cast<float>(scale),
              static_cast<int>(suffix_window),
              static_cast<float>(match_lambda),
              static_cast<float>(sink_threshold),
              static_cast<float>(tie_strength),
              static_cast<float>(logit_epsilon));
        };
        if (use_k_circular) {
          launch_forward(std::true_type{});
        } else {
          launch_forward(std::false_type{});
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> rosa_anchor_forward_with_bits_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon) {
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int dim = static_cast<int>(query.size(3));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));

  auto bits_options = query.options().dtype(torch::kInt32);
  auto q_bits = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto k_bits = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto output = torch::empty({batch_size, seq_len, num_heads, value_dim}, query.options());
  const bool save_row_stats = logit_epsilon > 0.0;
  auto row_max = torch::Tensor();
  auto row_denom = torch::Tensor();
  if (save_row_stats) {
    auto row_options = query.options().dtype(torch::kFloat32);
    row_max = torch::empty({batch_size, num_heads, seq_len}, row_options);
    row_denom = torch::empty({batch_size, num_heads, seq_len}, row_options);
  }

  const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len * num_heads;
  const int64_t total_rows = total_tokens;
  const bool use_k_circular =
      use_k_circular_buffer(static_cast<int>(suffix_window), logit_epsilon);
  const size_t shared_bytes =
      shared_bytes_for_forward(static_cast<int>(suffix_window), value_dim, use_k_circular);
  const auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(query.scalar_type(), "rosa_anchor_pack_forward_with_bits", [&] {
        launch_pack<scalar_t>(query, q_bits, total_tokens, dim, stream);
        launch_pack<scalar_t>(key, k_bits, total_tokens, dim, stream);
        auto launch_forward = [&](auto circular_tag) {
          constexpr bool kLaunchCircular = decltype(circular_tag)::value;
          rosa_anchor_forward_kernel<scalar_t, kLaunchCircular><<<
              total_rows,
              kBlockThreads,
              shared_bytes,
              stream>>>(
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              q_bits.data_ptr<int32_t>(),
              k_bits.data_ptr<int32_t>(),
              value.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              nullptr,
              row_max.defined() ? row_max.data_ptr<float>() : nullptr,
              row_denom.defined() ? row_denom.data_ptr<float>() : nullptr,
              batch_size,
              seq_len,
              num_heads,
              dim,
              num_value_heads,
              value_dim,
              static_cast<float>(scale),
              static_cast<int>(suffix_window),
              static_cast<float>(match_lambda),
              static_cast<float>(sink_threshold),
              static_cast<float>(tie_strength),
              static_cast<float>(logit_epsilon));
        };
        if (use_k_circular) {
          launch_forward(std::true_type{});
        } else {
          launch_forward(std::false_type{});
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (save_row_stats) {
    return {output, q_bits, k_bits, row_max, row_denom};
  }
  return {output, q_bits, k_bits};
}

std::vector<torch::Tensor> rosa_anchor_forward_with_stats_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon) {
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int dim = static_cast<int>(query.size(3));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));

  auto bits_options = query.options().dtype(torch::kInt32);
  auto q_bits = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto k_bits = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto output = torch::empty({batch_size, seq_len, num_heads, value_dim}, query.options());
  auto stats = torch::zeros({kTelemetryCount}, query.options().dtype(torch::kFloat32));
  auto row_options = query.options().dtype(torch::kFloat32);
  auto row_max = torch::empty({batch_size, num_heads, seq_len}, row_options);
  auto row_denom = torch::empty({batch_size, num_heads, seq_len}, row_options);

  const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len * num_heads;
  const int64_t total_rows = total_tokens;
  const bool use_k_circular =
      use_k_circular_buffer(static_cast<int>(suffix_window), logit_epsilon);
  const size_t shared_bytes =
      shared_bytes_for_forward(static_cast<int>(suffix_window), value_dim, use_k_circular);
  const auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(query.scalar_type(), "rosa_anchor_pack_forward_with_stats", [&] {
        launch_pack<scalar_t>(query, q_bits, total_tokens, dim, stream);
        launch_pack<scalar_t>(key, k_bits, total_tokens, dim, stream);
        auto launch_forward = [&](auto circular_tag) {
          constexpr bool kLaunchCircular = decltype(circular_tag)::value;
          rosa_anchor_forward_kernel<scalar_t, kLaunchCircular><<<
              total_rows,
              kBlockThreads,
              shared_bytes,
              stream>>>(
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              q_bits.data_ptr<int32_t>(),
              k_bits.data_ptr<int32_t>(),
              value.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              stats.data_ptr<float>(),
              row_max.data_ptr<float>(),
              row_denom.data_ptr<float>(),
              batch_size,
              seq_len,
              num_heads,
              dim,
              num_value_heads,
              value_dim,
              static_cast<float>(scale),
              static_cast<int>(suffix_window),
              static_cast<float>(match_lambda),
              static_cast<float>(sink_threshold),
              static_cast<float>(tie_strength),
              static_cast<float>(logit_epsilon));
        };
        if (use_k_circular) {
          launch_forward(std::true_type{});
        } else {
          launch_forward(std::false_type{});
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, stats, q_bits, k_bits, row_max, row_denom};
}

std::vector<torch::Tensor> rosa_anchor_backward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon,
    double qk_damper_strength) {
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int dim = static_cast<int>(query.size(3));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));

  auto bits_options = query.options().dtype(torch::kInt32);
  auto q_bits = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto k_bits = torch::empty({batch_size, seq_len, num_heads}, bits_options);

  auto grad_options = query.options().dtype(torch::kFloat32);
  auto grad_query = torch::empty(query.sizes(), grad_options);
  auto grad_key = torch::empty(key.sizes(), grad_options);
  auto grad_value = torch::empty(value.sizes(), grad_options);

  const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len * num_heads;
  const int64_t total_rows = total_tokens;
  const bool use_k_circular =
      use_k_circular_buffer(static_cast<int>(suffix_window), logit_epsilon);
  const size_t shared_bytes =
      shared_bytes_for_backward(
          static_cast<int>(suffix_window),
          value_dim,
          logit_epsilon > 0.0,
          use_k_circular);
  const auto stream = at::cuda::getCurrentCUDAStream();

  cudaMemsetAsync(
      grad_query.data_ptr<float>(),
      0,
      static_cast<size_t>(grad_query.numel()) * sizeof(float),
      stream);
  cudaMemsetAsync(
      grad_key.data_ptr<float>(),
      0,
      static_cast<size_t>(grad_key.numel()) * sizeof(float),
      stream);
  cudaMemsetAsync(
      grad_value.data_ptr<float>(),
      0,
      static_cast<size_t>(grad_value.numel()) * sizeof(float),
      stream);

  DISPATCH_ROSA_FLOAT_TYPES(query.scalar_type(), "rosa_anchor_backward", [&] {
        launch_pack<scalar_t>(query, q_bits, total_tokens, dim, stream);
        launch_pack<scalar_t>(key, k_bits, total_tokens, dim, stream);
        auto launch_backward = [&](auto circular_tag) {
          constexpr bool kLaunchCircular = decltype(circular_tag)::value;
          rosa_anchor_backward_kernel<scalar_t, kLaunchCircular><<<
              total_rows,
              kBlockThreads,
              shared_bytes,
              stream>>>(
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>(),
              grad_output.data_ptr<scalar_t>(),
              q_bits.data_ptr<int32_t>(),
              k_bits.data_ptr<int32_t>(),
              nullptr,
              nullptr,
              grad_query.data_ptr<float>(),
              grad_key.data_ptr<float>(),
              grad_value.data_ptr<float>(),
              batch_size,
              seq_len,
              num_heads,
              dim,
              num_value_heads,
              value_dim,
              static_cast<float>(scale),
              static_cast<int>(suffix_window),
              static_cast<float>(match_lambda),
              static_cast<float>(sink_threshold),
              static_cast<float>(tie_strength),
              static_cast<float>(logit_epsilon));
        };
        if (use_k_circular) {
          launch_backward(std::true_type{});
        } else {
          launch_backward(std::false_type{});
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (qk_damper_strength > 0.0) {
    DISPATCH_ROSA_FLOAT_TYPES(query.scalar_type(), "qk_directional_damper", [&] {
          const int threads = 256;
          const int64_t numel = query.numel();
          const int blocks = static_cast<int>((numel + threads - 1) / threads);
          const float strength = static_cast<float>(qk_damper_strength);
          qk_directional_damper_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              query.data_ptr<scalar_t>(),
              grad_query.data_ptr<float>(),
              numel,
              strength);
          qk_directional_damper_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              key.data_ptr<scalar_t>(),
              grad_key.data_ptr<float>(),
              numel,
              strength);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_query, grad_key, grad_value};
}

std::vector<torch::Tensor> rosa_anchor_backward_with_bits_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor q_bits,
    torch::Tensor k_bits,
    torch::Tensor row_max,
    torch::Tensor row_denom,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon,
    double qk_damper_strength) {
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int dim = static_cast<int>(query.size(3));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));

  auto grad_options = query.options().dtype(torch::kFloat32);
  auto grad_query = torch::empty(query.sizes(), grad_options);
  auto grad_key = torch::empty(key.sizes(), grad_options);
  auto grad_value = torch::empty(value.sizes(), grad_options);

  const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len * num_heads;
  const int64_t total_rows = total_tokens;
  const bool use_k_circular =
      use_k_circular_buffer(static_cast<int>(suffix_window), logit_epsilon);
  const size_t shared_bytes =
      shared_bytes_for_backward(
          static_cast<int>(suffix_window),
          value_dim,
          logit_epsilon > 0.0,
          use_k_circular);
  const auto stream = at::cuda::getCurrentCUDAStream();

  cudaMemsetAsync(
      grad_query.data_ptr<float>(),
      0,
      static_cast<size_t>(grad_query.numel()) * sizeof(float),
      stream);
  cudaMemsetAsync(
      grad_key.data_ptr<float>(),
      0,
      static_cast<size_t>(grad_key.numel()) * sizeof(float),
      stream);
  cudaMemsetAsync(
      grad_value.data_ptr<float>(),
      0,
      static_cast<size_t>(grad_value.numel()) * sizeof(float),
      stream);

  DISPATCH_ROSA_FLOAT_TYPES(query.scalar_type(), "rosa_anchor_backward_with_bits", [&] {
        auto launch_backward = [&](auto circular_tag) {
          constexpr bool kLaunchCircular = decltype(circular_tag)::value;
          rosa_anchor_backward_kernel<scalar_t, kLaunchCircular><<<
              total_rows,
              kBlockThreads,
              shared_bytes,
              stream>>>(
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>(),
              grad_output.data_ptr<scalar_t>(),
              q_bits.data_ptr<int32_t>(),
              k_bits.data_ptr<int32_t>(),
              row_max.defined() ? row_max.data_ptr<float>() : nullptr,
              row_denom.defined() ? row_denom.data_ptr<float>() : nullptr,
              grad_query.data_ptr<float>(),
              grad_key.data_ptr<float>(),
              grad_value.data_ptr<float>(),
              batch_size,
              seq_len,
              num_heads,
              dim,
              num_value_heads,
              value_dim,
              static_cast<float>(scale),
              static_cast<int>(suffix_window),
              static_cast<float>(match_lambda),
              static_cast<float>(sink_threshold),
              static_cast<float>(tie_strength),
              static_cast<float>(logit_epsilon));
        };
        if (use_k_circular) {
          launch_backward(std::true_type{});
        } else {
          launch_backward(std::false_type{});
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (qk_damper_strength > 0.0) {
    DISPATCH_ROSA_FLOAT_TYPES(query.scalar_type(), "qk_directional_damper_with_bits", [&] {
          const int threads = 256;
          const int64_t numel = query.numel();
          const int blocks = static_cast<int>((numel + threads - 1) / threads);
          const float strength = static_cast<float>(qk_damper_strength);
          qk_directional_damper_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              query.data_ptr<scalar_t>(),
              grad_query.data_ptr<float>(),
              numel,
              strength);
          qk_directional_damper_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              key.data_ptr<scalar_t>(),
              grad_key.data_ptr<float>(),
              numel,
              strength);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_query, grad_key, grad_value};
}
