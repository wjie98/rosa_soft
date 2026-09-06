#pragma once

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <limits>
#include <tuple>

#define DISPATCH_ROSA_FLOAT_TYPES(TYPE, NAME, ...)              \
  AT_DISPATCH_SWITCH(                                           \
      TYPE,                                                     \
      NAME,                                                     \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

namespace rosa_soft::cuda {

constexpr int kWarpSize = 32;
constexpr int kGradQuery = 1;
constexpr int kGradKey = 2;
constexpr int kGradValue = 4;
constexpr float kNullScore = 0.5f;
constexpr float kSqrtSuffixScale = 2.414213562373095f;

__device__ __forceinline__ uint32_t symbol_mask(int d) {
  return d == 32 ? 0xffffffffu : (1u << d) - 1u;
}

__device__ __forceinline__ __half2 binary_sign_half2(__half2 x) {
  const __half2 p = __hgt2(x, __float2half2_rn(0.0f));
  return __hfma2(p, __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
}

__device__ __forceinline__ void warp_forward_affine_scan(float& a, float& b) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float left_a = __shfl_up_sync(0xffffffffu, a, offset);
    const float left_b = __shfl_up_sync(0xffffffffu, b, offset);
    if (lane >= offset) {
      b = fmaf(a, left_b, b);
      a *= left_a;
    }
  }
}

__device__ __forceinline__ void warp_reverse_affine_scan(
    float& a, float& b, int n) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float right_a = __shfl_down_sync(0xffffffffu, a, offset);
    const float right_b = __shfl_down_sync(0xffffffffu, b, offset);
    if (lane + offset < n) {
      b = fmaf(a, right_b, b);
      a *= right_a;
    }
  }
}

template <typename scalar_t>
__device__ __forceinline__ float read_float(
    const scalar_t* __restrict__ pointer,
    int64_t index) {
  return static_cast<float>(pointer[index]);
}

__device__ __forceinline__ float softsign_derivative(float value) {
  const float denominator = 1.0f + fabsf(value);
  return 1.0f / (denominator * denominator);
}

__device__ __forceinline__ int sign_from_bit(uint32_t word, int bit) {
  return ((word >> bit) & 1u) != 0u ? 1 : -1;
}

__device__ __forceinline__ void initialize_mismatch_gate_lut(
    float* __restrict__ gate_lut,
    int symbol_dim,
    float mismatch_unit) {
  for (int mismatch = threadIdx.x;
       mismatch <= symbol_dim;
       mismatch += blockDim.x) {
    gate_lut[mismatch] =
        __expf(-mismatch_unit * static_cast<float>(mismatch));
  }
  __syncthreads();
}

__device__ __forceinline__ float mismatch_gate_from_lut(
    uint32_t query_word,
    uint32_t key_word,
    uint32_t symbol_mask,
    const float* __restrict__ gate_lut) {
  return gate_lut[__popc((query_word ^ key_word) & symbol_mask)];
}

__device__ __forceinline__ uint32_t hash_dropout_counter(uint32_t state) {
  state ^= state >> 16;
  state *= 0x7feb352du;
  state ^= state >> 15;
  state *= 0x846ca68bu;
  state ^= state >> 16;
  return state;
}

__device__ __forceinline__ float attention_dropout_scale(
    const int64_t* __restrict__ dropout_seed,
    float dropout_p,
    float inverse_keep_probability,
    int batch,
    int head,
    int query_position,
    int route_position) {
  if (dropout_p == 0.0f) {
    return 1.0f;
  }
  const uint64_t seed = static_cast<uint64_t>(*dropout_seed);
  uint32_t state = hash_dropout_counter(
      static_cast<uint32_t>(route_position) ^ 0x68e31da4u);
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(query_position) ^ 0xb5297a4du);
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(head) ^ 0x63d83595u);
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(batch) ^ 0xa511e9b3u);
  state = hash_dropout_counter(state ^ static_cast<uint32_t>(seed));
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(seed >> 32) ^ 0x9e3779b9u);
  const float uniform =
      static_cast<float>(state >> 8) * 0x1.0p-24f;
  return uniform >= dropout_p ? inverse_keep_probability : 0.0f;
}

struct SoftmaxStats {
  float maximum;
  float normalizer;
  float utility_numerator;
};

__device__ __forceinline__ SoftmaxStats merge_stats(
    SoftmaxStats left,
    SoftmaxStats right) {
  const float maximum = fmaxf(left.maximum, right.maximum);
  const float left_scale = __expf(left.maximum - maximum);
  const float right_scale = __expf(right.maximum - maximum);
  return {
      maximum,
      left.normalizer * left_scale + right.normalizer * right_scale,
      left.utility_numerator * left_scale +
          right.utility_numerator * right_scale};
}

__device__ __forceinline__ SoftmaxStats append_item(
    SoftmaxStats stats,
    float logit,
    float utility) {
  if (logit > stats.maximum) {
    const float old_scale = __expf(stats.maximum - logit);
    return {
        logit,
        stats.normalizer * old_scale + 1.0f,
        stats.utility_numerator * old_scale + utility};
  }
  const float weight = __expf(logit - stats.maximum);
  return {
      stats.maximum,
      stats.normalizer + weight,
      stats.utility_numerator + weight * utility};
}

__device__ __forceinline__ SoftmaxStats warp_reduce_stats(
    SoftmaxStats stats) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const SoftmaxStats right = {
        __shfl_down_sync(0xffffffffu, stats.maximum, offset),
        __shfl_down_sync(0xffffffffu, stats.normalizer, offset),
        __shfl_down_sync(
            0xffffffffu,
            stats.utility_numerator,
            offset)};
    if ((threadIdx.x & 31) + offset < 32) {
      stats = merge_stats(stats, right);
    }
  }
  return stats;
}

// Reduce independent candidates with one exponential per active lane. The
// generic merge above remains useful for already-normalized partial groups,
// but applying it to 32 singleton candidates performs redundant SFU work.
__device__ __forceinline__ SoftmaxStats warp_reduce_candidates(
    float logit,
    float utility,
    bool active) {
  float maximum = active ? logit : -FLT_MAX;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = fmaxf(
        maximum,
        __shfl_down_sync(0xffffffffu, maximum, offset));
  }
  maximum = __shfl_sync(0xffffffffu, maximum, 0);

  float weight = active ? __expf(logit - maximum) : 0.0f;
  float weighted_utility = weight * utility;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    weight += __shfl_down_sync(0xffffffffu, weight, offset);
    weighted_utility +=
        __shfl_down_sync(0xffffffffu, weighted_utility, offset);
  }
  return {maximum, weight, weighted_utility};
}

struct ScoreTransform {
  float route_score;
  float raw_vjp_multiplier;
};

__device__ __forceinline__ ScoreTransform transform_score(float raw_score) {
  const float root = sqrtf(1.0f + raw_score);
  return {
      kSqrtSuffixScale * (root - 1.0f),
      0.5f * kSqrtSuffixScale / root};
}

}  // namespace rosa_soft::cuda
