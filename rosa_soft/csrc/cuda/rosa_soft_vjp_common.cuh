#pragma once

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
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

__device__ __forceinline__ float local_match_gate(
    uint32_t query_word,
    uint32_t key_word,
    int symbol_dim,
    float mismatch_scale,
    float inverse_symbol_dim) {
  uint32_t mismatch = query_word ^ key_word;
  if (symbol_dim < 32) {
    mismatch &= (1u << symbol_dim) - 1u;
  }
  return __expf(
      -mismatch_scale *
      static_cast<float>(__popc(mismatch)) *
      inverse_symbol_dim);
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

template <typename scalar_t>
__global__ void finalize_vjp_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    float* __restrict__ grad_query,
    const float* __restrict__ grad_key_accumulator,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int64_t query_elements,
    int64_t key_elements,
    int64_t value_elements,
    int seq_len,
    int num_heads,
    int symbol_dim) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < query_elements) {
    grad_query[index] *= softsign_derivative(read_float(query, index));
    return;
  }
  const int64_t key_index = index - query_elements;
  if (key_index < key_elements) {
    const int bit = key_index % symbol_dim;
    const int64_t token_head = key_index / symbol_dim;
    const int head = token_head % num_heads;
    const int64_t token = token_head / num_heads;
    const int batch = static_cast<int>(token / seq_len);
    const int position = static_cast<int>(token - batch * seq_len);
    const int64_t accumulator_index =
        ((static_cast<int64_t>(batch) * num_heads + head) * symbol_dim + bit) *
            seq_len +
        position;
    grad_key[key_index] = grad_key_accumulator[accumulator_index] *
        softsign_derivative(read_float(key, key_index));
    return;
  }
  const int64_t value_index = key_index - key_elements;
  if (value_index < value_elements) {
    grad_value[value_index] *=
        softsign_derivative(read_float(value, value_index));
  }
}

}  // namespace rosa_soft::cuda
