#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <type_traits>
#include <vector>


namespace {

constexpr int kBlockThreads = 128;
constexpr size_t kPortableSharedMemoryLimit = 48 * 1024;
constexpr float kNullScore = 0.5f;
constexpr float kUniformScale = 1.0f / 16777216.0f;
constexpr uint64_t kCounterStride = 0x9e3779b97f4a7c15ULL;


enum class BackwardPlan {
  ScoreCached,
  KeyReduced,
  Generic,
};


template <BackwardPlan Plan>
struct BackwardPlanFeatures {
  static constexpr bool kCacheGradOutput =
      Plan != BackwardPlan::Generic;
  static constexpr bool kCacheQuery =
      Plan != BackwardPlan::Generic;
  static constexpr bool kCacheScores =
      Plan == BackwardPlan::ScoreCached;
  static constexpr bool kCacheGradKey =
      Plan == BackwardPlan::KeyReduced;
};


struct BackwardSharedLayout {
  size_t stats_offset;
  size_t grad_output_offset;
  size_t query_offset;
  size_t scores_offset;
  size_t grad_key_offset;
  size_t total_words;

  __host__ __device__ size_t total_bytes() const {
    return total_words * sizeof(float);
  }
};


template <BackwardPlan Plan>
__host__ __device__ BackwardSharedLayout make_backward_shared_layout(
    int seq_len,
    int dim,
    int value_dim,
    int max_suffix_length) {
  constexpr bool kCacheGradOutput =
      BackwardPlanFeatures<Plan>::kCacheGradOutput;
  constexpr bool kCacheQuery =
      BackwardPlanFeatures<Plan>::kCacheQuery;
  constexpr bool kCacheScores =
      BackwardPlanFeatures<Plan>::kCacheScores;
  constexpr bool kCacheGradKey =
      BackwardPlanFeatures<Plan>::kCacheGradKey;
  const int cached_query_len =
      max_suffix_length < seq_len ? max_suffix_length : seq_len;

  size_t cursor = 0;
  BackwardSharedLayout layout{};
  layout.stats_offset = cursor;
  cursor += 3 * kBlockThreads;
  layout.grad_output_offset = cursor;
  if constexpr (kCacheGradOutput) {
    cursor += static_cast<size_t>(value_dim);
  }
  layout.query_offset = cursor;
  if constexpr (kCacheQuery) {
    cursor += static_cast<size_t>(cached_query_len);
  }
  layout.scores_offset = cursor;
  if constexpr (kCacheScores) {
    cursor += static_cast<size_t>(seq_len);
  }
  layout.grad_key_offset = cursor;
  if constexpr (kCacheGradKey) {
    const int key_tile_positions =
        kBlockThreads + cached_query_len - 1;
    cursor += static_cast<size_t>(key_tile_positions) *
        static_cast<size_t>(dim);
  }
  layout.total_words = cursor;
  return layout;
}


struct BackwardSharedView {
  float* stats;
  float* grad_output;
  int32_t* query;
  float* scores;
  float* grad_key;
};


template <BackwardPlan Plan>
__device__ __forceinline__ BackwardSharedView bind_backward_shared(
    float* shared,
    int seq_len,
    int dim,
    int value_dim,
    int max_suffix_length) {
  const BackwardSharedLayout layout =
      make_backward_shared_layout<Plan>(
          seq_len,
          dim,
          value_dim,
          max_suffix_length);
  return {
      shared + layout.stats_offset,
      shared + layout.grad_output_offset,
      reinterpret_cast<int32_t*>(shared + layout.query_offset),
      shared + layout.scores_offset,
      shared + layout.grad_key_offset};
}


static_assert(
    sizeof(float) == sizeof(int32_t),
    "shared layout requires float and int32_t to occupy one word");


#define DISPATCH_ROSA_FLOAT_TYPES(TYPE, NAME, ...)              \
  AT_DISPATCH_SWITCH(                                           \
      TYPE,                                                     \
      NAME,                                                     \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))


template <typename scalar_t>
__device__ __forceinline__ float read_float(
    const scalar_t* __restrict__ ptr,
    int64_t idx) {
  return static_cast<float>(ptr[idx]);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t write_scalar(float value) {
  return static_cast<scalar_t>(value);
}


__device__ __forceinline__ float softsign_derivative(float value) {
  const float denom = 1.0f + fabsf(value);
  return 1.0f / (denom * denom);
}


__device__ __forceinline__ int sign_from_bit(uint32_t word, int bit) {
  return ((word >> bit) & 1u) != 0u ? 1 : -1;
}


__device__ __forceinline__ float contiguous_warp_sum(
    float value,
    unsigned mask) {
  const int lane = threadIdx.x & 31;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(mask, value, offset);
    if (lane + offset < 32 &&
        (mask & (1u << (lane + offset))) != 0u) {
      value += other;
    }
  }
  return value;
}


__device__ __forceinline__ unsigned action_offset_warp_mask(
    int tile_start,
    int row,
    int offset) {
  const int warp_action_start =
      tile_start + (threadIdx.x & ~31);
  const int first_lane =
      max(0, offset + 1 - warp_action_start);
  const int last_lane =
      min(31, row - warp_action_start);
  const unsigned through_last = last_lane == 31
      ? 0xffffffffu
      : (1u << (last_lane + 1)) - 1u;
  const unsigned before_first = first_lane == 0
      ? 0u
      : (1u << first_lane) - 1u;
  return through_last & ~before_first;
}


__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}


__device__ __forceinline__ float counter_uniform(
    uint64_t seed,
    uint64_t counter) {
  const uint64_t bits = splitmix64(seed + (counter + 1ULL) * kCounterStride);
  const uint32_t mantissa = static_cast<uint32_t>(bits >> 40);
  return (static_cast<float>(mantissa) + 0.5f) * kUniformScale;
}


__device__ __forceinline__ uint64_t mismatch_counter(
    int b,
    int h,
    int q_pos,
    int k_pos,
    int bit,
    int seq_len,
    int num_heads,
    int dim) {
  uint64_t index = static_cast<uint64_t>(b);
  index = index * static_cast<uint64_t>(num_heads) + static_cast<uint64_t>(h);
  index = index * static_cast<uint64_t>(seq_len) + static_cast<uint64_t>(q_pos);
  index = index * static_cast<uint64_t>(seq_len) + static_cast<uint64_t>(k_pos);
  return index * static_cast<uint64_t>(dim) + static_cast<uint64_t>(bit);
}


template <typename scalar_t>
__global__ void pack_sign_bits_kernel(
    const scalar_t* __restrict__ values,
    int32_t* __restrict__ packed,
    int64_t tokens,
    int dim) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= tokens) {
    return;
  }
  const int64_t base = index * dim;
  uint32_t word = 0u;
  for (int bit = 0; bit < dim; ++bit) {
    if (read_float(values, base + bit) > 0.0f) {
      word |= 1u << bit;
    }
  }
  packed[index] = static_cast<int32_t>(word);
}


__device__ __forceinline__ int64_t token_index(
    int b,
    int t,
    int h,
    int seq_len,
    int num_heads) {
  return (static_cast<int64_t>(b) * seq_len + t) * num_heads + h;
}


__device__ __forceinline__ uint32_t load_word(
    const int32_t* __restrict__ bits,
    int b,
    int t,
    int h,
    int seq_len,
    int num_heads) {
  return static_cast<uint32_t>(
      bits[token_index(b, t, h, seq_len, num_heads)]);
}


__device__ __forceinline__ int exact_suffix_length(
    const int32_t* __restrict__ query_symbols,
    const int32_t* __restrict__ key_symbols,
    int b,
    int h,
    int i,
    int action,
    int seq_len,
    int num_heads,
    int max_suffix_length) {
  const int valid = min(max_suffix_length, min(i + 1, action));
  for (int offset = 0; offset < valid; ++offset) {
    const uint32_t q_word =
        load_word(query_symbols, b, i - offset, h, seq_len, num_heads);
    const uint32_t k_word =
        load_word(key_symbols, b, action - 1 - offset, h, seq_len, num_heads);
    if (q_word != k_word) {
      return offset;
    }
  }
  return valid;
}


template <typename scalar_t>
__global__ void hard_forward_kernel(
    const int32_t* __restrict__ query_symbols,
    const int32_t* __restrict__ key_symbols,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int max_suffix_length) {
  extern __shared__ int shared_int[];
  int* lengths = shared_int;
  int* actions = lengths + blockDim.x;

  const int row = blockIdx.x;
  const int i = row % seq_len;
  const int h = (row / seq_len) % num_heads;
  const int b = row / (seq_len * num_heads);
  const int tid = threadIdx.x;

  int best_length = 0;
  int best_action = 0;
  for (int action = tid + 1; action <= i; action += blockDim.x) {
    const int length = exact_suffix_length(
        query_symbols,
        key_symbols,
        b,
        h,
        i,
        action,
        seq_len,
        num_heads,
        max_suffix_length);
    if (length > best_length ||
        (length > 0 && length == best_length && action > best_action)) {
      best_length = length;
      best_action = action;
    }
  }

  lengths[tid] = best_length;
  actions[tid] = best_action;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const int other_length = lengths[tid + stride];
      const int other_action = actions[tid + stride];
      if (other_length > lengths[tid] ||
          (other_length > 0 &&
           other_length == lengths[tid] &&
           other_action > actions[tid])) {
        lengths[tid] = other_length;
        actions[tid] = other_action;
      }
    }
    __syncthreads();
  }

  const int selected = lengths[0] > 0 ? actions[0] : 0;
  const int value_head = h / (num_heads / num_value_heads);
  for (int d = tid; d < value_dim; d += blockDim.x) {
    float result = 0.0f;
    if (selected > 0) {
      const int64_t value_idx =
          ((static_cast<int64_t>(b) * seq_len + selected) *
               num_value_heads +
           value_head) *
              value_dim +
          d;
      result = read_float(value, value_idx) > 0.0f ? 1.0f : -1.0f;
    }
    const int64_t output_idx =
        ((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) *
            value_dim +
        d;
    output[output_idx] = write_scalar<scalar_t>(result);
  }
}


struct LocalGate {
  float value;
  float derivative_ratio;
};


__device__ __forceinline__ LocalGate local_gate(
    uint32_t q_word,
    uint32_t k_word,
    uint64_t seed,
    int b,
    int h,
    int q_pos,
    int k_pos,
    int seq_len,
    int num_heads,
    int dim,
    float mismatch_penalty) {
  uint32_t mismatch = q_word ^ k_word;
  if (dim < 32) {
    mismatch &= (1u << dim) - 1u;
  }
  if (mismatch == 0u) {
    return {1.0f, 1.0f};
  }

  int hard_hamming = 0;
  float relaxed_hamming = 0.0f;
  while (mismatch != 0u) {
    const int bit = __ffs(static_cast<int>(mismatch)) - 1;
    const uint64_t counter = mismatch_counter(
        b,
        h,
        q_pos,
        k_pos,
        bit,
        seq_len,
        num_heads,
        dim);
    const float uniform = counter_uniform(seed, counter);
    const float alpha =
        1.0f - 0.5f * uniform * uniform * uniform;
    relaxed_hamming += alpha;
    ++hard_hamming;
    mismatch &= mismatch - 1u;
  }

  const float gate =
      __expf(-mismatch_penalty * relaxed_hamming);
  const float derivative_ratio =
      __expf(
          -mismatch_penalty *
          (static_cast<float>(hard_hamming) - relaxed_hamming));
  return {gate, derivative_ratio};
}


template <bool CacheQuery>
__device__ __forceinline__ float proxy_score(
    const int32_t* __restrict__ query_symbols,
    const int32_t* __restrict__ key_symbols,
    const int32_t* __restrict__ cached_query,
    uint64_t seed,
    int b,
    int h,
    int i,
    int action,
    int seq_len,
    int num_heads,
    int dim,
    int max_suffix_length,
    float mismatch_penalty) {
  const int valid = min(max_suffix_length, min(i + 1, action));
  float product = 1.0f;
  float score = 0.0f;
  for (int offset = 0; offset < valid; ++offset) {
    const int q_pos = i - offset;
    const int k_pos = action - 1 - offset;
    const uint32_t q_word = CacheQuery
        ? static_cast<uint32_t>(cached_query[offset])
        : load_word(query_symbols, b, q_pos, h, seq_len, num_heads);
    const uint32_t k_word =
        load_word(key_symbols, b, k_pos, h, seq_len, num_heads);
    product *= local_gate(
                   q_word,
                   k_word,
                   seed,
                   b,
                   h,
                   q_pos,
                   k_pos,
                   seq_len,
                   num_heads,
                   dim,
                   mismatch_penalty)
                   .value;
    score += product;
  }
  return score;
}


struct SoftmaxStats {
  float max_logit;
  float normalizer;
  float utility_numerator;
};


__device__ __forceinline__ SoftmaxStats merge_softmax_stats(
    SoftmaxStats left,
    SoftmaxStats right) {
  const float merged_max = fmaxf(left.max_logit, right.max_logit);
  const float left_scale = __expf(left.max_logit - merged_max);
  const float right_scale = __expf(right.max_logit - merged_max);
  return {
      merged_max,
      left.normalizer * left_scale + right.normalizer * right_scale,
      left.utility_numerator * left_scale +
          right.utility_numerator * right_scale};
}


__device__ __forceinline__ SoftmaxStats append_softmax_item(
    SoftmaxStats stats,
    float logit,
    float utility) {
  if (logit > stats.max_logit) {
    const float scale = __expf(stats.max_logit - logit);
    return {
        logit,
        stats.normalizer * scale + 1.0f,
        stats.utility_numerator * scale + utility};
  }
  const float weight = __expf(logit - stats.max_logit);
  return {
      stats.max_logit,
      stats.normalizer + weight,
      stats.utility_numerator + weight * utility};
}


__device__ SoftmaxStats block_reduce_softmax_stats(
    SoftmaxStats stats,
    float* shared) {
  const int tid = threadIdx.x;
  float* shared_max = shared;
  float* shared_normalizer = shared + blockDim.x;
  float* shared_utility = shared + 2 * blockDim.x;
  shared_max[tid] = stats.max_logit;
  shared_normalizer[tid] = stats.normalizer;
  shared_utility[tid] = stats.utility_numerator;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const SoftmaxStats merged = merge_softmax_stats(
          {
              shared_max[tid],
              shared_normalizer[tid],
              shared_utility[tid]},
          {
              shared_max[tid + stride],
              shared_normalizer[tid + stride],
              shared_utility[tid + stride]});
      shared_max[tid] = merged.max_logit;
      shared_normalizer[tid] = merged.normalizer;
      shared_utility[tid] = merged.utility_numerator;
    }
    __syncthreads();
  }
  return {shared_max[0], shared_normalizer[0], shared_utility[0]};
}


template <typename scalar_t, bool CacheGradOutput>
__device__ __forceinline__ float candidate_utility(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ cached_grad_output,
    int b,
    int h,
    int i,
    int action,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  const int value_head = h / (num_heads / num_value_heads);
  float utility = 0.0f;
  for (int d = 0; d < value_dim; ++d) {
    const int64_t value_idx =
        ((static_cast<int64_t>(b) * seq_len + action) *
             num_value_heads +
         value_head) *
            value_dim +
        d;
    const int64_t grad_idx =
        ((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) *
            value_dim +
        d;
    const float signed_value =
        read_float(value, value_idx) > 0.0f ? 1.0f : -1.0f;
    const float grad = CacheGradOutput
        ? cached_grad_output[d]
        : read_float(grad_output, grad_idx);
    utility += grad * signed_value;
  }
  return utility;
}


template <typename scalar_t, bool CacheGradKey>
__device__ __forceinline__ void accumulate_qk_vjp(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ cached_grad_key,
    int key_tile_start,
    int b,
    int h,
    int q_pos,
    int k_pos,
    int seq_len,
    int num_heads,
    int dim,
    uint32_t q_word,
    uint32_t k_word,
    float common,
    unsigned active) {
  const int64_t q_base =
      token_index(b, q_pos, h, seq_len, num_heads) * dim;
  const int64_t k_base =
      token_index(b, k_pos, h, seq_len, num_heads) * dim;
  const int lane = threadIdx.x & 31;
  const int leader = __ffs(static_cast<int>(active)) - 1;
  for (int bit = 0; bit < dim; ++bit) {
    const float q_sign = static_cast<float>(sign_from_bit(q_word, bit));
    const float k_sign = static_cast<float>(sign_from_bit(k_word, bit));
    float q_jacobian = 0.0f;
    if (lane == leader) {
      q_jacobian =
          softsign_derivative(read_float(query, q_base + bit));
    }
    q_jacobian = __shfl_sync(active, q_jacobian, leader);
    const float k_value = read_float(key, k_base + bit);
    const float q_contribution =
        0.5f * common * k_sign * q_jacobian;
    const float q_warp_sum =
        contiguous_warp_sum(q_contribution, active);
    if (lane == leader) {
      atomicAdd(&grad_query[q_base + bit], q_warp_sum);
    }
    const float k_contribution =
        0.5f * common * q_sign * softsign_derivative(k_value);
    if constexpr (CacheGradKey) {
      const int key_offset = k_pos - key_tile_start;
      atomicAdd(&cached_grad_key[key_offset * dim + bit], k_contribution);
    } else {
      atomicAdd(&grad_key[k_base + bit], k_contribution);
    }
  }
}


template <typename scalar_t, BackwardPlan Plan>
__global__ void soft_backward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ query_symbols,
    const int32_t* __restrict__ key_symbols,
    const int64_t* __restrict__ rng_seed,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int seq_len,
    int num_heads,
    int dim,
    int num_value_heads,
    int value_dim,
    int max_suffix_length,
    float inverse_route_temperature,
    float mismatch_penalty) {
  constexpr bool kCacheGradOutput =
      BackwardPlanFeatures<Plan>::kCacheGradOutput;
  constexpr bool kCacheQuery =
      BackwardPlanFeatures<Plan>::kCacheQuery;
  constexpr bool kCacheScores =
      BackwardPlanFeatures<Plan>::kCacheScores;
  constexpr bool kCacheGradKey =
      BackwardPlanFeatures<Plan>::kCacheGradKey;
  extern __shared__ float shared_float[];
  const BackwardSharedView shared = bind_backward_shared<Plan>(
      shared_float,
      seq_len,
      dim,
      value_dim,
      max_suffix_length);
  float* cached_grad_output = shared.grad_output;
  int32_t* cached_query = shared.query;
  float* cached_scores = shared.scores;
  float* cached_grad_key = shared.grad_key;
  const int row = blockIdx.x;
  const int i = row % seq_len;
  const int h = (row / seq_len) % num_heads;
  const int b = row / (seq_len * num_heads);
  const int tid = threadIdx.x;
  const uint64_t seed = static_cast<uint64_t>(rng_seed[0]);
  const float null_logit = kNullScore * inverse_route_temperature;

  if constexpr (kCacheGradOutput || kCacheQuery) {
    const int64_t grad_base =
        ((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) *
        value_dim;
    if constexpr (kCacheGradOutput) {
      for (int d = tid; d < value_dim; d += blockDim.x) {
        cached_grad_output[d] = read_float(grad_output, grad_base + d);
      }
    }
    if constexpr (kCacheQuery) {
      const int query_count = min(max_suffix_length, i + 1);
      for (int offset = tid; offset < query_count; offset += blockDim.x) {
        cached_query[offset] = static_cast<int32_t>(
            load_word(query_symbols, b, i - offset, h, seq_len, num_heads));
      }
    }
    __syncthreads();
  }

  SoftmaxStats local_stats =
      tid == 0
      ? SoftmaxStats{null_logit, 1.0f, 0.0f}
      : SoftmaxStats{-FLT_MAX, 0.0f, 0.0f};
  for (int tile_start = 1; tile_start <= i; tile_start += blockDim.x) {
    const int action = tile_start + tid;
    if (action > i) {
      continue;
    }
    const float score = proxy_score<kCacheQuery>(
        query_symbols,
        key_symbols,
        cached_query,
        seed,
        b,
        h,
        i,
        action,
        seq_len,
        num_heads,
        dim,
        max_suffix_length,
        mismatch_penalty);
    if constexpr (kCacheScores) {
      cached_scores[action - 1] = score;
    }
    const float utility = candidate_utility<scalar_t, kCacheGradOutput>(
        value,
        grad_output,
        cached_grad_output,
        b,
        h,
        i,
        action,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim);
    local_stats = append_softmax_item(
        local_stats,
        score * inverse_route_temperature,
        utility);
  }
  const SoftmaxStats row_stats =
      block_reduce_softmax_stats(local_stats, shared.stats);
  const float row_max = row_stats.max_logit;
  const float denominator = row_stats.normalizer;
  const float expected_utility =
      row_stats.utility_numerator / row_stats.normalizer;

  const int value_head = h / (num_heads / num_value_heads);
  for (int tile_start = 1; tile_start <= i; tile_start += blockDim.x) {
    const int action = tile_start + tid;
    const int key_tile_start = tile_start - max_suffix_length;
    const int key_tile_positions = blockDim.x + max_suffix_length - 1;
    const int key_tile_values = key_tile_positions * dim;
    if constexpr (kCacheGradKey) {
      for (int index = tid; index < key_tile_values; index += blockDim.x) {
        cached_grad_key[index] = 0.0f;
      }
      __syncthreads();
    }

    if (action <= i) {
      float score;
      if constexpr (kCacheScores) {
        score = cached_scores[action - 1];
      } else {
        score = proxy_score<kCacheQuery>(
            query_symbols,
            key_symbols,
            cached_query,
            seed,
            b,
            h,
            i,
            action,
            seq_len,
            num_heads,
            dim,
            max_suffix_length,
            mismatch_penalty);
      }
      const float probability =
          __expf(score * inverse_route_temperature - row_max) / denominator;
      const float utility = candidate_utility<scalar_t, kCacheGradOutput>(
          value,
          grad_output,
          cached_grad_output,
          b,
          h,
          i,
          action,
          seq_len,
          num_heads,
          num_value_heads,
          value_dim);
      const float dscore =
          inverse_route_temperature * probability * (utility - expected_utility);

      for (int d = 0; d < value_dim; ++d) {
        const int64_t value_idx =
            ((static_cast<int64_t>(b) * seq_len + action) *
                 num_value_heads +
             value_head) *
                value_dim +
            d;
        const float value_logit = read_float(value, value_idx);
        const int64_t grad_idx =
            ((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) *
                value_dim +
            d;
        const float grad = kCacheGradOutput
            ? cached_grad_output[d]
            : read_float(grad_output, grad_idx);
        atomicAdd(
            &grad_value[value_idx],
            probability * grad * softsign_derivative(value_logit));
      }

      const int valid = min(max_suffix_length, min(i + 1, action));
      float prefix_product = 1.0f;
      float tail_sum = score;
      for (int offset = 0; offset < valid; ++offset) {
        const unsigned offset_mask =
            action_offset_warp_mask(tile_start, i, offset);
        const int q_pos = i - offset;
        const int k_pos = action - 1 - offset;
        const uint32_t q_word = kCacheQuery
            ? static_cast<uint32_t>(cached_query[offset])
            : load_word(query_symbols, b, q_pos, h, seq_len, num_heads);
        const uint32_t k_word =
            load_word(key_symbols, b, k_pos, h, seq_len, num_heads);
        const LocalGate gate = local_gate(
            q_word,
            k_word,
            seed,
            b,
            h,
            q_pos,
            k_pos,
            seq_len,
            num_heads,
            dim,
            mismatch_penalty);
        prefix_product *= gate.value;
        const float common =
            mismatch_penalty *
            dscore *
            fmaxf(tail_sum, 0.0f) *
            gate.derivative_ratio;
        accumulate_qk_vjp<scalar_t, kCacheGradKey>(
            query,
            key,
            grad_query,
            grad_key,
            cached_grad_key,
            key_tile_start,
            b,
            h,
            q_pos,
            k_pos,
            seq_len,
            num_heads,
            dim,
            q_word,
            k_word,
            common,
            offset_mask);
        tail_sum -= prefix_product;
      }
    }

    if constexpr (kCacheGradKey) {
      __syncthreads();
      for (int index = tid; index < key_tile_values; index += blockDim.x) {
        const int key_offset = index / dim;
        const int bit = index - key_offset * dim;
        const int k_pos = key_tile_start + key_offset;
        const float gradient = cached_grad_key[index];
        if (k_pos >= 0 && k_pos < seq_len && gradient != 0.0f) {
          const int64_t k_base =
              token_index(b, k_pos, h, seq_len, num_heads) * dim;
          atomicAdd(&grad_key[k_base + bit], gradient);
        }
      }
      __syncthreads();
    }
  }
}


template <typename scalar_t>
void launch_pack(
    const torch::Tensor& values,
    torch::Tensor& packed,
    int64_t tokens,
    int dim,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((tokens + threads - 1) / threads);
  pack_sign_bits_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      values.data_ptr<scalar_t>(),
      packed.data_ptr<int32_t>(),
      tokens,
      dim);
}

}  // namespace


std::vector<torch::Tensor> soft_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t max_suffix_length) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int dim = static_cast<int>(query.size(3));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  const int64_t packed_tokens =
      static_cast<int64_t>(batch_size) * seq_len * num_heads;
  const int64_t rows = packed_tokens;

  auto bits_options = query.options().dtype(torch::kInt32);
  auto query_symbols = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto key_symbols = torch::empty({batch_size, seq_len, num_heads}, bits_options);
  auto output = torch::empty(
      {batch_size, seq_len, num_heads, value_dim},
      query.options());
  const auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_hard_forward",
      [&] {
        launch_pack<scalar_t>(
            query,
            query_symbols,
            packed_tokens,
            dim,
            stream);
        launch_pack<scalar_t>(
            key,
            key_symbols,
            packed_tokens,
            dim,
            stream);
        const size_t shared_bytes =
            2 * kBlockThreads * sizeof(int);
        hard_forward_kernel<scalar_t><<<
            rows,
            kBlockThreads,
            shared_bytes,
            stream>>>(
            query_symbols.data_ptr<int32_t>(),
            key_symbols.data_ptr<int32_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            seq_len,
            num_heads,
            num_value_heads,
            value_dim,
            static_cast<int>(max_suffix_length));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, query_symbols, key_symbols};
}


std::vector<torch::Tensor> soft_backward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor query_symbols,
    torch::Tensor key_symbols,
    torch::Tensor rng_seed,
    int64_t max_suffix_length,
    float inverse_route_temperature,
    float mismatch_penalty) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int dim = static_cast<int>(query.size(3));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  const int64_t rows =
      static_cast<int64_t>(batch_size) * seq_len * num_heads;

  auto grad_options = query.options().dtype(torch::kFloat32);
  auto grad_query = torch::zeros(query.sizes(), grad_options);
  auto grad_key = torch::zeros(key.sizes(), grad_options);
  auto grad_value = torch::zeros(value.sizes(), grad_options);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const BackwardSharedLayout score_layout =
      make_backward_shared_layout<BackwardPlan::ScoreCached>(
          seq_len,
          dim,
          value_dim,
          static_cast<int>(max_suffix_length));
  const BackwardSharedLayout key_layout =
      make_backward_shared_layout<BackwardPlan::KeyReduced>(
          seq_len,
          dim,
          value_dim,
          static_cast<int>(max_suffix_length));
  const BackwardSharedLayout generic_layout =
      make_backward_shared_layout<BackwardPlan::Generic>(
          seq_len,
          dim,
          value_dim,
          static_cast<int>(max_suffix_length));
  const size_t score_bytes =
      static_cast<size_t>(seq_len) * sizeof(float);
  const int cached_query_len =
      std::min(static_cast<int>(max_suffix_length), seq_len);
  const int key_tile_positions =
      kBlockThreads + cached_query_len - 1;
  const size_t key_gradient_bytes =
      static_cast<size_t>(key_tile_positions) *
      static_cast<size_t>(dim) *
      sizeof(float);
  const bool score_plan_fits =
      score_layout.total_bytes() <= kPortableSharedMemoryLimit;
  const bool key_plan_fits =
      key_layout.total_bytes() <= kPortableSharedMemoryLimit;

  BackwardPlan plan = BackwardPlan::Generic;
  size_t shared_bytes = generic_layout.total_bytes();
  if (score_plan_fits &&
      (!key_plan_fits || score_bytes <= key_gradient_bytes)) {
    plan = BackwardPlan::ScoreCached;
    shared_bytes = score_layout.total_bytes();
  } else if (key_plan_fits) {
    plan = BackwardPlan::KeyReduced;
    shared_bytes = key_layout.total_bytes();
  }

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_backward",
      [&] {
        const auto launch_backward = [&](auto plan_tag) {
          constexpr BackwardPlan kPlan =
              decltype(plan_tag)::value;
          soft_backward_kernel<scalar_t, kPlan><<<
              rows,
              kBlockThreads,
              shared_bytes,
              stream>>>(
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>(),
              grad_output.data_ptr<scalar_t>(),
              query_symbols.data_ptr<int32_t>(),
              key_symbols.data_ptr<int32_t>(),
              rng_seed.data_ptr<int64_t>(),
              grad_query.data_ptr<float>(),
              grad_key.data_ptr<float>(),
              grad_value.data_ptr<float>(),
              seq_len,
              num_heads,
              dim,
              num_value_heads,
              value_dim,
              static_cast<int>(max_suffix_length),
              inverse_route_temperature,
              mismatch_penalty);
        };
        switch (plan) {
          case BackwardPlan::ScoreCached:
            launch_backward(std::integral_constant<
                BackwardPlan,
                BackwardPlan::ScoreCached>{});
            break;
          case BackwardPlan::KeyReduced:
            launch_backward(std::integral_constant<
                BackwardPlan,
                BackwardPlan::KeyReduced>{});
            break;
          case BackwardPlan::Generic:
            launch_backward(std::integral_constant<
                BackwardPlan,
                BackwardPlan::Generic>{});
            break;
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_query, grad_key, grad_value};
}
