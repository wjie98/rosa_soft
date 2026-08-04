#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <tuple>
#include <type_traits>


namespace {

constexpr int kBlockThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr size_t kPortableSharedMemoryLimit = 48 * 1024;
constexpr float kNullScore = 0.5f;
constexpr float kNormalizedSqrtSuffixScale = 2.414213562373095f;
constexpr int kGradQuery = 1;
constexpr int kGradKey = 2;
constexpr int kGradValue = 4;
constexpr int kMinKeyAggregationSequenceLength = 256;
constexpr int kMinDenseKeyAggregationSequenceLength = 512;
constexpr int kMinDenseQueryKeyAggregationSequenceLength = 1024;
constexpr int kMinCooperativeSequenceLength = 256;
constexpr int kMinCooperativeValueDimension = 64;
constexpr int kMinPackedQkCooperativeValueDimension = 128;
constexpr int kMinPackedScoreCacheSequenceLength = 256;
constexpr int kPackedScoreCacheCapacity = 1024;
constexpr int kMinDenseRecomputeSequenceLength = 4096;
constexpr int kMinValueOnlyRoutesPerSuffixStepForRecompute = 64;


enum class BackwardPlan {
  CacheSuffixScores,
  RecomputeSuffixScores,
};


template <BackwardPlan Plan>
struct BackwardCacheTraits {
  static constexpr bool kCacheGradOutput =
      Plan == BackwardPlan::CacheSuffixScores;
  static constexpr bool kCacheQuery =
      Plan == BackwardPlan::CacheSuffixScores;
  static constexpr bool kCacheSuffixScores =
      Plan == BackwardPlan::CacheSuffixScores;
};


struct BackwardSharedLayout {
  size_t stats_word_offset;
  size_t grad_output_word_offset;
  size_t query_word_offset;
  size_t suffix_scores_word_offset;
  size_t total_words;

  __host__ __device__ size_t total_bytes() const {
    return total_words * sizeof(float);
  }
};


template <BackwardPlan Plan>
__host__ __device__ BackwardSharedLayout make_backward_shared_layout(
    int seq_len,
    int value_dim,
    int max_suffix_length) {
  constexpr bool kCacheGradOutput =
      BackwardCacheTraits<Plan>::kCacheGradOutput;
  constexpr bool kCacheQuery =
      BackwardCacheTraits<Plan>::kCacheQuery;
  constexpr bool kCacheSuffixScores =
      BackwardCacheTraits<Plan>::kCacheSuffixScores;
  size_t cursor = 0;
  BackwardSharedLayout layout{};
  layout.stats_word_offset = cursor;
  cursor += 3 * kWarpsPerBlock;
  layout.grad_output_word_offset = cursor;
  if constexpr (kCacheGradOutput) {
    cursor += static_cast<size_t>(value_dim);
  }
  layout.query_word_offset = cursor;
  if constexpr (kCacheQuery) {
    cursor += static_cast<size_t>(max_suffix_length);
  }
  layout.suffix_scores_word_offset = cursor;
  if constexpr (kCacheSuffixScores) {
    cursor += static_cast<size_t>(seq_len - 1);
  }
  layout.total_words = cursor;
  return layout;
}


struct BackwardSharedView {
  float* stats;
  float* grad_output;
  int32_t* query;
  float* suffix_scores;
};


template <BackwardPlan Plan>
__device__ __forceinline__ BackwardSharedView bind_backward_shared(
    float* shared,
    int seq_len,
    int value_dim,
    int max_suffix_length) {
  const BackwardSharedLayout layout =
      make_backward_shared_layout<Plan>(
          seq_len,
          value_dim,
          max_suffix_length);
  return {
      shared + layout.stats_word_offset,
      shared + layout.grad_output_word_offset,
      reinterpret_cast<int32_t*>(shared + layout.query_word_offset),
      shared + layout.suffix_scores_word_offset};
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
  for (int lane_delta = 16; lane_delta > 0; lane_delta >>= 1) {
    const float other = __shfl_down_sync(mask, value, lane_delta);
    if (lane + lane_delta < 32 &&
        (mask & (1u << (lane + lane_delta))) != 0u) {
      value += other;
    }
  }
  return value;
}


__device__ __forceinline__ unsigned route_suffix_warp_mask(
    int tile_start,
    int row,
    int suffix_offset_tokens) {
  const int warp_route_start =
      tile_start + (threadIdx.x & ~31);
  const int first_lane =
      max(0, suffix_offset_tokens + 1 - warp_route_start);
  const int last_lane =
      min(31, row - warp_route_start);
  const unsigned through_last = last_lane == 31
      ? 0xffffffffu
      : (1u << (last_lane + 1)) - 1u;
  const unsigned before_first = first_lane == 0
      ? 0u
      : (1u << first_lane) - 1u;
  return through_last & ~before_first;
}


template <typename scalar_t, bool HeadMajor>
__global__ void pack_sign_bits_pair_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    int32_t* __restrict__ packed_query,
    int32_t* __restrict__ packed_key,
    int64_t symbol_rows,
    int seq_len,
    int num_heads,
    int symbol_dim) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= symbol_rows) {
    return;
  }
  const int64_t source_row = index;
  int64_t packed_row = index;
  if constexpr (HeadMajor) {
    const int64_t tokens_per_batch =
        static_cast<int64_t>(seq_len) * num_heads;
    const int64_t b = index / tokens_per_batch;
    const int64_t within_batch = index - b * tokens_per_batch;
    const int64_t t = within_batch / num_heads;
    const int64_t h = within_batch - t * num_heads;
    packed_row = (b * num_heads + h) * seq_len + t;
  }
  const int64_t base = source_row * symbol_dim;
  uint32_t query_word = 0u;
  uint32_t key_word = 0u;
  for (int bit = 0; bit < symbol_dim; ++bit) {
    if (read_float(query, base + bit) > 0.0f) {
      query_word |= 1u << bit;
    }
    if (read_float(key, base + bit) > 0.0f) {
      key_word |= 1u << bit;
    }
  }
  packed_query[packed_row] = static_cast<int32_t>(query_word);
  packed_key[packed_row] = static_cast<int32_t>(key_word);
}


__device__ __forceinline__ int64_t token_index(
    int b,
    int t,
    int h,
    int seq_len,
    int num_heads) {
  return (static_cast<int64_t>(b) * seq_len + t) * num_heads + h;
}


template <bool HeadMajor>
__device__ __forceinline__ uint32_t load_word(
    const int32_t* __restrict__ packed_symbols,
    int b,
    int t,
    int h,
    int seq_len,
    int num_heads) {
  const int64_t index = HeadMajor
      ? (static_cast<int64_t>(b) * num_heads + h) * seq_len + t
      : token_index(b, t, h, seq_len, num_heads);
  return static_cast<uint32_t>(packed_symbols[index]);
}


__device__ __forceinline__ int find_packed_sequence(
    const int32_t* __restrict__ cu_seqlens,
    int num_sequences,
    int token_pos) {
  int low = 0;
  int high = num_sequences;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (cu_seqlens[middle + 1] <= token_pos) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}


__global__ void validate_varlen_offsets_kernel(
    const int32_t* __restrict__ cu_seqlens,
    int num_sequences,
    int total_tokens) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index == 0) {
    CUDA_KERNEL_ASSERT(
        cu_seqlens[0] == 0 &&
        "cu_seqlens must start at zero");
  }
  if (index < static_cast<int64_t>(num_sequences)) {
    CUDA_KERNEL_ASSERT(
        cu_seqlens[index] <= cu_seqlens[index + 1] &&
        "cu_seqlens must be nondecreasing");
  }
  if (index == static_cast<int64_t>(num_sequences)) {
    CUDA_KERNEL_ASSERT(
        cu_seqlens[num_sequences] == total_tokens &&
        "cu_seqlens must end at the packed token count");
  }
}


template <bool HeadMajor>
__device__ __forceinline__ int exact_suffix_length(
    const int32_t* __restrict__ packed_query_symbols,
    const int32_t* __restrict__ packed_key_symbols,
    int b,
    int h,
    int i,
    int route_position,
    int seq_len,
    int num_heads,
    int max_suffix_length) {
  const int suffix_steps =
      min(max_suffix_length, min(i + 1, route_position));
  for (int suffix_offset_tokens = 0;
       suffix_offset_tokens < suffix_steps;
       ++suffix_offset_tokens) {
    const uint32_t q_word =
        load_word<HeadMajor>(
            packed_query_symbols,
            b,
            i - suffix_offset_tokens,
            h,
            seq_len,
            num_heads);
    const uint32_t k_word =
        load_word<HeadMajor>(
            packed_key_symbols,
            b,
            route_position - 1 - suffix_offset_tokens,
            h,
            seq_len,
            num_heads);
    if (q_word != k_word) {
      return suffix_offset_tokens;
    }
  }
  return suffix_steps;
}


__device__ __forceinline__ void merge_hard_route(
    int other_length,
    int other_route_position,
    int& best_length,
    int& best_route_position) {
  if (other_length > best_length ||
      (other_length > 0 &&
       other_length == best_length &&
       other_route_position > best_route_position)) {
    best_length = other_length;
    best_route_position = other_route_position;
  }
}


template <typename scalar_t, bool PackedVarlen>
__global__ void hard_forward_kernel(
    const int32_t* __restrict__ packed_query_symbols,
    const int32_t* __restrict__ packed_key_symbols,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int max_suffix_length,
    const int32_t* __restrict__ cu_seqlens,
    int num_sequences,
    int total_tokens) {
  extern __shared__ int shared_int[];
  int* lengths = shared_int;
  int* route_positions = lengths + kWarpsPerBlock;

  const int row = blockIdx.x;
  int i;
  int h;
  int b;
  int symbol_h;
  if constexpr (PackedVarlen) {
    const int global_token_pos = row / num_heads;
    h = row % num_heads;
    const int sequence = find_packed_sequence(
        cu_seqlens,
        num_sequences,
        global_token_pos);
    if (sequence >= num_sequences) {
      return;
    }
    const int sequence_start = cu_seqlens[sequence];
    const int sequence_end = cu_seqlens[sequence + 1];
    if (sequence_start < 0 ||
        sequence_end <= sequence_start ||
        sequence_end > total_tokens ||
        global_token_pos < sequence_start ||
        global_token_pos >= sequence_end) {
      return;
    }
    seq_len = sequence_end - sequence_start;
    i = global_token_pos - sequence_start;
    b = 0;
    symbol_h = 0;
    packed_query_symbols +=
        static_cast<int64_t>(h) * total_tokens + sequence_start;
    packed_key_symbols +=
        static_cast<int64_t>(h) * total_tokens + sequence_start;
    value +=
        static_cast<int64_t>(sequence_start) *
        num_value_heads *
        value_dim;
    output +=
        static_cast<int64_t>(sequence_start) *
        num_heads *
        value_dim;
  } else {
    i = row % seq_len;
    h = (row / seq_len) % num_heads;
    b = row / (seq_len * num_heads);
    symbol_h = h;
  }
  const int tid = threadIdx.x;

  int best_length = 0;
  int best_route_position = 0;
  for (int route_position = tid + 1;
       route_position <= i;
       route_position += blockDim.x) {
    const int length = exact_suffix_length<true>(
        packed_query_symbols,
        packed_key_symbols,
        b,
        symbol_h,
        i,
        route_position,
        seq_len,
        num_heads,
        max_suffix_length);
    merge_hard_route(
        length,
        route_position,
        best_length,
        best_route_position);
  }

  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const int other_length =
        __shfl_down_sync(0xffffffffu, best_length, offset);
    const int other_route_position =
        __shfl_down_sync(
            0xffffffffu,
            best_route_position,
            offset);
    if (lane + offset < kWarpSize) {
      merge_hard_route(
          other_length,
          other_route_position,
          best_length,
          best_route_position);
    }
  }
  if (lane == 0) {
    lengths[warp] = best_length;
    route_positions[warp] = best_route_position;
  }
  __syncthreads();

  if (warp == 0 && lane < kWarpsPerBlock) {
    best_length = lengths[lane];
    best_route_position = route_positions[lane];
    constexpr unsigned kWarpLeaderMask =
        (1u << kWarpsPerBlock) - 1u;
    for (int offset = kWarpsPerBlock / 2;
         offset > 0;
         offset >>= 1) {
      const int other_length = __shfl_down_sync(
          kWarpLeaderMask,
          best_length,
          offset);
      const int other_route_position = __shfl_down_sync(
          kWarpLeaderMask,
          best_route_position,
          offset);
      if (lane + offset < kWarpsPerBlock) {
        merge_hard_route(
            other_length,
            other_route_position,
            best_length,
            best_route_position);
      }
    }
    if (lane == 0) {
      lengths[0] = best_length;
      route_positions[0] = best_route_position;
    }
  }
  __syncthreads();

  const int selected_route_position =
      lengths[0] > 0 ? route_positions[0] : 0;
  const int value_head = h / (num_heads / num_value_heads);
  for (int d = tid; d < value_dim; d += blockDim.x) {
    float result = 0.0f;
    if (selected_route_position > 0) {
      const int64_t value_idx =
          ((static_cast<int64_t>(b) * seq_len +
            selected_route_position) *
               num_value_heads +
           value_head) *
              value_dim +
          d;
      result =
          read_float(value, value_idx) > 0.0f ? 1.0f : -1.0f;
    }
    const int64_t output_idx =
        ((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) *
            value_dim +
        d;
    output[output_idx] = write_scalar<scalar_t>(result);
  }
}


__device__ __forceinline__ float local_match_gate(
    uint32_t q_word,
    uint32_t k_word,
    int symbol_dim,
    float mismatch_scale,
    float inverse_symbol_dim) {
  uint32_t mismatch_bits = q_word ^ k_word;
  if (symbol_dim < 32) {
    mismatch_bits &= (1u << symbol_dim) - 1u;
  }
  const float mismatch_rate =
      static_cast<float>(__popc(mismatch_bits)) * inverse_symbol_dim;
  return __expf(-mismatch_scale * mismatch_rate);
}

__device__ __forceinline__ uint32_t hash_dropout_counter(
    uint32_t state) {
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
    int query_pos,
    int route_pos) {
  if (dropout_p == 0.0f) {
    return 1.0f;
  }
  const uint64_t seed = static_cast<uint64_t>(*dropout_seed);
  uint32_t state = hash_dropout_counter(
      static_cast<uint32_t>(route_pos) ^ 0x68e31da4u);
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(query_pos) ^ 0xb5297a4du);
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(head) ^ 0x63d83595u);
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(batch) ^ 0xa511e9b3u);
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(seed));
  state = hash_dropout_counter(
      state ^ static_cast<uint32_t>(seed >> 32) ^ 0x9e3779b9u);
  const float uniform =
      static_cast<float>(state >> 8) * 0x1.0p-24f;
  return uniform >= dropout_p ? inverse_keep_probability : 0.0f;
}


template <bool CacheQuery, bool HeadMajor>
__device__ __forceinline__ float raw_suffix_score(
    const int32_t* __restrict__ packed_query_symbols,
    const int32_t* __restrict__ packed_key_symbols,
    const int32_t* __restrict__ cached_query,
    int b,
    int h,
    int i,
    int route_position,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    float inverse_symbol_dim) {
  const int suffix_steps =
      min(max_suffix_length, min(i + 1, route_position));
  float product = 1.0f;
  float score = 0.0f;
  for (int suffix_offset_tokens = 0;
       suffix_offset_tokens < suffix_steps;
       ++suffix_offset_tokens) {
    const int q_pos = i - suffix_offset_tokens;
    const int k_pos =
        route_position - 1 - suffix_offset_tokens;
    const uint32_t q_word = CacheQuery
        ? static_cast<uint32_t>(
              cached_query[suffix_offset_tokens])
        : load_word<HeadMajor>(
              packed_query_symbols,
              b,
              q_pos,
              h,
              seq_len,
              num_heads);
    const uint32_t k_word =
        load_word<HeadMajor>(
            packed_key_symbols,
            b,
            k_pos,
            h,
            seq_len,
            num_heads);
    product *= local_match_gate(
        q_word,
        k_word,
        symbol_dim,
        mismatch_scale,
        inverse_symbol_dim);
    score += product;
  }
  return score;
}


struct SuffixScoreTransform {
  float route_score;
  float raw_score_vjp_multiplier;
};


__device__ __forceinline__ SuffixScoreTransform transform_suffix_score(
    float raw_score) {
  const float root = sqrtf(1.0f + raw_score);
  return {
      kNormalizedSqrtSuffixScale * (root - 1.0f),
      0.5f * kNormalizedSqrtSuffixScale / root};
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
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  float* shared_max = shared;
  float* shared_normalizer = shared + kWarpsPerBlock;
  float* shared_utility = shared + 2 * kWarpsPerBlock;

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const SoftmaxStats right = {
        __shfl_down_sync(0xffffffffu, stats.max_logit, offset),
        __shfl_down_sync(0xffffffffu, stats.normalizer, offset),
        __shfl_down_sync(
            0xffffffffu,
            stats.utility_numerator,
            offset)};
    if (lane + offset < kWarpSize) {
      stats = merge_softmax_stats(stats, right);
    }
  }

  if (lane == 0) {
    shared_max[warp] = stats.max_logit;
    shared_normalizer[warp] = stats.normalizer;
    shared_utility[warp] = stats.utility_numerator;
  }
  __syncthreads();

  if (warp == 0 && lane < kWarpsPerBlock) {
    stats = {
        shared_max[lane],
        shared_normalizer[lane],
        shared_utility[lane]};
    constexpr unsigned kWarpLeaderMask =
        (1u << kWarpsPerBlock) - 1u;
    for (int offset = kWarpsPerBlock / 2;
         offset > 0;
         offset >>= 1) {
      const SoftmaxStats right = {
          __shfl_down_sync(
              kWarpLeaderMask,
              stats.max_logit,
              offset),
          __shfl_down_sync(
              kWarpLeaderMask,
              stats.normalizer,
              offset),
          __shfl_down_sync(
              kWarpLeaderMask,
              stats.utility_numerator,
              offset)};
      if (lane + offset < kWarpsPerBlock) {
        stats = merge_softmax_stats(stats, right);
      }
    }
    if (lane == 0) {
      shared_max[0] = stats.max_logit;
      shared_normalizer[0] = stats.normalizer;
      shared_utility[0] = stats.utility_numerator;
    }
  }
  __syncthreads();
  return {shared_max[0], shared_normalizer[0], shared_utility[0]};
}


template <
    typename scalar_t,
    bool CacheGradOutput,
    bool ComputeUtility,
    bool AccumulateValueVjp>
__device__ __forceinline__ float scan_route_value(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ cached_grad_output,
    float* __restrict__ grad_value,
    int b,
    int h,
    int i,
    int route_position,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    float route_probability) {
  static_assert(
      ComputeUtility || AccumulateValueVjp,
      "route value scan must produce utility or value VJP");
  const int value_head = h / (num_heads / num_value_heads);
  float utility = 0.0f;
  for (int d = 0; d < value_dim; ++d) {
    const int64_t value_idx =
        ((static_cast<int64_t>(b) * seq_len +
          route_position) *
             num_value_heads +
         value_head) *
            value_dim +
        d;
    const int64_t grad_idx =
        ((static_cast<int64_t>(b) * seq_len + i) * num_heads + h) *
            value_dim +
        d;
    const float grad = CacheGradOutput
        ? cached_grad_output[d]
        : read_float(grad_output, grad_idx);
    if constexpr (ComputeUtility) {
      const float value_logit = read_float(value, value_idx);
      const float signed_value =
          value_logit > 0.0f ? 1.0f : -1.0f;
      utility += grad * signed_value;
    }
    if constexpr (AccumulateValueVjp) {
      atomicAdd(
          &grad_value[value_idx],
          route_probability * grad);
    }
  }
  return utility;
}


template <
    typename scalar_t,
    bool CacheGradOutput,
    bool FullSequenceCache,
    int GroupSize>
__device__ __forceinline__ void cache_route_utility_tile(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ cached_grad_output,
    float* __restrict__ cached_route_utilities,
    int b,
    int h,
    int i,
    int route_tile_start,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  const int warp = threadIdx.x / kWarpSize;
  const int warp_lane = threadIdx.x & (kWarpSize - 1);
  const int lane = warp_lane & (GroupSize - 1);
  constexpr int kRouteSpan =
      FullSequenceCache ? kBlockThreads : kWarpSize;
  constexpr int kGroups = kRouteSpan / GroupSize;
  const int group = FullSequenceCache
      ? threadIdx.x / GroupSize
      : warp_lane / GroupSize;
  const int route_base = FullSequenceCache
      ? 0
      : warp * kWarpSize;
  const int value_head =
      h / (num_heads / num_value_heads);
  const int64_t grad_base =
      ((static_cast<int64_t>(b) * seq_len + i) *
           num_heads +
       h) *
      value_dim;
  for (int local_route_offset = group;
       local_route_offset < kRouteSpan;
       local_route_offset += kGroups) {
    const int route_offset =
        route_base + local_route_offset;
    const int route_position =
        route_tile_start + route_offset;
    float utility = 0.0f;
    if (route_position <= i) {
      const int64_t value_base =
          ((static_cast<int64_t>(b) * seq_len +
            route_position) *
               num_value_heads +
           value_head) *
          value_dim;
      for (int d = lane; d < value_dim; d += GroupSize) {
        const float value_logit =
            read_float(value, value_base + d);
        const float grad = CacheGradOutput
            ? cached_grad_output[d]
            : read_float(grad_output, grad_base + d);
        utility +=
            grad * (value_logit > 0.0f ? 1.0f : -1.0f);
      }
    }
    for (int delta = GroupSize / 2;
         delta > 0;
         delta >>= 1) {
      utility += __shfl_down_sync(
          0xffffffffu,
          utility,
          delta,
          GroupSize);
    }
    if (lane == 0 && route_position <= i) {
      const int cache_index = FullSequenceCache
          ? route_position - 1
          : route_offset;
      cached_route_utilities[cache_index] = utility;
    }
  }
}


template <
    typename scalar_t,
    bool CacheGradOutput,
    bool FullSequenceCache>
__device__ __forceinline__ void cache_route_utility_tile_dispatch(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ cached_grad_output,
    float* __restrict__ cached_route_utilities,
    int b,
    int h,
    int i,
    int route_tile_start,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  if (value_dim >= 64) {
    cache_route_utility_tile<
        scalar_t,
        CacheGradOutput,
        FullSequenceCache,
        16>(
        value,
        grad_output,
        cached_grad_output,
        cached_route_utilities,
        b,
        h,
        i,
        route_tile_start,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim);
  } else {
    cache_route_utility_tile<
        scalar_t,
        CacheGradOutput,
        FullSequenceCache,
        8>(
        value,
        grad_output,
        cached_grad_output,
        cached_route_utilities,
        b,
        h,
        i,
        route_tile_start,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim);
  }
}


template <bool KeyPositionContiguous, bool AggregateKeyVjp>
__device__ __forceinline__ void accumulate_local_match_qk_vjp(
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    int b,
    int h,
    int q_pos,
    int k_pos,
    int seq_len,
    int num_heads,
    int symbol_dim,
    uint32_t q_word,
    uint32_t k_word,
    float local_gate_vjp_scale,
    float* __restrict__ block_grad_key,
    int key_tile_base,
    int key_vjp_span,
    bool aggregate_key_vjp,
    int gradient_mask,
    unsigned active) {
  const int64_t q_base =
      token_index(b, q_pos, h, seq_len, num_heads) * symbol_dim;
  const int64_t k_base =
      token_index(b, k_pos, h, seq_len, num_heads) * symbol_dim;
  const int lane = threadIdx.x & 31;
  const int leader = __ffs(static_cast<int>(active)) - 1;
  for (int bit = 0; bit < symbol_dim; ++bit) {
    const float q_sign = static_cast<float>(sign_from_bit(q_word, bit));
    const float k_sign = static_cast<float>(sign_from_bit(k_word, bit));
    if ((gradient_mask & kGradQuery) != 0) {
      const float q_contribution =
          0.5f * local_gate_vjp_scale * k_sign;
      const float q_warp_sum =
          contiguous_warp_sum(q_contribution, active);
      if (lane == leader) {
        atomicAdd(&grad_query[q_base + bit], q_warp_sum);
      }
    }
    if ((gradient_mask & kGradKey) != 0) {
      const float k_contribution =
          0.5f * local_gate_vjp_scale * q_sign;
      if constexpr (AggregateKeyVjp) {
        if (aggregate_key_vjp) {
          atomicAdd(
              &block_grad_key[
                  bit * key_vjp_span + k_pos - key_tile_base],
              k_contribution);
          continue;
        }
      }
      int64_t grad_key_idx = k_base + bit;
      if constexpr (KeyPositionContiguous) {
        // Route lanes hold adjacent k_pos at a fixed bit.
        grad_key_idx =
            static_cast<int64_t>(bit) * seq_len + k_pos;
      }
      atomicAdd(&grad_key[grad_key_idx], k_contribution);
    }
  }
}


template <
    typename scalar_t,
    BackwardPlan Plan,
    bool PackedVarlen,
    int StaticGradientMask,
    bool AggregateKeyVjp,
    bool CooperativeRouteValues,
    bool CachePackedScores>
__global__ void surrogate_vjp_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query_symbols,
    const int32_t* __restrict__ packed_key_symbols,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int max_suffix_length,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_scale,
    const int64_t* __restrict__ dropout_seed,
    int gradient_mask,
    bool cache_route_utilities,
    bool aggregate_key_vjp,
    bool tile_value_vjp,
    const int32_t* __restrict__ cu_seqlens,
    int num_sequences,
    int total_tokens) {
  constexpr bool kCacheGradOutput =
      BackwardCacheTraits<Plan>::kCacheGradOutput;
  constexpr bool kCacheQuery =
      BackwardCacheTraits<Plan>::kCacheQuery;
  constexpr bool kCacheSuffixScores =
      BackwardCacheTraits<Plan>::kCacheSuffixScores;
  const int active_gradient_mask = StaticGradientMask == 0
      ? gradient_mask
      : StaticGradientMask;
  extern __shared__ float shared_float[];
  const int row = blockIdx.x;
  int i;
  int h;
  int b;
  int symbol_h;
  int dropout_batch;
  if constexpr (PackedVarlen) {
    const int global_token_pos = row / num_heads;
    h = row % num_heads;
    const int sequence = find_packed_sequence(
        cu_seqlens,
        num_sequences,
        global_token_pos);
    if (sequence >= num_sequences) {
      return;
    }
    const int sequence_start = cu_seqlens[sequence];
    const int sequence_end = cu_seqlens[sequence + 1];
    if (sequence_start < 0 ||
        sequence_end <= sequence_start ||
        sequence_end > total_tokens ||
        global_token_pos < sequence_start ||
        global_token_pos >= sequence_end) {
      return;
    }
    seq_len = sequence_end - sequence_start;
    i = global_token_pos - sequence_start;
    b = 0;
    symbol_h = 0;
    dropout_batch = sequence;
    value +=
        static_cast<int64_t>(sequence_start) *
        num_value_heads *
        value_dim;
    grad_output +=
        static_cast<int64_t>(sequence_start) *
        num_heads *
        value_dim;
    packed_query_symbols +=
        static_cast<int64_t>(h) * total_tokens + sequence_start;
    packed_key_symbols +=
        static_cast<int64_t>(h) * total_tokens + sequence_start;
    if ((active_gradient_mask & kGradQuery) != 0) {
      grad_query +=
          static_cast<int64_t>(sequence_start) *
          num_heads *
          symbol_dim;
    }
    if ((active_gradient_mask & kGradKey) != 0) {
      grad_key +=
          static_cast<int64_t>(sequence_start) *
          num_heads *
          symbol_dim;
    }
    if ((active_gradient_mask & kGradValue) != 0) {
      grad_value +=
          static_cast<int64_t>(sequence_start) *
          num_value_heads *
          value_dim;
    }
  } else {
    i = row % seq_len;
    h = (row / seq_len) % num_heads;
    b = row / (seq_len * num_heads);
    symbol_h = h;
    dropout_batch = b;
    if ((active_gradient_mask & kGradKey) != 0) {
      grad_key +=
          (static_cast<int64_t>(b) * num_heads + h) *
          symbol_dim *
          seq_len;
    }
  }
  const BackwardSharedView shared = bind_backward_shared<Plan>(
      shared_float,
      seq_len,
      value_dim,
      max_suffix_length);
  float* cached_grad_output = shared.grad_output;
  int32_t* cached_query = shared.query;
  float* cached_suffix_scores = shared.suffix_scores;
  const BackwardSharedLayout shared_layout =
      make_backward_shared_layout<Plan>(
          seq_len,
          value_dim,
          max_suffix_length);
  const int packed_score_cache_size = CachePackedScores
      ? min(kPackedScoreCacheCapacity, seq_len - 1)
      : 0;
  const int packed_score_cache_start =
      max(1, i - packed_score_cache_size + 1);
  float* cached_route_utilities =
      shared_float +
      shared_layout.total_words +
      packed_score_cache_size;
  const bool use_full_route_utility_cache =
      StaticGradientMask == kGradQuery &&
      cache_route_utilities;
  const int cached_utility_count =
      use_full_route_utility_cache
      ? seq_len - 1
      : (CooperativeRouteValues ? blockDim.x : 0);
  float* block_grad_key =
      cached_route_utilities + cached_utility_count;
  const int key_vjp_span =
      blockDim.x + max_suffix_length - 1;
  const bool use_key_vjp_aggregation =
      AggregateKeyVjp &&
      aggregate_key_vjp &&
      (active_gradient_mask & kGradKey) != 0 &&
      seq_len >= kMinKeyAggregationSequenceLength;
  float* cached_route_probabilities =
      block_grad_key +
      (use_key_vjp_aggregation
           ? symbol_dim * key_vjp_span
           : 0);
  const bool use_value_vjp_tiling =
      (StaticGradientMask == kGradValue ||
       (CooperativeRouteValues &&
        (active_gradient_mask & kGradValue) != 0)) &&
      tile_value_vjp;
  const int tid = threadIdx.x;
  const float inverse_symbol_dim =
      1.0f / static_cast<float>(symbol_dim);
  const float null_logit = kNullScore * scale;
  const float log_nonnull_candidate_count =
      i > 0 ? logf(static_cast<float>(i)) : 0.0f;
  const bool needs_qk_vjp =
      (active_gradient_mask & (kGradQuery | kGradKey)) != 0;

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
      for (int suffix_offset_tokens = tid;
           suffix_offset_tokens < query_count;
           suffix_offset_tokens += blockDim.x) {
        cached_query[suffix_offset_tokens] =
            static_cast<int32_t>(
                load_word<true>(
                    packed_query_symbols,
                    b,
                    i - suffix_offset_tokens,
                    symbol_h,
                    seq_len,
                    num_heads));
      }
    }
    __syncthreads();
  }
  SoftmaxStats local_stats =
      tid == 0
      ? SoftmaxStats{null_logit, 1.0f, 0.0f}
      : SoftmaxStats{-FLT_MAX, 0.0f, 0.0f};
  for (int route_tile_start = 1;
       route_tile_start <= i;
       route_tile_start += blockDim.x) {
    if constexpr (CooperativeRouteValues) {
      cache_route_utility_tile_dispatch<
          scalar_t,
          kCacheGradOutput,
          false>(
          value,
          grad_output,
          cached_grad_output,
          cached_route_utilities,
          b,
          h,
          i,
          route_tile_start,
          seq_len,
          num_heads,
          num_value_heads,
          value_dim);
      __syncwarp();
    } else if (use_full_route_utility_cache) {
      cache_route_utility_tile_dispatch<
          scalar_t,
          kCacheGradOutput,
          true>(
          value,
          grad_output,
          cached_grad_output,
          cached_route_utilities,
          b,
          h,
          i,
          route_tile_start,
          seq_len,
          num_heads,
          num_value_heads,
          value_dim);
      __syncthreads();
    }
    const int route_position =
        route_tile_start + tid;
    if constexpr (!CooperativeRouteValues) {
      if (route_position > i) {
        continue;
      }
    }
    if (route_position <= i) {
      const float raw_score =
          raw_suffix_score<kCacheQuery, true>(
              packed_query_symbols,
              packed_key_symbols,
              cached_query,
              b,
              symbol_h,
              i,
              route_position,
              seq_len,
              num_heads,
              symbol_dim,
              max_suffix_length,
              mismatch_scale,
              inverse_symbol_dim);
      if constexpr (kCacheSuffixScores) {
        cached_suffix_scores[route_position - 1] = raw_score;
      } else if constexpr (CachePackedScores) {
        if (route_position >= packed_score_cache_start) {
          cached_suffix_scores[
              route_position - packed_score_cache_start] = raw_score;
        }
      }
      const float route_score =
          transform_suffix_score(raw_score).route_score;
      float utility = 0.0f;
      if (needs_qk_vjp) {
        if constexpr (CooperativeRouteValues) {
          utility = cached_route_utilities[tid];
        } else if (use_full_route_utility_cache) {
          utility = cached_route_utilities[route_position - 1];
        } else {
          utility = scan_route_value<
              scalar_t,
              kCacheGradOutput,
              true,
              false>(
              value,
              grad_output,
              cached_grad_output,
              grad_value,
              b,
              h,
              i,
              route_position,
              seq_len,
              num_heads,
              num_value_heads,
              value_dim,
              0.0f);
        }
      }
      const float route_dropout_scale =
          attention_dropout_scale(
              dropout_seed,
              dropout_p,
              inverse_keep_probability,
              dropout_batch,
              h,
              i,
              route_position);
      local_stats = append_softmax_item(
          local_stats,
          route_score * scale -
              log_nonnull_candidate_count,
          route_dropout_scale * utility);
    }
    if constexpr (CooperativeRouteValues) {
      __syncwarp();
    }
  }
  const SoftmaxStats row_stats =
      block_reduce_softmax_stats(local_stats, shared.stats);
  const float row_max = row_stats.max_logit;
  const float denominator = row_stats.normalizer;
  const float expected_utility =
      row_stats.utility_numerator / row_stats.normalizer;

  for (int route_tile_start = 1;
       route_tile_start <= i;
       route_tile_start += blockDim.x) {
    const int route_position =
        route_tile_start + tid;
    const int key_tile_base =
        route_tile_start - max_suffix_length;
    if constexpr (CooperativeRouteValues) {
      cache_route_utility_tile_dispatch<
          scalar_t,
          kCacheGradOutput,
          false>(
          value,
          grad_output,
          cached_grad_output,
          cached_route_utilities,
          b,
          h,
          i,
          route_tile_start,
          seq_len,
          num_heads,
          num_value_heads,
          value_dim);
      __syncwarp();
    }
    if (use_key_vjp_aggregation) {
      const int element_count =
          symbol_dim * key_vjp_span;
      for (int index = tid;
           index < element_count;
           index += blockDim.x) {
        block_grad_key[index] = 0.0f;
      }
      __syncthreads();
    }

    if (route_position <= i) {
      float raw_score;
      if constexpr (kCacheSuffixScores) {
        raw_score = cached_suffix_scores[route_position - 1];
      } else if constexpr (CachePackedScores) {
        if (route_position >= packed_score_cache_start) {
          raw_score = cached_suffix_scores[
              route_position - packed_score_cache_start];
        } else {
          raw_score = raw_suffix_score<kCacheQuery, true>(
              packed_query_symbols,
              packed_key_symbols,
              cached_query,
              b,
              symbol_h,
              i,
              route_position,
              seq_len,
              num_heads,
              symbol_dim,
              max_suffix_length,
              mismatch_scale,
              inverse_symbol_dim);
        }
      } else {
        raw_score = raw_suffix_score<kCacheQuery, true>(
                packed_query_symbols,
                packed_key_symbols,
                cached_query,
                b,
                symbol_h,
                i,
                route_position,
                seq_len,
                num_heads,
                symbol_dim,
                max_suffix_length,
                mismatch_scale,
                inverse_symbol_dim);
      }
      const SuffixScoreTransform score_transform =
          transform_suffix_score(raw_score);
      const float probability =
          __expf(
              score_transform.route_score * scale -
              log_nonnull_candidate_count -
              row_max) /
          denominator;
      const float route_dropout_scale =
          attention_dropout_scale(
              dropout_seed,
              dropout_p,
              inverse_keep_probability,
              dropout_batch,
              h,
              i,
              route_position);
      const float dropped_probability =
          probability * route_dropout_scale;
      float utility = 0.0f;
      if constexpr (CooperativeRouteValues) {
        utility = cached_route_utilities[tid];
      } else if (use_full_route_utility_cache) {
        utility = cached_route_utilities[route_position - 1];
      }
      if ((active_gradient_mask & kGradValue) != 0) {
        if (use_value_vjp_tiling) {
          cached_route_probabilities[tid] =
              dropped_probability;
        } else if (
            needs_qk_vjp &&
            !use_full_route_utility_cache &&
            !CooperativeRouteValues) {
          utility = scan_route_value<
              scalar_t,
              kCacheGradOutput,
              true,
              true>(
              value,
              grad_output,
              cached_grad_output,
              grad_value,
              b,
              h,
              i,
              route_position,
              seq_len,
              num_heads,
              num_value_heads,
              value_dim,
              dropped_probability);
        } else {
          scan_route_value<
              scalar_t,
              kCacheGradOutput,
              false,
              true>(
              value,
              grad_output,
              cached_grad_output,
              grad_value,
              b,
              h,
              i,
              route_position,
              seq_len,
              num_heads,
              num_value_heads,
              value_dim,
              dropped_probability);
        }
      } else if (
          !use_full_route_utility_cache &&
          !CooperativeRouteValues) {
        utility = scan_route_value<
            scalar_t,
            kCacheGradOutput,
            true,
            false>(
            value,
            grad_output,
            cached_grad_output,
            grad_value,
            b,
            h,
            i,
            route_position,
            seq_len,
            num_heads,
            num_value_heads,
            value_dim,
            dropped_probability);
      }
      if (needs_qk_vjp) {
        const float route_score_vjp =
            scale * probability *
            (route_dropout_scale * utility - expected_utility);
        const float raw_score_vjp =
            route_score_vjp *
            score_transform.raw_score_vjp_multiplier;
        const int suffix_steps =
            min(max_suffix_length, min(i + 1, route_position));
        float prefix_product = 1.0f;
        float suffix_tail_sum = raw_score;
        for (int suffix_offset_tokens = 0;
             suffix_offset_tokens < suffix_steps;
             ++suffix_offset_tokens) {
          const unsigned active_route_mask =
              route_suffix_warp_mask(
                  route_tile_start,
                  i,
                  suffix_offset_tokens);
          const int q_pos = i - suffix_offset_tokens;
          const int k_pos =
              route_position - 1 - suffix_offset_tokens;
          const uint32_t q_word = kCacheQuery
              ? static_cast<uint32_t>(
                    cached_query[suffix_offset_tokens])
              : load_word<true>(
                    packed_query_symbols,
                    b,
                    q_pos,
                    symbol_h,
                    seq_len,
                    num_heads);
          const uint32_t k_word =
              load_word<true>(
                  packed_key_symbols,
                  b,
                  k_pos,
                  symbol_h,
                  seq_len,
                  num_heads);
          const float gate = local_match_gate(
              q_word,
              k_word,
              symbol_dim,
              mismatch_scale,
              inverse_symbol_dim);
          prefix_product *= gate;
          const float local_gate_vjp_scale =
              (mismatch_scale * inverse_symbol_dim) *
              raw_score_vjp *
              fmaxf(suffix_tail_sum, 0.0f);
          accumulate_local_match_qk_vjp<
              !PackedVarlen,
              AggregateKeyVjp>(
              grad_query,
              grad_key,
              b,
              h,
              q_pos,
              k_pos,
              seq_len,
              num_heads,
              symbol_dim,
              q_word,
              k_word,
              local_gate_vjp_scale,
              block_grad_key,
              key_tile_base,
              key_vjp_span,
              use_key_vjp_aggregation,
              active_gradient_mask,
              active_route_mask);
          suffix_tail_sum -= prefix_product;
        }
      }
    }
    if (use_value_vjp_tiling) {
      __syncthreads();
      const int route_count =
          min(blockDim.x, i - route_tile_start + 1);
      const int element_count =
          route_count * value_dim;
      const int value_head =
          h / (num_heads / num_value_heads);
      const int64_t grad_base =
          ((static_cast<int64_t>(b) * seq_len + i) *
               num_heads +
           h) *
          value_dim;
      for (int index = tid;
           index < element_count;
           index += blockDim.x) {
        const int route_offset = index / value_dim;
        const int d = index - route_offset * value_dim;
        const int route_position =
            route_tile_start + route_offset;
        const int64_t value_idx =
            ((static_cast<int64_t>(b) * seq_len +
              route_position) *
                 num_value_heads +
             value_head) *
                value_dim +
            d;
        const float grad = kCacheGradOutput
            ? cached_grad_output[d]
            : read_float(grad_output, grad_base + d);
        atomicAdd(
            &grad_value[value_idx],
            cached_route_probabilities[route_offset] *
                grad);
      }
      __syncthreads();
    }
    if (use_key_vjp_aggregation) {
      __syncthreads();
      const int element_count =
          symbol_dim * key_vjp_span;
      for (int index = tid;
           index < element_count;
           index += blockDim.x) {
        const int bit = index / key_vjp_span;
        const int local_key_pos =
            index - bit * key_vjp_span;
        const int k_pos =
            key_tile_base + local_key_pos;
        if (k_pos >= 0 && k_pos < i) {
          const float contribution =
              block_grad_key[index];
          if (contribution != 0.0f) {
            int64_t grad_key_idx =
                token_index(
                    b,
                    k_pos,
                    h,
                    seq_len,
                    num_heads) *
                    symbol_dim +
                bit;
            if constexpr (!PackedVarlen) {
              grad_key_idx =
                  static_cast<int64_t>(bit) *
                      seq_len +
                  k_pos;
            }
            atomicAdd(
                &grad_key[grad_key_idx],
                contribution);
          }
        }
      }
      __syncthreads();
    }
    if constexpr (CooperativeRouteValues) {
      if (!use_value_vjp_tiling &&
          !use_key_vjp_aggregation) {
        __syncwarp();
      }
    }
  }
}


template <typename scalar_t, bool DenseKeyAccumulator>
__global__ void finalize_surrogate_vjp_kernel(
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
    grad_query[index] *=
        softsign_derivative(read_float(query, index));
    return;
  }
  const int64_t key_index = index - query_elements;
  if (key_index < key_elements) {
    int64_t accumulator_index = key_index;
    if constexpr (DenseKeyAccumulator) {
      const int bit = key_index % symbol_dim;
      const int64_t token_head_index = key_index / symbol_dim;
      const int h = token_head_index % num_heads;
      const int64_t token_index = token_head_index / num_heads;
      const int t = token_index % seq_len;
      const int64_t b = token_index / seq_len;
      accumulator_index =
          ((b * num_heads + h) * symbol_dim + bit) * seq_len + t;
    }
    grad_key[key_index] =
        grad_key_accumulator[accumulator_index] *
        softsign_derivative(read_float(key, key_index));
    return;
  }
  const int64_t value_index =
      key_index - key_elements;
  if (value_index < value_elements) {
    grad_value[value_index] *=
        softsign_derivative(read_float(value, value_index));
  }
}


template <typename scalar_t, bool HeadMajor>
void launch_pack_pair(
    const torch::Tensor& query,
    const torch::Tensor& key,
    torch::Tensor& packed_query,
    torch::Tensor& packed_key,
    int64_t symbol_rows,
    int seq_len,
    int num_heads,
    int symbol_dim,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks =
      static_cast<int>((symbol_rows + threads - 1) / threads);
  pack_sign_bits_pair_kernel<scalar_t, HeadMajor><<<
      blocks,
      threads,
      0,
      stream>>>(
      query.data_ptr<scalar_t>(),
      key.data_ptr<scalar_t>(),
      packed_query.data_ptr<int32_t>(),
      packed_key.data_ptr<int32_t>(),
      symbol_rows,
      seq_len,
      num_heads,
      symbol_dim);
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hard_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t max_suffix_length) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int symbol_dim = static_cast<int>(query.size(3));
  const int num_value_heads =
      static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  const int64_t symbol_rows =
      static_cast<int64_t>(batch_size) * seq_len * num_heads;

  auto bits_options = query.options().dtype(torch::kInt32);
  auto packed_query_symbols = torch::empty(
      {batch_size, num_heads, seq_len},
      bits_options);
  auto packed_key_symbols = torch::empty(
      {batch_size, num_heads, seq_len},
      bits_options);
  auto output = torch::empty(
      {batch_size, seq_len, num_heads, value_dim},
      query.options());
  const auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_hard_forward",
      [&] {
        launch_pack_pair<scalar_t, true>(
            query,
            key,
            packed_query_symbols,
            packed_key_symbols,
            symbol_rows,
            seq_len,
            num_heads,
            symbol_dim,
            stream);
        const size_t shared_bytes =
            2 * kWarpsPerBlock * sizeof(int);
        hard_forward_kernel<scalar_t, false><<<
            symbol_rows,
            kBlockThreads,
            shared_bytes,
            stream>>>(
            packed_query_symbols.data_ptr<int32_t>(),
            packed_key_symbols.data_ptr<int32_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            seq_len,
            num_heads,
            num_value_heads,
            value_dim,
            static_cast<int>(max_suffix_length),
            nullptr,
            0,
            batch_size * seq_len);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(
      output,
      packed_query_symbols,
      packed_key_symbols);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
hard_forward_varlen_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens,
    int64_t max_suffix_length) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  const int total_tokens = static_cast<int>(query.size(0));
  const int num_heads = static_cast<int>(query.size(1));
  const int symbol_dim = static_cast<int>(query.size(2));
  const int num_value_heads =
      static_cast<int>(value.size(1));
  const int value_dim = static_cast<int>(value.size(2));
  const int num_sequences =
      static_cast<int>(cu_seqlens.numel() - 1);
  const int64_t symbol_rows =
      static_cast<int64_t>(total_tokens) * num_heads;

  auto bits_options = query.options().dtype(torch::kInt32);
  auto packed_query_symbols = torch::empty(
      {num_heads, total_tokens},
      bits_options);
  auto packed_key_symbols = torch::empty(
      {num_heads, total_tokens},
      bits_options);
  auto output = torch::empty(
      {total_tokens, num_heads, value_dim},
      query.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int validation_threads = 256;
  const int validation_blocks =
      (num_sequences + 1 + validation_threads - 1) /
      validation_threads;
  validate_varlen_offsets_kernel<<<
      validation_blocks,
      validation_threads,
      0,
      stream>>>(
      cu_seqlens.data_ptr<int32_t>(),
      num_sequences,
      total_tokens);

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_hard_forward_varlen",
      [&] {
        launch_pack_pair<scalar_t, true>(
            query,
            key,
            packed_query_symbols,
            packed_key_symbols,
            symbol_rows,
            total_tokens,
            num_heads,
            symbol_dim,
            stream);
        const size_t shared_bytes =
            2 * kWarpsPerBlock * sizeof(int);
        hard_forward_kernel<scalar_t, true><<<
            symbol_rows,
            kBlockThreads,
            shared_bytes,
            stream>>>(
            packed_query_symbols.data_ptr<int32_t>(),
            packed_key_symbols.data_ptr<int32_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            0,
            num_heads,
            num_value_heads,
            value_dim,
            static_cast<int>(max_suffix_length),
            cu_seqlens.data_ptr<int32_t>(),
            num_sequences,
            total_tokens);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(
      output,
      packed_query_symbols,
      packed_key_symbols);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> surrogate_vjp_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor packed_query_symbols,
    torch::Tensor packed_key_symbols,
    torch::Tensor dropout_seed,
    int64_t max_suffix_length,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_scale,
    int gradient_mask) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int symbol_dim = static_cast<int>(query.size(3));
  const int num_value_heads =
      static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  const int64_t symbol_rows =
      static_cast<int64_t>(batch_size) * seq_len * num_heads;
  const int64_t* dropout_seed_data = dropout_p > 0.0f
      ? dropout_seed.data_ptr<int64_t>()
      : nullptr;

  auto grad_options = query.options().dtype(torch::kFloat32);
  auto grad_query = (gradient_mask & kGradQuery) != 0
      ? torch::zeros(query.sizes(), grad_options)
      : torch::empty({0}, grad_options);
  auto grad_key_accumulator = (gradient_mask & kGradKey) != 0
      ? torch::zeros(
            {batch_size, num_heads, symbol_dim, seq_len},
            grad_options)
      : torch::empty({0}, grad_options);
  auto grad_key = (gradient_mask & kGradKey) != 0
      ? torch::empty(key.sizes(), grad_options)
      : torch::empty({0}, grad_options);
  auto grad_value = (gradient_mask & kGradValue) != 0
      ? torch::zeros(value.sizes(), grad_options)
      : torch::empty({0}, grad_options);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const BackwardSharedLayout suffix_score_layout =
      make_backward_shared_layout<
          BackwardPlan::CacheSuffixScores>(
          seq_len,
          value_dim,
          static_cast<int>(max_suffix_length));
  const BackwardSharedLayout recompute_layout =
      make_backward_shared_layout<BackwardPlan::RecomputeSuffixScores>(
          seq_len,
          value_dim,
          static_cast<int>(max_suffix_length));
  const bool suffix_score_cache_fits =
      suffix_score_layout.total_bytes() <=
      kPortableSharedMemoryLimit;

  const bool needs_qk_vjp =
      (gradient_mask & (kGradQuery | kGradKey)) != 0;
  const bool needs_value_vjp =
      (gradient_mask & kGradValue) != 0;
  struct DenseBackwardLaunchConfig {
    BackwardPlan plan;
    size_t shared_bytes;
    bool cache_route_utilities;
    bool aggregate_key_vjp;
    bool cooperative_route_values;
    bool tile_value_vjp;
  };
  const auto make_launch_config = [&](
      BackwardPlan plan,
      size_t base_shared_bytes) {
    DenseBackwardLaunchConfig config{
        plan,
        base_shared_bytes,
        false,
        false,
        false,
        false};
    const size_t route_utility_bytes =
        static_cast<size_t>(seq_len - 1) *
        sizeof(float);
    config.cache_route_utilities =
        gradient_mask == kGradQuery &&
        value_dim >= 32 &&
        config.shared_bytes + route_utility_bytes <=
            kPortableSharedMemoryLimit;
    if (config.cache_route_utilities) {
      config.shared_bytes += route_utility_bytes;
    }
    const size_t key_vjp_bytes =
        static_cast<size_t>(
            kBlockThreads + max_suffix_length - 1) *
        symbol_dim *
        sizeof(float);
    config.aggregate_key_vjp =
        (gradient_mask & kGradKey) != 0 &&
        symbol_dim <= 8 &&
        seq_len >=
            ((gradient_mask & kGradQuery) == 0
                 ? kMinDenseKeyAggregationSequenceLength
                 : kMinDenseQueryKeyAggregationSequenceLength) &&
        config.shared_bytes + key_vjp_bytes <=
            kPortableSharedMemoryLimit;
    if (config.aggregate_key_vjp) {
      config.shared_bytes += key_vjp_bytes;
    }
    const size_t cooperative_route_value_bytes =
        static_cast<size_t>(
            1 + (needs_value_vjp ? 1 : 0)) *
        kBlockThreads *
        sizeof(float);
    config.cooperative_route_values =
        needs_qk_vjp &&
        seq_len >= kMinCooperativeSequenceLength &&
        value_dim >= kMinCooperativeValueDimension &&
        !config.cache_route_utilities &&
        config.shared_bytes + cooperative_route_value_bytes <=
            kPortableSharedMemoryLimit;
    if (config.cooperative_route_values) {
      config.shared_bytes += cooperative_route_value_bytes;
    }
    config.tile_value_vjp =
        config.cooperative_route_values && needs_value_vjp;
    if (!config.tile_value_vjp &&
        gradient_mask == kGradValue &&
        value_dim >= 32 &&
        config.shared_bytes + kBlockThreads * sizeof(float) <=
            kPortableSharedMemoryLimit) {
      config.tile_value_vjp = true;
      config.shared_bytes += kBlockThreads * sizeof(float);
    }
    return config;
  };
  const DenseBackwardLaunchConfig recompute_config =
      make_launch_config(
          BackwardPlan::RecomputeSuffixScores,
          recompute_layout.total_bytes());
  const DenseBackwardLaunchConfig cache_config =
      make_launch_config(
          BackwardPlan::CacheSuffixScores,
          suffix_score_layout.total_bytes());
  DenseBackwardLaunchConfig launch_config =
      max_suffix_length > 1 && suffix_score_cache_fits
      ? cache_config
      : recompute_config;
  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_surrogate_vjp",
      [&] {
        const auto resident_blocks = [&](
            auto plan_tag,
            auto gradient_mask_tag,
            auto aggregate_key_vjp_tag,
            auto cooperative_route_values_tag,
            size_t dynamic_shared_bytes) {
          constexpr BackwardPlan kPlan =
              decltype(plan_tag)::value;
          constexpr int kStaticGradientMask =
              decltype(gradient_mask_tag)::value;
          constexpr bool kAggregateKeyVjp =
              decltype(aggregate_key_vjp_tag)::value;
          constexpr bool kCooperativeRouteValues =
              decltype(cooperative_route_values_tag)::value;
          int blocks = 0;
          C10_CUDA_CHECK(
              cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks,
                  surrogate_vjp_kernel<
                      scalar_t,
                      kPlan,
                      false,
                      kStaticGradientMask,
                      kAggregateKeyVjp,
                      kCooperativeRouteValues,
                      false>,
                  kBlockThreads,
                  dynamic_shared_bytes));
          return blocks;
        };
        const auto config_resident_blocks = [&](
            auto plan_tag,
            auto gradient_mask_tag,
            const DenseBackwardLaunchConfig& config) {
          if (config.aggregate_key_vjp) {
            return config.cooperative_route_values
                ? resident_blocks(
                      plan_tag,
                      gradient_mask_tag,
                      std::true_type{},
                      std::true_type{},
                      config.shared_bytes)
                : resident_blocks(
                      plan_tag,
                      gradient_mask_tag,
                      std::true_type{},
                      std::false_type{},
                      config.shared_bytes);
          }
          return config.cooperative_route_values
              ? resident_blocks(
                    plan_tag,
                    gradient_mask_tag,
                    std::false_type{},
                    std::true_type{},
                    config.shared_bytes)
              : resident_blocks(
                    plan_tag,
                    gradient_mask_tag,
                    std::false_type{},
                    std::false_type{},
                    config.shared_bytes);
        };
        const auto select_score_plan = [&](auto gradient_mask_tag) {
          const int cache_blocks = config_resident_blocks(
              std::integral_constant<
                  BackwardPlan,
                  BackwardPlan::CacheSuffixScores>{},
              gradient_mask_tag,
              cache_config);
          const int recompute_blocks = config_resident_blocks(
              std::integral_constant<
                  BackwardPlan,
                  BackwardPlan::RecomputeSuffixScores>{},
              gradient_mask_tag,
              recompute_config);
          TORCH_CHECK(
              recompute_blocks > 0,
              "RosaSoft recompute backward kernel has zero CUDA "
              "occupancy");
          if (cache_blocks <= 0) {
            launch_config = recompute_config;
            return;
          }
          const int multiprocessor_count =
              at::cuda::getCurrentDeviceProperties()
                  ->multiProcessorCount;
          const auto grid_waves = [&](
              int resident_blocks) {
            const int64_t concurrent_blocks =
                static_cast<int64_t>(resident_blocks) *
                multiprocessor_count;
            return (
                symbol_rows + concurrent_blocks - 1) /
                concurrent_blocks;
          };
          const bool recompute_work_is_amortized =
              seq_len >= kMinDenseRecomputeSequenceLength &&
              (gradient_mask != kGradValue ||
               static_cast<int64_t>(seq_len) >=
                   static_cast<int64_t>(
                       kMinValueOnlyRoutesPerSuffixStepForRecompute) *
                       max_suffix_length);
          if (recompute_work_is_amortized &&
              grid_waves(cache_blocks) >
              grid_waves(recompute_blocks)) {
            launch_config = recompute_config;
          }
        };
        if (launch_config.plan ==
            BackwardPlan::CacheSuffixScores) {
          if (gradient_mask == kGradQuery) {
            select_score_plan(
                std::integral_constant<int, kGradQuery>{});
          } else if (gradient_mask == kGradValue) {
            select_score_plan(
                std::integral_constant<int, kGradValue>{});
          } else {
            select_score_plan(
                std::integral_constant<int, 0>{});
          }
        }
        const BackwardPlan plan = launch_config.plan;
        const size_t shared_bytes =
            launch_config.shared_bytes;
        const bool cache_route_utilities =
            launch_config.cache_route_utilities;
        const bool aggregate_key_vjp =
            launch_config.aggregate_key_vjp;
        const bool cooperative_route_values =
            launch_config.cooperative_route_values;
        const bool tile_value_vjp =
            launch_config.tile_value_vjp;
        const auto launch_backward = [&](
            auto plan_tag,
            auto gradient_mask_tag,
            auto aggregate_key_vjp_tag,
            auto cooperative_route_values_tag) {
          constexpr BackwardPlan kPlan =
              decltype(plan_tag)::value;
          constexpr int kStaticGradientMask =
              decltype(gradient_mask_tag)::value;
          constexpr bool kAggregateKeyVjp =
              decltype(aggregate_key_vjp_tag)::value;
          constexpr bool kCooperativeRouteValues =
              decltype(cooperative_route_values_tag)::value;
          surrogate_vjp_kernel<
              scalar_t,
              kPlan,
              false,
              kStaticGradientMask,
              kAggregateKeyVjp,
              kCooperativeRouteValues,
              false><<<
              symbol_rows,
              kBlockThreads,
              shared_bytes,
              stream>>>(
              value.data_ptr<scalar_t>(),
              grad_output.data_ptr<scalar_t>(),
              packed_query_symbols.data_ptr<int32_t>(),
              packed_key_symbols.data_ptr<int32_t>(),
              grad_query.data_ptr<float>(),
              grad_key_accumulator.data_ptr<float>(),
              grad_value.data_ptr<float>(),
              seq_len,
              num_heads,
              symbol_dim,
              num_value_heads,
              value_dim,
              static_cast<int>(max_suffix_length),
              scale,
              dropout_p,
              inverse_keep_probability,
              mismatch_scale,
              dropout_seed_data,
              gradient_mask,
              cache_route_utilities,
              aggregate_key_vjp,
              tile_value_vjp,
              nullptr,
              0,
              batch_size * seq_len);
        };
        const auto launch_selected = [&](
            auto plan_tag,
            auto gradient_mask_tag) {
          if (aggregate_key_vjp) {
            if (cooperative_route_values) {
              launch_backward(
                  plan_tag,
                  gradient_mask_tag,
                  std::true_type{},
                  std::true_type{});
            } else {
              launch_backward(
                  plan_tag,
                  gradient_mask_tag,
                  std::true_type{},
                  std::false_type{});
            }
          } else if (cooperative_route_values) {
            launch_backward(
                plan_tag,
                gradient_mask_tag,
                std::false_type{},
                std::true_type{});
          } else {
            launch_backward(
                plan_tag,
                gradient_mask_tag,
                std::false_type{},
                std::false_type{});
          }
        };
        switch (plan) {
          case BackwardPlan::CacheSuffixScores: {
            const auto plan_tag = std::integral_constant<
                BackwardPlan,
                BackwardPlan::CacheSuffixScores>{};
            if (gradient_mask == kGradQuery) {
              launch_selected(
                  plan_tag,
                  std::integral_constant<int, kGradQuery>{});
            } else if (gradient_mask == kGradValue) {
              launch_backward(
                  plan_tag,
                  std::integral_constant<int, kGradValue>{},
                  std::false_type{},
                  std::false_type{});
            } else {
              launch_selected(
                  plan_tag,
                  std::integral_constant<int, 0>{});
            }
            break;
          }
          case BackwardPlan::RecomputeSuffixScores:
            if (gradient_mask == kGradQuery) {
              launch_selected(
                  std::integral_constant<
                      BackwardPlan,
                      BackwardPlan::RecomputeSuffixScores>{},
                  std::integral_constant<int, kGradQuery>{});
            } else if (gradient_mask == kGradValue) {
              launch_backward(
                  std::integral_constant<
                      BackwardPlan,
                      BackwardPlan::RecomputeSuffixScores>{},
                  std::integral_constant<int, kGradValue>{},
                  std::false_type{},
                  std::false_type{});
            } else {
              launch_selected(
                  std::integral_constant<
                      BackwardPlan,
                      BackwardPlan::RecomputeSuffixScores>{},
                  std::integral_constant<int, 0>{});
            }
            break;
        }
        const int64_t query_elements =
            (gradient_mask & kGradQuery) != 0
            ? query.numel()
            : 0;
        const int64_t key_elements =
            (gradient_mask & kGradKey) != 0
            ? key.numel()
            : 0;
        const int64_t value_elements =
            (gradient_mask & kGradValue) != 0
            ? value.numel()
            : 0;
        const int64_t finalize_elements =
            query_elements + key_elements + value_elements;
        constexpr int kFinalizeThreads = 256;
        const int finalize_blocks = static_cast<int>(
            (finalize_elements + kFinalizeThreads - 1) /
            kFinalizeThreads);
        finalize_surrogate_vjp_kernel<scalar_t, true><<<
            finalize_blocks,
            kFinalizeThreads,
            0,
            stream>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            grad_query.data_ptr<float>(),
            grad_key_accumulator.data_ptr<float>(),
            grad_key.data_ptr<float>(),
            grad_value.data_ptr<float>(),
            query_elements,
            key_elements,
            value_elements,
            seq_len,
            num_heads,
            symbol_dim);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(
      grad_query,
      grad_key,
      grad_value);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
surrogate_vjp_varlen_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens,
    torch::Tensor grad_output,
    torch::Tensor packed_query_symbols,
    torch::Tensor packed_key_symbols,
    torch::Tensor dropout_seed,
    int64_t max_suffix_length,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_scale,
    int gradient_mask) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  const int total_tokens = static_cast<int>(query.size(0));
  const int num_heads = static_cast<int>(query.size(1));
  const int symbol_dim = static_cast<int>(query.size(2));
  const int num_value_heads =
      static_cast<int>(value.size(1));
  const int value_dim = static_cast<int>(value.size(2));
  const int num_sequences =
      static_cast<int>(cu_seqlens.numel() - 1);
  const int64_t symbol_rows =
      static_cast<int64_t>(total_tokens) * num_heads;
  const int64_t* dropout_seed_data = dropout_p > 0.0f
      ? dropout_seed.data_ptr<int64_t>()
      : nullptr;

  auto grad_options = query.options().dtype(torch::kFloat32);
  auto grad_query = (gradient_mask & kGradQuery) != 0
      ? torch::zeros(query.sizes(), grad_options)
      : torch::empty({0}, grad_options);
  auto grad_key = (gradient_mask & kGradKey) != 0
      ? torch::zeros(key.sizes(), grad_options)
      : torch::empty({0}, grad_options);
  auto grad_value = (gradient_mask & kGradValue) != 0
      ? torch::zeros(value.sizes(), grad_options)
      : torch::empty({0}, grad_options);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const BackwardSharedLayout recompute_layout =
      make_backward_shared_layout<BackwardPlan::RecomputeSuffixScores>(
          0,
          value_dim,
          static_cast<int>(max_suffix_length));
  size_t shared_bytes = recompute_layout.total_bytes();
  const bool cache_packed_scores =
      gradient_mask == kGradQuery &&
      static_cast<int64_t>(total_tokens) >=
          static_cast<int64_t>(num_sequences) *
              kMinPackedScoreCacheSequenceLength;
  const size_t packed_score_cache_bytes =
      cache_packed_scores
      ? static_cast<size_t>(
            std::min(
                kPackedScoreCacheCapacity,
                total_tokens - 1)) *
          sizeof(float)
      : 0;
  shared_bytes += packed_score_cache_bytes;
  const size_t key_vjp_bytes =
      static_cast<size_t>(
          kBlockThreads + max_suffix_length - 1) *
      symbol_dim *
      sizeof(float);
  const bool long_average_sequence =
      static_cast<int64_t>(total_tokens) >=
      static_cast<int64_t>(num_sequences) *
          kMinKeyAggregationSequenceLength;
  const bool aggregate_key_vjp =
      (gradient_mask & kGradKey) != 0 &&
      symbol_dim <= 8 &&
      long_average_sequence &&
      shared_bytes + key_vjp_bytes <=
          kPortableSharedMemoryLimit;
  if (aggregate_key_vjp) {
    shared_bytes += key_vjp_bytes;
  }
  const bool needs_qk_vjp =
      (gradient_mask & (kGradQuery | kGradKey)) != 0;
  const bool needs_value_vjp =
      (gradient_mask & kGradValue) != 0;
  const size_t cooperative_route_value_bytes =
      static_cast<size_t>(
          1 + (needs_value_vjp ? 1 : 0)) *
      kBlockThreads *
      sizeof(float);
  const bool cooperative_route_values =
      needs_qk_vjp &&
      long_average_sequence &&
      value_dim >= kMinCooperativeValueDimension &&
      (gradient_mask != (kGradQuery | kGradKey) ||
       value_dim >= kMinPackedQkCooperativeValueDimension) &&
      shared_bytes + cooperative_route_value_bytes <=
          kPortableSharedMemoryLimit;
  if (cooperative_route_values) {
    shared_bytes += cooperative_route_value_bytes;
  }
  bool tile_value_vjp =
      cooperative_route_values && needs_value_vjp;
  if (!tile_value_vjp &&
      gradient_mask == kGradValue &&
      value_dim >= 32 &&
      shared_bytes + kBlockThreads * sizeof(float) <=
          kPortableSharedMemoryLimit) {
    tile_value_vjp = true;
    shared_bytes += kBlockThreads * sizeof(float);
  }

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_surrogate_vjp_varlen",
      [&] {
        const auto launch_backward = [&](
            auto gradient_mask_tag,
            auto aggregate_key_vjp_tag,
            auto cooperative_route_values_tag,
            auto cache_packed_scores_tag) {
          constexpr int kStaticGradientMask =
              decltype(gradient_mask_tag)::value;
          constexpr bool kAggregateKeyVjp =
              decltype(aggregate_key_vjp_tag)::value;
          constexpr bool kCooperativeRouteValues =
              decltype(cooperative_route_values_tag)::value;
          constexpr bool kCachePackedScores =
              decltype(cache_packed_scores_tag)::value;
          surrogate_vjp_kernel<
              scalar_t,
              BackwardPlan::RecomputeSuffixScores,
              true,
              kStaticGradientMask,
              kAggregateKeyVjp,
              kCooperativeRouteValues,
              kCachePackedScores><<<
              symbol_rows,
              kBlockThreads,
              shared_bytes,
              stream>>>(
              value.data_ptr<scalar_t>(),
              grad_output.data_ptr<scalar_t>(),
              packed_query_symbols.data_ptr<int32_t>(),
              packed_key_symbols.data_ptr<int32_t>(),
              grad_query.data_ptr<float>(),
              grad_key.data_ptr<float>(),
              grad_value.data_ptr<float>(),
              0,
              num_heads,
              symbol_dim,
              num_value_heads,
              value_dim,
              static_cast<int>(max_suffix_length),
              scale,
              dropout_p,
              inverse_keep_probability,
              mismatch_scale,
              dropout_seed_data,
              gradient_mask,
              false,
              aggregate_key_vjp,
              tile_value_vjp,
              cu_seqlens.data_ptr<int32_t>(),
              num_sequences,
              total_tokens);
        };
        const auto launch_selected = [&](
            auto gradient_mask_tag,
            auto aggregate_key_vjp_tag,
            auto cache_packed_scores_tag) {
          if (cooperative_route_values) {
            launch_backward(
                gradient_mask_tag,
                aggregate_key_vjp_tag,
                std::true_type{},
                cache_packed_scores_tag);
          } else {
            launch_backward(
                gradient_mask_tag,
                aggregate_key_vjp_tag,
                std::false_type{},
                cache_packed_scores_tag);
          }
        };
        if (gradient_mask == kGradQuery) {
          if (cache_packed_scores) {
            launch_selected(
                std::integral_constant<int, kGradQuery>{},
                std::false_type{},
                std::true_type{});
          } else {
            launch_selected(
                std::integral_constant<int, kGradQuery>{},
                std::false_type{},
                std::false_type{});
          }
        } else if (gradient_mask == kGradValue) {
          launch_backward(
              std::integral_constant<int, kGradValue>{},
              std::false_type{},
              std::false_type{},
              std::false_type{});
        } else if (aggregate_key_vjp) {
          launch_selected(
              std::integral_constant<int, 0>{},
              std::true_type{},
              std::false_type{});
        } else {
          launch_selected(
              std::integral_constant<int, 0>{},
              std::false_type{},
              std::false_type{});
        }
        const int64_t query_elements =
            (gradient_mask & kGradQuery) != 0
            ? query.numel()
            : 0;
        const int64_t key_elements =
            (gradient_mask & kGradKey) != 0
            ? key.numel()
            : 0;
        const int64_t value_elements =
            (gradient_mask & kGradValue) != 0
            ? value.numel()
            : 0;
        const int64_t finalize_elements =
            query_elements + key_elements + value_elements;
        constexpr int kFinalizeThreads = 256;
        const int finalize_blocks = static_cast<int>(
            (finalize_elements + kFinalizeThreads - 1) /
            kFinalizeThreads);
        finalize_surrogate_vjp_kernel<scalar_t, false><<<
            finalize_blocks,
            kFinalizeThreads,
            0,
            stream>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            grad_query.data_ptr<float>(),
            grad_key.data_ptr<float>(),
            grad_key.data_ptr<float>(),
            grad_value.data_ptr<float>(),
            query_elements,
            key_elements,
            value_elements,
            0,
            num_heads,
            symbol_dim);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(
      grad_query,
      grad_key,
      grad_value);
}
