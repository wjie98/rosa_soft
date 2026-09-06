#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <tuple>

#include "../../rosa_soft/csrc/cuda/rosa_soft_vjp_common.cuh"


namespace {

namespace cg = cooperative_groups;

constexpr int kWarpSize = 32;
constexpr int kMicroTile = 32;
constexpr int kScoreStride = kMicroTile + 1;


__device__ __forceinline__ uint32_t symbol_mask(int symbol_dim) {
  return symbol_dim == 32
      ? 0xffffffffu
      : (1u << symbol_dim) - 1u;
}


struct PackedDiagonalLane {
  int delta;
  int offset;
  int segment_start;
  int nominal_length;
};


__device__ __forceinline__ PackedDiagonalLane packed_diagonal_lane(
    int pair,
    int lane) {
  if (pair < 30) {
    const int short_length = pair / 2 + 1;
    const int sign = (pair & 1) == 0 ? 1 : -1;
    if (lane < short_length) {
      return {
          sign * (kMicroTile - short_length),
          lane,
          0,
          short_length};
    }
    return {
        sign * short_length,
        lane - short_length,
        short_length,
        kMicroTile - short_length};
  }
  if (pair == 30) {
    if (lane < kMicroTile / 2) {
      return {kMicroTile / 2, lane, 0, kMicroTile / 2};
    }
    return {
        -kMicroTile / 2,
        lane - kMicroTile / 2,
        kMicroTile / 2,
        kMicroTile / 2};
  }
  return {0, lane, 0, kMicroTile};
}


__device__ __forceinline__ void warp_segmented_affine_scan(
    float& coefficient,
    float& bias,
    int segment_start) {
  const int lane = threadIdx.x & (kWarpSize - 1);
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const float left_coefficient = __shfl_up_sync(
        0xffffffffu, coefficient, offset);
    const float left_bias = __shfl_up_sync(
        0xffffffffu, bias, offset);
    if (lane - offset >= segment_start) {
      bias = fmaf(coefficient, left_bias, bias);
      coefficient *= left_coefficient;
    }
  }
}


__host__ __device__ __forceinline__ int triangular_slot(int row, int col) {
  return row * (row + 1) / 2 + col;
}


__device__ __forceinline__ int triangular_row(int slot) {
  int row = __float2int_rd(
      0.5f * (sqrtf(8.0f * static_cast<float>(slot) + 1.0f) - 1.0f));
  while (triangular_slot(row + 1, 0) <= slot) {
    ++row;
  }
  while (triangular_slot(row, 0) > slot) {
    --row;
  }
  return row;
}


__host__ __device__ __forceinline__ int first_column_on_wave(
    int wave,
    int tile_count) {
  return max(0, wave - (tile_count - 1));
}


__host__ __device__ __forceinline__ int last_causal_column_on_wave(
    int wave,
    int tile_count) {
  return min(min(tile_count - 1, wave), wave / 2);
}


template <int TileSize>
__device__ __forceinline__ size_t score_shared_bytes() {
  // Q, K, and the padded 2L diagonal carry vector.
  return static_cast<size_t>(4 * TileSize) * sizeof(uint32_t);
}


template <int TileSize>
__host__ __device__ __forceinline__ size_t stats_shared_bytes() {
  constexpr int kMicroTiles = TileSize / kMicroTile;
  // Q/K words, diagonal carries, three row statistics, and one padded score
  // microtile per active warp.
  return static_cast<size_t>(
      2 * TileSize * sizeof(uint32_t) +
      5 * TileSize * sizeof(float) +
      kMicroTiles * kMicroTile * kScoreStride * sizeof(float));
}


template <int TileSize>
__host__ __device__ __forceinline__ size_t reverse_shared_bytes() {
  constexpr int kMicroTiles = TileSize / kMicroTile;
  constexpr int kMicroEdgeStride = 2 * kMicroTile;
  return static_cast<size_t>(
      2 * TileSize * sizeof(uint32_t) +
      4 * TileSize * sizeof(float) +
      kMicroTiles * kMicroTiles * kMicroEdgeStride * sizeof(float) +
      kMicroTiles * kMicroTile * kScoreStride * sizeof(float));
}


template <int TileSize>
__device__ __forceinline__ void process_score_macro_tile(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ dense_scores,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int series,
    int tile_row,
    int tile_col,
    int seq_len,
    int symbol_dim,
    int macro_tile_count,
    float mismatch_unit,
    unsigned char* __restrict__ shared_storage) {
  constexpr int kMicroTiles = TileSize / kMicroTile;
  constexpr int kThreads = kMicroTiles * kWarpSize;

  uint32_t* shared_query =
      reinterpret_cast<uint32_t*>(shared_storage);
  uint32_t* shared_key = shared_query + TileSize;
  float* shared_carry = reinterpret_cast<float*>(
      shared_key + TileSize);

  const int query_start = tile_row * TileSize;
  const int key_start = tile_col * TileSize;
  const int query_count = min(TileSize, seq_len - query_start);
  const int key_count = min(TileSize, seq_len - key_start);
  const int center_delta = (tile_row - tile_col) * TileSize;
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;

  for (int index = threadIdx.x; index < TileSize; index += kThreads) {
    shared_query[index] = index < query_count
        ? static_cast<uint32_t>(packed_query[series_offset + query_start + index])
        : 0u;
    shared_key[index] = index < key_count
        ? static_cast<uint32_t>(packed_key[series_offset + key_start + index])
        : 0u;
  }
  for (int index = threadIdx.x; index < 2 * TileSize; index += kThreads) {
    const int local_delta = index - (TileSize - 1);
    const int delta = center_delta + local_delta;
    shared_carry[index] = delta > 0 && delta < seq_len
        ? diagonal_carry[series_offset + delta]
        : 0.0f;
  }
  __syncthreads();

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const uint32_t mask = symbol_mask(symbol_dim);

  for (int micro_wave = 0;
       micro_wave < 2 * kMicroTiles - 1;
       ++micro_wave) {
    const int first_micro_col = max(0, micro_wave - (kMicroTiles - 1));
    const int final_micro_col = min(kMicroTiles - 1, micro_wave);
    const int micro_col = first_micro_col + warp;
    if (micro_col <= final_micro_col) {
      const int micro_row = micro_wave - micro_col;
      const int query_micro_start = micro_row * kMicroTile;
      const int key_micro_start = micro_col * kMicroTile;
      const int micro_query_count = min(
          kMicroTile, max(0, query_count - query_micro_start));
      const int micro_key_count = min(
          kMicroTile, max(0, key_count - key_micro_start));

      for (int pair = 0; pair < kMicroTile; ++pair) {
        const PackedDiagonalLane diagonal =
            packed_diagonal_lane(pair, lane);
        const int micro_delta = diagonal.delta;
        const int local_delta =
            query_micro_start - key_micro_start + micro_delta;
        const int delta = center_delta + local_delta;
        const int local_query_begin = max(0, micro_delta);
        const int local_query_end = min(
            micro_query_count - 1,
            micro_delta + micro_key_count - 1);
        const int segment_length =
            local_query_end >= local_query_begin
            ? local_query_end - local_query_begin + 1
            : 0;
        const bool active =
            diagonal.offset < segment_length && delta > 0;
        const int local_query = local_query_begin + diagonal.offset;
        const int local_key = local_query - micro_delta;
        const int macro_query = query_micro_start + local_query;
        const int macro_key = key_micro_start + local_key;
        const int mismatch = active
            ? __popc(
                  (shared_query[macro_query] ^ shared_key[macro_key]) & mask)
            : 0;
        const float gate = active
            ? __expf(-mismatch_unit * static_cast<float>(mismatch))
            : 1.0f;
        float coefficient = gate;
        float bias = active ? gate : 0.0f;
        warp_segmented_affine_scan(
            coefficient, bias, diagonal.segment_start);
        const int carry_index = local_delta + TileSize - 1;
        float incoming = 0.0f;
        if (diagonal.offset == 0 && delta > 0 && segment_length > 0) {
          incoming = shared_carry[carry_index];
        }
        incoming = __shfl_sync(
            0xffffffffu, incoming, diagonal.segment_start);
        __syncwarp();
        const float score = fmaf(coefficient, incoming, bias);
        if (active) {
          const int query_position = query_start + macro_query;
          const int key_position = key_start + macro_key;
          const int64_t output =
              (static_cast<int64_t>(series) * seq_len + query_position) *
                  seq_len +
              key_position + 1;
          dense_scores[output] = score;
        }
        if (diagonal.offset == segment_length - 1 && delta > 0) {
          shared_carry[carry_index] = score;
        }
      }
    }
    __syncthreads();
  }

  const int checkpoint_slot = triangular_slot(tile_row, tile_col);
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;
  const int64_t checkpoint_base =
      (static_cast<int64_t>(series) * macro_tile_slots + checkpoint_slot) *
      (2 * TileSize);
  for (int index = threadIdx.x; index < 2 * TileSize; index += kThreads) {
    const int local_delta = index - (TileSize - 1);
    const int delta = center_delta + local_delta;
    const int local_query_begin = max(0, local_delta);
    const int local_query_end = min(
        query_count - 1,
        local_delta + key_count - 1);
    const bool has_segment = delta > 0 && delta < seq_len &&
        local_query_end >= local_query_begin;
    const float outgoing = has_segment ? shared_carry[index] : 0.0f;
    checkpoints[checkpoint_base + index] = outgoing;
    if (has_segment) {
      diagonal_carry[series_offset + delta] = outgoing;
    }
  }
  __syncthreads();
}


template <typename scalar_t>
__device__ __forceinline__ float route_utility(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    int batch,
    int head,
    int value_head,
    int query_position,
    int route_position,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  const int64_t grad_base =
      ((static_cast<int64_t>(batch) * seq_len + query_position) *
           num_heads +
       head) * value_dim;
  const int64_t value_base =
      ((static_cast<int64_t>(batch) * seq_len + route_position) *
           num_value_heads +
       value_head) * value_dim;
  float utility = 0.0f;
  for (int feature = 0; feature < value_dim; ++feature) {
    const float value_sign = rosa_soft::cuda::read_float(
        value, value_base + feature) > 0.0f
        ? 1.0f
        : -1.0f;
    utility = fmaf(
        rosa_soft::cuda::read_float(
            grad_output, grad_base + feature),
        value_sign,
        utility);
  }
  return utility;
}


template <int GroupSize, typename scalar_t>
__device__ __forceinline__ float grouped_route_utility(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    int batch,
    int head,
    int value_head,
    int query_start,
    int route_start,
    int candidate_count,
    int query_stride,
    int route_stride,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  constexpr int kGroups = kWarpSize / GroupSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int group = lane / GroupSize;
  const int group_lane = lane & (GroupSize - 1);
  float lane_utility = 0.0f;
  for (int candidate_start = 0;
       candidate_start < candidate_count;
       candidate_start += kGroups) {
    const int candidate = candidate_start + group;
    const int query_position = query_start + candidate * query_stride;
    const int route_position = route_start + candidate * route_stride;
    const bool active = candidate < candidate_count &&
        query_position >= 0 && query_position < seq_len &&
        route_position >= 1 && route_position <= query_position;
    float utility = 0.0f;
    if (active) {
      const int64_t grad_base =
          ((static_cast<int64_t>(batch) * seq_len + query_position) *
               num_heads +
           head) *
          value_dim;
      const int64_t value_base =
          ((static_cast<int64_t>(batch) * seq_len + route_position) *
               num_value_heads +
           value_head) *
          value_dim;
      for (int feature = group_lane;
           feature < value_dim;
           feature += GroupSize) {
        const float value_sign = rosa_soft::cuda::read_float(
            value, value_base + feature) > 0.0f
            ? 1.0f
            : -1.0f;
        utility = fmaf(
            rosa_soft::cuda::read_float(
                grad_output, grad_base + feature),
            value_sign,
            utility);
      }
    }
#pragma unroll
    for (int offset = GroupSize / 2; offset > 0; offset >>= 1) {
      utility += __shfl_down_sync(
          0xffffffffu, utility, offset, GroupSize);
    }
#pragma unroll
    for (int source_group = 0;
         source_group < kGroups;
         ++source_group) {
      const int target_candidate = candidate_start + source_group;
      const float candidate_utility = __shfl_sync(
          0xffffffffu, utility, source_group * GroupSize);
      if (lane == target_candidate && target_candidate < candidate_count) {
        lane_utility = candidate_utility;
      }
    }
  }
  return lane_utility;
}


template <typename scalar_t>
__device__ __forceinline__ float grouped_route_utility_dispatch(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    int batch,
    int head,
    int value_head,
    int query_start,
    int route_start,
    int candidate_count,
    int query_stride,
    int route_stride,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  if (value_dim >= 64) {
    return grouped_route_utility<16>(
        value,
        grad_output,
        batch,
        head,
        value_head,
        query_start,
        route_start,
        candidate_count,
        query_stride,
        route_stride,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim);
  }
  return grouped_route_utility<8>(
      value,
      grad_output,
      batch,
      head,
      value_head,
      query_start,
      route_start,
      candidate_count,
      query_stride,
      route_stride,
      seq_len,
      num_heads,
      num_value_heads,
      value_dim);
}


template <int TileSize>
__device__ __forceinline__ float macro_incoming_score(
    const float* __restrict__ checkpoints,
    int series,
    int tile_row,
    int tile_col,
    int local_delta,
    int query_count,
    int key_count,
    int seq_len,
    int macro_tile_count);


template <int TileSize, typename scalar_t, bool ReplayFromCheckpoints = false>
__device__ __forceinline__ void process_stats_macro_tile(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    float* __restrict__ row_stats,
    float* __restrict__ partial_stats,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int series,
    int tile_row,
    int tile_col,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    unsigned char* __restrict__ shared_storage) {
  constexpr int kMicroTiles = TileSize / kMicroTile;
  constexpr int kThreads = kMicroTiles * kWarpSize;

  uint32_t* shared_query =
      reinterpret_cast<uint32_t*>(shared_storage);
  uint32_t* shared_key = shared_query + TileSize;
  float* shared_carry = reinterpret_cast<float*>(shared_key + TileSize);
  float* shared_maximum = shared_carry + 2 * TileSize;
  float* shared_normalizer = shared_maximum + TileSize;
  float* shared_utility = shared_normalizer + TileSize;
  float* shared_score_tiles = shared_utility + TileSize;

  const int query_start = tile_row * TileSize;
  const int key_start = tile_col * TileSize;
  const int query_count = min(TileSize, seq_len - query_start);
  const int key_count = min(TileSize, seq_len - key_start);
  const int center_delta = (tile_row - tile_col) * TileSize;
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int value_head = head / (num_heads / num_value_heads);

  for (int index = threadIdx.x; index < TileSize; index += kThreads) {
    shared_query[index] = index < query_count
        ? static_cast<uint32_t>(packed_query[series_offset + query_start + index])
        : 0u;
    shared_key[index] = index < key_count
        ? static_cast<uint32_t>(packed_key[series_offset + key_start + index])
        : 0u;
    if (index < query_count) {
      if (partial_stats != nullptr) {
        shared_maximum[index] = -FLT_MAX;
        shared_normalizer[index] = 0.0f;
        shared_utility[index] = 0.0f;
      } else {
        const int64_t stats_index =
            (series_offset + query_start + index) * 3;
        shared_maximum[index] = row_stats[stats_index];
        shared_normalizer[index] = row_stats[stats_index + 1];
        shared_utility[index] = row_stats[stats_index + 2];
      }
    }
  }
  for (int index = threadIdx.x; index < 2 * TileSize; index += kThreads) {
    const int local_delta = index - (TileSize - 1);
    const int delta = center_delta + local_delta;
    if constexpr (ReplayFromCheckpoints) {
      shared_carry[index] = macro_incoming_score<TileSize>(
          checkpoints,
          series,
          tile_row,
          tile_col,
          local_delta,
          query_count,
          key_count,
          seq_len,
          macro_tile_count);
    } else {
      shared_carry[index] = delta > 0 && delta < seq_len
          ? diagonal_carry[series_offset + delta]
          : 0.0f;
    }
  }
  __syncthreads();

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const uint32_t mask = symbol_mask(symbol_dim);
  float* warp_scores =
      shared_score_tiles + warp * kMicroTile * kScoreStride;

  for (int micro_wave = 0;
       micro_wave < 2 * kMicroTiles - 1;
       ++micro_wave) {
    const int first_micro_col = max(0, micro_wave - (kMicroTiles - 1));
    const int final_micro_col = min(kMicroTiles - 1, micro_wave);
    const int micro_col = first_micro_col + warp;
    if (micro_col <= final_micro_col) {
      const int micro_row = micro_wave - micro_col;
      const int query_micro_start = micro_row * kMicroTile;
      const int key_micro_start = micro_col * kMicroTile;
      const int micro_query_count = min(
          kMicroTile, max(0, query_count - query_micro_start));
      const int micro_key_count = min(
          kMicroTile, max(0, key_count - key_micro_start));

      for (int pair = 0; pair < kMicroTile; ++pair) {
        const PackedDiagonalLane diagonal =
            packed_diagonal_lane(pair, lane);
        const int micro_delta = diagonal.delta;
        const int local_delta =
            query_micro_start - key_micro_start + micro_delta;
        const int delta = center_delta + local_delta;
        const int local_query_begin = max(0, micro_delta);
        const int local_query_end = min(
            micro_query_count - 1,
            micro_delta + micro_key_count - 1);
        const int segment_length =
            local_query_end >= local_query_begin
            ? local_query_end - local_query_begin + 1
            : 0;
        const bool active =
            diagonal.offset < segment_length && delta > 0;
        const int local_query = local_query_begin + diagonal.offset;
        const int local_key = local_query - micro_delta;
        const int macro_query = query_micro_start + local_query;
        const int macro_key = key_micro_start + local_key;
        const int mismatch = active
            ? __popc(
                  (shared_query[macro_query] ^ shared_key[macro_key]) & mask)
            : 0;
        const float gate = active
            ? __expf(-mismatch_unit * static_cast<float>(mismatch))
            : 1.0f;
        float coefficient = gate;
        float bias = active ? gate : 0.0f;
        warp_segmented_affine_scan(
            coefficient, bias, diagonal.segment_start);
        const int carry_index = local_delta + TileSize - 1;
        float incoming = 0.0f;
        if (diagonal.offset == 0 && delta > 0 && segment_length > 0) {
          incoming = shared_carry[carry_index];
        }
        incoming = __shfl_sync(
            0xffffffffu, incoming, diagonal.segment_start);
        __syncwarp();
        const float score = fmaf(coefficient, incoming, bias);
        if (active) {
          warp_scores[local_query * kScoreStride + local_key] = score;
        }
        if (diagonal.offset == segment_length - 1 && delta > 0) {
          shared_carry[carry_index] = score;
        }
      }
      __syncwarp();

      for (int local_query = 0;
           local_query < micro_query_count;
           ++local_query) {
        const int query_position =
            query_start + query_micro_start + local_query;
        const bool use_grouped_utility =
            partial_stats != nullptr && value_dim >= 32;
        const float grouped_utility = use_grouped_utility
            ? grouped_route_utility_dispatch(
                  value,
                  grad_output,
                  batch,
                  head,
                  value_head,
                  query_position,
                  key_start + key_micro_start + 1,
                  micro_key_count,
                  0,
                  1,
                  seq_len,
                  num_heads,
                  num_value_heads,
                  value_dim)
            : 0.0f;
        const int local_key = lane;
        const int key_position = key_start + key_micro_start + local_key;
        const bool active = local_key < micro_key_count &&
            key_position < query_position;
        rosa_soft::cuda::SoftmaxStats local = {
            -FLT_MAX, 0.0f, 0.0f};
        if (active) {
          const float raw_score =
              warp_scores[local_query * kScoreStride + local_key];
          const rosa_soft::cuda::ScoreTransform transformed =
              rosa_soft::cuda::transform_score(raw_score);
          const int route_position = key_position + 1;
          const float utility = use_grouped_utility
              ? grouped_utility
              : route_utility(
                    value,
                    grad_output,
                    batch,
                    head,
                    value_head,
                    query_position,
                    route_position,
                    seq_len,
                    num_heads,
                    num_value_heads,
                    value_dim);
          const float dropout_scale =
              rosa_soft::cuda::attention_dropout_scale(
                  dropout_seed,
                  dropout_p,
                  inverse_keep_probability,
                  batch,
                  head,
                  query_position,
                  route_position);
          local = rosa_soft::cuda::append_item(
              local,
              transformed.route_score * scale -
                  logf(static_cast<float>(query_position)),
              dropout_scale * utility);
        }
        local = rosa_soft::cuda::warp_reduce_stats(local);
        if (lane == 0 && local.normalizer != 0.0f) {
          const int macro_query = query_micro_start + local_query;
          const rosa_soft::cuda::SoftmaxStats previous = {
              shared_maximum[macro_query],
              shared_normalizer[macro_query],
              shared_utility[macro_query]};
          const rosa_soft::cuda::SoftmaxStats merged =
              rosa_soft::cuda::merge_stats(previous, local);
          shared_maximum[macro_query] = merged.maximum;
          shared_normalizer[macro_query] = merged.normalizer;
          shared_utility[macro_query] = merged.utility_numerator;
        }
      }
    }
    __syncthreads();
  }

  if constexpr (!ReplayFromCheckpoints) {
    const int checkpoint_slot = triangular_slot(tile_row, tile_col);
    const int macro_tile_slots =
        macro_tile_count * (macro_tile_count + 1) / 2;
    const int64_t checkpoint_base =
        (static_cast<int64_t>(series) * macro_tile_slots + checkpoint_slot) *
        (2 * TileSize);
    for (int index = threadIdx.x; index < 2 * TileSize; index += kThreads) {
      const int local_delta = index - (TileSize - 1);
      const int delta = center_delta + local_delta;
      const int local_query_begin = max(0, local_delta);
      const int local_query_end = min(
          query_count - 1,
          local_delta + key_count - 1);
      const bool has_segment = delta > 0 && delta < seq_len &&
          local_query_end >= local_query_begin;
      const float outgoing = has_segment ? shared_carry[index] : 0.0f;
      checkpoints[checkpoint_base + index] = outgoing;
      if (has_segment) {
        diagonal_carry[series_offset + delta] = outgoing;
      }
    }
  }
  for (int index = threadIdx.x; index < 3 * query_count; index += kThreads) {
    const int local_query = index / 3;
    const int field = index - local_query * 3;
    const int checkpoint_slot = triangular_slot(tile_row, tile_col);
    const int macro_tile_slots =
        macro_tile_count * (macro_tile_count + 1) / 2;
    int64_t stats_index =
        (series_offset + query_start + local_query) * 3 + field;
    float* output_stats = row_stats;
    if (partial_stats != nullptr) {
      stats_index =
          ((static_cast<int64_t>(series) * macro_tile_slots +
            checkpoint_slot) *
               TileSize +
           local_query) *
              3 +
          field;
      output_stats = partial_stats;
    }
    if (field == 0) {
      output_stats[stats_index] = shared_maximum[local_query];
    } else if (field == 1) {
      output_stats[stats_index] = shared_normalizer[local_query];
    } else {
      output_stats[stats_index] = shared_utility[local_query];
    }
  }
  __syncthreads();
}


__device__ __forceinline__ void warp_segmented_reverse_affine_scan(
    float& coefficient,
    float& bias,
    int segment_end) {
  const int lane = threadIdx.x & (kWarpSize - 1);
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const float right_coefficient = __shfl_down_sync(
        0xffffffffu, coefficient, offset);
    const float right_bias = __shfl_down_sync(
        0xffffffffu, bias, offset);
    if (lane + offset < segment_end) {
      bias = fmaf(coefficient, right_bias, bias);
      coefficient *= right_coefficient;
    }
  }
}


template <int TileSize>
__device__ __forceinline__ float macro_incoming_score(
    const float* __restrict__ checkpoints,
    int series,
    int tile_row,
    int tile_col,
    int local_delta,
    int query_count,
    int key_count,
    int seq_len,
    int macro_tile_count) {
  const int center_delta = (tile_row - tile_col) * TileSize;
  const int delta = center_delta + local_delta;
  const int local_query_begin = max(0, local_delta);
  const int local_query_end = min(
      query_count - 1,
      local_delta + key_count - 1);
  if (delta <= 0 || delta >= seq_len ||
      local_query_end < local_query_begin) {
    return 0.0f;
  }
  const int local_key_begin = local_query_begin - local_delta;
  const int query_position = tile_row * TileSize + local_query_begin;
  const int key_position = tile_col * TileSize + local_key_begin;
  if (query_position == 0 || key_position == 0) {
    return 0.0f;
  }
  const int predecessor_query = query_position - 1;
  const int predecessor_key = key_position - 1;
  const int predecessor_tile_row = predecessor_query / TileSize;
  const int predecessor_tile_col = predecessor_key / TileSize;
  const int predecessor_local_delta =
      predecessor_query - predecessor_tile_row * TileSize -
      (predecessor_key - predecessor_tile_col * TileSize);
  const int predecessor_slot = triangular_slot(
      predecessor_tile_row, predecessor_tile_col);
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;
  const int64_t checkpoint_index =
      (static_cast<int64_t>(series) * macro_tile_slots + predecessor_slot) *
          (2 * TileSize) +
      predecessor_local_delta + TileSize - 1;
  return checkpoints[checkpoint_index];
}


template <int TileSize>
__global__ void folded_diagonal_checkpoint_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ dense_scores,
    float* __restrict__ checkpoints,
    int series_count,
    int seq_len,
    int symbol_dim,
    int macro_tile_count,
    float mismatch_unit) {
  constexpr int kWarps = TileSize / kMicroTile;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int pair_count = seq_len / 2;
  const int task_count = series_count * pair_count;
  const int first_task = blockIdx.x * kWarps + warp;
  const int task_stride = gridDim.x * kWarps;
  const uint32_t mask = symbol_mask(symbol_dim);
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;

  for (int task = first_task; task < task_count; task += task_stride) {
    const int series = task / pair_count;
    const int pair = task - series * pair_count;
    const int long_delta = pair + 1;
    const int short_delta = seq_len - long_delta;
    const int long_length = seq_len - long_delta;
    const bool has_short = short_delta != long_delta;
    const int short_length = has_short ? long_delta : 0;
    const int stream_length = long_length + short_length;
    const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
    float long_carry = 0.0f;
    float short_carry = 0.0f;

    for (int stream_start = 0;
         stream_start < stream_length;
         stream_start += kWarpSize) {
      const int stream_position = stream_start + lane;
      const bool active = stream_position < stream_length;
      const bool in_short = has_short && stream_position >= long_length;
      const int diagonal = in_short ? short_delta : long_delta;
      const int diagonal_position = in_short
          ? stream_position - long_length
          : stream_position;
      const int query_position = diagonal_position + diagonal;
      const int key_position = diagonal_position;
      const int mismatch = active
          ? __popc(
                (static_cast<uint32_t>(
                     packed_query[series_offset + query_position]) ^
                 static_cast<uint32_t>(
                     packed_key[series_offset + key_position])) &
                mask)
          : 0;
      const float gate = active
          ? __expf(-mismatch_unit * static_cast<float>(mismatch))
          : 1.0f;
      float coefficient = gate;
      float bias = active ? gate : 0.0f;
      const int short_segment_start = long_length - stream_start;
      const int segment_start = in_short && short_segment_start > 0
          ? short_segment_start
          : 0;
      warp_segmented_affine_scan(coefficient, bias, segment_start);
      const float incoming = in_short ? short_carry : long_carry;
      const float score = fmaf(coefficient, incoming, bias);

      if (active) {
        if (dense_scores != nullptr) {
          const int64_t output =
              (static_cast<int64_t>(series) * seq_len + query_position) *
                  seq_len +
              key_position + 1;
          dense_scores[output] = score;
        }
        const bool tile_segment_end = query_position + 1 == seq_len ||
            key_position + 1 == seq_len ||
            (query_position + 1) % TileSize == 0 ||
            (key_position + 1) % TileSize == 0;
        if (tile_segment_end) {
          const int tile_row = query_position / TileSize;
          const int tile_col = key_position / TileSize;
          const int local_delta = query_position - tile_row * TileSize -
              (key_position - tile_col * TileSize);
          const int checkpoint_slot = triangular_slot(tile_row, tile_col);
          const int64_t checkpoint_index =
              (static_cast<int64_t>(series) * macro_tile_slots +
               checkpoint_slot) *
                  (2 * TileSize) +
              local_delta + TileSize - 1;
          checkpoints[checkpoint_index] = score;
        }
      }

      const unsigned long_mask = __ballot_sync(
          0xffffffffu, active && !in_short);
      if (long_mask != 0) {
        const int final_lane = 31 - __clz(long_mask);
        long_carry = __shfl_sync(0xffffffffu, score, final_lane);
      }
      const unsigned short_mask = __ballot_sync(
          0xffffffffu, active && in_short);
      if (short_mask != 0) {
        const int final_lane = 31 - __clz(short_mask);
        short_carry = __shfl_sync(0xffffffffu, score, final_lane);
      }
    }
  }
}


template <int TileSize, typename scalar_t>
__global__ void folded_tile_stats_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    float* __restrict__ row_stats,
    float* __restrict__ partial_stats,
    float* __restrict__ checkpoints,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  const int series_count = batch_size * num_heads;
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;
  const int task_count = series_count * macro_tile_slots;
  for (int task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int series = task / macro_tile_slots;
    const int slot = task - series * macro_tile_slots;
    const int tile_row = triangular_row(slot);
    const int tile_col = slot - triangular_slot(tile_row, 0);
    process_stats_macro_tile<TileSize, scalar_t, true>(
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        row_stats,
        partial_stats,
        nullptr,
        checkpoints,
        series,
        tile_row,
        tile_col,
        batch_size,
        seq_len,
        num_heads,
        symbol_dim,
        num_value_heads,
        value_dim,
        macro_tile_count,
        scale,
        dropout_p,
        inverse_keep_probability,
        mismatch_unit,
        shared_storage);
  }
}


template <int TileSize>
__global__ void reduce_folded_tile_stats_kernel(
    const float* __restrict__ partial_stats,
    float* __restrict__ row_stats,
    int series_count,
    int seq_len,
    int macro_tile_count) {
  const int64_t total_rows =
      static_cast<int64_t>(series_count) * seq_len;
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;
  for (int64_t row_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row_index < total_rows;
       row_index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int series = row_index / seq_len;
    const int query_position = row_index -
        static_cast<int64_t>(series) * seq_len;
    const int tile_row = query_position / TileSize;
    const int local_query = query_position - tile_row * TileSize;
    rosa_soft::cuda::SoftmaxStats total = {
        row_stats[row_index * 3],
        row_stats[row_index * 3 + 1],
        row_stats[row_index * 3 + 2]};
    for (int tile_col = 0; tile_col <= tile_row; ++tile_col) {
      const int slot = triangular_slot(tile_row, tile_col);
      const int64_t partial_index =
          ((static_cast<int64_t>(series) * macro_tile_slots + slot) *
               TileSize +
           local_query) *
          3;
      const rosa_soft::cuda::SoftmaxStats partial = {
          partial_stats[partial_index],
          partial_stats[partial_index + 1],
          partial_stats[partial_index + 2]};
      if (partial.normalizer != 0.0f) {
        total = rosa_soft::cuda::merge_stats(total, partial);
      }
    }
    row_stats[row_index * 3] = total.maximum;
    row_stats[row_index * 3 + 1] = total.normalizer;
    row_stats[row_index * 3 + 2] = total.utility_numerator;
  }
}


template <int TileSize>
__device__ __forceinline__ float checkpoint_score_at_cell(
    const float* __restrict__ checkpoints,
    int series,
    int query_position,
    int key_position,
    int macro_tile_count) {
  const int tile_row = query_position / TileSize;
  const int tile_col = key_position / TileSize;
  const int local_delta = query_position - tile_row * TileSize -
      (key_position - tile_col * TileSize);
  const int checkpoint_slot = triangular_slot(tile_row, tile_col);
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;
  const int64_t checkpoint_index =
      (static_cast<int64_t>(series) * macro_tile_slots + checkpoint_slot) *
          (2 * TileSize) +
      local_delta + TileSize - 1;
  return checkpoints[checkpoint_index];
}


template <int TileSize, typename scalar_t>
__device__ __forceinline__ void process_folded_reverse_diagonal(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    const float* __restrict__ checkpoints,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ shared_scores,
    int gradient_mask,
    int series,
    int diagonal,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    float symbol_scale) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int value_head = head / (num_heads / num_value_heads);
  const int diagonal_length = seq_len - diagonal;
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  float incoming_adjoint = 0.0f;

  for (int chunk_start =
           ((diagonal_length - 1) / TileSize) * TileSize;
       chunk_start >= 0;
       chunk_start -= TileSize) {
    const int chunk_length = min(TileSize, diagonal_length - chunk_start);
    float forward_carry = chunk_start == 0
        ? 0.0f
        : checkpoint_score_at_cell<TileSize>(
              checkpoints,
              series,
              chunk_start + diagonal - 1,
              chunk_start - 1,
              macro_tile_count);

    for (int offset = 0; offset < chunk_length; offset += kWarpSize) {
      const int diagonal_position = chunk_start + offset + lane;
      const bool active = offset + lane < chunk_length;
      const int query_position = diagonal_position + diagonal;
      const int key_position = diagonal_position;
      const int mismatch = active
          ? __popc(
                (static_cast<uint32_t>(
                     packed_query[series_offset + query_position]) ^
                 static_cast<uint32_t>(
                     packed_key[series_offset + key_position])) &
                mask)
          : 0;
      const float gate = active
          ? __expf(-mismatch_unit * static_cast<float>(mismatch))
          : 1.0f;
      float coefficient = gate;
      float bias = active ? gate : 0.0f;
      warp_segmented_affine_scan(coefficient, bias, 0);
      const float score = fmaf(coefficient, forward_carry, bias);
      if (active) {
        shared_scores[offset + lane] = score;
      }
      const unsigned active_mask = __ballot_sync(0xffffffffu, active);
      const int final_lane = 31 - __clz(active_mask);
      forward_carry = __shfl_sync(0xffffffffu, score, final_lane);
    }
    __syncwarp();

    for (int offset = ((chunk_length - 1) / kWarpSize) * kWarpSize;
         offset >= 0;
         offset -= kWarpSize) {
      const int active_count = min(kWarpSize, chunk_length - offset);
      const bool use_grouped_utility = value_dim >= 32;
      const float grouped_utility = use_grouped_utility
          ? grouped_route_utility_dispatch(
                value,
                grad_output,
                batch,
                head,
                value_head,
                chunk_start + offset + diagonal,
                chunk_start + offset + 1,
                active_count,
                1,
                1,
                seq_len,
                num_heads,
                num_value_heads,
                value_dim)
          : 0.0f;
      const bool active = lane < active_count;
      const int diagonal_position = chunk_start + offset + lane;
      const int query_position = diagonal_position + diagonal;
      const int key_position = diagonal_position;
      float gate = 0.0f;
      float raw_score = 0.0f;
      float direct_vjp = 0.0f;
      if (active) {
        const int mismatch = __popc(
            (static_cast<uint32_t>(
                 packed_query[series_offset + query_position]) ^
             static_cast<uint32_t>(
                 packed_key[series_offset + key_position])) &
            mask);
        gate = __expf(-mismatch_unit * static_cast<float>(mismatch));
        raw_score = shared_scores[offset + lane];
        const auto transformed =
            rosa_soft::cuda::transform_score(raw_score);
        const int64_t stats_index =
            (series_offset + query_position) * 3;
        const float probability = __expf(
            transformed.route_score * scale -
            logf(static_cast<float>(query_position)) -
            row_stats[stats_index]) /
            row_stats[stats_index + 1];
        const int route_position = key_position + 1;
        const float dropout_scale =
            rosa_soft::cuda::attention_dropout_scale(
                dropout_seed,
                dropout_p,
                inverse_keep_probability,
                batch,
                head,
                query_position,
                route_position);
        const float utility = use_grouped_utility
            ? grouped_utility
            : route_utility(
                  value,
                  grad_output,
                  batch,
                  head,
                  value_head,
                  query_position,
                  route_position,
                  seq_len,
                  num_heads,
                  num_value_heads,
                  value_dim);
        direct_vjp = scale * probability *
            (dropout_scale * utility - row_stats[stats_index + 2]) *
            transformed.raw_vjp_multiplier;
      }
      const float successor_gate = __shfl_down_sync(
          0xffffffffu, gate, 1);
      float coefficient =
          active && lane + 1 < active_count ? successor_gate : 1.0f;
      warp_segmented_reverse_affine_scan(
          coefficient, direct_vjp, active_count);
      const float score_vjp = fmaf(
          coefficient, incoming_adjoint, direct_vjp);
      if (active) {
        const float log_gate_vjp = raw_score * score_vjp;
        if ((gradient_mask & rosa_soft::cuda::kGradQuery) != 0) {
          const int64_t output_base =
              ((static_cast<int64_t>(batch) * seq_len + query_position) *
                   num_heads +
               head) *
              symbol_dim;
          const uint32_t key_word = static_cast<uint32_t>(
              packed_key[series_offset + key_position]);
          for (int bit = 0; bit < symbol_dim; ++bit) {
            atomicAdd(
                grad_query + output_base + bit,
                symbol_scale * log_gate_vjp *
                    static_cast<float>(
                        rosa_soft::cuda::sign_from_bit(key_word, bit)));
          }
        }
        if ((gradient_mask & rosa_soft::cuda::kGradKey) != 0) {
          const int64_t output_base =
              ((static_cast<int64_t>(batch) * seq_len + key_position) *
                   num_heads +
               head) *
              symbol_dim;
          const uint32_t query_word = static_cast<uint32_t>(
              packed_query[series_offset + query_position]);
          for (int bit = 0; bit < symbol_dim; ++bit) {
            atomicAdd(
                grad_key + output_base + bit,
                symbol_scale * log_gate_vjp *
                    static_cast<float>(
                        rosa_soft::cuda::sign_from_bit(query_word, bit)));
          }
        }
      }
      const float first_score_vjp = __shfl_sync(
          0xffffffffu, score_vjp, 0);
      const float first_gate = __shfl_sync(0xffffffffu, gate, 0);
      incoming_adjoint = first_gate * first_score_vjp;
    }
    __syncwarp();
  }
}


template <int TileSize, typename scalar_t>
__global__ void folded_diagonal_reverse_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    const float* __restrict__ checkpoints,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    int gradient_mask,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    float symbol_scale) {
  extern __shared__ __align__(16) float shared_scores[];
  constexpr int kWarps = TileSize / kMicroTile;
  const int warp = threadIdx.x / kWarpSize;
  const int pair_count = seq_len / 2;
  const int task_count = batch_size * num_heads * pair_count;
  const int first_task = blockIdx.x * kWarps + warp;
  const int task_stride = gridDim.x * kWarps;
  for (int task = first_task; task < task_count; task += task_stride) {
    const int series = task / pair_count;
    const int pair = task - series * pair_count;
    const int long_delta = pair + 1;
    const int short_delta = seq_len - long_delta;
    float* warp_scores = shared_scores + warp * TileSize;
    process_folded_reverse_diagonal<TileSize, scalar_t>(
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        row_stats,
        checkpoints,
        grad_query,
        grad_key,
        warp_scores,
        gradient_mask,
        series,
        long_delta,
        seq_len,
        num_heads,
        symbol_dim,
        num_value_heads,
        value_dim,
        macro_tile_count,
        scale,
        dropout_p,
        inverse_keep_probability,
        mismatch_unit,
        symbol_scale);
    if (short_delta != long_delta) {
      process_folded_reverse_diagonal<TileSize, scalar_t>(
          value,
          grad_output,
          packed_query,
          packed_key,
          dropout_seed,
          row_stats,
          checkpoints,
          grad_query,
          grad_key,
          warp_scores,
          gradient_mask,
          series,
          short_delta,
          seq_len,
          num_heads,
          symbol_dim,
          num_value_heads,
          value_dim,
          macro_tile_count,
          scale,
          dropout_p,
          inverse_keep_probability,
          mismatch_unit,
          symbol_scale);
    }
  }
}


template <int TileSize>
__device__ __forceinline__ float micro_incoming_score(
    const float* __restrict__ macro_input,
    const float* __restrict__ micro_edges,
    int micro_row,
    int micro_col,
    int micro_delta,
    int macro_local_delta) {
  constexpr int kMicroTiles = TileSize / kMicroTile;
  constexpr int kMicroEdgeStride = 2 * kMicroTile;
  const int local_query_begin = max(0, micro_delta);
  const int local_key_begin = local_query_begin - micro_delta;
  const int macro_query = micro_row * kMicroTile + local_query_begin;
  const int macro_key = micro_col * kMicroTile + local_key_begin;
  if (macro_query == 0 || macro_key == 0) {
    return macro_input[macro_local_delta + TileSize - 1];
  }
  const int predecessor_query = macro_query - 1;
  const int predecessor_key = macro_key - 1;
  const int predecessor_micro_row = predecessor_query / kMicroTile;
  const int predecessor_micro_col = predecessor_key / kMicroTile;
  if (predecessor_micro_row < 0 ||
      predecessor_micro_row >= kMicroTiles ||
      predecessor_micro_col < 0 ||
      predecessor_micro_col >= kMicroTiles) {
    return macro_input[macro_local_delta + TileSize - 1];
  }
  const int predecessor_delta =
      predecessor_query - predecessor_micro_row * kMicroTile -
      (predecessor_key - predecessor_micro_col * kMicroTile);
  const int predecessor_slot =
      predecessor_micro_row * kMicroTiles + predecessor_micro_col;
  return micro_edges[
      predecessor_slot * kMicroEdgeStride +
      predecessor_delta + kMicroTile - 1];
}


template <int TileSize>
__device__ __forceinline__ void build_micro_edge_checkpoints(
    const uint32_t* __restrict__ shared_query,
    const uint32_t* __restrict__ shared_key,
    const float* __restrict__ macro_input,
    float* __restrict__ micro_edges,
    int tile_row,
    int tile_col,
    int query_count,
    int key_count,
    int symbol_dim,
    float mismatch_unit) {
  constexpr int kMicroTiles = TileSize / kMicroTile;
  constexpr int kMicroEdgeStride = 2 * kMicroTile;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const uint32_t mask = symbol_mask(symbol_dim);
  const int center_delta = (tile_row - tile_col) * TileSize;

  for (int index = threadIdx.x;
       index < kMicroTiles * kMicroTiles * kMicroEdgeStride;
       index += blockDim.x) {
    micro_edges[index] = 0.0f;
  }
  __syncthreads();

  for (int micro_wave = 0;
       micro_wave < 2 * kMicroTiles - 1;
       ++micro_wave) {
    const int first_micro_col = max(0, micro_wave - (kMicroTiles - 1));
    const int final_micro_col = min(kMicroTiles - 1, micro_wave);
    const int micro_col = first_micro_col + warp;
    if (micro_col <= final_micro_col) {
      const int micro_row = micro_wave - micro_col;
      const int query_micro_start = micro_row * kMicroTile;
      const int key_micro_start = micro_col * kMicroTile;
      const int micro_query_count = min(
          kMicroTile, max(0, query_count - query_micro_start));
      const int micro_key_count = min(
          kMicroTile, max(0, key_count - key_micro_start));
      const int micro_slot = micro_row * kMicroTiles + micro_col;

      for (int pair = 0; pair < kMicroTile; ++pair) {
        const PackedDiagonalLane diagonal =
            packed_diagonal_lane(pair, lane);
        const int micro_delta = diagonal.delta;
        const int macro_local_delta =
            query_micro_start - key_micro_start + micro_delta;
        const int delta = center_delta + macro_local_delta;
        const int local_query_begin = max(0, micro_delta);
        const int local_query_end = min(
            micro_query_count - 1,
            micro_delta + micro_key_count - 1);
        const int segment_length =
            local_query_end >= local_query_begin
            ? local_query_end - local_query_begin + 1
            : 0;
        const bool active =
            diagonal.offset < segment_length && delta > 0;
        const int local_query = local_query_begin + diagonal.offset;
        const int local_key = local_query - micro_delta;
        const int macro_query = query_micro_start + local_query;
        const int macro_key = key_micro_start + local_key;
        const int mismatch = active
            ? __popc(
                  (shared_query[macro_query] ^ shared_key[macro_key]) & mask)
            : 0;
        const float gate = active
            ? __expf(-mismatch_unit * static_cast<float>(mismatch))
            : 1.0f;
        float coefficient = gate;
        float bias = active ? gate : 0.0f;
        warp_segmented_affine_scan(
            coefficient, bias, diagonal.segment_start);
        const float incoming =
            segment_length > 0 && delta > 0
            ? micro_incoming_score<TileSize>(
                  macro_input,
                  micro_edges,
                  micro_row,
                  micro_col,
                  micro_delta,
                  macro_local_delta)
            : 0.0f;
        const float score = fmaf(coefficient, incoming, bias);
        if (diagonal.offset == segment_length - 1 && delta > 0) {
          micro_edges[
              micro_slot * kMicroEdgeStride +
              micro_delta + kMicroTile - 1] = score;
        }
      }
    }
    __syncthreads();
  }
}


template <int TileSize, typename scalar_t, bool PublishAdjoint = true>
__device__ __forceinline__ void process_reverse_macro_tile(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ checkpoints,
    float* __restrict__ reverse_carry,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int gradient_mask,
    int series,
    int tile_row,
    int tile_col,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    float symbol_scale,
    unsigned char* __restrict__ shared_storage) {
  constexpr int kMicroTiles = TileSize / kMicroTile;
  constexpr int kThreads = kMicroTiles * kWarpSize;
  constexpr int kMicroEdgeStride = 2 * kMicroTile;

  uint32_t* shared_query =
      reinterpret_cast<uint32_t*>(shared_storage);
  uint32_t* shared_key = shared_query + TileSize;
  float* macro_input = reinterpret_cast<float*>(shared_key + TileSize);
  float* shared_adjoint = macro_input + 2 * TileSize;
  float* micro_edges = shared_adjoint + 2 * TileSize;
  float* shared_score_tiles =
      micro_edges + kMicroTiles * kMicroTiles * kMicroEdgeStride;

  const int query_start = tile_row * TileSize;
  const int key_start = tile_col * TileSize;
  const int query_count = min(TileSize, seq_len - query_start);
  const int key_count = min(TileSize, seq_len - key_start);
  const int center_delta = (tile_row - tile_col) * TileSize;
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int value_head = head / (num_heads / num_value_heads);

  for (int index = threadIdx.x; index < TileSize; index += kThreads) {
    shared_query[index] = index < query_count
        ? static_cast<uint32_t>(packed_query[series_offset + query_start + index])
        : 0u;
    shared_key[index] = index < key_count
        ? static_cast<uint32_t>(packed_key[series_offset + key_start + index])
        : 0u;
  }
  for (int index = threadIdx.x; index < 2 * TileSize; index += kThreads) {
    const int local_delta = index - (TileSize - 1);
    const int delta = center_delta + local_delta;
    macro_input[index] = macro_incoming_score<TileSize>(
        checkpoints,
        series,
        tile_row,
        tile_col,
        local_delta,
        query_count,
        key_count,
        seq_len,
        macro_tile_count);
    shared_adjoint[index] = delta > 0 && delta < seq_len
        ? reverse_carry[series_offset + delta]
        : 0.0f;
  }
  __syncthreads();

  build_micro_edge_checkpoints<TileSize>(
      shared_query,
      shared_key,
      macro_input,
      micro_edges,
      tile_row,
      tile_col,
      query_count,
      key_count,
      symbol_dim,
      mismatch_unit);

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const uint32_t mask = symbol_mask(symbol_dim);
  float* warp_scores =
      shared_score_tiles + warp * kMicroTile * kScoreStride;

  for (int micro_wave = 2 * kMicroTiles - 2;
       micro_wave >= 0;
       --micro_wave) {
    const int first_micro_col = max(0, micro_wave - (kMicroTiles - 1));
    const int final_micro_col = min(kMicroTiles - 1, micro_wave);
    const int micro_col = first_micro_col + warp;
    if (micro_col <= final_micro_col) {
      const int micro_row = micro_wave - micro_col;
      const int query_micro_start = micro_row * kMicroTile;
      const int key_micro_start = micro_col * kMicroTile;
      const int micro_query_count = min(
          kMicroTile, max(0, query_count - query_micro_start));
      const int micro_key_count = min(
          kMicroTile, max(0, key_count - key_micro_start));

      // Reconstruct every raw score in this microtile from its exact incoming
      // micro-edge checkpoint.
      for (int pair = 0; pair < kMicroTile; ++pair) {
        const PackedDiagonalLane diagonal =
            packed_diagonal_lane(pair, lane);
        const int micro_delta = diagonal.delta;
        const int macro_local_delta =
            query_micro_start - key_micro_start + micro_delta;
        const int delta = center_delta + macro_local_delta;
        const int local_query_begin = max(0, micro_delta);
        const int local_query_end = min(
            micro_query_count - 1,
            micro_delta + micro_key_count - 1);
        const int segment_length =
            local_query_end >= local_query_begin
            ? local_query_end - local_query_begin + 1
            : 0;
        const bool active =
            diagonal.offset < segment_length && delta > 0;
        const int local_query = local_query_begin + diagonal.offset;
        const int local_key = local_query - micro_delta;
        const int macro_query = query_micro_start + local_query;
        const int macro_key = key_micro_start + local_key;
        const int mismatch = active
            ? __popc(
                  (shared_query[macro_query] ^ shared_key[macro_key]) & mask)
            : 0;
        const float gate = active
            ? __expf(-mismatch_unit * static_cast<float>(mismatch))
            : 1.0f;
        float coefficient = gate;
        float bias = active ? gate : 0.0f;
        warp_segmented_affine_scan(
            coefficient, bias, diagonal.segment_start);
        const float incoming =
            segment_length > 0 && delta > 0
            ? micro_incoming_score<TileSize>(
                  macro_input,
                  micro_edges,
                  micro_row,
                  micro_col,
                  micro_delta,
                  macro_local_delta)
            : 0.0f;
        const float score = fmaf(coefficient, incoming, bias);
        if (active) {
          warp_scores[local_query * kScoreStride + local_key] = score;
        }
      }
      __syncwarp();

      if ((gradient_mask & rosa_soft::cuda::kGradValue) != 0) {
        for (int local_key = 0;
             local_key < micro_key_count;
             ++local_key) {
          const int key_position =
              key_start + key_micro_start + local_key;
          const int route_position = key_position + 1;
          for (int feature = lane;
               feature < value_dim;
               feature += kWarpSize) {
            float contribution = 0.0f;
            for (int local_query = 0;
                 local_query < micro_query_count;
                 ++local_query) {
              const int query_position =
                  query_start + query_micro_start + local_query;
              if (key_position >= query_position) {
                continue;
              }
              const float raw_score =
                  warp_scores[local_query * kScoreStride + local_key];
              const auto transformed =
                  rosa_soft::cuda::transform_score(raw_score);
              const int64_t stats_index =
                  (series_offset + query_position) * 3;
              float probability = __expf(
                  transformed.route_score * scale -
                  logf(static_cast<float>(query_position)) -
                  row_stats[stats_index]) /
                  row_stats[stats_index + 1];
              probability *= rosa_soft::cuda::attention_dropout_scale(
                  dropout_seed,
                  dropout_p,
                  inverse_keep_probability,
                  batch,
                  head,
                  query_position,
                  route_position);
              const int64_t grad_index =
                  ((static_cast<int64_t>(batch) * seq_len + query_position) *
                       num_heads +
                   head) * value_dim + feature;
              contribution = fmaf(
                  probability,
                  rosa_soft::cuda::read_float(
                      grad_output, grad_index),
                  contribution);
            }
            if (contribution != 0.0f) {
              const int64_t value_index =
                  ((static_cast<int64_t>(batch) * seq_len + route_position) *
                       num_value_heads +
                   value_head) * value_dim + feature;
              atomicAdd(grad_value + value_index, contribution);
            }
          }
        }
      }
      __syncwarp();

      if ((gradient_mask &
           (rosa_soft::cuda::kGradQuery | rosa_soft::cuda::kGradKey)) != 0) {
        // Reverse each local diagonal. shared_adjoint carries
        // g(successor) * dS(successor) across micro- and macro-tile edges.
        for (int pair = 0; pair < kMicroTile; ++pair) {
          const PackedDiagonalLane diagonal =
              packed_diagonal_lane(pair, lane);
          const int micro_delta = diagonal.delta;
          const int macro_local_delta =
              query_micro_start - key_micro_start + micro_delta;
          const int delta = center_delta + macro_local_delta;
          const int local_query_begin = max(0, micro_delta);
          const int local_query_end = min(
              micro_query_count - 1,
              micro_delta + micro_key_count - 1);
          const int segment_length =
              local_query_end >= local_query_begin
              ? local_query_end - local_query_begin + 1
              : 0;
          const bool active =
              diagonal.offset < segment_length && delta > 0;
          const int local_query = local_query_begin + diagonal.offset;
          const int local_key = local_query - micro_delta;
          const int macro_query = query_micro_start + local_query;
          const int macro_key = key_micro_start + local_key;
          const int mismatch = active
              ? __popc(
                    (shared_query[macro_query] ^ shared_key[macro_key]) & mask)
              : 0;
          const float gate = active
              ? __expf(-mismatch_unit * static_cast<float>(mismatch))
              : 0.0f;
          const float successor_gate = __shfl_down_sync(
              0xffffffffu, gate, 1);
          float coefficient = 1.0f;
          float direct_vjp = 0.0f;
          float raw_score = 0.0f;
          if (active) {
            raw_score = warp_scores[
                local_query * kScoreStride + local_key];
            if (diagonal.offset < segment_length - 1) {
              coefficient = successor_gate;
            }
            const auto transformed =
                rosa_soft::cuda::transform_score(raw_score);
            const int query_position = query_start + macro_query;
            const int key_position = key_start + macro_key;
            const int64_t stats_index =
                (series_offset + query_position) * 3;
            const float probability = __expf(
                transformed.route_score * scale -
                logf(static_cast<float>(query_position)) -
                row_stats[stats_index]) /
                row_stats[stats_index + 1];
            const int route_position = key_position + 1;
            const float dropout_scale =
                rosa_soft::cuda::attention_dropout_scale(
                    dropout_seed,
                    dropout_p,
                    inverse_keep_probability,
                    batch,
                    head,
                    query_position,
                    route_position);
            const float utility = route_utility(
                value,
                grad_output,
                batch,
                head,
                value_head,
                query_position,
                route_position,
                seq_len,
                num_heads,
                num_value_heads,
                value_dim);
            direct_vjp = scale * probability *
                (dropout_scale * utility - row_stats[stats_index + 2]) *
                transformed.raw_vjp_multiplier;
          }
          warp_segmented_reverse_affine_scan(
              coefficient,
              direct_vjp,
              diagonal.segment_start + diagonal.nominal_length);
          const int carry_index = macro_local_delta + TileSize - 1;
          float incoming_adjoint = 0.0f;
          if (diagonal.offset == 0 && segment_length > 0 && delta > 0) {
            incoming_adjoint = shared_adjoint[carry_index];
          }
          incoming_adjoint = __shfl_sync(
              0xffffffffu,
              incoming_adjoint,
              diagonal.segment_start);
          __syncwarp();
          const float score_vjp = fmaf(
              coefficient, incoming_adjoint, direct_vjp);
          if (active) {
            warp_scores[local_query * kScoreStride + local_key] =
                raw_score * score_vjp;
          }
          if (diagonal.offset == 0 && segment_length > 0 && delta > 0) {
            shared_adjoint[carry_index] = gate * score_vjp;
          }
        }
        __syncwarp();

        if ((gradient_mask & rosa_soft::cuda::kGradQuery) != 0) {
          for (int local_query = 0;
               local_query < micro_query_count;
               ++local_query) {
            if (lane < symbol_dim) {
              float contribution = 0.0f;
              for (int local_key = 0;
                   local_key < micro_key_count;
                   ++local_key) {
                const int query_position =
                    query_start + query_micro_start + local_query;
                const int key_position =
                    key_start + key_micro_start + local_key;
                if (key_position < query_position) {
                  contribution = fmaf(
                      warp_scores[local_query * kScoreStride + local_key],
                      static_cast<float>(rosa_soft::cuda::sign_from_bit(
                          shared_key[key_micro_start + local_key], lane)),
                      contribution);
                }
              }
              const int query_position =
                  query_start + query_micro_start + local_query;
              const int64_t output =
                  ((static_cast<int64_t>(batch) * seq_len + query_position) *
                       num_heads +
                   head) * symbol_dim + lane;
              grad_query[output] += symbol_scale * contribution;
            }
          }
        }
        if ((gradient_mask & rosa_soft::cuda::kGradKey) != 0) {
          for (int local_key = 0;
               local_key < micro_key_count;
               ++local_key) {
            if (lane < symbol_dim) {
              float contribution = 0.0f;
              for (int local_query = 0;
                   local_query < micro_query_count;
                   ++local_query) {
                const int query_position =
                    query_start + query_micro_start + local_query;
                const int key_position =
                    key_start + key_micro_start + local_key;
                if (key_position < query_position) {
                  contribution = fmaf(
                      warp_scores[local_query * kScoreStride + local_key],
                      static_cast<float>(rosa_soft::cuda::sign_from_bit(
                          shared_query[query_micro_start + local_query], lane)),
                      contribution);
                }
              }
              const int key_position =
                  key_start + key_micro_start + local_key;
              const int64_t output =
                  ((static_cast<int64_t>(batch) * seq_len + key_position) *
                       num_heads +
                   head) * symbol_dim + lane;
              grad_key[output] += symbol_scale * contribution;
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if constexpr (PublishAdjoint) {
    const int checkpoint_slot = triangular_slot(tile_row, tile_col);
    const int macro_tile_slots =
        macro_tile_count * (macro_tile_count + 1) / 2;
    const int64_t checkpoint_base =
        (static_cast<int64_t>(series) * macro_tile_slots + checkpoint_slot) *
        (2 * TileSize);
    for (int index = threadIdx.x; index < 2 * TileSize; index += kThreads) {
      const int local_delta = index - (TileSize - 1);
      const int delta = center_delta + local_delta;
      const int local_query_begin = max(0, local_delta);
      const int local_query_end = min(
          query_count - 1,
          local_delta + key_count - 1);
      const bool has_segment = delta > 0 && delta < seq_len &&
          local_query_end >= local_query_begin;
      const float outgoing = has_segment ? shared_adjoint[index] : 0.0f;
      checkpoints[checkpoint_base + index] = outgoing;
      if (has_segment) {
        reverse_carry[series_offset + delta] = outgoing;
      }
    }
  }
  __syncthreads();
}


template <int TileSize, typename scalar_t>
__global__ void folded_tile_value_grad_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ checkpoints,
    float* __restrict__ reverse_carry,
    float* __restrict__ grad_value,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    float symbol_scale) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  const int series_count = batch_size * num_heads;
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;
  const int task_count = series_count * macro_tile_slots;
  for (int task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int series = task / macro_tile_slots;
    const int slot = task - series * macro_tile_slots;
    const int tile_row = triangular_row(slot);
    const int tile_col = slot - triangular_slot(tile_row, 0);
    process_reverse_macro_tile<TileSize, scalar_t, false>(
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        row_stats,
        checkpoints,
        reverse_carry,
        nullptr,
        nullptr,
        grad_value,
        rosa_soft::cuda::kGradValue,
        series,
        tile_row,
        tile_col,
        batch_size,
        seq_len,
        num_heads,
        symbol_dim,
        num_value_heads,
        value_dim,
        macro_tile_count,
        scale,
        dropout_p,
        inverse_keep_probability,
        mismatch_unit,
        symbol_scale,
        shared_storage);
  }
}


__device__ __forceinline__ void scheduler_backoff() {
#if __CUDA_ARCH__ >= 700
  __nanosleep(128);
#endif
}


__device__ __forceinline__ int dequeue_ready_task(
    const int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_ready,
    int32_t* __restrict__ counters,
    int total_tasks) {
  while (true) {
    if (atomicAdd(counters + 2, 0) >= total_tasks) {
      return -1;
    }
    const int head = atomicAdd(counters, 0);
    const int tail = atomicAdd(counters + 1, 0);
    if (head < tail && atomicCAS(counters, head, head + 1) == head) {
      while (atomicAdd(queue_ready + head, 0) == 0) {
        scheduler_backoff();
      }
      __threadfence();
      return queue[head];
    }
    scheduler_backoff();
  }
}


__device__ __forceinline__ void enqueue_ready_task(
    int task,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_ready,
    int32_t* __restrict__ counters) {
  const int slot = atomicAdd(counters + 1, 1);
  queue[slot] = task;
  __threadfence();
  atomicExch(queue_ready + slot, 1);
}


__device__ __forceinline__ void release_forward_successors(
    int task,
    int series,
    int tile_row,
    int tile_col,
    int macro_tile_count,
    int32_t* __restrict__ dependencies,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_ready,
    int32_t* __restrict__ counters) {
  const int series_base = series * macro_tile_count * macro_tile_count;
  if (tile_col + 1 <= tile_row) {
    const int successor = task + 1;
    if (atomicSub(dependencies + successor, 1) == 1) {
      enqueue_ready_task(successor, queue, queue_ready, counters);
    }
  }
  if (tile_row + 1 < macro_tile_count) {
    const int successor =
        series_base + (tile_row + 1) * macro_tile_count + tile_col;
    if (atomicSub(dependencies + successor, 1) == 1) {
      enqueue_ready_task(successor, queue, queue_ready, counters);
    }
  }
}


__device__ __forceinline__ void release_reverse_predecessors(
    int task,
    int series,
    int tile_row,
    int tile_col,
    int macro_tile_count,
    int32_t* __restrict__ dependencies,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_ready,
    int32_t* __restrict__ counters) {
  const int series_base = series * macro_tile_count * macro_tile_count;
  if (tile_col > 0) {
    const int predecessor = task - 1;
    if (atomicSub(dependencies + predecessor, 1) == 1) {
      enqueue_ready_task(predecessor, queue, queue_ready, counters);
    }
  }
  if (tile_row > 0 && tile_col < tile_row) {
    const int predecessor =
        series_base + (tile_row - 1) * macro_tile_count + tile_col;
    if (atomicSub(dependencies + predecessor, 1) == 1) {
      enqueue_ready_task(predecessor, queue, queue_ready, counters);
    }
  }
}


__global__ void initialize_ready_queue_kernel(
    int32_t* __restrict__ dependencies,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_ready,
    int32_t* __restrict__ counters,
    int series_count,
    int macro_tile_count,
    int reverse) {
  const int square_tiles = macro_tile_count * macro_tile_count;
  const int task_capacity = series_count * square_tiles;
  for (int task = blockIdx.x * blockDim.x + threadIdx.x;
       task < task_capacity;
       task += gridDim.x * blockDim.x) {
    const int local_tile = task % square_tiles;
    const int tile_row = local_tile / macro_tile_count;
    const int tile_col = local_tile - tile_row * macro_tile_count;
    if (tile_col > tile_row) {
      dependencies[task] = -1;
      continue;
    }
    const int dependency_count = reverse != 0
        ? (tile_row + 1 < macro_tile_count ? 1 : 0) +
            (tile_col < tile_row ? 1 : 0)
        : (tile_col > 0 ? 1 : 0) + (tile_row > tile_col ? 1 : 0);
    dependencies[task] = dependency_count;
    if (dependency_count == 0) {
      const int slot = atomicAdd(counters + 1, 1);
      queue[slot] = task;
      queue_ready[slot] = 1;
    }
  }
}


template <int TileSize>
__global__ void macro_wavefront_score_wave_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ dense_scores,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int series_count,
    int seq_len,
    int symbol_dim,
    int macro_tile_count,
    int tile_wave,
    int first_tile_col,
    int tiles_on_wave,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  const int task = blockIdx.x;
  const int series = task / tiles_on_wave;
  const int local_tile = task - series * tiles_on_wave;
  if (series >= series_count) {
    return;
  }
  const int tile_col = first_tile_col + local_tile;
  const int tile_row = tile_wave - tile_col;
  process_score_macro_tile<TileSize>(
      packed_query,
      packed_key,
      dense_scores,
      diagonal_carry,
      checkpoints,
      series,
      tile_row,
      tile_col,
      seq_len,
      symbol_dim,
      macro_tile_count,
      mismatch_unit,
      shared_storage);
}


template <int TileSize>
__global__ void persistent_macro_wavefront_score_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ dense_scores,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int series_count,
    int seq_len,
    int symbol_dim,
    int macro_tile_count,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  cg::grid_group grid = cg::this_grid();
  for (int tile_wave = 0;
       tile_wave < 2 * macro_tile_count - 1;
       ++tile_wave) {
    const int first_tile_col =
        first_column_on_wave(tile_wave, macro_tile_count);
    const int last_tile_col =
        last_causal_column_on_wave(tile_wave, macro_tile_count);
    const int tiles_on_wave = max(0, last_tile_col - first_tile_col + 1);
    const int task_count = series_count * tiles_on_wave;
    for (int task = blockIdx.x;
         task < task_count;
         task += gridDim.x) {
      const int series = task / tiles_on_wave;
      const int local_tile = task - series * tiles_on_wave;
      const int tile_col = first_tile_col + local_tile;
      const int tile_row = tile_wave - tile_col;
      process_score_macro_tile<TileSize>(
          packed_query,
          packed_key,
          dense_scores,
          diagonal_carry,
          checkpoints,
          series,
          tile_row,
          tile_col,
          seq_len,
          symbol_dim,
          macro_tile_count,
          mismatch_unit,
          shared_storage);
    }
    grid.sync();
  }
}


template <int TileSize>
__global__ void queued_macro_wavefront_score_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ dense_scores,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int32_t* __restrict__ dependencies,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_ready,
    int32_t* __restrict__ counters,
    int seq_len,
    int symbol_dim,
    int macro_tile_count,
    int total_tasks,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  __shared__ int shared_task;
  const int square_tiles = macro_tile_count * macro_tile_count;
  while (true) {
    if (threadIdx.x == 0) {
      shared_task = dequeue_ready_task(
          queue, queue_ready, counters, total_tasks);
    }
    __syncthreads();
    const int task = shared_task;
    if (task < 0) {
      return;
    }
    const int series = task / square_tiles;
    const int local_tile = task - series * square_tiles;
    const int tile_row = local_tile / macro_tile_count;
    const int tile_col = local_tile - tile_row * macro_tile_count;
    process_score_macro_tile<TileSize>(
        packed_query,
        packed_key,
        dense_scores,
        diagonal_carry,
        checkpoints,
        series,
        tile_row,
        tile_col,
        seq_len,
        symbol_dim,
        macro_tile_count,
        mismatch_unit,
        shared_storage);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      release_forward_successors(
          task,
          series,
          tile_row,
          tile_col,
          macro_tile_count,
          dependencies,
          queue,
          queue_ready,
          counters);
      atomicAdd(counters + 2, 1);
    }
    __syncthreads();
  }
}


constexpr int kRowStreamSlots = 4;


// A slot owns one macro row and advances it in order. One monotone progress
// value per row therefore replaces a completion flag for every tile.
struct RowStreamState {
  int selected_slot;
  int next_selection;
  int row_ranks[kRowStreamSlots];
  int tile_columns[kRowStreamSlots];
};


__device__ __forceinline__ int32_t* row_progress_signal(
    int32_t* row_progress,
    int series,
    int tile_row,
    int macro_tile_count) {
  return row_progress + series * macro_tile_count + tile_row;
}


template <bool Reverse>
__device__ __forceinline__ void initialize_row_stream(
    RowStreamState& state,
    int series_count,
    int macro_tile_count) {
  if (threadIdx.x != 0) {
    return;
  }
  const int total_rows = series_count * macro_tile_count;
#pragma unroll
  for (int slot = 0; slot < kRowStreamSlots; ++slot) {
    const int row_rank = blockIdx.x + slot * gridDim.x;
    state.row_ranks[slot] = row_rank < total_rows ? row_rank : -1;
    if (row_rank < total_rows) {
      const int row = row_rank / series_count;
      state.tile_columns[slot] = Reverse ? macro_tile_count - 1 - row : 0;
    } else {
      state.tile_columns[slot] = 0;
    }
  }
  state.next_selection = 0;
}


template <bool Reverse>
__device__ __forceinline__ void poll_row_stream(
    RowStreamState& state,
    int32_t* __restrict__ row_progress,
    int series_count,
    int macro_tile_count) {
  if (threadIdx.x >= kWarpSize) {
    return;
  }
  const int lane = threadIdx.x;
  const int row_rank = lane < kRowStreamSlots ? state.row_ranks[lane] : -1;
  bool ready = false;
  if (row_rank >= 0) {
    const int row = row_rank / series_count;
    const int series = row_rank - row * series_count;
    const int tile_row = Reverse ? macro_tile_count - 1 - row : row;
    const int tile_col = state.tile_columns[lane];
    if constexpr (Reverse) {
      if (tile_row + 1 == macro_tile_count) {
        ready = true;
      } else {
        int32_t* signal = row_progress_signal(
            row_progress, series, tile_row + 1, macro_tile_count);
        cuda::atomic_ref<int32_t, cuda::thread_scope_device> flag(*signal);
        ready = flag.load(cuda::memory_order_relaxed) <= tile_col;
      }
    } else {
      if (tile_col == tile_row) {
        ready = true;
      } else {
        int32_t* signal = row_progress_signal(
            row_progress, series, tile_row - 1, macro_tile_count);
        cuda::atomic_ref<int32_t, cuda::thread_scope_device> flag(*signal);
        ready = flag.load(cuda::memory_order_relaxed) >= tile_col;
      }
    }
  }
  const uint32_t ready_mask = __ballot_sync(0xffffffffu, ready);
  if (lane != 0) {
    return;
  }
  if (ready_mask != 0) {
    int selected = -1;
#pragma unroll
    for (int offset = 0; offset < kRowStreamSlots; ++offset) {
      const int candidate =
          (state.next_selection + offset) & (kRowStreamSlots - 1);
      if (selected < 0 && (ready_mask & (1u << candidate)) != 0) {
        selected = candidate;
      }
    }
    state.selected_slot = selected;
    state.next_selection = (selected + 1) & (kRowStreamSlots - 1);
    return;
  }
  bool any_active = false;
#pragma unroll
  for (int slot = 0; slot < kRowStreamSlots; ++slot) {
    any_active |= state.row_ranks[slot] >= 0;
  }
  state.selected_slot = any_active ? -2 : -1;
}


template <bool Reverse>
__device__ __forceinline__ void row_stream_coordinates(
    const RowStreamState& state,
    int slot,
    int series_count,
    int macro_tile_count,
    int& series,
    int& tile_row,
    int& tile_col) {
  const int row_rank = state.row_ranks[slot];
  const int row = row_rank / series_count;
  series = row_rank - row * series_count;
  tile_row = Reverse ? macro_tile_count - 1 - row : row;
  tile_col = state.tile_columns[slot];
}


template <bool Reverse>
__device__ __forceinline__ void acquire_row_stream_predecessor(
    const RowStreamState& state,
    int slot,
    int32_t* __restrict__ row_progress,
    int series_count,
    int macro_tile_count) {
  int series = 0;
  int tile_row = 0;
  int tile_col = 0;
  row_stream_coordinates<Reverse>(
      state,
      slot,
      series_count,
      macro_tile_count,
      series,
      tile_row,
      tile_col);
  const bool has_external_predecessor = Reverse
      ? tile_row + 1 < macro_tile_count
      : tile_col < tile_row;
  if ((threadIdx.x & (kWarpSize - 1)) == 0 &&
      has_external_predecessor) {
    const int predecessor_row = tile_row + (Reverse ? 1 : -1);
    int32_t* signal = row_progress_signal(
        row_progress, series, predecessor_row, macro_tile_count);
    cuda::atomic_ref<int32_t, cuda::thread_scope_device> flag(*signal);
    flag.load(cuda::memory_order_acquire);
  }
  __syncwarp();
}


template <bool Reverse>
__device__ __forceinline__ void publish_row_stream_tile(
    RowStreamState& state,
    int slot,
    int32_t* __restrict__ row_progress,
    int series_count,
    int macro_tile_count) {
  if (threadIdx.x != 0) {
    return;
  }
  int series = 0;
  int tile_row = 0;
  int tile_col = 0;
  row_stream_coordinates<Reverse>(
      state,
      slot,
      series_count,
      macro_tile_count,
      series,
      tile_row,
      tile_col);
  int32_t* signal = row_progress_signal(
      row_progress, series, tile_row, macro_tile_count);
  cuda::atomic_ref<int32_t, cuda::thread_scope_device> flag(*signal);
  flag.store(tile_col, cuda::memory_order_release);
  const int next_col = tile_col + (Reverse ? -1 : 1);
  const bool row_done = Reverse ? next_col < 0 : next_col > tile_row;
  if (!row_done) {
    state.tile_columns[slot] = next_col;
    return;
  }
  const int replacement =
      state.row_ranks[slot] + kRowStreamSlots * gridDim.x;
  const int total_rows = series_count * macro_tile_count;
  state.row_ranks[slot] = replacement < total_rows ? replacement : -1;
  if (replacement < total_rows) {
    const int row = replacement / series_count;
    state.tile_columns[slot] = Reverse ? macro_tile_count - 1 - row : 0;
  }
}


template <int TileSize>
__global__ void row_stream_macro_wavefront_score_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ dense_scores,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int32_t* __restrict__ row_progress,
    int series_count,
    int seq_len,
    int symbol_dim,
    int macro_tile_count,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  __shared__ RowStreamState state;
  initialize_row_stream<false>(state, series_count, macro_tile_count);
  __syncthreads();

  while (true) {
    poll_row_stream<false>(
        state, row_progress, series_count, macro_tile_count);
    __syncthreads();
    const int slot = state.selected_slot;
    if (slot == -1) {
      return;
    }
    if (slot == -2) {
      if (threadIdx.x == 0) {
        scheduler_backoff();
      }
      __syncthreads();
      continue;
    }

    acquire_row_stream_predecessor<false>(
        state, slot, row_progress, series_count, macro_tile_count);
    int series = 0;
    int tile_row = 0;
    int tile_col = 0;
    row_stream_coordinates<false>(
        state,
        slot,
        series_count,
        macro_tile_count,
        series,
        tile_row,
        tile_col);
    process_score_macro_tile<TileSize>(
        packed_query,
        packed_key,
        dense_scores,
        diagonal_carry,
        checkpoints,
        series,
        tile_row,
        tile_col,
        seq_len,
        symbol_dim,
        macro_tile_count,
        mismatch_unit,
        shared_storage);
    __threadfence();
    __syncthreads();
    publish_row_stream_tile<false>(
        state, slot, row_progress, series_count, macro_tile_count);
    __syncthreads();
  }
}


__global__ void initialize_stats_kernel(
    float* __restrict__ row_stats,
    int64_t row_count,
    float scale) {
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < row_count;
       row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    row_stats[row * 3] = rosa_soft::cuda::kNullScore * scale;
    row_stats[row * 3 + 1] = 1.0f;
    row_stats[row * 3 + 2] = 0.0f;
  }
}


__global__ void finalize_stats_kernel(
    float* __restrict__ row_stats,
    int64_t row_count) {
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < row_count;
       row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    row_stats[row * 3 + 2] /= row_stats[row * 3 + 1];
  }
}


template <int TileSize, typename scalar_t>
__global__ void macro_wavefront_stats_wave_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    float* __restrict__ row_stats,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    int tile_wave,
    int first_tile_col,
    int tiles_on_wave,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  const int series_count = batch_size * num_heads;
  const int task = blockIdx.x;
  const int series = task / tiles_on_wave;
  const int local_tile = task - series * tiles_on_wave;
  if (series >= series_count) {
    return;
  }
  const int tile_col = first_tile_col + local_tile;
  const int tile_row = tile_wave - tile_col;
  process_stats_macro_tile<TileSize>(
      value,
      grad_output,
      packed_query,
      packed_key,
      dropout_seed,
      row_stats,
      nullptr,
      diagonal_carry,
      checkpoints,
      series,
      tile_row,
      tile_col,
      batch_size,
      seq_len,
      num_heads,
      symbol_dim,
      num_value_heads,
      value_dim,
      macro_tile_count,
      scale,
      dropout_p,
      inverse_keep_probability,
      mismatch_unit,
      shared_storage);
}


template <int TileSize, typename scalar_t>
__global__ void persistent_macro_wavefront_stats_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    float* __restrict__ row_stats,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  cg::grid_group grid = cg::this_grid();
  const int series_count = batch_size * num_heads;
  for (int tile_wave = 0;
       tile_wave < 2 * macro_tile_count - 1;
       ++tile_wave) {
    const int first_tile_col =
        first_column_on_wave(tile_wave, macro_tile_count);
    const int last_tile_col =
        last_causal_column_on_wave(tile_wave, macro_tile_count);
    const int tiles_on_wave = max(0, last_tile_col - first_tile_col + 1);
    const int task_count = series_count * tiles_on_wave;
    for (int task = blockIdx.x;
         task < task_count;
         task += gridDim.x) {
      const int series = task / tiles_on_wave;
      const int local_tile = task - series * tiles_on_wave;
      const int tile_col = first_tile_col + local_tile;
      const int tile_row = tile_wave - tile_col;
      process_stats_macro_tile<TileSize>(
          value,
          grad_output,
          packed_query,
          packed_key,
          dropout_seed,
          row_stats,
          nullptr,
          diagonal_carry,
          checkpoints,
          series,
          tile_row,
          tile_col,
          batch_size,
          seq_len,
          num_heads,
          symbol_dim,
          num_value_heads,
          value_dim,
          macro_tile_count,
          scale,
          dropout_p,
          inverse_keep_probability,
          mismatch_unit,
          shared_storage);
    }
    grid.sync();
  }
}


template <int TileSize, typename scalar_t>
__global__ void queued_macro_wavefront_stats_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    float* __restrict__ row_stats,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int32_t* __restrict__ dependencies,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_ready,
    int32_t* __restrict__ counters,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    int total_tasks,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  __shared__ int shared_task;
  const int square_tiles = macro_tile_count * macro_tile_count;
  while (true) {
    if (threadIdx.x == 0) {
      shared_task = dequeue_ready_task(
          queue, queue_ready, counters, total_tasks);
    }
    __syncthreads();
    const int task = shared_task;
    if (task < 0) {
      return;
    }
    const int series = task / square_tiles;
    const int local_tile = task - series * square_tiles;
    const int tile_row = local_tile / macro_tile_count;
    const int tile_col = local_tile - tile_row * macro_tile_count;
    process_stats_macro_tile<TileSize>(
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        row_stats,
        nullptr,
        diagonal_carry,
        checkpoints,
        series,
        tile_row,
        tile_col,
        batch_size,
        seq_len,
        num_heads,
        symbol_dim,
        num_value_heads,
        value_dim,
        macro_tile_count,
        scale,
        dropout_p,
        inverse_keep_probability,
        mismatch_unit,
        shared_storage);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      release_forward_successors(
          task,
          series,
          tile_row,
          tile_col,
          macro_tile_count,
          dependencies,
          queue,
          queue_ready,
          counters);
      atomicAdd(counters + 2, 1);
    }
    __syncthreads();
  }
}


template <int TileSize, typename scalar_t>
__global__ void row_stream_macro_wavefront_stats_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    float* __restrict__ row_stats,
    float* __restrict__ diagonal_carry,
    float* __restrict__ checkpoints,
    int32_t* __restrict__ row_progress,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  __shared__ RowStreamState state;
  const int series_count = batch_size * num_heads;
  initialize_row_stream<false>(state, series_count, macro_tile_count);
  __syncthreads();

  while (true) {
    poll_row_stream<false>(
        state, row_progress, series_count, macro_tile_count);
    __syncthreads();
    const int slot = state.selected_slot;
    if (slot == -1) {
      return;
    }
    if (slot == -2) {
      if (threadIdx.x == 0) {
        scheduler_backoff();
      }
      __syncthreads();
      continue;
    }

    acquire_row_stream_predecessor<false>(
        state, slot, row_progress, series_count, macro_tile_count);
    int series = 0;
    int tile_row = 0;
    int tile_col = 0;
    row_stream_coordinates<false>(
        state,
        slot,
        series_count,
        macro_tile_count,
        series,
        tile_row,
        tile_col);
    process_stats_macro_tile<TileSize>(
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        row_stats,
        nullptr,
        diagonal_carry,
        checkpoints,
        series,
        tile_row,
        tile_col,
        batch_size,
        seq_len,
        num_heads,
        symbol_dim,
        num_value_heads,
        value_dim,
        macro_tile_count,
        scale,
        dropout_p,
        inverse_keep_probability,
        mismatch_unit,
        shared_storage);
    __threadfence();
    __syncthreads();
    publish_row_stream_tile<false>(
        state, slot, row_progress, series_count, macro_tile_count);
    __syncthreads();
  }
}


template <int TileSize, typename scalar_t>
__global__ void macro_wavefront_reverse_wave_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ checkpoints,
    float* __restrict__ reverse_carry,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int gradient_mask,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    int tile_wave,
    int first_tile_col,
    int tiles_on_wave,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    float symbol_scale) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  const int series_count = batch_size * num_heads;
  const int task = blockIdx.x;
  const int series = task / tiles_on_wave;
  const int local_tile = task - series * tiles_on_wave;
  if (series >= series_count) {
    return;
  }
  const int tile_col = first_tile_col + local_tile;
  const int tile_row = tile_wave - tile_col;
  process_reverse_macro_tile<TileSize>(
      query,
      key,
      value,
      grad_output,
      packed_query,
      packed_key,
      dropout_seed,
      row_stats,
      checkpoints,
      reverse_carry,
      grad_query,
      grad_key,
      grad_value,
      gradient_mask,
      series,
      tile_row,
      tile_col,
      batch_size,
      seq_len,
      num_heads,
      symbol_dim,
      num_value_heads,
      value_dim,
      macro_tile_count,
      scale,
      dropout_p,
      inverse_keep_probability,
      mismatch_unit,
      symbol_scale,
      shared_storage);
}


template <int TileSize, typename scalar_t>
__global__ void persistent_macro_wavefront_reverse_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ checkpoints,
    float* __restrict__ reverse_carry,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int gradient_mask,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    float symbol_scale) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  cg::grid_group grid = cg::this_grid();
  const int series_count = batch_size * num_heads;
  for (int tile_wave = 2 * macro_tile_count - 2;
       tile_wave >= 0;
       --tile_wave) {
    const int first_tile_col =
        first_column_on_wave(tile_wave, macro_tile_count);
    const int last_tile_col =
        last_causal_column_on_wave(tile_wave, macro_tile_count);
    const int tiles_on_wave = max(0, last_tile_col - first_tile_col + 1);
    const int task_count = series_count * tiles_on_wave;
    for (int task = blockIdx.x;
         task < task_count;
         task += gridDim.x) {
      const int series = task / tiles_on_wave;
      const int local_tile = task - series * tiles_on_wave;
      const int tile_col = first_tile_col + local_tile;
      const int tile_row = tile_wave - tile_col;
      process_reverse_macro_tile<TileSize>(
          query,
          key,
          value,
          grad_output,
          packed_query,
          packed_key,
          dropout_seed,
          row_stats,
          checkpoints,
          reverse_carry,
          grad_query,
          grad_key,
          grad_value,
          gradient_mask,
          series,
          tile_row,
          tile_col,
          batch_size,
          seq_len,
          num_heads,
          symbol_dim,
          num_value_heads,
          value_dim,
          macro_tile_count,
          scale,
          dropout_p,
          inverse_keep_probability,
          mismatch_unit,
          symbol_scale,
          shared_storage);
    }
    grid.sync();
  }
}


template <int TileSize, typename scalar_t>
__global__ void queued_macro_wavefront_reverse_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ checkpoints,
    float* __restrict__ reverse_carry,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int32_t* __restrict__ dependencies,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_ready,
    int32_t* __restrict__ counters,
    int gradient_mask,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    int total_tasks,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    float symbol_scale) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  __shared__ int shared_task;
  const int square_tiles = macro_tile_count * macro_tile_count;
  while (true) {
    if (threadIdx.x == 0) {
      shared_task = dequeue_ready_task(
          queue, queue_ready, counters, total_tasks);
    }
    __syncthreads();
    const int task = shared_task;
    if (task < 0) {
      return;
    }
    const int series = task / square_tiles;
    const int local_tile = task - series * square_tiles;
    const int tile_row = local_tile / macro_tile_count;
    const int tile_col = local_tile - tile_row * macro_tile_count;
    process_reverse_macro_tile<TileSize>(
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        row_stats,
        checkpoints,
        reverse_carry,
        grad_query,
        grad_key,
        grad_value,
        gradient_mask,
        series,
        tile_row,
        tile_col,
        batch_size,
        seq_len,
        num_heads,
        symbol_dim,
        num_value_heads,
        value_dim,
        macro_tile_count,
        scale,
        dropout_p,
        inverse_keep_probability,
        mismatch_unit,
        symbol_scale,
        shared_storage);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      release_reverse_predecessors(
          task,
          series,
          tile_row,
          tile_col,
          macro_tile_count,
          dependencies,
          queue,
          queue_ready,
          counters);
      atomicAdd(counters + 2, 1);
    }
    __syncthreads();
  }
}


template <int TileSize, typename scalar_t>
__global__ void row_stream_macro_wavefront_reverse_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ checkpoints,
    float* __restrict__ reverse_carry,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int32_t* __restrict__ row_progress,
    int gradient_mask,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    float symbol_scale) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  __shared__ RowStreamState state;
  const int series_count = batch_size * num_heads;
  initialize_row_stream<true>(state, series_count, macro_tile_count);
  __syncthreads();

  while (true) {
    poll_row_stream<true>(
        state, row_progress, series_count, macro_tile_count);
    __syncthreads();
    const int slot = state.selected_slot;
    if (slot == -1) {
      return;
    }
    if (slot == -2) {
      if (threadIdx.x == 0) {
        scheduler_backoff();
      }
      __syncthreads();
      continue;
    }

    acquire_row_stream_predecessor<true>(
        state, slot, row_progress, series_count, macro_tile_count);
    int series = 0;
    int tile_row = 0;
    int tile_col = 0;
    row_stream_coordinates<true>(
        state,
        slot,
        series_count,
        macro_tile_count,
        series,
        tile_row,
        tile_col);
    process_reverse_macro_tile<TileSize>(
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        row_stats,
        checkpoints,
        reverse_carry,
        grad_query,
        grad_key,
        grad_value,
        gradient_mask,
        series,
        tile_row,
        tile_col,
        batch_size,
        seq_len,
        num_heads,
        symbol_dim,
        num_value_heads,
        value_dim,
        macro_tile_count,
        scale,
        dropout_p,
        inverse_keep_probability,
        mismatch_unit,
        symbol_scale,
        shared_storage);
    __threadfence();
    __syncthreads();
    publish_row_stream_tile<true>(
        state, slot, row_progress, series_count, macro_tile_count);
    __syncthreads();
  }
}


template <typename scalar_t>
__global__ void finalize_vjp_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int64_t query_items,
    int64_t key_items,
    int64_t value_items) {
  const int64_t total_items = query_items + key_items + value_items;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total_items;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (index < query_items) {
      grad_query[index] *= rosa_soft::cuda::softsign_derivative(
          rosa_soft::cuda::read_float(query, index));
    } else if (index < query_items + key_items) {
      const int64_t key_index = index - query_items;
      grad_key[key_index] *= rosa_soft::cuda::softsign_derivative(
          rosa_soft::cuda::read_float(key, key_index));
    } else {
      const int64_t value_index = index - query_items - key_items;
      grad_value[value_index] *= rosa_soft::cuda::softsign_derivative(
          rosa_soft::cuda::read_float(value, value_index));
    }
  }
}


struct ReadyQueueWorkspace {
  torch::Tensor dependencies;
  torch::Tensor queue;
  torch::Tensor queue_ready;
  torch::Tensor counters;
  int total_tasks;
};


ReadyQueueWorkspace make_ready_queue(
    const torch::TensorOptions& options,
    int series_count,
    int macro_tile_count,
    bool reverse,
    cudaStream_t stream) {
  const int64_t square_capacity =
      static_cast<int64_t>(series_count) * macro_tile_count *
      macro_tile_count;
  const int64_t total_tasks =
      static_cast<int64_t>(series_count) * macro_tile_count *
      (macro_tile_count + 1) / 2;
  TORCH_CHECK(
      square_capacity <= std::numeric_limits<int>::max() &&
          total_tasks <= std::numeric_limits<int>::max(),
      "macro-wavefront ready queue exceeds int32 indexing capacity");
  ReadyQueueWorkspace workspace{
      torch::empty({square_capacity}, options.dtype(torch::kInt32)),
      torch::empty({total_tasks}, options.dtype(torch::kInt32)),
      torch::zeros({total_tasks}, options.dtype(torch::kInt32)),
      torch::zeros({3}, options.dtype(torch::kInt32)),
      static_cast<int>(total_tasks)};
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      65535,
      (square_capacity + kThreads - 1) / kThreads));
  initialize_ready_queue_kernel<<<blocks, kThreads, 0, stream>>>(
      workspace.dependencies.data_ptr<int32_t>(),
      workspace.queue.data_ptr<int32_t>(),
      workspace.queue_ready.data_ptr<int32_t>(),
      workspace.counters.data_ptr<int32_t>(),
      series_count,
      macro_tile_count,
      reverse ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return workspace;
}


template <int TileSize>
int active_blocks_per_sm(int execution_plan) {
  constexpr int kThreads = (TileSize / kMicroTile) * kWarpSize;
  const size_t shared_bytes = 4 * TileSize * sizeof(uint32_t);
  int active_blocks = 0;
  if (execution_plan == 1) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        queued_macro_wavefront_score_kernel<TileSize>,
        kThreads,
        shared_bytes));
  } else if (execution_plan == 4) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        row_stream_macro_wavefront_score_kernel<TileSize>,
        kThreads,
        shared_bytes));
  } else if (execution_plan == 2) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        persistent_macro_wavefront_score_kernel<TileSize>,
        kThreads,
        shared_bytes));
  } else if (execution_plan == 3) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        folded_diagonal_checkpoint_kernel<TileSize>,
        kThreads,
        0));
  } else {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        macro_wavefront_score_wave_kernel<TileSize>,
        kThreads,
        shared_bytes));
  }
  return active_blocks;
}


template <int TileSize, typename scalar_t>
int active_stats_blocks_per_sm(int execution_plan) {
  constexpr int kThreads = (TileSize / kMicroTile) * kWarpSize;
  const size_t shared_bytes = stats_shared_bytes<TileSize>();
  int active_blocks = 0;
  if (execution_plan == 1) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        queued_macro_wavefront_stats_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  } else if (execution_plan == 4) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        row_stream_macro_wavefront_stats_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  } else if (execution_plan == 2) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        persistent_macro_wavefront_stats_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  } else if (execution_plan == 3) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        folded_tile_stats_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  } else {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        macro_wavefront_stats_wave_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  }
  return active_blocks;
}


template <int TileSize, typename scalar_t>
int active_reverse_blocks_per_sm(int execution_plan) {
  constexpr int kThreads = (TileSize / kMicroTile) * kWarpSize;
  const size_t shared_bytes = reverse_shared_bytes<TileSize>();
  int active_blocks = 0;
  if (execution_plan == 1) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        queued_macro_wavefront_reverse_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  } else if (execution_plan == 4) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        row_stream_macro_wavefront_reverse_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  } else if (execution_plan == 2) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        persistent_macro_wavefront_reverse_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  } else {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        macro_wavefront_reverse_wave_kernel<TileSize, scalar_t>,
        kThreads,
        shared_bytes));
  }
  return active_blocks;
}


template <int TileSize>
int launch_scores(
    const int32_t* packed_query,
    const int32_t* packed_key,
    float* dense_scores,
    float* diagonal_carry,
    float* checkpoints,
    int series_count,
    int seq_len,
    int symbol_dim,
    int macro_tile_count,
    float mismatch_unit,
    int execution_plan,
    const torch::TensorOptions& queue_options,
    const cudaDeviceProp* properties,
    cudaStream_t stream) {
  constexpr int kThreads = (TileSize / kMicroTile) * kWarpSize;
  const size_t shared_bytes = 4 * TileSize * sizeof(uint32_t);
  const int resident_per_sm = active_blocks_per_sm<TileSize>(execution_plan);
  TORCH_CHECK(
      resident_per_sm > 0,
      "macro-wavefront kernel has zero occupancy for tile_size=",
      TileSize);

  if (execution_plan == 0) {
    for (int wave = 0; wave < 2 * macro_tile_count - 1; ++wave) {
      const int first_col = first_column_on_wave(wave, macro_tile_count);
      const int last_col =
          last_causal_column_on_wave(wave, macro_tile_count);
      const int tile_count = max(0, last_col - first_col + 1);
      if (tile_count == 0) {
        continue;
      }
      macro_wavefront_score_wave_kernel<TileSize><<<
          series_count * tile_count,
          kThreads,
          shared_bytes,
          stream>>>(
          packed_query,
          packed_key,
          dense_scores,
          diagonal_carry,
          checkpoints,
          series_count,
          seq_len,
          symbol_dim,
          macro_tile_count,
          wave,
          first_col,
          tile_count,
          mismatch_unit);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return 0;
  }

  if (execution_plan == 3) {
    if (seq_len <= 1) {
      return 0;
    }
    const int persistent_blocks =
        properties->multiProcessorCount * resident_per_sm;
    folded_diagonal_checkpoint_kernel<TileSize><<<
        persistent_blocks,
        kThreads,
        0,
        stream>>>(
        packed_query,
        packed_key,
        dense_scores,
        checkpoints,
        series_count,
        seq_len,
        symbol_dim,
        macro_tile_count,
        mismatch_unit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return persistent_blocks;
  }

  TORCH_CHECK(
      properties->cooperativeLaunch,
      "persistent macro-wavefront requires cooperative launch support");
  const int resident_grid = properties->multiProcessorCount * resident_per_sm;
  const int persistent_blocks = resident_grid;

  if (execution_plan == 4) {
    const int64_t progress_slots =
        static_cast<int64_t>(series_count) * macro_tile_count;
    torch::Tensor row_progress = torch::full(
        {progress_slots}, -1, queue_options.dtype(torch::kInt32));
    const int32_t* query_pointer = packed_query;
    const int32_t* key_pointer = packed_key;
    float* score_pointer = dense_scores;
    float* carry_pointer = diagonal_carry;
    float* checkpoint_pointer = checkpoints;
    int32_t* progress_pointer = row_progress.data_ptr<int32_t>();
    int series_value = series_count;
    int length_value = seq_len;
    int dimension_value = symbol_dim;
    int tile_count_value = macro_tile_count;
    float mismatch_value = mismatch_unit;
    void* arguments[] = {
        &query_pointer,
        &key_pointer,
        &score_pointer,
        &carry_pointer,
        &checkpoint_pointer,
        &progress_pointer,
        &series_value,
        &length_value,
        &dimension_value,
        &tile_count_value,
        &mismatch_value};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            row_stream_macro_wavefront_score_kernel<TileSize>),
        dim3(persistent_blocks),
        dim3(kThreads),
        arguments,
        shared_bytes,
        stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return persistent_blocks;
  }

  if (execution_plan == 1) {
    auto scheduler = make_ready_queue(
        queue_options,
        series_count,
        macro_tile_count,
        false,
        stream);
    const int32_t* query_pointer = packed_query;
    const int32_t* key_pointer = packed_key;
    float* score_pointer = dense_scores;
    float* carry_pointer = diagonal_carry;
    float* checkpoint_pointer = checkpoints;
    int32_t* dependency_pointer =
        scheduler.dependencies.data_ptr<int32_t>();
    int32_t* queue_pointer = scheduler.queue.data_ptr<int32_t>();
    int32_t* ready_pointer = scheduler.queue_ready.data_ptr<int32_t>();
    int32_t* counter_pointer = scheduler.counters.data_ptr<int32_t>();
    int length_value = seq_len;
    int dimension_value = symbol_dim;
    int tile_count_value = macro_tile_count;
    int total_tasks_value = scheduler.total_tasks;
    float mismatch_value = mismatch_unit;
    void* arguments[] = {
        &query_pointer,
        &key_pointer,
        &score_pointer,
        &carry_pointer,
        &checkpoint_pointer,
        &dependency_pointer,
        &queue_pointer,
        &ready_pointer,
        &counter_pointer,
        &length_value,
        &dimension_value,
        &tile_count_value,
        &total_tasks_value,
        &mismatch_value};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            queued_macro_wavefront_score_kernel<TileSize>),
        dim3(persistent_blocks),
        dim3(kThreads),
        arguments,
        shared_bytes,
        stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return persistent_blocks;
  }

  const int32_t* query_pointer = packed_query;
  const int32_t* key_pointer = packed_key;
  float* score_pointer = dense_scores;
  float* carry_pointer = diagonal_carry;
  float* checkpoint_pointer = checkpoints;
  int series_value = series_count;
  int length_value = seq_len;
  int dimension_value = symbol_dim;
  int tile_count_value = macro_tile_count;
  float mismatch_value = mismatch_unit;
  void* arguments[] = {
      &query_pointer,
      &key_pointer,
      &score_pointer,
      &carry_pointer,
      &checkpoint_pointer,
      &series_value,
      &length_value,
      &dimension_value,
      &tile_count_value,
      &mismatch_value};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          persistent_macro_wavefront_score_kernel<TileSize>),
      dim3(persistent_blocks),
      dim3(kThreads),
      arguments,
      shared_bytes,
      stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return persistent_blocks;
}


template <int TileSize, typename scalar_t>
int launch_stats(
    const scalar_t* value,
    const scalar_t* grad_output,
    const int32_t* packed_query,
    const int32_t* packed_key,
    const int64_t* dropout_seed,
    float* row_stats,
    float* diagonal_carry,
    float* checkpoints,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float mismatch_unit,
    int execution_plan,
    const torch::TensorOptions& queue_options,
    const cudaDeviceProp* properties,
    cudaStream_t stream) {
  constexpr int kThreads = (TileSize / kMicroTile) * kWarpSize;
  const size_t shared_bytes = stats_shared_bytes<TileSize>();
  const int resident_per_sm =
      active_stats_blocks_per_sm<TileSize, scalar_t>(execution_plan);
  TORCH_CHECK(
      resident_per_sm > 0,
      "macro-wavefront stats kernel has zero occupancy for tile_size=",
      TileSize);
  const int series_count = batch_size * num_heads;
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);

  if (execution_plan == 0) {
    for (int wave = 0; wave < 2 * macro_tile_count - 1; ++wave) {
      const int first_col = first_column_on_wave(wave, macro_tile_count);
      const int last_col =
          last_causal_column_on_wave(wave, macro_tile_count);
      const int tile_count = max(0, last_col - first_col + 1);
      if (tile_count == 0) {
        continue;
      }
      macro_wavefront_stats_wave_kernel<TileSize, scalar_t><<<
          series_count * tile_count,
          kThreads,
          shared_bytes,
          stream>>>(
          value,
          grad_output,
          packed_query,
          packed_key,
          dropout_seed,
          row_stats,
          diagonal_carry,
          checkpoints,
          batch_size,
          seq_len,
          num_heads,
          symbol_dim,
          num_value_heads,
          value_dim,
          macro_tile_count,
          wave,
          first_col,
          tile_count,
          scale,
          dropout_p,
          inverse_keep_probability,
          mismatch_unit);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return 0;
  }

  if (execution_plan == 3) {
    if (seq_len > 1) {
      int checkpoint_resident_per_sm = 0;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &checkpoint_resident_per_sm,
          folded_diagonal_checkpoint_kernel<TileSize>,
          kThreads,
          0));
      const int checkpoint_blocks =
          properties->multiProcessorCount * checkpoint_resident_per_sm;
      folded_diagonal_checkpoint_kernel<TileSize><<<
          checkpoint_blocks,
          kThreads,
          0,
          stream>>>(
          packed_query,
          packed_key,
          nullptr,
          checkpoints,
          series_count,
          seq_len,
          symbol_dim,
          macro_tile_count,
          mismatch_unit);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int persistent_blocks =
        properties->multiProcessorCount * resident_per_sm;
    const int macro_tile_slots =
        macro_tile_count * (macro_tile_count + 1) / 2;
    torch::Tensor partial_stats = torch::empty(
        {series_count, macro_tile_slots, TileSize, 3},
        queue_options.dtype(torch::kFloat32));
    folded_tile_stats_kernel<TileSize, scalar_t><<<
        persistent_blocks,
        kThreads,
        shared_bytes,
        stream>>>(
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        row_stats,
        partial_stats.data_ptr<float>(),
        checkpoints,
        batch_size,
        seq_len,
        num_heads,
        symbol_dim,
        num_value_heads,
        value_dim,
        macro_tile_count,
        scale,
        dropout_p,
        inverse_keep_probability,
        mismatch_unit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    constexpr int kReductionThreads = 256;
    const int64_t total_rows =
        static_cast<int64_t>(series_count) * seq_len;
    const int reduction_blocks = static_cast<int>(std::min<int64_t>(
        65535,
        (total_rows + kReductionThreads - 1) / kReductionThreads));
    reduce_folded_tile_stats_kernel<TileSize><<<
        reduction_blocks,
        kReductionThreads,
        0,
        stream>>>(
        partial_stats.data_ptr<float>(),
        row_stats,
        series_count,
        seq_len,
        macro_tile_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return persistent_blocks;
  }

  TORCH_CHECK(
      properties->cooperativeLaunch,
      "persistent macro-wavefront requires cooperative launch support");
  const int resident_grid = properties->multiProcessorCount * resident_per_sm;
  const int persistent_blocks = resident_grid;

  if (execution_plan == 4) {
    torch::Tensor row_progress = torch::full(
        {static_cast<int64_t>(series_count) * macro_tile_count},
        -1,
        queue_options.dtype(torch::kInt32));
    const scalar_t* value_pointer = value;
    const scalar_t* grad_pointer = grad_output;
    const int32_t* query_pointer = packed_query;
    const int32_t* key_pointer = packed_key;
    const int64_t* seed_pointer = dropout_seed;
    float* stats_pointer = row_stats;
    float* carry_pointer = diagonal_carry;
    float* checkpoint_pointer = checkpoints;
    int32_t* progress_pointer = row_progress.data_ptr<int32_t>();
    int batch_value = batch_size;
    int length_value = seq_len;
    int heads_value = num_heads;
    int dimension_value = symbol_dim;
    int value_heads_value = num_value_heads;
    int value_dimension_value = value_dim;
    int tile_count_value = macro_tile_count;
    float scale_value = scale;
    float dropout_value = dropout_p;
    float inverse_keep_value = inverse_keep_probability;
    float mismatch_value = mismatch_unit;
    void* arguments[] = {
        &value_pointer,
        &grad_pointer,
        &query_pointer,
        &key_pointer,
        &seed_pointer,
        &stats_pointer,
        &carry_pointer,
        &checkpoint_pointer,
        &progress_pointer,
        &batch_value,
        &length_value,
        &heads_value,
        &dimension_value,
        &value_heads_value,
        &value_dimension_value,
        &tile_count_value,
        &scale_value,
        &dropout_value,
        &inverse_keep_value,
        &mismatch_value};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            row_stream_macro_wavefront_stats_kernel<TileSize, scalar_t>),
        dim3(persistent_blocks),
        dim3(kThreads),
        arguments,
        shared_bytes,
        stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return persistent_blocks;
  }

  if (execution_plan == 1) {
    auto scheduler = make_ready_queue(
        queue_options,
        series_count,
        macro_tile_count,
        false,
        stream);
    const scalar_t* value_pointer = value;
    const scalar_t* grad_pointer = grad_output;
    const int32_t* query_pointer = packed_query;
    const int32_t* key_pointer = packed_key;
    const int64_t* seed_pointer = dropout_seed;
    float* stats_pointer = row_stats;
    float* carry_pointer = diagonal_carry;
    float* checkpoint_pointer = checkpoints;
    int32_t* dependency_pointer =
        scheduler.dependencies.data_ptr<int32_t>();
    int32_t* queue_pointer = scheduler.queue.data_ptr<int32_t>();
    int32_t* ready_pointer = scheduler.queue_ready.data_ptr<int32_t>();
    int32_t* counter_pointer = scheduler.counters.data_ptr<int32_t>();
    int batch_value = batch_size;
    int length_value = seq_len;
    int heads_value = num_heads;
    int dimension_value = symbol_dim;
    int value_heads_value = num_value_heads;
    int value_dimension_value = value_dim;
    int tile_count_value = macro_tile_count;
    int total_tasks_value = scheduler.total_tasks;
    float scale_value = scale;
    float dropout_value = dropout_p;
    float inverse_keep_value = inverse_keep_probability;
    float mismatch_value = mismatch_unit;
    void* arguments[] = {
        &value_pointer,
        &grad_pointer,
        &query_pointer,
        &key_pointer,
        &seed_pointer,
        &stats_pointer,
        &carry_pointer,
        &checkpoint_pointer,
        &dependency_pointer,
        &queue_pointer,
        &ready_pointer,
        &counter_pointer,
        &batch_value,
        &length_value,
        &heads_value,
        &dimension_value,
        &value_heads_value,
        &value_dimension_value,
        &tile_count_value,
        &total_tasks_value,
        &scale_value,
        &dropout_value,
        &inverse_keep_value,
        &mismatch_value};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            queued_macro_wavefront_stats_kernel<TileSize, scalar_t>),
        dim3(persistent_blocks),
        dim3(kThreads),
        arguments,
        shared_bytes,
        stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return persistent_blocks;
  }

  const scalar_t* value_pointer = value;
  const scalar_t* grad_pointer = grad_output;
  const int32_t* query_pointer = packed_query;
  const int32_t* key_pointer = packed_key;
  const int64_t* seed_pointer = dropout_seed;
  float* stats_pointer = row_stats;
  float* carry_pointer = diagonal_carry;
  float* checkpoint_pointer = checkpoints;
  int batch_value = batch_size;
  int length_value = seq_len;
  int heads_value = num_heads;
  int dimension_value = symbol_dim;
  int value_heads_value = num_value_heads;
  int value_dimension_value = value_dim;
  int tile_count_value = macro_tile_count;
  float scale_value = scale;
  float dropout_value = dropout_p;
  float inverse_keep_value = inverse_keep_probability;
  float mismatch_value = mismatch_unit;
  void* arguments[] = {
      &value_pointer,
      &grad_pointer,
      &query_pointer,
      &key_pointer,
      &seed_pointer,
      &stats_pointer,
      &carry_pointer,
      &checkpoint_pointer,
      &batch_value,
      &length_value,
      &heads_value,
      &dimension_value,
      &value_heads_value,
      &value_dimension_value,
      &tile_count_value,
      &scale_value,
      &dropout_value,
      &inverse_keep_value,
      &mismatch_value};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          persistent_macro_wavefront_stats_kernel<TileSize, scalar_t>),
      dim3(persistent_blocks),
      dim3(kThreads),
      arguments,
      shared_bytes,
      stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return persistent_blocks;
}


template <int TileSize, typename scalar_t>
int launch_reverse(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const scalar_t* grad_output,
    const int32_t* packed_query,
    const int32_t* packed_key,
    const int64_t* dropout_seed,
    const float* row_stats,
    float* checkpoints,
    float* reverse_carry,
    float* grad_query,
    float* grad_key,
    float* grad_value,
    int gradient_mask,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int macro_tile_count,
    float scale,
    float dropout_p,
    float mismatch_unit,
    int execution_plan,
    const torch::TensorOptions& queue_options,
    const cudaDeviceProp* properties,
    cudaStream_t stream) {
  constexpr int kThreads = (TileSize / kMicroTile) * kWarpSize;
  const size_t shared_bytes = reverse_shared_bytes<TileSize>();
  const int resident_per_sm =
      active_reverse_blocks_per_sm<TileSize, scalar_t>(execution_plan);
  TORCH_CHECK(
      resident_per_sm > 0,
      "macro-wavefront reverse kernel has zero occupancy for tile_size=",
      TileSize);
  const int series_count = batch_size * num_heads;
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);
  const float symbol_scale =
      0.5f * mismatch_unit;

  if (execution_plan == 3) {
    int folded_blocks = 0;
    if ((gradient_mask &
         (rosa_soft::cuda::kGradQuery | rosa_soft::cuda::kGradKey)) != 0 &&
        seq_len > 1) {
      const size_t folded_shared_bytes =
          static_cast<size_t>(TileSize / kMicroTile) * TileSize *
          sizeof(float);
      int folded_resident_per_sm = 0;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &folded_resident_per_sm,
          folded_diagonal_reverse_kernel<TileSize, scalar_t>,
          kThreads,
          folded_shared_bytes));
      folded_blocks =
          properties->multiProcessorCount * folded_resident_per_sm;
      folded_diagonal_reverse_kernel<TileSize, scalar_t><<<
          folded_blocks,
          kThreads,
          folded_shared_bytes,
          stream>>>(
          value,
          grad_output,
          packed_query,
          packed_key,
          dropout_seed,
          row_stats,
          checkpoints,
          grad_query,
          grad_key,
          gradient_mask,
          batch_size,
          seq_len,
          num_heads,
          symbol_dim,
          num_value_heads,
          value_dim,
          macro_tile_count,
          scale,
          dropout_p,
          inverse_keep_probability,
          mismatch_unit,
          symbol_scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if ((gradient_mask & rosa_soft::cuda::kGradValue) != 0) {
      int value_resident_per_sm = 0;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &value_resident_per_sm,
          folded_tile_value_grad_kernel<TileSize, scalar_t>,
          kThreads,
          shared_bytes));
      const int value_blocks =
          properties->multiProcessorCount * value_resident_per_sm;
      folded_tile_value_grad_kernel<TileSize, scalar_t><<<
          value_blocks,
          kThreads,
          shared_bytes,
          stream>>>(
          query,
          key,
          value,
          grad_output,
          packed_query,
          packed_key,
          dropout_seed,
          row_stats,
          checkpoints,
          reverse_carry,
          grad_value,
          batch_size,
          seq_len,
          num_heads,
          symbol_dim,
          num_value_heads,
          value_dim,
          macro_tile_count,
          scale,
          dropout_p,
          inverse_keep_probability,
          mismatch_unit,
          symbol_scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      folded_blocks = max(folded_blocks, value_blocks);
    }
    return folded_blocks;
  }

  if (execution_plan == 0) {
    for (int wave = 2 * macro_tile_count - 2; wave >= 0; --wave) {
      const int first_col = first_column_on_wave(wave, macro_tile_count);
      const int last_col =
          last_causal_column_on_wave(wave, macro_tile_count);
      const int tile_count = max(0, last_col - first_col + 1);
      if (tile_count == 0) {
        continue;
      }
      macro_wavefront_reverse_wave_kernel<TileSize, scalar_t><<<
          series_count * tile_count,
          kThreads,
          shared_bytes,
          stream>>>(
          query,
          key,
          value,
          grad_output,
          packed_query,
          packed_key,
          dropout_seed,
          row_stats,
          checkpoints,
          reverse_carry,
          grad_query,
          grad_key,
          grad_value,
          gradient_mask,
          batch_size,
          seq_len,
          num_heads,
          symbol_dim,
          num_value_heads,
          value_dim,
          macro_tile_count,
          wave,
          first_col,
          tile_count,
          scale,
          dropout_p,
          inverse_keep_probability,
          mismatch_unit,
          symbol_scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return 0;
  }

  TORCH_CHECK(
      properties->cooperativeLaunch,
      "persistent macro-wavefront requires cooperative launch support");
  const int resident_grid = properties->multiProcessorCount * resident_per_sm;
  const int persistent_blocks = resident_grid;

  if (execution_plan == 4) {
    torch::Tensor row_progress = torch::full(
        {static_cast<int64_t>(series_count) * macro_tile_count},
        macro_tile_count,
        queue_options.dtype(torch::kInt32));
    const scalar_t* query_pointer = query;
    const scalar_t* key_pointer = key;
    const scalar_t* value_pointer = value;
    const scalar_t* grad_pointer = grad_output;
    const int32_t* packed_query_pointer = packed_query;
    const int32_t* packed_key_pointer = packed_key;
    const int64_t* seed_pointer = dropout_seed;
    const float* stats_pointer = row_stats;
    float* checkpoint_pointer = checkpoints;
    float* carry_pointer = reverse_carry;
    float* grad_query_pointer = grad_query;
    float* grad_key_pointer = grad_key;
    float* grad_value_pointer = grad_value;
    int32_t* progress_pointer = row_progress.data_ptr<int32_t>();
    int gradient_mask_value = gradient_mask;
    int batch_value = batch_size;
    int length_value = seq_len;
    int heads_value = num_heads;
    int dimension_value = symbol_dim;
    int value_heads_value = num_value_heads;
    int value_dimension_value = value_dim;
    int tile_count_value = macro_tile_count;
    float scale_value = scale;
    float dropout_value = dropout_p;
    float inverse_keep_value = inverse_keep_probability;
    float mismatch_value = mismatch_unit;
    float symbol_value = symbol_scale;
    void* arguments[] = {
        &query_pointer,
        &key_pointer,
        &value_pointer,
        &grad_pointer,
        &packed_query_pointer,
        &packed_key_pointer,
        &seed_pointer,
        &stats_pointer,
        &checkpoint_pointer,
        &carry_pointer,
        &grad_query_pointer,
        &grad_key_pointer,
        &grad_value_pointer,
        &progress_pointer,
        &gradient_mask_value,
        &batch_value,
        &length_value,
        &heads_value,
        &dimension_value,
        &value_heads_value,
        &value_dimension_value,
        &tile_count_value,
        &scale_value,
        &dropout_value,
        &inverse_keep_value,
        &mismatch_value,
        &symbol_value};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            row_stream_macro_wavefront_reverse_kernel<TileSize, scalar_t>),
        dim3(persistent_blocks),
        dim3(kThreads),
        arguments,
        shared_bytes,
        stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return persistent_blocks;
  }

  if (execution_plan == 1) {
    auto scheduler = make_ready_queue(
        queue_options,
        series_count,
        macro_tile_count,
        true,
        stream);
    const scalar_t* query_pointer = query;
    const scalar_t* key_pointer = key;
    const scalar_t* value_pointer = value;
    const scalar_t* grad_pointer = grad_output;
    const int32_t* packed_query_pointer = packed_query;
    const int32_t* packed_key_pointer = packed_key;
    const int64_t* seed_pointer = dropout_seed;
    const float* stats_pointer = row_stats;
    float* checkpoint_pointer = checkpoints;
    float* carry_pointer = reverse_carry;
    float* grad_query_pointer = grad_query;
    float* grad_key_pointer = grad_key;
    float* grad_value_pointer = grad_value;
    int32_t* dependency_pointer =
        scheduler.dependencies.data_ptr<int32_t>();
    int32_t* queue_pointer = scheduler.queue.data_ptr<int32_t>();
    int32_t* ready_pointer = scheduler.queue_ready.data_ptr<int32_t>();
    int32_t* counter_pointer = scheduler.counters.data_ptr<int32_t>();
    int gradient_mask_value = gradient_mask;
    int batch_value = batch_size;
    int length_value = seq_len;
    int heads_value = num_heads;
    int dimension_value = symbol_dim;
    int value_heads_value = num_value_heads;
    int value_dimension_value = value_dim;
    int tile_count_value = macro_tile_count;
    int total_tasks_value = scheduler.total_tasks;
    float scale_value = scale;
    float dropout_value = dropout_p;
    float inverse_keep_value = inverse_keep_probability;
    float mismatch_value = mismatch_unit;
    float symbol_value = symbol_scale;
    void* arguments[] = {
        &query_pointer,
        &key_pointer,
        &value_pointer,
        &grad_pointer,
        &packed_query_pointer,
        &packed_key_pointer,
        &seed_pointer,
        &stats_pointer,
        &checkpoint_pointer,
        &carry_pointer,
        &grad_query_pointer,
        &grad_key_pointer,
        &grad_value_pointer,
        &dependency_pointer,
        &queue_pointer,
        &ready_pointer,
        &counter_pointer,
        &gradient_mask_value,
        &batch_value,
        &length_value,
        &heads_value,
        &dimension_value,
        &value_heads_value,
        &value_dimension_value,
        &tile_count_value,
        &total_tasks_value,
        &scale_value,
        &dropout_value,
        &inverse_keep_value,
        &mismatch_value,
        &symbol_value};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            queued_macro_wavefront_reverse_kernel<TileSize, scalar_t>),
        dim3(persistent_blocks),
        dim3(kThreads),
        arguments,
        shared_bytes,
        stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return persistent_blocks;
  }

  const scalar_t* query_pointer = query;
  const scalar_t* key_pointer = key;
  const scalar_t* value_pointer = value;
  const scalar_t* grad_pointer = grad_output;
  const int32_t* packed_query_pointer = packed_query;
  const int32_t* packed_key_pointer = packed_key;
  const int64_t* seed_pointer = dropout_seed;
  const float* stats_pointer = row_stats;
  float* checkpoint_pointer = checkpoints;
  float* carry_pointer = reverse_carry;
  float* grad_query_pointer = grad_query;
  float* grad_key_pointer = grad_key;
  float* grad_value_pointer = grad_value;
  int gradient_mask_value = gradient_mask;
  int batch_value = batch_size;
  int length_value = seq_len;
  int heads_value = num_heads;
  int dimension_value = symbol_dim;
  int value_heads_value = num_value_heads;
  int value_dimension_value = value_dim;
  int tile_count_value = macro_tile_count;
  float scale_value = scale;
  float dropout_value = dropout_p;
  float inverse_keep_value = inverse_keep_probability;
  float mismatch_value = mismatch_unit;
  float symbol_value = symbol_scale;
  void* arguments[] = {
      &query_pointer,
      &key_pointer,
      &value_pointer,
      &grad_pointer,
      &packed_query_pointer,
      &packed_key_pointer,
      &seed_pointer,
      &stats_pointer,
      &checkpoint_pointer,
      &carry_pointer,
      &grad_query_pointer,
      &grad_key_pointer,
      &grad_value_pointer,
      &gradient_mask_value,
      &batch_value,
      &length_value,
      &heads_value,
      &dimension_value,
      &value_heads_value,
      &value_dimension_value,
      &tile_count_value,
      &scale_value,
      &dropout_value,
      &inverse_keep_value,
      &mismatch_value,
      &symbol_value};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          persistent_macro_wavefront_reverse_kernel<TileSize, scalar_t>),
      dim3(persistent_blocks),
      dim3(kThreads),
      arguments,
      shared_bytes,
      stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return persistent_blocks;
}


int choose_tile_size(
    int seq_len,
    int requested_tile_size,
    int execution_plan) {
  if (requested_tile_size != 0) {
    return requested_tile_size;
  }
  if (execution_plan == 3) {
    return 32;
  }
  if (seq_len <= 64) {
    return 32;
  }
  if (seq_len <= 256) {
    return 64;
  }
  return 128;
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_macro_wavefront_scores_cuda(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t requested_tile_size,
    float mismatch_scale,
    int execution_plan) {
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int series_count = batch_size * num_heads;
  const int tile_size = choose_tile_size(
      seq_len, static_cast<int>(requested_tile_size), execution_plan);
  const int macro_tile_count = (seq_len + tile_size - 1) / tile_size;
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;
  const auto float_options =
      packed_query_symbols.options().dtype(torch::kFloat32);
  torch::Tensor dense_scores = torch::zeros(
      {batch_size, num_heads, seq_len, seq_len},
      float_options);
  torch::Tensor diagonal_carry = torch::zeros(
      {series_count, seq_len},
      float_options);
  torch::Tensor checkpoints = torch::zeros(
      {batch_size,
       num_heads,
       macro_tile_slots,
       2 * tile_size},
      float_options);
  const cudaDeviceProp* properties =
      at::cuda::getCurrentDeviceProperties();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  int persistent_blocks = 0;
  int resident_per_sm = 0;

  switch (tile_size) {
    case 32:
      resident_per_sm = active_blocks_per_sm<32>(execution_plan);
      persistent_blocks = launch_scores<32>(
          packed_query_symbols.data_ptr<int32_t>(),
          packed_key_symbols.data_ptr<int32_t>(),
          dense_scores.data_ptr<float>(),
          diagonal_carry.data_ptr<float>(),
          checkpoints.data_ptr<float>(),
          series_count,
          seq_len,
          static_cast<int>(symbol_dim),
          macro_tile_count,
          mismatch_unit,
          execution_plan,
          packed_query_symbols.options(),
          properties,
          stream);
      break;
    case 64:
      resident_per_sm = active_blocks_per_sm<64>(execution_plan);
      persistent_blocks = launch_scores<64>(
          packed_query_symbols.data_ptr<int32_t>(),
          packed_key_symbols.data_ptr<int32_t>(),
          dense_scores.data_ptr<float>(),
          diagonal_carry.data_ptr<float>(),
          checkpoints.data_ptr<float>(),
          series_count,
          seq_len,
          static_cast<int>(symbol_dim),
          macro_tile_count,
          mismatch_unit,
          execution_plan,
          packed_query_symbols.options(),
          properties,
          stream);
      break;
    case 96:
      resident_per_sm = active_blocks_per_sm<96>(execution_plan);
      persistent_blocks = launch_scores<96>(
          packed_query_symbols.data_ptr<int32_t>(),
          packed_key_symbols.data_ptr<int32_t>(),
          dense_scores.data_ptr<float>(),
          diagonal_carry.data_ptr<float>(),
          checkpoints.data_ptr<float>(),
          series_count,
          seq_len,
          static_cast<int>(symbol_dim),
          macro_tile_count,
          mismatch_unit,
          execution_plan,
          packed_query_symbols.options(),
          properties,
          stream);
      break;
    case 128:
      resident_per_sm = active_blocks_per_sm<128>(execution_plan);
      persistent_blocks = launch_scores<128>(
          packed_query_symbols.data_ptr<int32_t>(),
          packed_key_symbols.data_ptr<int32_t>(),
          dense_scores.data_ptr<float>(),
          diagonal_carry.data_ptr<float>(),
          checkpoints.data_ptr<float>(),
          series_count,
          seq_len,
          static_cast<int>(symbol_dim),
          macro_tile_count,
          mismatch_unit,
          execution_plan,
          packed_query_symbols.options(),
          properties,
          stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported tile_size");
  }

  torch::Tensor launch_info = torch::tensor(
      {static_cast<int64_t>(tile_size),
       static_cast<int64_t>(persistent_blocks),
       static_cast<int64_t>(resident_per_sm),
       static_cast<int64_t>(properties->multiProcessorCount),
       static_cast<int64_t>(4 * tile_size * sizeof(uint32_t)),
       static_cast<int64_t>(macro_tile_slots)},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  return std::make_tuple(dense_scores, checkpoints, launch_info);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_macro_wavefront_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t requested_tile_size,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int execution_plan) {
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int num_value_heads = value.size(2);
  const int value_dim = value.size(3);
  const int series_count = batch_size * num_heads;
  const int tile_size = choose_tile_size(
      seq_len, static_cast<int>(requested_tile_size), execution_plan);
  const int macro_tile_count = (seq_len + tile_size - 1) / tile_size;
  const int macro_tile_slots =
      macro_tile_count * (macro_tile_count + 1) / 2;
  const auto float_options = value.options().dtype(torch::kFloat32);
  torch::Tensor row_stats = torch::empty(
      {batch_size, num_heads, seq_len, 3},
      float_options);
  torch::Tensor diagonal_carry = torch::zeros(
      {series_count, seq_len},
      float_options);
  torch::Tensor checkpoints = torch::zeros(
      {batch_size,
       num_heads,
       macro_tile_slots,
       2 * tile_size},
      float_options);
  const cudaDeviceProp* properties =
      at::cuda::getCurrentDeviceProperties();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  constexpr int kLinearThreads = 256;
  const int64_t row_count =
      static_cast<int64_t>(series_count) * seq_len;
  const int linear_blocks = static_cast<int>(std::min<int64_t>(
      65535,
      (row_count + kLinearThreads - 1) / kLinearThreads));
  initialize_stats_kernel<<<
      linear_blocks,
      kLinearThreads,
      0,
      stream>>>(row_stats.data_ptr<float>(), row_count, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int persistent_blocks = 0;
  int resident_per_sm = 0;
  size_t shared_bytes = 0;
  DISPATCH_ROSA_FLOAT_TYPES(
      value.scalar_type(),
      "rosa_soft_macro_wavefront_stats_cuda",
      [&] {
        switch (tile_size) {
          case 32:
            resident_per_sm =
                active_stats_blocks_per_sm<32, scalar_t>(execution_plan);
            shared_bytes = stats_shared_bytes<32>();
            persistent_blocks = launch_stats<32, scalar_t>(
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                dropout_seed.data_ptr<int64_t>(),
                row_stats.data_ptr<float>(),
                diagonal_carry.data_ptr<float>(),
                checkpoints.data_ptr<float>(),
                batch_size,
                seq_len,
                num_heads,
                static_cast<int>(symbol_dim),
                num_value_heads,
                value_dim,
                macro_tile_count,
                scale,
                dropout_p,
                mismatch_unit,
                execution_plan,
                packed_query_symbols.options(),
                properties,
                stream);
            break;
          case 64:
            resident_per_sm =
                active_stats_blocks_per_sm<64, scalar_t>(execution_plan);
            shared_bytes = stats_shared_bytes<64>();
            persistent_blocks = launch_stats<64, scalar_t>(
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                dropout_seed.data_ptr<int64_t>(),
                row_stats.data_ptr<float>(),
                diagonal_carry.data_ptr<float>(),
                checkpoints.data_ptr<float>(),
                batch_size,
                seq_len,
                num_heads,
                static_cast<int>(symbol_dim),
                num_value_heads,
                value_dim,
                macro_tile_count,
                scale,
                dropout_p,
                mismatch_unit,
                execution_plan,
                packed_query_symbols.options(),
                properties,
                stream);
            break;
          case 96:
            resident_per_sm =
                active_stats_blocks_per_sm<96, scalar_t>(execution_plan);
            shared_bytes = stats_shared_bytes<96>();
            persistent_blocks = launch_stats<96, scalar_t>(
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                dropout_seed.data_ptr<int64_t>(),
                row_stats.data_ptr<float>(),
                diagonal_carry.data_ptr<float>(),
                checkpoints.data_ptr<float>(),
                batch_size,
                seq_len,
                num_heads,
                static_cast<int>(symbol_dim),
                num_value_heads,
                value_dim,
                macro_tile_count,
                scale,
                dropout_p,
                mismatch_unit,
                execution_plan,
                packed_query_symbols.options(),
                properties,
                stream);
            break;
          case 128:
            resident_per_sm =
                active_stats_blocks_per_sm<128, scalar_t>(execution_plan);
            shared_bytes = stats_shared_bytes<128>();
            persistent_blocks = launch_stats<128, scalar_t>(
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                dropout_seed.data_ptr<int64_t>(),
                row_stats.data_ptr<float>(),
                diagonal_carry.data_ptr<float>(),
                checkpoints.data_ptr<float>(),
                batch_size,
                seq_len,
                num_heads,
                static_cast<int>(symbol_dim),
                num_value_heads,
                value_dim,
                macro_tile_count,
                scale,
                dropout_p,
                mismatch_unit,
                execution_plan,
                packed_query_symbols.options(),
                properties,
                stream);
            break;
          default:
            TORCH_CHECK(false, "unsupported tile_size");
        }
      });

  finalize_stats_kernel<<<
      linear_blocks,
      kLinearThreads,
      0,
      stream>>>(row_stats.data_ptr<float>(), row_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  torch::Tensor launch_info = torch::tensor(
      {static_cast<int64_t>(tile_size),
       static_cast<int64_t>(persistent_blocks),
       static_cast<int64_t>(resident_per_sm),
       static_cast<int64_t>(properties->multiProcessorCount),
       static_cast<int64_t>(shared_bytes),
       static_cast<int64_t>(macro_tile_slots)},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  return std::make_tuple(row_stats, checkpoints, launch_info);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_macro_wavefront_vjp_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t requested_tile_size,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask,
    int execution_plan) {
  const int batch_size = query.size(0);
  const int seq_len = query.size(1);
  const int num_heads = query.size(2);
  const int symbol_dim = query.size(3);
  const int num_value_heads = value.size(2);
  const int value_dim = value.size(3);
  const int series_count = batch_size * num_heads;
  const int tile_size = choose_tile_size(
      seq_len, static_cast<int>(requested_tile_size), execution_plan);
  const int macro_tile_count = (seq_len + tile_size - 1) / tile_size;
  auto stats_result = rosa_soft_macro_wavefront_stats_cuda(
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      symbol_dim,
      tile_size,
      scale,
      dropout_p,
      mismatch_scale,
      execution_plan);
  torch::Tensor row_stats = std::get<0>(stats_result);
  torch::Tensor checkpoints = std::get<1>(stats_result);
  const auto float_options = query.options().dtype(torch::kFloat32);
  torch::Tensor grad_query =
      (gradient_mask & rosa_soft::cuda::kGradQuery) != 0
      ? torch::zeros(query.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_key =
      (gradient_mask & rosa_soft::cuda::kGradKey) != 0
      ? torch::zeros(key.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_value =
      (gradient_mask & rosa_soft::cuda::kGradValue) != 0
      ? torch::zeros(value.sizes(), float_options)
      : torch::empty({0}, float_options);
  if (seq_len <= 1) {
    return std::make_tuple(grad_query, grad_key, grad_value);
  }
  torch::Tensor reverse_carry = torch::zeros(
      {series_count, seq_len},
      float_options);
  const cudaDeviceProp* properties =
      at::cuda::getCurrentDeviceProperties();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_macro_wavefront_vjp_cuda",
      [&] {
        switch (tile_size) {
          case 32:
            launch_reverse<32, scalar_t>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                dropout_seed.data_ptr<int64_t>(),
                row_stats.data_ptr<float>(),
                checkpoints.data_ptr<float>(),
                reverse_carry.data_ptr<float>(),
                grad_query.numel() != 0 ? grad_query.data_ptr<float>() : nullptr,
                grad_key.numel() != 0 ? grad_key.data_ptr<float>() : nullptr,
                grad_value.numel() != 0 ? grad_value.data_ptr<float>() : nullptr,
                gradient_mask,
                batch_size,
                seq_len,
                num_heads,
                symbol_dim,
                num_value_heads,
                value_dim,
                macro_tile_count,
                scale,
                dropout_p,
                mismatch_unit,
                execution_plan,
                packed_query_symbols.options(),
                properties,
                stream);
            break;
          case 64:
            launch_reverse<64, scalar_t>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                dropout_seed.data_ptr<int64_t>(),
                row_stats.data_ptr<float>(),
                checkpoints.data_ptr<float>(),
                reverse_carry.data_ptr<float>(),
                grad_query.numel() != 0 ? grad_query.data_ptr<float>() : nullptr,
                grad_key.numel() != 0 ? grad_key.data_ptr<float>() : nullptr,
                grad_value.numel() != 0 ? grad_value.data_ptr<float>() : nullptr,
                gradient_mask,
                batch_size,
                seq_len,
                num_heads,
                symbol_dim,
                num_value_heads,
                value_dim,
                macro_tile_count,
                scale,
                dropout_p,
                mismatch_unit,
                execution_plan,
                packed_query_symbols.options(),
                properties,
                stream);
            break;
          case 96:
            launch_reverse<96, scalar_t>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                dropout_seed.data_ptr<int64_t>(),
                row_stats.data_ptr<float>(),
                checkpoints.data_ptr<float>(),
                reverse_carry.data_ptr<float>(),
                grad_query.numel() != 0 ? grad_query.data_ptr<float>() : nullptr,
                grad_key.numel() != 0 ? grad_key.data_ptr<float>() : nullptr,
                grad_value.numel() != 0 ? grad_value.data_ptr<float>() : nullptr,
                gradient_mask,
                batch_size,
                seq_len,
                num_heads,
                symbol_dim,
                num_value_heads,
                value_dim,
                macro_tile_count,
                scale,
                dropout_p,
                mismatch_unit,
                execution_plan,
                packed_query_symbols.options(),
                properties,
                stream);
            break;
          case 128:
            launch_reverse<128, scalar_t>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                dropout_seed.data_ptr<int64_t>(),
                row_stats.data_ptr<float>(),
                checkpoints.data_ptr<float>(),
                reverse_carry.data_ptr<float>(),
                grad_query.numel() != 0 ? grad_query.data_ptr<float>() : nullptr,
                grad_key.numel() != 0 ? grad_key.data_ptr<float>() : nullptr,
                grad_value.numel() != 0 ? grad_value.data_ptr<float>() : nullptr,
                gradient_mask,
                batch_size,
                seq_len,
                num_heads,
                symbol_dim,
                num_value_heads,
                value_dim,
                macro_tile_count,
                scale,
                dropout_p,
                mismatch_unit,
                execution_plan,
                packed_query_symbols.options(),
                properties,
                stream);
            break;
          default:
            TORCH_CHECK(false, "unsupported tile_size");
        }

        constexpr int kFinalizeThreads = 256;
        const int64_t query_items = grad_query.numel();
        const int64_t key_items = grad_key.numel();
        const int64_t value_items = grad_value.numel();
        const int64_t total_items = query_items + key_items + value_items;
        if (total_items != 0) {
          const int blocks = static_cast<int>(std::min<int64_t>(
              65535,
              (total_items + kFinalizeThreads - 1) / kFinalizeThreads));
          finalize_vjp_kernel<scalar_t><<<
              blocks,
              kFinalizeThreads,
              0,
              stream>>>(
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>(),
              grad_query.numel() != 0 ? grad_query.data_ptr<float>() : nullptr,
              grad_key.numel() != 0 ? grad_key.data_ptr<float>() : nullptr,
              grad_value.numel() != 0 ? grad_value.data_ptr<float>() : nullptr,
              query_items,
              key_items,
              value_items);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(grad_query, grad_key, grad_value);
}
