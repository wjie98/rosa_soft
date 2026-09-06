#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>
#include <mma.h>
#include <tuple>

#include "rosa_soft_vjp_common.cuh"


using namespace rosa_soft::cuda;


namespace {

namespace wmma = nvcuda::wmma;

// Slabs partition independent q-k diagonal chains into contiguous ranges.
// Capacity is private and bounded: it changes storage granularity, never
// suffix horizon or candidate coverage.
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kWorkspaceRowTile = kWarpSize;
constexpr int kWorkspaceDiagonalTile = kWarpsPerBlock;
constexpr int kLinearThreads = 256;
constexpr int kValueOwnerThreads = 128;
constexpr int kRowStatsWidth = 3;
constexpr int kRowMaximum = 0;
constexpr int kRowNormalizer = 1;
constexpr int kRowUtility = 2;
constexpr int kTensorTile = 16;
constexpr int kFusedRows = 32;
constexpr int kFusedDiagonals = 32;
constexpr int kFusedScoreStride = kFusedDiagonals + 1;
constexpr int kFusedRoutes = 64;
constexpr int kFusedValueDim = 64;
constexpr int kFusedThreads = 8 * kWarpSize;
constexpr int kValueRouteTile = 32;
constexpr int kMismatchGateCount = 33;

enum ReplayPlan : int {
  kPlanFusedStats = 1 << 0,
  kPlanFusedReverse = 1 << 1,
  kPlanTensorValue = 1 << 2,
  kPlanTiledSymbols = 1 << 3,
};


__device__ __forceinline__ int64_t workspace_index(
    int series,
    int row,
    int slot,
    int seq_len,
    int workspace_stride) {
  const int row_tiles =
      (seq_len + kWorkspaceRowTile - 1) / kWorkspaceRowTile;
  const int diagonal_tiles =
      (workspace_stride + kWorkspaceDiagonalTile - 1) /
      kWorkspaceDiagonalTile;
  const int row_tile = row / kWorkspaceRowTile;
  const int row_lane = row & (kWorkspaceRowTile - 1);
  const int diagonal_tile = slot / kWorkspaceDiagonalTile;
  const int diagonal_lane = slot & (kWorkspaceDiagonalTile - 1);
  return (((static_cast<int64_t>(series) * diagonal_tiles + diagonal_tile) *
               row_tiles +
           row_tile) *
              kWorkspaceRowTile +
          row_lane) *
      kWorkspaceDiagonalTile + diagonal_lane;
}


__device__ __forceinline__ uint32_t symbol_mask(int symbol_dim) {
  return symbol_dim == 32
      ? 0xffffffffu
      : (1u << symbol_dim) - 1u;
}


__device__ __forceinline__ __half2 binary_sign_half2(__half2 values) {
  const __half2 positive = __hgt2(values, __float2half2_rn(0.0f));
  return __hfma2(
      positive,
      __float2half2_rn(2.0f),
      __float2half2_rn(-1.0f));
}


__device__ __forceinline__ void warp_forward_affine_scan(
    float& coefficient,
    float& bias) {
  const int lane = threadIdx.x & (kWarpSize - 1);
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const float left_coefficient = __shfl_up_sync(
        0xffffffffu, coefficient, offset);
    const float left_bias = __shfl_up_sync(
        0xffffffffu, bias, offset);
    if (lane >= offset) {
      bias = fmaf(coefficient, left_bias, bias);
      coefficient *= left_coefficient;
    }
  }
}


__device__ __forceinline__ void warp_reverse_affine_scan(
    float& coefficient,
    float& bias,
    int active_count) {
  const int lane = threadIdx.x & (kWarpSize - 1);
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const float right_coefficient = __shfl_down_sync(
        0xffffffffu, coefficient, offset);
    const float right_bias = __shfl_down_sync(
        0xffffffffu, bias, offset);
    if (lane + offset < active_count) {
      bias = fmaf(coefficient, right_bias, bias);
      coefficient *= right_coefficient;
    }
  }
}


__global__ void initialize_stats_kernel(
    float* __restrict__ row_stats,
    int64_t row_count,
    float scale) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < row_count) {
    const int64_t offset = row * kRowStatsWidth;
    row_stats[offset + kRowMaximum] = kNullScore * scale;
    row_stats[offset + kRowNormalizer] = 1.0f;
    row_stats[offset + kRowUtility] = 0.0f;
  }
}


__global__ void initialize_row_prior_kernel(
    float* __restrict__ row_prior,
    int seq_len) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < seq_len) {
    row_prior[row] = row > 0 ? logf(static_cast<float>(row)) : 0.0f;
  }
}


__global__ void finalize_stats_kernel(
    float* __restrict__ row_stats,
    int64_t row_count) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < row_count) {
    const int64_t offset = row * kRowStatsWidth;
    const float inverse_normalizer =
        1.0f / row_stats[offset + kRowNormalizer];
    row_stats[offset + kRowNormalizer] = inverse_normalizer;
    row_stats[offset + kRowUtility] *= inverse_normalizer;
  }
}


__global__ void fused_slab_stats_fp16_kernel(
    const c10::Half* __restrict__ value,
    const c10::Half* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_prior,
    float* __restrict__ partial_stats,
    int series_count,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int diagonal_tile_count,
    int partial_tile_stride,
    float mismatch_unit,
    float scale,
    float dropout_p,
    float inverse_keep_probability) {
#if __CUDA_ARCH__ >= 700
  __shared__ __align__(16) __half grad_tile[
      kFusedRows * kFusedValueDim];
  __shared__ __align__(16) __half value_tile[
      kFusedRoutes * kFusedValueDim];
  __shared__ __align__(16) float utility_tile[
      kFusedRows * kFusedRoutes];
  __shared__ float score_tile[kFusedRows * kFusedScoreStride];
  __shared__ float gate_lut[kMismatchGateCount];

  const int series = blockIdx.x / diagonal_tile_count;
  const int diagonal_tile = blockIdx.x - series * diagonal_tile_count;
  if (series >= series_count) {
    return;
  }
  const int thread = threadIdx.x;
  const int warp = thread / kWarpSize;
  const int lane = thread & (kWarpSize - 1);
  const int diagonal_start =
      slab_start + diagonal_tile * kFusedDiagonals;
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int value_head = head / (num_heads / num_value_heads);
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  constexpr int kDiagonalRounds =
      kFusedDiagonals / (kFusedThreads / kWarpSize);
  float score_carry[kDiagonalRounds] = {};
  initialize_mismatch_gate_lut(gate_lut, symbol_dim, mismatch_unit);

  // The CTA begins at the first 32-row block that can contain one of its
  // diagonals, then carries all 32 recurrences through the rest of the slab.
  const int first_row_start =
      (diagonal_start / kFusedRows) * kFusedRows;
  for (int row_start = first_row_start;
       row_start < seq_len;
       row_start += kFusedRows) {
    const int row_count = min(kFusedRows, seq_len - row_start);
    const int route_start = row_start - diagonal_start -
        (kFusedDiagonals - 1) + 1;
    const int row = row_start + lane;
    const bool lane_has_row = lane < row_count;
    const uint32_t query_word = lane_has_row
        ? static_cast<uint32_t>(packed_query[series_offset + row])
        : 0u;
    const int tile_width = min(
        kFusedDiagonals,
        slab_width - diagonal_tile * kFusedDiagonals);

    constexpr int kFusedValuePairs = kFusedValueDim / 2;
    for (int pair = thread;
         pair < kFusedRows * kFusedValuePairs;
         pair += blockDim.x) {
      const int row_offset = pair / kFusedValuePairs;
      const int feature = (pair - row_offset * kFusedValuePairs) * 2;
      __half2 gradient = __float2half2_rn(0.0f);
      if (row_offset < row_count) {
        const int source_row = row_start + row_offset;
        const int64_t source =
            ((static_cast<int64_t>(batch) * seq_len + source_row) *
                 num_heads +
             head) * kFusedValueDim + feature;
        gradient = *reinterpret_cast<const __half2*>(grad_output + source);
      }
      reinterpret_cast<__half2*>(grad_tile)[pair] = gradient;
    }
    for (int pair = thread;
         pair < kFusedRoutes * kFusedValuePairs;
         pair += blockDim.x) {
      const int route_offset = pair / kFusedValuePairs;
      const int feature = (pair - route_offset * kFusedValuePairs) * 2;
      const int route = route_start + route_offset;
      __half2 sign = __float2half2_rn(0.0f);
      if (route >= 1 && route < seq_len) {
        const int64_t source =
            ((static_cast<int64_t>(batch) * seq_len + route) *
                 num_value_heads +
             value_head) * kFusedValueDim + feature;
        sign = binary_sign_half2(
            *reinterpret_cast<const __half2*>(value + source));
      }
      reinterpret_cast<__half2*>(value_tile)[pair] = sign;
    }

    // Each warp scans one diagonal at a time across 32 query rows. Four
    // rounds cover all 32 diagonals while keeping every warp active.
#pragma unroll
    for (int round = 0; round < kDiagonalRounds; ++round) {
      const int diagonal_offset =
          round * (kFusedThreads / kWarpSize) + warp;
      const int delta = diagonal_start + diagonal_offset;
      const bool active = lane_has_row &&
          diagonal_offset < tile_width &&
          delta <= row;
      const int key_position = row - delta;
      const float gate = active
          ? mismatch_gate_from_lut(
                query_word,
                static_cast<uint32_t>(
                    packed_key[series_offset + key_position]),
                mask,
                gate_lut)
          : 1.0f;
      float coefficient = gate;
      float bias = active ? gate : 0.0f;
      warp_forward_affine_scan(coefficient, bias);
      const float score =
          fmaf(coefficient, score_carry[round], bias);
      if (lane_has_row) {
        score_tile[lane * kFusedScoreStride + diagonal_offset] =
            active ? score : 0.0f;
      }
      score_carry[round] = __shfl_sync(
          0xffffffffu, score, row_count - 1);
    }
    __syncthreads();

    constexpr int kRouteTiles = kFusedRoutes / kTensorTile;
    constexpr int kMmaWarps =
        (kFusedRows / kTensorTile) * kRouteTiles;
    if (warp < kMmaWarps) {
      const int output_row = (warp / kRouteTiles) * kTensorTile;
      const int output_route = (warp % kRouteTiles) * kTensorTile;
      wmma::fragment<
          wmma::accumulator,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          float>
          accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
      for (int feature_start = 0;
           feature_start < kFusedValueDim;
           feature_start += kTensorTile) {
        wmma::fragment<
            wmma::matrix_a,
            kTensorTile,
            kTensorTile,
            kTensorTile,
            __half,
            wmma::row_major>
            gradient;
        wmma::fragment<
            wmma::matrix_b,
            kTensorTile,
            kTensorTile,
            kTensorTile,
            __half,
            wmma::col_major>
            signed_value;
        wmma::load_matrix_sync(
            gradient,
            grad_tile + output_row * kFusedValueDim + feature_start,
            kFusedValueDim);
        wmma::load_matrix_sync(
            signed_value,
            value_tile + output_route * kFusedValueDim + feature_start,
            kFusedValueDim);
        wmma::mma_sync(accumulator, gradient, signed_value, accumulator);
      }
      wmma::store_matrix_sync(
          utility_tile + output_row * kFusedRoutes + output_route,
          accumulator,
          kFusedRoutes,
          wmma::mem_row_major);
    }
    __syncthreads();

    for (int reduction_row = warp;
         reduction_row < row_count;
         reduction_row += kFusedThreads / kWarpSize) {
      const int row = row_start + reduction_row;
      const bool active =
          lane < kFusedDiagonals &&
          lane < slab_width - diagonal_tile * kFusedDiagonals &&
          diagonal_start + lane <= row;
      float logit = -FLT_MAX;
      float utility = 0.0f;
      if (active) {
        const int route_offset =
            reduction_row - lane + kFusedDiagonals - 1;
        const int route = route_start + route_offset;
        const ScoreTransform transformed = transform_score(
            score_tile[reduction_row * kFusedScoreStride + lane]);
        utility = attention_dropout_scale(
            dropout_seed,
            dropout_p,
            inverse_keep_probability,
            batch,
            head,
            row,
            route) * utility_tile[
                reduction_row * kFusedRoutes + route_offset];
        logit = transformed.route_score * scale - row_prior[row];
      }
      const SoftmaxStats item =
          warp_reduce_candidates(logit, utility, active);
      if (lane == 0) {
        const int64_t target =
            (((static_cast<int64_t>(series) * seq_len + row) *
                   partial_tile_stride +
               diagonal_tile) *
              kRowStatsWidth);
        partial_stats[target + kRowMaximum] = item.maximum;
        partial_stats[target + kRowNormalizer] = item.normalizer;
        partial_stats[target + kRowUtility] = item.utility_numerator;
      }
    }
    __syncthreads();
  }
#endif
}


#ifdef ROSA_SOFT_BENCHMARK_COMPAT
template <int MacroDiagonals>
__global__ void macro_slab_stats_fp16_kernel(
    const c10::Half* __restrict__ value,
    const c10::Half* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_prior,
    float* __restrict__ partial_stats,
    int series_count,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int macro_tile_count,
    int partial_tile_stride,
    float mismatch_unit,
    float scale,
    float dropout_p,
    float inverse_keep_probability) {
#if __CUDA_ARCH__ >= 700
  static_assert(MacroDiagonals % kFusedDiagonals == 0);
  constexpr int kWarps = kFusedThreads / kWarpSize;
  constexpr int kMicroTiles = MacroDiagonals / kFusedDiagonals;
  constexpr int kRoundsPerMicro = kFusedDiagonals / kWarps;
  constexpr int kMacroRounds = MacroDiagonals / kWarps;

  __shared__ __align__(16) __half grad_tile[
      kFusedRows * kFusedValueDim];
  __shared__ __align__(16) __half value_tile[
      kFusedRoutes * kFusedValueDim];
  __shared__ __align__(16) float utility_tile[
      kFusedRows * kFusedRoutes];
  __shared__ float score_tile[kFusedRows * kFusedScoreStride];
  __shared__ float macro_stats[kFusedRows * kRowStatsWidth];
  __shared__ float gate_lut[kMismatchGateCount];

  const int series = blockIdx.x / macro_tile_count;
  const int macro_tile = blockIdx.x - series * macro_tile_count;
  if (series >= series_count) {
    return;
  }
  const int thread = threadIdx.x;
  const int warp = thread / kWarpSize;
  const int lane = thread & (kWarpSize - 1);
  const int macro_start = slab_start + macro_tile * MacroDiagonals;
  const int macro_width = min(
      MacroDiagonals, slab_width - macro_tile * MacroDiagonals);
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int value_head = head / (num_heads / num_value_heads);
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  float score_carry[kMacroRounds] = {};
  initialize_mismatch_gate_lut(gate_lut, symbol_dim, mismatch_unit);

  const int first_row_start =
      (macro_start / kFusedRows) * kFusedRows;
  for (int row_start = first_row_start;
       row_start < seq_len;
       row_start += kFusedRows) {
    const int row_count = min(kFusedRows, seq_len - row_start);
    const int row = row_start + lane;
    const bool lane_has_row = lane < row_count;
    const uint32_t query_word = lane_has_row
        ? static_cast<uint32_t>(packed_query[series_offset + row])
        : 0u;

    constexpr int kFusedValuePairs = kFusedValueDim / 2;
    for (int pair = thread;
         pair < kFusedRows * kFusedValuePairs;
         pair += blockDim.x) {
      const int row_offset = pair / kFusedValuePairs;
      const int feature = (pair - row_offset * kFusedValuePairs) * 2;
      __half2 gradient = __float2half2_rn(0.0f);
      if (row_offset < row_count) {
        const int source_row = row_start + row_offset;
        const int64_t source =
            ((static_cast<int64_t>(batch) * seq_len + source_row) *
                 num_heads +
             head) * kFusedValueDim + feature;
        gradient = *reinterpret_cast<const __half2*>(grad_output + source);
      }
      reinterpret_cast<__half2*>(grad_tile)[pair] = gradient;
    }
    for (int index = thread;
         index < kFusedRows * kRowStatsWidth;
         index += blockDim.x) {
      const int field = index % kRowStatsWidth;
      macro_stats[index] = field == kRowMaximum ? -FLT_MAX : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int micro = 0; micro < kMicroTiles; ++micro) {
      const int local_start = micro * kFusedDiagonals;
      const int diagonal_start = macro_start + local_start;
      const int tile_width = min(
          kFusedDiagonals, max(0, macro_width - local_start));
      const int route_start = row_start - diagonal_start -
          (kFusedDiagonals - 1) + 1;

      for (int pair = thread;
           pair < kFusedRoutes * kFusedValuePairs;
           pair += blockDim.x) {
        const int route_offset = pair / kFusedValuePairs;
        const int feature = (pair - route_offset * kFusedValuePairs) * 2;
        const int route = route_start + route_offset;
        __half2 sign = __float2half2_rn(0.0f);
        if (route >= 1 && route < seq_len) {
          const int64_t source =
              ((static_cast<int64_t>(batch) * seq_len + route) *
                   num_value_heads +
               value_head) * kFusedValueDim + feature;
          sign = binary_sign_half2(
              *reinterpret_cast<const __half2*>(value + source));
        }
        reinterpret_cast<__half2*>(value_tile)[pair] = sign;
      }

#pragma unroll
      for (int round = 0; round < kRoundsPerMicro; ++round) {
        const int diagonal_offset = round * kWarps + warp;
        const int macro_round = micro * kRoundsPerMicro + round;
        const int delta = diagonal_start + diagonal_offset;
        const bool active = lane_has_row &&
            diagonal_offset < tile_width && delta <= row;
        const float gate = active
            ? mismatch_gate_from_lut(
                  query_word,
                  static_cast<uint32_t>(
                      packed_key[series_offset + row - delta]),
                  mask,
                  gate_lut)
            : 1.0f;
        float coefficient = gate;
        float bias = active ? gate : 0.0f;
        warp_forward_affine_scan(coefficient, bias);
        const float score = fmaf(
            coefficient, score_carry[macro_round], bias);
        if (lane_has_row) {
          score_tile[lane * kFusedScoreStride + diagonal_offset] =
              active ? score : 0.0f;
        }
        score_carry[macro_round] = __shfl_sync(
            0xffffffffu, score, row_count - 1);
      }
      __syncthreads();

      constexpr int kRouteTiles = kFusedRoutes / kTensorTile;
      constexpr int kMmaWarps =
          (kFusedRows / kTensorTile) * kRouteTiles;
      if (warp < kMmaWarps) {
        const int output_row = (warp / kRouteTiles) * kTensorTile;
        const int output_route = (warp % kRouteTiles) * kTensorTile;
        wmma::fragment<
            wmma::accumulator,
            kTensorTile,
            kTensorTile,
            kTensorTile,
            float>
            accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
        for (int feature_start = 0;
             feature_start < kFusedValueDim;
             feature_start += kTensorTile) {
          wmma::fragment<
              wmma::matrix_a,
              kTensorTile,
              kTensorTile,
              kTensorTile,
              __half,
              wmma::row_major>
              gradient;
          wmma::fragment<
              wmma::matrix_b,
              kTensorTile,
              kTensorTile,
              kTensorTile,
              __half,
              wmma::col_major>
              signed_value;
          wmma::load_matrix_sync(
              gradient,
              grad_tile + output_row * kFusedValueDim + feature_start,
              kFusedValueDim);
          wmma::load_matrix_sync(
              signed_value,
              value_tile + output_route * kFusedValueDim + feature_start,
              kFusedValueDim);
          wmma::mma_sync(accumulator, gradient, signed_value, accumulator);
        }
        wmma::store_matrix_sync(
            utility_tile + output_row * kFusedRoutes + output_route,
            accumulator,
            kFusedRoutes,
            wmma::mem_row_major);
      }
      __syncthreads();

      for (int reduction_row = warp;
           reduction_row < row_count;
           reduction_row += kWarps) {
        const int reduction_position = row_start + reduction_row;
        const bool active = lane < tile_width &&
            diagonal_start + lane <= reduction_position;
        float logit = -FLT_MAX;
        float utility = 0.0f;
        if (active) {
          const int route_offset =
              reduction_row - lane + kFusedDiagonals - 1;
          const int route = route_start + route_offset;
          const ScoreTransform transformed = transform_score(
              score_tile[
                  reduction_row * kFusedScoreStride + lane]);
          utility = attention_dropout_scale(
              dropout_seed,
              dropout_p,
              inverse_keep_probability,
              batch,
              head,
              reduction_position,
              route) * utility_tile[
                  reduction_row * kFusedRoutes + route_offset];
          logit = transformed.route_score * scale -
              row_prior[reduction_position];
        }
        const SoftmaxStats item =
            warp_reduce_candidates(logit, utility, active);
        if (lane == 0 && item.normalizer != 0.0f) {
          const int stats_offset = reduction_row * kRowStatsWidth;
          if (macro_stats[stats_offset + kRowNormalizer] == 0.0f) {
            macro_stats[stats_offset + kRowMaximum] = item.maximum;
            macro_stats[stats_offset + kRowNormalizer] = item.normalizer;
            macro_stats[stats_offset + kRowUtility] =
                item.utility_numerator;
          } else {
            const SoftmaxStats previous = {
                macro_stats[stats_offset + kRowMaximum],
                macro_stats[stats_offset + kRowNormalizer],
                macro_stats[stats_offset + kRowUtility]};
            const SoftmaxStats merged = merge_stats(previous, item);
            macro_stats[stats_offset + kRowMaximum] = merged.maximum;
            macro_stats[stats_offset + kRowNormalizer] = merged.normalizer;
            macro_stats[stats_offset + kRowUtility] =
                merged.utility_numerator;
          }
        }
      }
      __syncthreads();
    }

    for (int index = thread;
         index < row_count * kRowStatsWidth;
         index += blockDim.x) {
      const int row_offset = index / kRowStatsWidth;
      const int field = index - row_offset * kRowStatsWidth;
      const int64_t target =
          (((static_cast<int64_t>(series) * seq_len +
              row_start + row_offset) * partial_tile_stride + macro_tile) *
           kRowStatsWidth) + field;
      partial_stats[target] =
          macro_stats[row_offset * kRowStatsWidth + field];
    }
    __syncthreads();
  }
#endif
}
#endif


__global__ void merge_fused_slab_stats_kernel(
    const float* __restrict__ partial_stats,
    float* __restrict__ row_stats,
    int series_count,
    int seq_len,
    int slab_start,
    int diagonal_tile_count,
    int partial_tile_stride,
    int diagonals_per_tile) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  constexpr int kRowsPerBlock = kFusedThreads / kWarpSize;
  const int row_blocks =
      (seq_len - slab_start + kRowsPerBlock - 1) / kRowsPerBlock;
  const int series = blockIdx.x / row_blocks;
  const int row_block = blockIdx.x - series * row_blocks;
  const int row = slab_start + row_block * kRowsPerBlock + warp;
  if (series >= series_count || row >= seq_len) {
    return;
  }
  const int active_tiles = min(
      diagonal_tile_count,
      (row - slab_start) / diagonals_per_tile + 1);
  SoftmaxStats local = {-FLT_MAX, 0.0f, 0.0f};
  for (int tile = lane; tile < active_tiles; tile += kWarpSize) {
    const int64_t source =
        (((static_cast<int64_t>(series) * seq_len + row) *
              partial_tile_stride +
          tile) *
         kRowStatsWidth);
    const SoftmaxStats partial = {
        partial_stats[source + kRowMaximum],
        partial_stats[source + kRowNormalizer],
        partial_stats[source + kRowUtility]};
    local = merge_stats(local, partial);
  }
  local = warp_reduce_stats(local);
  if (lane == 0) {
    const int64_t target =
        (static_cast<int64_t>(series) * seq_len + row) * kRowStatsWidth;
    const SoftmaxStats previous = {
        row_stats[target + kRowMaximum],
        row_stats[target + kRowNormalizer],
        row_stats[target + kRowUtility]};
    const SoftmaxStats merged = merge_stats(previous, local);
    row_stats[target + kRowMaximum] = merged.maximum;
    row_stats[target + kRowNormalizer] = merged.normalizer;
    row_stats[target + kRowUtility] = merged.utility_numerator;
  }
}


__global__ void slab_diagonal_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ scores,
    int series_count,
    int seq_len,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float mismatch_unit) {
  __shared__ float gate_lut[kMismatchGateCount];
  const int blocks_per_series =
      (slab_width + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int series = blockIdx.x / blocks_per_series;
  const int diagonal_block = blockIdx.x - series * blocks_per_series;
  if (series >= series_count) {
    return;
  }
  initialize_mismatch_gate_lut(gate_lut, symbol_dim, mismatch_unit);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int slot = diagonal_block * kWarpsPerBlock + warp;
  if (slot >= slab_width) {
    return;
  }
  const int delta = slab_start + slot;
  const uint32_t mask = symbol_mask(symbol_dim);
  const int diagonal_length = seq_len - delta;
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  float incoming_score = 0.0f;
  for (int position_start = 0;
       position_start < diagonal_length;
       position_start += kWarpSize) {
    const int active_count = min(kWarpSize, diagonal_length - position_start);
    const int diagonal_position = position_start + lane;
    const bool active = lane < active_count;
    const int row = diagonal_position + delta;
    const float gate = active
        ? mismatch_gate_from_lut(
              static_cast<uint32_t>(packed_query[series_offset + row]),
              static_cast<uint32_t>(
                  packed_key[series_offset + diagonal_position]),
              mask,
              gate_lut)
        : 1.0f;
    float coefficient = gate;
    float bias = active ? gate : 0.0f;
    warp_forward_affine_scan(coefficient, bias);
    const float score = fmaf(coefficient, incoming_score, bias);
    if (active) {
      scores[workspace_index(
          series, row, slot, seq_len, workspace_stride)] = score;
    }
    incoming_score = __shfl_sync(0xffffffffu, score, active_count - 1);
  }
}


constexpr int kUtilityTile = 16;
constexpr int kUtilityRouteSpan = 2 * kUtilityTile - 1;
constexpr int kUtilitySharedStride = kUtilityTile + 1;
constexpr int kUtilityThreads = kUtilityTile * kUtilityTile;


template <typename scalar_t>
__global__ void slab_utilities_tiled_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    float* __restrict__ utilities,
    int series_count,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int slab_start,
    int slab_width,
    int workspace_stride) {
  __shared__ float grad_tile[kUtilityTile * kUtilityTile];
  __shared__ float value_tile[
      kUtilityRouteSpan * kUtilitySharedStride];
  const int row_tiles = (seq_len + kUtilityTile - 1) / kUtilityTile;
  const int diagonal_tiles =
      (slab_width + kUtilityTile - 1) / kUtilityTile;
  const int tiles_per_series = row_tiles * diagonal_tiles;
  const int series = blockIdx.x / tiles_per_series;
  const int local_tile = blockIdx.x - series * tiles_per_series;
  if (series >= series_count) {
    return;
  }
  const int diagonal_tile = local_tile / row_tiles;
  const int row_tile = local_tile - diagonal_tile * row_tiles;
  const int local_diagonal_start = diagonal_tile * kUtilityTile;
  const int diagonal_start = slab_start + local_diagonal_start;
  const int row_start = row_tile * kUtilityTile;
  if (row_start + kUtilityTile - 1 < diagonal_start) {
    return;
  }
  const int route_start = row_start - diagonal_start - kUtilityTile + 2;
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int value_head = head / (num_heads / num_value_heads);
  const int row_offset = threadIdx.x / kUtilityTile;
  const int diagonal_offset = threadIdx.x & (kUtilityTile - 1);
  const int row = row_start + row_offset;
  const int local_diagonal = local_diagonal_start + diagonal_offset;
  const int delta = diagonal_start + diagonal_offset;
  const int route_offset = row_offset - diagonal_offset + kUtilityTile - 1;
  const bool active = row < seq_len && local_diagonal < slab_width &&
      delta <= row;
  float utility = 0.0f;

  for (int feature_start = 0;
       feature_start < value_dim;
       feature_start += kUtilityTile) {
    const int feature = feature_start + diagonal_offset;
    float gradient = 0.0f;
    if (row < seq_len && feature < value_dim) {
      const int64_t source =
          ((static_cast<int64_t>(batch) * seq_len + row) * num_heads + head) *
              value_dim +
          feature;
      gradient = read_float(grad_output, source);
    }
    grad_tile[threadIdx.x] = gradient;
    for (int index = threadIdx.x;
         index < kUtilityRouteSpan * kUtilityTile;
         index += blockDim.x) {
      const int staged_route = index / kUtilityTile;
      const int staged_feature = index - staged_route * kUtilityTile;
      const int route = route_start + staged_route;
      const int value_feature = feature_start + staged_feature;
      float value_sign = 0.0f;
      if (route >= 1 && route < seq_len && value_feature < value_dim) {
        const int64_t source =
            ((static_cast<int64_t>(batch) * seq_len + route) *
                 num_value_heads +
             value_head) * value_dim + value_feature;
        value_sign = read_float(value, source) > 0.0f ? 1.0f : -1.0f;
      }
      value_tile[
          staged_route * kUtilitySharedStride + staged_feature] = value_sign;
    }
    __syncthreads();
    if (active) {
#pragma unroll
      for (int feature_offset = 0;
           feature_offset < kUtilityTile;
           ++feature_offset) {
        utility = fmaf(
            grad_tile[row_offset * kUtilityTile + feature_offset],
            value_tile[
                route_offset * kUtilitySharedStride + feature_offset],
            utility);
      }
    }
    __syncthreads();
  }
  if (active) {
    utilities[workspace_index(
        series,
        row,
        local_diagonal,
        seq_len,
        workspace_stride)] = utility;
  }
}


__global__ void accumulate_slab_stats_kernel(
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_prior,
    const float* __restrict__ scores,
    const float* __restrict__ utilities,
    float* __restrict__ row_stats,
    int seq_len,
    int num_heads,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    int compute_utility) {
  const int active_rows = seq_len - slab_start;
  const int series = blockIdx.x / active_rows;
  const int row = slab_start + blockIdx.x - series * active_rows;
  const int64_t row_index = static_cast<int64_t>(series) * seq_len + row;
  const int head = series % num_heads;
  const int batch = series / num_heads;
  SoftmaxStats local = {-FLT_MAX, 0.0f, 0.0f};
  const int valid_width = min(slab_width, row - slab_start + 1);

  for (int slot = threadIdx.x; slot < valid_width; slot += blockDim.x) {
    const int delta = slab_start + slot;
    const int route = row - delta + 1;
    const int64_t index = workspace_index(
        series, row, slot, seq_len, workspace_stride);
    const ScoreTransform transformed = transform_score(scores[index]);
    float utility = 0.0f;
    if (compute_utility != 0) {
      utility = attention_dropout_scale(
          dropout_seed,
          dropout_p,
          inverse_keep_probability,
          batch,
          head,
          row,
          route) * utilities[index];
    }
    local = append_item(
        local,
        transformed.route_score * scale - row_prior[row],
        utility);
  }

  local = warp_reduce_stats(local);
  __shared__ SoftmaxStats warp_stats[kWarpsPerBlock];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (lane == 0) {
    warp_stats[warp] = local;
  }
  __syncthreads();
  if (warp == 0) {
    SoftmaxStats slab = lane < kWarpsPerBlock
        ? warp_stats[lane]
        : SoftmaxStats{-FLT_MAX, 0.0f, 0.0f};
    slab = warp_reduce_stats(slab);
    if (lane == 0 && slab.normalizer != 0.0f) {
      const int64_t target = row_index * kRowStatsWidth;
      const SoftmaxStats previous = {
          row_stats[target + kRowMaximum],
          row_stats[target + kRowNormalizer],
          row_stats[target + kRowUtility]};
      const SoftmaxStats merged = merge_stats(previous, slab);
      row_stats[target + kRowMaximum] = merged.maximum;
      row_stats[target + kRowNormalizer] = merged.normalizer;
      row_stats[target + kRowUtility] = merged.utility_numerator;
    }
  }
}


__device__ __forceinline__ void reverse_diagonal(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ scores,
    const float* __restrict__ utilities,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_prior,
    const float* __restrict__ row_stats,
    const float* __restrict__ gate_lut,
    int series,
    int slot,
    int delta,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int workspace_stride,
    float mismatch_unit,
    float scale,
    float dropout_p,
    float inverse_keep_probability) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const int diagonal_length = seq_len - delta;
  float successor_adjoint = 0.0f;
  float successor_gate = 0.0f;

  for (int position_start =
           ((diagonal_length - 1) / kWarpSize) * kWarpSize;
       position_start >= 0;
       position_start -= kWarpSize) {
    const int active_count = min(kWarpSize, diagonal_length - position_start);
    const bool active = lane < active_count;
    const int diagonal_position = position_start + lane;
    const int row = diagonal_position + delta;
    const int key_position = diagonal_position;
    const float gate = active
        ? mismatch_gate_from_lut(
              static_cast<uint32_t>(packed_query[series_offset + row]),
              static_cast<uint32_t>(
                  packed_key[series_offset + key_position]),
              mask,
              gate_lut)
        : 0.0f;
    const float next_lane_gate = __shfl_down_sync(
        0xffffffffu, gate, 1);
    float coefficient = 1.0f;
    float direct_vjp = 0.0f;
    if (active) {
      if (lane + 1 < active_count) {
        coefficient = next_lane_gate;
      } else if (position_start + active_count < diagonal_length) {
        coefficient = successor_gate;
      } else {
        coefficient = 0.0f;
      }
      const int64_t index = workspace_index(
          series, row, slot, seq_len, workspace_stride);
      const ScoreTransform transformed = transform_score(scores[index]);
      const int64_t stats_index =
          (series_offset + row) * kRowStatsWidth;
      const float probability = __expf(
          transformed.route_score * scale -
          row_prior[row] -
          row_stats[stats_index + kRowMaximum]) *
          row_stats[stats_index + kRowNormalizer];
      const int route = key_position + 1;
      const float dropout_scale = attention_dropout_scale(
          dropout_seed,
          dropout_p,
          inverse_keep_probability,
          batch,
          head,
          row,
          route);
      direct_vjp = scale * probability *
          (dropout_scale * utilities[index] -
           row_stats[stats_index + kRowUtility]) *
          transformed.raw_vjp_multiplier;
    }
    warp_reverse_affine_scan(coefficient, direct_vjp, active_count);
    const float score_vjp = fmaf(
        coefficient, successor_adjoint, direct_vjp);
    if (active) {
      const int64_t index = workspace_index(
          series, row, slot, seq_len, workspace_stride);
      scores[index] *= score_vjp;
    }
    successor_adjoint = __shfl_sync(0xffffffffu, score_vjp, 0);
    successor_gate = __shfl_sync(0xffffffffu, gate, 0);
  }
}


__global__ void slab_diagonal_reverse_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ scores,
    const float* __restrict__ utilities,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_prior,
    const float* __restrict__ row_stats,
    int series_count,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float mismatch_unit,
    float scale,
    float dropout_p,
    float inverse_keep_probability) {
  __shared__ float gate_lut[kMismatchGateCount];
  const int blocks_per_series =
      (slab_width + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int series = blockIdx.x / blocks_per_series;
  const int diagonal_block = blockIdx.x - series * blocks_per_series;
  if (series >= series_count) {
    return;
  }
  initialize_mismatch_gate_lut(gate_lut, symbol_dim, mismatch_unit);
  const int warp = threadIdx.x / kWarpSize;
  const int slot = diagonal_block * kWarpsPerBlock + warp;
  if (slot >= slab_width) {
    return;
  }
  reverse_diagonal(
      packed_query,
      packed_key,
      scores,
      utilities,
      dropout_seed,
      row_prior,
      row_stats,
      gate_lut,
      series,
      slot,
      slab_start + slot,
      seq_len,
      num_heads,
      symbol_dim,
      workspace_stride,
      mismatch_unit,
      scale,
      dropout_p,
      inverse_keep_probability);
}


__global__ void fused_slab_reverse_fp16_kernel(
    const c10::Half* __restrict__ value,
    const c10::Half* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_prior,
    const float* __restrict__ row_stats,
    float* __restrict__ scores,
    int series_count,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int diagonal_tile_count,
    int workspace_stride,
    float mismatch_unit,
    float scale,
    float dropout_p,
    float inverse_keep_probability) {
#if __CUDA_ARCH__ >= 700
  constexpr int kScoreStride = kFusedRows + 1;
  constexpr int kRouteCount =
      ((kFusedRows + kFusedDiagonals - 1 + kTensorTile - 1) /
       kTensorTile) * kTensorTile;
  constexpr int kWarpCount = kFusedThreads / kWarpSize;
  constexpr int kDiagonalRounds = kFusedDiagonals / kWarpCount;
  __shared__ __align__(16) __half grad_tile[
      kFusedRows * kFusedValueDim];
  __shared__ __align__(16) __half value_tile[
      kRouteCount * kFusedValueDim];
  __shared__ __align__(16) float utility_tile[
      kFusedRows * kRouteCount];
  __shared__ float score_tile[kFusedDiagonals * kScoreStride];
  __shared__ float gate_lut[kMismatchGateCount];

  const int series = blockIdx.x / diagonal_tile_count;
  const int diagonal_tile = blockIdx.x - series * diagonal_tile_count;
  if (series >= series_count) {
    return;
  }
  const int thread = threadIdx.x;
  const int warp = thread / kWarpSize;
  const int lane = thread & (kWarpSize - 1);
  const int local_diagonal_start = diagonal_tile * kFusedDiagonals;
  const int diagonal_start = slab_start + local_diagonal_start;
  const int tile_width = min(
      kFusedDiagonals, slab_width - local_diagonal_start);
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int value_head = head / (num_heads / num_value_heads);
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  float successor_adjoint[kDiagonalRounds] = {};
  float successor_gate[kDiagonalRounds] = {};
  initialize_mismatch_gate_lut(gate_lut, symbol_dim, mismatch_unit);

  const int first_row_start =
      (diagonal_start / kFusedRows) * kFusedRows;
  for (int row_start =
           ((seq_len - 1) / kFusedRows) * kFusedRows;
       row_start >= first_row_start;
       row_start -= kFusedRows) {
    const int row_count = min(kFusedRows, seq_len - row_start);
    const int route_start = row_start - diagonal_start -
        (kFusedDiagonals - 1) + 1;

    for (int index = thread;
         index < kFusedRows * kFusedValueDim;
         index += blockDim.x) {
      const int row_offset = index / kFusedValueDim;
      const int feature = index - row_offset * kFusedValueDim;
      float gradient = 0.0f;
      if (row_offset < row_count) {
        const int row = row_start + row_offset;
        const int64_t source =
            ((static_cast<int64_t>(batch) * seq_len + row) * num_heads +
             head) * kFusedValueDim + feature;
        gradient = read_float(grad_output, source);
      }
      grad_tile[index] = __float2half_rn(gradient);
    }
    for (int index = thread;
         index < kRouteCount * kFusedValueDim;
         index += blockDim.x) {
      const int route_offset = index / kFusedValueDim;
      const int feature = index - route_offset * kFusedValueDim;
      const int route = route_start + route_offset;
      float sign = 0.0f;
      if (route >= 1 && route < seq_len) {
        const int64_t source =
            ((static_cast<int64_t>(batch) * seq_len + route) *
                 num_value_heads +
             value_head) * kFusedValueDim + feature;
        sign = read_float(value, source) > 0.0f ? 1.0f : -1.0f;
      }
      value_tile[index] = __float2half_rn(sign);
    }

    // The workspace is row-major within four-diagonal groups. Read those
    // groups contiguously and transpose to one shared-memory row per diagonal
    // so each warp can perform a contiguous reverse scan.
    for (int linear = thread;
         linear < kFusedRows * kFusedDiagonals;
         linear += blockDim.x) {
      constexpr int kGroupItems =
          kFusedRows * kWorkspaceDiagonalTile;
      const int group = linear / kGroupItems;
      const int within_group = linear - group * kGroupItems;
      const int row_offset = within_group / kWorkspaceDiagonalTile;
      const int diagonal_lane =
          within_group & (kWorkspaceDiagonalTile - 1);
      const int diagonal_offset =
          group * kWorkspaceDiagonalTile + diagonal_lane;
      const int row = row_start + row_offset;
      const int delta = diagonal_start + diagonal_offset;
      const bool active = row_offset < row_count &&
          diagonal_offset < tile_width && delta <= row;
      score_tile[diagonal_offset * kScoreStride + row_offset] = active
          ? scores[workspace_index(
                series,
                row,
                local_diagonal_start + diagonal_offset,
                seq_len,
                workspace_stride)]
          : 0.0f;
    }
    __syncthreads();

    constexpr int kRouteTiles = kRouteCount / kTensorTile;
    constexpr int kMmaWarps =
        (kFusedRows / kTensorTile) * kRouteTiles;
    if (warp < kMmaWarps) {
      const int output_row = (warp / kRouteTiles) * kTensorTile;
      const int output_route = (warp % kRouteTiles) * kTensorTile;
      wmma::fragment<
          wmma::accumulator,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          float>
          accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
      for (int feature_start = 0;
           feature_start < kFusedValueDim;
           feature_start += kTensorTile) {
        wmma::fragment<
            wmma::matrix_a,
            kTensorTile,
            kTensorTile,
            kTensorTile,
            __half,
            wmma::row_major>
            gradient;
        wmma::fragment<
            wmma::matrix_b,
            kTensorTile,
            kTensorTile,
            kTensorTile,
            __half,
            wmma::col_major>
            signed_value;
        wmma::load_matrix_sync(
            gradient,
            grad_tile + output_row * kFusedValueDim + feature_start,
            kFusedValueDim);
        wmma::load_matrix_sync(
            signed_value,
            value_tile + output_route * kFusedValueDim + feature_start,
            kFusedValueDim);
        wmma::mma_sync(accumulator, gradient, signed_value, accumulator);
      }
      wmma::store_matrix_sync(
          utility_tile + output_row * kRouteCount + output_route,
          accumulator,
          kRouteCount,
          wmma::mem_row_major);
    }
    __syncthreads();

#pragma unroll
    for (int diagonal_round = 0;
         diagonal_round < kDiagonalRounds;
         ++diagonal_round) {
      const int diagonal_offset =
          diagonal_round * kWarpCount + warp;
      const int delta = diagonal_start + diagonal_offset;
      const int row = row_start + lane;
      const bool active = diagonal_offset < tile_width &&
          lane < row_count && delta <= row;
      const float gate = active
          ? mismatch_gate_from_lut(
                static_cast<uint32_t>(packed_query[series_offset + row]),
                static_cast<uint32_t>(
                    packed_key[series_offset + row - delta]),
                mask,
                gate_lut)
          : 0.0f;
      const float next_lane_gate = __shfl_down_sync(
          0xffffffffu, gate, 1);
      float coefficient = 1.0f;
      float direct_vjp = 0.0f;
      float raw_score = 0.0f;
      if (active) {
        coefficient = lane + 1 < row_count
            ? next_lane_gate
            : successor_gate[diagonal_round];
        raw_score =
            score_tile[diagonal_offset * kScoreStride + lane];
        const ScoreTransform transformed = transform_score(raw_score);
        const int64_t stats_index =
            (series_offset + row) * kRowStatsWidth;
        const float probability = __expf(
            transformed.route_score * scale -
            row_prior[row] -
            row_stats[stats_index + kRowMaximum]) *
            row_stats[stats_index + kRowNormalizer];
        const int route_offset =
            lane - diagonal_offset + kFusedDiagonals - 1;
        const int route = route_start + route_offset;
        const float dropout_scale = attention_dropout_scale(
            dropout_seed,
            dropout_p,
            inverse_keep_probability,
            batch,
            head,
            row,
            route);
        direct_vjp = scale * probability *
            (dropout_scale * utility_tile[
                 lane * kRouteCount + route_offset] -
             row_stats[stats_index + kRowUtility]) *
            transformed.raw_vjp_multiplier;
      }
      warp_reverse_affine_scan(coefficient, direct_vjp, row_count);
      const float score_vjp = fmaf(
          coefficient,
          successor_adjoint[diagonal_round],
          direct_vjp);
      if (active) {
        score_tile[diagonal_offset * kScoreStride + lane] =
            raw_score * score_vjp;
      }
      successor_adjoint[diagonal_round] = __shfl_sync(
          0xffffffffu, score_vjp, 0);
      successor_gate[diagonal_round] = __shfl_sync(
          0xffffffffu, gate, 0);
    }
    __syncthreads();

    for (int linear = thread;
         linear < kFusedRows * kFusedDiagonals;
         linear += blockDim.x) {
      constexpr int kGroupItems =
          kFusedRows * kWorkspaceDiagonalTile;
      const int group = linear / kGroupItems;
      const int within_group = linear - group * kGroupItems;
      const int row_offset = within_group / kWorkspaceDiagonalTile;
      const int diagonal_lane =
          within_group & (kWorkspaceDiagonalTile - 1);
      const int diagonal_offset =
          group * kWorkspaceDiagonalTile + diagonal_lane;
      const int row = row_start + row_offset;
      const int delta = diagonal_start + diagonal_offset;
      if (row_offset < row_count && diagonal_offset < tile_width &&
          delta <= row) {
        scores[workspace_index(
            series,
            row,
            local_diagonal_start + diagonal_offset,
            seq_len,
            workspace_stride)] =
            score_tile[diagonal_offset * kScoreStride + row_offset];
      }
    }
    __syncthreads();
  }
#endif
}


__global__ void accumulate_slab_symbol_vjp_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const float* __restrict__ log_gate_vjp,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    int64_t output_items,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float symbol_scale,
    int gradient_mask) {
  const int64_t tensor_items =
      static_cast<int64_t>(batch_size) * seq_len * num_heads * symbol_dim;
  for (int64_t output =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output < output_items;
       output += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const bool query_output = output < tensor_items;
    const int64_t local_output = query_output
        ? output
        : output - tensor_items;
    const int bit = local_output % symbol_dim;
    const int64_t token_head = local_output / symbol_dim;
    const int head = token_head % num_heads;
    const int64_t token = token_head / num_heads;
    const int position = token % seq_len;
    const int batch = token / seq_len;
    const int series = batch * num_heads + head;
    float contribution = 0.0f;
    if (query_output) {
      if ((gradient_mask & kGradQuery) == 0 || position < slab_start) {
        continue;
      }
      const int valid_width = min(
          slab_width, position - slab_start + 1);
      for (int slot = 0; slot < valid_width; ++slot) {
        const int delta = slab_start + slot;
        const int key_position = position - delta;
        contribution += log_gate_vjp[workspace_index(
            series, position, slot, seq_len, workspace_stride)] *
            static_cast<float>(sign_from_bit(
                static_cast<uint32_t>(
                    packed_key[static_cast<int64_t>(series) * seq_len +
                               key_position]),
                bit));
      }
      grad_query[local_output] += symbol_scale * contribution;
    } else {
      const int valid_width = min(
          slab_width, seq_len - position - slab_start);
      if ((gradient_mask & kGradKey) == 0 || valid_width <= 0) {
        continue;
      }
      for (int slot = 0; slot < valid_width; ++slot) {
        const int delta = slab_start + slot;
        const int row = position + delta;
        contribution += log_gate_vjp[workspace_index(
            series, row, slot, seq_len, workspace_stride)] *
            static_cast<float>(sign_from_bit(
                static_cast<uint32_t>(
                    packed_query[static_cast<int64_t>(series) * seq_len +
                                 row]),
                bit));
      }
      grad_key[local_output] += symbol_scale * contribution;
    }
  }
}


__global__ void accumulate_slab_symbol_vjp_tiled_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const float* __restrict__ log_gate_vjp,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    int series_count,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float symbol_scale,
    int gradient_mask) {
  constexpr int kOwnerWarps = kFusedThreads / kWarpSize;
  constexpr int kBitsPerWarp = 8;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t owners_per_side =
      static_cast<int64_t>(series_count) * seq_len;
  const int bit_groups =
      (symbol_dim + kBitsPerWarp - 1) / kBitsPerWarp;
  const int64_t work =
      static_cast<int64_t>(blockIdx.x) * kOwnerWarps + warp;
  if (work >= 2 * owners_per_side * bit_groups) {
    return;
  }
  const int bit_group = work % bit_groups;
  const int bit_start = bit_group * kBitsPerWarp;
  const int64_t owner = work / bit_groups;
  const bool query_owner = owner < owners_per_side;
  if ((query_owner && (gradient_mask & kGradQuery) == 0) ||
      (!query_owner && (gradient_mask & kGradKey) == 0)) {
    return;
  }
  const int64_t local_owner = query_owner
      ? owner
      : owner - owners_per_side;
  const int position = local_owner % seq_len;
  const int series = local_owner / seq_len;
  const int head = series % num_heads;
  const int batch = series / num_heads;
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  float contribution[kBitsPerWarp] = {};

  if (query_owner) {
    const int valid_width = position >= slab_start
        ? min(slab_width, position - slab_start + 1)
        : 0;
    for (int slot = lane; slot < valid_width; slot += kWarpSize) {
      const int delta = slab_start + slot;
      const float log_gate = log_gate_vjp[workspace_index(
          series, position, slot, seq_len, workspace_stride)];
      const uint32_t word = static_cast<uint32_t>(
          packed_key[series_offset + position - delta]);
#pragma unroll
      for (int local_bit = 0;
           local_bit < kBitsPerWarp && bit_start + local_bit < symbol_dim;
           ++local_bit) {
        contribution[local_bit] = fmaf(
            log_gate,
            static_cast<float>(
                sign_from_bit(word, bit_start + local_bit)),
            contribution[local_bit]);
      }
    }
  } else {
    const int valid_width = min(
        slab_width, seq_len - position - slab_start);
    for (int slot = lane; slot < valid_width; slot += kWarpSize) {
      const int delta = slab_start + slot;
      const int row = position + delta;
      const float log_gate = log_gate_vjp[workspace_index(
          series, row, slot, seq_len, workspace_stride)];
      const uint32_t word = static_cast<uint32_t>(
          packed_query[series_offset + row]);
#pragma unroll
      for (int local_bit = 0;
           local_bit < kBitsPerWarp && bit_start + local_bit < symbol_dim;
           ++local_bit) {
        contribution[local_bit] = fmaf(
            log_gate,
            static_cast<float>(
                sign_from_bit(word, bit_start + local_bit)),
            contribution[local_bit]);
      }
    }
  }

#pragma unroll
  for (int local_bit = 0;
       local_bit < kBitsPerWarp && bit_start + local_bit < symbol_dim;
       ++local_bit) {
    float total = contribution[local_bit];
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    if (lane == 0) {
      const int64_t output =
          ((static_cast<int64_t>(batch) * seq_len + position) * num_heads +
           head) * symbol_dim + bit_start + local_bit;
      if (query_owner) {
        grad_query[output] += symbol_scale * total;
      } else {
        grad_key[output] += symbol_scale * total;
      }
    }
  }
}


template <typename scalar_t>
__global__ void accumulate_slab_value_vjp_kernel(
    const scalar_t* __restrict__ grad_output,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_prior,
    const float* __restrict__ row_stats,
    const float* __restrict__ scores,
    float* __restrict__ grad_value,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float scale,
    float dropout_p,
    float inverse_keep_probability) {
  constexpr int kCandidateTile = 256;
  __shared__ float probability_tile[kCandidateTile];
  __shared__ int64_t gradient_base_tile[kCandidateTile];
  const int active_routes = seq_len - slab_start;
  const int routes_per_batch = num_value_heads * active_routes;
  const int batch = blockIdx.x / routes_per_batch;
  const int local_block = blockIdx.x - batch * routes_per_batch;
  const int value_head = local_block / active_routes;
  const int route = local_block - value_head * active_routes + 1;
  if (batch >= batch_size) {
    return;
  }
  const int heads_per_value = num_heads / num_value_heads;
  const int valid_width = min(
      slab_width, seq_len - route - slab_start + 1);
  if (valid_width <= 0) {
    return;
  }
  const int candidate_count = heads_per_value * valid_width;

  for (int candidate_start = 0;
       candidate_start < candidate_count;
       candidate_start += kCandidateTile) {
    const int tile_count = min(
        kCandidateTile, candidate_count - candidate_start);
    for (int local_candidate = threadIdx.x;
         local_candidate < tile_count;
         local_candidate += blockDim.x) {
      const int candidate = candidate_start + local_candidate;
      const int head_offset = candidate / valid_width;
      const int slot = candidate - head_offset * valid_width;
      const int head = value_head * heads_per_value + head_offset;
      const int series = batch * num_heads + head;
      const int row = route + slab_start + slot - 1;
      const int64_t score_index = workspace_index(
          series, row, slot, seq_len, workspace_stride);
      const int64_t stats_index =
          (static_cast<int64_t>(series) * seq_len + row) * kRowStatsWidth;
      const ScoreTransform transformed =
          transform_score(scores[score_index]);
      float probability = __expf(
          transformed.route_score * scale -
          row_prior[row] -
          row_stats[stats_index + kRowMaximum]) *
          row_stats[stats_index + kRowNormalizer];
      probability *= attention_dropout_scale(
          dropout_seed,
          dropout_p,
          inverse_keep_probability,
          batch,
          head,
          row,
          route);
      const int64_t gradient_base =
          ((static_cast<int64_t>(batch) * seq_len + row) * num_heads +
           head) * value_dim;
      probability_tile[local_candidate] = probability;
      gradient_base_tile[local_candidate] = gradient_base;
    }
    __syncthreads();

    for (int feature = threadIdx.x;
         feature < value_dim;
         feature += blockDim.x) {
      float contribution = 0.0f;
      for (int local_candidate = 0;
           local_candidate < tile_count;
           ++local_candidate) {
        const int64_t gradient_base = gradient_base_tile[local_candidate];
        contribution = fmaf(
            probability_tile[local_candidate],
            read_float(grad_output, gradient_base + feature),
            contribution);
      }
      const int64_t output =
          ((static_cast<int64_t>(batch) * seq_len + route) *
               num_value_heads +
           value_head) * value_dim + feature;
      grad_value[output] += contribution;
    }
    __syncthreads();
  }
}


__global__ void tensor_slab_value_vjp_fp16_kernel(
    const c10::Half* __restrict__ grad_output,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_prior,
    const float* __restrict__ row_stats,
    const float* __restrict__ scores,
    float* __restrict__ grad_value,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float scale,
    float dropout_p,
    float inverse_keep_probability) {
#if __CUDA_ARCH__ >= 700
  constexpr int kFeatureTiles = kFusedValueDim / kTensorTile;
  constexpr int kValueWarps =
      (kValueRouteTile / kTensorTile) * kFeatureTiles;
  constexpr int kValueThreads = kValueWarps * kWarpSize;
  __shared__ __align__(16) __half probability_high[
      kTensorTile * kValueRouteTile];
  __shared__ __align__(16) __half probability_low[
      kTensorTile * kValueRouteTile];
  __shared__ __align__(16) __half gradient_tile[
      kTensorTile * kFusedValueDim];
  __shared__ __align__(16) float output_tile[
      kValueRouteTile * kFusedValueDim];

  const int route_tiles =
      (seq_len - slab_start + kValueRouteTile - 1) / kValueRouteTile;
  const int tiles_per_batch = num_value_heads * route_tiles;
  const int batch = blockIdx.x / tiles_per_batch;
  const int local_tile = blockIdx.x - batch * tiles_per_batch;
  const int value_head = local_tile / route_tiles;
  const int route_tile = local_tile - value_head * route_tiles;
  if (batch >= batch_size) {
    return;
  }
  const int route_start = 1 + route_tile * kValueRouteTile;
  const int heads_per_value = num_heads / num_value_heads;
  const int warp = threadIdx.x / kWarpSize;
  const int output_route = (warp / kFeatureTiles) * kTensorTile;
  const int feature_start = (warp % kFeatureTiles) * kTensorTile;
  wmma::fragment<
      wmma::accumulator,
      kTensorTile,
      kTensorTile,
      kTensorTile,
      float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  const int first_query = route_start + slab_start - 1;
  const int final_query = min(
      seq_len - 1,
      route_start + kValueRouteTile - 1 + slab_start + slab_width - 2);
  const int first_query_tile =
      (first_query / kTensorTile) * kTensorTile;
  const int final_query_tile =
      (final_query / kTensorTile) * kTensorTile;
  for (int head_offset = 0;
       head_offset < heads_per_value;
       ++head_offset) {
    const int head = value_head * heads_per_value + head_offset;
    const int series = batch * num_heads + head;
    for (int query_start = first_query_tile;
         query_start <= final_query_tile;
         query_start += kTensorTile) {
      for (int index = threadIdx.x;
           index < kTensorTile * kValueRouteTile;
           index += kValueThreads) {
        const int query_offset = index / kValueRouteTile;
        const int route_offset = index - query_offset * kValueRouteTile;
        const int query_position = query_start + query_offset;
        const int route_position = route_start + route_offset;
        const int delta = query_position - route_position + 1;
        float probability = 0.0f;
        if (query_position >= 1 && query_position < seq_len &&
            route_position >= 1 && route_position < seq_len &&
            delta >= slab_start &&
            delta < slab_start + slab_width) {
          const int slot = delta - slab_start;
          const float raw_score = scores[workspace_index(
              series,
              query_position,
              slot,
              seq_len,
              workspace_stride)];
          const ScoreTransform transformed = transform_score(raw_score);
          const int64_t stats_index =
              (static_cast<int64_t>(series) * seq_len + query_position) *
              kRowStatsWidth;
          probability = __expf(
              transformed.route_score * scale -
              row_prior[query_position] -
              row_stats[stats_index + kRowMaximum]) *
              row_stats[stats_index + kRowNormalizer];
          probability *= attention_dropout_scale(
              dropout_seed,
              dropout_p,
              inverse_keep_probability,
              batch,
              head,
              query_position,
              route_position);
        }
        const __half high = __float2half_rn(probability);
        probability_high[index] = high;
        probability_low[index] = __float2half_rn(
            probability - __half2float(high));
      }
      for (int index = threadIdx.x;
           index < kTensorTile * kFusedValueDim;
           index += kValueThreads) {
        const int query_offset = index / kFusedValueDim;
        const int feature = index - query_offset * kFusedValueDim;
        const int query_position = query_start + query_offset;
        float gradient = 0.0f;
        if (query_position >= 0 && query_position < seq_len) {
          const int64_t source =
              ((static_cast<int64_t>(batch) * seq_len + query_position) *
                   num_heads +
               head) * kFusedValueDim + feature;
          gradient = read_float(grad_output, source);
        }
        gradient_tile[index] = __float2half_rn(gradient);
      }
      __syncthreads();

      wmma::fragment<
          wmma::matrix_a,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          __half,
          wmma::col_major>
          probability;
      wmma::fragment<
          wmma::matrix_b,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          __half,
          wmma::row_major>
          gradient;
      wmma::load_matrix_sync(
          probability, probability_high + output_route, kValueRouteTile);
      wmma::load_matrix_sync(
          gradient,
          gradient_tile + feature_start,
          kFusedValueDim);
      wmma::mma_sync(accumulator, probability, gradient, accumulator);
      wmma::load_matrix_sync(
          probability, probability_low + output_route, kValueRouteTile);
      wmma::mma_sync(accumulator, probability, gradient, accumulator);
      __syncthreads();
    }
  }

  wmma::store_matrix_sync(
      output_tile + output_route * kFusedValueDim + feature_start,
      accumulator,
      kFusedValueDim,
      wmma::mem_row_major);
  __syncthreads();
  for (int index = threadIdx.x;
       index < kValueRouteTile * kFusedValueDim;
       index += kValueThreads) {
    const int route_offset = index / kFusedValueDim;
    const int feature = index - route_offset * kFusedValueDim;
    const int route = route_start + route_offset;
    if (route < seq_len) {
      const int64_t output =
          ((static_cast<int64_t>(batch) * seq_len + route) *
               num_value_heads +
           value_head) * kFusedValueDim + feature;
      grad_value[output] += output_tile[index];
    }
  }
#endif
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
  const int64_t total = query_items + key_items + value_items;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (index < query_items) {
      grad_query[index] *= softsign_derivative(read_float(query, index));
    } else if (index < query_items + key_items) {
      const int64_t key_index = index - query_items;
      grad_key[key_index] *= softsign_derivative(read_float(key, key_index));
    } else {
      const int64_t value_index = index - query_items - key_items;
      grad_value[value_index] *=
          softsign_derivative(read_float(value, value_index));
    }
  }
}


struct SlabShape {
  int total_diagonals;
  int workspace_stride;
  int row_tiles;
  int diagonal_tiles;
};


SlabShape slab_shape(int seq_len, int slab_size) {
  const int total_diagonals = std::max(0, seq_len - 1);
  const int maximum_diagonals =
      std::min(total_diagonals, slab_size);
  const int workspace_stride = std::max(
      1, maximum_diagonals);
  return {
      total_diagonals,
      workspace_stride,
      (seq_len + kWorkspaceRowTile - 1) / kWorkspaceRowTile,
      (workspace_stride + kWorkspaceDiagonalTile - 1) /
          kWorkspaceDiagonalTile};
}


torch::Tensor slabbed_replay_stats(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    const torch::Tensor& row_prior,
    int symbol_dim,
    float scale,
    float dropout_p,
    float mismatch_scale,
    bool compute_utility,
    int slab_size) {
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int num_value_heads = value.size(2);
  const int value_dim = value.size(3);
  const int series_count = batch_size * num_heads;
  const SlabShape shape = slab_shape(seq_len, slab_size);
  const auto options = value.options().dtype(torch::kFloat32);
  torch::Tensor scores = torch::empty(
      {series_count,
       shape.diagonal_tiles,
       shape.row_tiles,
       kWorkspaceRowTile,
       kWorkspaceDiagonalTile},
      options);
  torch::Tensor utilities = compute_utility
      ? torch::empty_like(scores)
      : torch::empty({0}, options);
  torch::Tensor row_stats = torch::empty(
      {batch_size, num_heads, seq_len, kRowStatsWidth}, options);
  const int64_t total_rows =
      static_cast<int64_t>(series_count) * seq_len;
  const int linear_blocks = static_cast<int>(
      (total_rows + kLinearThreads - 1) / kLinearThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  initialize_stats_kernel<<<linear_blocks, kLinearThreads, 0, stream>>>(
      row_stats.data_ptr<float>(), total_rows, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (seq_len > 1) {
    const float mismatch_unit =
        mismatch_scale / static_cast<float>(symbol_dim);
    const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);
    DISPATCH_ROSA_FLOAT_TYPES(
        value.scalar_type(),
        "rosa_soft_slabbed_replay_stats",
        [&] {
          for (int slab_start = 1;
               slab_start < seq_len;
               slab_start += slab_size) {
            const int slab_width = std::min(
                slab_size, seq_len - slab_start);
            const int blocks_per_series =
                (slab_width + kWarpsPerBlock - 1) / kWarpsPerBlock;
            const int diagonal_blocks = series_count * blocks_per_series;
            slab_diagonal_scores_kernel<<<
                diagonal_blocks, kThreads, 0, stream>>>(
                packed_query_symbols.data_ptr<int32_t>(),
                packed_key_symbols.data_ptr<int32_t>(),
                scores.data_ptr<float>(),
                series_count,
                seq_len,
                symbol_dim,
                slab_start,
                slab_width,
                shape.workspace_stride,
                mismatch_unit);
            if (compute_utility) {
              const int row_tiles =
                  (seq_len + kUtilityTile - 1) / kUtilityTile;
              const int diagonal_tiles =
                  (slab_width + kUtilityTile - 1) / kUtilityTile;
              slab_utilities_tiled_kernel<scalar_t><<<
                  series_count * row_tiles * diagonal_tiles,
                  kUtilityThreads,
                  0,
                  stream>>>(
                  value.data_ptr<scalar_t>(),
                  grad_output.data_ptr<scalar_t>(),
                  utilities.data_ptr<float>(),
                  series_count,
                  seq_len,
                  num_heads,
                  num_value_heads,
                  value_dim,
                  slab_start,
                  slab_width,
                  shape.workspace_stride);
            }
            accumulate_slab_stats_kernel<<<
                series_count * (seq_len - slab_start),
                kThreads,
                0,
                stream>>>(
                dropout_seed.data_ptr<int64_t>(),
                row_prior.data_ptr<float>(),
                scores.data_ptr<float>(),
                compute_utility ? utilities.data_ptr<float>() : nullptr,
                row_stats.data_ptr<float>(),
                seq_len,
                num_heads,
                slab_start,
                slab_width,
                shape.workspace_stride,
                scale,
                dropout_p,
                inverse_keep_probability,
                compute_utility ? 1 : 0);
          }
        });
  }
  finalize_stats_kernel<<<linear_blocks, kLinearThreads, 0, stream>>>(
      row_stats.data_ptr<float>(), total_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return row_stats;
}


torch::Tensor fused_replay_stats_fp16(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    const torch::Tensor& row_prior,
    int symbol_dim,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int slab_size) {
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int num_value_heads = value.size(2);
  const int series_count = batch_size * num_heads;
  const int partial_tile_stride =
      (slab_size + kFusedDiagonals - 1) / kFusedDiagonals;
  const auto options = value.options().dtype(torch::kFloat32);
  torch::Tensor partial_stats = torch::empty(
      {series_count, seq_len, partial_tile_stride, kRowStatsWidth},
      options);
  torch::Tensor row_stats = torch::empty(
      {batch_size, num_heads, seq_len, kRowStatsWidth}, options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t total_rows =
      static_cast<int64_t>(series_count) * seq_len;
  const int linear_blocks = static_cast<int>(
      (total_rows + kLinearThreads - 1) / kLinearThreads);
  initialize_stats_kernel<<<linear_blocks, kLinearThreads, 0, stream>>>(
      row_stats.data_ptr<float>(), total_rows, scale);
  if (seq_len <= 1) {
    finalize_stats_kernel<<<linear_blocks, kLinearThreads, 0, stream>>>(
        row_stats.data_ptr<float>(), total_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return row_stats;
  }
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);

  TORCH_INTERNAL_ASSERT(value.scalar_type() == torch::kFloat16);
  for (int slab_start = 1;
       slab_start < seq_len;
       slab_start += slab_size) {
    const int slab_width = std::min(
        slab_size, seq_len - slab_start);
    const int diagonal_tile_count =
        (slab_width + kFusedDiagonals - 1) / kFusedDiagonals;
    fused_slab_stats_fp16_kernel<<<
        series_count * diagonal_tile_count,
        kFusedThreads,
        0,
        stream>>>(
        value.data_ptr<c10::Half>(),
        grad_output.data_ptr<c10::Half>(),
        packed_query_symbols.data_ptr<int32_t>(),
        packed_key_symbols.data_ptr<int32_t>(),
        dropout_seed.data_ptr<int64_t>(),
        row_prior.data_ptr<float>(),
        partial_stats.data_ptr<float>(),
        series_count,
        seq_len,
        num_heads,
        num_value_heads,
        symbol_dim,
        slab_start,
        slab_width,
        diagonal_tile_count,
        partial_tile_stride,
        mismatch_unit,
        scale,
        dropout_p,
        inverse_keep_probability);
    constexpr int kRowsPerBlock = kFusedThreads / kWarpSize;
    const int row_blocks =
        (seq_len - slab_start + kRowsPerBlock - 1) / kRowsPerBlock;
    merge_fused_slab_stats_kernel<<<
        series_count * row_blocks, kFusedThreads, 0, stream>>>(
        partial_stats.data_ptr<float>(),
        row_stats.data_ptr<float>(),
        series_count,
        seq_len,
        slab_start,
        diagonal_tile_count,
        partial_tile_stride,
        kFusedDiagonals);
  }
  finalize_stats_kernel<<<linear_blocks, kLinearThreads, 0, stream>>>(
      row_stats.data_ptr<float>(), total_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return row_stats;
}

#ifdef ROSA_SOFT_BENCHMARK_COMPAT
template <int MacroDiagonals>
torch::Tensor macro_replay_stats_fp16(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    const torch::Tensor& row_prior,
    int symbol_dim,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int slab_size) {
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int num_value_heads = value.size(2);
  const int series_count = batch_size * num_heads;
  const int partial_tile_stride =
      (slab_size + MacroDiagonals - 1) / MacroDiagonals;
  const auto options = value.options().dtype(torch::kFloat32);
  torch::Tensor partial_stats = torch::empty(
      {series_count, seq_len, partial_tile_stride, kRowStatsWidth},
      options);
  torch::Tensor row_stats = torch::empty(
      {batch_size, num_heads, seq_len, kRowStatsWidth}, options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t total_rows =
      static_cast<int64_t>(series_count) * seq_len;
  const int linear_blocks = static_cast<int>(
      (total_rows + kLinearThreads - 1) / kLinearThreads);
  initialize_stats_kernel<<<linear_blocks, kLinearThreads, 0, stream>>>(
      row_stats.data_ptr<float>(), total_rows, scale);
  if (seq_len <= 1) {
    finalize_stats_kernel<<<linear_blocks, kLinearThreads, 0, stream>>>(
        row_stats.data_ptr<float>(), total_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return row_stats;
  }

  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);
  for (int slab_start = 1;
       slab_start < seq_len;
       slab_start += slab_size) {
    const int slab_width = std::min(slab_size, seq_len - slab_start);
    const int macro_tile_count =
        (slab_width + MacroDiagonals - 1) / MacroDiagonals;
    macro_slab_stats_fp16_kernel<MacroDiagonals><<<
        series_count * macro_tile_count,
        kFusedThreads,
        0,
        stream>>>(
        value.data_ptr<c10::Half>(),
        grad_output.data_ptr<c10::Half>(),
        packed_query_symbols.data_ptr<int32_t>(),
        packed_key_symbols.data_ptr<int32_t>(),
        dropout_seed.data_ptr<int64_t>(),
        row_prior.data_ptr<float>(),
        partial_stats.data_ptr<float>(),
        series_count,
        seq_len,
        num_heads,
        num_value_heads,
        symbol_dim,
        slab_start,
        slab_width,
        macro_tile_count,
        partial_tile_stride,
        mismatch_unit,
        scale,
        dropout_p,
        inverse_keep_probability);
    constexpr int kRowsPerBlock = kFusedThreads / kWarpSize;
    const int row_blocks =
        (seq_len - slab_start + kRowsPerBlock - 1) / kRowsPerBlock;
    merge_fused_slab_stats_kernel<<<
        series_count * row_blocks, kFusedThreads, 0, stream>>>(
        partial_stats.data_ptr<float>(),
        row_stats.data_ptr<float>(),
        series_count,
        seq_len,
        slab_start,
        macro_tile_count,
        partial_tile_stride,
        MacroDiagonals);
  }
  finalize_stats_kernel<<<linear_blocks, kLinearThreads, 0, stream>>>(
      row_stats.data_ptr<float>(), total_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return row_stats;
}
#endif

}  // namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_slabbed_replay_vjp_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask,
    int execution_plan,
    int slab_size) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  const int batch_size = query.size(0);
  const int seq_len = query.size(1);
  const int num_heads = query.size(2);
  const int symbol_dim = query.size(3);
  const int num_value_heads = value.size(2);
  const int value_dim = value.size(3);
  const int series_count = batch_size * num_heads;
  const SlabShape shape = slab_shape(seq_len, slab_size);
  const auto options = query.options().dtype(torch::kFloat32);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor row_prior = torch::empty({seq_len}, options);
  initialize_row_prior_kernel<<<
      (seq_len + kLinearThreads - 1) / kLinearThreads,
      kLinearThreads,
      0,
      stream>>>(row_prior.data_ptr<float>(), seq_len);
  torch::Tensor grad_query = (gradient_mask & kGradQuery) != 0
      ? torch::zeros(query.sizes(), options)
      : torch::empty({0}, options);
  torch::Tensor grad_key = (gradient_mask & kGradKey) != 0
      ? torch::zeros(key.sizes(), options)
      : torch::empty({0}, options);
  torch::Tensor grad_value = (gradient_mask & kGradValue) != 0
      ? torch::zeros(value.sizes(), options)
      : torch::empty({0}, options);
  const bool needs_symbols =
      (gradient_mask & (kGradQuery | kGradKey)) != 0;
  const bool tensor_shape_supported =
      value.scalar_type() == torch::kFloat16 && value_dim == kFusedValueDim;
  const bool fused_shape_supported =
      tensor_shape_supported && needs_symbols;
  const bool use_fused_stats =
      fused_shape_supported && (execution_plan & kPlanFusedStats) != 0;
  const bool use_fused_reverse =
      fused_shape_supported && (execution_plan & kPlanFusedReverse) != 0;
  const bool use_tensor_value =
      tensor_shape_supported && (execution_plan & kPlanTensorValue) != 0;
  const bool use_tiled_symbol =
      needs_symbols && (execution_plan & kPlanTiledSymbols) != 0;
  torch::Tensor row_stats = use_fused_stats
      ? fused_replay_stats_fp16(
            value,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            row_prior,
            symbol_dim,
            scale,
            dropout_p,
            mismatch_scale,
            slab_size)
      : slabbed_replay_stats(
            value,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            row_prior,
            symbol_dim,
            scale,
            dropout_p,
            mismatch_scale,
            needs_symbols,
            slab_size);
  if (seq_len <= 1) {
    return std::make_tuple(grad_query, grad_key, grad_value);
  }

  torch::Tensor scores = torch::empty(
      {series_count,
       shape.diagonal_tiles,
       shape.row_tiles,
       kWorkspaceRowTile,
       kWorkspaceDiagonalTile},
      options);
  torch::Tensor utilities = needs_symbols && !use_fused_reverse
      ? torch::empty_like(scores)
      : torch::empty({0}, options);
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  const float symbol_scale =
      0.5f * mismatch_scale / static_cast<float>(symbol_dim);
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);
  const int64_t symbol_tensor_items = query.numel();
  const int64_t symbol_output_items = 2 * symbol_tensor_items;
  const int symbol_blocks = static_cast<int>(std::min<int64_t>(
      65535,
      (symbol_output_items + kLinearThreads - 1) / kLinearThreads));
  constexpr int kSymbolOwnerWarps = kFusedThreads / kWarpSize;
  constexpr int kBitsPerSymbolWarp = 8;
  const int symbol_bit_groups =
      (symbol_dim + kBitsPerSymbolWarp - 1) / kBitsPerSymbolWarp;
  const int tiled_symbol_blocks = static_cast<int>(
      (2 * static_cast<int64_t>(series_count) * seq_len *
           symbol_bit_groups +
       kSymbolOwnerWarps - 1) /
      kSymbolOwnerWarps);
  const int final_slab_start = 1 +
      ((shape.total_diagonals - 1) / slab_size) * slab_size;

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_slabbed_replay_vjp",
      [&] {
        for (int slab_start = final_slab_start;
             slab_start >= 1;
             slab_start -= slab_size) {
          const int slab_width = std::min(
              slab_size, seq_len - slab_start);
          const int blocks_per_series =
              (slab_width + kWarpsPerBlock - 1) / kWarpsPerBlock;
          const int diagonal_blocks = series_count * blocks_per_series;
          slab_diagonal_scores_kernel<<<
              diagonal_blocks, kThreads, 0, stream>>>(
              packed_query_symbols.data_ptr<int32_t>(),
              packed_key_symbols.data_ptr<int32_t>(),
              scores.data_ptr<float>(),
              series_count,
              seq_len,
              symbol_dim,
              slab_start,
              slab_width,
              shape.workspace_stride,
              mismatch_unit);
          if (needs_symbols && !use_fused_reverse) {
            const int row_tiles =
                (seq_len + kUtilityTile - 1) / kUtilityTile;
            const int diagonal_tiles =
                (slab_width + kUtilityTile - 1) / kUtilityTile;
            slab_utilities_tiled_kernel<scalar_t><<<
                series_count * row_tiles * diagonal_tiles,
                kUtilityThreads,
                0,
                stream>>>(
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                utilities.data_ptr<float>(),
                series_count,
                seq_len,
                num_heads,
                num_value_heads,
                value_dim,
                slab_start,
                slab_width,
                shape.workspace_stride);
          }
          if ((gradient_mask & kGradValue) != 0) {
            if (use_tensor_value) {
              const int value_route_tiles =
                  (seq_len - slab_start + kValueRouteTile - 1) /
                  kValueRouteTile;
              tensor_slab_value_vjp_fp16_kernel<<<
                  batch_size * num_value_heads * value_route_tiles,
                  8 * kWarpSize,
                  0,
                  stream>>>(
                  grad_output.data_ptr<c10::Half>(),
                  dropout_seed.data_ptr<int64_t>(),
                  row_prior.data_ptr<float>(),
                  row_stats.data_ptr<float>(),
                  scores.data_ptr<float>(),
                  grad_value.data_ptr<float>(),
                  batch_size,
                  seq_len,
                  num_heads,
                  num_value_heads,
                  slab_start,
                  slab_width,
                  shape.workspace_stride,
                  scale,
                  dropout_p,
                  inverse_keep_probability);
            } else {
              const int value_blocks =
                  batch_size * num_value_heads * (seq_len - slab_start);
              accumulate_slab_value_vjp_kernel<scalar_t><<<
                  value_blocks, kValueOwnerThreads, 0, stream>>>(
                  grad_output.data_ptr<scalar_t>(),
                  dropout_seed.data_ptr<int64_t>(),
                  row_prior.data_ptr<float>(),
                  row_stats.data_ptr<float>(),
                  scores.data_ptr<float>(),
                  grad_value.data_ptr<float>(),
                  batch_size,
                  seq_len,
                  num_heads,
                  num_value_heads,
                  value_dim,
                  slab_start,
                  slab_width,
                  shape.workspace_stride,
                  scale,
                  dropout_p,
                  inverse_keep_probability);
            }
          }
          if (needs_symbols) {
            if (use_fused_reverse) {
              const int fused_diagonal_tiles =
                  (slab_width + kFusedDiagonals - 1) /
                  kFusedDiagonals;
              fused_slab_reverse_fp16_kernel<<<
                  series_count * fused_diagonal_tiles,
                  kFusedThreads,
                  0,
                  stream>>>(
                  value.data_ptr<c10::Half>(),
                  grad_output.data_ptr<c10::Half>(),
                  packed_query_symbols.data_ptr<int32_t>(),
                  packed_key_symbols.data_ptr<int32_t>(),
                  dropout_seed.data_ptr<int64_t>(),
                  row_prior.data_ptr<float>(),
                  row_stats.data_ptr<float>(),
                  scores.data_ptr<float>(),
                  series_count,
                  seq_len,
                  num_heads,
                  num_value_heads,
                  symbol_dim,
                  slab_start,
                  slab_width,
                  fused_diagonal_tiles,
                  shape.workspace_stride,
                  mismatch_unit,
                  scale,
                  dropout_p,
                  inverse_keep_probability);
            } else {
              slab_diagonal_reverse_kernel<<<
                  diagonal_blocks, kThreads, 0, stream>>>(
                  packed_query_symbols.data_ptr<int32_t>(),
                  packed_key_symbols.data_ptr<int32_t>(),
                  scores.data_ptr<float>(),
                  utilities.data_ptr<float>(),
                  dropout_seed.data_ptr<int64_t>(),
                  row_prior.data_ptr<float>(),
                  row_stats.data_ptr<float>(),
                  series_count,
                  seq_len,
                  num_heads,
                  symbol_dim,
                  slab_start,
                  slab_width,
                  shape.workspace_stride,
                  mismatch_unit,
                  scale,
                  dropout_p,
                  inverse_keep_probability);
            }
            if (use_tiled_symbol) {
              accumulate_slab_symbol_vjp_tiled_kernel<<<
                  tiled_symbol_blocks, kFusedThreads, 0, stream>>>(
                  packed_query_symbols.data_ptr<int32_t>(),
                  packed_key_symbols.data_ptr<int32_t>(),
                  scores.data_ptr<float>(),
                  grad_query.numel() != 0
                      ? grad_query.data_ptr<float>()
                      : nullptr,
                  grad_key.numel() != 0
                      ? grad_key.data_ptr<float>()
                      : nullptr,
                  series_count,
                  seq_len,
                  num_heads,
                  symbol_dim,
                  slab_start,
                  slab_width,
                  shape.workspace_stride,
                  symbol_scale,
                  gradient_mask);
            } else {
              accumulate_slab_symbol_vjp_kernel<<<
                  symbol_blocks, kLinearThreads, 0, stream>>>(
                  packed_query_symbols.data_ptr<int32_t>(),
                  packed_key_symbols.data_ptr<int32_t>(),
                  scores.data_ptr<float>(),
                  grad_query.numel() != 0
                      ? grad_query.data_ptr<float>()
                      : nullptr,
                  grad_key.numel() != 0
                      ? grad_key.data_ptr<float>()
                      : nullptr,
                  symbol_output_items,
                  batch_size,
                  seq_len,
                  num_heads,
                  symbol_dim,
                  slab_start,
                  slab_width,
                  shape.workspace_stride,
                  symbol_scale,
                  gradient_mask);
            }
          }
        }

        const int64_t query_items = grad_query.numel();
        const int64_t key_items = grad_key.numel();
        const int64_t value_items = grad_value.numel();
        const int64_t final_items = query_items + key_items + value_items;
        if (final_items != 0) {
          const int final_blocks = static_cast<int>(std::min<int64_t>(
              65535,
              (final_items + kLinearThreads - 1) / kLinearThreads));
          finalize_vjp_kernel<scalar_t><<<
              final_blocks, kLinearThreads, 0, stream>>>(
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


namespace {

constexpr int kAutomaticMaximumSlabSize = 8192;
constexpr int64_t kAutomaticSlabWorkspaceBudgetBytes = int64_t{1} << 30;


int automatic_slab_size(
    int series_count,
    int seq_len,
    int workspace_count) {
  const int diagonal_count = std::max(1, seq_len - 1);
  const int64_t padded_rows =
      ((static_cast<int64_t>(seq_len) + kWorkspaceRowTile - 1) /
       kWorkspaceRowTile) * kWorkspaceRowTile;
  const int64_t bytes_per_slot =
      static_cast<int64_t>(workspace_count) * series_count * padded_rows *
      sizeof(float);
  int64_t budget_slots = bytes_per_slot == 0
      ? kAutomaticMaximumSlabSize
      : kAutomaticSlabWorkspaceBudgetBytes / bytes_per_slot;
  budget_slots = std::max<int64_t>(1, budget_slots);
  const int alignment = budget_slots >= kWarpSize
      ? kWarpSize
      : kWorkspaceDiagonalTile;
  if (budget_slots >= alignment) {
    budget_slots = (budget_slots / alignment) * alignment;
  }
  return std::min({
      diagonal_count,
      kAutomaticMaximumSlabSize,
      static_cast<int>(budget_slots)});
}


bool should_fuse_slab_replay(int series_count, int seq_len) {
  if (seq_len < 2048) {
    return false;
  }
  return series_count >= 8 ||
      (seq_len >= 4096 &&
       static_cast<int64_t>(series_count) * seq_len >= 8192);
}


constexpr int kGroupedStatsMaximumSlabSize = 4096;
constexpr int64_t kGroupedStatsWorkspaceBudgetBytes = int64_t{64} << 20;


int automatic_grouped_stats_size(int series_count, int seq_len) {
  const int64_t bytes_per_diagonal_tile =
      static_cast<int64_t>(series_count) * seq_len * kRowStatsWidth *
      sizeof(float);
  const int64_t tile_count = std::max<int64_t>(
      1, kGroupedStatsWorkspaceBudgetBytes / bytes_per_diagonal_tile);
  const int64_t slab_size = std::min<int64_t>(
      kGroupedStatsMaximumSlabSize, tile_count * kFusedDiagonals);
  return std::min(seq_len, static_cast<int>(slab_size));
}


#ifdef ROSA_SOFT_BENCHMARK_COMPAT
__global__ void contiguous_group_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ scores,
    int series_count,
    int seq_len,
    int symbol_dim,
    int group_start,
    int group_width,
    float mismatch_unit) {
  const int blocks_per_series =
      (group_width + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int series = blockIdx.x / blocks_per_series;
  const int diagonal_block = blockIdx.x - series * blocks_per_series;
  if (series >= series_count) {
    return;
  }
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int local_diagonal = diagonal_block * kWarpsPerBlock + warp;
  if (local_diagonal >= group_width) {
    return;
  }
  const int delta = group_start + local_diagonal;
  const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  float incoming_score = 0.0f;
  const int first_row_start = (delta / kWarpSize) * kWarpSize;
  for (int row_start = first_row_start;
       row_start < seq_len;
       row_start += kWarpSize) {
    const int row = row_start + lane;
    const bool active = row >= delta && row < seq_len;
    const int key_position = row - delta;
    const int mismatch = active
        ? __popc(
              (static_cast<uint32_t>(
                   packed_query[series_offset + row]) ^
               static_cast<uint32_t>(
                   packed_key[series_offset + key_position])) &
              mask)
        : 0;
    const float gate = active
        ? __expf(-mismatch_unit * static_cast<float>(mismatch))
        : 1.0f;
    float coefficient = gate;
    float bias = active ? gate : 0.0f;
    warp_forward_affine_scan(coefficient, bias);
    const float score = fmaf(coefficient, incoming_score, bias);
    if (active) {
      scores[(series_offset + row) * group_width + local_diagonal] = score;
    }
    incoming_score = fmaf(
        __shfl_sync(0xffffffffu, coefficient, kWarpSize - 1),
        incoming_score,
        __shfl_sync(0xffffffffu, bias, kWarpSize - 1));
  }
}


#endif

}  // namespace


torch::Tensor rosa_soft_grouped_checkpoint_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int symbol_dim,
    float scale,
    float dropout_p,
    float mismatch_scale) {
  const c10::cuda::CUDAGuard device_guard(value.device());
  const int seq_len = packed_query_symbols.size(2);
  const int series_count =
      packed_query_symbols.size(0) * packed_query_symbols.size(1);
  const auto float_options = value.options().dtype(torch::kFloat32);
  torch::Tensor row_prior = torch::empty({seq_len}, float_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  initialize_row_prior_kernel<<<
      (seq_len + kLinearThreads - 1) / kLinearThreads,
      kLinearThreads,
      0,
      stream>>>(row_prior.data_ptr<float>(), seq_len);
  torch::Tensor row_stats = fused_replay_stats_fp16(
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      row_prior,
      symbol_dim,
      scale,
      dropout_p,
      mismatch_scale,
      automatic_grouped_stats_size(series_count, seq_len));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return row_stats;
}


#ifdef ROSA_SOFT_BENCHMARK_COMPAT
torch::Tensor rosa_soft_macro_checkpoint_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int symbol_dim,
    int macro_diagonals,
    int slab_size,
    float scale,
    float dropout_p,
    float mismatch_scale) {
  const c10::cuda::CUDAGuard device_guard(value.device());
  const int seq_len = packed_query_symbols.size(2);
  const auto float_options = value.options().dtype(torch::kFloat32);
  torch::Tensor row_prior = torch::empty({seq_len}, float_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  initialize_row_prior_kernel<<<
      (seq_len + kLinearThreads - 1) / kLinearThreads,
      kLinearThreads,
      0,
      stream>>>(row_prior.data_ptr<float>(), seq_len);
  switch (macro_diagonals) {
    case 32:
      return macro_replay_stats_fp16<32>(
          value,
          grad_output,
          packed_query_symbols,
          packed_key_symbols,
          dropout_seed,
          row_prior,
          symbol_dim,
          scale,
          dropout_p,
          mismatch_scale,
          slab_size);
    case 64:
      return macro_replay_stats_fp16<64>(
          value,
          grad_output,
          packed_query_symbols,
          packed_key_symbols,
          dropout_seed,
          row_prior,
          symbol_dim,
          scale,
          dropout_p,
          mismatch_scale,
          slab_size);
    case 96:
      return macro_replay_stats_fp16<96>(
          value,
          grad_output,
          packed_query_symbols,
          packed_key_symbols,
          dropout_seed,
          row_prior,
          symbol_dim,
          scale,
          dropout_p,
          mismatch_scale,
          slab_size);
    default:
      TORCH_CHECK(false, "macro_diagonals must be 32, 64, or 96");
  }
}


torch::Tensor rosa_soft_unbounded_group_scores_cuda(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t group_start,
    int64_t group_width,
    float mismatch_scale) {
  const c10::cuda::CUDAGuard device_guard(
      packed_query_symbols.device());
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int series_count = batch_size * num_heads;
  torch::Tensor scores = torch::zeros(
      {batch_size, num_heads, seq_len, group_width},
      packed_query_symbols.options().dtype(torch::kFloat32));
  const int blocks_per_series =
      (static_cast<int>(group_width) + kWarpsPerBlock - 1) /
      kWarpsPerBlock;
  contiguous_group_scores_kernel<<<
      series_count * blocks_per_series,
      kThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      packed_query_symbols.data_ptr<int32_t>(),
      packed_key_symbols.data_ptr<int32_t>(),
      scores.data_ptr<float>(),
      series_count,
      seq_len,
      static_cast<int>(symbol_dim),
      static_cast<int>(group_start),
      static_cast<int>(group_width),
      mismatch_scale / static_cast<float>(symbol_dim));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}


torch::Tensor rosa_soft_unbounded_replay_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t group_size,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int compute_utility) {
  const int seq_len = packed_query_symbols.size(2);
  torch::Tensor row_prior = torch::empty(
      {seq_len}, value.options().dtype(torch::kFloat32));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  initialize_row_prior_kernel<<<
      (seq_len + kLinearThreads - 1) / kLinearThreads,
      kLinearThreads,
      0,
      stream>>>(row_prior.data_ptr<float>(), seq_len);
  const bool use_fused_stats =
      compute_utility != 0 && value.scalar_type() == torch::kFloat16 &&
      value.size(3) == kFusedValueDim;
  torch::Tensor row_stats = use_fused_stats
      ? fused_replay_stats_fp16(
            value,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            row_prior,
            static_cast<int>(symbol_dim),
            scale,
            dropout_p,
            mismatch_scale,
            static_cast<int>(group_size))
      : slabbed_replay_stats(
            value,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            row_prior,
            static_cast<int>(symbol_dim),
            scale,
            dropout_p,
            mismatch_scale,
            compute_utility != 0,
            static_cast<int>(group_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return row_stats;
}
#endif


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_unbounded_replay_vjp_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t requested_slab_size,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask) {
  const int series_count = query.size(0) * query.size(2);
  const int seq_len = query.size(1);
  const bool needs_symbols =
      (gradient_mask & (kGradQuery | kGradKey)) != 0;
  const bool fp16_value_tiles =
      query.scalar_type() == torch::kFloat16 && value.size(3) == 64;
  const bool fuse_replay =
      fp16_value_tiles && needs_symbols &&
      should_fuse_slab_replay(series_count, seq_len);
  const bool tensor_value =
      fp16_value_tiles && (gradient_mask & kGradValue) != 0 &&
      (seq_len >= 4096 ||
       static_cast<int64_t>(series_count) * seq_len >= 8192);
  const bool tiled_symbols = needs_symbols && query.size(3) >= 4;
  const int execution_plan =
      (fuse_replay ? kPlanFusedStats | kPlanFusedReverse : 0) |
      (tensor_value ? kPlanTensorValue : 0) |
      (tiled_symbols ? kPlanTiledSymbols : 0);
  const int workspace_count =
      needs_symbols && !fuse_replay ? 2 : 1;
  const int slab_size = requested_slab_size > 0
      ? static_cast<int>(requested_slab_size)
      : automatic_slab_size(series_count, seq_len, workspace_count);
  return rosa_soft_slabbed_replay_vjp_cuda(
      query,
      key,
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      scale,
      dropout_p,
      mismatch_scale,
      gradient_mask,
      execution_plan,
      slab_size);
}
