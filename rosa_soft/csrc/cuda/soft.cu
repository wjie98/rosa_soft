#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>
#include <mma.h>
#include <tuple>

#include "common.cuh"
#include "../soft.h"


using namespace rosa_soft::cuda;


namespace rosa::soft {
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
constexpr int kFusedUtilityStride = 72;
constexpr int kFusedValueDim = 64;
constexpr int kInputStride = 72;
constexpr int kFusedThreads = 8 * kWarpSize;
constexpr int kValueRouteTile = 32;
constexpr int kMismatchGateCount = 33;

template <int Rows, bool Binary>
__device__ __forceinline__ void load_tile(
    const c10::Half* __restrict__ input,
    __half* __restrict__ tile,
    int thread, int batch, int seq_len, int num_heads, int head, int first_row) {
  constexpr int kPairs = kFusedValueDim / 2;
  static_assert(Rows * kPairs % (2 * kFusedThreads) == 0,
                "Load pairs must cover the tile exactly");
#pragma unroll 1
  for (int base = thread; base < Rows * kPairs; base += 2 * kFusedThreads) {
    __half2 pending[2];
    bool valid[2];
    // Issue two independent loads before converting or writing either one.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int pair = base + i * kFusedThreads;
      const int row = first_row + pair / kPairs;
      valid[i] = row >= (Binary ? 1 : 0) && row < seq_len;
      pending[i] = __float2half2_rn(0.0f);
      if (valid[i]) {
        const int64_t source =
            ((static_cast<int64_t>(batch) * seq_len + row) * num_heads + head) *
                kFusedValueDim + (pair % kPairs) * 2;
        pending[i] = *reinterpret_cast<const __half2*>(input + source);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int pair = base + i * kFusedThreads;
      __half2 x = pending[i];
      if constexpr (Binary) {
        // Invalid routes stay zero; sign(0) would give -1.
        if (valid[i]) x = binary_sign_half2(x);
      }
      reinterpret_cast<__half2*>(tile)[
          (pair / kPairs) * (kInputStride / 2) + pair % kPairs] = x;
    }
  }
}

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
    const c10::Half* __restrict__ dy,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ seed,
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
      kFusedRows * kInputStride];
  __shared__ __align__(16) __half value_tile[
      kFusedRoutes * kInputStride];
  __shared__ __align__(16) float utility_tile[
      kFusedRows * kFusedUtilityStride];
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
  float score_carry = 0.0f;
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
    const int tile_width = min(
        kFusedDiagonals,
        slab_width - diagonal_tile * kFusedDiagonals);

    load_tile<kFusedRows, false>(
        dy, grad_tile, thread, batch, seq_len, num_heads, head, row_start);
    load_tile<kFusedRoutes, true>(
        value, value_tile, thread, batch, seq_len, num_value_heads, value_head, route_start);

    const int diagonal = thread / 8;
    const int delta = diagonal_start + diagonal;
    Affine prefix[4];
    Affine total{1.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = row_start + (thread % 8) * 4 + i;
      const bool active = row < seq_len && diagonal < tile_width && delta <= row;
      const float gate = active
          ? mismatch_gate_from_lut(
                packed_query[series_offset + row],
                packed_key[series_offset + row - delta], mask, gate_lut)
          : 1.0f;
      total = Compose{}(total, Affine{gate, active ? gate : 0.0f});
      prefix[i] = total;
    }
    group_affine_scan(prefix, score_carry);
    score_carry = __shfl_sync(0xffffffffu, prefix[3].b, 7, 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = (thread % 8) * 4 + i;
      if (r < row_count) {
        score_tile[r * kFusedScoreStride + diagonal] =
            diagonal < tile_width && delta <= row_start + r ? prefix[i].b : 0.0f;
      }
    }
    __syncthreads();

    constexpr int kRouteTiles = kFusedRoutes / kTensorTile;
    constexpr int kMmaWarps =
        (kFusedRows / kTensorTile) * kRouteTiles;
    // These two corners of the rectangle contain no diagonal candidates.
    if (warp < kMmaWarps && warp != 3 && warp != 4) {
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
            grad_tile + output_row * kInputStride + feature_start,
            kInputStride);
        wmma::load_matrix_sync(
            signed_value,
            value_tile + output_route * kInputStride + feature_start,
            kInputStride);
        wmma::mma_sync(accumulator, gradient, signed_value, accumulator);
      }
      wmma::store_matrix_sync(
          utility_tile + output_row * kFusedUtilityStride + output_route,
          accumulator,
          kFusedUtilityStride,
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
            seed,
            dropout_p,
            inverse_keep_probability,
            batch,
            head,
            row,
            route) * utility_tile[
                reduction_row * kFusedUtilityStride + route_offset];
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
    const scalar_t* __restrict__ dy,
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
      gradient = read_float(dy, source);
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
    const int64_t* __restrict__ seed,
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
          seed,
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
    const int64_t* __restrict__ seed,
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
          seed,
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
    const int64_t* __restrict__ seed,
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
      seed,
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
    const c10::Half* __restrict__ dy,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ seed,
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
        gradient = read_float(dy, source);
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
            seed,
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
    float* __restrict__ dq,
    float* __restrict__ dk,
    int64_t output_items,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float symbol_scale,
    int mask) {
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
      if ((mask & kGradQuery) == 0 || position < slab_start) {
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
      dq[local_output] += symbol_scale * contribution;
    } else {
      const int valid_width = min(
          slab_width, seq_len - position - slab_start);
      if ((mask & kGradKey) == 0 || valid_width <= 0) {
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
      dk[local_output] += symbol_scale * contribution;
    }
  }
}


__global__ void accumulate_slab_symbol_vjp_tiled_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const float* __restrict__ log_gate_vjp,
    float* __restrict__ dq,
    float* __restrict__ dk,
    int series_count,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int slab_start,
    int slab_width,
    int workspace_stride,
    float symbol_scale,
    int mask) {
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
  if ((query_owner && (mask & kGradQuery) == 0) ||
      (!query_owner && (mask & kGradKey) == 0)) {
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
        dq[output] += symbol_scale * total;
      } else {
        dk[output] += symbol_scale * total;
      }
    }
  }
}


template <typename scalar_t>
__global__ void accumulate_slab_value_vjp_kernel(
    const scalar_t* __restrict__ dy,
    const int64_t* __restrict__ seed,
    const float* __restrict__ row_prior,
    const float* __restrict__ row_stats,
    const float* __restrict__ scores,
    float* __restrict__ dv,
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
          seed,
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
            read_float(dy, gradient_base + feature),
            contribution);
      }
      const int64_t output =
          ((static_cast<int64_t>(batch) * seq_len + route) *
               num_value_heads +
           value_head) * value_dim + feature;
      dv[output] += contribution;
    }
    __syncthreads();
  }
}


__global__ void tensor_slab_value_vjp_fp16_kernel(
    const c10::Half* __restrict__ dy,
    const int64_t* __restrict__ seed,
    const float* __restrict__ row_prior,
    const float* __restrict__ row_stats,
    const float* __restrict__ scores,
    float* __restrict__ dv,
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
              seed,
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
          gradient = read_float(dy, source);
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
      dv[output] += output_tile[index];
    }
  }
#endif
}


template <typename scalar_t>
__global__ void finalize_vjp_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    float* __restrict__ dq,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int64_t query_items,
    int64_t key_items,
    int64_t value_items) {
  const int64_t total = query_items + key_items + value_items;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (index < query_items) {
      dq[index] *= softsign_derivative(read_float(query, index));
    } else if (index < query_items + key_items) {
      const int64_t key_index = index - query_items;
      dk[key_index] *= softsign_derivative(read_float(key, key_index));
    } else {
      const int64_t value_index = index - query_items - key_items;
      dv[value_index] *=
          softsign_derivative(read_float(value, value_index));
    }
  }
}


struct Slab {
  int stride, rows, diagonals;
  Slab(int t, int size)
      : stride(std::max(1, std::min(t - 1, size))),
        rows((t + kWorkspaceRowTile - 1) / kWorkspaceRowTile),
        diagonals((stride + kWorkspaceDiagonalTile - 1) / kWorkspaceDiagonalTile) {}
  Tensor allocate(int s, const torch::TensorOptions& options) const {
    return torch::empty(
        {s, diagonals, rows, kWorkspaceRowTile, kWorkspaceDiagonalTile}, options);
  }
};

struct Shape {
  int b, t, h, d, hv, dv, s;
  explicit Shape(const Input& x)
      : b(x.q.size(0)), t(x.q.size(1)), h(x.q.size(2)), d(x.q.size(3)),
        hv(x.v.size(2)), dv(x.v.size(3)), s(b * h) {}
};

struct Plan {
  bool fp16, fused, value, tiled;
  int slab;
};

// All shape-dependent execution choices live here; none change the estimator.
Plan plan(const Input& x, int mask) {
  const Shape shape(x);
  const int t = shape.t, d = shape.d, dv = shape.dv, s = shape.s;
  const bool symbols = (mask & 3) != 0;
  const bool half = x.q.scalar_type() == torch::kHalf && dv == 64;
  const auto* device = at::cuda::getCurrentDeviceProperties();
  const bool fp16 = half && symbols && t >= 2048 &&
      ((s >= 2 && static_cast<int64_t>(s) * t >= 8192) || t >= 32768) &&
      10 * device->major + device->minor >= 75;
  const bool fused = half && symbols && t >= 2048 &&
      (s >= 8 || (t >= 4096 && static_cast<int64_t>(s) * t >= 8192));
  const bool value = half && (mask & 4) &&
      (t >= 4096 || static_cast<int64_t>(s) * t >= 8192);
  const int buffers = symbols && !fused ? 2 : 1;
  const int64_t padded = ((int64_t(t) + kWorkspaceRowTile - 1) /
                         kWorkspaceRowTile) * kWorkspaceRowTile;
  const int64_t bytes = buffers * int64_t(s) * padded * sizeof(float);
  int64_t slots = std::max<int64_t>(1, (int64_t{1} << 30) / bytes);
  const int alignment = slots >= kWarpSize ? kWarpSize : kWorkspaceDiagonalTile;
  if (slots >= alignment) slots = slots / alignment * alignment;
  const int slab = std::min<int64_t>({std::max(1, t - 1), 8192, slots});
  return {fp16, fused, value, symbols && d >= 4, slab};
}

Tensor initialize_stats(const Input& x, const Args& a) {
  const Shape shape(x);
  const int b = shape.b, t = shape.t, h = shape.h, s = shape.s;
  auto stats = torch::empty({b, h, t, kRowStatsWidth}, x.q.options().dtype(torch::kFloat32));
  const int64_t n = int64_t(s) * t;
  initialize_stats_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      stats.data_ptr<float>(), n, a.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return stats;
}

void finalize_stats(const Tensor& stats) {
  const int64_t n = stats.numel() / kRowStatsWidth;
  finalize_stats_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      stats.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Reused by the statistics pass and backward replay, with identical launches.
void replay(const Input& x, const Args& a, const Slab& slab, int start, int width,
            const Tensor& scores, const Tensor& utilities) {
  const Shape shape(x);
  const int t = shape.t, h = shape.h, d = shape.d,
            hv = shape.hv, dv = shape.dv, s = shape.s;
  const auto stream = at::cuda::getCurrentCUDAStream();
  slab_diagonal_scores_kernel<<<s * ((width + kWarpsPerBlock - 1) / kWarpsPerBlock),
                                kThreads, 0, stream>>>(
      x.pq.data_ptr<int32_t>(), x.pk.data_ptr<int32_t>(), scores.data_ptr<float>(),
      s, t, d, start, width, slab.stride, a.mismatch / float(d));
  if (utilities.numel()) {
    const int tiles = ((t + kUtilityTile - 1) / kUtilityTile) *
                      ((width + kUtilityTile - 1) / kUtilityTile);
    DISPATCH_ROSA_FLOAT_TYPES(x.v.scalar_type(), "rosa_soft_utility", [&] {
      slab_utilities_tiled_kernel<scalar_t><<<s * tiles, kUtilityThreads, 0, stream>>>(
          x.v.data_ptr<scalar_t>(), x.dy.data_ptr<scalar_t>(), utilities.data_ptr<float>(),
          s, t, h, hv, dv, start, width, slab.stride);
    });
  }
}

Tensor replay_stats(const Input& x, const Args& a, const Tensor& prior,
                    bool symbols, int size) {
  const Shape shape(x);
  const int t = shape.t, h = shape.h, s = shape.s;
  const Slab slab(t, size);
  const auto options = x.q.options().dtype(torch::kFloat32);
  auto scores = slab.allocate(s, options);
  auto utilities = symbols ? torch::empty_like(scores) : torch::empty({0}, options);
  auto stats = initialize_stats(x, a);
  const auto stream = at::cuda::getCurrentCUDAStream();
  for (int start = 1; start < t; start += size) {
    const int width = std::min(size, t - start);
    replay(x, a, slab, start, width, scores, utilities);
    accumulate_slab_stats_kernel<<<s * (t - start), kThreads, 0, stream>>>(
        x.seed.data_ptr<int64_t>(), prior.data_ptr<float>(), scores.data_ptr<float>(),
        symbols ? utilities.data_ptr<float>() : nullptr, stats.data_ptr<float>(),
        t, h, start, width, slab.stride, a.scale, a.dropout,
        1.0f / (1.0f - a.dropout), symbols);
  }
  finalize_stats(stats);
  return stats;
}

Tensor fused_stats(const Input& x, const Args& a, const Tensor& prior, int size) {
  const Shape shape(x);
  const int t = shape.t, h = shape.h, d = shape.d, hv = shape.hv, s = shape.s;
  const int stride = (size + kFusedDiagonals - 1) / kFusedDiagonals;
  auto partial = torch::empty({s, t, stride, kRowStatsWidth},
                              x.q.options().dtype(torch::kFloat32));
  auto stats = initialize_stats(x, a);
  const auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_INTERNAL_ASSERT(x.v.scalar_type() == torch::kFloat16);
  for (int start = 1; start < t; start += size) {
    const int width = std::min(size, t - start);
    const int tiles = (width + kFusedDiagonals - 1) / kFusedDiagonals;
    fused_slab_stats_fp16_kernel<<<s * tiles, kFusedThreads, 0, stream>>>(
        x.v.data_ptr<c10::Half>(), x.dy.data_ptr<c10::Half>(),
        x.pq.data_ptr<int32_t>(), x.pk.data_ptr<int32_t>(), x.seed.data_ptr<int64_t>(),
        prior.data_ptr<float>(), partial.data_ptr<float>(), s, t, h, hv, d,
        start, width, tiles, stride, a.mismatch / float(d), a.scale, a.dropout,
        1.0f / (1.0f - a.dropout));
    constexpr int rows = kFusedThreads / kWarpSize;
    merge_fused_slab_stats_kernel<<<s * ((t - start + rows - 1) / rows),
                                   kFusedThreads, 0, stream>>>(
        partial.data_ptr<float>(), stats.data_ptr<float>(), s, t, start, tiles,
        stride, kFusedDiagonals);
  }
  finalize_stats(stats);
  return stats;
}

Grads backward_impl(const Input& x, const Args& a, int mask,
                    const Tensor& prior, const Plan& p) {
  const Shape shape(x);
  const int b = shape.b, t = shape.t, h = shape.h, d = shape.d,
            hv = shape.hv, dv = shape.dv, s = shape.s;
  const Slab slab(t, p.slab);
  auto grad = gradients(x, mask);
  auto& dq = std::get<0>(grad);
  auto& dk = std::get<1>(grad);
  auto& out_v = std::get<2>(grad);
  const bool symbols = (mask & 3) != 0;
  const auto options = x.q.options().dtype(torch::kFloat32);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto stats = p.fused ? fused_stats(x, a, prior, p.slab)
                       : replay_stats(x, a, prior, symbols, p.slab);
  if (t <= 1) return grad;
  auto scores = slab.allocate(s, options);
  auto utilities = symbols && !p.fused ? torch::empty_like(scores) : torch::empty({0}, options);
  const float gate_scale = a.mismatch / float(d), symbol_scale = .5f * a.mismatch / float(d);
  const float keep = 1.0f / (1.0f - a.dropout);
  const int64_t items = 2 * x.q.numel();
  const int symbol_blocks = std::min<int64_t>(65535, (items + 255) / 256);
  constexpr int owner_warps = kFusedThreads / kWarpSize;
  const int groups = (d + 7) / 8;
  const int tiled_blocks = (2 * int64_t(s) * t * groups + owner_warps - 1) / owner_warps;
  float* gq = dq.numel() ? dq.data_ptr<float>() : nullptr;
  float* gk = dk.numel() ? dk.data_ptr<float>() : nullptr;
  const int last = 1 + ((t - 2) / p.slab) * p.slab;
  DISPATCH_ROSA_FLOAT_TYPES(x.q.scalar_type(), "rosa_soft_backward", [&] {
    for (int start = last; start >= 1; start -= p.slab) {
      const int width = std::min(p.slab, t - start);
      replay(x, a, slab, start, width, scores, utilities);
      if (mask & kGradValue) {
        if (p.value) {
          const int tiles = (t - start + kValueRouteTile - 1) / kValueRouteTile;
          tensor_slab_value_vjp_fp16_kernel<<<b * hv * tiles, 8 * kWarpSize, 0, stream>>>(
              x.dy.data_ptr<c10::Half>(), x.seed.data_ptr<int64_t>(), prior.data_ptr<float>(),
              stats.data_ptr<float>(), scores.data_ptr<float>(), out_v.data_ptr<float>(),
              b, t, h, hv, start, width, slab.stride, a.scale, a.dropout, keep);
        } else {
          accumulate_slab_value_vjp_kernel<scalar_t><<<b * hv * (t - start),
                                                      kValueOwnerThreads, 0, stream>>>(
              x.dy.data_ptr<scalar_t>(), x.seed.data_ptr<int64_t>(), prior.data_ptr<float>(),
              stats.data_ptr<float>(), scores.data_ptr<float>(), out_v.data_ptr<float>(),
              b, t, h, hv, dv, start, width, slab.stride, a.scale, a.dropout, keep);
        }
      }
      if (symbols) {
        if (p.fused) {
          const int tiles = (width + kFusedDiagonals - 1) / kFusedDiagonals;
          fused_slab_reverse_fp16_kernel<<<s * tiles, kFusedThreads, 0, stream>>>(
              x.v.data_ptr<c10::Half>(), x.dy.data_ptr<c10::Half>(),
              x.pq.data_ptr<int32_t>(), x.pk.data_ptr<int32_t>(), x.seed.data_ptr<int64_t>(),
              prior.data_ptr<float>(), stats.data_ptr<float>(), scores.data_ptr<float>(),
              s, t, h, hv, d, start, width, tiles, slab.stride,
              gate_scale, a.scale, a.dropout, keep);
        } else {
          const int blocks = s * ((width + kWarpsPerBlock - 1) / kWarpsPerBlock);
          slab_diagonal_reverse_kernel<<<blocks, kThreads, 0, stream>>>(
              x.pq.data_ptr<int32_t>(), x.pk.data_ptr<int32_t>(), scores.data_ptr<float>(),
              utilities.data_ptr<float>(), x.seed.data_ptr<int64_t>(), prior.data_ptr<float>(),
              stats.data_ptr<float>(), s, t, h, d, start, width, slab.stride,
              gate_scale, a.scale, a.dropout, keep);
        }
        if (p.tiled) {
          accumulate_slab_symbol_vjp_tiled_kernel<<<tiled_blocks, kFusedThreads, 0, stream>>>(
              x.pq.data_ptr<int32_t>(), x.pk.data_ptr<int32_t>(), scores.data_ptr<float>(),
              gq, gk, s, t, h, d, start, width, slab.stride, symbol_scale, mask);
        } else {
          accumulate_slab_symbol_vjp_kernel<<<symbol_blocks, kLinearThreads, 0, stream>>>(
              x.pq.data_ptr<int32_t>(), x.pk.data_ptr<int32_t>(), scores.data_ptr<float>(),
              gq, gk, items, b, t, h, d, start, width, slab.stride, symbol_scale, mask);
        }
      }
    }
  });
  finish(x, grad);
  return grad;
}
}  // namespace

void finish(const Input& x, Grads& grad) {
  auto& dq = std::get<0>(grad);
  auto& dk = std::get<1>(grad);
  auto& dv = std::get<2>(grad);
  const int64_t n = dq.numel() + dk.numel() + dv.numel();
  if (n) {
    DISPATCH_ROSA_FLOAT_TYPES(x.q.scalar_type(), "rosa_soft_finish", [&] {
      finalize_vjp_kernel<scalar_t><<<std::min<int64_t>(65535, (n + 255) / 256),
                                      256, 0, at::cuda::getCurrentCUDAStream()>>>(
          x.q.data_ptr<scalar_t>(), x.k.data_ptr<scalar_t>(), x.v.data_ptr<scalar_t>(),
          dq.numel() ? dq.data_ptr<float>() : nullptr,
          dk.numel() ? dk.data_ptr<float>() : nullptr,
          dv.numel() ? dv.data_ptr<float>() : nullptr, dq.numel(), dk.numel(), dv.numel());
    });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor stats_fp16(const Input& x, const Args& a, const Tensor& prior) {
  const int64_t bytes = x.pq.numel() * kRowStatsWidth * sizeof(float);
  const int64_t tiles = std::max<int64_t>(1, (int64_t{64} << 20) / bytes);
  const int size = std::min<int64_t>({x.q.size(1), 4096, tiles * kFusedDiagonals});
  return fused_stats(x, a, prior, size);
}

Grads backward(const Input& x, const Args& a, int mask) {
  const c10::cuda::CUDAGuard guard(x.q.device());
  const int t = x.q.size(1);
  const Plan p = plan(x, mask);
  auto prior = torch::empty({t}, x.q.options().dtype(torch::kFloat32));
  initialize_row_prior_kernel<<<(t + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      prior.data_ptr<float>(), t);
  return p.fp16 ? backward_fp16(x, a, mask, prior) : backward_impl(x, a, mask, prior, p);
}
}  // namespace rosa::soft
