#include "rosa_soft_vjp_common.cuh"

#include <mma.h>

using namespace rosa_soft::cuda;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_streaming_vjp_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t max_suffix_length,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask,
    int query_tile_size);

namespace {

namespace wmma = nvcuda::wmma;

#ifndef ROSA_BLOCK_ROWS
#define ROSA_BLOCK_ROWS 64
#endif
#ifndef ROSA_BLOCK_ROUTES
#define ROSA_BLOCK_ROUTES 64
#endif
#ifndef ROSA_BLOCK_THREADS
#define ROSA_BLOCK_THREADS 256
#endif
#ifndef ROSA_BLOCK_MIN_BLOCKS
#define ROSA_BLOCK_MIN_BLOCKS 1
#endif
#ifndef ROSA_BLOCK_VALUE_DIM
#define ROSA_BLOCK_VALUE_DIM ROSA_BLOCK_ROUTES
#endif

constexpr int kBlockRows = ROSA_BLOCK_ROWS;
constexpr int kBlockRoutes = ROSA_BLOCK_ROUTES;
constexpr int kBlockThreads = ROSA_BLOCK_THREADS;
constexpr int kBlockMinBlocks = ROSA_BLOCK_MIN_BLOCKS;
constexpr int kBlockValueDim = ROSA_BLOCK_VALUE_DIM;
constexpr int kBlockWindow = 32;
constexpr int kBlockSpan = kBlockRows + kBlockWindow - 1;
constexpr int kBlockSpanStride = ((kBlockSpan + 15) / 16) * 16;
constexpr int kBlockDiagonals = kBlockRows + kBlockRoutes - 1;
constexpr int kBlockMatrixItems = kBlockRows * kBlockRoutes;
constexpr int kBlockGradItems = kBlockRows * kBlockValueDim;
constexpr int kBlockValueItems = kBlockRoutes * kBlockValueDim;
constexpr int kTensorTile = 16;
constexpr int kTensorReduction = 8;
constexpr int kTensorWarps = kBlockThreads / kWarpSize;
constexpr int kTensorWarpScratchItems =
    kTensorWarps * kTensorTile * kTensorReduction;
constexpr int kTensorMatrixScratchItems =
    kBlockSpanStride * 32 + kTensorWarpScratchItems;
constexpr int kBlockPrimaryItems =
    kBlockMatrixItems > kTensorMatrixScratchItems
    ? kBlockMatrixItems
    : kTensorMatrixScratchItems;
// WMMA reads the logical gate band through a zero-padded physical matrix.
constexpr int kBlockGateItems = kBlockSpanStride * kBlockSpanStride;
constexpr int kBlockRingStride = kBlockWindow + 1;
constexpr int kBlockRingItems = kBlockDiagonals * kBlockRingStride;
constexpr int kUtilityWorkspaceItems =
    kBlockMatrixItems + kBlockGradItems + kBlockValueItems;
constexpr int kUtilityRingOverlapItems =
    kUtilityWorkspaceItems > kBlockGateItems
    ? kUtilityWorkspaceItems - kBlockGateItems
    : 0;
constexpr int kBlockRingStorageItems =
    kBlockRingItems > kTensorMatrixScratchItems
    ? (kBlockRingItems > kUtilityRingOverlapItems
           ? kBlockRingItems
           : kUtilityRingOverlapItems)
    : (kTensorMatrixScratchItems > kUtilityRingOverlapItems
           ? kTensorMatrixScratchItems
           : kUtilityRingOverlapItems);
constexpr int kBlockPayloadItems =
    kBlockPrimaryItems + kBlockGateItems + kBlockRingStorageItems;
constexpr int kBlockStatsItems = 3 * kBlockRows;
constexpr int kBlockSharedItems = kBlockPayloadItems + kBlockStatsItems;

static_assert(kBlockRows % kTensorTile == 0);
static_assert(kBlockRoutes % kTensorTile == 0);
static_assert(kBlockValueDim % kTensorTile == 0);
static_assert(kBlockThreads % kWarpSize == 0);


struct BlockShared {
  float* primary;
  float* secondary;
  float* ring;
  float* row_maximum;
  float* row_normalizer;
  float* row_expected_utility;
};


__device__ __forceinline__ BlockShared bind_block_shared(float* storage) {
  float* primary = storage;
  float* secondary = primary + kBlockPrimaryItems;
  float* ring = secondary + kBlockGateItems;
  float* stats = storage + kBlockPayloadItems;
  return {
      primary,
      secondary,
      ring,
      stats,
      stats + kBlockRows,
      stats + 2 * kBlockRows};
}


__device__ __forceinline__ uint32_t block_symbol_mask(int symbol_dim) {
  return symbol_dim == 32
      ? 0xffffffffu
      : (1u << symbol_dim) - 1u;
}


__device__ __forceinline__ int block_mismatch(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int query_position,
    int key_position,
    uint32_t mask) {
  const uint32_t query_word = static_cast<uint32_t>(query[query_position]);
  const uint32_t key_word = static_cast<uint32_t>(key[key_position]);
  return __popc((query_word ^ key_word) & mask);
}


__device__ __forceinline__ float block_gate(
    int mismatch,
    float mismatch_unit) {
  return __expf(-mismatch_unit * static_cast<float>(mismatch));
}


__device__ __forceinline__ void initialize_block_state(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int route,
    uint32_t mask,
    float mismatch_unit,
    float& score,
    int& rolling_mismatch,
    int& rolling_count) {
  const int available = min(row + 1, route);
  rolling_count = min(kBlockWindow + 1, available);
  rolling_mismatch = 0;
  float product = 1.0f;
  score = 0.0f;
#pragma unroll
  for (int suffix = 0; suffix < kBlockWindow + 1; ++suffix) {
    if (suffix < rolling_count) {
      const int mismatch = block_mismatch(
          query,
          key,
          row - suffix,
          route - 1 - suffix,
          mask);
      rolling_mismatch += mismatch;
      if (suffix < kBlockWindow) {
        product *= block_gate(mismatch, mismatch_unit);
        score += product;
      }
    }
  }
}


__device__ __forceinline__ void advance_block_state(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int route,
    uint32_t mask,
    float mismatch_unit,
    float& score,
    int& rolling_mismatch,
    int& rolling_count) {
  const int mismatch = block_mismatch(query, key, row, route - 1, mask);
  if (rolling_count == kBlockWindow + 1) {
    rolling_mismatch -= block_mismatch(
        query,
        key,
        row - kBlockWindow - 1,
        route - kBlockWindow - 2,
        mask);
  } else {
    ++rolling_count;
  }
  rolling_mismatch += mismatch;
  const float gate = block_gate(mismatch, mismatch_unit);
  const float correction = rolling_count == kBlockWindow + 1
      ? block_gate(rolling_mismatch, mismatch_unit)
      : 0.0f;
  score = fmaf(gate, 1.0f + score, -correction);
}


__device__ __forceinline__ void generate_block_scores(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    BlockShared shared,
    int row_start,
    int row_count,
    int route_start,
    int route_count,
    uint32_t mask,
    float mismatch_unit) {
  for (int index = threadIdx.x; index < kBlockMatrixItems;
       index += blockDim.x) {
    shared.primary[index] = 0.0f;
  }
  __syncthreads();
  const int diagonal_count = row_count + route_count - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int difference = diagonal - (route_count - 1);
    const int first_row_offset = max(difference, 0);
    const int first_route_offset = max(-difference, 0);
    const int count = min(
        row_count - first_row_offset,
        route_count - first_route_offset);
    int row = row_start + first_row_offset;
    int route = route_start + first_route_offset;
    if (count <= 0 || route > row) {
      continue;
    }
    float score;
    int rolling_mismatch;
    int rolling_count;
    initialize_block_state(
        query,
        key,
        row,
        route,
        mask,
        mismatch_unit,
        score,
        rolling_mismatch,
        rolling_count);
    shared.primary[
        first_row_offset * kBlockRoutes + first_route_offset] = score;
    for (int step = 1; step < count; ++step) {
      ++row;
      ++route;
      advance_block_state(
          query,
          key,
          row,
          route,
          mask,
          mismatch_unit,
          score,
          rolling_mismatch,
          rolling_count);
      shared.primary[
          (first_row_offset + step) * kBlockRoutes +
          first_route_offset + step] = score;
    }
  }
  __syncthreads();
}


template <typename scalar_t>
__device__ __forceinline__ void generate_block_utilities_tf32(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    BlockShared shared,
    int batch,
    int head,
    int value_head,
    int row_start,
    int row_count,
    int route_start,
    int route_count,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    bool needs_qk) {
#if __CUDA_ARCH__ >= 800
  float* grad_tile = shared.secondary + kBlockMatrixItems;
  float* value_tile = grad_tile + kBlockGradItems;
  for (int index = threadIdx.x; index < kBlockGradItems;
       index += blockDim.x) {
    const int item = index / kBlockValueDim;
    const int feature = index - item * kBlockValueDim;
    float grad = 0.0f;
    if (item < row_count) {
      const int row = row_start + item;
      const int64_t source =
          ((static_cast<int64_t>(batch) * seq_len + row) * num_heads + head) *
          value_dim + feature;
      grad = read_float(grad_output, source);
    }
    grad_tile[index] = wmma::__float_to_tf32(grad);
  }
  for (int index = threadIdx.x; index < kBlockValueItems;
       index += blockDim.x) {
    const int item = index / kBlockValueDim;
    const int feature = index - item * kBlockValueDim;
    float sign = 0.0f;
    if (needs_qk && item < route_count) {
      const int route = route_start + item;
      const int64_t source =
          ((static_cast<int64_t>(batch) * seq_len + route) *
               num_value_heads +
           value_head) * value_dim + feature;
      sign = read_float(value, source) > 0.0f ? 1.0f : -1.0f;
    }
    value_tile[index] = wmma::__float_to_tf32(sign);
  }
  __syncthreads();

  const int warp = threadIdx.x / kWarpSize;
  constexpr int kRouteTiles = kBlockRoutes / kTensorTile;
  constexpr int kUtilityTiles =
      (kBlockRows / kTensorTile) * kRouteTiles;
  for (int tile = warp; tile < kUtilityTiles; tile += kTensorWarps) {
    const int output_row = (tile / kRouteTiles) * kTensorTile;
    const int output_column = (tile % kRouteTiles) * kTensorTile;
    wmma::fragment<
        wmma::accumulator,
        kTensorTile,
        kTensorTile,
        kTensorReduction,
        float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    if (needs_qk) {
      for (int reduction = 0; reduction < kBlockValueDim;
           reduction += kTensorReduction) {
        wmma::fragment<
            wmma::matrix_a,
            kTensorTile,
            kTensorTile,
            kTensorReduction,
            wmma::precision::tf32,
            wmma::row_major>
            left;
        wmma::fragment<
            wmma::matrix_b,
            kTensorTile,
            kTensorTile,
            kTensorReduction,
            wmma::precision::tf32,
            wmma::col_major>
            right;
        wmma::load_matrix_sync(
            left,
            grad_tile + output_row * kBlockValueDim + reduction,
            kBlockValueDim);
        wmma::load_matrix_sync(
            right,
            value_tile + output_column * kBlockValueDim + reduction,
            kBlockValueDim);
        wmma::mma_sync(accumulator, left, right, accumulator);
      }
    }
    wmma::store_matrix_sync(
        shared.secondary + output_row * kBlockRoutes + output_column,
        accumulator,
        kBlockRoutes,
        wmma::mem_row_major);
  }
  __syncthreads();
#endif
}


template <typename scalar_t>
__device__ __forceinline__ void accumulate_block_value_vjp_tf32(
    const scalar_t* __restrict__ grad_output,
    BlockShared shared,
    float* __restrict__ grad_value,
    int batch,
    int head,
    int value_head,
    int row_start,
    int row_count,
    int route_start,
    int route_count,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
#if __CUDA_ARCH__ >= 800
  float* grad_tile = shared.secondary + kBlockMatrixItems;
  float* output_tile = grad_tile + kBlockGradItems;
  for (int index = threadIdx.x; index < kBlockGradItems;
       index += blockDim.x) {
    const int row_offset = index / kBlockValueDim;
    const int feature = index - row_offset * kBlockValueDim;
    float grad = 0.0f;
    if (row_offset < row_count) {
      const int row = row_start + row_offset;
      const int64_t source =
          ((static_cast<int64_t>(batch) * seq_len + row) * num_heads + head) *
          value_dim + feature;
      grad = read_float(grad_output, source);
    }
    grad_tile[index] = wmma::__float_to_tf32(grad);
  }
  __syncthreads();

  const int warp = threadIdx.x / kWarpSize;
  constexpr int kRouteTiles = kBlockRoutes / kTensorTile;
  constexpr int kValueColumnTiles = kBlockValueDim / kTensorTile;
  constexpr int kValueTiles = kRouteTiles * kValueColumnTiles;
  for (int tile = warp; tile < kValueTiles; tile += kTensorWarps) {
    const int output_row = (tile / kValueColumnTiles) * kTensorTile;
    const int output_column = (tile % kValueColumnTiles) * kTensorTile;
    wmma::fragment<
        wmma::accumulator,
        kTensorTile,
        kTensorTile,
        kTensorReduction,
        float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int reduction = 0; reduction < kBlockRows;
         reduction += kTensorReduction) {
      wmma::fragment<
          wmma::matrix_a,
          kTensorTile,
          kTensorTile,
          kTensorReduction,
          wmma::precision::tf32,
          wmma::col_major>
          probability;
      wmma::fragment<
          wmma::matrix_b,
          kTensorTile,
          kTensorTile,
          kTensorReduction,
          wmma::precision::tf32,
          wmma::row_major>
          gradient;
      wmma::load_matrix_sync(
          probability,
          shared.primary + reduction * kBlockRoutes + output_row,
          kBlockRoutes);
      wmma::load_matrix_sync(
          gradient,
          grad_tile + reduction * kBlockValueDim + output_column,
          kBlockValueDim);
      wmma::mma_sync(accumulator, probability, gradient, accumulator);
    }
    wmma::store_matrix_sync(
        output_tile + output_row * kBlockValueDim + output_column,
        accumulator,
        kBlockValueDim,
        wmma::mem_row_major);
  }
  __syncthreads();

  const int items = route_count * value_dim;
  for (int index = threadIdx.x; index < items; index += blockDim.x) {
    const int route_offset = index / value_dim;
    const int feature = index - route_offset * value_dim;
    const float contribution =
        output_tile[route_offset * kBlockValueDim + feature];
    if (contribution != 0.0f) {
      const int route = route_start + route_offset;
      const int64_t target =
          ((static_cast<int64_t>(batch) * seq_len + route) * num_value_heads +
           value_head) * value_dim + feature;
      atomicAdd(&grad_value[target], contribution);
    }
  }
  __syncthreads();
#endif
}


__device__ __forceinline__ void fill_block_gate_scores(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    BlockShared shared,
    int row_start,
    int row_count,
    int route_start,
    int route_count,
    uint32_t mask,
    float mismatch_unit) {
  const int diagonal_count = row_count + route_count - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int difference = diagonal - (route_count - 1);
    const int first_row_offset = max(difference, 0);
    const int first_route_offset = max(-difference, 0);
    const int endpoint_count = min(
        row_count - first_row_offset,
        route_count - first_route_offset);
    const int first_row = row_start + first_row_offset;
    const int first_route = route_start + first_route_offset;
    if (endpoint_count <= 0 || first_route > first_row) {
      continue;
    }
    const int history = min(
        kBlockWindow - 1,
        min(first_row, first_route - 1));
    int row = first_row - history;
    int route = first_route - history;
    const int count = history + endpoint_count;
    float score;
    int rolling_mismatch;
    int rolling_count;
    initialize_block_state(
        query,
        key,
        row,
        route,
        mask,
        mismatch_unit,
        score,
        rolling_mismatch,
        rolling_count);
    for (int index = 0; index < count; ++index) {
      if (index > 0) {
        ++row;
        ++route;
        advance_block_state(
            query,
            key,
            row,
            route,
            mask,
            mismatch_unit,
            score,
            rolling_mismatch,
            rolling_count);
      }
      const int query_local = row - (row_start - kBlockWindow + 1);
      const int key_local = route - 1 - (route_start - kBlockWindow);
      shared.secondary[
          query_local * kBlockSpanStride + key_local] = score;
    }
  }
  __syncthreads();
}


__device__ __forceinline__ void reverse_block_gates(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    BlockShared shared,
    int row_start,
    int row_count,
    int route_start,
    int route_count,
    uint32_t mask,
    float mismatch_unit) {
  const int diagonal_count = row_count + route_count - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int difference = diagonal - (route_count - 1);
    const int first_row_offset = max(difference, 0);
    const int first_route_offset = max(-difference, 0);
    const int endpoint_count = min(
        row_count - first_row_offset,
        route_count - first_route_offset);
    const int first_row = row_start + first_row_offset;
    const int first_route = route_start + first_route_offset;
    if (endpoint_count <= 0 || first_route > first_row) {
      continue;
    }
    const int history = min(
        kBlockWindow - 1,
        min(first_row, first_route - 1));
    const int first_extended_row = first_row - history;
    const int first_extended_route = first_route - history;
    const int count = history + endpoint_count;
    int row = first_extended_row + count - 1;
    int route = first_extended_route + count - 1;
    int rolling_count = min(kBlockWindow + 1, min(row + 1, route));
    int rolling_mismatch = 0;
#pragma unroll
    for (int suffix = 0; suffix < kBlockWindow + 1; ++suffix) {
      if (suffix < rolling_count) {
        rolling_mismatch += block_mismatch(
            query,
            key,
            row - suffix,
            route - 1 - suffix,
            mask);
      }
    }
    float next_gate = 0.0f;
    float next_score_vjp = 0.0f;
    float future_correction_vjp = 0.0f;
    float* correction_ring = shared.ring + diagonal * kBlockRingStride;

    for (int index = count - 1; index >= 0; --index) {
      const int query_local = row - (row_start - kBlockWindow + 1);
      const int key_local = route - 1 - (route_start - kBlockWindow);
      const int gate_index = query_local * kBlockSpanStride + key_local;
      const float score = shared.secondary[gate_index];
      const float correction = rolling_count == kBlockWindow + 1
          ? block_gate(rolling_mismatch, mismatch_unit)
          : 0.0f;
      float route_vjp = 0.0f;
      if (index >= history) {
        const int endpoint = index - history;
        route_vjp = shared.primary[
            (first_row_offset + endpoint) * kBlockRoutes +
            first_route_offset + endpoint];
      }
      const float score_vjp = route_vjp + next_gate * next_score_vjp;
      const float correction_vjp = score_vjp * correction;
      const int ring_slot = index % kBlockRingStride;
      const float outgoing = index + kBlockRingStride < count
          ? correction_ring[ring_slot]
          : 0.0f;
      future_correction_vjp += correction_vjp - outgoing;
      shared.secondary[gate_index] =
          score_vjp * (score + correction) - future_correction_vjp;
      correction_ring[ring_slot] = correction_vjp;

      const int current_mismatch = block_mismatch(
          query,
          key,
          row,
          route - 1,
          mask);
      next_gate = block_gate(current_mismatch, mismatch_unit);
      next_score_vjp = score_vjp;
      if (index > 0) {
        const int next_count = min(kBlockWindow + 1, min(row, route - 1));
        rolling_mismatch -= current_mismatch;
        if (next_count == kBlockWindow + 1) {
          rolling_mismatch += block_mismatch(
              query,
              key,
              row - kBlockWindow - 1,
              route - kBlockWindow - 2,
              mask);
        }
        rolling_count = next_count;
        --row;
        --route;
      }
    }
  }
  __syncthreads();
}


template <bool Transpose>
__device__ __forceinline__ void contract_block_symbol_vjp_tf32(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    BlockShared shared,
    float* __restrict__ target,
    int batch,
    int head,
    int row_start,
    int route_start,
    int seq_len,
    int num_heads,
    int symbol_dim,
    float mismatch_scale) {
#if __CUDA_ARCH__ >= 800
  const int padded_dim = symbol_dim <= 16 ? 16 : 32;
  float* sign_matrix = shared.primary;
  float* output_matrix = shared.ring;
  float* high_scratch = shared.primary + kBlockSpanStride * 32;
  float* low_scratch = shared.ring + kBlockSpanStride * 32;
  const int position_base = Transpose
      ? route_start - kBlockWindow
      : row_start - kBlockWindow + 1;
  const int sign_base = Transpose
      ? row_start - kBlockWindow + 1
      : route_start - kBlockWindow;
  const int sign_items = kBlockSpanStride * padded_dim;
  for (int index = threadIdx.x; index < sign_items; index += blockDim.x) {
    const int position_offset = index / padded_dim;
    const int bit = index - position_offset * padded_dim;
    const int position = sign_base + position_offset;
    float sign = 0.0f;
    if (position_offset < kBlockSpan && position >= 0 &&
        position < seq_len && bit < symbol_dim) {
      const uint32_t word = static_cast<uint32_t>(
          Transpose ? query[position] : key[position]);
      sign = static_cast<float>(sign_from_bit(word, bit));
    }
    sign_matrix[index] = wmma::__float_to_tf32(sign);
  }
  __syncthreads();

  const int column_tiles = padded_dim / kTensorTile;
  const int row_tiles = kBlockSpanStride / kTensorTile;
  const int output_tiles = row_tiles * column_tiles;
  const int warp = threadIdx.x / kWarpSize;
  for (int tile = warp; tile < output_tiles; tile += kTensorWarps) {
    const int output_row = (tile / column_tiles) * kTensorTile;
    const int output_column = (tile % column_tiles) * kTensorTile;
    wmma::fragment<
        wmma::accumulator,
        kTensorTile,
        kTensorTile,
        kTensorReduction,
        float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int reduction = 0; reduction < kBlockSpanStride;
         reduction += kTensorReduction) {
      const int lane = threadIdx.x & (kWarpSize - 1);
      float* warp_high =
          high_scratch + warp * kTensorTile * kTensorReduction;
      float* warp_low =
          low_scratch + warp * kTensorTile * kTensorReduction;
      for (int index = lane; index < kTensorTile * kTensorReduction;
           index += kWarpSize) {
        const int local_row = index / kTensorReduction;
        const int local_reduction = index % kTensorReduction;
        const int gate_row = Transpose
            ? reduction + local_reduction
            : output_row + local_row;
        const int gate_column = Transpose
            ? output_row + local_row
            : reduction + local_reduction;
        const float gate = shared.secondary[
            gate_row * kBlockSpanStride + gate_column];
        const float high = wmma::__float_to_tf32(gate);
        warp_high[index] = high;
        warp_low[index] = wmma::__float_to_tf32(gate - high);
      }
      __syncwarp();
      wmma::fragment<
          wmma::matrix_a,
          kTensorTile,
          kTensorTile,
          kTensorReduction,
          wmma::precision::tf32,
          wmma::row_major>
          gate_high;
      wmma::fragment<
          wmma::matrix_a,
          kTensorTile,
          kTensorTile,
          kTensorReduction,
          wmma::precision::tf32,
          wmma::row_major>
          gate_low;
      wmma::fragment<
          wmma::matrix_b,
          kTensorTile,
          kTensorTile,
          kTensorReduction,
          wmma::precision::tf32,
          wmma::row_major>
          signs;
      wmma::load_matrix_sync(
          gate_high,
          warp_high,
          kTensorReduction);
      wmma::load_matrix_sync(
          gate_low,
          warp_low,
          kTensorReduction);
      wmma::load_matrix_sync(
          signs,
          sign_matrix + reduction * padded_dim + output_column,
          padded_dim);
      wmma::mma_sync(accumulator, gate_high, signs, accumulator);
      wmma::mma_sync(accumulator, gate_low, signs, accumulator);
      __syncwarp();
    }
    wmma::store_matrix_sync(
        output_matrix + output_row * padded_dim + output_column,
        accumulator,
        padded_dim,
        wmma::mem_row_major);
  }
  __syncthreads();

  const float derivative_scale =
      mismatch_scale / (2.0f * static_cast<float>(symbol_dim));
  const int output_items = kBlockSpan * symbol_dim;
  for (int item = threadIdx.x; item < output_items; item += blockDim.x) {
    const int position_offset = item / symbol_dim;
    const int bit = item - position_offset * symbol_dim;
    const int position = position_base + position_offset;
    if (position >= 0 && position < seq_len) {
      const float contribution =
          output_matrix[position_offset * padded_dim + bit] *
          derivative_scale;
      if (contribution != 0.0f) {
        const int64_t target_index = Transpose
            ? ((static_cast<int64_t>(batch) * num_heads + head) * symbol_dim +
               bit) * seq_len + position
            : ((static_cast<int64_t>(batch) * seq_len + position) *
                   num_heads +
               head) * symbol_dim + bit;
        atomicAdd(&target[target_index], contribution);
      }
    }
  }
  __syncthreads();
#endif
}


template <typename scalar_t>
__global__ __launch_bounds__(kBlockThreads, kBlockMinBlocks)
void block_diagonal_vjp_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query_symbols,
    const int32_t* __restrict__ packed_key_symbols,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_scale,
    const int64_t* __restrict__ dropout_seed,
    int gradient_mask) {
  extern __shared__ float storage[];
  const BlockShared shared = bind_block_shared(storage);
  const int query_tiles = (seq_len + kBlockRows - 1) / kBlockRows;
  const int query_tile = blockIdx.x % query_tiles;
  const int head = (blockIdx.x / query_tiles) % num_heads;
  const int batch = blockIdx.x / (query_tiles * num_heads);
  const int row_start = query_tile * kBlockRows;
  const int row_count = min(kBlockRows, seq_len - row_start);
  const int row_end = row_start + row_count - 1;
  const int value_head = head / (num_heads / num_value_heads);
  const int32_t* query = packed_query_symbols +
      (static_cast<int64_t>(batch) * num_heads + head) * seq_len;
  const int32_t* key = packed_key_symbols +
      (static_cast<int64_t>(batch) * num_heads + head) * seq_len;
  const uint32_t mask = block_symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);
  const bool needs_qk = (gradient_mask & (kGradQuery | kGradKey)) != 0;
  const bool needs_value = (gradient_mask & kGradValue) != 0;
  const int stats_row = threadIdx.x;
  const bool owns_stats_row = stats_row < row_count;

  SoftmaxStats row_stats = owns_stats_row
      ? SoftmaxStats{kNullScore * scale, 1.0f, 0.0f}
      : SoftmaxStats{-FLT_MAX, 0.0f, 0.0f};
  for (int route_start = 1; route_start <= row_end;
       route_start += kBlockRoutes) {
    const int route_count = min(kBlockRoutes, seq_len - route_start);
    generate_block_scores(
        query,
        key,
        shared,
        row_start,
        row_count,
        route_start,
        route_count,
        mask,
        mismatch_unit);
    generate_block_utilities_tf32(
        value,
        grad_output,
        shared,
        batch,
        head,
        value_head,
        row_start,
        row_count,
        route_start,
        route_count,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim,
        needs_qk);
    if (owns_stats_row) {
      const int row = row_start + stats_row;
      for (int route_offset = 0; route_offset < route_count; ++route_offset) {
        const int route = route_start + route_offset;
        if (route <= row) {
          const float raw_score =
              shared.primary[stats_row * kBlockRoutes + route_offset];
          const ScoreTransform transformed = transform_score(raw_score);
          const float dropout_scale = attention_dropout_scale(
              dropout_seed,
              dropout_p,
              inverse_keep_probability,
              batch,
              head,
              row,
              route);
          row_stats = append_item(
              row_stats,
              transformed.route_score * scale -
                  logf(static_cast<float>(row)),
              dropout_scale * shared.secondary[
                  stats_row * kBlockRoutes + route_offset]);
        }
      }
    }
    __syncthreads();
  }
  if (owns_stats_row) {
    shared.row_maximum[stats_row] = row_stats.maximum;
    shared.row_normalizer[stats_row] = row_stats.normalizer;
    shared.row_expected_utility[stats_row] =
        row_stats.utility_numerator / row_stats.normalizer;
  }
  __syncthreads();

  for (int route_start = 1; route_start <= row_end;
       route_start += kBlockRoutes) {
    const int route_count = min(kBlockRoutes, seq_len - route_start);
    generate_block_scores(
        query,
        key,
        shared,
        row_start,
        row_count,
        route_start,
        route_count,
        mask,
        mismatch_unit);
    generate_block_utilities_tf32(
        value,
        grad_output,
        shared,
        batch,
        head,
        value_head,
        row_start,
        row_count,
        route_start,
        route_count,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim,
        needs_qk);

    const int route_items = row_count * route_count;
    for (int index = threadIdx.x; index < route_items; index += blockDim.x) {
      const int row_offset = index / route_count;
      const int route_offset = index - row_offset * route_count;
      const int row = row_start + row_offset;
      const int route = route_start + route_offset;
      float dropped_probability = 0.0f;
      float raw_score_vjp = 0.0f;
      if (route <= row) {
        const float raw_score =
            shared.primary[row_offset * kBlockRoutes + route_offset];
        const ScoreTransform transformed = transform_score(raw_score);
        const float probability = __expf(
            transformed.route_score * scale - logf(static_cast<float>(row)) -
            shared.row_maximum[row_offset]) /
            shared.row_normalizer[row_offset];
        const float dropout_scale = attention_dropout_scale(
            dropout_seed,
            dropout_p,
            inverse_keep_probability,
            batch,
            head,
            row,
            route);
        dropped_probability = probability * dropout_scale;
        if (needs_qk) {
          raw_score_vjp =
              scale * probability *
              (dropout_scale *
                   shared.secondary[
                       row_offset * kBlockRoutes + route_offset] -
               shared.row_expected_utility[row_offset]) *
              transformed.raw_vjp_multiplier;
        }
      }
      shared.primary[row_offset * kBlockRoutes + route_offset] =
          dropped_probability;
      shared.secondary[row_offset * kBlockRoutes + route_offset] =
          raw_score_vjp;
    }
    __syncthreads();

    if (needs_value) {
      accumulate_block_value_vjp_tf32(
          grad_output,
          shared,
          grad_value,
          batch,
          head,
          value_head,
          row_start,
          row_count,
          route_start,
          route_count,
          seq_len,
          num_heads,
          num_value_heads,
          value_dim);
    }
    __syncthreads();

    if (needs_qk) {
      for (int index = threadIdx.x; index < kBlockMatrixItems;
           index += blockDim.x) {
        shared.primary[index] = shared.secondary[index];
      }
      for (int index = threadIdx.x; index < kBlockGateItems;
           index += blockDim.x) {
        shared.secondary[index] = 0.0f;
      }
      for (int index = threadIdx.x; index < kBlockRingItems;
           index += blockDim.x) {
        shared.ring[index] = 0.0f;
      }
      __syncthreads();
      fill_block_gate_scores(
          query,
          key,
          shared,
          row_start,
          row_count,
          route_start,
          route_count,
          mask,
          mismatch_unit);
      reverse_block_gates(
          query,
          key,
          shared,
          row_start,
          row_count,
          route_start,
          route_count,
          mask,
          mismatch_unit);
      if ((gradient_mask & kGradQuery) != 0) {
        contract_block_symbol_vjp_tf32<false>(
            query,
            key,
            shared,
            grad_query,
            batch,
            head,
            row_start,
            route_start,
            seq_len,
            num_heads,
            symbol_dim,
            mismatch_scale);
      }
      if ((gradient_mask & kGradKey) != 0) {
        contract_block_symbol_vjp_tf32<true>(
            query,
            key,
            shared,
            grad_key,
            batch,
            head,
            row_start,
            route_start,
            seq_len,
            num_heads,
            symbol_dim,
            mismatch_scale);
      }
    }
  }
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_block_diagonal_vjp_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t max_suffix_length,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  if (max_suffix_length != kBlockWindow ||
      value.size(3) != kBlockValueDim ||
      at::cuda::getCurrentDeviceProperties()->major < 8) {
    return rosa_soft_streaming_vjp_cuda(
        query,
        key,
        value,
        grad_output,
        packed_query_symbols,
        packed_key_symbols,
        dropout_seed,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
        gradient_mask,
        32);
  }

  const int batch_size = query.size(0);
  const int seq_len = query.size(1);
  const int num_heads = query.size(2);
  const int symbol_dim = query.size(3);
  const int num_value_heads = value.size(2);
  const int value_dim = value.size(3);
  const auto float_options = query.options().dtype(torch::kFloat32);
  torch::Tensor grad_query = (gradient_mask & kGradQuery) != 0
      ? torch::zeros(query.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_key_accumulator = (gradient_mask & kGradKey) != 0
      ? torch::zeros(
            {batch_size, num_heads, symbol_dim, seq_len}, float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_key = (gradient_mask & kGradKey) != 0
      ? torch::zeros(query.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_value = (gradient_mask & kGradValue) != 0
      ? torch::zeros(value.sizes(), float_options)
      : torch::empty({0}, float_options);
  const int query_tiles = (seq_len + kBlockRows - 1) / kBlockRows;
  const int64_t blocks64 =
      static_cast<int64_t>(batch_size) * num_heads * query_tiles;
  TORCH_CHECK(blocks64 <= std::numeric_limits<int>::max());
  const size_t shared_bytes = kBlockSharedItems * sizeof(float);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_block_diagonal_vjp_cuda",
      [&] {
          C10_CUDA_CHECK(cudaFuncSetAttribute(
              block_diagonal_vjp_kernel<scalar_t>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(shared_bytes)));
          block_diagonal_vjp_kernel<scalar_t><<<
              static_cast<int>(blocks64),
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
              batch_size,
              seq_len,
              num_heads,
              symbol_dim,
              num_value_heads,
              value_dim,
              scale,
              dropout_p,
              inverse_keep_probability,
              mismatch_scale,
              dropout_seed.data_ptr<int64_t>(),
              gradient_mask);

        const int64_t query_elements =
            (gradient_mask & kGradQuery) != 0 ? query.numel() : 0;
        const int64_t key_elements =
            (gradient_mask & kGradKey) != 0 ? key.numel() : 0;
        const int64_t value_elements =
            (gradient_mask & kGradValue) != 0 ? value.numel() : 0;
        const int64_t final_elements =
            query_elements + key_elements + value_elements;
        constexpr int kFinalizeThreads = 256;
        const int final_blocks = static_cast<int>(
            (final_elements + kFinalizeThreads - 1) / kFinalizeThreads);
        finalize_vjp_kernel<scalar_t><<<
            final_blocks,
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
  return std::make_tuple(grad_query, grad_key, grad_value);
}
