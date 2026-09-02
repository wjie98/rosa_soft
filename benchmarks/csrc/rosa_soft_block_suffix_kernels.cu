#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>


namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kRowThreads = kWarpSize * kWarpsPerBlock;


__device__ __forceinline__ uint32_t symbol_mask(int symbol_dim) {
  return symbol_dim == 32
      ? 0xffffffffu
      : (1u << symbol_dim) - 1u;
}


__device__ __forceinline__ int mismatch_count(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int query_position,
    int key_position,
    uint32_t mask) {
  const uint32_t query_word = static_cast<uint32_t>(query[query_position]);
  const uint32_t key_word = static_cast<uint32_t>(key[key_position]);
  return __popc((query_word ^ key_word) & mask);
}


__device__ __forceinline__ float gate_from_mismatch(
    int mismatch,
    float mismatch_unit) {
  return __expf(-mismatch_unit * static_cast<float>(mismatch));
}


__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}


__global__ void thread_suffix_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale) {
  const int row = blockIdx.x % seq_len;
  const int series = blockIdx.x / seq_len;
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* row_output =
      output + (static_cast<int64_t>(series) * seq_len + row) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);

  for (int route = threadIdx.x; route < seq_len; route += blockDim.x) {
    float score = 0.0f;
    if (route >= 1 && route <= row) {
      const int suffix_steps = min(max_suffix_length, min(row + 1, route));
      float product = 1.0f;
      for (int suffix = 0; suffix < suffix_steps; ++suffix) {
        const int mismatch = mismatch_count(
            query,
            key,
            row - suffix,
            route - 1 - suffix,
            mask);
        product *= gate_from_mismatch(mismatch, mismatch_unit);
        score += product;
      }
    }
    row_output[route] = score;
  }
}


__global__ void warp_suffix_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale) {
  const int row = blockIdx.x % seq_len;
  const int series = blockIdx.x / seq_len;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* row_output =
      output + (static_cast<int64_t>(series) * seq_len + row) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);

  for (int route = threadIdx.x; route < seq_len; route += blockDim.x) {
    if (route == 0 || route > row) {
      row_output[route] = 0.0f;
    }
  }

  for (int route = 1 + warp; route <= row; route += kWarpsPerBlock) {
    const int suffix_steps = min(max_suffix_length, min(row + 1, route));
    float carry = 1.0f;
    float score = 0.0f;
    for (int suffix_start = 0; suffix_start < suffix_steps;
         suffix_start += kWarpSize) {
      const int count = min(kWarpSize, suffix_steps - suffix_start);
      const int suffix = suffix_start + lane;
      float prefix = 1.0f;
      if (lane < count) {
        const int mismatch = mismatch_count(
            query,
            key,
            row - suffix,
            route - 1 - suffix,
            mask);
        prefix = gate_from_mismatch(mismatch, mismatch_unit);
      }
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float previous =
            __shfl_up_sync(0xffffffffu, prefix, offset);
        if (lane >= offset) {
          prefix *= previous;
        }
      }
      score += warp_sum(lane < count ? carry * prefix : 0.0f);
      carry *= __shfl_sync(0xffffffffu, prefix, count - 1);
    }
    if (lane == 0) {
      row_output[route] = score;
    }
  }
}


__device__ __forceinline__ void initialize_diagonal_state(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int route,
    int max_suffix_length,
    uint32_t mask,
    float mismatch_unit,
    float& score,
    int& rolling_mismatch,
    int& rolling_count) {
  const int available = min(row + 1, route);
  rolling_count = min(max_suffix_length + 1, available);
  rolling_mismatch = 0;
  float product = 1.0f;
  score = 0.0f;
  for (int suffix = 0; suffix < rolling_count; ++suffix) {
    const int mismatch = mismatch_count(
        query,
        key,
        row - suffix,
        route - 1 - suffix,
        mask);
    rolling_mismatch += mismatch;
    if (suffix < max_suffix_length) {
      product *= gate_from_mismatch(mismatch, mismatch_unit);
      score += product;
    }
  }
}


__device__ __forceinline__ void advance_diagonal_state(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int route,
    int max_suffix_length,
    uint32_t mask,
    float mismatch_unit,
    float& score,
    int& rolling_mismatch,
    int& rolling_count) {
  const int mismatch = mismatch_count(query, key, row, route - 1, mask);
  if (rolling_count == max_suffix_length + 1) {
    rolling_mismatch -= mismatch_count(
        query,
        key,
        row - max_suffix_length - 1,
        route - max_suffix_length - 2,
        mask);
  } else {
    ++rolling_count;
  }
  rolling_mismatch += mismatch;
  const float gate = gate_from_mismatch(mismatch, mismatch_unit);
  const float correction = rolling_count == max_suffix_length + 1
      ? gate_from_mismatch(rolling_mismatch, mismatch_unit)
      : 0.0f;
  score = fmaf(gate, 1.0f + score, -correction);
}


template <int Tile>
__global__ void diagonal_block_suffix_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale) {
  const int query_tiles = (seq_len + Tile - 1) / Tile;
  const int query_tile = blockIdx.x % query_tiles;
  const int series = blockIdx.x / query_tiles;
  const int row_start = query_tile * Tile;
  const int row_count = min(Tile, seq_len - row_start);
  const int row_end = row_start + row_count - 1;
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* matrix = output + static_cast<int64_t>(series) * seq_len * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);

  const int row_items = row_count * seq_len;
  for (int index = threadIdx.x; index < row_items; index += blockDim.x) {
    const int row_offset = index / seq_len;
    const int route = index - row_offset * seq_len;
    const int row = row_start + row_offset;
    if (route == 0 || route > row) {
      matrix[static_cast<int64_t>(row) * seq_len + route] = 0.0f;
    }
  }

  for (int route_start = 1; route_start <= row_end; route_start += Tile) {
    const int route_count = min(Tile, seq_len - route_start);
    const int diagonal_count = row_count + route_count - 1;
    for (int diagonal = threadIdx.x; diagonal < diagonal_count;
         diagonal += blockDim.x) {
      const int difference = diagonal - (route_count - 1);
      int row_offset = max(difference, 0);
      int route_offset = max(-difference, 0);
      const int count = min(
          row_count - row_offset,
          route_count - route_offset);
      int row = row_start + row_offset;
      int route = route_start + route_offset;
      if (count <= 0 || route > row) {
        continue;
      }

      float score;
      int rolling_mismatch;
      int rolling_count;
      initialize_diagonal_state(
          query,
          key,
          row,
          route,
          max_suffix_length,
          mask,
          mismatch_unit,
          score,
          rolling_mismatch,
          rolling_count);
      matrix[static_cast<int64_t>(row) * seq_len + route] = score;

      for (int step = 1; step < count; ++step) {
        ++row;
        ++route;
        advance_diagonal_state(
            query,
            key,
            row,
            route,
            max_suffix_length,
            mask,
            mismatch_unit,
            score,
            rolling_mismatch,
            rolling_count);
        matrix[static_cast<int64_t>(row) * seq_len + route] = score;
      }
    }
  }
}

constexpr int kTailPhysicalTile = 64;


__global__ void physical_block_tail_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int route_start,
    int active_queries) {
  extern __shared__ float score_tile[];
  const int series = blockIdx.x;
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* series_output =
      output + static_cast<int64_t>(series) * active_queries * active_queries;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);
  const int physical_row_start =
      route_start - (kTailPhysicalTile - active_queries);

  for (int index = threadIdx.x;
       index < kTailPhysicalTile * kTailPhysicalTile;
       index += blockDim.x) {
    score_tile[index] = 0.0f;
  }
  __syncthreads();

  constexpr int kDiagonals = 2 * kTailPhysicalTile - 1;
  for (int diagonal = threadIdx.x; diagonal < kDiagonals;
       diagonal += blockDim.x) {
    const int difference = diagonal - (kTailPhysicalTile - 1);
    const int first_row_offset = max(difference, 0);
    const int first_route_offset = max(-difference, 0);
    const int count = min(
        kTailPhysicalTile - first_row_offset,
        kTailPhysicalTile - first_route_offset);
    int row = physical_row_start + first_row_offset;
    int route = route_start + first_route_offset;
    if (count <= 0 || route > row) {
      continue;
    }
    float score;
    int rolling_mismatch;
    int rolling_count;
    initialize_diagonal_state(
        query,
        key,
        row,
        route,
        max_suffix_length,
        mask,
        mismatch_unit,
        score,
        rolling_mismatch,
        rolling_count);
    score_tile[
        first_row_offset * kTailPhysicalTile + first_route_offset] = score;
    for (int step = 1; step < count; ++step) {
      ++row;
      ++route;
      advance_diagonal_state(
          query,
          key,
          row,
          route,
          max_suffix_length,
          mask,
          mismatch_unit,
          score,
          rolling_mismatch,
          rolling_count);
      score_tile[
          (first_row_offset + step) * kTailPhysicalTile +
          first_route_offset + step] = score;
    }
  }
  __syncthreads();

  const int active_items = active_queries * active_queries;
  const int active_row_base = kTailPhysicalTile - active_queries;
  for (int index = threadIdx.x; index < active_items; index += blockDim.x) {
    const int row_offset = index / active_queries;
    const int route_offset = index - row_offset * active_queries;
    series_output[index] = route_offset <= row_offset
        ? score_tile[
              (active_row_base + row_offset) * kTailPhysicalTile +
              route_offset]
        : 0.0f;
  }
}


__global__ void thread_tail_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int route_start,
    int active_queries) {
  const int series = blockIdx.x;
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* series_output =
      output + static_cast<int64_t>(series) * active_queries * active_queries;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);
  const int items = active_queries * active_queries;
  for (int index = threadIdx.x; index < items; index += blockDim.x) {
    const int row_offset = index / active_queries;
    const int route_offset = index - row_offset * active_queries;
    float score = 0.0f;
    if (route_offset <= row_offset) {
      const int row = route_start + row_offset;
      const int route = route_start + route_offset;
      const int suffix_steps = min(max_suffix_length, min(row + 1, route));
      float product = 1.0f;
      for (int suffix = 0; suffix < suffix_steps; ++suffix) {
        const int mismatch = mismatch_count(
            query,
            key,
            row - suffix,
            route - 1 - suffix,
            mask);
        product *= gate_from_mismatch(mismatch, mismatch_unit);
        score += product;
      }
    }
    series_output[index] = score;
  }
}


__global__ void warp_suffix_tail_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int route_start,
    int active_queries) {
  const int series = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* series_output =
      output + static_cast<int64_t>(series) * active_queries * active_queries;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);
  const int items = active_queries * active_queries;

  for (int index = threadIdx.x; index < items; index += blockDim.x) {
    const int row_offset = index / active_queries;
    const int route_offset = index - row_offset * active_queries;
    if (route_offset > row_offset) {
      series_output[index] = 0.0f;
    }
  }
  for (int index = warp; index < items; index += kWarpsPerBlock) {
    const int row_offset = index / active_queries;
    const int route_offset = index - row_offset * active_queries;
    if (route_offset > row_offset) {
      continue;
    }
    const int row = route_start + row_offset;
    const int route = route_start + route_offset;
    const int suffix_steps = min(max_suffix_length, min(row + 1, route));
    float carry = 1.0f;
    float score = 0.0f;
    for (int suffix_start = 0; suffix_start < suffix_steps;
         suffix_start += kWarpSize) {
      const int count = min(kWarpSize, suffix_steps - suffix_start);
      const int suffix = suffix_start + lane;
      float prefix = 1.0f;
      if (lane < count) {
        const int mismatch = mismatch_count(
            query,
            key,
            row - suffix,
            route - 1 - suffix,
            mask);
        prefix = gate_from_mismatch(mismatch, mismatch_unit);
      }
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float previous =
            __shfl_up_sync(0xffffffffu, prefix, offset);
        if (lane >= offset) {
          prefix *= previous;
        }
      }
      score += warp_sum(lane < count ? carry * prefix : 0.0f);
      carry *= __shfl_sync(0xffffffffu, prefix, count - 1);
    }
    if (lane == 0) {
      series_output[index] = score;
    }
  }
}


__global__ void diagonal_thread_tail_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int route_start,
    int active_queries) {
  const int series = blockIdx.x;
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* series_output =
      output + static_cast<int64_t>(series) * active_queries * active_queries;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);
  const int items = active_queries * active_queries;
  for (int index = threadIdx.x; index < items; index += blockDim.x) {
    series_output[index] = 0.0f;
  }
  __syncthreads();

  for (int difference = threadIdx.x; difference < active_queries;
       difference += blockDim.x) {
    int row = route_start + difference;
    int route = route_start;
    const int count = active_queries - difference;
    float score;
    int rolling_mismatch;
    int rolling_count;
    initialize_diagonal_state(
        query,
        key,
        row,
        route,
        max_suffix_length,
        mask,
        mismatch_unit,
        score,
        rolling_mismatch,
        rolling_count);
    series_output[difference * active_queries] = score;
    for (int step = 1; step < count; ++step) {
      ++row;
      ++route;
      advance_diagonal_state(
          query,
          key,
          row,
          route,
          max_suffix_length,
          mask,
          mismatch_unit,
          score,
          rolling_mismatch,
          rolling_count);
      series_output[
          (difference + step) * active_queries + step] = score;
    }
  }
}


__global__ void diagonal_warp_tail_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    float mismatch_scale,
    int route_start,
    int active_queries) {
  const int series = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* series_output =
      output + static_cast<int64_t>(series) * active_queries * active_queries;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);
  const int items = active_queries * active_queries;
  for (int index = threadIdx.x; index < items; index += blockDim.x) {
    series_output[index] = 0.0f;
  }
  __syncthreads();

  for (int difference = warp; difference < active_queries;
       difference += kWarpsPerBlock) {
    const int first_row = route_start + difference;
    const int first_route = route_start;
    const int count = active_queries - difference;

    const int history_mismatch = mismatch_count(
        query,
        key,
        first_row - kWarpSize + lane,
        first_route - kWarpSize - 1 + lane,
        mask);
    int history_prefix = history_mismatch;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int previous =
          __shfl_up_sync(0xffffffffu, history_prefix, offset);
      if (lane >= offset) {
        history_prefix += previous;
      }
    }
    const int history_total =
        __shfl_sync(0xffffffffu, history_prefix, kWarpSize - 1);
    const int shifted_history_prefix =
        __shfl_up_sync(0xffffffffu, history_prefix, 1);
    const int history_before = lane == 0 ? 0 : shifted_history_prefix;

    const int reverse_history_mismatch = mismatch_count(
        query,
        key,
        first_row - 1 - lane,
        first_route - 2 - lane,
        mask);
    float history_product = gate_from_mismatch(
        reverse_history_mismatch,
        mismatch_unit);
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const float previous =
          __shfl_up_sync(0xffffffffu, history_product, offset);
      if (lane >= offset) {
        history_product *= previous;
      }
    }
    float history_score = warp_sum(history_product);
    history_score = __shfl_sync(0xffffffffu, history_score, 0);

    int current_mismatch = 0;
    if (lane < count) {
      current_mismatch = mismatch_count(
          query,
          key,
          first_row + lane,
          first_route - 1 + lane,
          mask);
    }
    int current_prefix = current_mismatch;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int previous =
          __shfl_up_sync(0xffffffffu, current_prefix, offset);
      if (lane >= offset) {
        current_prefix += previous;
      }
    }

    float affine_scale = 1.0f;
    float affine_bias = 0.0f;
    if (lane < count) {
      const float gate = gate_from_mismatch(current_mismatch, mismatch_unit);
      const int correction_mismatch =
          history_total - history_before + current_prefix;
      const float correction = gate_from_mismatch(
          correction_mismatch,
          mismatch_unit);
      affine_scale = gate;
      affine_bias = gate - correction;
    }
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const float previous_scale =
          __shfl_up_sync(0xffffffffu, affine_scale, offset);
      const float previous_bias =
          __shfl_up_sync(0xffffffffu, affine_bias, offset);
      if (lane >= offset) {
        affine_bias = fmaf(affine_scale, previous_bias, affine_bias);
        affine_scale *= previous_scale;
      }
    }
    if (lane < count) {
      series_output[
          (difference + lane) * active_queries + lane] =
          fmaf(affine_scale, history_score, affine_bias);
    }
  }
}


constexpr int kHybridTile = 64;


template <int Method>
__global__ void hybrid_tile_scores_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    float mismatch_scale,
    int tile_start,
    int tail_queries) {
  const int series = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  float* tile_output =
      output + static_cast<int64_t>(series) * kHybridTile * kHybridTile;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);

  for (int index = threadIdx.x; index < kHybridTile * kHybridTile;
       index += blockDim.x) {
    tile_output[index] = 0.0f;
  }
  __syncthreads();

  constexpr bool kHasSpecialTail = Method != 0;
  const int main_route_count =
      kHasSpecialTail ? kHybridTile - tail_queries : kHybridTile;
  for (int difference = threadIdx.x; difference < kHybridTile;
       difference += blockDim.x) {
    const int count = kHybridTile - difference;
    const int main_count = min(count, main_route_count);
    if (main_count <= 0) {
      continue;
    }
    int row = tile_start + difference;
    int route = tile_start;
    float score;
    int rolling_mismatch;
    int rolling_count;
    initialize_diagonal_state(
        query,
        key,
        row,
        route,
        kWarpSize,
        mask,
        mismatch_unit,
        score,
        rolling_mismatch,
        rolling_count);
    tile_output[difference * kHybridTile] = score;
    for (int step = 1; step < main_count; ++step) {
      ++row;
      ++route;
      advance_diagonal_state(
          query,
          key,
          row,
          route,
          kWarpSize,
          mask,
          mismatch_unit,
          score,
          rolling_mismatch,
          rolling_count);
      tile_output[(difference + step) * kHybridTile + step] = score;
    }
  }

  if constexpr (Method == 0) {
    return;
  }

  const int tail_start = tile_start + kHybridTile - tail_queries;
  if constexpr (Method == 1) {
    const int tail_items = tail_queries * tail_queries;
    for (int index = threadIdx.x; index < tail_items; index += blockDim.x) {
      const int row_offset = index / tail_queries;
      const int route_offset = index - row_offset * tail_queries;
      if (route_offset > row_offset) {
        continue;
      }
      const int row = tail_start + row_offset;
      const int route = tail_start + route_offset;
      float product = 1.0f;
      float score = 0.0f;
#pragma unroll
      for (int suffix = 0; suffix < kWarpSize; ++suffix) {
        const int mismatch = mismatch_count(
            query,
            key,
            row - suffix,
            route - 1 - suffix,
            mask);
        product *= gate_from_mismatch(mismatch, mismatch_unit);
        score += product;
      }
      tile_output[
          (kHybridTile - tail_queries + row_offset) * kHybridTile +
          kHybridTile - tail_queries + route_offset] = score;
    }
  } else if constexpr (Method == 2) {
    const int tail_items = tail_queries * tail_queries;
    for (int index = warp; index < tail_items; index += kWarpsPerBlock) {
      const int row_offset = index / tail_queries;
      const int route_offset = index - row_offset * tail_queries;
      if (route_offset > row_offset) {
        continue;
      }
      const int row = tail_start + row_offset;
      const int route = tail_start + route_offset;
      const int mismatch = mismatch_count(
          query,
          key,
          row - lane,
          route - 1 - lane,
          mask);
      float prefix = gate_from_mismatch(mismatch, mismatch_unit);
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float previous =
            __shfl_up_sync(0xffffffffu, prefix, offset);
        if (lane >= offset) {
          prefix *= previous;
        }
      }
      const float score = warp_sum(prefix);
      if (lane == 0) {
        tile_output[
            (kHybridTile - tail_queries + row_offset) * kHybridTile +
            kHybridTile - tail_queries + route_offset] = score;
      }
    }
  } else {
    for (int difference = warp; difference < tail_queries;
         difference += kWarpsPerBlock) {
      const int first_row = tail_start + difference;
      const int first_route = tail_start;
      const int count = tail_queries - difference;

      const int history_mismatch = mismatch_count(
          query,
          key,
          first_row - kWarpSize + lane,
          first_route - kWarpSize - 1 + lane,
          mask);
      int history_prefix = history_mismatch;
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int previous =
            __shfl_up_sync(0xffffffffu, history_prefix, offset);
        if (lane >= offset) {
          history_prefix += previous;
        }
      }
      const int history_total =
          __shfl_sync(0xffffffffu, history_prefix, kWarpSize - 1);
      const int shifted_history_prefix =
          __shfl_up_sync(0xffffffffu, history_prefix, 1);
      const int history_before = lane == 0 ? 0 : shifted_history_prefix;

      const int reverse_history_mismatch = mismatch_count(
          query,
          key,
          first_row - 1 - lane,
          first_route - 2 - lane,
          mask);
      float history_product = gate_from_mismatch(
          reverse_history_mismatch,
          mismatch_unit);
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float previous =
            __shfl_up_sync(0xffffffffu, history_product, offset);
        if (lane >= offset) {
          history_product *= previous;
        }
      }
      float history_score = warp_sum(history_product);
      history_score = __shfl_sync(0xffffffffu, history_score, 0);

      int current_mismatch = 0;
      if (lane < count) {
        current_mismatch = mismatch_count(
            query,
            key,
            first_row + lane,
            first_route - 1 + lane,
            mask);
      }
      int current_prefix = current_mismatch;
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int previous =
            __shfl_up_sync(0xffffffffu, current_prefix, offset);
        if (lane >= offset) {
          current_prefix += previous;
        }
      }

      float affine_scale = 1.0f;
      float affine_bias = 0.0f;
      if (lane < count) {
        const float gate = gate_from_mismatch(current_mismatch, mismatch_unit);
        const int correction_mismatch =
            history_total - history_before + current_prefix;
        affine_scale = gate;
        affine_bias = gate - gate_from_mismatch(
            correction_mismatch,
            mismatch_unit);
      }
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float previous_scale =
            __shfl_up_sync(0xffffffffu, affine_scale, offset);
        const float previous_bias =
            __shfl_up_sync(0xffffffffu, affine_bias, offset);
        if (lane >= offset) {
          affine_bias = fmaf(affine_scale, previous_bias, affine_bias);
          affine_scale *= previous_scale;
        }
      }
      if (lane < count) {
        tile_output[
            (kHybridTile - tail_queries + difference + lane) * kHybridTile +
            kHybridTile - tail_queries + lane] =
            fmaf(affine_scale, history_score, affine_bias);
      }
    }
  }
}


}  // namespace


torch::Tensor rosa_block_suffix_scores_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int method) {
  const int64_t batch = packed_query.size(0);
  const int64_t heads = packed_query.size(1);
  const int64_t seq_len64 = packed_query.size(2);
  TORCH_CHECK(seq_len64 <= std::numeric_limits<int>::max());
  const int seq_len = static_cast<int>(seq_len64);
  const int64_t series64 = batch * heads;
  auto output = torch::empty(
      {batch, heads, seq_len64, seq_len64},
      packed_query.options().dtype(torch::kFloat32));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (method == 0 || method == 2) {
    const int64_t blocks64 = series64 * seq_len;
    TORCH_CHECK(blocks64 <= std::numeric_limits<int>::max());
    if (method == 0) {
      thread_suffix_scores_kernel<<<
          static_cast<int>(blocks64),
          kRowThreads,
          0,
          stream>>>(
          packed_query.data_ptr<int32_t>(),
          packed_key.data_ptr<int32_t>(),
          output.data_ptr<float>(),
          seq_len,
          symbol_dim,
          max_suffix_length,
          mismatch_scale);
    } else {
      warp_suffix_scores_kernel<<<
          static_cast<int>(blocks64),
          kRowThreads,
          0,
          stream>>>(
          packed_query.data_ptr<int32_t>(),
          packed_key.data_ptr<int32_t>(),
          output.data_ptr<float>(),
          seq_len,
          symbol_dim,
          max_suffix_length,
          mismatch_scale);
    }
  } else {
    const int tile = method == 3 ? 32 : (method == 4 ? 128 : 64);
    const int query_tiles = (seq_len + tile - 1) / tile;
    const int64_t blocks64 = series64 * query_tiles;
    TORCH_CHECK(blocks64 <= std::numeric_limits<int>::max());
    if (tile == 32) {
      diagonal_block_suffix_scores_kernel<32><<<
          static_cast<int>(blocks64), 64, 0, stream>>>(
          packed_query.data_ptr<int32_t>(),
          packed_key.data_ptr<int32_t>(),
          output.data_ptr<float>(),
          seq_len,
          symbol_dim,
          max_suffix_length,
          mismatch_scale);
    } else if (tile == 64) {
      diagonal_block_suffix_scores_kernel<64><<<
          static_cast<int>(blocks64), 128, 0, stream>>>(
          packed_query.data_ptr<int32_t>(),
          packed_key.data_ptr<int32_t>(),
          output.data_ptr<float>(),
          seq_len,
          symbol_dim,
          max_suffix_length,
          mismatch_scale);
    } else {
      diagonal_block_suffix_scores_kernel<128><<<
          static_cast<int>(blocks64), 256, 0, stream>>>(
          packed_query.data_ptr<int32_t>(),
          packed_key.data_ptr<int32_t>(),
          output.data_ptr<float>(),
          seq_len,
          symbol_dim,
          max_suffix_length,
          mismatch_scale);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}


torch::Tensor rosa_block_suffix_tail_scores_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int route_start,
    int active_queries,
    int method) {
  const int64_t batch = packed_query.size(0);
  const int64_t heads = packed_query.size(1);
  const int seq_len = static_cast<int>(packed_query.size(2));
  const int64_t series64 = batch * heads;
  TORCH_CHECK(series64 <= std::numeric_limits<int>::max());
  auto output = torch::empty(
      {batch, heads, active_queries, active_queries},
      packed_query.options().dtype(torch::kFloat32));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(series64);
  if (method == 0) {
    physical_block_tail_scores_kernel<<<
        blocks,
        kRowThreads,
        kTailPhysicalTile * kTailPhysicalTile * sizeof(float),
        stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        max_suffix_length,
        mismatch_scale,
        route_start,
        active_queries);
  } else if (method == 1) {
    thread_tail_scores_kernel<<<blocks, kRowThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        max_suffix_length,
        mismatch_scale,
        route_start,
        active_queries);
  } else if (method == 2) {
    warp_suffix_tail_scores_kernel<<<blocks, kRowThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        max_suffix_length,
        mismatch_scale,
        route_start,
        active_queries);
  } else if (method == 3) {
    diagonal_thread_tail_scores_kernel<<<blocks, kRowThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        max_suffix_length,
        mismatch_scale,
        route_start,
        active_queries);
  } else {
    diagonal_warp_tail_scores_kernel<<<blocks, kRowThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        mismatch_scale,
        route_start,
        active_queries);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}


torch::Tensor rosa_block_suffix_hybrid_scores_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int tile_start,
    int tail_queries,
    int method) {
  (void)max_suffix_length;
  const int64_t batch = packed_query.size(0);
  const int64_t heads = packed_query.size(1);
  const int seq_len = static_cast<int>(packed_query.size(2));
  const int64_t series64 = batch * heads;
  TORCH_CHECK(series64 <= std::numeric_limits<int>::max());
  auto output = torch::empty(
      {batch, heads, kHybridTile, kHybridTile},
      packed_query.options().dtype(torch::kFloat32));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(series64);
  if (method == 0) {
    hybrid_tile_scores_kernel<0><<<blocks, kRowThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        mismatch_scale,
        tile_start,
        tail_queries);
  } else if (method == 1) {
    hybrid_tile_scores_kernel<1><<<blocks, kRowThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        mismatch_scale,
        tile_start,
        tail_queries);
  } else if (method == 2) {
    hybrid_tile_scores_kernel<2><<<blocks, kRowThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        mismatch_scale,
        tile_start,
        tail_queries);
  } else {
    hybrid_tile_scores_kernel<3><<<blocks, kRowThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        mismatch_scale,
        tile_start,
        tail_queries);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
