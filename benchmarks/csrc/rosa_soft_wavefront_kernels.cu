#include "../../rosa_soft/csrc/cuda/rosa_soft_vjp_common.cuh"

#include <cooperative_groups.h>

using namespace rosa_soft::cuda;

namespace {

constexpr int kTile = 16;
constexpr int kThreads = 256;
constexpr int kSubwarp = 16;
constexpr int kSubwarps = kThreads / kSubwarp;
constexpr int kTileDiagonals = 2 * kTile - 1;


struct TileCoordinate {
  int row_tile;
  int key_tile;
};


__host__ __device__ int ceil_half(int value) {
  return (value + 1) / 2;
}


__host__ __device__ int wave_first_row_tile(int wave, int tile_count) {
  return max(ceil_half(wave), wave - (tile_count - 1));
}


__host__ __device__ int wave_last_row_tile(int wave, int tile_count) {
  return min(wave, tile_count - 1);
}


__device__ __forceinline__ TileCoordinate decode_tile(
    int local_tile,
    int wave,
    int first_row_tile) {
  const int row_tile = first_row_tile + local_tile;
  return {row_tile, wave - row_tile};
}


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
  return __popc(
      (static_cast<uint32_t>(query[query_position]) ^
       static_cast<uint32_t>(key[key_position])) &
      mask);
}


__device__ __forceinline__ void diagonal_geometry(
    int diagonal,
    int row_count,
    int key_count,
    int& first_row,
    int& first_key,
    int& count) {
  const int difference = diagonal - (key_count - 1);
  first_row = max(difference, 0);
  first_key = max(-difference, 0);
  count = min(row_count - first_row, key_count - first_key);
}


__global__ void wavefront_score_sequential_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ scores,
    float* __restrict__ score_carry,
    int32_t* __restrict__ mismatch_prefix,
    int32_t* __restrict__ prefix_history,
    int batch_heads,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_unit,
    int wave,
    int first_row_tile,
    int tiles_in_wave) {
  const int group = blockIdx.x / tiles_in_wave;
  const int local_tile = blockIdx.x - group * tiles_in_wave;
  if (group >= batch_heads) {
    return;
  }
  const TileCoordinate tile =
      decode_tile(local_tile, wave, first_row_tile);
  const int row_start = tile.row_tile * kTile;
  const int key_start = tile.key_tile * kTile;
  const int row_count = min(kTile, seq_len - row_start);
  const int key_count = min(kTile, seq_len - key_start);
  const int32_t* query = packed_query +
      static_cast<int64_t>(group) * seq_len;
  const int32_t* key = packed_key +
      static_cast<int64_t>(group) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const int history_stride = max_suffix_length + 1;

  for (int diagonal = threadIdx.x;
       diagonal < row_count + key_count - 1;
       diagonal += blockDim.x) {
    int first_row;
    int first_key;
    int count;
    diagonal_geometry(
        diagonal,
        row_count,
        key_count,
        first_row,
        first_key,
        count);
    const int query_position = row_start + first_row;
    const int key_position = key_start + first_key;
    const int delta = query_position - key_position;
    if (count <= 0 || delta <= 0) {
      continue;
    }
    const int64_t state_index =
        static_cast<int64_t>(group) * seq_len + delta;
    const int64_t history_base = state_index * history_stride;
    float score = score_carry[state_index];
    int cumulative = mismatch_prefix[state_index];
    for (int step = 0; step < count; ++step) {
      const int row = query_position + step;
      const int key_pos = key_position + step;
      const int mismatch = mismatch_count(
          query,
          key,
          row,
          key_pos,
          mask);
      int old_prefix = 0;
      if (key_pos >= max_suffix_length + 1) {
        old_prefix = prefix_history[
            history_base + key_pos % history_stride];
      }
      cumulative += mismatch;
      const float gate = __expf(
          -mismatch_unit * static_cast<float>(mismatch));
      const float correction = key_pos >= max_suffix_length
          ? __expf(
                -mismatch_unit *
                static_cast<float>(cumulative - old_prefix))
          : 0.0f;
      score = fmaf(gate, 1.0f + score, -correction);
      prefix_history[
          history_base + key_pos % history_stride] = cumulative;
      scores[
          (static_cast<int64_t>(group) * seq_len + row) * seq_len +
          key_pos + 1] = score;
    }
    score_carry[state_index] = score;
    mismatch_prefix[state_index] = cumulative;
  }
}


__device__ __forceinline__ int subwarp_inclusive_sum(int value) {
#pragma unroll
  for (int offset = 1; offset < kSubwarp; offset <<= 1) {
    const int left = __shfl_up_sync(
        0xffffffffu,
        value,
        offset,
        kSubwarp);
    if ((threadIdx.x & (kSubwarp - 1)) >= offset) {
      value += left;
    }
  }
  return value;
}


__device__ __forceinline__ void subwarp_affine_inclusive_scan(
    float& coefficient,
    float& bias) {
#pragma unroll
  for (int offset = 1; offset < kSubwarp; offset <<= 1) {
    const float left_coefficient = __shfl_up_sync(
        0xffffffffu,
        coefficient,
        offset,
        kSubwarp);
    const float left_bias = __shfl_up_sync(
        0xffffffffu,
        bias,
        offset,
        kSubwarp);
    if ((threadIdx.x & (kSubwarp - 1)) >= offset) {
      bias = fmaf(coefficient, left_bias, bias);
      coefficient *= left_coefficient;
    }
  }
}


template <typename scalar_t>
__device__ __forceinline__ void contract_tile_utilities(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    float* __restrict__ utilities,
    float* __restrict__ left_stage,
    float* __restrict__ right_stage,
    int batch_index,
    int head,
    int value_head,
    int row_start,
    int key_start,
    int row_count,
    int key_count,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  const int row_offset = threadIdx.x / kTile;
  const int key_offset = threadIdx.x & (kTile - 1);
  float utility = 0.0f;
  for (int value_start = 0; value_start < value_dim;
       value_start += kTile) {
    const int stage_row = threadIdx.x / kTile;
    const int stage_offset = threadIdx.x & (kTile - 1);
    const int value_offset = value_start + stage_offset;
    float left = 0.0f;
    float right = 0.0f;
    if (stage_row < row_count && value_offset < value_dim) {
      const int row = row_start + stage_row;
      const int64_t index =
          ((static_cast<int64_t>(batch_index) * seq_len + row) *
               num_heads +
           head) *
              value_dim +
          value_offset;
      left = read_float(grad_output, index);
    }
    if (stage_row < key_count && value_offset < value_dim) {
      const int route = key_start + stage_row + 1;
      if (route < seq_len) {
        const int64_t index =
            ((static_cast<int64_t>(batch_index) * seq_len + route) *
                 num_value_heads +
             value_head) *
                value_dim +
            value_offset;
        right = read_float(value, index) > 0.0f ? 1.0f : -1.0f;
      }
    }
    left_stage[threadIdx.x] = left;
    right_stage[threadIdx.x] = right;
    __syncthreads();
    if (row_offset < row_count && key_offset < key_count) {
#pragma unroll
      for (int offset = 0; offset < kTile; ++offset) {
        utility = fmaf(
            left_stage[row_offset * kTile + offset],
            right_stage[key_offset * kTile + offset],
            utility);
      }
    }
    __syncthreads();
  }
  utilities[threadIdx.x] =
      row_offset < row_count && key_offset < key_count ? utility : 0.0f;
}


template <typename scalar_t>
__device__ __forceinline__ float direct_candidate_utility(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    int batch_index,
    int head,
    int value_head,
    int row,
    int route,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  float utility = 0.0f;
  for (int value_offset = 0; value_offset < value_dim; ++value_offset) {
    const int64_t grad_index =
        ((static_cast<int64_t>(batch_index) * seq_len + row) * num_heads +
         head) *
            value_dim +
        value_offset;
    const int64_t value_index =
        ((static_cast<int64_t>(batch_index) * seq_len + route) *
             num_value_heads +
         value_head) *
            value_dim +
        value_offset;
    utility += read_float(grad_output, grad_index) *
        (read_float(value, value_index) > 0.0f ? 1.0f : -1.0f);
  }
  return utility;
}


__global__ void wavefront_score_affine_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ scores,
    float* __restrict__ score_carry,
    int32_t* __restrict__ mismatch_prefix,
    int32_t* __restrict__ prefix_history,
    int batch_heads,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_unit,
    int wave,
    int first_row_tile,
    int tiles_in_wave) {
  const int group = blockIdx.x / tiles_in_wave;
  const int local_tile = blockIdx.x - group * tiles_in_wave;
  if (group >= batch_heads) {
    return;
  }
  const TileCoordinate tile =
      decode_tile(local_tile, wave, first_row_tile);
  const int row_start = tile.row_tile * kTile;
  const int key_start = tile.key_tile * kTile;
  const int row_count = min(kTile, seq_len - row_start);
  const int key_count = min(kTile, seq_len - key_start);
  const int32_t* query = packed_query +
      static_cast<int64_t>(group) * seq_len;
  const int32_t* key = packed_key +
      static_cast<int64_t>(group) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const int history_stride = max_suffix_length + 1;
  const int lane = threadIdx.x & (kSubwarp - 1);
  const int subwarp = threadIdx.x / kSubwarp;
  const int diagonal_count = row_count + key_count - 1;

  for (int batch = 0; batch * kSubwarps < diagonal_count; ++batch) {
    const int diagonal = batch * kSubwarps + subwarp;
    int first_row = 0;
    int first_key = 0;
    int count = 0;
    if (diagonal < diagonal_count) {
      diagonal_geometry(
          diagonal,
          row_count,
          key_count,
          first_row,
          first_key,
          count);
    }
    const int row = row_start + first_row + lane;
    const int key_pos = key_start + first_key + lane;
    const int delta =
        diagonal < diagonal_count
        ? row_start + first_row - (key_start + first_key)
        : 0;
    const bool active = diagonal < diagonal_count && lane < count && delta > 0;
    const int64_t state_index =
        static_cast<int64_t>(group) * seq_len + max(delta, 0);
    const int64_t history_base = state_index * history_stride;
    const int incoming_cumulative = delta > 0
        ? mismatch_prefix[state_index]
        : 0;
    const float incoming_score = delta > 0
        ? score_carry[state_index]
        : 0.0f;
    const int mismatch = active
        ? mismatch_count(query, key, row, key_pos, mask)
        : 0;
    const int local_prefix = subwarp_inclusive_sum(mismatch);
    const int cumulative = incoming_cumulative + local_prefix;
    const int old_key = key_pos - max_suffix_length - 1;
    const int old_lane = old_key - (key_start + first_key);
    const int local_old_prefix = __shfl_sync(
        0xffffffffu,
        cumulative,
        min(max(old_lane, 0), kSubwarp - 1),
        kSubwarp);
    int old_prefix = 0;
    if (active && key_pos >= max_suffix_length + 1) {
      old_prefix = old_lane >= 0
          ? local_old_prefix
          : prefix_history[history_base + key_pos % history_stride];
    }
    const float gate = active
        ? __expf(-mismatch_unit * static_cast<float>(mismatch))
        : 1.0f;
    const float correction = active && key_pos >= max_suffix_length
        ? __expf(
              -mismatch_unit *
              static_cast<float>(cumulative - old_prefix))
        : 0.0f;
    float coefficient = gate;
    float bias = active ? gate - correction : 0.0f;
    subwarp_affine_inclusive_scan(coefficient, bias);
    const float score = fmaf(coefficient, incoming_score, bias);
    if (active && lane >= count - history_stride) {
      prefix_history[
          history_base + key_pos % history_stride] = cumulative;
    }
    if (active) {
      scores[
          (static_cast<int64_t>(group) * seq_len + row) * seq_len +
          key_pos + 1] = score;
    }
    if (diagonal < diagonal_count && delta > 0 && lane == count - 1) {
      score_carry[state_index] = score;
      mismatch_prefix[state_index] = cumulative;
    }
  }
}


__global__ void initialize_row_stats_kernel(
    float* __restrict__ row_stats,
    int64_t row_count,
    float scale) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < row_count) {
    row_stats[row * 3] = kNullScore * scale;
    row_stats[row * 3 + 1] = 1.0f;
    row_stats[row * 3 + 2] = 0.0f;
  }
}


template <typename scalar_t>
__global__ void wavefront_stats_affine_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    float* __restrict__ row_stats,
    float* __restrict__ score_carry,
    int32_t* __restrict__ mismatch_prefix,
    int32_t* __restrict__ prefix_history,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int max_suffix_length,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    int compute_utility,
    int wave,
    int first_row_tile,
    int tiles_in_wave) {
  __shared__ float tile_scores[kTile * kTile];
  __shared__ float tile_utilities[kTile * kTile];
  __shared__ float utility_left[kTile * kTile];
  __shared__ float utility_right[kTile * kTile];
  const int group = blockIdx.x / tiles_in_wave;
  const int local_tile = blockIdx.x - group * tiles_in_wave;
  const int batch_heads = batch_size * num_heads;
  if (group >= batch_heads) {
    return;
  }
  const int head = group % num_heads;
  const int batch_index = group / num_heads;
  const int value_head = head / (num_heads / num_value_heads);
  const TileCoordinate tile =
      decode_tile(local_tile, wave, first_row_tile);
  const int row_start = tile.row_tile * kTile;
  const int key_start = tile.key_tile * kTile;
  const int row_count = min(kTile, seq_len - row_start);
  const int key_count = min(kTile, seq_len - key_start);
  const int32_t* query_words = packed_query +
      static_cast<int64_t>(group) * seq_len;
  const int32_t* key_words = packed_key +
      static_cast<int64_t>(group) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const int history_stride = max_suffix_length + 1;
  const int lane = threadIdx.x & (kSubwarp - 1);
  const int subwarp = threadIdx.x / kSubwarp;
  const int diagonal_count = row_count + key_count - 1;

  for (int index = threadIdx.x; index < kTile * kTile;
       index += blockDim.x) {
    tile_scores[index] = 0.0f;
  }
  __syncthreads();
  for (int diagonal_batch = 0;
       diagonal_batch * kSubwarps < diagonal_count;
       ++diagonal_batch) {
    const int diagonal = diagonal_batch * kSubwarps + subwarp;
    int first_row = 0;
    int first_key = 0;
    int count = 0;
    if (diagonal < diagonal_count) {
      diagonal_geometry(
          diagonal,
          row_count,
          key_count,
          first_row,
          first_key,
          count);
    }
    const int row_offset = first_row + lane;
    const int key_offset = first_key + lane;
    const int row = row_start + row_offset;
    const int key_pos = key_start + key_offset;
    const int delta = diagonal < diagonal_count
        ? row_start + first_row - (key_start + first_key)
        : 0;
    const bool active = diagonal < diagonal_count && lane < count && delta > 0;
    const int64_t state_index =
        static_cast<int64_t>(group) * seq_len + max(delta, 0);
    const int64_t history_base = state_index * history_stride;
    const int incoming_cumulative = delta > 0
        ? mismatch_prefix[state_index]
        : 0;
    const float incoming_score = delta > 0
        ? score_carry[state_index]
        : 0.0f;
    const int mismatch = active
        ? mismatch_count(query_words, key_words, row, key_pos, mask)
        : 0;
    const int local_prefix = subwarp_inclusive_sum(mismatch);
    const int cumulative = incoming_cumulative + local_prefix;
    const int old_key = key_pos - max_suffix_length - 1;
    const int old_lane = old_key - (key_start + first_key);
    const int local_old_prefix = __shfl_sync(
        0xffffffffu,
        cumulative,
        min(max(old_lane, 0), kSubwarp - 1),
        kSubwarp);
    int old_prefix = 0;
    if (active && key_pos >= max_suffix_length + 1) {
      old_prefix = old_lane >= 0
          ? local_old_prefix
          : prefix_history[
                history_base + key_pos % history_stride];
    }
    const float gate = active
        ? __expf(-mismatch_unit * static_cast<float>(mismatch))
        : 1.0f;
    const float correction = active && key_pos >= max_suffix_length
        ? __expf(
              -mismatch_unit *
              static_cast<float>(cumulative - old_prefix))
        : 0.0f;
    float coefficient = gate;
    float bias = active ? gate - correction : 0.0f;
    subwarp_affine_inclusive_scan(coefficient, bias);
    const float score = fmaf(coefficient, incoming_score, bias);
    if (active && lane >= count - history_stride) {
      prefix_history[
          history_base + key_pos % history_stride] = cumulative;
    }
    if (active) {
      tile_scores[row_offset * kTile + key_offset] = score;
    }
    if (diagonal < diagonal_count && delta > 0 && lane == count - 1) {
      score_carry[state_index] = score;
      mismatch_prefix[state_index] = cumulative;
    }
  }
  __syncthreads();
  if (compute_utility == 1) {
    contract_tile_utilities(
        value,
        grad_output,
        tile_utilities,
        utility_left,
        utility_right,
        batch_index,
        head,
        value_head,
        row_start,
        key_start,
        row_count,
        key_count,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim);
  }
  __syncthreads();

  if (threadIdx.x < row_count) {
    const int row_offset = threadIdx.x;
    const int row = row_start + row_offset;
    const int64_t stats_index =
        (static_cast<int64_t>(group) * seq_len + row) * 3;
    SoftmaxStats stats = {
        row_stats[stats_index],
        row_stats[stats_index + 1],
        row_stats[stats_index + 2]};
    for (int key_offset = 0; key_offset < key_count; ++key_offset) {
      const int key_pos = key_start + key_offset;
      if (key_pos >= row) {
        continue;
      }
      const int route = key_pos + 1;
      float utility = 0.0f;
      if (compute_utility == 1) {
        utility = tile_utilities[row_offset * kTile + key_offset];
      } else if (compute_utility == 2) {
        utility = direct_candidate_utility(
            value,
            grad_output,
            batch_index,
            head,
            value_head,
            row,
            route,
            seq_len,
            num_heads,
            num_value_heads,
            value_dim);
      }
      const float dropout_scale = compute_utility != 0
          ? attention_dropout_scale(
                dropout_seed,
                dropout_p,
                inverse_keep_probability,
                batch_index,
                head,
                row,
                route)
          : 0.0f;
      const ScoreTransform transformed = transform_score(
          tile_scores[row_offset * kTile + key_offset]);
      stats = append_item(
          stats,
          transformed.route_score * scale - logf(static_cast<float>(row)),
          dropout_scale * utility);
    }
    row_stats[stats_index] = stats.maximum;
    row_stats[stats_index + 1] = stats.normalizer;
    row_stats[stats_index + 2] = stats.utility_numerator;
  }
}


__global__ void finalize_row_stats_kernel(
    float* __restrict__ row_stats,
    int64_t row_count) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < row_count) {
    row_stats[row * 3 + 2] /= row_stats[row * 3 + 1];
  }
}


template <typename scalar_t>
__global__ void persistent_wavefront_stats_kernel(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    float* __restrict__ row_stats,
    float* __restrict__ score_carry,
    int32_t* __restrict__ mismatch_prefix,
    int32_t* __restrict__ prefix_history,
    int batch_size,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int num_value_heads,
    int value_dim,
    int max_suffix_length,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_unit,
    int compute_utility,
    int tile_count) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  __shared__ float tile_scores[kTile * kTile];
  __shared__ float tile_utilities[kTile * kTile];
  __shared__ float utility_left[kTile * kTile];
  __shared__ float utility_right[kTile * kTile];
  const int64_t total_rows =
      static_cast<int64_t>(batch_size) * num_heads * seq_len;
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t global_stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t row = global_thread; row < total_rows;
       row += global_stride) {
    row_stats[row * 3] = kNullScore * scale;
    row_stats[row * 3 + 1] = 1.0f;
    row_stats[row * 3 + 2] = 0.0f;
  }
  grid.sync();

  const int batch_heads = batch_size * num_heads;
  for (int wave = 0; wave < 2 * tile_count - 1; ++wave) {
    const int first_row_tile = wave_first_row_tile(wave, tile_count);
    const int last_row_tile = wave_last_row_tile(wave, tile_count);
    const int tiles_in_wave = max(0, last_row_tile - first_row_tile + 1);
    const int tasks = batch_heads * tiles_in_wave;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
      const int group = task / tiles_in_wave;
      const int local_tile = task - group * tiles_in_wave;
      const int head = group % num_heads;
      const int batch_index = group / num_heads;
      const int value_head = head / (num_heads / num_value_heads);
      const TileCoordinate tile =
          decode_tile(local_tile, wave, first_row_tile);
      const int row_start = tile.row_tile * kTile;
      const int key_start = tile.key_tile * kTile;
      const int row_count = min(kTile, seq_len - row_start);
      const int key_count = min(kTile, seq_len - key_start);
      const int32_t* query_words = packed_query +
          static_cast<int64_t>(group) * seq_len;
      const int32_t* key_words = packed_key +
          static_cast<int64_t>(group) * seq_len;
      const uint32_t mask = symbol_mask(symbol_dim);
      const int history_stride = max_suffix_length + 1;
      const int lane = threadIdx.x & (kSubwarp - 1);
      const int subwarp = threadIdx.x / kSubwarp;
      const int diagonal_count = row_count + key_count - 1;

      for (int index = threadIdx.x; index < kTile * kTile;
           index += blockDim.x) {
        tile_scores[index] = 0.0f;
      }
      __syncthreads();
      for (int diagonal_batch = 0;
           diagonal_batch * kSubwarps < diagonal_count;
           ++diagonal_batch) {
        const int diagonal = diagonal_batch * kSubwarps + subwarp;
        int first_row = 0;
        int first_key = 0;
        int count = 0;
        if (diagonal < diagonal_count) {
          diagonal_geometry(
              diagonal,
              row_count,
              key_count,
              first_row,
              first_key,
              count);
        }
        const int row_offset = first_row + lane;
        const int key_offset = first_key + lane;
        const int row = row_start + row_offset;
        const int key_pos = key_start + key_offset;
        const int delta = diagonal < diagonal_count
            ? row_start + first_row - (key_start + first_key)
            : 0;
        const bool active =
            diagonal < diagonal_count && lane < count && delta > 0;
        const int64_t state_index =
            static_cast<int64_t>(group) * seq_len + max(delta, 0);
        const int64_t history_base = state_index * history_stride;
        const int incoming_cumulative = delta > 0
            ? mismatch_prefix[state_index]
            : 0;
        const float incoming_score = delta > 0
            ? score_carry[state_index]
            : 0.0f;
        const int mismatch = active
            ? mismatch_count(query_words, key_words, row, key_pos, mask)
            : 0;
        const int local_prefix = subwarp_inclusive_sum(mismatch);
        const int cumulative = incoming_cumulative + local_prefix;
        const int old_key = key_pos - max_suffix_length - 1;
        const int old_lane = old_key - (key_start + first_key);
        const int local_old_prefix = __shfl_sync(
            0xffffffffu,
            cumulative,
            min(max(old_lane, 0), kSubwarp - 1),
            kSubwarp);
        int old_prefix = 0;
        if (active && key_pos >= max_suffix_length + 1) {
          old_prefix = old_lane >= 0
              ? local_old_prefix
              : prefix_history[
                    history_base + key_pos % history_stride];
        }
        const float gate = active
            ? __expf(-mismatch_unit * static_cast<float>(mismatch))
            : 1.0f;
        const float correction = active && key_pos >= max_suffix_length
            ? __expf(
                  -mismatch_unit *
                  static_cast<float>(cumulative - old_prefix))
            : 0.0f;
        float coefficient = gate;
        float bias = active ? gate - correction : 0.0f;
        subwarp_affine_inclusive_scan(coefficient, bias);
        const float score = fmaf(coefficient, incoming_score, bias);
        if (active && lane >= count - history_stride) {
          prefix_history[
              history_base + key_pos % history_stride] = cumulative;
        }
        if (active) {
          tile_scores[row_offset * kTile + key_offset] = score;
        }
        if (diagonal < diagonal_count && delta > 0 &&
            lane == count - 1) {
          score_carry[state_index] = score;
          mismatch_prefix[state_index] = cumulative;
        }
      }
      __syncthreads();
      if (compute_utility == 1) {
        contract_tile_utilities(
            value,
            grad_output,
            tile_utilities,
            utility_left,
            utility_right,
            batch_index,
            head,
            value_head,
            row_start,
            key_start,
            row_count,
            key_count,
            seq_len,
            num_heads,
            num_value_heads,
            value_dim);
      }
      __syncthreads();

      if (threadIdx.x < row_count) {
        const int row_offset = threadIdx.x;
        const int row = row_start + row_offset;
        const int64_t stats_index =
            (static_cast<int64_t>(group) * seq_len + row) * 3;
        SoftmaxStats stats = {
            row_stats[stats_index],
            row_stats[stats_index + 1],
            row_stats[stats_index + 2]};
        for (int key_offset = 0; key_offset < key_count; ++key_offset) {
          const int key_pos = key_start + key_offset;
          if (key_pos >= row) {
            continue;
          }
          const int route = key_pos + 1;
          float utility = 0.0f;
          if (compute_utility == 1) {
            utility = tile_utilities[row_offset * kTile + key_offset];
          } else if (compute_utility == 2) {
            utility = direct_candidate_utility(
                value,
                grad_output,
                batch_index,
                head,
                value_head,
                row,
                route,
                seq_len,
                num_heads,
                num_value_heads,
                value_dim);
          }
          const float dropout_scale = compute_utility != 0
              ? attention_dropout_scale(
                    dropout_seed,
                    dropout_p,
                    inverse_keep_probability,
                    batch_index,
                    head,
                    row,
                    route)
              : 0.0f;
          const ScoreTransform transformed = transform_score(
              tile_scores[row_offset * kTile + key_offset]);
          stats = append_item(
              stats,
              transformed.route_score * scale -
                  logf(static_cast<float>(row)),
              dropout_scale * utility);
        }
        row_stats[stats_index] = stats.maximum;
        row_stats[stats_index + 1] = stats.normalizer;
        row_stats[stats_index + 2] = stats.utility_numerator;
      }
      __syncthreads();
    }
    grid.sync();
  }

  for (int64_t row = global_thread; row < total_rows;
       row += global_stride) {
    row_stats[row * 3 + 2] /= row_stats[row * 3 + 1];
  }
}


__device__ __forceinline__ float direct_window_correction(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int key_position,
    int max_suffix_length,
    uint32_t mask,
    float mismatch_unit) {
  if (key_position < max_suffix_length) {
    return 0.0f;
  }
  int mismatch_sum = 0;
  for (int offset = 0; offset <= max_suffix_length; ++offset) {
    mismatch_sum += mismatch_count(
        query,
        key,
        row - offset,
        key_position - offset,
        mask);
  }
  return __expf(
      -mismatch_unit * static_cast<float>(mismatch_sum));
}


__global__ void wavefront_reverse_log_gate_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const float* __restrict__ raw_scores,
    const float* __restrict__ raw_score_vjp,
    float* __restrict__ log_gate_vjp,
    float* __restrict__ next_gate_state,
    float* __restrict__ next_score_vjp_state,
    float* __restrict__ future_correction_state,
    float* __restrict__ correction_ring,
    int batch_heads,
    int seq_len,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_unit,
    int wave,
    int first_row_tile,
    int tiles_in_wave) {
  const int group = blockIdx.x / tiles_in_wave;
  const int local_tile = blockIdx.x - group * tiles_in_wave;
  if (group >= batch_heads) {
    return;
  }
  const TileCoordinate tile =
      decode_tile(local_tile, wave, first_row_tile);
  const int row_start = tile.row_tile * kTile;
  const int key_start = tile.key_tile * kTile;
  const int row_count = min(kTile, seq_len - row_start);
  const int key_count = min(kTile, seq_len - key_start);
  const int32_t* query = packed_query +
      static_cast<int64_t>(group) * seq_len;
  const int32_t* key = packed_key +
      static_cast<int64_t>(group) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const int ring_stride = max_suffix_length + 1;

  for (int diagonal = threadIdx.x;
       diagonal < row_count + key_count - 1;
       diagonal += blockDim.x) {
    int first_row;
    int first_key;
    int count;
    diagonal_geometry(
        diagonal,
        row_count,
        key_count,
        first_row,
        first_key,
        count);
    const int query_position = row_start + first_row;
    const int key_position = key_start + first_key;
    const int delta = query_position - key_position;
    if (count <= 0 || delta <= 0) {
      continue;
    }
    const int64_t state_index =
        static_cast<int64_t>(group) * seq_len + delta;
    const int64_t ring_base = state_index * ring_stride;
    float next_gate = next_gate_state[state_index];
    float next_score_vjp = next_score_vjp_state[state_index];
    float future_correction = future_correction_state[state_index];
    const int chain_end_key = seq_len - delta - 1;
    for (int step = count - 1; step >= 0; --step) {
      const int row = query_position + step;
      const int key_pos = key_position + step;
      const int route = key_pos + 1;
      const int64_t matrix_index =
          (static_cast<int64_t>(group) * seq_len + row) * seq_len + route;
      const float score = raw_scores[matrix_index];
      const float route_vjp = raw_score_vjp[matrix_index];
      const float score_vjp =
          fmaf(next_gate, next_score_vjp, route_vjp);
      const float correction = direct_window_correction(
          query,
          key,
          row,
          key_pos,
          max_suffix_length,
          mask,
          mismatch_unit);
      const float correction_vjp = score_vjp * correction;
      const int ring_slot = key_pos % ring_stride;
      const float outgoing =
          key_pos + ring_stride <= chain_end_key
          ? correction_ring[ring_base + ring_slot]
          : 0.0f;
      future_correction += correction_vjp - outgoing;
      log_gate_vjp[matrix_index] =
          score_vjp * (score + correction) - future_correction;
      correction_ring[ring_base + ring_slot] = correction_vjp;
      const int mismatch = mismatch_count(
          query,
          key,
          row,
          key_pos,
          mask);
      next_gate = __expf(
          -mismatch_unit * static_cast<float>(mismatch));
      next_score_vjp = score_vjp;
    }
    next_gate_state[state_index] = next_gate;
    next_score_vjp_state[state_index] = next_score_vjp;
    future_correction_state[state_index] = future_correction;
  }
}


__device__ __forceinline__ float direct_raw_suffix_score(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int key_position,
    int max_suffix_length,
    uint32_t mask,
    float mismatch_unit) {
  const int steps = min(max_suffix_length, key_position + 1);
  float product = 1.0f;
  float score = 0.0f;
  for (int offset = 0; offset < steps; ++offset) {
    const int mismatch = mismatch_count(
        query,
        key,
        row - offset,
        key_position - offset,
        mask);
    product *= __expf(
        -mismatch_unit * static_cast<float>(mismatch));
    score += product;
  }
  return score;
}


__device__ __forceinline__ int subwarp_sum(int value) {
#pragma unroll
  for (int offset = kSubwarp / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(
        0xffffffffu,
        value,
        offset,
        kSubwarp);
  }
  return __shfl_sync(0xffffffffu, value, 0, kSubwarp);
}


__device__ __forceinline__ void recompute_tile_scores_affine(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    float* __restrict__ tile_scores,
    float* __restrict__ tile_corrections,
    int row_start,
    int key_start,
    int row_count,
    int key_count,
    int max_suffix_length,
    uint32_t mask,
    float mismatch_unit) {
  const int lane = threadIdx.x & (kSubwarp - 1);
  const int subwarp = threadIdx.x / kSubwarp;
  const int diagonal_count = row_count + key_count - 1;

  for (int diagonal_batch = 0;
       diagonal_batch * kSubwarps < diagonal_count;
       ++diagonal_batch) {
    const int diagonal = diagonal_batch * kSubwarps + subwarp;
    int first_row = 0;
    int first_key = 0;
    int count = 0;
    if (diagonal < diagonal_count) {
      diagonal_geometry(
          diagonal,
          row_count,
          key_count,
          first_row,
          first_key,
          count);
    }
    const int first_query_position = row_start + first_row;
    const int first_key_position = key_start + first_key;
    const int delta = diagonal < diagonal_count
        ? first_query_position - first_key_position
        : 0;
    const bool valid_diagonal =
        diagonal < diagonal_count && count > 0 && delta > 0;

    float incoming_score = 0.0f;
    const int halo_begin = max(0, first_key_position - max_suffix_length);
    const int halo_count = valid_diagonal
        ? first_key_position - halo_begin
        : 0;
    int base_mismatch = 0;
    for (int halo_offset = 0;
         halo_offset < max_suffix_length;
         halo_offset += kSubwarp) {
      const int local_halo = halo_offset + lane;
      const bool active_halo = local_halo < halo_count;
      const int key_position = halo_begin + local_halo;
      const int query_position = key_position + delta;
      const int mismatch = active_halo
          ? mismatch_count(
                query,
                key,
                query_position,
                key_position,
                mask)
          : 0;
      base_mismatch += mismatch;
      const float gate = active_halo
          ? __expf(-mismatch_unit * static_cast<float>(mismatch))
          : 1.0f;
      float coefficient = gate;
      float bias = active_halo ? gate : 0.0f;
      subwarp_affine_inclusive_scan(coefficient, bias);
      const float block_coefficient = __shfl_sync(
          0xffffffffu,
          coefficient,
          kSubwarp - 1,
          kSubwarp);
      const float block_bias = __shfl_sync(
          0xffffffffu,
          bias,
          kSubwarp - 1,
          kSubwarp);
      incoming_score = fmaf(
          block_coefficient,
          incoming_score,
          block_bias);
    }
    base_mismatch = subwarp_sum(base_mismatch);

    const int row_offset = first_row + lane;
    const int key_offset = first_key + lane;
    const int row = row_start + row_offset;
    const int key_position = key_start + key_offset;
    const bool active = valid_diagonal && lane < count;
    const int mismatch = active
        ? mismatch_count(query, key, row, key_position, mask)
        : 0;
    const int local_prefix = subwarp_inclusive_sum(mismatch);
    const int outgoing_position =
        first_key_position - max_suffix_length + lane;
    const int outgoing_mismatch =
        valid_diagonal && outgoing_position >= 0
        ? mismatch_count(
              query,
              key,
              outgoing_position + delta,
              outgoing_position,
              mask)
        : 0;
    const int outgoing_prefix =
        subwarp_inclusive_sum(outgoing_mismatch) - outgoing_mismatch;
    const int correction_mismatch =
        base_mismatch - outgoing_prefix + local_prefix;
    const float gate = active
        ? __expf(-mismatch_unit * static_cast<float>(mismatch))
        : 1.0f;
    const float correction = active && key_position >= max_suffix_length
        ? __expf(
              -mismatch_unit * static_cast<float>(correction_mismatch))
        : 0.0f;
    float coefficient = gate;
    float bias = active ? gate - correction : 0.0f;
    subwarp_affine_inclusive_scan(coefficient, bias);
    const float score = fmaf(coefficient, incoming_score, bias);
    if (active) {
      tile_scores[row_offset * kTile + key_offset] = score;
      tile_corrections[row_offset * kTile + key_offset] = correction;
    }
  }
}


template <typename scalar_t>
__device__ __forceinline__ void process_full_reverse_tile(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    float* __restrict__ next_gate_state,
    float* __restrict__ next_score_vjp_state,
    float* __restrict__ future_correction_state,
    float* __restrict__ correction_ring,
    float* __restrict__ tile_scores,
    float* __restrict__ tile_raw_vjp,
    float* __restrict__ tile_probability,
    float* __restrict__ tile_gate_vjp,
    float* __restrict__ utility_stage,
    int batch_size,
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
    float mismatch_unit,
    int gradient_mask,
    int task_index,
    int head_group_offset,
    int wave,
    int first_row_tile,
    int tiles_in_wave) {
  const int value_group = task_index / tiles_in_wave;
  const int local_tile = task_index - value_group * tiles_in_wave;
  const int batch_value_heads = batch_size * num_value_heads;
  if (value_group >= batch_value_heads) {
    return;
  }
  const int value_head = value_group % num_value_heads;
  const int batch_index = value_group / num_value_heads;
  const int heads_per_value = num_heads / num_value_heads;
  const int head = value_head * heads_per_value + head_group_offset;
  const int group = batch_index * num_heads + head;
  const TileCoordinate tile =
      decode_tile(local_tile, wave, first_row_tile);
  const int row_start = tile.row_tile * kTile;
  const int key_start = tile.key_tile * kTile;
  const int row_count = min(kTile, seq_len - row_start);
  const int key_count = min(kTile, seq_len - key_start);
  const int32_t* query_words = packed_query +
      static_cast<int64_t>(group) * seq_len;
  const int32_t* key_words = packed_key +
      static_cast<int64_t>(group) * seq_len;
  const uint32_t mask = symbol_mask(symbol_dim);
  const int matrix_items = row_count * key_count;

  for (int index = threadIdx.x; index < kTile * kTile;
       index += blockDim.x) {
    tile_scores[index] = 0.0f;
    tile_raw_vjp[index] = 0.0f;
    tile_probability[index] = 0.0f;
    tile_gate_vjp[index] = 0.0f;
  }
  __syncthreads();
  if (max_suffix_length >= kTile) {
    recompute_tile_scores_affine(
        query_words,
        key_words,
        tile_scores,
        tile_gate_vjp,
        row_start,
        key_start,
        row_count,
        key_count,
        max_suffix_length,
        mask,
        mismatch_unit);
  }
  __syncthreads();
  if ((gradient_mask & (kGradQuery | kGradKey)) != 0) {
    contract_tile_utilities(
        value,
        grad_output,
        tile_raw_vjp,
        tile_probability,
        utility_stage,
        batch_index,
        head,
        value_head,
        row_start,
        key_start,
        row_count,
        key_count,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim);
  }
  __syncthreads();
  for (int index = threadIdx.x; index < matrix_items;
       index += blockDim.x) {
    const int row_offset = index / key_count;
    const int key_offset = index - row_offset * key_count;
    const int row = row_start + row_offset;
    const int key_pos = key_start + key_offset;
    if (key_pos < row) {
      const int route = key_pos + 1;
      const float raw_score = max_suffix_length >= kTile
          ? tile_scores[row_offset * kTile + key_offset]
          : direct_raw_suffix_score(
                query_words,
                key_words,
                row,
                key_pos,
                max_suffix_length,
                mask,
                mismatch_unit);
      const float utility =
          tile_raw_vjp[row_offset * kTile + key_offset];
      const int64_t stats_index =
          (static_cast<int64_t>(group) * seq_len + row) * 3;
      const ScoreTransform transformed = transform_score(raw_score);
      const float probability = __expf(
          transformed.route_score * scale - logf(static_cast<float>(row)) -
          row_stats[stats_index]) / row_stats[stats_index + 1];
      const float dropout_scale = attention_dropout_scale(
          dropout_seed,
          dropout_p,
          inverse_keep_probability,
          batch_index,
          head,
          row,
          route);
      tile_scores[row_offset * kTile + key_offset] = raw_score;
      tile_probability[row_offset * kTile + key_offset] =
          probability * dropout_scale;
      tile_raw_vjp[row_offset * kTile + key_offset] =
          (gradient_mask & (kGradQuery | kGradKey)) != 0
          ? scale * probability *
                (dropout_scale * utility - row_stats[stats_index + 2]) *
                transformed.raw_vjp_multiplier
          : 0.0f;
    }
  }
  __syncthreads();

  if ((gradient_mask & kGradValue) != 0) {
    const int value_items = key_count * value_dim;
    for (int index = threadIdx.x; index < value_items;
         index += blockDim.x) {
      const int key_offset = index / value_dim;
      const int value_offset = index - key_offset * value_dim;
      const int key_pos = key_start + key_offset;
      const int route = key_pos + 1;
      if (route < seq_len) {
        float contribution = 0.0f;
        for (int row_offset = 0; row_offset < row_count; ++row_offset) {
          const int row = row_start + row_offset;
          if (key_pos < row) {
            const int64_t grad_index =
                ((static_cast<int64_t>(batch_index) * seq_len + row) *
                     num_heads +
                 head) *
                    value_dim +
                value_offset;
            contribution +=
                tile_probability[row_offset * kTile + key_offset] *
                read_float(grad_output, grad_index);
          }
        }
        const int64_t target =
            ((static_cast<int64_t>(batch_index) * seq_len + route) *
                 num_value_heads +
             value_head) *
                value_dim +
            value_offset;
        grad_value[target] += contribution;
      }
    }
  }
  __syncthreads();

  if ((gradient_mask & (kGradQuery | kGradKey)) != 0) {
    const int ring_stride = max_suffix_length + 1;
    const int diagonal_count = row_count + key_count - 1;
    for (int diagonal = threadIdx.x;
         diagonal < diagonal_count;
         diagonal += blockDim.x) {
      int first_row;
      int first_key;
      int count;
      diagonal_geometry(
          diagonal,
          row_count,
          key_count,
          first_row,
          first_key,
          count);
      const int query_position = row_start + first_row;
      const int key_position = key_start + first_key;
      const int delta = query_position - key_position;
      if (count <= 0 || delta <= 0) {
        continue;
      }
      const int64_t state_index =
          static_cast<int64_t>(group) * seq_len + delta;
      const int64_t ring_base = state_index * ring_stride;
      float next_gate = next_gate_state[state_index];
      float next_score_vjp = next_score_vjp_state[state_index];
      float future_correction = future_correction_state[state_index];
      const int chain_end_key = seq_len - delta - 1;
      for (int step = count - 1; step >= 0; --step) {
        const int row_offset = first_row + step;
        const int key_offset = first_key + step;
        const int row = row_start + row_offset;
        const int key_pos = key_start + key_offset;
        const float raw_score =
            tile_scores[row_offset * kTile + key_offset];
        const float route_vjp =
            tile_raw_vjp[row_offset * kTile + key_offset];
        const float score_vjp =
            fmaf(next_gate, next_score_vjp, route_vjp);
        const float correction = max_suffix_length >= kTile
            ? tile_gate_vjp[row_offset * kTile + key_offset]
            : direct_window_correction(
                  query_words,
                  key_words,
                  row,
                  key_pos,
                  max_suffix_length,
                  mask,
                  mismatch_unit);
        const float correction_vjp = score_vjp * correction;
        const int ring_slot = key_pos % ring_stride;
        const float outgoing =
            key_pos + ring_stride <= chain_end_key
            ? correction_ring[ring_base + ring_slot]
            : 0.0f;
        future_correction += correction_vjp - outgoing;
        tile_gate_vjp[row_offset * kTile + key_offset] =
            score_vjp * (raw_score + correction) - future_correction;
        correction_ring[ring_base + ring_slot] = correction_vjp;
        const int mismatch = mismatch_count(
            query_words,
            key_words,
            row,
            key_pos,
            mask);
        next_gate = __expf(
            -mismatch_unit * static_cast<float>(mismatch));
        next_score_vjp = score_vjp;
      }
      next_gate_state[state_index] = next_gate;
      next_score_vjp_state[state_index] = next_score_vjp;
      future_correction_state[state_index] = future_correction;
    }
  }
  __syncthreads();

  const float symbol_scale =
      0.5f * mismatch_scale / static_cast<float>(symbol_dim);
  if ((gradient_mask & kGradQuery) != 0) {
    const int query_items = row_count * symbol_dim;
    for (int index = threadIdx.x; index < query_items;
         index += blockDim.x) {
      const int row_offset = index / symbol_dim;
      const int bit = index - row_offset * symbol_dim;
      const int row = row_start + row_offset;
      float contribution = 0.0f;
      for (int key_offset = 0; key_offset < key_count; ++key_offset) {
        const int key_pos = key_start + key_offset;
        if (key_pos < row) {
          contribution +=
              tile_gate_vjp[row_offset * kTile + key_offset] *
              static_cast<float>(sign_from_bit(
                  static_cast<uint32_t>(key_words[key_pos]),
                  bit));
        }
      }
      const int64_t target =
          ((static_cast<int64_t>(batch_index) * seq_len + row) * num_heads +
           head) *
              symbol_dim +
          bit;
      grad_query[target] += symbol_scale * contribution;
    }
  }
  if ((gradient_mask & kGradKey) != 0) {
    const int key_items = key_count * symbol_dim;
    for (int index = threadIdx.x; index < key_items;
         index += blockDim.x) {
      const int key_offset = index / symbol_dim;
      const int bit = index - key_offset * symbol_dim;
      const int key_pos = key_start + key_offset;
      float contribution = 0.0f;
      for (int row_offset = 0; row_offset < row_count; ++row_offset) {
        const int row = row_start + row_offset;
        if (key_pos < row) {
          contribution +=
              tile_gate_vjp[row_offset * kTile + key_offset] *
              static_cast<float>(sign_from_bit(
                  static_cast<uint32_t>(query_words[row]),
                  bit));
        }
      }
      const int64_t target =
          ((static_cast<int64_t>(batch_index) * seq_len + key_pos) *
               num_heads +
           head) *
              symbol_dim +
          bit;
      grad_key[target] += symbol_scale * contribution;
    }
  }
}


template <typename scalar_t>
__global__ void wavefront_full_reverse_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    float* __restrict__ next_gate_state,
    float* __restrict__ next_score_vjp_state,
    float* __restrict__ future_correction_state,
    float* __restrict__ correction_ring,
    int batch_size,
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
    float mismatch_unit,
    int gradient_mask,
    int head_group_offset,
    int wave,
    int first_row_tile,
    int tiles_in_wave) {
  __shared__ float tile_scores[kTile * kTile];
  __shared__ float tile_raw_vjp[kTile * kTile];
  __shared__ float tile_probability[kTile * kTile];
  __shared__ float tile_gate_vjp[kTile * kTile];
  __shared__ float utility_stage[kTile * kTile];
  process_full_reverse_tile(
      query,
      key,
      value,
      grad_output,
      packed_query,
      packed_key,
      dropout_seed,
      row_stats,
      grad_query,
      grad_key,
      grad_value,
      next_gate_state,
      next_score_vjp_state,
      future_correction_state,
      correction_ring,
      tile_scores,
      tile_raw_vjp,
      tile_probability,
      tile_gate_vjp,
      utility_stage,
      batch_size,
      seq_len,
      num_heads,
      symbol_dim,
      num_value_heads,
      value_dim,
      max_suffix_length,
      scale,
      dropout_p,
      inverse_keep_probability,
      mismatch_scale,
      mismatch_unit,
      gradient_mask,
      blockIdx.x,
      head_group_offset,
      wave,
      first_row_tile,
      tiles_in_wave);
}


template <typename scalar_t>
__global__ void persistent_wavefront_full_reverse_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ dropout_seed,
    const float* __restrict__ row_stats,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    float* __restrict__ next_gate_state,
    float* __restrict__ next_score_vjp_state,
    float* __restrict__ future_correction_state,
    float* __restrict__ correction_ring,
    int batch_size,
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
    float mismatch_unit,
    int gradient_mask,
    int head_group_offset,
    int tile_count) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  __shared__ float tile_scores[kTile * kTile];
  __shared__ float tile_raw_vjp[kTile * kTile];
  __shared__ float tile_probability[kTile * kTile];
  __shared__ float tile_gate_vjp[kTile * kTile];
  __shared__ float utility_stage[kTile * kTile];
  const int tasks_per_group = batch_size * num_value_heads;

  for (int wave = 2 * tile_count - 2; wave >= 0; --wave) {
    const int first_row_tile = wave_first_row_tile(wave, tile_count);
    const int last_row_tile = wave_last_row_tile(wave, tile_count);
    const int tiles_in_wave = max(0, last_row_tile - first_row_tile + 1);
    const int tasks = tasks_per_group * tiles_in_wave;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
      process_full_reverse_tile(
          query,
          key,
          value,
          grad_output,
          packed_query,
          packed_key,
          dropout_seed,
          row_stats,
          grad_query,
          grad_key,
          grad_value,
          next_gate_state,
          next_score_vjp_state,
          future_correction_state,
          correction_ring,
          tile_scores,
          tile_raw_vjp,
          tile_probability,
          tile_gate_vjp,
          utility_stage,
          batch_size,
          seq_len,
          num_heads,
          symbol_dim,
          num_value_heads,
          value_dim,
          max_suffix_length,
          scale,
          dropout_p,
          inverse_keep_probability,
          mismatch_scale,
          mismatch_unit,
          gradient_mask,
          task,
          head_group_offset,
          wave,
          first_row_tile,
          tiles_in_wave);
      __syncthreads();
    }
    grid.sync();
  }
}


template <typename scalar_t>
__global__ void finalize_direct_vjp_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int64_t query_elements,
    int64_t key_elements,
    int64_t value_elements) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < query_elements) {
    grad_query[index] *= softsign_derivative(read_float(query, index));
    return;
  }
  const int64_t key_index = index - query_elements;
  if (key_index < key_elements) {
    grad_key[key_index] *= softsign_derivative(read_float(key, key_index));
    return;
  }
  const int64_t value_index = key_index - key_elements;
  if (value_index < value_elements) {
    grad_value[value_index] *=
        softsign_derivative(read_float(value, value_index));
  }
}

}  // namespace


torch::Tensor rosa_soft_wavefront_scores_cuda(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    float mismatch_scale,
    int plan) {
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int batch_heads = batch_size * num_heads;
  const int tile_count = (seq_len + kTile - 1) / kTile;
  const auto float_options =
      packed_query_symbols.options().dtype(torch::kFloat32);
  const auto int_options =
      packed_query_symbols.options().dtype(torch::kInt32);
  torch::Tensor scores = torch::zeros(
      {batch_size, num_heads, seq_len, seq_len},
      float_options);
  torch::Tensor score_carry = torch::zeros(
      {batch_heads, seq_len},
      float_options);
  torch::Tensor mismatch_prefix = torch::zeros(
      {batch_heads, seq_len},
      int_options);
  torch::Tensor prefix_history = torch::zeros(
      {batch_heads, seq_len, max_suffix_length + 1},
      int_options);
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int wave = 0; wave < 2 * tile_count - 1; ++wave) {
    const int first_row_tile = wave_first_row_tile(wave, tile_count);
    const int last_row_tile = wave_last_row_tile(wave, tile_count);
    const int tiles_in_wave = max(0, last_row_tile - first_row_tile + 1);
    if (tiles_in_wave == 0) {
      continue;
    }
    const int blocks = batch_heads * tiles_in_wave;
    if (plan == 0) {
      wavefront_score_sequential_kernel<<<
          blocks,
          kThreads,
          0,
          stream>>>(
          packed_query_symbols.data_ptr<int32_t>(),
          packed_key_symbols.data_ptr<int32_t>(),
          scores.data_ptr<float>(),
          score_carry.data_ptr<float>(),
          mismatch_prefix.data_ptr<int32_t>(),
          prefix_history.data_ptr<int32_t>(),
          batch_heads,
          seq_len,
          static_cast<int>(symbol_dim),
          static_cast<int>(max_suffix_length),
          mismatch_unit,
          wave,
          first_row_tile,
          tiles_in_wave);
    } else {
      wavefront_score_affine_kernel<<<
          blocks,
          kThreads,
          0,
          stream>>>(
          packed_query_symbols.data_ptr<int32_t>(),
          packed_key_symbols.data_ptr<int32_t>(),
          scores.data_ptr<float>(),
          score_carry.data_ptr<float>(),
          mismatch_prefix.data_ptr<int32_t>(),
          prefix_history.data_ptr<int32_t>(),
          batch_heads,
          seq_len,
          static_cast<int>(symbol_dim),
          static_cast<int>(max_suffix_length),
          mismatch_unit,
          wave,
          first_row_tile,
          tiles_in_wave);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return scores;
}


torch::Tensor rosa_soft_wavefront_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int compute_utility) {
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int num_value_heads = value.size(2);
  const int value_dim = value.size(3);
  const int batch_heads = batch_size * num_heads;
  const int tile_count = (seq_len + kTile - 1) / kTile;
  const auto float_options = value.options().dtype(torch::kFloat32);
  const auto int_options =
      packed_query_symbols.options().dtype(torch::kInt32);
  torch::Tensor row_stats = torch::empty(
      {batch_size, num_heads, seq_len, 3},
      float_options);
  torch::Tensor score_carry = torch::zeros(
      {batch_heads, seq_len},
      float_options);
  torch::Tensor mismatch_prefix = torch::zeros(
      {batch_heads, seq_len},
      int_options);
  torch::Tensor prefix_history = torch::zeros(
      {batch_heads, seq_len, max_suffix_length + 1},
      int_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t total_rows =
      static_cast<int64_t>(batch_heads) * seq_len;
  constexpr int kLinearThreads = 256;
  const int row_blocks = static_cast<int>(
      (total_rows + kLinearThreads - 1) / kLinearThreads);
  initialize_row_stats_kernel<<<
      row_blocks,
      kLinearThreads,
      0,
      stream>>>(row_stats.data_ptr<float>(), total_rows, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);

  DISPATCH_ROSA_FLOAT_TYPES(
      value.scalar_type(),
      "rosa_soft_wavefront_stats_cuda",
      [&] {
        for (int wave = 0; wave < 2 * tile_count - 1; ++wave) {
          const int first_row_tile = wave_first_row_tile(wave, tile_count);
          const int last_row_tile = wave_last_row_tile(wave, tile_count);
          const int tiles_in_wave =
              max(0, last_row_tile - first_row_tile + 1);
          if (tiles_in_wave == 0) {
            continue;
          }
          wavefront_stats_affine_kernel<scalar_t><<<
              batch_heads * tiles_in_wave,
              kThreads,
              0,
              stream>>>(
              value.data_ptr<scalar_t>(),
              grad_output.data_ptr<scalar_t>(),
              packed_query_symbols.data_ptr<int32_t>(),
              packed_key_symbols.data_ptr<int32_t>(),
              dropout_seed.data_ptr<int64_t>(),
              row_stats.data_ptr<float>(),
              score_carry.data_ptr<float>(),
              mismatch_prefix.data_ptr<int32_t>(),
              prefix_history.data_ptr<int32_t>(),
              batch_size,
              seq_len,
              num_heads,
              static_cast<int>(symbol_dim),
              num_value_heads,
              value_dim,
              static_cast<int>(max_suffix_length),
              scale,
              dropout_p,
              inverse_keep_probability,
              mismatch_unit,
              compute_utility,
              wave,
              first_row_tile,
              tiles_in_wave);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  finalize_row_stats_kernel<<<
      row_blocks,
      kLinearThreads,
      0,
      stream>>>(row_stats.data_ptr<float>(), total_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return row_stats;
}


torch::Tensor rosa_soft_persistent_wavefront_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int compute_utility) {
  const cudaDeviceProp* properties =
      at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(
      properties->cooperativeLaunch,
      "persistent wavefront requires cooperative launch support");
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int num_value_heads = value.size(2);
  const int value_dim = value.size(3);
  const int batch_heads = batch_size * num_heads;
  const int tile_count = (seq_len + kTile - 1) / kTile;
  const auto float_options = value.options().dtype(torch::kFloat32);
  const auto int_options =
      packed_query_symbols.options().dtype(torch::kInt32);
  torch::Tensor row_stats = torch::empty(
      {batch_size, num_heads, seq_len, 3},
      float_options);
  torch::Tensor score_carry = torch::zeros(
      {batch_heads, seq_len},
      float_options);
  torch::Tensor mismatch_prefix = torch::zeros(
      {batch_heads, seq_len},
      int_options);
  torch::Tensor prefix_history = torch::zeros(
      {batch_heads, seq_len, max_suffix_length + 1},
      int_options);
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(
      value.scalar_type(),
      "rosa_soft_persistent_wavefront_stats_cuda",
      [&] {
        int active_blocks_per_sm = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            persistent_wavefront_stats_kernel<scalar_t>,
            kThreads,
            0));
        TORCH_CHECK(
            active_blocks_per_sm > 0,
            "persistent wavefront kernel has zero occupancy");
        const int persistent_blocks =
            properties->multiProcessorCount * active_blocks_per_sm;
        const scalar_t* value_pointer = value.data_ptr<scalar_t>();
        const scalar_t* grad_output_pointer =
            grad_output.data_ptr<scalar_t>();
        const int32_t* query_pointer =
            packed_query_symbols.data_ptr<int32_t>();
        const int32_t* key_pointer =
            packed_key_symbols.data_ptr<int32_t>();
        const int64_t* seed_pointer = dropout_seed.data_ptr<int64_t>();
        float* stats_pointer = row_stats.data_ptr<float>();
        float* score_pointer = score_carry.data_ptr<float>();
        int32_t* mismatch_pointer = mismatch_prefix.data_ptr<int32_t>();
        int32_t* history_pointer = prefix_history.data_ptr<int32_t>();
        int batch_size_value = batch_size;
        int seq_len_value = seq_len;
        int num_heads_value = num_heads;
        int symbol_dim_value = static_cast<int>(symbol_dim);
        int num_value_heads_value = num_value_heads;
        int value_dim_value = value_dim;
        int window_value = static_cast<int>(max_suffix_length);
        int tile_count_value = tile_count;
        int compute_utility_value = compute_utility;
        float scale_value = scale;
        float dropout_value = dropout_p;
        float inverse_keep_value = inverse_keep_probability;
        float mismatch_unit_value = mismatch_unit;
        void* arguments[] = {
            &value_pointer,
            &grad_output_pointer,
            &query_pointer,
            &key_pointer,
            &seed_pointer,
            &stats_pointer,
            &score_pointer,
            &mismatch_pointer,
            &history_pointer,
            &batch_size_value,
            &seq_len_value,
            &num_heads_value,
            &symbol_dim_value,
            &num_value_heads_value,
            &value_dim_value,
            &window_value,
            &scale_value,
            &dropout_value,
            &inverse_keep_value,
            &mismatch_unit_value,
            &compute_utility_value,
            &tile_count_value};
        C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(
                persistent_wavefront_stats_kernel<scalar_t>),
            dim3(persistent_blocks),
            dim3(kThreads),
            arguments,
            0,
            stream));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return row_stats;
}


torch::Tensor rosa_soft_wavefront_log_gate_vjp_cuda(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& raw_scores,
    const torch::Tensor& raw_score_vjp,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    float mismatch_scale) {
  const int batch_size = packed_query_symbols.size(0);
  const int num_heads = packed_query_symbols.size(1);
  const int seq_len = packed_query_symbols.size(2);
  const int batch_heads = batch_size * num_heads;
  const int tile_count = (seq_len + kTile - 1) / kTile;
  const auto float_options = raw_scores.options().dtype(torch::kFloat32);
  torch::Tensor log_gate_vjp = torch::zeros_like(raw_scores);
  torch::Tensor next_gate = torch::zeros(
      {batch_heads, seq_len},
      float_options);
  torch::Tensor next_score_vjp = torch::zeros(
      {batch_heads, seq_len},
      float_options);
  torch::Tensor future_correction = torch::zeros(
      {batch_heads, seq_len},
      float_options);
  torch::Tensor correction_ring = torch::zeros(
      {batch_heads, seq_len, max_suffix_length + 1},
      float_options);
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int wave = 2 * tile_count - 2; wave >= 0; --wave) {
    const int first_row_tile = wave_first_row_tile(wave, tile_count);
    const int last_row_tile = wave_last_row_tile(wave, tile_count);
    const int tiles_in_wave = max(0, last_row_tile - first_row_tile + 1);
    if (tiles_in_wave == 0) {
      continue;
    }
    wavefront_reverse_log_gate_kernel<<<
        batch_heads * tiles_in_wave,
        kThreads,
        0,
        stream>>>(
        packed_query_symbols.data_ptr<int32_t>(),
        packed_key_symbols.data_ptr<int32_t>(),
        raw_scores.data_ptr<float>(),
        raw_score_vjp.data_ptr<float>(),
        log_gate_vjp.data_ptr<float>(),
        next_gate.data_ptr<float>(),
        next_score_vjp.data_ptr<float>(),
        future_correction.data_ptr<float>(),
        correction_ring.data_ptr<float>(),
        batch_heads,
        seq_len,
        static_cast<int>(symbol_dim),
        static_cast<int>(max_suffix_length),
        mismatch_unit,
        wave,
        first_row_tile,
        tiles_in_wave);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return log_gate_vjp;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_wavefront_vjp_cuda(
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
    int execution_plan) {
  const int batch_size = query.size(0);
  const int seq_len = query.size(1);
  const int num_heads = query.size(2);
  const int symbol_dim = query.size(3);
  const int num_value_heads = value.size(2);
  const int value_dim = value.size(3);
  const int batch_heads = batch_size * num_heads;
  const int heads_per_value = num_heads / num_value_heads;
  const int tile_count = (seq_len + kTile - 1) / kTile;
  const int compute_utility =
      (gradient_mask & (kGradQuery | kGradKey)) != 0 ? 1 : 0;
  const auto float_options = query.options().dtype(torch::kFloat32);
  torch::Tensor row_stats = execution_plan == 0
      ? rosa_soft_wavefront_stats_cuda(
            value,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            symbol_dim,
            max_suffix_length,
            scale,
            dropout_p,
            mismatch_scale,
            compute_utility)
      : rosa_soft_persistent_wavefront_stats_cuda(
            value,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            symbol_dim,
            max_suffix_length,
            scale,
            dropout_p,
            mismatch_scale,
            compute_utility);
  torch::Tensor grad_query = (gradient_mask & kGradQuery) != 0
      ? torch::zeros(query.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_key = (gradient_mask & kGradKey) != 0
      ? torch::zeros(key.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_value = (gradient_mask & kGradValue) != 0
      ? torch::zeros(value.sizes(), float_options)
      : torch::empty({0}, float_options);
  const bool needs_symbol_vjp = compute_utility != 0;
  torch::Tensor next_gate = needs_symbol_vjp
      ? torch::zeros({batch_heads, seq_len}, float_options)
      : torch::empty({0}, float_options);
  torch::Tensor next_score_vjp = needs_symbol_vjp
      ? torch::zeros({batch_heads, seq_len}, float_options)
      : torch::empty({0}, float_options);
  torch::Tensor future_correction = needs_symbol_vjp
      ? torch::zeros({batch_heads, seq_len}, float_options)
      : torch::empty({0}, float_options);
  torch::Tensor correction_ring = needs_symbol_vjp
      ? torch::zeros(
            {batch_heads, seq_len, max_suffix_length + 1},
            float_options)
      : torch::empty({0}, float_options);
  const float mismatch_unit =
      mismatch_scale / static_cast<float>(symbol_dim);
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_wavefront_vjp_cuda",
      [&] {
        if (execution_plan == 0) {
          for (int wave = 2 * tile_count - 2; wave >= 0; --wave) {
            const int first_row_tile = wave_first_row_tile(wave, tile_count);
            const int last_row_tile = wave_last_row_tile(wave, tile_count);
            const int tiles_in_wave =
                max(0, last_row_tile - first_row_tile + 1);
            if (tiles_in_wave == 0) {
              continue;
            }
            for (int head_group_offset = 0;
                 head_group_offset < heads_per_value;
                 ++head_group_offset) {
              wavefront_full_reverse_kernel<scalar_t><<<
                  batch_size * num_value_heads * tiles_in_wave,
                  kThreads,
                  0,
                  stream>>>(
                  query.data_ptr<scalar_t>(),
                  key.data_ptr<scalar_t>(),
                  value.data_ptr<scalar_t>(),
                  grad_output.data_ptr<scalar_t>(),
                  packed_query_symbols.data_ptr<int32_t>(),
                  packed_key_symbols.data_ptr<int32_t>(),
                  dropout_seed.data_ptr<int64_t>(),
                  row_stats.data_ptr<float>(),
                  grad_query.data_ptr<float>(),
                  grad_key.data_ptr<float>(),
                  grad_value.data_ptr<float>(),
                  next_gate.data_ptr<float>(),
                  next_score_vjp.data_ptr<float>(),
                  future_correction.data_ptr<float>(),
                  correction_ring.data_ptr<float>(),
                  batch_size,
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
                  mismatch_unit,
                  gradient_mask,
                  head_group_offset,
                  wave,
                  first_row_tile,
                  tiles_in_wave);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
          }
        } else {
          const cudaDeviceProp* properties =
              at::cuda::getCurrentDeviceProperties();
          TORCH_CHECK(
              properties->cooperativeLaunch,
              "persistent wavefront requires cooperative launch support");
          int active_blocks_per_sm = 0;
          C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_blocks_per_sm,
              persistent_wavefront_full_reverse_kernel<scalar_t>,
              kThreads,
              0));
          TORCH_CHECK(
              active_blocks_per_sm > 0,
              "persistent reverse wavefront kernel has zero occupancy");
          const int persistent_blocks =
              properties->multiProcessorCount * active_blocks_per_sm;
          const scalar_t* query_pointer = query.data_ptr<scalar_t>();
          const scalar_t* key_pointer = key.data_ptr<scalar_t>();
          const scalar_t* value_pointer = value.data_ptr<scalar_t>();
          const scalar_t* grad_output_pointer =
              grad_output.data_ptr<scalar_t>();
          const int32_t* packed_query_pointer =
              packed_query_symbols.data_ptr<int32_t>();
          const int32_t* packed_key_pointer =
              packed_key_symbols.data_ptr<int32_t>();
          const int64_t* seed_pointer = dropout_seed.data_ptr<int64_t>();
          const float* stats_pointer = row_stats.data_ptr<float>();
          float* grad_query_pointer = grad_query.data_ptr<float>();
          float* grad_key_pointer = grad_key.data_ptr<float>();
          float* grad_value_pointer = grad_value.data_ptr<float>();
          float* next_gate_pointer = next_gate.data_ptr<float>();
          float* next_score_pointer = next_score_vjp.data_ptr<float>();
          float* future_pointer = future_correction.data_ptr<float>();
          float* ring_pointer = correction_ring.data_ptr<float>();
          int batch_size_value = batch_size;
          int seq_len_value = seq_len;
          int num_heads_value = num_heads;
          int symbol_dim_value = symbol_dim;
          int num_value_heads_value = num_value_heads;
          int value_dim_value = value_dim;
          int window_value = static_cast<int>(max_suffix_length);
          float scale_value = scale;
          float dropout_value = dropout_p;
          float inverse_keep_value = inverse_keep_probability;
          float mismatch_scale_value = mismatch_scale;
          float mismatch_unit_value = mismatch_unit;
          int gradient_mask_value = gradient_mask;
          int tile_count_value = tile_count;
          for (int head_group_offset = 0;
               head_group_offset < heads_per_value;
               ++head_group_offset) {
            int head_group_value = head_group_offset;
            void* arguments[] = {
                &query_pointer,
                &key_pointer,
                &value_pointer,
                &grad_output_pointer,
                &packed_query_pointer,
                &packed_key_pointer,
                &seed_pointer,
                &stats_pointer,
                &grad_query_pointer,
                &grad_key_pointer,
                &grad_value_pointer,
                &next_gate_pointer,
                &next_score_pointer,
                &future_pointer,
                &ring_pointer,
                &batch_size_value,
                &seq_len_value,
                &num_heads_value,
                &symbol_dim_value,
                &num_value_heads_value,
                &value_dim_value,
                &window_value,
                &scale_value,
                &dropout_value,
                &inverse_keep_value,
                &mismatch_scale_value,
                &mismatch_unit_value,
                &gradient_mask_value,
                &head_group_value,
                &tile_count_value};
            C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(
                    persistent_wavefront_full_reverse_kernel<scalar_t>),
                dim3(persistent_blocks),
                dim3(kThreads),
                arguments,
                0,
                stream));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }

        const int64_t query_elements =
            (gradient_mask & kGradQuery) != 0 ? query.numel() : 0;
        const int64_t key_elements =
            (gradient_mask & kGradKey) != 0 ? key.numel() : 0;
        const int64_t value_elements =
            (gradient_mask & kGradValue) != 0 ? value.numel() : 0;
        const int64_t total_elements =
            query_elements + key_elements + value_elements;
        constexpr int kFinalizeThreads = 256;
        const int finalize_blocks = static_cast<int>(
            (total_elements + kFinalizeThreads - 1) / kFinalizeThreads);
        finalize_direct_vjp_kernel<scalar_t><<<
            finalize_blocks,
            kFinalizeThreads,
            0,
            stream>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            grad_query.data_ptr<float>(),
            grad_key.data_ptr<float>(),
            grad_value.data_ptr<float>(),
            query_elements,
            key_elements,
            value_elements);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {grad_query, grad_key, grad_value};
}
