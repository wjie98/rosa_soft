#define rosa_soft_streaming_vjp_cuda rosa_soft_flash_tc_baseline_vjp_cuda
#include "../../rosa_soft/csrc/cuda/rosa_soft_streaming_kernels.cu"
#undef rosa_soft_streaming_vjp_cuda

#include <cuda_fp16.h>
#include <mma.h>


namespace {

namespace wmma = nvcuda::wmma;

constexpr int kFlashQueryTile = 32;
constexpr int kLocalWords = kFlashQueryTile + kSuffixTile - 1;
constexpr int kLocalDiagonals = kFlashQueryTile + kRouteTile - 1;
constexpr int kTensorWarps = 4;
constexpr int kGateTensorWarps = 8;
constexpr int kTensorTile = 16;


__device__ __forceinline__ float warp_sum_tc(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}


__device__ __forceinline__ float warp_max_tc(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(
        value,
        __shfl_down_sync(0xffffffffu, value, offset));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}


__device__ __forceinline__ float warp_prefix_product_tc(float value) {
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const float previous = __shfl_up_sync(0xffffffffu, value, offset);
    if ((threadIdx.x & (kWarpSize - 1)) >= offset) {
      value *= previous;
    }
  }
  return value;
}


__device__ __forceinline__ void split_half(
    float value,
    __half& high,
    __half& low) {
  high = __float2half_rn(value);
  low = __float2half_rn(value - __half2float(high));
}


__device__ __forceinline__ float* tensor_storage(
    SharedView<kFlashQueryTile> shared) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(shared.gates);
  return reinterpret_cast<float*>((address + 31u) & ~uintptr_t(31u));
}


__device__ __forceinline__ void compute_raw_score_tile_warp(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    SharedView<kFlashQueryTile> shared,
    int row_start,
    int route_start,
    int row_end,
    int seq_len,
    int symbol_dim,
    int maximum_steps,
    float mismatch_scale,
    float inverse_symbol_dim) {
  const int query_base = row_start - (kSuffixTile - 1);
  const int key_base = route_start - 1 - (kSuffixTile - 1);
  stage_gate_tile<kFlashQueryTile>(
      packed_query,
      packed_key,
      shared,
      query_base,
      key_base,
      seq_len,
      symbol_dim,
      maximum_steps,
      mismatch_scale,
      inverse_symbol_dim);

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  for (int diagonal = warp; diagonal < kLocalDiagonals;
       diagonal += kFlashQueryTile) {
    const int delta = diagonal - (kRouteTile - 1);
    const int first_row = max(delta, 0);
    const int first_route = max(-delta, 0);
    const int diagonal_count = kRouteTile - abs(delta);
    float carry = 0.0f;

    for (int segment = 0; segment < diagonal_count;
         segment += maximum_steps) {
      const int count = min(maximum_steps, diagonal_count - segment);
      const int first_query_index =
          first_row + segment + (kSuffixTile - 1);
      const int history_count =
          segment == 0 ? maximum_steps - 1 : maximum_steps;
      float history_prefix = lane < history_count
          ? shared.gates[
                diagonal * kLocalWords + first_query_index - 1 - lane]
          : 1.0f;
      history_prefix = warp_prefix_product_tc(history_prefix);
      if (segment == 0) {
        carry = warp_sum_tc(
            lane < history_count ? history_prefix : 0.0f);
      }

      const int row_offset = first_row + segment + lane;
      const int route_offset = first_route + segment + lane;
      const float gate = lane < count
          ? shared.gates[
                diagonal * kLocalWords + row_offset + (kSuffixTile - 1)]
          : 1.0f;
      const float current_prefix = warp_prefix_product_tc(gate);
      const int history_lane = max(
          0,
          min(kWarpSize - 1, maximum_steps - lane - 1));
      const float correction_history = __shfl_sync(
          0xffffffffu, history_prefix, history_lane);
      float correction = 0.0f;
      if (lane < count && (segment > 0 || lane > 0)) {
        correction = current_prefix * correction_history;
      }

      float affine_scale = lane < count ? gate : 1.0f;
      float affine_bias = lane < count ? gate - correction : 0.0f;
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float previous_scale =
            __shfl_up_sync(0xffffffffu, affine_scale, offset);
        const float previous_bias =
            __shfl_up_sync(0xffffffffu, affine_bias, offset);
        if (lane >= offset) {
          affine_bias = affine_scale * previous_bias + affine_bias;
          affine_scale *= previous_scale;
        }
      }
      const float score = affine_scale * carry + affine_bias;
      if (lane < count) {
        const int row = row_start + row_offset;
        const int route_position = route_start + route_offset;
        shared.route_workspace[row_offset * kRouteTile + route_offset] =
            row < seq_len && route_position <= row ? score : 0.0f;
      }
      carry = __shfl_sync(0xffffffffu, score, count - 1);
    }
  }
  __syncthreads();
}


__device__ __forceinline__ float compute_raw_score_tile_direct(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    SharedView<kFlashQueryTile> shared,
    int row_start,
    int route_start,
    int row_end,
    int seq_len,
    int symbol_dim,
    int maximum_steps,
    float mismatch_scale,
    float inverse_symbol_dim) {
  const int query_base = row_start - (kSuffixTile - 1);
  const int key_base = route_start - 1 - (kSuffixTile - 1);
  stage_gate_tile<kFlashQueryTile>(
      packed_query,
      packed_key,
      shared,
      query_base,
      key_base,
      seq_len,
      symbol_dim,
      maximum_steps,
      mismatch_scale,
      inverse_symbol_dim);
  const int row_offset = threadIdx.x / kWarpSize;
  const int route_offset = threadIdx.x & (kWarpSize - 1);
  const int row = row_start + row_offset;
  const int route_position = route_start + route_offset;
  const int suffix_steps = row < seq_len && route_position <= row
      ? min(maximum_steps, min(row + 1, route_position))
      : 0;
  const int diagonal = row_offset - route_offset + (kRouteTile - 1);
  float product = 1.0f;
  float score = 0.0f;
  for (int suffix = 0; suffix < suffix_steps; ++suffix) {
    const int query_index =
        row_offset + (kSuffixTile - 1) - suffix;
    product *= shared.gates[diagonal * kLocalWords + query_index];
    score += product;
  }
  return score;
}


__device__ __forceinline__ void tensor_utility_tile(
    SharedView<kFlashQueryTile> shared,
    int value_count) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  float* output = tensor_storage(shared);
  __half* scratch = reinterpret_cast<__half*>(
      output + kFlashQueryTile * kRouteTile);
  constexpr int kHalfWordsPerWarp = 3 * kTensorTile * kTensorTile;

  if (warp < kTensorWarps) {
    const int row_start = (warp / 2) * kTensorTile;
    const int route_start = (warp % 2) * kTensorTile;
    __half* warp_scratch = scratch + warp * kHalfWordsPerWarp;
    __half* left_high = warp_scratch;
    __half* left_low = left_high + kTensorTile * kTensorTile;
    __half* right_sign = left_low + kTensorTile * kTensorTile;
    wmma::fragment<
        wmma::accumulator,
        kTensorTile,
        kTensorTile,
        kTensorTile,
        float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (int value_start = 0; value_start < value_count;
         value_start += kTensorTile) {
      for (int index = lane; index < kTensorTile * kTensorTile;
           index += kWarpSize) {
        const int row = row_start + index / kTensorTile;
        const int value_offset = value_start + index % kTensorTile;
        const float left_value = value_offset < value_count
            ? shared.grad_output[
                  row * TileTraits<kFlashQueryTile>::kValueTile + value_offset]
            : 0.0f;
        split_half(left_value, left_high[index], left_low[index]);

        const int route = route_start + index / kTensorTile;
        const int right_value_offset = value_start + index % kTensorTile;
        const float sign = right_value_offset < value_count
            ? shared.value_signs[
                  right_value_offset * kValueSharedStride + route]
            : 0.0f;
        right_sign[index] = __float2half_rn(sign);
      }
      __syncwarp();
      using left_fragment_t = wmma::fragment<
          wmma::matrix_a,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          __half,
          wmma::row_major>;
      using right_fragment_t = wmma::fragment<
          wmma::matrix_b,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          __half,
          wmma::col_major>;
      left_fragment_t high_fragment;
      left_fragment_t low_fragment;
      right_fragment_t sign_fragment;
      wmma::load_matrix_sync(high_fragment, left_high, kTensorTile);
      wmma::load_matrix_sync(low_fragment, left_low, kTensorTile);
      wmma::load_matrix_sync(sign_fragment, right_sign, kTensorTile);
      wmma::mma_sync(
          accumulator, high_fragment, sign_fragment, accumulator);
      wmma::mma_sync(
          accumulator, low_fragment, sign_fragment, accumulator);
      __syncwarp();
    }
    wmma::store_matrix_sync(
        output + row_start * kRouteTile + route_start,
        accumulator,
        kRouteTile,
        wmma::mem_row_major);
  }
  __syncthreads();
}


__device__ __forceinline__ void tensor_value_gradient_tile(
    SharedView<kFlashQueryTile> shared,
    int value_count) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  float* output = tensor_storage(shared);
  __half* scratch = reinterpret_cast<__half*>(
      output + kRouteTile * kFlashQueryTile);
  constexpr int kHalfWordsPerWarp = 4 * kTensorTile * kTensorTile;

  if (warp < kTensorWarps) {
    const int route_start = (warp / 2) * kTensorTile;
    const int value_start = (warp % 2) * kTensorTile;
    __half* warp_scratch = scratch + warp * kHalfWordsPerWarp;
    __half* left_high = warp_scratch;
    __half* left_low = left_high + kTensorTile * kTensorTile;
    __half* right_high = left_low + kTensorTile * kTensorTile;
    __half* right_low = right_high + kTensorTile * kTensorTile;
    wmma::fragment<
        wmma::accumulator,
        kTensorTile,
        kTensorTile,
        kTensorTile,
        float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (int row_start = 0; row_start < kFlashQueryTile;
         row_start += kTensorTile) {
      for (int index = lane; index < kTensorTile * kTensorTile;
           index += kWarpSize) {
        const int route = route_start + index / kTensorTile;
        const int row = row_start + index % kTensorTile;
        const float probability =
            shared.route_workspace[row * kRouteTile + route];
        split_half(probability, left_high[index], left_low[index]);

        const int column = value_start + index / kTensorTile;
        const int right_row = row_start + index % kTensorTile;
        const float gradient = column < value_count
            ? shared.grad_output[
                  right_row * TileTraits<kFlashQueryTile>::kValueTile + column]
            : 0.0f;
        split_half(gradient, right_high[index], right_low[index]);
      }
      __syncwarp();
      using left_fragment_t = wmma::fragment<
          wmma::matrix_a,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          __half,
          wmma::row_major>;
      using right_fragment_t = wmma::fragment<
          wmma::matrix_b,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          __half,
          wmma::col_major>;
      left_fragment_t left_hi_fragment;
      left_fragment_t left_lo_fragment;
      right_fragment_t right_hi_fragment;
      right_fragment_t right_lo_fragment;
      wmma::load_matrix_sync(left_hi_fragment, left_high, kTensorTile);
      wmma::load_matrix_sync(left_lo_fragment, left_low, kTensorTile);
      wmma::load_matrix_sync(right_hi_fragment, right_high, kTensorTile);
      wmma::load_matrix_sync(right_lo_fragment, right_low, kTensorTile);
      wmma::mma_sync(
          accumulator,
          left_hi_fragment,
          right_hi_fragment,
          accumulator);
      wmma::mma_sync(
          accumulator,
          left_hi_fragment,
          right_lo_fragment,
          accumulator);
      wmma::mma_sync(
          accumulator,
          left_lo_fragment,
          right_hi_fragment,
          accumulator);
      wmma::mma_sync(
          accumulator,
          left_lo_fragment,
          right_lo_fragment,
          accumulator);
      __syncwarp();
    }
    wmma::store_matrix_sync(
        output + route_start * kRouteTile + value_start,
        accumulator,
        kRouteTile,
        wmma::mem_row_major);
  }
  __syncthreads();
}


template <bool Transpose>
__device__ __forceinline__ void tensor_gate_credit(
    SharedView<kFlashQueryTile> shared,
    float* __restrict__ target,
    int batch,
    int head,
    int query_base,
    int key_base,
    int seq_len,
    int num_heads,
    int symbol_dim) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  constexpr int kHalfWordsPerWarp = 3 * kTensorTile * kTensorTile;
  __half* scratch = reinterpret_cast<__half*>(tensor_storage(shared));
  const int column_tiles = (symbol_dim + kTensorTile - 1) / kTensorTile;
  const int output_tiles = 4 * column_tiles;

  if (warp < kGateTensorWarps) {
    __half* warp_scratch = scratch + warp * kHalfWordsPerWarp;
    __half* left_high = warp_scratch;
    __half* left_low = left_high + kTensorTile * kTensorTile;
    __half* right_sign = left_low + kTensorTile * kTensorTile;
    float* output_tile = reinterpret_cast<float*>(warp_scratch);

    for (int tile = warp; tile < output_tiles; tile += kGateTensorWarps) {
      const int row_start = (tile / column_tiles) * kTensorTile;
      const int bit_start = (tile % column_tiles) * kTensorTile;
      wmma::fragment<
          wmma::accumulator,
          kTensorTile,
          kTensorTile,
          kTensorTile,
          float>
          accumulator;
      wmma::fill_fragment(accumulator, 0.0f);

      for (int reduction_start = 0; reduction_start < 64;
           reduction_start += kTensorTile) {
        float lane_maximum = 0.0f;
        bool requires_scaling = false;
        for (int index = lane; index < kTensorTile * kTensorTile;
             index += kWarpSize) {
          const int output_row = row_start + index / kTensorTile;
          const int reduction = reduction_start + index % kTensorTile;
          const int query_index = Transpose ? reduction : output_row;
          const int key_index = Transpose ? output_row : reduction;
          float local_scale = 0.0f;
          if (query_index < kLocalWords && key_index < kLocalWords) {
            const int diagonal =
                query_index - key_index + (kRouteTile - 1);
            if (diagonal >= 0 && diagonal < kLocalDiagonals) {
              local_scale = 0.5f * shared.route_workspace[
                  diagonal * kLocalWords + query_index];
            }
          }
          lane_maximum = fmaxf(lane_maximum, fabsf(local_scale));
          requires_scaling =
              requires_scaling || fabsf(local_scale) > 60000.0f;
          split_half(local_scale, left_high[index], left_low[index]);

          const int bit = bit_start + index / kTensorTile;
          const int sign_index = reduction_start + index % kTensorTile;
          int sign = 0;
          if (bit < symbol_dim && sign_index < kLocalWords) {
            const uint32_t word = static_cast<uint32_t>(
                Transpose
                ? shared.query_words[sign_index]
                : shared.key_words[sign_index]);
            sign = sign_from_bit(word, bit);
          }
          right_sign[index] = __int2half_rn(sign);
        }
        requires_scaling = __any_sync(0xffffffffu, requires_scaling);
        float tile_scale = 1.0f;
        if (requires_scaling) {
          tile_scale = warp_max_tc(lane_maximum);
          const float inverse_scale = 1.0f / tile_scale;
          for (int index = lane; index < kTensorTile * kTensorTile;
               index += kWarpSize) {
            const int output_row = row_start + index / kTensorTile;
            const int reduction = reduction_start + index % kTensorTile;
            const int query_index = Transpose ? reduction : output_row;
            const int key_index = Transpose ? output_row : reduction;
            float local_scale = 0.0f;
            if (query_index < kLocalWords && key_index < kLocalWords) {
              const int diagonal =
                  query_index - key_index + (kRouteTile - 1);
              if (diagonal >= 0 && diagonal < kLocalDiagonals) {
                local_scale = 0.5f * shared.route_workspace[
                    diagonal * kLocalWords + query_index];
              }
            }
            split_half(
                local_scale * inverse_scale,
                left_high[index],
                left_low[index]);
          }
        }
        __syncwarp();
        using left_fragment_t = wmma::fragment<
            wmma::matrix_a,
            kTensorTile,
            kTensorTile,
            kTensorTile,
            __half,
            wmma::row_major>;
        using right_fragment_t = wmma::fragment<
            wmma::matrix_b,
            kTensorTile,
            kTensorTile,
            kTensorTile,
            __half,
            wmma::col_major>;
        left_fragment_t high_fragment;
        left_fragment_t low_fragment;
        right_fragment_t sign_fragment;
        wmma::load_matrix_sync(high_fragment, left_high, kTensorTile);
        wmma::load_matrix_sync(low_fragment, left_low, kTensorTile);
        wmma::load_matrix_sync(sign_fragment, right_sign, kTensorTile);
        if (requires_scaling) {
          wmma::fragment<
              wmma::accumulator,
              kTensorTile,
              kTensorTile,
              kTensorTile,
              float>
              partial;
          wmma::fill_fragment(partial, 0.0f);
          wmma::mma_sync(
              partial, high_fragment, sign_fragment, partial);
          wmma::mma_sync(
              partial, low_fragment, sign_fragment, partial);
#pragma unroll
          for (int index = 0; index < accumulator.num_elements; ++index) {
            accumulator.x[index] += tile_scale * partial.x[index];
          }
        } else {
          wmma::mma_sync(
              accumulator, high_fragment, sign_fragment, accumulator);
          wmma::mma_sync(
              accumulator, low_fragment, sign_fragment, accumulator);
        }
        __syncwarp();
      }
      wmma::store_matrix_sync(
          output_tile, accumulator, kTensorTile, wmma::mem_row_major);
      __syncwarp();
      for (int index = lane; index < kTensorTile * kTensorTile;
           index += kWarpSize) {
        const int local_position = row_start + index / kTensorTile;
        const int bit = bit_start + index % kTensorTile;
        if (local_position < kLocalWords && bit < symbol_dim) {
          const int position =
              (Transpose ? key_base : query_base) + local_position;
          const float contribution = output_tile[index];
          if (position >= 0 && position < seq_len && contribution != 0.0f) {
            const int64_t target_index = Transpose
                ? ((static_cast<int64_t>(batch) * num_heads + head) *
                       symbol_dim +
                   bit) *
                          seq_len +
                      position
                : ((static_cast<int64_t>(batch) * seq_len + position) *
                       num_heads +
                   head) *
                          symbol_dim +
                      bit;
            atomicAdd(&target[target_index], contribution);
          }
        }
      }
      __syncwarp();
    }
  }
  __syncthreads();
}


__device__ __forceinline__ void direct_gate_credit(
    SharedView<kFlashQueryTile> shared,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    int batch,
    int head,
    int query_base,
    int key_base,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int suffix_count,
    int gradient_mask) {
  if ((gradient_mask & kGradQuery) != 0) {
    const int query_items = suffix_count * kFlashQueryTile * symbol_dim;
    for (int index = threadIdx.x; index < query_items; index += blockDim.x) {
      const int suffix = index / (kFlashQueryTile * symbol_dim);
      const int within_suffix =
          index - suffix * kFlashQueryTile * symbol_dim;
      const int row_offset = within_suffix / symbol_dim;
      const int bit = within_suffix - row_offset * symbol_dim;
      const int query_index = row_offset + kSuffixTile - 1 - suffix;
      const int query_position = query_base + query_index;
      if (query_position >= 0 && query_position < seq_len) {
        float contribution = 0.0f;
        for (int route_offset = 0; route_offset < kRouteTile; ++route_offset) {
          const float local_scale = shared.route_workspace[
              (suffix * kFlashQueryTile + row_offset) * kRouteTile +
              route_offset];
          const int key_index = route_offset + kSuffixTile - 1 - suffix;
          contribution += 0.5f * local_scale * static_cast<float>(
              sign_from_bit(
                  static_cast<uint32_t>(shared.key_words[key_index]), bit));
        }
        if (contribution != 0.0f) {
          const int64_t target =
              ((static_cast<int64_t>(batch) * seq_len + query_position) *
                   num_heads +
               head) *
                  symbol_dim +
              bit;
          atomicAdd(&grad_query[target], contribution);
        }
      }
    }
  }
  if ((gradient_mask & kGradKey) != 0) {
    const int key_items = suffix_count * kRouteTile * symbol_dim;
    for (int index = threadIdx.x; index < key_items; index += blockDim.x) {
      const int suffix = index / (kRouteTile * symbol_dim);
      const int within_suffix = index - suffix * kRouteTile * symbol_dim;
      const int route_offset = within_suffix / symbol_dim;
      const int bit = within_suffix - route_offset * symbol_dim;
      const int key_index = route_offset + kSuffixTile - 1 - suffix;
      const int key_position = key_base + key_index;
      if (key_position >= 0 && key_position < seq_len) {
        float contribution = 0.0f;
        for (int row_offset = 0; row_offset < kFlashQueryTile; ++row_offset) {
          const float local_scale = shared.route_workspace[
              (suffix * kFlashQueryTile + row_offset) * kRouteTile +
              route_offset];
          const int query_index = row_offset + kSuffixTile - 1 - suffix;
          contribution += 0.5f * local_scale * static_cast<float>(
              sign_from_bit(
                  static_cast<uint32_t>(shared.query_words[query_index]), bit));
        }
        if (contribution != 0.0f) {
          const int64_t target =
              ((static_cast<int64_t>(batch) * num_heads + head) * symbol_dim +
               bit) *
                  seq_len +
              key_position;
          atomicAdd(&grad_key[target], contribution);
        }
      }
    }
  }
  __syncthreads();
}


__device__ __forceinline__ void scalar_aggregated_gate_credit(
    SharedView<kFlashQueryTile> shared,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    int batch,
    int head,
    int query_base,
    int key_base,
    int seq_len,
    int num_heads,
    int symbol_dim,
    int suffix_count,
    int gradient_mask) {
  if ((gradient_mask & kGradQuery) != 0) {
    const int first_query_index = kSuffixTile - suffix_count;
    const int active_query_words = kFlashQueryTile + suffix_count - 1;
    const int query_items = active_query_words * symbol_dim;
    for (int index = threadIdx.x; index < query_items; index += blockDim.x) {
      const int local_query_index = index / symbol_dim;
      const int bit = index - local_query_index * symbol_dim;
      const int query_index = first_query_index + local_query_index;
      const int query_position = query_base + query_index;
      if (query_position >= 0 && query_position < seq_len) {
        float contribution = 0.0f;
        const int first_row_offset =
            max(0, query_index - (kSuffixTile - 1));
        const int last_row_offset = min(
            kFlashQueryTile - 1,
            query_index - (kSuffixTile - 1) + suffix_count - 1);
        for (int diagonal = first_row_offset;
             diagonal <= last_row_offset + (kRouteTile - 1);
             ++diagonal) {
          const int key_index =
              query_index - (diagonal - (kRouteTile - 1));
          if (key_index >= 0 && key_index < kLocalWords) {
            contribution += 0.5f * shared.route_workspace[
                diagonal * kLocalWords + query_index] *
                static_cast<float>(sign_from_bit(
                    static_cast<uint32_t>(shared.key_words[key_index]), bit));
          }
        }
        if (contribution != 0.0f) {
          const int64_t target =
              ((static_cast<int64_t>(batch) * seq_len + query_position) *
                   num_heads +
               head) *
                  symbol_dim +
              bit;
          atomicAdd(&grad_query[target], contribution);
        }
      }
    }
  }
  if ((gradient_mask & kGradKey) != 0) {
    const int first_key_index = kSuffixTile - suffix_count;
    const int active_key_words = kRouteTile + suffix_count - 1;
    const int key_items = active_key_words * symbol_dim;
    for (int index = threadIdx.x; index < key_items; index += blockDim.x) {
      const int local_key_index = index / symbol_dim;
      const int bit = index - local_key_index * symbol_dim;
      const int key_index = first_key_index + local_key_index;
      const int key_position = key_base + key_index;
      if (key_position >= 0 && key_position < seq_len) {
        float contribution = 0.0f;
        const int first_route_offset =
            max(0, key_index - (kSuffixTile - 1));
        const int last_route_offset = min(
            kRouteTile - 1,
            key_index - (kSuffixTile - 1) + suffix_count - 1);
        const int first_diagonal =
            (kRouteTile - 1) - last_route_offset;
        const int last_diagonal =
            kFlashQueryTile - 1 + (kRouteTile - 1) - first_route_offset;
        for (int diagonal = first_diagonal;
             diagonal <= last_diagonal;
             ++diagonal) {
          const int query_index =
              key_index + diagonal - (kRouteTile - 1);
          if (query_index >= 0 && query_index < kLocalWords) {
            contribution += 0.5f * shared.route_workspace[
                diagonal * kLocalWords + query_index] *
                static_cast<float>(sign_from_bit(
                    static_cast<uint32_t>(shared.query_words[query_index]),
                    bit));
          }
        }
        if (contribution != 0.0f) {
          const int64_t target =
              ((static_cast<int64_t>(batch) * num_heads + head) * symbol_dim +
               bit) *
                  seq_len +
              key_position;
          atomicAdd(&grad_key[target], contribution);
        }
      }
    }
  }
  __syncthreads();
}


template <
    typename scalar_t,
    bool UseTensorUtility,
    bool UseTensorValue,
    bool UseTensorGate,
    bool UseWarpSuffix>
__global__ __launch_bounds__(kFlashQueryTile * kWarpSize, 1)
void tiled_flash_tc_vjp_kernel(
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
    int max_suffix_length,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_scale,
    const int64_t* __restrict__ dropout_seed,
    int gradient_mask) {
  constexpr int kValueTile = TileTraits<kFlashQueryTile>::kValueTile;
  constexpr int kGateCount = kLocalDiagonals * kLocalWords;
  extern __shared__ int32_t shared_storage[];
  const SharedView<kFlashQueryTile> shared =
      bind_shared<kFlashQueryTile>(shared_storage);
  const int tiles_per_head =
      (seq_len + kFlashQueryTile - 1) / kFlashQueryTile;
  const int tile_index = blockIdx.x % tiles_per_head;
  const int head = (blockIdx.x / tiles_per_head) % num_heads;
  const int batch = blockIdx.x / (tiles_per_head * num_heads);
  const int row_start = tile_index * kFlashQueryTile;
  const int row_end = min(seq_len - 1, row_start + kFlashQueryTile - 1);
  const int row_offset = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int row = row_start + row_offset;
  const int value_head = head / (num_heads / num_value_heads);
  const int32_t* packed_query =
      packed_query_symbols +
      (static_cast<int64_t>(batch) * num_heads + head) * seq_len;
  const int32_t* packed_key =
      packed_key_symbols +
      (static_cast<int64_t>(batch) * num_heads + head) * seq_len;
  const float inverse_symbol_dim = 1.0f / static_cast<float>(symbol_dim);
  const bool needs_qk = (gradient_mask & (kGradQuery | kGradKey)) != 0;
  const bool needs_value = (gradient_mask & kGradValue) != 0;

  SoftmaxStats stats = row < seq_len && lane == 0
      ? SoftmaxStats{kNullScore * scale, 1.0f, 0.0f}
      : SoftmaxStats{-FLT_MAX, 0.0f, 0.0f};

  for (int route_start = 1; route_start <= row_end;
       route_start += kRouteTile) {
    const int maximum_steps = tile_suffix_steps<kFlashQueryTile>(
        row_end, route_start, max_suffix_length);
    float raw_score;
    if constexpr (UseWarpSuffix) {
      compute_raw_score_tile_warp(
          packed_query,
          packed_key,
          shared,
          row_start,
          route_start,
          row_end,
          seq_len,
          symbol_dim,
          maximum_steps,
          mismatch_scale,
          inverse_symbol_dim);
      raw_score = shared.route_workspace[row_offset * kRouteTile + lane];
    } else {
      raw_score = compute_raw_score_tile_direct(
          packed_query,
          packed_key,
          shared,
          row_start,
          route_start,
          row_end,
          seq_len,
          symbol_dim,
          maximum_steps,
          mismatch_scale,
          inverse_symbol_dim);
    }

    float utility = 0.0f;
    if (needs_qk) {
      for (int value_start = 0; value_start < value_dim;
           value_start += kValueTile) {
        const int value_count = min(kValueTile, value_dim - value_start);
        stage_value_tile<scalar_t, kFlashQueryTile>(
            value,
            grad_output,
            shared,
            batch,
            head,
            value_head,
            row_start,
            route_start,
            value_start,
            seq_len,
            num_heads,
            num_value_heads,
            value_dim,
            value_count,
            true);
        if constexpr (UseTensorUtility) {
          tensor_utility_tile(shared, value_count);
          utility += tensor_storage(shared)[
              row_offset * kRouteTile + lane];
        } else if (row < seq_len && route_start + lane <= row) {
          for (int value_offset = 0; value_offset < value_count;
               ++value_offset) {
            utility +=
                shared.grad_output[row_offset * kValueTile + value_offset] *
                shared.value_signs[
                    value_offset * kValueSharedStride + lane];
          }
        }
      }
    }
    const int route_position = route_start + lane;
    if (row < seq_len && route_position <= row) {
      const ScoreTransform transformed = transform_score(raw_score);
      const float dropout_scale = attention_dropout_scale(
          dropout_seed,
          dropout_p,
          inverse_keep_probability,
          batch,
          head,
          row,
          route_position);
      stats = append_item(
          stats,
          transformed.route_score * scale - logf(static_cast<float>(row)),
          dropout_scale * utility);
    }
  }

  const SoftmaxStats reduced = warp_reduce_stats(stats);
  if (lane == 0 && row < seq_len) {
    shared.row_maximum[row_offset] = reduced.maximum;
    shared.row_normalizer[row_offset] = reduced.normalizer;
    shared.row_expected_utility[row_offset] =
        reduced.utility_numerator / reduced.normalizer;
  }
  __syncthreads();

  for (int route_start = 1; route_start <= row_end;
       route_start += kRouteTile) {
    const int maximum_steps = tile_suffix_steps<kFlashQueryTile>(
        row_end, route_start, max_suffix_length);
    float raw_score;
    if constexpr (UseWarpSuffix) {
      compute_raw_score_tile_warp(
          packed_query,
          packed_key,
          shared,
          row_start,
          route_start,
          row_end,
          seq_len,
          symbol_dim,
          maximum_steps,
          mismatch_scale,
          inverse_symbol_dim);
      raw_score = shared.route_workspace[row_offset * kRouteTile + lane];
    } else {
      raw_score = compute_raw_score_tile_direct(
          packed_query,
          packed_key,
          shared,
          row_start,
          route_start,
          row_end,
          seq_len,
          symbol_dim,
          maximum_steps,
          mismatch_scale,
          inverse_symbol_dim);
    }

    const int route_position = route_start + lane;
    float probability = 0.0f;
    float dropout_scale = 0.0f;
    if (row < seq_len && route_position <= row) {
      const ScoreTransform transformed = transform_score(raw_score);
      probability = __expf(
          transformed.route_score * scale - logf(static_cast<float>(row)) -
          shared.row_maximum[row_offset]) /
          shared.row_normalizer[row_offset];
      dropout_scale = attention_dropout_scale(
          dropout_seed,
          dropout_p,
          inverse_keep_probability,
          batch,
          head,
          row,
          route_position);
    }
    shared.route_workspace[row_offset * kRouteTile + lane] =
        probability * dropout_scale;

    float utility = 0.0f;
    if (needs_qk || needs_value) {
      for (int value_start = 0; value_start < value_dim;
           value_start += kValueTile) {
        const int value_count = min(kValueTile, value_dim - value_start);
        stage_value_tile<scalar_t, kFlashQueryTile>(
            value,
            grad_output,
            shared,
            batch,
            head,
            value_head,
            row_start,
            route_start,
            value_start,
            seq_len,
            num_heads,
            num_value_heads,
            value_dim,
            value_count,
            needs_qk);
        if (needs_qk) {
          if constexpr (UseTensorUtility) {
            tensor_utility_tile(shared, value_count);
            utility += tensor_storage(shared)[
                row_offset * kRouteTile + lane];
          } else if (row < seq_len && route_position <= row) {
            for (int value_offset = 0; value_offset < value_count;
                 ++value_offset) {
              utility +=
                  shared.grad_output[row_offset * kValueTile + value_offset] *
                  shared.value_signs[
                      value_offset * kValueSharedStride + lane];
            }
          }
        }
        if constexpr (UseTensorUtility && UseTensorValue) {
          if (needs_qk && needs_value) {
            __syncthreads();
          }
        }
        if (needs_value) {
          if constexpr (UseTensorValue) {
            tensor_value_gradient_tile(shared, value_count);
            const int item_count = kRouteTile * value_count;
            for (int index = threadIdx.x; index < item_count;
                 index += blockDim.x) {
              const int route_offset = index / value_count;
              const int value_offset = index - route_offset * value_count;
              const int target_position = route_start + route_offset;
              if (target_position <= row_end) {
                const int64_t target =
                    ((static_cast<int64_t>(batch) * seq_len + target_position) *
                         num_value_heads +
                     value_head) *
                        value_dim +
                    value_start + value_offset;
                atomicAdd(
                    &grad_value[target],
                    tensor_storage(shared)[
                        route_offset * kRouteTile + value_offset]);
              }
            }
          } else {
            const int item_count = kRouteTile * value_count;
            for (int index = threadIdx.x; index < item_count;
                 index += blockDim.x) {
              const int route_offset = index / value_count;
              const int value_offset = index - route_offset * value_count;
              const int target_position = route_start + route_offset;
              if (target_position <= row_end) {
                float contribution = 0.0f;
                for (int source_row = 0; source_row < kFlashQueryTile;
                     ++source_row) {
                  contribution += shared.route_workspace[
                                      source_row * kRouteTile + route_offset] *
                      shared.grad_output[
                          source_row * kValueTile + value_offset];
                }
                const int64_t target =
                    ((static_cast<int64_t>(batch) * seq_len + target_position) *
                         num_value_heads +
                     value_head) *
                        value_dim +
                    value_start + value_offset;
                atomicAdd(&grad_value[target], contribution);
              }
            }
          }
        }
      }
    }

    // Every route/value consumer must finish before Q/K reuses either the
    // probability workspace or the Tensor-Core output scratch.
    __syncthreads();
    if (needs_qk) {
      float raw_score_vjp = 0.0f;
      if (row < seq_len && route_position <= row) {
        const ScoreTransform transformed = transform_score(raw_score);
        raw_score_vjp =
            scale * probability *
            (dropout_scale * utility -
             shared.row_expected_utility[row_offset]) *
            transformed.raw_vjp_multiplier;
      }
      const int query_base = row_start - (kSuffixTile - 1);
      const int key_base = route_start - 1 - (kSuffixTile - 1);
      const bool use_direct_suffix_vjp =
          maximum_steps == 1 ||
          (maximum_steps <= 4 && maximum_steps * symbol_dim <= 16);
      if (!use_direct_suffix_vjp) {
        for (int index = threadIdx.x; index < kGateCount;
             index += blockDim.x) {
          shared.route_workspace[index] = 0.0f;
        }
      }
      stage_gate_tile<kFlashQueryTile>(
          packed_query,
          packed_key,
          shared,
          query_base,
          key_base,
          seq_len,
          symbol_dim,
          maximum_steps,
          mismatch_scale,
          inverse_symbol_dim);

      float prefix = 1.0f;
      float tail = raw_score;
      const int suffix_steps = row < seq_len && route_position <= row
          ? min(maximum_steps, min(row + 1, route_position))
          : 0;
      const int diagonal = row_offset - lane + (kRouteTile - 1);
      for (int suffix = 0; suffix < maximum_steps; ++suffix) {
        const int query_index =
            row_offset + (kSuffixTile - 1) - suffix;
        float local_scale = 0.0f;
        if (suffix < suffix_steps) {
          prefix *= shared.gates[diagonal * kLocalWords + query_index];
          local_scale = mismatch_scale * inverse_symbol_dim * raw_score_vjp *
              fmaxf(tail, 0.0f);
          tail -= prefix;
        }
        if (use_direct_suffix_vjp) {
          shared.route_workspace[
              (suffix * kFlashQueryTile + row_offset) * kRouteTile + lane] =
              local_scale;
        } else if (local_scale != 0.0f) {
          atomicAdd(
              &shared.route_workspace[
                  diagonal * kLocalWords + query_index],
              local_scale);
        }
      }
      __syncthreads();

      if (use_direct_suffix_vjp) {
        direct_gate_credit(
            shared,
            grad_query,
            grad_key,
            batch,
            head,
            query_base,
            key_base,
            seq_len,
            num_heads,
            symbol_dim,
            maximum_steps,
            gradient_mask);
      } else if constexpr (UseTensorGate) {
        if (symbol_dim == 32) {
          if ((gradient_mask & kGradQuery) != 0) {
            tensor_gate_credit<false>(
                shared,
                grad_query,
                batch,
                head,
                query_base,
                key_base,
                seq_len,
                num_heads,
                symbol_dim);
          }
          if ((gradient_mask & kGradKey) != 0) {
            tensor_gate_credit<true>(
                shared,
                grad_key,
                batch,
                head,
                query_base,
                key_base,
                seq_len,
                num_heads,
                symbol_dim);
          }
        } else {
          scalar_aggregated_gate_credit(
              shared,
              grad_query,
              grad_key,
              batch,
              head,
              query_base,
              key_base,
              seq_len,
              num_heads,
              symbol_dim,
              maximum_steps,
              gradient_mask);
        }
      } else {
        scalar_aggregated_gate_credit(
            shared,
            grad_query,
            grad_key,
            batch,
            head,
            query_base,
            key_base,
            seq_len,
            num_heads,
            symbol_dim,
            maximum_steps,
            gradient_mask);
      }
    }
  }
}


template <
    typename scalar_t,
    bool UseTensorUtility,
    bool UseTensorValue,
    bool UseTensorGate,
    bool UseWarpSuffix>
void launch_flash_tc_kernel(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    torch::Tensor& grad_query,
    torch::Tensor& grad_key_accumulator,
    torch::Tensor& grad_value,
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
    const torch::Tensor& dropout_seed,
    int gradient_mask,
    cudaStream_t stream) {
  const int tiles_per_head =
      (seq_len + kFlashQueryTile - 1) / kFlashQueryTile;
  const int64_t block_count64 =
      static_cast<int64_t>(batch_size) * num_heads * tiles_per_head;
  TORCH_CHECK(
      block_count64 <= std::numeric_limits<int>::max(),
      "FlashROSA-TC grid is too large");
  const size_t shared_bytes =
      shared_word_count<kFlashQueryTile>() * sizeof(int32_t);
  tiled_flash_tc_vjp_kernel<
      scalar_t,
      UseTensorUtility,
      UseTensorValue,
      UseTensorGate,
      UseWarpSuffix><<<
      static_cast<int>(block_count64),
      kFlashQueryTile * kWarpSize,
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
      max_suffix_length,
      scale,
      dropout_p,
      inverse_keep_probability,
      mismatch_scale,
      dropout_seed.data_ptr<int64_t>(),
      gradient_mask);
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_flash_tc_vjp_cuda(
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
int plan) {
  const bool tensor_gate_shape =
      query.size(3) == 32 && max_suffix_length >= 31 &&
      max_suffix_length <= 32 &&
      (gradient_mask & (kGradQuery | kGradKey)) != 0;
  if (plan == 0 || max_suffix_length > 32 ||
      (plan == 4 && !tensor_gate_shape)) {
    return rosa_soft_flash_tc_baseline_vjp_cuda(
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
  const c10::cuda::CUDAGuard device_guard(query.device());
  if (plan >= 2) {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major >= 8,
        "FlashROSA Tensor-Core plans require compute capability 8.0 or newer");
  }
  const int batch_size = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int symbol_dim = static_cast<int>(query.size(3));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  const auto float_options = query.options().dtype(torch::kFloat32);
  torch::Tensor grad_query = (gradient_mask & kGradQuery) != 0
      ? torch::zeros(query.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_key = (gradient_mask & kGradKey) != 0
      ? torch::empty(key.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_key_accumulator = (gradient_mask & kGradKey) != 0
      ? torch::zeros(
            {batch_size, num_heads, symbol_dim, seq_len}, float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_value = (gradient_mask & kGradValue) != 0
      ? torch::zeros(value.sizes(), float_options)
      : torch::empty({0}, float_options);
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_flash_tc_vjp_cuda",
      [&] {
        if (plan == 1) {
          launch_flash_tc_kernel<scalar_t, false, false, false, true>(
              value,
              grad_output,
              packed_query_symbols,
              packed_key_symbols,
              grad_query,
              grad_key_accumulator,
              grad_value,
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
              dropout_seed,
              gradient_mask,
              stream);
        } else if (plan == 2) {
          launch_flash_tc_kernel<scalar_t, true, false, false, false>(
              value,
              grad_output,
              packed_query_symbols,
              packed_key_symbols,
              grad_query,
              grad_key_accumulator,
              grad_value,
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
              dropout_seed,
              gradient_mask,
              stream);
        } else if (plan == 3) {
          launch_flash_tc_kernel<scalar_t, false, true, false, false>(
              value,
              grad_output,
              packed_query_symbols,
              packed_key_symbols,
              grad_query,
              grad_key_accumulator,
              grad_value,
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
              dropout_seed,
              gradient_mask,
              stream);
        } else if (plan == 4) {
          launch_flash_tc_kernel<scalar_t, false, false, true, false>(
              value,
              grad_output,
              packed_query_symbols,
              packed_key_symbols,
              grad_query,
              grad_key_accumulator,
              grad_value,
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
              dropout_seed,
              gradient_mask,
              stream);
        } else if (plan == 5) {
          launch_flash_tc_kernel<scalar_t, true, true, true, false>(
              value,
              grad_output,
              packed_query_symbols,
              packed_key_symbols,
              grad_query,
              grad_key_accumulator,
              grad_value,
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
              dropout_seed,
              gradient_mask,
              stream);
        } else {
          launch_flash_tc_kernel<scalar_t, true, true, true, true>(
              value,
              grad_output,
              packed_query_symbols,
              packed_key_symbols,
              grad_query,
              grad_key_accumulator,
              grad_value,
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
              dropout_seed,
              gradient_mask,
              stream);
        }

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
