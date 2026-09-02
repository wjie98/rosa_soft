#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>


namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWarpSize = 32;
constexpr int kWmmaTile = 16;


__device__ __forceinline__ uint32_t symbol_mask(int symbol_bits) {
  return symbol_bits == 32 ? 0xffffffffu : (1u << symbol_bits) - 1u;
}


template <typename element_t>
__device__ __forceinline__ element_t gate_element(int value);


template <>
__device__ __forceinline__ __half gate_element<__half>(int value) {
  return __int2half_rn(value);
}


template <>
__device__ __forceinline__ signed char gate_element<signed char>(int value) {
  return static_cast<signed char>(value);
}


__global__ void gate_scalar_kernel(
    const int32_t* __restrict__ query_codes,
    const int32_t* __restrict__ key_codes,
    float* __restrict__ output,
    int query_count,
    int key_count,
    int symbol_bits,
    float mismatch_scale) {
  const int key_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int query_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (query_index >= query_count || key_index >= key_count) {
    return;
  }
  const uint32_t mask = symbol_mask(symbol_bits);
  const uint32_t mismatch =
      (static_cast<uint32_t>(query_codes[query_index]) ^
       static_cast<uint32_t>(key_codes[key_index])) &
      mask;
  output[static_cast<int64_t>(query_index) * key_count + key_index] =
      __expf(
          -mismatch_scale * static_cast<float>(__popc(mismatch)) /
          static_cast<float>(symbol_bits));
}


template <typename element_t, bool Integer>
__global__ void gate_wmma_kernel(
    const int32_t* __restrict__ query_codes,
    const int32_t* __restrict__ key_codes,
    float* __restrict__ output,
    int query_count,
    int key_count,
    int symbol_bits,
    float mismatch_scale) {
  __shared__ __align__(16) element_t query_signs[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) element_t key_signs[kWmmaTile * kWmmaTile];
  using accumulator_t = typename std::conditional<Integer, int, float>::type;
  __shared__ __align__(16) accumulator_t dots[kWmmaTile * kWmmaTile];

  const int lane = threadIdx.x;
  const int query_start = blockIdx.y * kWmmaTile;
  const int key_start = blockIdx.x * kWmmaTile;

  wmma::fragment<
      wmma::accumulator,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      accumulator_t>
      accumulator;
  wmma::fill_fragment(accumulator, accumulator_t(0));

  for (int bit_start = 0; bit_start < symbol_bits; bit_start += kWmmaTile) {
    for (int index = lane; index < kWmmaTile * kWmmaTile;
         index += kWarpSize) {
      const int row = index / kWmmaTile;
      const int bit = bit_start + index % kWmmaTile;
      const int query_index = query_start + row;
      int sign = 0;
      if (query_index < query_count && bit < symbol_bits) {
        sign =
            ((static_cast<uint32_t>(query_codes[query_index]) >> bit) & 1u)
            ? 1
            : -1;
      }
      query_signs[index] = gate_element<element_t>(sign);

      const int column = index / kWmmaTile;
      const int key_index = key_start + column;
      sign = 0;
      if (key_index < key_count && bit < symbol_bits) {
        sign =
            ((static_cast<uint32_t>(key_codes[key_index]) >> bit) & 1u)
            ? 1
            : -1;
      }
      // A col-major B fragment sees each logical key as one matrix column.
      key_signs[index] = gate_element<element_t>(sign);
    }
    __syncwarp();

    wmma::fragment<
        wmma::matrix_a,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        element_t,
        wmma::row_major>
        query_fragment;
    wmma::fragment<
        wmma::matrix_b,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        element_t,
        wmma::col_major>
        key_fragment;
    wmma::load_matrix_sync(query_fragment, query_signs, kWmmaTile);
    wmma::load_matrix_sync(key_fragment, key_signs, kWmmaTile);
    wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
    __syncwarp();
  }

  wmma::store_matrix_sync(dots, accumulator, kWmmaTile, wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kWmmaTile * kWmmaTile;
       index += kWarpSize) {
    const int row = index / kWmmaTile;
    const int column = index % kWmmaTile;
    const int query_index = query_start + row;
    const int key_index = key_start + column;
    if (query_index < query_count && key_index < key_count) {
      const float dot = static_cast<float>(dots[index]);
      const float mismatches =
          0.5f * (static_cast<float>(symbol_bits) - dot);
      output[static_cast<int64_t>(query_index) * key_count + key_index] =
          __expf(
              -mismatch_scale * mismatches /
              static_cast<float>(symbol_bits));
    }
  }
}


__global__ void gate_b1_wmma_kernel(
    const int32_t* __restrict__ query_codes,
    const int32_t* __restrict__ key_codes,
    float* __restrict__ output,
    int query_count,
    int key_count,
    int symbol_bits,
    float mismatch_scale) {
#if __CUDA_ARCH__ >= 750
  constexpr int kTile = 8;
  constexpr int kWordsPerCode = 4;
  __shared__ __align__(16) uint32_t query_bits[kTile * kWordsPerCode];
  __shared__ __align__(16) uint32_t key_bits[kTile * kWordsPerCode];
  __shared__ __align__(16) int mismatches[kTile * kTile];

  const int lane = threadIdx.x;
  const int query_start = blockIdx.y * kTile;
  const int key_start = blockIdx.x * kTile;
  const uint32_t mask = symbol_mask(symbol_bits);
  for (int index = lane; index < kTile * kWordsPerCode;
       index += kWarpSize) {
    const int item = index / kWordsPerCode;
    const int word = index % kWordsPerCode;
    const int query_index = query_start + item;
    const int key_index = key_start + item;
    query_bits[index] =
        word == 0 && query_index < query_count
        ? static_cast<uint32_t>(query_codes[query_index]) & mask
        : 0u;
    key_bits[index] =
        word == 0 && key_index < key_count
        ? static_cast<uint32_t>(key_codes[key_index]) & mask
        : 0u;
  }
  __syncwarp();

  wmma::fragment<
      wmma::matrix_a,
      kTile,
      kTile,
      128,
      wmma::experimental::precision::b1,
      wmma::row_major>
      query_fragment;
  wmma::fragment<
      wmma::matrix_b,
      kTile,
      kTile,
      128,
      wmma::experimental::precision::b1,
      wmma::col_major>
      key_fragment;
  wmma::fragment<wmma::accumulator, kTile, kTile, 128, int> accumulator;
  wmma::fill_fragment(accumulator, 0);
  wmma::load_matrix_sync(query_fragment, query_bits, 128);
  wmma::load_matrix_sync(key_fragment, key_bits, 128);
  wmma::bmma_sync(
      accumulator,
      query_fragment,
      key_fragment,
      accumulator,
      wmma::experimental::bmmaBitOpXOR,
      wmma::experimental::bmmaAccumulateOpPOPC);
  wmma::store_matrix_sync(
      mismatches, accumulator, kTile, wmma::mem_row_major);
  __syncwarp();

  for (int index = lane; index < kTile * kTile; index += kWarpSize) {
    const int row = index / kTile;
    const int column = index % kTile;
    const int query_index = query_start + row;
    const int key_index = key_start + column;
    if (query_index < query_count && key_index < key_count) {
      output[static_cast<int64_t>(query_index) * key_count + key_index] =
          __expf(
              -mismatch_scale * static_cast<float>(mismatches[index]) /
              static_cast<float>(symbol_bits));
    }
  }
#endif
}


__global__ void batched_matmul_scalar_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ output,
    int batch_count,
    int rows,
    int inner,
    int columns) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch = blockIdx.z;
  if (batch >= batch_count || row >= rows || column >= columns) {
    return;
  }
  const int64_t left_base =
      (static_cast<int64_t>(batch) * rows + row) * inner;
  const int64_t right_base = static_cast<int64_t>(batch) * inner * columns;
  float sum = 0.0f;
  for (int reduction = 0; reduction < inner; ++reduction) {
    sum = fmaf(
        left[left_base + reduction],
        right[right_base + static_cast<int64_t>(reduction) * columns + column],
        sum);
  }
  output[(static_cast<int64_t>(batch) * rows + row) * columns + column] = sum;
}


template <typename element_t>
__device__ __forceinline__ element_t convert_element(float value);


template <>
__device__ __forceinline__ __half convert_element<__half>(float value) {
  return __float2half_rn(value);
}


template <>
__device__ __forceinline__ __nv_bfloat16
convert_element<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}


template <typename element_t>
__global__ void batched_matmul_wmma_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ output,
    int rows,
    int inner,
    int columns) {
  __shared__ __align__(16) element_t left_tile[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) element_t right_tile[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) float result[kWmmaTile * kWmmaTile];

  const int lane = threadIdx.x;
  const int batch = blockIdx.z;
  const int row_start = blockIdx.y * kWmmaTile;
  const int column_start = blockIdx.x * kWmmaTile;
  const int64_t left_batch = static_cast<int64_t>(batch) * rows * inner;
  const int64_t right_batch = static_cast<int64_t>(batch) * inner * columns;

  wmma::fragment<
      wmma::accumulator,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int reduction_start = 0; reduction_start < inner;
       reduction_start += kWmmaTile) {
    for (int index = lane; index < kWmmaTile * kWmmaTile;
         index += kWarpSize) {
      const int row = index / kWmmaTile;
      const int reduction = reduction_start + index % kWmmaTile;
      const float left_value = row_start + row < rows && reduction < inner
          ? left[left_batch +
                 static_cast<int64_t>(row_start + row) * inner + reduction]
          : 0.0f;
      left_tile[index] = convert_element<element_t>(left_value);

      const int column = index / kWmmaTile;
      const float right_value =
          column_start + column < columns && reduction < inner
          ? right[right_batch +
                  static_cast<int64_t>(reduction) * columns +
                  column_start + column]
          : 0.0f;
      right_tile[index] = convert_element<element_t>(right_value);
    }
    __syncwarp();
    wmma::fragment<
        wmma::matrix_a,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        element_t,
        wmma::row_major>
        left_fragment;
    wmma::fragment<
        wmma::matrix_b,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        element_t,
        wmma::col_major>
        right_fragment;
    wmma::load_matrix_sync(left_fragment, left_tile, kWmmaTile);
    wmma::load_matrix_sync(right_fragment, right_tile, kWmmaTile);
    wmma::mma_sync(accumulator, left_fragment, right_fragment, accumulator);
    __syncwarp();
  }
  wmma::store_matrix_sync(result, accumulator, kWmmaTile, wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kWmmaTile * kWmmaTile;
       index += kWarpSize) {
    const int row = row_start + index / kWmmaTile;
    const int column = column_start + index % kWmmaTile;
    if (row < rows && column < columns) {
      output[(static_cast<int64_t>(batch) * rows + row) * columns + column] =
          result[index];
    }
  }
}


__global__ void batched_matmul_tf32_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ output,
    int rows,
    int inner,
    int columns) {
#if __CUDA_ARCH__ >= 800
  constexpr int kReductionTile = 8;
  __shared__ __align__(16) float left_tile[kWmmaTile * kReductionTile];
  __shared__ __align__(16) float right_tile[kWmmaTile * kReductionTile];
  __shared__ __align__(16) float result[kWmmaTile * kWmmaTile];

  const int lane = threadIdx.x;
  const int batch = blockIdx.z;
  const int row_start = blockIdx.y * kWmmaTile;
  const int column_start = blockIdx.x * kWmmaTile;
  const int64_t left_batch = static_cast<int64_t>(batch) * rows * inner;
  const int64_t right_batch = static_cast<int64_t>(batch) * inner * columns;
  wmma::fragment<
      wmma::accumulator,
      kWmmaTile,
      kWmmaTile,
      kReductionTile,
      float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int reduction_start = 0; reduction_start < inner;
       reduction_start += kReductionTile) {
    for (int index = lane; index < kWmmaTile * kReductionTile;
         index += kWarpSize) {
      const int row = index / kReductionTile;
      const int reduction = reduction_start + index % kReductionTile;
      const float left_value = row_start + row < rows && reduction < inner
          ? left[left_batch +
                 static_cast<int64_t>(row_start + row) * inner + reduction]
          : 0.0f;
      left_tile[index] = wmma::__float_to_tf32(left_value);

      const int column = index / kReductionTile;
      const float right_value =
          column_start + column < columns && reduction < inner
          ? right[right_batch +
                  static_cast<int64_t>(reduction) * columns +
                  column_start + column]
          : 0.0f;
      right_tile[index] = wmma::__float_to_tf32(right_value);
    }
    __syncwarp();
    wmma::fragment<
        wmma::matrix_a,
        kWmmaTile,
        kWmmaTile,
        kReductionTile,
        wmma::precision::tf32,
        wmma::row_major>
        left_fragment;
    wmma::fragment<
        wmma::matrix_b,
        kWmmaTile,
        kWmmaTile,
        kReductionTile,
        wmma::precision::tf32,
        wmma::col_major>
        right_fragment;
    wmma::load_matrix_sync(left_fragment, left_tile, kReductionTile);
    wmma::load_matrix_sync(right_fragment, right_tile, kReductionTile);
    wmma::mma_sync(accumulator, left_fragment, right_fragment, accumulator);
    __syncwarp();
  }
  wmma::store_matrix_sync(result, accumulator, kWmmaTile, wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kWmmaTile * kWmmaTile;
       index += kWarpSize) {
    const int row = row_start + index / kWmmaTile;
    const int column = column_start + index % kWmmaTile;
    if (row < rows && column < columns) {
      output[(static_cast<int64_t>(batch) * rows + row) * columns + column] =
          result[index];
    }
  }
#endif
}


__device__ __forceinline__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}


__global__ void batched_matmul_fp16_scaled_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ output,
    int rows,
    int inner,
    int columns) {
  __shared__ __align__(16) __half left_tile[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half right_tile[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) float result[kWmmaTile * kWmmaTile];

  const int lane = threadIdx.x;
  const int batch = blockIdx.z;
  const int row_start = blockIdx.y * kWmmaTile;
  const int column_start = blockIdx.x * kWmmaTile;
  const int64_t left_batch = static_cast<int64_t>(batch) * rows * inner;
  const int64_t right_batch = static_cast<int64_t>(batch) * inner * columns;
  wmma::fragment<
      wmma::accumulator,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int reduction_start = 0; reduction_start < inner;
       reduction_start += kWmmaTile) {
    float left_maximum = 0.0f;
    float right_maximum = 0.0f;
    for (int index = lane; index < kWmmaTile * kWmmaTile;
         index += kWarpSize) {
      const int row = index / kWmmaTile;
      const int reduction = reduction_start + index % kWmmaTile;
      if (row_start + row < rows && reduction < inner) {
        left_maximum = fmaxf(
            left_maximum,
            fabsf(left[left_batch +
                       static_cast<int64_t>(row_start + row) * inner +
                       reduction]));
      }
      const int column = index / kWmmaTile;
      if (column_start + column < columns && reduction < inner) {
        right_maximum = fmaxf(
            right_maximum,
            fabsf(right[right_batch +
                        static_cast<int64_t>(reduction) * columns +
                        column_start + column]));
      }
    }
    left_maximum = warp_max(left_maximum);
    right_maximum = warp_max(right_maximum);
    const float inverse_left = left_maximum > 0.0f ? 1.0f / left_maximum : 0.0f;
    const float inverse_right =
        right_maximum > 0.0f ? 1.0f / right_maximum : 0.0f;

    for (int index = lane; index < kWmmaTile * kWmmaTile;
         index += kWarpSize) {
      const int row = index / kWmmaTile;
      const int reduction = reduction_start + index % kWmmaTile;
      const float left_value = row_start + row < rows && reduction < inner
          ? left[left_batch +
                 static_cast<int64_t>(row_start + row) * inner + reduction]
          : 0.0f;
      left_tile[index] = __float2half_rn(left_value * inverse_left);
      const int column = index / kWmmaTile;
      const float right_value =
          column_start + column < columns && reduction < inner
          ? right[right_batch +
                  static_cast<int64_t>(reduction) * columns +
                  column_start + column]
          : 0.0f;
      right_tile[index] = __float2half_rn(right_value * inverse_right);
    }
    __syncwarp();
    wmma::fragment<
        wmma::matrix_a,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        __half,
        wmma::row_major>
        left_fragment;
    wmma::fragment<
        wmma::matrix_b,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        __half,
        wmma::col_major>
        right_fragment;
    wmma::fragment<
        wmma::accumulator,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        float>
        partial;
    wmma::fill_fragment(partial, 0.0f);
    wmma::load_matrix_sync(left_fragment, left_tile, kWmmaTile);
    wmma::load_matrix_sync(right_fragment, right_tile, kWmmaTile);
    wmma::mma_sync(partial, left_fragment, right_fragment, partial);
    const float scale = left_maximum * right_maximum;
#pragma unroll
    for (int index = 0; index < accumulator.num_elements; ++index) {
      accumulator.x[index] += scale * partial.x[index];
    }
    __syncwarp();
  }
  wmma::store_matrix_sync(result, accumulator, kWmmaTile, wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kWmmaTile * kWmmaTile;
       index += kWarpSize) {
    const int row = row_start + index / kWmmaTile;
    const int column = column_start + index % kWmmaTile;
    if (row < rows && column < columns) {
      output[(static_cast<int64_t>(batch) * rows + row) * columns + column] =
          result[index];
    }
  }
}


__global__ void batched_matmul_fp16_hilo_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ output,
    int rows,
    int inner,
    int columns) {
  __shared__ __align__(16) __half left_high[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half left_low[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half right_high[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half right_low[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) float result[kWmmaTile * kWmmaTile];

  const int lane = threadIdx.x;
  const int batch = blockIdx.z;
  const int row_start = blockIdx.y * kWmmaTile;
  const int column_start = blockIdx.x * kWmmaTile;
  const int64_t left_batch = static_cast<int64_t>(batch) * rows * inner;
  const int64_t right_batch = static_cast<int64_t>(batch) * inner * columns;
  wmma::fragment<
      wmma::accumulator,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int reduction_start = 0; reduction_start < inner;
       reduction_start += kWmmaTile) {
    for (int index = lane; index < kWmmaTile * kWmmaTile;
         index += kWarpSize) {
      const int row = index / kWmmaTile;
      const int reduction = reduction_start + index % kWmmaTile;
      const float left_value = row_start + row < rows && reduction < inner
          ? left[left_batch +
                 static_cast<int64_t>(row_start + row) * inner + reduction]
          : 0.0f;
      const __half left_hi = __float2half_rn(left_value);
      left_high[index] = left_hi;
      left_low[index] = __float2half_rn(left_value - __half2float(left_hi));

      const int column = index / kWmmaTile;
      const float right_value =
          column_start + column < columns && reduction < inner
          ? right[right_batch +
                  static_cast<int64_t>(reduction) * columns +
                  column_start + column]
          : 0.0f;
      const __half right_hi = __float2half_rn(right_value);
      right_high[index] = right_hi;
      right_low[index] = __float2half_rn(right_value - __half2float(right_hi));
    }
    __syncwarp();

    using left_fragment_t = wmma::fragment<
        wmma::matrix_a,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        __half,
        wmma::row_major>;
    using right_fragment_t = wmma::fragment<
        wmma::matrix_b,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        __half,
        wmma::col_major>;
    left_fragment_t left_hi_fragment;
    left_fragment_t left_lo_fragment;
    right_fragment_t right_hi_fragment;
    right_fragment_t right_lo_fragment;
    wmma::load_matrix_sync(left_hi_fragment, left_high, kWmmaTile);
    wmma::load_matrix_sync(left_lo_fragment, left_low, kWmmaTile);
    wmma::load_matrix_sync(right_hi_fragment, right_high, kWmmaTile);
    wmma::load_matrix_sync(right_lo_fragment, right_low, kWmmaTile);
    wmma::mma_sync(
        accumulator, left_hi_fragment, right_hi_fragment, accumulator);
    wmma::mma_sync(
        accumulator, left_hi_fragment, right_lo_fragment, accumulator);
    wmma::mma_sync(
        accumulator, left_lo_fragment, right_hi_fragment, accumulator);
    wmma::mma_sync(
        accumulator, left_lo_fragment, right_lo_fragment, accumulator);
    __syncwarp();
  }
  wmma::store_matrix_sync(result, accumulator, kWmmaTile, wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kWmmaTile * kWmmaTile;
       index += kWarpSize) {
    const int row = row_start + index / kWmmaTile;
    const int column = column_start + index % kWmmaTile;
    if (row < rows && column < columns) {
      output[(static_cast<int64_t>(batch) * rows + row) * columns + column] =
          result[index];
    }
  }
}


__global__ void suffix_direct_kernel(
    const float* __restrict__ gates,
    float* __restrict__ scores,
    int length,
    int max_suffix_length) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  const int sequence = blockIdx.y;
  if (position >= length) {
    return;
  }
  const float* sequence_gates = gates + static_cast<int64_t>(sequence) * length;
  float product = 1.0f;
  float score = 0.0f;
  const int suffix_count = min(max_suffix_length, position + 1);
  for (int offset = 0; offset < suffix_count; ++offset) {
    product *= sequence_gates[position - offset];
    score += product;
  }
  scores[static_cast<int64_t>(sequence) * length + position] = score;
}


__global__ void suffix_warp_scan_kernel(
    const float* __restrict__ gates,
    float* __restrict__ scores,
    int sequence_count,
    int length,
    int max_suffix_length) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int sequence = blockIdx.x * warps_per_block + warp;
  if (sequence >= sequence_count) {
    return;
  }
  const float* sequence_gates = gates + static_cast<int64_t>(sequence) * length;
  float* sequence_scores = scores + static_cast<int64_t>(sequence) * length;
  float carry = 0.0f;

  for (int start = 0; start < length; start += max_suffix_length) {
    const int count = min(max_suffix_length, length - start);

    float history_prefix = lane < max_suffix_length && start - 1 - lane >= 0
        ? sequence_gates[start - 1 - lane]
        : 1.0f;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const float previous =
          __shfl_up_sync(0xffffffffu, history_prefix, offset);
      if (lane >= offset) {
        history_prefix *= previous;
      }
    }

    const float gate = lane < count ? sequence_gates[start + lane] : 1.0f;
    float current_prefix = gate;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const float previous =
          __shfl_up_sync(0xffffffffu, current_prefix, offset);
      if (lane >= offset) {
        current_prefix *= previous;
      }
    }

    float correction = 0.0f;
    const int history_lane =
        max(0, min(kWarpSize - 1, max_suffix_length - lane - 1));
    const float correction_history = __shfl_sync(
        0xffffffffu, history_prefix, history_lane);
    if (start > 0 && lane < count) {
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
      sequence_scores[start + lane] = score;
    }
    carry = __shfl_sync(0xffffffffu, score, count - 1);
  }
}

}  // namespace


torch::Tensor rosa_gate_matrix_cuda(
    const torch::Tensor& query_codes,
    const torch::Tensor& key_codes,
    int symbol_bits,
    float mismatch_scale,
    int method) {
  const c10::cuda::CUDAGuard device_guard(query_codes.device());
  auto output = torch::empty(
      {query_codes.numel(), key_codes.numel()},
      query_codes.options().dtype(torch::kFloat32));
  const int query_count = static_cast<int>(query_codes.numel());
  const int key_count = static_cast<int>(key_codes.numel());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (method == 0) {
    const dim3 block(16, 16);
    const dim3 grid(
        (key_count + block.x - 1) / block.x,
        (query_count + block.y - 1) / block.y);
    gate_scalar_kernel<<<grid, block, 0, stream>>>(
        query_codes.data_ptr<int32_t>(),
        key_codes.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        query_count,
        key_count,
        symbol_bits,
        mismatch_scale);
  } else if (method == 1) {
    const dim3 grid(
        (key_count + kWmmaTile - 1) / kWmmaTile,
        (query_count + kWmmaTile - 1) / kWmmaTile);
    gate_wmma_kernel<__half, false><<<grid, kWarpSize, 0, stream>>>(
        query_codes.data_ptr<int32_t>(),
        key_codes.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        query_count,
        key_count,
        symbol_bits,
        mismatch_scale);
  } else if (method == 2) {
    const dim3 grid(
        (key_count + kWmmaTile - 1) / kWmmaTile,
        (query_count + kWmmaTile - 1) / kWmmaTile);
    gate_wmma_kernel<signed char, true><<<grid, kWarpSize, 0, stream>>>(
        query_codes.data_ptr<int32_t>(),
        key_codes.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        query_count,
        key_count,
        symbol_bits,
        mismatch_scale);
  } else {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        properties->major > 7 ||
            (properties->major == 7 && properties->minor >= 5),
        "B1 WMMA requires compute capability 7.5 or newer");
    constexpr int kTile = 8;
    const dim3 grid(
        (key_count + kTile - 1) / kTile,
        (query_count + kTile - 1) / kTile);
    gate_b1_wmma_kernel<<<grid, kWarpSize, 0, stream>>>(
        query_codes.data_ptr<int32_t>(),
        key_codes.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        query_count,
        key_count,
        symbol_bits,
        mismatch_scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}


torch::Tensor rosa_batched_matmul_cuda(
    const torch::Tensor& left,
    const torch::Tensor& right,
    int method) {
  const c10::cuda::CUDAGuard device_guard(left.device());
  const int batch_count = static_cast<int>(left.size(0));
  const int rows = static_cast<int>(left.size(1));
  const int inner = static_cast<int>(left.size(2));
  const int columns = static_cast<int>(right.size(2));
  auto output = torch::empty(
      {batch_count, rows, columns}, left.options().dtype(torch::kFloat32));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (method == 0) {
    const dim3 block(16, 16);
    const dim3 grid(
        (columns + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y,
        batch_count);
    batched_matmul_scalar_kernel<<<grid, block, 0, stream>>>(
        left.data_ptr<float>(),
        right.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_count,
        rows,
        inner,
        columns);
  } else {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        method != 2 && method != 3 || properties->major >= 8,
        "BF16 and TF32 WMMA require compute capability 8.0 or newer");
    const dim3 grid(
        (columns + kWmmaTile - 1) / kWmmaTile,
        (rows + kWmmaTile - 1) / kWmmaTile,
        batch_count);
    if (method == 1) {
      batched_matmul_wmma_kernel<__half><<<grid, kWarpSize, 0, stream>>>(
          left.data_ptr<float>(),
          right.data_ptr<float>(),
          output.data_ptr<float>(),
          rows,
          inner,
          columns);
    } else if (method == 2) {
      batched_matmul_wmma_kernel<__nv_bfloat16>
          <<<grid, kWarpSize, 0, stream>>>(
              left.data_ptr<float>(),
              right.data_ptr<float>(),
              output.data_ptr<float>(),
              rows,
              inner,
              columns);
    } else if (method == 3) {
      batched_matmul_tf32_kernel<<<grid, kWarpSize, 0, stream>>>(
          left.data_ptr<float>(),
          right.data_ptr<float>(),
          output.data_ptr<float>(),
          rows,
          inner,
          columns);
    } else if (method == 4) {
      batched_matmul_fp16_scaled_kernel<<<grid, kWarpSize, 0, stream>>>(
          left.data_ptr<float>(),
          right.data_ptr<float>(),
          output.data_ptr<float>(),
          rows,
          inner,
          columns);
    } else {
      batched_matmul_fp16_hilo_kernel<<<grid, kWarpSize, 0, stream>>>(
          left.data_ptr<float>(),
          right.data_ptr<float>(),
          output.data_ptr<float>(),
          rows,
          inner,
          columns);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}


torch::Tensor rosa_suffix_scores_cuda(
    const torch::Tensor& gates,
    int max_suffix_length,
    int method) {
  const c10::cuda::CUDAGuard device_guard(gates.device());
  const int sequence_count = static_cast<int>(gates.size(0));
  const int length = static_cast<int>(gates.size(1));
  auto output = torch::empty_like(gates);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (method == 0) {
    constexpr int kThreads = 256;
    const dim3 grid((length + kThreads - 1) / kThreads, sequence_count);
    suffix_direct_kernel<<<grid, kThreads, 0, stream>>>(
        gates.data_ptr<float>(),
        output.data_ptr<float>(),
        length,
        max_suffix_length);
  } else {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / kWarpSize;
    const int blocks = (sequence_count + kWarps - 1) / kWarps;
    suffix_warp_scan_kernel<<<blocks, kThreads, 0, stream>>>(
        gates.data_ptr<float>(),
        output.data_ptr<float>(),
        sequence_count,
        length,
        max_suffix_length);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
