#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cfloat>
#include <cstdint>
#include <limits>


namespace {

namespace wmma = nvcuda::wmma;

constexpr int kRows = 64;
constexpr int kRoutes = 64;
constexpr int kValueDim = 64;
constexpr int kWindow = 32;
constexpr int kThreads = 256;
constexpr int kGroupThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kGroupWarps = kGroupThreads / kWarpSize;
constexpr int kMatrixItems = kRows * kRoutes;
constexpr int kTensorTile = 16;
constexpr int kTensorReduction = 8;
constexpr int kScoreBuffers = 2;
constexpr int kScoreItems = kScoreBuffers * kMatrixItems;
constexpr int kUtilityItems = kMatrixItems;
constexpr int kGradItems = kRows * kValueDim;
constexpr int kValueItems = kRoutes * kValueDim;
constexpr int kFloatItems =
    kScoreItems + kUtilityItems + kGradItems + kValueItems;


struct PipelineShared {
  float* score[2];
  float* utility;
  float* grad;
  float* value;
};


__device__ __forceinline__ PipelineShared bind_shared(float* storage) {
  float* score0 = storage;
  float* score1 = score0 + kMatrixItems;
  float* utility = score1 + kMatrixItems;
  float* grad = utility + kUtilityItems;
  float* value = grad + kGradItems;
  return {{score0, score1}, utility, grad, value};
}


// Barriers 1/2 synchronize one role group. Ready barriers 3/4 and free
// barriers 5/6 complete after one 128-thread arrive and one 128-thread wait.
__device__ __forceinline__ void producer_group_sync() {
  asm volatile("bar.sync 1, 128;" : : : "memory");
}


__device__ __forceinline__ void consumer_group_sync() {
  asm volatile("bar.sync 2, 128;" : : : "memory");
}


__device__ __forceinline__ void producer_wait_free(int buffer) {
  if (buffer == 0) {
    asm volatile("bar.sync 5, 256;" : : : "memory");
  } else {
    asm volatile("bar.sync 6, 256;" : : : "memory");
  }
}


__device__ __forceinline__ void producer_signal_ready(int buffer) {
  if (buffer == 0) {
    asm volatile("bar.arrive 3, 256;" : : : "memory");
  } else {
    asm volatile("bar.arrive 4, 256;" : : : "memory");
  }
}


__device__ __forceinline__ void consumer_wait_ready(int buffer) {
  if (buffer == 0) {
    asm volatile("bar.sync 3, 256;" : : : "memory");
  } else {
    asm volatile("bar.sync 4, 256;" : : : "memory");
  }
}


__device__ __forceinline__ void consumer_signal_free(int buffer) {
  if (buffer == 0) {
    asm volatile("bar.arrive 5, 256;" : : : "memory");
  } else {
    asm volatile("bar.arrive 6, 256;" : : : "memory");
  }
}


__device__ __forceinline__ uint32_t symbol_mask(int symbol_dim) {
  return symbol_dim == 32
      ? 0xffffffffu
      : (1u << symbol_dim) - 1u;
}


__device__ __forceinline__ int mismatch_count(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int route,
    uint32_t mask) {
  return __popc(
      (static_cast<uint32_t>(query[row]) ^
       static_cast<uint32_t>(key[route - 1])) &
      mask);
}


__device__ __forceinline__ float mismatch_gate(
    int mismatch,
    float mismatch_unit) {
  return __expf(-mismatch_unit * static_cast<float>(mismatch));
}


__device__ __forceinline__ void initialize_state(
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
  rolling_count = min(kWindow + 1, available);
  rolling_mismatch = 0;
  float product = 1.0f;
  score = 0.0f;
#pragma unroll
  for (int suffix = 0; suffix < kWindow + 1; ++suffix) {
    if (suffix < rolling_count) {
      const int mismatch = mismatch_count(
          query,
          key,
          row - suffix,
          route - suffix,
          mask);
      rolling_mismatch += mismatch;
      if (suffix < kWindow) {
        product *= mismatch_gate(mismatch, mismatch_unit);
        score += product;
      }
    }
  }
}


__device__ __forceinline__ void advance_state(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int route,
    uint32_t mask,
    float mismatch_unit,
    float& score,
    int& rolling_mismatch,
    int& rolling_count) {
  const int mismatch = mismatch_count(query, key, row, route, mask);
  if (rolling_count == kWindow + 1) {
    rolling_mismatch -= mismatch_count(
        query,
        key,
        row - kWindow - 1,
        route - kWindow - 1,
        mask);
  } else {
    ++rolling_count;
  }
  rolling_mismatch += mismatch;
  const float gate = mismatch_gate(mismatch, mismatch_unit);
  const float correction = rolling_count == kWindow + 1
      ? mismatch_gate(rolling_mismatch, mismatch_unit)
      : 0.0f;
  score = fmaf(gate, 1.0f + score, -correction);
}


__device__ __forceinline__ void clear_score(
    float* score,
    int worker,
    int workers) {
  for (int index = worker; index < kMatrixItems; index += workers) {
    score[index] = 0.0f;
  }
}


__device__ __forceinline__ void generate_score_diagonals(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    float* score_tile,
    int row_start,
    int route_start,
    int route_count,
    uint32_t mask,
    float mismatch_unit,
    int worker,
    int workers) {
  const int diagonal_count = kRows + route_count - 1;
  for (int diagonal = worker; diagonal < diagonal_count;
       diagonal += workers) {
    const int difference = diagonal - (route_count - 1);
    const int first_row_offset = max(difference, 0);
    const int first_route_offset = max(-difference, 0);
    const int count = min(
        kRows - first_row_offset,
        route_count - first_route_offset);
    int row = row_start + first_row_offset;
    int route = route_start + first_route_offset;
    if (count <= 0 || route > row) {
      continue;
    }
    float score;
    int rolling_mismatch;
    int rolling_count;
    initialize_state(
        query,
        key,
        row,
        route,
        mask,
        mismatch_unit,
        score,
        rolling_mismatch,
        rolling_count);
    score_tile[first_row_offset * kRoutes + first_route_offset] = score;
    for (int step = 1; step < count; ++step) {
      ++row;
      ++route;
      advance_state(
          query,
          key,
          row,
          route,
          mask,
          mismatch_unit,
          score,
          rolling_mismatch,
          rolling_count);
      score_tile[
          (first_row_offset + step) * kRoutes +
          first_route_offset + step] = score;
    }
  }
}


__device__ __forceinline__ void load_utility_inputs(
    const float* __restrict__ grad_output,
    const float* __restrict__ value,
    PipelineShared shared,
    int series,
    int seq_len,
    int route_start,
    int route_count,
    int worker,
    int workers) {
  for (int index = worker; index < kGradItems; index += workers) {
    shared.grad[index] = wmma::__float_to_tf32(
        grad_output[static_cast<int64_t>(series) * kGradItems + index]);
  }
  for (int index = worker; index < kValueItems; index += workers) {
    const int route_offset = index / kValueDim;
    const int feature = index - route_offset * kValueDim;
    float sign = 0.0f;
    if (route_offset < route_count) {
      const int route = route_start + route_offset;
      const int64_t source =
          (static_cast<int64_t>(series) * seq_len + route) * kValueDim +
          feature;
      sign = value[source] > 0.0f ? 1.0f : -1.0f;
    }
    shared.value[index] = wmma::__float_to_tf32(sign);
  }
}


__device__ __forceinline__ void compute_utilities(
    PipelineShared shared,
    int warp,
    int warp_count) {
#if __CUDA_ARCH__ >= 800
  constexpr int kOutputTiles = 16;
  for (int tile = warp; tile < kOutputTiles; tile += warp_count) {
    const int output_row = (tile / 4) * kTensorTile;
    const int output_column = (tile % 4) * kTensorTile;
    wmma::fragment<
        wmma::accumulator,
        kTensorTile,
        kTensorTile,
        kTensorReduction,
        float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int reduction = 0; reduction < kValueDim;
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
          shared.grad + output_row * kValueDim + reduction,
          kValueDim);
      wmma::load_matrix_sync(
          right,
          shared.value + output_column * kValueDim + reduction,
          kValueDim);
      wmma::mma_sync(accumulator, left, right, accumulator);
    }
    wmma::store_matrix_sync(
        shared.utility + output_row * kRoutes + output_column,
        accumulator,
        kRoutes,
        wmma::mem_row_major);
  }
#endif
}


__device__ __forceinline__ float consume_tile(
    const float* score,
    const float* utility,
    int row_start,
    int route_start,
    int route_count,
    int worker) {
  if (worker >= kRows) {
    return 0.0f;
  }
  const int row = row_start + worker;
  float result = 0.0f;
  for (int route_offset = 0; route_offset < route_count; ++route_offset) {
    const int route = route_start + route_offset;
    if (route <= row) {
      const float raw = score[worker * kRoutes + route_offset];
      const float weight = __expf(0.125f * sqrtf(1.0f + fmaxf(raw, 0.0f)));
      result = fmaf(weight, utility[worker * kRoutes + route_offset], result);
    }
  }
  return result;
}


__global__ __launch_bounds__(kThreads, 1)
void sequential_pipeline_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const float* __restrict__ grad_output,
    const float* __restrict__ value,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    float mismatch_scale) {
  extern __shared__ float storage[];
  PipelineShared shared = bind_shared(storage);
  const int series = blockIdx.x;
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  const int row_start = seq_len - kRows;
  const int row_end = seq_len - 1;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);
  float row_total = 0.0f;

  for (int route_start = 1; route_start <= row_end;
       route_start += kRoutes) {
    const int route_count = min(kRoutes, seq_len - route_start);
    clear_score(shared.score[0], threadIdx.x, kThreads);
    __syncthreads();
    generate_score_diagonals(
        query,
        key,
        shared.score[0],
        row_start,
        route_start,
        route_count,
        mask,
        mismatch_unit,
        threadIdx.x,
        kThreads);
    __syncthreads();
    load_utility_inputs(
        grad_output,
        value,
        shared,
        series,
        seq_len,
        route_start,
        route_count,
        threadIdx.x,
        kThreads);
    __syncthreads();
    compute_utilities(shared, threadIdx.x / kWarpSize, kWarps);
    __syncthreads();
    if (threadIdx.x < kRows) {
      row_total += consume_tile(
          shared.score[0],
          shared.utility,
          row_start,
          route_start,
          route_count,
          threadIdx.x);
    }
    __syncthreads();
  }
  if (threadIdx.x < kRows) {
    output[static_cast<int64_t>(series) * kRows + threadIdx.x] = row_total;
  }
}


__global__ __launch_bounds__(kThreads, 1)
void warp_specialized_pipeline_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const float* __restrict__ grad_output,
    const float* __restrict__ value,
    float* __restrict__ output,
    int seq_len,
    int symbol_dim,
    float mismatch_scale) {
  extern __shared__ float storage[];
  PipelineShared shared = bind_shared(storage);
  const int series = blockIdx.x;
  const int group_id = threadIdx.x / kGroupThreads;
  const int local_thread = threadIdx.x % kGroupThreads;
  const int local_warp = local_thread / kWarpSize;
  const int32_t* query = packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key = packed_key + static_cast<int64_t>(series) * seq_len;
  const int row_start = seq_len - kRows;
  const int row_end = seq_len - 1;
  const int route_tiles = (row_end + kRoutes - 1) / kRoutes;
  const uint32_t mask = symbol_mask(symbol_dim);
  const float mismatch_unit = mismatch_scale / static_cast<float>(symbol_dim);

  if (group_id == 0) {
    for (int tile = 0; tile < route_tiles; ++tile) {
      const int buffer = tile & 1;
      if (tile >= kScoreBuffers) {
        producer_wait_free(buffer);
      }
      const int route_start = 1 + tile * kRoutes;
      const int route_count = min(kRoutes, seq_len - route_start);
      clear_score(
          shared.score[buffer],
          local_thread,
          kGroupThreads);
      producer_group_sync();
      generate_score_diagonals(
          query,
          key,
          shared.score[buffer],
          row_start,
          route_start,
          route_count,
          mask,
          mismatch_unit,
          local_thread,
          kGroupThreads);
      producer_group_sync();
      producer_signal_ready(buffer);
    }
  } else {
    float row_total = 0.0f;
    for (int tile = 0; tile < route_tiles; ++tile) {
      const int buffer = tile & 1;
      consumer_wait_ready(buffer);
      const int route_start = 1 + tile * kRoutes;
      const int route_count = min(kRoutes, seq_len - route_start);
      load_utility_inputs(
          grad_output,
          value,
          shared,
          series,
          seq_len,
          route_start,
          route_count,
          local_thread,
          kGroupThreads);
      consumer_group_sync();
      compute_utilities(shared, local_warp, kGroupWarps);
      consumer_group_sync();
      if (local_thread < kRows) {
        row_total += consume_tile(
            shared.score[buffer],
            shared.utility,
            row_start,
            route_start,
            route_count,
            local_thread);
      }
      consumer_group_sync();
      consumer_signal_free(buffer);
    }
    if (local_thread < kRows) {
      output[static_cast<int64_t>(series) * kRows + local_thread] = row_total;
    }
  }
  __syncthreads();
}

}  // namespace


torch::Tensor rosa_block_pipeline_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& grad_output,
    const torch::Tensor& value,
    int symbol_dim,
    float mismatch_scale,
    int method) {
  const int64_t series64 = packed_query.size(0);
  const int64_t seq_len64 = packed_query.size(1);
  TORCH_CHECK(series64 <= std::numeric_limits<int>::max());
  TORCH_CHECK(seq_len64 <= std::numeric_limits<int>::max());
  const int series = static_cast<int>(series64);
  const int seq_len = static_cast<int>(seq_len64);
  auto output = torch::empty(
      {series64, kRows},
      grad_output.options());
  constexpr size_t kSharedBytes =
      kFloatItems * sizeof(float);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (method == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sequential_pipeline_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes)));
    sequential_pipeline_kernel<<<
        series,
        kThreads,
        kSharedBytes,
        stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        grad_output.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        mismatch_scale);
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        warp_specialized_pipeline_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes)));
    warp_specialized_pipeline_kernel<<<
        series,
        kThreads,
        kSharedBytes,
        stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        grad_output.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len,
        symbol_dim,
        mismatch_scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
