#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>


namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;


__device__ __forceinline__ uint32_t symbol_mask(int symbol_dim) {
  return symbol_dim == 32
      ? 0xffffffffu
      : (1u << symbol_dim) - 1u;
}


__device__ __forceinline__ void warp_affine_scan(
    float& coefficient,
    float& bias) {
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const float left_coefficient = __shfl_up_sync(
        0xffffffffu, coefficient, offset);
    const float left_bias = __shfl_up_sync(
        0xffffffffu, bias, offset);
    if ((threadIdx.x & (kWarpSize - 1)) >= offset) {
      bias = fmaf(coefficient, left_bias, bias);
      coefficient *= left_coefficient;
    }
  }
}


__device__ __forceinline__ void scan_diagonal(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ final_scores,
    int series,
    int delta,
    int seq_len,
    int symbol_dim,
    float mismatch_unit) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int32_t* query =
      packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
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
              (static_cast<uint32_t>(query[row]) ^
               static_cast<uint32_t>(key[key_position])) &
              mask)
        : 0;
    const float gate = active
        ? __expf(-mismatch_unit * static_cast<float>(mismatch))
        : 1.0f;
    float coefficient = gate;
    float bias = active ? gate : 0.0f;
    warp_affine_scan(coefficient, bias);
    const float chunk_coefficient = __shfl_sync(
        0xffffffffu, coefficient, kWarpSize - 1);
    const float chunk_bias = __shfl_sync(
        0xffffffffu, bias, kWarpSize - 1);
    incoming_score = fmaf(
        chunk_coefficient, incoming_score, chunk_bias);
  }
  if (lane == 0) {
    final_scores[
        static_cast<int64_t>(series) * (seq_len - 1) + delta - 1] =
        incoming_score;
  }
}


__global__ void one_diagonal_per_warp_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ final_scores,
    int series_count,
    int seq_len,
    int symbol_dim,
    float mismatch_unit) {
  const int warp = threadIdx.x / kWarpSize;
  const int64_t task =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int diagonals = seq_len - 1;
  const int64_t task_count = static_cast<int64_t>(series_count) * diagonals;
  if (task < task_count) {
    scan_diagonal(
        packed_query,
        packed_key,
        final_scores,
        static_cast<int>(task / diagonals),
        static_cast<int>(task % diagonals) + 1,
        seq_len,
        symbol_dim,
        mismatch_unit);
  }
}


__device__ __forceinline__ void scan_paired_task(
    const int32_t* packed_query,
    const int32_t* packed_key,
    float* final_scores,
    int64_t task,
    int pair_count,
    int seq_len,
    int symbol_dim,
    float mismatch_unit) {
  const int series = static_cast<int>(task / pair_count);
  const int pair = static_cast<int>(task % pair_count);
  const int first_delta = pair + 1;
  const int second_delta = seq_len - 1 - pair;
  scan_diagonal(
      packed_query,
      packed_key,
      final_scores,
      series,
      first_delta,
      seq_len,
      symbol_dim,
      mismatch_unit);
  if (second_delta != first_delta) {
    scan_diagonal(
        packed_query,
        packed_key,
        final_scores,
        series,
        second_delta,
        seq_len,
        symbol_dim,
        mismatch_unit);
  }
}


__global__ void paired_diagonal_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ final_scores,
    int series_count,
    int seq_len,
    int symbol_dim,
    float mismatch_unit) {
  const int warp = threadIdx.x / kWarpSize;
  const int64_t task =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int pair_count = seq_len / 2;
  const int64_t task_count = static_cast<int64_t>(series_count) * pair_count;
  if (task < task_count) {
    scan_paired_task(
        packed_query,
        packed_key,
        final_scores,
        task,
        pair_count,
        seq_len,
        symbol_dim,
        mismatch_unit);
  }
}


template <bool Paired>
__global__ void fixed_worker_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ final_scores,
    int series_count,
    int seq_len,
    int symbol_dim,
    float mismatch_unit) {
  const int warp = threadIdx.x / kWarpSize;
  const int64_t worker =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t worker_count =
      static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  const int span = Paired ? seq_len / 2 : seq_len - 1;
  const int64_t task_count = static_cast<int64_t>(series_count) * span;
  for (int64_t task = worker; task < task_count; task += worker_count) {
    if constexpr (Paired) {
      scan_paired_task(
          packed_query,
          packed_key,
          final_scores,
          task,
          span,
          seq_len,
          symbol_dim,
          mismatch_unit);
    } else {
      scan_diagonal(
          packed_query,
          packed_key,
          final_scores,
          static_cast<int>(task / span),
          static_cast<int>(task % span) + 1,
          seq_len,
          symbol_dim,
          mismatch_unit);
    }
  }
}


__global__ void persistent_queue_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    float* __restrict__ final_scores,
    unsigned long long* __restrict__ task_counter,
    int series_count,
    int seq_len,
    int symbol_dim,
    float mismatch_unit) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int diagonals = seq_len - 1;
  const int64_t task_count = static_cast<int64_t>(series_count) * diagonals;
  while (true) {
    unsigned long long task = 0;
    if (lane == 0) {
      task = atomicAdd(task_counter, 1ULL);
    }
    task = __shfl_sync(0xffffffffu, task, 0);
    if (task >= static_cast<unsigned long long>(task_count)) {
      return;
    }
    scan_diagonal(
        packed_query,
        packed_key,
        final_scores,
        static_cast<int>(task / diagonals),
        static_cast<int>(task % diagonals) + 1,
        seq_len,
        symbol_dim,
        mismatch_unit);
  }
}

}  // namespace


torch::Tensor rosa_diagonal_schedule_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int64_t symbol_dim,
    double mismatch_scale,
    int64_t method,
    int64_t worker_blocks) {
  const int batch = static_cast<int>(packed_query.size(0));
  const int heads = static_cast<int>(packed_query.size(1));
  const int seq_len = static_cast<int>(packed_query.size(2));
  const int series_count = batch * heads;
  auto output = torch::empty(
      {batch, heads, std::max(0, seq_len - 1)},
      packed_query.options().dtype(torch::kFloat32));
  if (seq_len <= 1) {
    return output;
  }
  const float mismatch_unit =
      static_cast<float>(mismatch_scale) / static_cast<float>(symbol_dim);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int diagonals = seq_len - 1;
  const int pair_count = seq_len / 2;
  const int normal_blocks = static_cast<int>(
      (static_cast<int64_t>(series_count) * diagonals +
       kWarpsPerBlock - 1) /
      kWarpsPerBlock);
  const int paired_blocks = static_cast<int>(
      (static_cast<int64_t>(series_count) * pair_count +
       kWarpsPerBlock - 1) /
      kWarpsPerBlock);
  if (worker_blocks == 0) {
    worker_blocks =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4;
  }
  const int fixed_blocks = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(worker_blocks, normal_blocks)));

  if (method == 0) {
    one_diagonal_per_warp_kernel<<<normal_blocks, kThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        series_count,
        seq_len,
        static_cast<int>(symbol_dim),
        mismatch_unit);
  } else if (method == 1) {
    paired_diagonal_kernel<<<paired_blocks, kThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        series_count,
        seq_len,
        static_cast<int>(symbol_dim),
        mismatch_unit);
  } else if (method == 2) {
    fixed_worker_kernel<false><<<fixed_blocks, kThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        series_count,
        seq_len,
        static_cast<int>(symbol_dim),
        mismatch_unit);
  } else if (method == 3) {
    auto counter = torch::zeros(
        {}, packed_query.options().dtype(torch::kInt64));
    persistent_queue_kernel<<<fixed_blocks, kThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(counter.data_ptr<int64_t>()),
        series_count,
        seq_len,
        static_cast<int>(symbol_dim),
        mismatch_unit);
  } else {
    fixed_worker_kernel<true><<<
        std::min(fixed_blocks, std::max(1, paired_blocks)),
        kThreads,
        0,
        stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        series_count,
        seq_len,
        static_cast<int>(symbol_dim),
        mismatch_unit);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
