#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>


namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 128;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr int kIndexThreads = 256;
constexpr int kIndexWarps = kIndexThreads / kWarpSize;
constexpr int kAlphabetCapacity = 256;
constexpr int kSurvivorPrefix = 8;


#define DISPATCH_ROSA_FLOAT_TYPES(TYPE, NAME, ...)              \
  AT_DISPATCH_SWITCH(                                           \
      TYPE,                                                     \
      NAME,                                                     \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))


template <typename scalar_t>
__device__ __forceinline__ float read_float(
    const scalar_t* __restrict__ values,
    int64_t index) {
  return static_cast<float>(values[index]);
}


template <typename scalar_t>
__global__ void pack_sign_bits_pair_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    int32_t* __restrict__ packed_query,
    int32_t* __restrict__ packed_key,
    int64_t rows,
    int seq_len,
    int num_heads,
    int symbol_dim) {
  const int64_t source_row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (source_row >= rows) {
    return;
  }
  const int64_t tokens_per_batch =
      static_cast<int64_t>(seq_len) * num_heads;
  const int64_t batch = source_row / tokens_per_batch;
  const int64_t within_batch = source_row - batch * tokens_per_batch;
  const int64_t token = within_batch / num_heads;
  const int64_t head = within_batch - token * num_heads;
  const int64_t packed_row =
      (batch * num_heads + head) * seq_len + token;
  const int64_t base = source_row * symbol_dim;
  uint32_t query_word = 0;
  uint32_t key_word = 0;
  for (int bit = 0; bit < symbol_dim; ++bit) {
    query_word |=
        static_cast<uint32_t>(read_float(query, base + bit) > 0.0f) << bit;
    key_word |=
        static_cast<uint32_t>(read_float(key, base + bit) > 0.0f) << bit;
  }
  packed_query[packed_row] = static_cast<int32_t>(query_word);
  packed_key[packed_row] = static_cast<int32_t>(key_word);
}


__global__ void build_occurrence_index_kernel(
    const int32_t* __restrict__ packed_key,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ occurrences,
    int seq_len,
    int alphabet_size) {
  static_cast<void>(alphabet_size);
  __shared__ int shared_counts[kAlphabetCapacity];
  __shared__ int shared_cursors[kAlphabetCapacity];
  const int series = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int32_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
  int32_t* series_offsets =
      offsets + static_cast<int64_t>(series) * (kAlphabetCapacity + 1);
  int32_t* output =
      occurrences + static_cast<int64_t>(series) * seq_len;

  shared_counts[tid] = 0;
  __syncthreads();
  for (int chunk = 0; chunk < seq_len; chunk += blockDim.x) {
    const int position = chunk + tid;
    const unsigned active = __ballot_sync(
        0xffffffffu,
        position < seq_len);
    if (position < seq_len) {
      const int code = key[position];
      const unsigned peers = __match_any_sync(active, code);
      const int leader = __ffs(peers) - 1;
      if (lane == leader) {
        atomicAdd(&shared_counts[code], __popc(peers));
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    int prefix = 0;
    for (int candidate_code = 0;
         candidate_code < kAlphabetCapacity;
         ++candidate_code) {
      series_offsets[candidate_code] = prefix;
      shared_cursors[candidate_code] = prefix;
      prefix += shared_counts[candidate_code];
    }
    series_offsets[kAlphabetCapacity] = prefix;
  }
  __syncthreads();

  for (int chunk = 0; chunk < seq_len; chunk += blockDim.x) {
    const int position = chunk + tid;
    const unsigned active = __ballot_sync(
        0xffffffffu,
        position < seq_len);
    const int code = position < seq_len ? key[position] : 0;
    for (int active_warp = 0;
         active_warp < kIndexWarps;
         ++active_warp) {
      if (warp == active_warp && position < seq_len) {
        const unsigned peers = __match_any_sync(active, code);
        const int leader = __ffs(peers) - 1;
        int base = 0;
        if (lane == leader) {
          base = atomicAdd(&shared_cursors[code], __popc(peers));
        }
        base = __shfl_sync(peers, base, leader);
        const unsigned preceding =
            peers & ((1u << lane) - 1u);
        output[base + __popc(preceding)] = position;
      }
      __syncthreads();
    }
  }
}


__device__ __forceinline__ int upper_bound_position(
    const int32_t* __restrict__ occurrences,
    int begin,
    int end,
    int maximum_position) {
  int low = begin;
  int high = end;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (occurrences[middle] <= maximum_position) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}


__device__ __forceinline__ int exact_suffix_length(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int route,
    int max_suffix_length,
    int start_offset = 0) {
  const int suffix_steps = min(max_suffix_length, min(row + 1, route));
  for (int offset = start_offset; offset < suffix_steps; ++offset) {
    if (query[row - offset] != key[route - 1 - offset]) {
      return offset;
    }
  }
  return suffix_steps;
}


__device__ __forceinline__ void merge_route(
    int other_length,
    int other_route,
    int& best_length,
    int& best_route) {
  if (other_length > best_length ||
      (other_length > 0 &&
       other_length == best_length &&
       other_route > best_route)) {
    best_length = other_length;
    best_route = other_route;
  }
}


__device__ __forceinline__ void warp_reduce_route(
    int& length,
    int& route) {
  const int lane = threadIdx.x & (kWarpSize - 1);
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const int other_length =
        __shfl_down_sync(0xffffffffu, length, offset);
    const int other_route =
        __shfl_down_sync(0xffffffffu, route, offset);
    if (lane + offset < kWarpSize) {
      merge_route(other_length, other_route, length, route);
    }
  }
}


__device__ __forceinline__ int warp_suffix_length(
    const int32_t* __restrict__ query,
    const int32_t* __restrict__ key,
    int row,
    int route,
    int max_suffix_length,
    int start_offset) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int suffix_steps = min(max_suffix_length, min(row + 1, route));
  int length = suffix_steps;
  for (int chunk = start_offset; chunk < suffix_steps; chunk += kWarpSize) {
    const int offset = chunk + lane;
    const bool active = offset < suffix_steps;
    const bool mismatch = active &&
        query[row - offset] != key[route - 1 - offset];
    const unsigned mismatch_mask = __ballot_sync(0xffffffffu, mismatch);
    if (mismatch_mask != 0u) {
      length = chunk + __ffs(mismatch_mask) - 1;
      break;
    }
  }
  return __shfl_sync(0xffffffffu, length, 0);
}


template <typename scalar_t>
__device__ __forceinline__ void write_selected_value(
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    int32_t* __restrict__ selected_routes,
    int32_t* __restrict__ selected_lengths,
    int batch,
    int row,
    int head,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int route,
    int length) {
  if (threadIdx.x == 0) {
    const int64_t route_index =
        (static_cast<int64_t>(batch) * num_heads + head) * seq_len + row;
    selected_routes[route_index] = route;
    selected_lengths[route_index] = length;
  }
  const int value_head = head / (num_heads / num_value_heads);
  for (int feature = threadIdx.x;
       feature < value_dim;
       feature += blockDim.x) {
    float result = 0.0f;
    if (route > 0) {
      const int64_t value_index =
          ((static_cast<int64_t>(batch) * seq_len + route) *
               num_value_heads +
           value_head) *
              value_dim +
          feature;
      result = read_float(value, value_index) > 0.0f ? 1.0f : -1.0f;
    }
    const int64_t output_index =
        ((static_cast<int64_t>(batch) * seq_len + row) * num_heads + head) *
            value_dim +
        feature;
    output[output_index] = static_cast<scalar_t>(result);
  }
}


template <typename scalar_t, bool UseCertificate>
__global__ void indexed_lane_hard_forward_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const scalar_t* __restrict__ value,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ occurrences,
    scalar_t* __restrict__ output,
    int32_t* __restrict__ selected_routes,
    int32_t* __restrict__ selected_lengths,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int max_suffix_length) {
  __shared__ int shared[2 * kWarpsPerBlock + 6];
  const int linear_row = blockIdx.x;
  const int row = linear_row % seq_len;
  const int head = (linear_row / seq_len) % num_heads;
  const int batch = linear_row / (seq_len * num_heads);
  const int series = batch * num_heads + head;
  const int32_t* query =
      packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
  const int32_t* series_offsets =
      offsets + static_cast<int64_t>(series) * (kAlphabetCapacity + 1);
  const int32_t* series_occurrences =
      occurrences + static_cast<int64_t>(series) * seq_len;
  const int code = query[row];

  if (threadIdx.x == 0) {
    const int begin = series_offsets[code];
    const int end = series_offsets[code + 1];
    shared[0] = begin;
    shared[1] = upper_bound_position(
        series_occurrences,
        begin,
        end,
        row - 1);
    shared[2] = 0;
    shared[3] = 0;
  }
  __syncthreads();

  const int begin = shared[0];
  const int upper = shared[1];
  const int row_maximum = min(max_suffix_length, row);
  if constexpr (UseCertificate) {
    if (upper > begin && threadIdx.x < kWarpSize) {
      const int latest_route = series_occurrences[upper - 1] + 1;
      const int latest_length = warp_suffix_length(
          query,
          key,
          row,
          latest_route,
          max_suffix_length,
          1);
      if (threadIdx.x == 0) {
        shared[2] = latest_length;
        shared[3] = latest_route;
      }
    }
    __syncthreads();
    if (shared[2] == row_maximum && row_maximum > 0) {
      write_selected_value(
          value,
          output,
          selected_routes,
          selected_lengths,
          batch,
          row,
          head,
          seq_len,
          num_heads,
          num_value_heads,
          value_dim,
          shared[3],
          shared[2]);
      return;
    }
  }

  int best_length = 0;
  int best_route = 0;
  for (int occurrence_index = upper - 1 - threadIdx.x;
       occurrence_index >= begin;
       occurrence_index -= blockDim.x) {
    const int route = series_occurrences[occurrence_index] + 1;
    const int length = exact_suffix_length(
        query,
        key,
        row,
        route,
        max_suffix_length,
        1);
    merge_route(length, route, best_length, best_route);
  }

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  warp_reduce_route(best_length, best_route);
  if (lane == 0) {
    shared[4 + warp] = best_length;
    shared[4 + kWarpsPerBlock + warp] = best_route;
  }
  __syncthreads();
  if (warp == 0) {
    best_length = lane < kWarpsPerBlock ? shared[4 + lane] : 0;
    best_route = lane < kWarpsPerBlock
        ? shared[4 + kWarpsPerBlock + lane]
        : 0;
    warp_reduce_route(best_length, best_route);
    if (lane == 0) {
      shared[2] = best_length;
      shared[3] = best_route;
    }
  }
  __syncthreads();
  write_selected_value(
      value,
      output,
      selected_routes,
      selected_lengths,
      batch,
      row,
      head,
      seq_len,
      num_heads,
      num_value_heads,
      value_dim,
      shared[3],
      shared[2]);
}


template <typename scalar_t>
__global__ void indexed_hybrid_hard_forward_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const scalar_t* __restrict__ value,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ occurrences,
    scalar_t* __restrict__ output,
    int32_t* __restrict__ selected_routes,
    int32_t* __restrict__ selected_lengths,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int max_suffix_length) {
  __shared__ int shared[2 * kWarpsPerBlock + 6];
  const int linear_row = blockIdx.x;
  const int row = linear_row % seq_len;
  const int head = (linear_row / seq_len) % num_heads;
  const int batch = linear_row / (seq_len * num_heads);
  const int series = batch * num_heads + head;
  const int32_t* query =
      packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
  const int32_t* series_offsets =
      offsets + static_cast<int64_t>(series) * (kAlphabetCapacity + 1);
  const int32_t* series_occurrences =
      occurrences + static_cast<int64_t>(series) * seq_len;
  const int code = query[row];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

  if (threadIdx.x == 0) {
    const int begin = series_offsets[code];
    const int end = series_offsets[code + 1];
    shared[0] = begin;
    shared[1] = upper_bound_position(
        series_occurrences,
        begin,
        end,
        row - 1);
    shared[2] = 0;
    shared[3] = 0;
  }
  __syncthreads();

  const int begin = shared[0];
  const int upper = shared[1];
  const int row_maximum = min(max_suffix_length, row);
  if (upper > begin && warp == 0) {
    const int latest_route = series_occurrences[upper - 1] + 1;
    const int latest_length = warp_suffix_length(
        query,
        key,
        row,
        latest_route,
        max_suffix_length,
        1);
    if (lane == 0) {
      shared[2] = latest_length;
      shared[3] = latest_route;
    }
  }
  __syncthreads();

  if (shared[2] == row_maximum && row_maximum > 0) {
    write_selected_value(
        value,
        output,
        selected_routes,
        selected_lengths,
        batch,
        row,
        head,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim,
        shared[3],
        shared[2]);
    return;
  }

  int best_length = 0;
  int best_route = 0;
  for (int tile = 0; tile < upper - begin; tile += kBlockThreads) {
    const int occurrence_index = upper - 1 - tile - threadIdx.x;
    int route = 0;
    int length = 0;
    bool survivor = false;
    if (occurrence_index >= begin) {
      route = series_occurrences[occurrence_index] + 1;
      const int suffix_steps = min(max_suffix_length, min(row + 1, route));
      length = 1;
      for (int offset = 1;
           offset < min(suffix_steps, kSurvivorPrefix);
           ++offset) {
        if (query[row - offset] != key[route - 1 - offset]) {
          break;
        }
        length = offset + 1;
      }
      if (length == kSurvivorPrefix &&
          suffix_steps > kSurvivorPrefix) {
        survivor = true;
      }
    }

    unsigned survivors = __ballot_sync(0xffffffffu, survivor);
    while (survivors != 0u) {
      const int owner = __ffs(survivors) - 1;
      const int survivor_route =
          __shfl_sync(0xffffffffu, route, owner);
      const int survivor_length = warp_suffix_length(
          query,
          key,
          row,
          survivor_route,
          max_suffix_length,
          kSurvivorPrefix);
      if (lane == owner) {
        length = survivor_length;
      }
      survivors &= ~(1u << owner);
    }
    merge_route(length, route, best_length, best_route);
  }

  warp_reduce_route(best_length, best_route);
  if (lane == 0) {
    shared[4 + warp] = best_length;
    shared[4 + kWarpsPerBlock + warp] = best_route;
  }
  __syncthreads();
  if (warp == 0) {
    best_length = lane < kWarpsPerBlock ? shared[4 + lane] : 0;
    best_route = lane < kWarpsPerBlock
        ? shared[4 + kWarpsPerBlock + lane]
        : 0;
    warp_reduce_route(best_length, best_route);
    if (lane == 0) {
      shared[2] = best_length;
      shared[3] = best_route;
    }
  }
  __syncthreads();

  write_selected_value(
      value,
      output,
      selected_routes,
      selected_lengths,
      batch,
      row,
      head,
      seq_len,
      num_heads,
      num_value_heads,
      value_dim,
      shared[3],
      shared[2]);
}


__global__ void diagonal_winner_kernel(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    int64_t* __restrict__ winners,
    int seq_len,
    int series_count,
    int max_suffix_length) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t global_warp =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int diagonals = seq_len - 1;
  const int64_t total_warps = static_cast<int64_t>(series_count) * diagonals;
  if (global_warp >= total_warps) {
    return;
  }
  const int series = static_cast<int>(global_warp / diagonals);
  const int diagonal = static_cast<int>(global_warp -
      static_cast<int64_t>(series) * diagonals) + 1;
  const int count = seq_len - diagonal;
  const int32_t* query =
      packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
  int latest_mismatch = -1;

  for (int chunk = 0; chunk < count; chunk += kWarpSize) {
    const int key_position = chunk + lane;
    const bool active = key_position < count;
    const int query_position = key_position + diagonal;
    const bool equal = active && query[query_position] == key[key_position];
    int prefix_mismatch = active && !equal ? key_position : -1;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int previous =
          __shfl_up_sync(0xffffffffu, prefix_mismatch, offset);
      if (lane >= offset) {
        prefix_mismatch = max(prefix_mismatch, previous);
      }
    }
    const int preceding_mismatch = max(latest_mismatch, prefix_mismatch);
    if (equal) {
      const int length = min(
          max_suffix_length,
          key_position - preceding_mismatch);
      const uint64_t priority =
          (static_cast<uint64_t>(length) << 32) |
          static_cast<uint32_t>(key_position + 1);
      atomicMax(
          reinterpret_cast<unsigned long long*>(
              winners +
              static_cast<int64_t>(series) * seq_len +
              query_position),
          static_cast<unsigned long long>(priority));
    }
    const int chunk_mismatch =
        __shfl_sync(0xffffffffu, prefix_mismatch, kWarpSize - 1);
    latest_mismatch = max(latest_mismatch, chunk_mismatch);
  }
}


template <typename scalar_t>
__global__ void gather_diagonal_winners_kernel(
    const int64_t* __restrict__ winners,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    int32_t* __restrict__ selected_routes,
    int32_t* __restrict__ selected_lengths,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim) {
  const int linear_row = blockIdx.x;
  const int row = linear_row % seq_len;
  const int head = (linear_row / seq_len) % num_heads;
  const int batch = linear_row / (seq_len * num_heads);
  const int series = batch * num_heads + head;
  const uint64_t priority = static_cast<uint64_t>(
      winners[static_cast<int64_t>(series) * seq_len + row]);
  const int route = static_cast<int>(priority & 0xffffffffu);
  const int length = static_cast<int>(priority >> 32);
  write_selected_value(
      value,
      output,
      selected_routes,
      selected_lengths,
      batch,
      row,
      head,
      seq_len,
      num_heads,
      num_value_heads,
      value_dim,
      route,
      length);
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor> rosa_pack_sign_bits_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key) {
  const int batch = static_cast<int>(query.size(0));
  const int seq_len = static_cast<int>(query.size(1));
  const int num_heads = static_cast<int>(query.size(2));
  const int symbol_dim = static_cast<int>(query.size(3));
  const int64_t rows =
      static_cast<int64_t>(batch) * seq_len * num_heads;
  auto options = query.options().dtype(torch::kInt32);
  auto packed_query = torch::empty({batch, num_heads, seq_len}, options);
  auto packed_key = torch::empty({batch, num_heads, seq_len}, options);
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_pack_sign_bits",
      [&] {
        pack_sign_bits_pair_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            packed_query.data_ptr<int32_t>(),
            packed_key.data_ptr<int32_t>(),
            rows,
            seq_len,
            num_heads,
            symbol_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_query, packed_key};
}


std::tuple<torch::Tensor, torch::Tensor> rosa_build_occurrence_index_cuda(
    const torch::Tensor& packed_key,
    int symbol_dim) {
  const int batch = static_cast<int>(packed_key.size(0));
  const int num_heads = static_cast<int>(packed_key.size(1));
  const int seq_len = static_cast<int>(packed_key.size(2));
  auto offsets = torch::empty(
      {batch, num_heads, kAlphabetCapacity + 1},
      packed_key.options());
  auto occurrences = torch::empty_like(packed_key);
  const auto stream = at::cuda::getCurrentCUDAStream();
  build_occurrence_index_kernel<<<
      batch * num_heads,
      kIndexThreads,
      0,
      stream>>>(
      packed_key.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>(),
      occurrences.data_ptr<int32_t>(),
      seq_len,
      1 << symbol_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {offsets, occurrences};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_indexed_hard_forward_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& value,
    const torch::Tensor& offsets,
    const torch::Tensor& occurrences,
    int max_suffix_length,
    int method) {
  const int batch = static_cast<int>(packed_query.size(0));
  const int num_heads = static_cast<int>(packed_query.size(1));
  const int seq_len = static_cast<int>(packed_query.size(2));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  auto output = torch::empty(
      {batch, seq_len, num_heads, value_dim},
      value.options());
  auto selected_routes = torch::empty_like(packed_query);
  auto selected_lengths = torch::empty_like(packed_query);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = batch * num_heads * seq_len;
  DISPATCH_ROSA_FLOAT_TYPES(
      value.scalar_type(),
      "rosa_indexed_hard_forward",
      [&] {
        if (method == 0) {
          indexed_lane_hard_forward_kernel<scalar_t, false><<<
              blocks,
              kBlockThreads,
              0,
              stream>>>(
              packed_query.data_ptr<int32_t>(),
              packed_key.data_ptr<int32_t>(),
              value.data_ptr<scalar_t>(),
              offsets.data_ptr<int32_t>(),
              occurrences.data_ptr<int32_t>(),
              output.data_ptr<scalar_t>(),
              selected_routes.data_ptr<int32_t>(),
              selected_lengths.data_ptr<int32_t>(),
              seq_len,
              num_heads,
              num_value_heads,
              value_dim,
              max_suffix_length);
        } else if (method == 1) {
          indexed_lane_hard_forward_kernel<scalar_t, true><<<
              blocks,
              kBlockThreads,
              0,
              stream>>>(
              packed_query.data_ptr<int32_t>(),
              packed_key.data_ptr<int32_t>(),
              value.data_ptr<scalar_t>(),
              offsets.data_ptr<int32_t>(),
              occurrences.data_ptr<int32_t>(),
              output.data_ptr<scalar_t>(),
              selected_routes.data_ptr<int32_t>(),
              selected_lengths.data_ptr<int32_t>(),
              seq_len,
              num_heads,
              num_value_heads,
              value_dim,
              max_suffix_length);
        } else {
          indexed_hybrid_hard_forward_kernel<scalar_t><<<
              blocks,
              kBlockThreads,
              0,
              stream>>>(
              packed_query.data_ptr<int32_t>(),
              packed_key.data_ptr<int32_t>(),
              value.data_ptr<scalar_t>(),
              offsets.data_ptr<int32_t>(),
              occurrences.data_ptr<int32_t>(),
              output.data_ptr<scalar_t>(),
              selected_routes.data_ptr<int32_t>(),
              selected_lengths.data_ptr<int32_t>(),
              seq_len,
              num_heads,
              num_value_heads,
              value_dim,
              max_suffix_length);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, selected_routes, selected_lengths};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_diagonal_hard_forward_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& value,
    int max_suffix_length) {
  const int batch = static_cast<int>(packed_query.size(0));
  const int num_heads = static_cast<int>(packed_query.size(1));
  const int seq_len = static_cast<int>(packed_query.size(2));
  const int series_count = batch * num_heads;
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  auto winners = torch::zeros(
      {batch, num_heads, seq_len},
      packed_query.options().dtype(torch::kInt64));
  auto output = torch::empty(
      {batch, seq_len, num_heads, value_dim},
      value.options());
  auto selected_routes = torch::empty_like(packed_query);
  auto selected_lengths = torch::empty_like(packed_query);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (seq_len > 1) {
    const int64_t total_warps =
        static_cast<int64_t>(series_count) * (seq_len - 1);
    const int blocks = static_cast<int>(
        (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
    diagonal_winner_kernel<<<blocks, kBlockThreads, 0, stream>>>(
        packed_query.data_ptr<int32_t>(),
        packed_key.data_ptr<int32_t>(),
        winners.data_ptr<int64_t>(),
        seq_len,
        series_count,
        max_suffix_length);
  }
  DISPATCH_ROSA_FLOAT_TYPES(
      value.scalar_type(),
      "rosa_diagonal_hard_forward",
      [&] {
        gather_diagonal_winners_kernel<scalar_t><<<
            series_count * seq_len,
            kBlockThreads,
            0,
            stream>>>(
            winners.data_ptr<int64_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            selected_routes.data_ptr<int32_t>(),
            selected_lengths.data_ptr<int32_t>(),
            seq_len,
            num_heads,
            num_value_heads,
            value_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, selected_routes, selected_lengths};
}
