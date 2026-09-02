#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace {

constexpr int kHardWarpSize = 32;
constexpr int kHardWarpsPerBlock = 4;
constexpr int kHardBlockThreads = kHardWarpsPerBlock * kHardWarpSize;
constexpr int kTrajectoryWarpsPerBlock = 2;
constexpr int kTrajectoryBlockThreads =
    kTrajectoryWarpsPerBlock * kHardWarpSize;
constexpr int kIndexThreads = 256;
constexpr int kIndexWarps = kIndexThreads / kHardWarpSize;
constexpr int kCodeCapacity = 256;
constexpr int kCodeBits = 8;
constexpr int kParallelIndexMinSequenceLength = 8192;
constexpr int kIndexPositionsPerChunk = 2048;
constexpr int kMaxIndexChunksPerSeries = 32;
constexpr int kTrajectoryMinSequenceLength = 4096;
constexpr int kTrajectoryWorkersPerSeries = 256;
constexpr int kHeavyDiagonalMinSequenceLength = 8192;
constexpr int kHeavyRestartSamples = 64;
constexpr int kHeavyRestartMinimumCount = 64;


#define DISPATCH_ROSA_HARD_FLOAT_TYPES(TYPE, NAME, ...)         \
  AT_DISPATCH_SWITCH(                                           \
      TYPE,                                                     \
      NAME,                                                     \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))


template <typename scalar_t>
__device__ __forceinline__ float hard_read_float(
    const scalar_t* __restrict__ values,
    int64_t index) {
  return static_cast<float>(values[index]);
}


__device__ __forceinline__ int hard_ngram_code(
    const uint8_t* __restrict__ key,
    int position,
    int symbol_dim,
    int symbols_per_code) {
  int code = 0;
  for (int offset = symbols_per_code - 1; offset >= 0; --offset) {
    code = (code << symbol_dim) | key[position - offset];
  }
  return code;
}


__global__ void build_hard_occurrence_index_kernel(
    const uint8_t* __restrict__ packed_key,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ occurrences,
    int32_t* __restrict__ first_symbol_positions,
    int seq_len,
    int symbol_dim,
    int symbols_per_code) {
  __shared__ int counts[kCodeCapacity];
  __shared__ int cursors[kCodeCapacity];
  __shared__ int first_positions[kCodeCapacity];
  const int series = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kHardWarpSize - 1);
  const int warp = tid / kHardWarpSize;
  const uint8_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
  int32_t* series_offsets =
      offsets + static_cast<int64_t>(series) * (kCodeCapacity + 1);
  int32_t* series_occurrences =
      occurrences + static_cast<int64_t>(series) * seq_len;
  int32_t* series_first_positions = nullptr;
  if (first_symbol_positions != nullptr) {
    series_first_positions = first_symbol_positions +
        static_cast<int64_t>(series) * kCodeCapacity;
  }

  counts[tid] = 0;
  if (first_symbol_positions != nullptr) {
    first_positions[tid] = seq_len;
  }
  __syncthreads();
  for (int chunk = 0; chunk < seq_len; chunk += blockDim.x) {
    const int position = chunk + tid;
    const bool symbol_valid = position < seq_len;
    if (first_symbol_positions != nullptr) {
      const unsigned symbol_active = __ballot_sync(
          0xffffffffu,
          symbol_valid);
      if (symbol_valid) {
        const int symbol = key[position];
        const unsigned symbol_peers =
            __match_any_sync(symbol_active, symbol);
        const int symbol_leader = __ffs(symbol_peers) - 1;
        if (lane == symbol_leader) {
          atomicMin(&first_positions[symbol], position);
        }
      }
    }
    const bool valid = symbol_valid &&
        position + 1 >= symbols_per_code;
    const unsigned active = __ballot_sync(
        0xffffffffu,
        valid);
    if (valid) {
      const int code = hard_ngram_code(
          key,
          position,
          symbol_dim,
          symbols_per_code);
      const unsigned peers = __match_any_sync(active, code);
      const int leader = __ffs(peers) - 1;
      if (lane == leader) {
        atomicAdd(&counts[code], __popc(peers));
      }
    }
  }
  __syncthreads();

  if (first_symbol_positions != nullptr) {
    series_first_positions[tid] = first_positions[tid];
  }
  if (tid == 0) {
    int prefix = 0;
    for (int code = 0; code < kCodeCapacity; ++code) {
      series_offsets[code] = prefix;
      cursors[code] = prefix;
      prefix += counts[code];
    }
    series_offsets[kCodeCapacity] = prefix;
  }
  __syncthreads();

  for (int chunk = 0; chunk < seq_len; chunk += blockDim.x) {
    const int position = chunk + tid;
    const bool valid = position < seq_len &&
        position + 1 >= symbols_per_code;
    const unsigned active = __ballot_sync(
        0xffffffffu,
        valid);
    const int code = valid
        ? hard_ngram_code(
              key,
              position,
              symbol_dim,
              symbols_per_code)
        : 0;
    for (int active_warp = 0; active_warp < kIndexWarps; ++active_warp) {
      if (warp == active_warp && valid) {
        const unsigned peers = __match_any_sync(active, code);
        const int leader = __ffs(peers) - 1;
        int base = 0;
        if (lane == leader) {
          base = atomicAdd(&cursors[code], __popc(peers));
        }
        base = __shfl_sync(peers, base, leader);
        const unsigned preceding = peers & ((1u << lane) - 1u);
        series_occurrences[base + __popc(preceding)] = position;
      }
      __syncthreads();
    }
  }
}


__device__ __forceinline__ void hard_index_chunk_bounds(
    int item_count,
    int chunk,
    int chunks_per_series,
    int* begin,
    int* end) {
  *begin = static_cast<int>(
      (static_cast<int64_t>(item_count) * chunk) / chunks_per_series);
  *end = static_cast<int>(
      (static_cast<int64_t>(item_count) * (chunk + 1)) /
      chunks_per_series);
}


__global__ void count_hard_occurrence_chunks_kernel(
    const uint8_t* __restrict__ packed_key,
    int32_t* __restrict__ chunk_offsets,
    int32_t* __restrict__ first_symbol_positions,
    int seq_len,
    int chunks_per_series,
    int symbol_dim,
    int symbols_per_code) {
  __shared__ int counts[kCodeCapacity];
  __shared__ int first_positions[kCodeCapacity];
  const int series = blockIdx.x / chunks_per_series;
  const int chunk = blockIdx.x % chunks_per_series;
  const int tid = threadIdx.x;
  const int lane = tid & (kHardWarpSize - 1);
  const uint8_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
  int32_t* series_first_positions = nullptr;
  if (first_symbol_positions != nullptr) {
    series_first_positions = first_symbol_positions +
        static_cast<int64_t>(series) * kCodeCapacity;
  }

  counts[tid] = 0;
  first_positions[tid] = seq_len;
  __syncthreads();

  if (first_symbol_positions != nullptr) {
    int symbol_begin = 0;
    int symbol_end = 0;
    hard_index_chunk_bounds(
        seq_len,
        chunk,
        chunks_per_series,
        &symbol_begin,
        &symbol_end);
    for (int tile = symbol_begin; tile < symbol_end; tile += blockDim.x) {
      const int position = tile + tid;
      const bool valid = position < symbol_end;
      const unsigned active = __ballot_sync(0xffffffffu, valid);
      if (valid) {
        const int symbol = key[position];
        const unsigned peers = __match_any_sync(active, symbol);
        const int leader = __ffs(peers) - 1;
        if (lane == leader) {
          atomicMin(&first_positions[symbol], position);
        }
      }
    }
  }

  const int ngram_count = seq_len - symbols_per_code + 1;
  int ngram_begin = 0;
  int ngram_end = 0;
  hard_index_chunk_bounds(
      ngram_count,
      chunk,
      chunks_per_series,
      &ngram_begin,
      &ngram_end);
  ngram_begin += symbols_per_code - 1;
  ngram_end += symbols_per_code - 1;
  for (int tile = ngram_begin; tile < ngram_end; tile += blockDim.x) {
    const int position = tile + tid;
    const bool valid = position < ngram_end;
    const unsigned active = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      const int code = hard_ngram_code(
          key,
          position,
          symbol_dim,
          symbols_per_code);
      const unsigned peers = __match_any_sync(active, code);
      const int leader = __ffs(peers) - 1;
      if (lane == leader) {
        atomicAdd(&counts[code], __popc(peers));
      }
    }
  }
  __syncthreads();

  const int64_t chunk_index =
      (static_cast<int64_t>(series) * chunks_per_series + chunk) *
      kCodeCapacity;
  chunk_offsets[chunk_index + tid] = counts[tid];
  if (series_first_positions != nullptr &&
      first_positions[tid] < seq_len) {
    atomicMin(
        &series_first_positions[tid],
        first_positions[tid]);
  }
}


__global__ void prefix_hard_occurrence_chunks_kernel(
    int32_t* __restrict__ chunk_offsets,
    int32_t* __restrict__ offsets,
    int chunks_per_series) {
  __shared__ int warp_totals[kIndexWarps];
  const int series = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kHardWarpSize - 1);
  const int warp = tid / kHardWarpSize;
  int32_t* series_chunks = chunk_offsets +
      static_cast<int64_t>(series) * chunks_per_series * kCodeCapacity;
  int32_t* series_offsets = offsets +
      static_cast<int64_t>(series) * (kCodeCapacity + 1);

  int total = 0;
  for (int chunk = 0; chunk < chunks_per_series; ++chunk) {
    total += series_chunks[chunk * kCodeCapacity + tid];
  }
  int inclusive = total;
#pragma unroll
  for (int delta = 1; delta < kHardWarpSize; delta <<= 1) {
    const int preceding = __shfl_up_sync(
        0xffffffffu,
        inclusive,
        delta);
    if (lane >= delta) {
      inclusive += preceding;
    }
  }
  if (lane == kHardWarpSize - 1) {
    warp_totals[warp] = inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    int warp_total = lane < kIndexWarps ? warp_totals[lane] : 0;
#pragma unroll
    for (int delta = 1; delta < kHardWarpSize; delta <<= 1) {
      const int preceding = __shfl_up_sync(
          0xffffffffu,
          warp_total,
          delta);
      if (lane >= delta) {
        warp_total += preceding;
      }
    }
    if (lane < kIndexWarps) {
      warp_totals[lane] = warp_total;
    }
  }
  __syncthreads();
  const int warp_base = warp == 0 ? 0 : warp_totals[warp - 1];
  const int code_base = warp_base + inclusive - total;
  series_offsets[tid] = code_base;
  if (tid == kCodeCapacity - 1) {
    series_offsets[kCodeCapacity] = code_base + total;
  }

  int chunk_base = code_base;
  for (int chunk = 0; chunk < chunks_per_series; ++chunk) {
    const int index = chunk * kCodeCapacity + tid;
    const int count = series_chunks[index];
    series_chunks[index] = chunk_base;
    chunk_base += count;
  }
}


__global__ void scatter_hard_occurrence_chunks_kernel(
    const uint8_t* __restrict__ packed_key,
    const int32_t* __restrict__ chunk_offsets,
    int32_t* __restrict__ occurrences,
    int seq_len,
    int chunks_per_series,
    int symbol_dim,
    int symbols_per_code) {
  __shared__ int cursors[kCodeCapacity];
  const int series = blockIdx.x / chunks_per_series;
  const int chunk = blockIdx.x % chunks_per_series;
  const int tid = threadIdx.x;
  const int lane = tid & (kHardWarpSize - 1);
  const int warp = tid / kHardWarpSize;
  const uint8_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
  int32_t* series_occurrences =
      occurrences + static_cast<int64_t>(series) * seq_len;
  const int64_t chunk_index =
      (static_cast<int64_t>(series) * chunks_per_series + chunk) *
      kCodeCapacity;
  cursors[tid] = chunk_offsets[chunk_index + tid];
  __syncthreads();

  const int ngram_count = seq_len - symbols_per_code + 1;
  int ngram_begin = 0;
  int ngram_end = 0;
  hard_index_chunk_bounds(
      ngram_count,
      chunk,
      chunks_per_series,
      &ngram_begin,
      &ngram_end);
  ngram_begin += symbols_per_code - 1;
  ngram_end += symbols_per_code - 1;
  for (int tile = ngram_begin; tile < ngram_end; tile += blockDim.x) {
    const int position = tile + tid;
    const bool valid = position < ngram_end;
    const unsigned active = __ballot_sync(0xffffffffu, valid);
    const int code = valid
        ? hard_ngram_code(
              key,
              position,
              symbol_dim,
              symbols_per_code)
        : 0;
    for (int active_warp = 0; active_warp < kIndexWarps; ++active_warp) {
      if (warp == active_warp && valid) {
        const unsigned peers = __match_any_sync(active, code);
        const int leader = __ffs(peers) - 1;
        int base = 0;
        if (lane == leader) {
          base = atomicAdd(&cursors[code], __popc(peers));
        }
        base = __shfl_sync(peers, base, leader);
        const unsigned preceding = peers & ((1u << lane) - 1u);
        series_occurrences[base + __popc(preceding)] = position;
      }
      __syncthreads();
    }
  }
}


__device__ __forceinline__ int hard_upper_bound(
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


__device__ __forceinline__ int latest_ngram_candidate_end(
    const uint8_t* __restrict__ query,
    const int32_t* __restrict__ series_offsets,
    const int32_t* __restrict__ series_occurrences,
    int row,
    int symbol_dim,
    int symbols_per_code) {
  if (row + 1 < symbols_per_code) {
    return -1;
  }
  const int code = hard_ngram_code(
      query,
      row,
      symbol_dim,
      symbols_per_code);
  const int begin = series_offsets[code];
  const int end = series_offsets[code + 1];
  const int upper = hard_upper_bound(
      series_occurrences,
      begin,
      end,
      row - 1);
  return upper > begin ? series_occurrences[upper - 1] : -1;
}


__global__ void candidate_trajectory_local_scan_kernel(
    const uint8_t* __restrict__ packed_query,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ occurrences,
    int32_t* __restrict__ latest_ends,
    int32_t* __restrict__ segment_starts,
    int seq_len,
    int chunks_per_series,
    int symbol_dim,
    int symbols_per_code) {
  __shared__ int tile_candidates[kIndexThreads];
  __shared__ int warp_prefixes[kIndexWarps];
  const int series = blockIdx.x / chunks_per_series;
  const int chunk = blockIdx.x % chunks_per_series;
  const int tid = threadIdx.x;
  const int lane = tid & (kHardWarpSize - 1);
  const int warp = tid / kHardWarpSize;
  const int row = chunk * kIndexThreads + tid;
  const uint8_t* query =
      packed_query + static_cast<int64_t>(series) * seq_len;
  const int32_t* series_offsets =
      offsets + static_cast<int64_t>(series) * (kCodeCapacity + 1);
  const int32_t* series_occurrences =
      occurrences + static_cast<int64_t>(series) * seq_len;
  int32_t* series_latest =
      latest_ends + static_cast<int64_t>(series) * seq_len;
  int32_t* series_starts =
      segment_starts + static_cast<int64_t>(series) * seq_len;
  const bool valid = row < seq_len;
  const int candidate = valid
      ? latest_ngram_candidate_end(
            query,
            series_offsets,
            series_occurrences,
            row,
            symbol_dim,
            symbols_per_code)
      : -1;
  tile_candidates[tid] = candidate;
  __syncthreads();
  int previous = -1;
  if (valid && row > 0) {
    previous = tid > 0
        ? tile_candidates[tid - 1]
        : latest_ngram_candidate_end(
              query,
              series_offsets,
              series_occurrences,
              row - 1,
              symbol_dim,
              symbols_per_code);
  }
  if (valid) {
    series_latest[row] = candidate;
  }
  const bool starts_segment = valid &&
      (candidate < 0 || row == 0 || previous < 0 ||
       previous != candidate - 1);
  int segment_start = starts_segment ? row : -1;
#pragma unroll
  for (int offset = 1; offset < kHardWarpSize; offset <<= 1) {
    const int preceding = __shfl_up_sync(
        0xffffffffu,
        segment_start,
        offset);
    if (lane >= offset) {
      segment_start = max(segment_start, preceding);
    }
  }
  if (lane == kHardWarpSize - 1) {
    warp_prefixes[warp] = segment_start;
  }
  __syncthreads();
  if (warp == 0) {
    int warp_start = lane < kIndexWarps ? warp_prefixes[lane] : -1;
#pragma unroll
    for (int offset = 1; offset < kHardWarpSize; offset <<= 1) {
      const int preceding = __shfl_up_sync(
          0xffffffffu,
          warp_start,
          offset);
      if (lane >= offset) {
        warp_start = max(warp_start, preceding);
      }
    }
    if (lane < kIndexWarps) {
      warp_prefixes[lane] = warp_start;
    }
  }
  __syncthreads();
  if (warp > 0) {
    segment_start = max(segment_start, warp_prefixes[warp - 1]);
  }
  if (valid) {
    series_starts[row] = segment_start;
  }
}


__global__ void candidate_segment_carry_kernel(
    int32_t* __restrict__ segment_starts,
    int seq_len,
    int chunks_per_series) {
  const int series = blockIdx.x;
  const int lane = threadIdx.x;
  int32_t* series_starts =
      segment_starts + static_cast<int64_t>(series) * seq_len;
  int carried_start = -1;
  for (int chunk_base = 0;
       chunk_base < chunks_per_series;
       chunk_base += kHardWarpSize) {
    const int chunk = chunk_base + lane;
    const bool valid = chunk < chunks_per_series;
    const int last_row = valid
        ? min((chunk + 1) * kIndexThreads, seq_len) - 1
        : 0;
    int segment_start = valid ? series_starts[last_row] : -1;
    segment_start = max(segment_start, carried_start);
#pragma unroll
    for (int offset = 1; offset < kHardWarpSize; offset <<= 1) {
      const int preceding = __shfl_up_sync(
          0xffffffffu,
          segment_start,
          offset);
      if (lane >= offset) {
        segment_start = max(segment_start, preceding);
      }
    }
    if (valid) {
      series_starts[last_row] = segment_start;
    }
    const int last_lane = min(
        kHardWarpSize - 1,
        chunks_per_series - chunk_base - 1);
    carried_start = __shfl_sync(
        0xffffffffu,
        segment_start,
        last_lane);
  }
}


__device__ __forceinline__ int exact_indexed_suffix_length(
    const uint8_t* __restrict__ query,
    const uint8_t* __restrict__ key,
    int row,
    int route,
    int known_prefix) {
  const int suffix_steps = min(row + 1, route);
  for (int offset = known_prefix; offset < suffix_steps; ++offset) {
    if (query[row - offset] != key[route - 1 - offset]) {
      return offset;
    }
  }
  return suffix_steps;
}


__device__ __forceinline__ int certificate_suffix_length(
    const uint8_t* __restrict__ query,
    const uint8_t* __restrict__ key,
    int row,
    int route,
    int known_prefix);


__global__ void select_heavy_restart_delta_kernel(
    const uint8_t* __restrict__ packed_query,
    const uint8_t* __restrict__ packed_key,
    const int32_t* __restrict__ latest_ends,
    const int32_t* __restrict__ segment_starts,
    int32_t* __restrict__ heavy_deltas,
    int seq_len) {
  __shared__ int sampled_deltas[kHeavyRestartSamples];
  __shared__ int warp_mode_deltas[kIndexWarps];
  __shared__ int warp_mode_counts[kIndexWarps];
  __shared__ int exact_count;
  __shared__ int restart_count;
  __shared__ int diagonal_match_count;
  const int series = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kHardWarpSize - 1);
  const int warp = tid / kHardWarpSize;
  const int64_t series_base = static_cast<int64_t>(series) * seq_len;
  const uint8_t* query = packed_query + series_base;
  const uint8_t* key = packed_key + series_base;
  const int32_t* series_latest = latest_ends + series_base;
  const int32_t* series_starts = segment_starts + series_base;

  if (tid < kHeavyRestartSamples) {
    int sample_begin = 0;
    int sample_end = 0;
    hard_index_chunk_bounds(
        seq_len,
        tid,
        kHeavyRestartSamples,
        &sample_begin,
        &sample_end);
    const int sample_size = sample_end - sample_begin;
    const int sample_offset =
        (tid * 37 + series * 17) % sample_size;
    const int row = sample_begin + sample_offset;
    const int latest_end = series_latest[row];
    const int sampled_delta =
        series_starts[row] == row && latest_end >= 0
        ? row - latest_end
        : -1;
    sampled_deltas[tid] = sampled_delta;
  }
  __syncthreads();

  int mode_delta = tid < kHeavyRestartSamples
      ? sampled_deltas[tid]
      : -1;
  int mode_count = 0;
  if (mode_delta > 0) {
#pragma unroll
    for (int sample = 0; sample < kHeavyRestartSamples; ++sample) {
      mode_count += sampled_deltas[sample] == mode_delta;
    }
  }
#pragma unroll
  for (int offset = kHardWarpSize / 2; offset > 0; offset >>= 1) {
    const int other_delta = __shfl_down_sync(
        0xffffffffu,
        mode_delta,
        offset);
    const int other_count = __shfl_down_sync(
        0xffffffffu,
        mode_count,
        offset);
    if (lane + offset < kHardWarpSize &&
        (other_count > mode_count ||
         (other_count == mode_count &&
          other_count > 0 &&
          other_delta < mode_delta))) {
      mode_delta = other_delta;
      mode_count = other_count;
    }
  }
  if (lane == 0) {
    warp_mode_deltas[warp] = mode_delta;
    warp_mode_counts[warp] = mode_count;
  }
  __syncthreads();
  if (warp == 0) {
    mode_delta = lane < kIndexWarps ? warp_mode_deltas[lane] : -1;
    mode_count = lane < kIndexWarps ? warp_mode_counts[lane] : 0;
#pragma unroll
    for (int offset = kHardWarpSize / 2; offset > 0; offset >>= 1) {
      const int other_delta = __shfl_down_sync(
          0xffffffffu,
          mode_delta,
          offset);
      const int other_count = __shfl_down_sync(
          0xffffffffu,
          mode_count,
          offset);
      if (lane + offset < kHardWarpSize &&
          (other_count > mode_count ||
           (other_count == mode_count &&
            other_count > 0 &&
            other_delta < mode_delta))) {
        mode_delta = other_delta;
        mode_count = other_count;
      }
    }
    if (lane == 0) {
      sampled_deltas[0] = mode_count > 0 ? mode_delta : -1;
    }
  }
  __syncthreads();
  const int sampled_delta = sampled_deltas[0];
  if (sampled_delta < 0) {
    if (tid == 0) {
      heavy_deltas[series] = -1;
    }
    return;
  }

  if (tid == 0) {
    exact_count = 0;
    restart_count = 0;
  }
  __syncthreads();

  int local_restart_count = 0;
  int local_exact_count = 0;
  for (int row = tid; row < seq_len; row += blockDim.x) {
    const int latest_end = series_latest[row];
    if (series_starts[row] != row || latest_end < 0) {
      continue;
    }
    ++local_restart_count;
    const int delta = row - latest_end;
    local_exact_count += delta == sampled_delta;
  }
#pragma unroll
  for (int offset = kHardWarpSize / 2; offset > 0; offset >>= 1) {
    local_restart_count += __shfl_down_sync(
        0xffffffffu,
        local_restart_count,
        offset);
    local_exact_count += __shfl_down_sync(
        0xffffffffu,
        local_exact_count,
        offset);
  }
  if (lane == 0) {
    atomicAdd(&restart_count, local_restart_count);
    atomicAdd(&exact_count, local_exact_count);
  }
  __syncthreads();

  if (tid == 0) {
    const int sequence_minimum =
        (seq_len + 64 - 1) / 64;
    const int minimum_count = max(
        kHeavyRestartMinimumCount,
        sequence_minimum);
    sampled_deltas[0] =
        exact_count >= minimum_count ? sampled_delta : -1;
  }
  __syncthreads();
  if (sampled_deltas[0] < 0) {
    if (tid == 0) {
      heavy_deltas[series] = -1;
    }
    return;
  }

  if (static_cast<int64_t>(exact_count) * 8 < restart_count) {
    if (tid == 0) {
      diagonal_match_count = 0;
    }
    __syncthreads();
    int local_match_count = 0;
    for (int row = sampled_delta + tid;
         row < seq_len;
         row += blockDim.x) {
      local_match_count +=
          query[row] == key[row - sampled_delta];
    }
#pragma unroll
    for (int offset = kHardWarpSize / 2; offset > 0; offset >>= 1) {
      local_match_count += __shfl_down_sync(
          0xffffffffu,
          local_match_count,
          offset);
    }
    if (lane == 0) {
      atomicAdd(&diagonal_match_count, local_match_count);
    }
    __syncthreads();
    if (tid == 0 &&
        static_cast<int64_t>(diagonal_match_count) * 8 <
            static_cast<int64_t>(seq_len - sampled_delta) * 7) {
      sampled_deltas[0] = -1;
    }
    __syncthreads();
  }
  const int heavy_delta = sampled_deltas[0];
  if (tid == 0) {
    heavy_deltas[series] = heavy_delta;
  }
}


__global__ void build_heavy_diagonal_mismatch_groups_kernel(
    const uint8_t* __restrict__ packed_query,
    const uint8_t* __restrict__ packed_key,
    const int32_t* __restrict__ heavy_deltas,
    int32_t* __restrict__ heavy_mismatch_prefixes,
    int seq_len,
    int mismatch_groups_per_series,
    int64_t total_groups) {
  const int warp = threadIdx.x / kHardWarpSize;
  const int lane = threadIdx.x & (kHardWarpSize - 1);
  const int64_t linear_group =
      static_cast<int64_t>(blockIdx.x) * kHardWarpsPerBlock + warp;
  if (linear_group >= total_groups) {
    return;
  }
  const int group = static_cast<int>(
      linear_group % mismatch_groups_per_series);
  const int series = static_cast<int>(
      linear_group / mismatch_groups_per_series);
  const int heavy_delta = heavy_deltas[series];
  if (heavy_delta < 0) {
    return;
  }
  const int row = group * kHardWarpSize + lane;
  const int64_t series_base = static_cast<int64_t>(series) * seq_len;
  const uint8_t* query = packed_query + series_base;
  const uint8_t* key = packed_key + series_base;
  const bool mismatch = row < seq_len &&
      (row < heavy_delta || query[row] != key[row - heavy_delta]);
  const unsigned mismatches = __ballot_sync(0xffffffffu, mismatch);
  if (lane == 0) {
    const int last_mismatch = mismatches == 0u
        ? -1
        : group * kHardWarpSize +
            (kHardWarpSize - 1 - __clz(mismatches));
    heavy_mismatch_prefixes[linear_group] = last_mismatch;
  }
}


__global__ void carry_heavy_diagonal_mismatch_groups_kernel(
    const int32_t* __restrict__ heavy_deltas,
    int32_t* __restrict__ heavy_mismatch_prefixes,
    int mismatch_groups_per_series) {
  const int series = blockIdx.x;
  const int lane = threadIdx.x;
  if (heavy_deltas[series] < 0) {
    return;
  }
  int32_t* series_prefixes = heavy_mismatch_prefixes +
      static_cast<int64_t>(series) * mismatch_groups_per_series;
  int carried_mismatch = -1;
  for (int group_base = 0;
       group_base < mismatch_groups_per_series;
       group_base += kHardWarpSize) {
    const int group = group_base + lane;
    const bool valid = group < mismatch_groups_per_series;
    int last_mismatch = valid ? series_prefixes[group] : -1;
    last_mismatch = max(last_mismatch, carried_mismatch);
#pragma unroll
    for (int offset = 1; offset < kHardWarpSize; offset <<= 1) {
      const int preceding = __shfl_up_sync(
          0xffffffffu,
          last_mismatch,
          offset);
      if (lane >= offset) {
        last_mismatch = max(last_mismatch, preceding);
      }
    }
    if (valid) {
      series_prefixes[group] = last_mismatch;
    }
    const int last_lane = min(
        kHardWarpSize - 1,
        mismatch_groups_per_series - group_base - 1);
    carried_mismatch = __shfl_sync(
        0xffffffffu,
        last_mismatch,
        last_lane);
  }
}


__device__ __forceinline__ int heavy_diagonal_suffix_length(
    const uint8_t* __restrict__ query,
    const uint8_t* __restrict__ key,
    const int32_t* __restrict__ mismatch_prefixes,
    int row,
    int heavy_delta) {
  const int lane = threadIdx.x & (kHardWarpSize - 1);
  const int group_start =
      (row / kHardWarpSize) * kHardWarpSize;
  const int position = row - lane;
  const bool mismatch =
      position >= group_start &&
      (position < heavy_delta ||
       query[position] != key[position - heavy_delta]);
  const unsigned mismatches = __ballot_sync(0xffffffffu, mismatch);
  int length = 0;
  if (lane == 0) {
    if (mismatches != 0u) {
      length = __ffs(mismatches) - 1;
    } else {
      const int group = row / kHardWarpSize;
      const int previous_mismatch =
          group > 0 ? mismatch_prefixes[group - 1] : -1;
      length = row - previous_mismatch;
    }
  }
  return __shfl_sync(0xffffffffu, length, 0);
}


__global__ void candidate_segment_base_kernel(
    const uint8_t* __restrict__ packed_query,
    const uint8_t* __restrict__ packed_key,
    int32_t* __restrict__ latest_ends_and_base_lengths,
    const int32_t* __restrict__ segment_starts,
    const int32_t* __restrict__ heavy_deltas,
    const int32_t* __restrict__ heavy_mismatch_prefixes,
    int64_t total_rows,
    int seq_len,
    int mismatch_groups_per_series,
    int symbols_per_code) {
  const int warp = threadIdx.x / kHardWarpSize;
  const int lane = threadIdx.x & (kHardWarpSize - 1);
  for (int64_t linear_row =
           static_cast<int64_t>(blockIdx.x) *
               kTrajectoryWarpsPerBlock +
           warp;
       linear_row < total_rows;
       linear_row +=
           static_cast<int64_t>(gridDim.x) *
           kTrajectoryWarpsPerBlock) {
    const int row = static_cast<int>(linear_row % seq_len);
    const int latest_end = latest_ends_and_base_lengths[linear_row];
    if (segment_starts[linear_row] != row || latest_end < 0) {
      continue;
    }
    const int series = static_cast<int>(linear_row / seq_len);
    const uint8_t* query =
        packed_query + static_cast<int64_t>(series) * seq_len;
    const uint8_t* key =
        packed_key + static_cast<int64_t>(series) * seq_len;
    const int route = latest_end + 1;
    const int heavy_delta = heavy_deltas != nullptr
        ? heavy_deltas[series]
        : -1;
    const int length =
        heavy_delta > 0 && row - latest_end == heavy_delta
        ? heavy_diagonal_suffix_length(
              query,
              key,
              heavy_mismatch_prefixes +
                  static_cast<int64_t>(series) *
                      mismatch_groups_per_series,
              row,
              heavy_delta)
        : certificate_suffix_length(
              query,
              key,
              row,
              route,
              symbols_per_code);
    if (lane == 0) {
      latest_ends_and_base_lengths[linear_row] = length;
    }
  }
}


__device__ __forceinline__ int certificate_suffix_length(
    const uint8_t* __restrict__ query,
    const uint8_t* __restrict__ key,
    int row,
    int route,
    int known_prefix) {
  const int lane = threadIdx.x & (kHardWarpSize - 1);
  const int suffix_steps = min(row + 1, route);
  int length = suffix_steps;
  int chunk = known_prefix;
  if (chunk < suffix_steps) {
    const int offset = chunk + lane;
    const bool mismatch = offset < suffix_steps &&
        query[row - offset] != key[route - 1 - offset];
    const unsigned mismatches = __ballot_sync(0xffffffffu, mismatch);
    if (mismatches != 0u) {
      length = chunk + __ffs(mismatches) - 1;
      return __shfl_sync(0xffffffffu, length, 0);
    }
    chunk += kHardWarpSize;
  }
  constexpr int kSymbolsPerLane = 8;
  for (; chunk < suffix_steps;
       chunk += kSymbolsPerLane * kHardWarpSize) {
    unsigned lane_mismatches = 0u;
#pragma unroll
    for (int symbol = 0; symbol < kSymbolsPerLane; ++symbol) {
      const int offset =
          chunk + lane * kSymbolsPerLane + symbol;
      const bool mismatch = offset < suffix_steps &&
          query[row - offset] != key[route - 1 - offset];
      if (mismatch) {
        lane_mismatches |= 1u << symbol;
      }
    }
    const unsigned mismatch_lanes = __ballot_sync(
        0xffffffffu,
        lane_mismatches != 0u);
    if (mismatch_lanes != 0u) {
      const int first_lane = __ffs(mismatch_lanes) - 1;
      const unsigned first_lane_mismatches = __shfl_sync(
          0xffffffffu,
          lane_mismatches,
          first_lane);
      length = chunk + first_lane * kSymbolsPerLane +
          __ffs(first_lane_mismatches) - 1;
      return __shfl_sync(0xffffffffu, length, 0);
    }
  }
  return __shfl_sync(0xffffffffu, length, 0);
}


__device__ __forceinline__ void merge_indexed_route(
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


__device__ __forceinline__ void reduce_indexed_route_in_warp(
    int& length,
    int& route) {
  const int lane = threadIdx.x & (kHardWarpSize - 1);
#pragma unroll
  for (int offset = kHardWarpSize / 2; offset > 0; offset >>= 1) {
    const int other_length =
        __shfl_down_sync(0xffffffffu, length, offset);
    const int other_route =
        __shfl_down_sync(0xffffffffu, route, offset);
    if (lane + offset < kHardWarpSize) {
      merge_indexed_route(other_length, other_route, length, route);
    }
  }
}


template <typename scalar_t>
__device__ __forceinline__ void write_indexed_value(
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    int batch,
    int row,
    int head,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int route) {
  const int lane = threadIdx.x & (kHardWarpSize - 1);
  const int value_head = head / (num_heads / num_value_heads);
  for (int feature = lane;
       feature < value_dim;
       feature += kHardWarpSize) {
    float result = 0.0f;
    if (route > 0) {
      const int64_t value_index =
          ((static_cast<int64_t>(batch) * seq_len + route) *
               num_value_heads +
           value_head) *
              value_dim +
          feature;
      result = hard_read_float(value, value_index) > 0.0f ? 1.0f : -1.0f;
    }
    const int64_t output_index =
        ((static_cast<int64_t>(batch) * seq_len + row) * num_heads + head) *
            value_dim +
        feature;
    output[output_index] = static_cast<scalar_t>(result);
  }
}


template <typename scalar_t>
__global__ void indexed_hard_forward_kernel(
    const uint8_t* __restrict__ packed_query,
    const uint8_t* __restrict__ packed_key,
    const scalar_t* __restrict__ value,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ occurrences,
    const int32_t* __restrict__ first_symbol_positions,
    const int32_t* __restrict__ trajectory_segment_starts,
    const int32_t* __restrict__ trajectory_base_lengths,
    const int32_t* __restrict__ trajectory_heavy_deltas,
    const int32_t* __restrict__ trajectory_heavy_mismatch_prefixes,
    scalar_t* __restrict__ output,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int symbol_dim,
    int symbols_per_code,
    int mismatch_groups_per_series,
    int64_t total_rows) {
  const int warp = threadIdx.x / kHardWarpSize;
  const int64_t linear_row =
      static_cast<int64_t>(blockIdx.x) * kHardWarpsPerBlock + warp;
  if (linear_row >= total_rows) {
    return;
  }
  const int row = static_cast<int>(linear_row % seq_len);
  const int head = static_cast<int>(
      (linear_row / seq_len) % num_heads);
  const int batch = static_cast<int>(
      linear_row / (static_cast<int64_t>(seq_len) * num_heads));
  const int series = batch * num_heads + head;
  const uint8_t* query =
      packed_query + static_cast<int64_t>(series) * seq_len;
  const uint8_t* key =
      packed_key + static_cast<int64_t>(series) * seq_len;
  const int32_t* series_offsets =
      offsets + static_cast<int64_t>(series) * (kCodeCapacity + 1);
  const int32_t* series_occurrences =
      occurrences + static_cast<int64_t>(series) * seq_len;
  const int32_t* series_first_positions = nullptr;
  if (first_symbol_positions != nullptr) {
    series_first_positions = first_symbol_positions +
        static_cast<int64_t>(series) * kCodeCapacity;
  }
  const int lane = threadIdx.x & (kHardWarpSize - 1);

  int begin = 0;
  int upper = row;
  int known_prefix = 0;
  if (lane == 0) {
    if (row + 1 >= symbols_per_code) {
      const int code = hard_ngram_code(
          query,
          row,
          symbol_dim,
          symbols_per_code);
      const int code_begin = series_offsets[code];
      const int code_end = series_offsets[code + 1];
      const int code_upper = hard_upper_bound(
          series_occurrences,
          code_begin,
          code_end,
          row - 1);
      if (code_upper > code_begin) {
        begin = code_begin;
        upper = code_upper;
        known_prefix = symbols_per_code;
      }
    }
    if (known_prefix == 0 &&
        (symbols_per_code == 1 ||
         series_first_positions[query[row]] >= row)) {
      upper = 0;
    }
  }
  begin = __shfl_sync(0xffffffffu, begin, 0);
  upper = __shfl_sync(0xffffffffu, upper, 0);
  known_prefix = __shfl_sync(0xffffffffu, known_prefix, 0);
  const int alphabet_size = known_prefix > 0
      ? 1 << (symbol_dim * known_prefix)
      : 1;
  int best_length = 0;
  int best_route = 0;
  if (upper > begin) {
    best_route = known_prefix > 0
        ? series_occurrences[upper - 1] + 1
        : upper;
    if (trajectory_segment_starts != nullptr && known_prefix > 0) {
      const int64_t trajectory_index =
          static_cast<int64_t>(series) * seq_len + row;
      int segment_start =
          trajectory_segment_starts[trajectory_index];
      if (segment_start < 0) {
        const int previous_chunk_last =
            (row / kIndexThreads) * kIndexThreads - 1;
        segment_start = trajectory_segment_starts[
            static_cast<int64_t>(series) * seq_len +
            previous_chunk_last];
      }
      const int base_length =
          trajectory_base_lengths[
              static_cast<int64_t>(series) * seq_len + segment_start];
      const int trajectory_length =
          base_length + row - segment_start;
      best_length = min(
          trajectory_length,
          min(row + 1, best_route));
    } else {
      best_length = certificate_suffix_length(
          query,
          key,
          row,
          best_route,
          known_prefix);
    }
    if (best_length == 0) {
      best_route = 0;
    }
  }
  if (trajectory_heavy_deltas != nullptr) {
    const int heavy_delta = trajectory_heavy_deltas[series];
    if (heavy_delta > 0 && row >= heavy_delta) {
      const int heavy_length = heavy_diagonal_suffix_length(
          query,
          key,
          trajectory_heavy_mismatch_prefixes +
              static_cast<int64_t>(series) *
                  mismatch_groups_per_series,
          row,
          heavy_delta);
      merge_indexed_route(
          heavy_length,
          row - heavy_delta + 1,
          best_length,
          best_route);
    }
  }
  const int second_index = upper - 2;
  const int second_route = second_index >= begin
      ? (known_prefix > 0
             ? series_occurrences[second_index] + 1
             : second_index + 1)
      : 0;
  if (second_route <= best_length) {
    write_indexed_value(
        value,
        output,
        batch,
        row,
        head,
        seq_len,
        num_heads,
        num_value_heads,
        value_dim,
        best_route);
    return;
  }

  int first_unprobed = second_index;
  const int expected_collision_count =
      (2 * (row + 1) + alphabet_size - 1) / alphabet_size;
  const int additional_probe_threshold =
      expected_collision_count > 8 ? expected_collision_count : 8;
  if (known_prefix > 0 &&
      upper - begin >= additional_probe_threshold) {
    constexpr int kAdditionalProbes = 3;
    int probe_index = second_index;
    int probe_best_length = best_length;
    int probe_best_route = best_route;
    bool resolved = false;
#pragma unroll
    for (int probe = 0; probe < kAdditionalProbes; ++probe) {
      if (probe_index < begin) {
        resolved = true;
        break;
      }
      const int probe_route = series_occurrences[probe_index] + 1;
      const int probe_length = certificate_suffix_length(
          query,
          key,
          row,
          probe_route,
          known_prefix);
      merge_indexed_route(
          probe_length,
          probe_route,
          probe_best_length,
          probe_best_route);
      --probe_index;
      const int next_route = probe_index >= begin
          ? series_occurrences[probe_index] + 1
          : 0;
      if (next_route <= probe_best_length) {
        resolved = true;
        break;
      }
    }
    best_length = probe_best_length;
    best_route = probe_best_route;
    first_unprobed = probe_index;
    if (resolved) {
      write_indexed_value(
          value,
          output,
          batch,
          row,
          head,
          seq_len,
          num_heads,
          num_value_heads,
          value_dim,
          best_route);
      return;
    }
  }

  for (int occurrence_index = first_unprobed - lane;
       occurrence_index >= begin;
       occurrence_index -= kHardWarpSize) {
    const int route = known_prefix > 0
        ? series_occurrences[occurrence_index] + 1
        : occurrence_index + 1;
    if (route <= best_length) {
      break;
    }
    const int length = exact_indexed_suffix_length(
        query,
        key,
        row,
        route,
        known_prefix);
    merge_indexed_route(length, route, best_length, best_route);
  }

  reduce_indexed_route_in_warp(best_length, best_route);
  best_route = __shfl_sync(0xffffffffu, best_route, 0);
  write_indexed_value(
      value,
      output,
      batch,
      row,
      head,
      seq_len,
      num_heads,
      num_value_heads,
      value_dim,
      best_route);
}

}  // namespace


void rosa_soft_hard_index_cuda(
    const torch::Tensor& query_codes,
    const torch::Tensor& key_codes,
    const torch::Tensor& value,
    torch::Tensor& output,
    int symbol_dim) {
  const c10::cuda::CUDAGuard device_guard(query_codes.device());
  const int batch = static_cast<int>(query_codes.size(0));
  const int num_heads = static_cast<int>(query_codes.size(1));
  const int seq_len = static_cast<int>(query_codes.size(2));
  const int num_value_heads = static_cast<int>(value.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  const int symbols_per_code = kCodeBits / symbol_dim;
  auto index_options = query_codes.options().dtype(torch::kInt32);
  auto offsets = torch::empty(
      {batch, num_heads, kCodeCapacity + 1},
      index_options);
  auto occurrences = torch::empty(
      {batch, num_heads, seq_len},
      index_options);
  torch::Tensor first_symbol_positions;
  int32_t* first_symbol_positions_ptr = nullptr;
  if (symbols_per_code > 1) {
    first_symbol_positions = torch::empty(
        {batch, num_heads, kCodeCapacity},
        index_options);
    first_symbol_positions_ptr =
        first_symbol_positions.data_ptr<int32_t>();
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_series = batch * num_heads;
  const int index_chunks = min(
      kMaxIndexChunksPerSeries,
      (seq_len + kIndexPositionsPerChunk - 1) /
          kIndexPositionsPerChunk);
  if (seq_len >= kParallelIndexMinSequenceLength && index_chunks > 1) {
    if (first_symbol_positions_ptr != nullptr) {
      first_symbol_positions.fill_(seq_len);
    }
    {
      auto chunk_offsets = torch::empty(
          {num_series, index_chunks, kCodeCapacity},
          index_options);
      count_hard_occurrence_chunks_kernel<<<
          num_series * index_chunks,
          kIndexThreads,
          0,
          stream>>>(
          key_codes.data_ptr<uint8_t>(),
          chunk_offsets.data_ptr<int32_t>(),
          first_symbol_positions_ptr,
          seq_len,
          index_chunks,
          symbol_dim,
          symbols_per_code);
      prefix_hard_occurrence_chunks_kernel<<<
          num_series,
          kIndexThreads,
          0,
          stream>>>(
          chunk_offsets.data_ptr<int32_t>(),
          offsets.data_ptr<int32_t>(),
          index_chunks);
      scatter_hard_occurrence_chunks_kernel<<<
          num_series * index_chunks,
          kIndexThreads,
          0,
          stream>>>(
          key_codes.data_ptr<uint8_t>(),
          chunk_offsets.data_ptr<int32_t>(),
          occurrences.data_ptr<int32_t>(),
          seq_len,
          index_chunks,
          symbol_dim,
          symbols_per_code);
    }
  } else {
    build_hard_occurrence_index_kernel<<<
        num_series,
        kIndexThreads,
        0,
        stream>>>(
        key_codes.data_ptr<uint8_t>(),
        offsets.data_ptr<int32_t>(),
        occurrences.data_ptr<int32_t>(),
        first_symbol_positions_ptr,
        seq_len,
        symbol_dim,
        symbols_per_code);
  }
  torch::Tensor trajectory_segment_starts;
  torch::Tensor trajectory_latest_or_base;
  torch::Tensor trajectory_heavy_deltas;
  torch::Tensor trajectory_heavy_mismatch_prefixes;
  int32_t* trajectory_segment_starts_ptr = nullptr;
  int32_t* trajectory_base_lengths_ptr = nullptr;
  int32_t* trajectory_heavy_deltas_ptr = nullptr;
  int32_t* trajectory_heavy_mismatch_prefixes_ptr = nullptr;
  int mismatch_groups_per_series = 0;
  if (seq_len >= kTrajectoryMinSequenceLength) {
    const int64_t trajectory_rows =
        static_cast<int64_t>(num_series) * seq_len;
    trajectory_latest_or_base = torch::empty(
        {batch, num_heads, seq_len},
        index_options);
    trajectory_segment_starts =
        torch::empty_like(trajectory_latest_or_base);
    trajectory_segment_starts_ptr =
        trajectory_segment_starts.data_ptr<int32_t>();
    trajectory_base_lengths_ptr =
        trajectory_latest_or_base.data_ptr<int32_t>();
    if (seq_len >= kHeavyDiagonalMinSequenceLength) {
      mismatch_groups_per_series =
          (seq_len + kHardWarpSize - 1) / kHardWarpSize;
      trajectory_heavy_deltas = torch::empty(
          {batch, num_heads},
          index_options);
      trajectory_heavy_mismatch_prefixes = torch::empty(
          {batch, num_heads, mismatch_groups_per_series},
          index_options);
      trajectory_heavy_deltas_ptr =
          trajectory_heavy_deltas.data_ptr<int32_t>();
      trajectory_heavy_mismatch_prefixes_ptr =
          trajectory_heavy_mismatch_prefixes.data_ptr<int32_t>();
    }
    const int segment_chunks =
        (seq_len + kIndexThreads - 1) / kIndexThreads;
    candidate_trajectory_local_scan_kernel<<<
        num_series * segment_chunks,
        kIndexThreads,
        0,
        stream>>>(
        query_codes.data_ptr<uint8_t>(),
        offsets.data_ptr<int32_t>(),
        occurrences.data_ptr<int32_t>(),
        trajectory_base_lengths_ptr,
        trajectory_segment_starts_ptr,
        seq_len,
        segment_chunks,
        symbol_dim,
        symbols_per_code);
    candidate_segment_carry_kernel<<<
        num_series,
        kHardWarpSize,
        0,
        stream>>>(
        trajectory_segment_starts_ptr,
        seq_len,
        segment_chunks);
    if (trajectory_heavy_deltas_ptr != nullptr) {
      select_heavy_restart_delta_kernel<<<
          num_series,
          kIndexThreads,
          0,
          stream>>>(
          query_codes.data_ptr<uint8_t>(),
          key_codes.data_ptr<uint8_t>(),
          trajectory_base_lengths_ptr,
          trajectory_segment_starts_ptr,
          trajectory_heavy_deltas_ptr,
          seq_len);
      const int64_t mismatch_groups =
          static_cast<int64_t>(num_series) *
          mismatch_groups_per_series;
      const int mismatch_group_blocks = static_cast<int>(
          (mismatch_groups + kHardWarpsPerBlock - 1) /
          kHardWarpsPerBlock);
      build_heavy_diagonal_mismatch_groups_kernel<<<
          mismatch_group_blocks,
          kHardBlockThreads,
          0,
          stream>>>(
          query_codes.data_ptr<uint8_t>(),
          key_codes.data_ptr<uint8_t>(),
          trajectory_heavy_deltas_ptr,
          trajectory_heavy_mismatch_prefixes_ptr,
          seq_len,
          mismatch_groups_per_series,
          mismatch_groups);
      carry_heavy_diagonal_mismatch_groups_kernel<<<
          num_series,
          kHardWarpSize,
          0,
          stream>>>(
          trajectory_heavy_deltas_ptr,
          trajectory_heavy_mismatch_prefixes_ptr,
          mismatch_groups_per_series);
    }
    const int trajectory_workers = static_cast<int>(
        trajectory_rows <
                static_cast<int64_t>(num_series) *
                    kTrajectoryWorkersPerSeries
            ? trajectory_rows
            : static_cast<int64_t>(num_series) *
                  kTrajectoryWorkersPerSeries);
    const int trajectory_blocks =
        (trajectory_workers + kTrajectoryWarpsPerBlock - 1) /
        kTrajectoryWarpsPerBlock;
    candidate_segment_base_kernel<<<
        trajectory_blocks,
        kTrajectoryBlockThreads,
        0,
        stream>>>(
        query_codes.data_ptr<uint8_t>(),
        key_codes.data_ptr<uint8_t>(),
        trajectory_base_lengths_ptr,
        trajectory_segment_starts_ptr,
        trajectory_heavy_deltas_ptr,
        trajectory_heavy_mismatch_prefixes_ptr,
        trajectory_rows,
        seq_len,
        mismatch_groups_per_series,
        symbols_per_code);
  }
  const int64_t hard_rows =
      static_cast<int64_t>(batch) * num_heads * seq_len;
  const int hard_blocks = static_cast<int>(
      (hard_rows + kHardWarpsPerBlock - 1) /
      kHardWarpsPerBlock);
  DISPATCH_ROSA_HARD_FLOAT_TYPES(
      value.scalar_type(),
      "rosa_soft_hard_index",
      [&] {
        indexed_hard_forward_kernel<scalar_t><<<
          hard_blocks,
          kHardBlockThreads,
          0,
          stream>>>(
          query_codes.data_ptr<uint8_t>(),
          key_codes.data_ptr<uint8_t>(),
          value.data_ptr<scalar_t>(),
          offsets.data_ptr<int32_t>(),
          occurrences.data_ptr<int32_t>(),
          first_symbol_positions_ptr,
          trajectory_segment_starts_ptr,
          trajectory_base_lengths_ptr,
          trajectory_heavy_deltas_ptr,
          trajectory_heavy_mismatch_prefixes_ptr,
          output.data_ptr<scalar_t>(),
          seq_len,
          num_heads,
          num_value_heads,
          value_dim,
          symbol_dim,
          symbols_per_code,
          mismatch_groups_per_series,
          hard_rows);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
