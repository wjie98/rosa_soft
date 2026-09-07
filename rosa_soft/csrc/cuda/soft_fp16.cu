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


using namespace rosa_soft::cuda;


torch::Tensor rosa_soft_stats_fp16_cuda(
    const torch::Tensor& value,
    const torch::Tensor& dy,
    const torch::Tensor& packed_q,
    const torch::Tensor& packed_k,
    const torch::Tensor& seed,
    int symbol_dim,
    float scale,
    float dropout_p,
    float mismatch);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_backward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& dy,
    const torch::Tensor& packed_q,
    const torch::Tensor& packed_k,
    const torch::Tensor& seed,
    float scale,
    float dropout_p,
    float mismatch,
    int mask);


namespace {

namespace wmma = nvcuda::wmma;

constexpr int kRows = 32;
constexpr int kDiagonals = 32;
constexpr int kRouteCount = 64;
constexpr int kValueDim = 64;
// Physical strides keep WMMA loads off repeated shared-memory banks.
constexpr int kUtilityStride = 72;
constexpr int kInputStride = 72;
constexpr int kProbabilityStride = 40;
constexpr int kTensorTile = 16;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kWarpSize;
constexpr int kDiagonalRounds = kDiagonals / kWarps;
constexpr int kScoreStride = kRows + 1;
constexpr int kRowStatsWidth = 3;
constexpr int kRowMaximum = 0;
constexpr int kRowInverseNormalizer = 1;
constexpr int kRowUtility = 2;
constexpr int kMismatchGateCount = 33;


struct __align__(16) ProbabilityStorage {
  __half probability_high[kRouteCount * kProbabilityStride];
  __half probability_low[kRouteCount * kProbabilityStride];
};


union RouteHalfStorage {
  __half value[kRouteCount * kInputStride];
  ProbabilityStorage probability;
};


struct HalfPhaseStorage {
  // The same output gradient is consumed by utility and dV, so keep it outside
  // the value/probability overlay instead of loading it twice per row block.
  __half grad[kRows * kInputStride];
  RouteHalfStorage route;
};


union FloatPhaseStorage {
  float utility[kRows * kUtilityStride];
  float mma_output[kWarps * kTensorTile * kTensorTile];
};


__global__ void initialize_prior_kernel(
    float* __restrict__ prior,
    int seq_len) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < seq_len) {
    prior[row] = row > 0 ? logf(static_cast<float>(row)) : 0.0f;
  }
}


template <int TensorSymbolMask, bool SpecializedReplay>
__global__ void backward_kernel(
    const c10::Half* __restrict__ dy,
    const c10::Half* __restrict__ value,
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    const int64_t* __restrict__ seed,
    const float* __restrict__ prior,
    const float* __restrict__ stats,
    float* __restrict__ checkpoints,
    float* __restrict__ dq,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int series_count,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int symbol_dim,
    int row_block_count,
    int diagonal_tile_count,
    float mismatch_unit,
    float symbol_scale,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    int mask) {
#if __CUDA_ARCH__ >= 700
  __shared__ HalfPhaseStorage half_phase;
  __shared__ FloatPhaseStorage float_phase;
  __shared__ float score_tile[kDiagonals * kScoreStride];
  __shared__ float replay_gate_tile[kDiagonals * kScoreStride];
  __shared__ float gate_lut[kMismatchGateCount];

  const int thread = threadIdx.x;
  const int warp = thread / kWarpSize;
  const int lane = thread & (kWarpSize - 1);
  const uint32_t bit_mask = symbol_mask(symbol_dim);
  const int task_pairs_per_series = (diagonal_tile_count + 1) / 2;
  const int task_pair_count = series_count * task_pairs_per_series;
  const int worker_stride = gridDim.x;
  const int64_t checkpoint_block_stride =
      static_cast<int64_t>(row_block_count) * kDiagonals;
  float* const block_checkpoints =
      checkpoints + static_cast<int64_t>(blockIdx.x) *
          checkpoint_block_stride;

  initialize_mismatch_gate_lut(gate_lut, symbol_dim, mismatch_unit);

  for (int pair_ordinal = blockIdx.x;
       pair_ordinal < task_pair_count;
       pair_ordinal += worker_stride) {
    const int series = pair_ordinal / task_pairs_per_series;
    const int pair_index =
        pair_ordinal - series * task_pairs_per_series;
    const int tasks_in_pair =
        pair_index != diagonal_tile_count - 1 - pair_index
        ? 2
        : 1;
    for (int pair_side = 0;
         pair_side < tasks_in_pair;
         ++pair_side) {
      // Reverse the pair on alternating workers to dephase atomic updates.
      const int ordered_side = tasks_in_pair == 2
          ? pair_side ^ (blockIdx.x & 1)
          : 0;
      const int diagonal_tile = ordered_side == 0
          ? pair_index
          : diagonal_tile_count - 1 - pair_index;
      const int diagonal_start = 1 + diagonal_tile * kDiagonals;
      const int tile_width = min(kDiagonals, seq_len - diagonal_start);
      const int first_row_block = diagonal_start / kRows;
      const int head = series % num_heads;
      const int batch = series / num_heads;
      const int value_head = head / (num_heads / num_value_heads);
      const int64_t series_offset = static_cast<int64_t>(series) * seq_len;
      float score_carry = 0.0f;

      // Save only the incoming state of each 32-row replay interval. Scratch is
      // private to the physical CTA and reused for every logical task it owns.
      for (int row_block = first_row_block;
           row_block < row_block_count;
           ++row_block) {
        const int row_start = row_block * kRows;
        const int group = thread / 8;
        const int local_lane = thread % 8;
        const int delta = diagonal_start + group;
        if (local_lane == 0) {
          block_checkpoints[
              static_cast<int64_t>(row_block) * kDiagonals + group] = score_carry;
        }
        Affine local{1.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = row_start + local_lane * 4 + i;
          const bool active = group < tile_width && row < seq_len && delta <= row;
          const float gate = active
              ? mismatch_gate_from_lut(
                    packed_query[series_offset + row],
                    packed_key[series_offset + row - delta], bit_mask, gate_lut)
              : 1.0f;
          local = Compose{}(local, Affine{gate, active ? gate : 0.0f});
        }
        using Reduce = cub::WarpReduce<Affine, 8>;
        __shared__ typename Reduce::TempStorage storage[kDiagonals];
        const Affine total = Reduce(storage[group]).Reduce(local, Compose{});
        __syncwarp();
        const float score = fmaf(total.a, score_carry, total.b);
        score_carry = __shfl_sync(0xffffffffu, score, 0, 8);
      }
      __syncthreads();

      float successor_adjoint[kDiagonalRounds] = {};
      float successor_gate[kDiagonalRounds] = {};
      for (int row_block = row_block_count - 1;
           row_block >= first_row_block;
           --row_block) {
        const int row_start = row_block * kRows;
        const int row_count = min(kRows, seq_len - row_start);
        const int route_start =
            row_start - diagonal_start - (kDiagonals - 1) + 1;
        const int row = row_start + lane;
        const bool lane_has_row = lane < row_count;
        const uint32_t query_word = lane_has_row
            ? static_cast<uint32_t>(packed_query[series_offset + row])
            : 0u;
        const int64_t stats_index = (series_offset + row) * kRowStatsWidth;
        const float log_n = lane_has_row ? prior[row] : 0.0f;
        const float row_maximum = lane_has_row
            ? stats[stats_index + kRowMaximum]
            : 0.0f;
        const float row_inverse_normalizer = lane_has_row
            ? stats[stats_index + kRowInverseNormalizer]
            : 0.0f;
        const float row_utility = lane_has_row
            ? stats[stats_index + kRowUtility]
            : 0.0f;
        float replay_gate[kDiagonalRounds];

        constexpr int kValuePairs = kValueDim / 2;
        if constexpr (SpecializedReplay) {
          constexpr int kProducerWarps = kWarps / 2;
          constexpr int kProducerThreads = kProducerWarps * kWarpSize;
          if (warp < kProducerWarps) {
#pragma unroll
            for (int round = 0;
                 round < kDiagonals / kProducerWarps;
                 ++round) {
              const int diagonal_offset =
                  round * kProducerWarps + warp;
              const int delta = diagonal_start + diagonal_offset;
              const float incoming = block_checkpoints[
                  static_cast<int64_t>(row_block) * kDiagonals +
                  diagonal_offset];
              const bool active = diagonal_offset < tile_width &&
                  lane_has_row && delta <= row;
              const float gate = active
                  ? mismatch_gate_from_lut(
                        query_word,
                        static_cast<uint32_t>(
                            packed_key[series_offset + row - delta]),
                            bit_mask,
                        gate_lut)
                  : 1.0f;
              float coefficient = gate;
              float bias = active ? gate : 0.0f;
              warp_forward_affine_scan(coefficient, bias);
              const float score = fmaf(coefficient, incoming, bias);
              score_tile[diagonal_offset * kScoreStride + lane] =
                  active ? score : 0.0f;
              replay_gate_tile[
                  diagonal_offset * kScoreStride + lane] =
                  active ? gate : 0.0f;
            }
          } else {
            const int loader_thread = thread - kProducerThreads;
            for (int pair = loader_thread;
                 pair < kRows * kValuePairs;
                 pair += kProducerThreads) {
              const int row_offset = pair / kValuePairs;
              const int feature = (pair - row_offset * kValuePairs) * 2;
              const int source_row = row_start + row_offset;
              __half2 gradient = __float2half2_rn(0.0f);
              if (row_offset < row_count) {
                const int64_t source =
                    ((static_cast<int64_t>(batch) * seq_len + source_row) *
                         num_heads +
                     head) * kValueDim + feature;
                gradient = *reinterpret_cast<const __half2*>(
                    dy + source);
              }
              reinterpret_cast<__half2*>(half_phase.grad)[
                  row_offset * (kInputStride / 2) + feature / 2] = gradient;
            }
            for (int pair = loader_thread;
                 pair < kRouteCount * kValuePairs;
                 pair += kProducerThreads) {
              const int route_offset = pair / kValuePairs;
              const int feature = (pair - route_offset * kValuePairs) * 2;
              const int route = route_start + route_offset;
              __half2 value_sign = __float2half2_rn(0.0f);
              if (route >= 1 && route < seq_len) {
                const int64_t source =
                    ((static_cast<int64_t>(batch) * seq_len + route) *
                         num_value_heads +
                     value_head) * kValueDim + feature;
                value_sign = binary_sign_half2(
                    *reinterpret_cast<const __half2*>(value + source));
              }
              reinterpret_cast<__half2*>(half_phase.route.value)[
                  route_offset * (kInputStride / 2) + feature / 2] =
                  value_sign;
            }
          }
        } else {
#pragma unroll
          for (int round = 0; round < kDiagonalRounds; ++round) {
            const int diagonal_offset = round * kWarps + warp;
            const int delta = diagonal_start + diagonal_offset;
            const float incoming = block_checkpoints[
                static_cast<int64_t>(row_block) * kDiagonals +
                diagonal_offset];
            const bool active = diagonal_offset < tile_width &&
                lane_has_row && delta <= row;
            const float gate = active
                ? mismatch_gate_from_lut(
                      query_word,
                      static_cast<uint32_t>(
                          packed_key[series_offset + row - delta]),
                      bit_mask,
                      gate_lut)
                : 1.0f;
            float coefficient = gate;
            float bias = active ? gate : 0.0f;
            warp_forward_affine_scan(coefficient, bias);
            const float score = fmaf(coefficient, incoming, bias);
            score_tile[diagonal_offset * kScoreStride + lane] =
                active ? score : 0.0f;
            replay_gate[round] = active ? gate : 0.0f;
          }
          for (int pair = thread;
               pair < kRows * kValuePairs;
               pair += blockDim.x) {
            const int row_offset = pair / kValuePairs;
            const int feature = (pair - row_offset * kValuePairs) * 2;
            const int source_row = row_start + row_offset;
            __half2 gradient = __float2half2_rn(0.0f);
            if (row_offset < row_count) {
              const int64_t source =
                  ((static_cast<int64_t>(batch) * seq_len + source_row) *
                       num_heads +
                   head) * kValueDim + feature;
              gradient = *reinterpret_cast<const __half2*>(
                  dy + source);
            }
            reinterpret_cast<__half2*>(half_phase.grad)[
                row_offset * (kInputStride / 2) + feature / 2] = gradient;
          }
          for (int pair = thread;
               pair < kRouteCount * kValuePairs;
               pair += blockDim.x) {
            const int route_offset = pair / kValuePairs;
            const int feature = (pair - route_offset * kValuePairs) * 2;
            const int route = route_start + route_offset;
            __half2 value_sign = __float2half2_rn(0.0f);
            if (route >= 1 && route < seq_len) {
              const int64_t source =
                  ((static_cast<int64_t>(batch) * seq_len + route) *
                       num_value_heads +
                   value_head) * kValueDim + feature;
              value_sign = binary_sign_half2(
                  *reinterpret_cast<const __half2*>(value + source));
            }
            reinterpret_cast<__half2*>(half_phase.route.value)[
                route_offset * (kInputStride / 2) + feature / 2] =
                value_sign;
          }
        }
        __syncthreads();

        constexpr int kUtilityRouteTiles = kRouteCount / kTensorTile;
        const int utility_row =
            (warp / kUtilityRouteTiles) * kTensorTile;
        const int utility_route =
            (warp % kUtilityRouteTiles) * kTensorTile;
        // The unused corners contain no causal diagonal candidates.
        if (warp != 3 && warp != 4) {
          wmma::fragment<
              wmma::accumulator,
              kTensorTile,
              kTensorTile,
              kTensorTile,
              float>
              utility_accumulator;
          wmma::fill_fragment(utility_accumulator, 0.0f);
#pragma unroll
          for (int feature_start = 0;
               feature_start < kValueDim;
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
                half_phase.grad +
                    utility_row * kInputStride + feature_start,
                kInputStride);
            wmma::load_matrix_sync(
                signed_value,
                half_phase.route.value +
                    utility_route * kInputStride + feature_start,
                kInputStride);
            wmma::mma_sync(
                utility_accumulator,
                gradient,
                signed_value,
                utility_accumulator);
          }
          wmma::store_matrix_sync(
              float_phase.utility +
                  utility_row * kUtilityStride + utility_route,
              utility_accumulator,
              kUtilityStride,
              wmma::mem_row_major);
        }
        __syncthreads();

        if ((mask & kGradValue) != 0) {
          constexpr int kProbabilityBytes = sizeof(ProbabilityStorage);
          constexpr int kZeroWords = kProbabilityBytes / sizeof(uint4);
          uint4* const probability_words = reinterpret_cast<uint4*>(
              &half_phase.route.probability);
          for (int index = thread;
               index < kZeroWords;
               index += blockDim.x) {
            probability_words[index] = make_uint4(0u, 0u, 0u, 0u);
          }
          __syncthreads();
        }

#pragma unroll
        for (int round = 0; round < kDiagonalRounds; ++round) {
          const int diagonal_offset = round * kWarps + warp;
          const int delta = diagonal_start + diagonal_offset;
          const bool active = diagonal_offset < tile_width &&
              lane_has_row && delta <= row;
          const float gate = SpecializedReplay
              ? replay_gate_tile[
                    diagonal_offset * kScoreStride + lane]
              : replay_gate[round];
          const float next_lane_gate =
              __shfl_down_sync(0xffffffffu, gate, 1);
          float coefficient = 1.0f;
          float direct_vjp = 0.0f;
          float raw_score = 0.0f;
          if (active) {
            coefficient = lane + 1 < row_count
                ? next_lane_gate
                : successor_gate[round];
            raw_score = score_tile[
                diagonal_offset * kScoreStride + lane];
            const ScoreTransform transformed = transform_score(raw_score);
            const float probability = __expf(
                transformed.route_score * scale - log_n - row_maximum) *
                row_inverse_normalizer;
            const int route_offset =
                lane - diagonal_offset + kDiagonals - 1;
            const int route = route_start + route_offset;
            const float dropout_scale = attention_dropout_scale(
                seed,
                dropout_p,
                inverse_keep_probability,
                batch,
                head,
                row,
                route);
            if ((mask & kGradValue) != 0) {
              const int probability_index = route_offset * kProbabilityStride + lane;
              const float dropped_probability = probability * dropout_scale;
              const __half high = __float2half_rn(dropped_probability);
              half_phase.route.probability.probability_high[
                  probability_index] = high;
              half_phase.route.probability.probability_low[
                  probability_index] = __float2half_rn(
                      dropped_probability - __half2float(high));
            }
            if ((mask & (kGradQuery | kGradKey)) != 0) {
              direct_vjp = scale * probability *
                  (dropout_scale * float_phase.utility[
                       lane * kUtilityStride + route_offset] -
                   row_utility) *
                  transformed.raw_vjp_multiplier;
            }
          }
          warp_reverse_affine_scan(coefficient, direct_vjp, row_count);
          const float score_vjp = fmaf(
              coefficient, successor_adjoint[round], direct_vjp);
          score_tile[diagonal_offset * kScoreStride + lane] =
              active ? raw_score * score_vjp : 0.0f;
          successor_adjoint[round] =
              __shfl_sync(0xffffffffu, score_vjp, 0);
          successor_gate[round] =
              __shfl_sync(0xffffffffu, gate, 0);
        }
        __syncthreads();

        if constexpr ((TensorSymbolMask & kGradQuery) == 0) {
          if ((mask & kGradQuery) != 0) {
            const int output_count = row_count * symbol_dim;
            for (int output = thread;
                 output < output_count;
                 output += blockDim.x) {
              const int row_offset = output / symbol_dim;
              const int bit = output - row_offset * symbol_dim;
              const int row = row_start + row_offset;
              float contribution = 0.0f;
              for (int diagonal_offset = 0;
                   diagonal_offset < tile_width;
                   ++diagonal_offset) {
                const int delta = diagonal_start + diagonal_offset;
                if (delta <= row) {
                  const uint32_t key_word = static_cast<uint32_t>(
                      packed_key[series_offset + row - delta]);
                  contribution = fmaf(
                      score_tile[
                          diagonal_offset * kScoreStride + row_offset],
                      static_cast<float>(sign_from_bit(key_word, bit)),
                      contribution);
                }
              }
              const int64_t target =
                  ((static_cast<int64_t>(batch) * seq_len + row) *
                       num_heads +
                   head) * symbol_dim + bit;
              atomicAdd(dq + target, symbol_scale * contribution);
            }
          }
        }

        if constexpr ((TensorSymbolMask & kGradKey) == 0) {
          if ((mask & kGradKey) != 0) {
            const int output_count =
                (kRows + kDiagonals - 1) * symbol_dim;
            for (int output = thread;
                 output < output_count;
                 output += blockDim.x) {
              const int route_offset = output / symbol_dim;
              const int bit = output - route_offset * symbol_dim;
              const int key_position = route_start + route_offset - 1;
              if (key_position < 0 || key_position >= seq_len) {
                continue;
              }
              float contribution = 0.0f;
              for (int diagonal_offset = 0;
                   diagonal_offset < tile_width;
                   ++diagonal_offset) {
                const int row_offset =
                    route_offset + diagonal_offset - (kDiagonals - 1);
                if (row_offset >= 0 && row_offset < row_count) {
                  const int row = row_start + row_offset;
                  const int delta = diagonal_start + diagonal_offset;
                  if (delta <= row) {
                    const uint32_t query_word = static_cast<uint32_t>(
                        packed_query[series_offset + row]);
                    contribution = fmaf(
                        score_tile[
                            diagonal_offset * kScoreStride + row_offset],
                        static_cast<float>(sign_from_bit(query_word, bit)),
                        contribution);
                  }
                }
              }
              const int64_t target =
                  ((static_cast<int64_t>(batch) * seq_len + key_position) *
                       num_heads +
                   head) * symbol_dim + bit;
              atomicAdd(dk + target, symbol_scale * contribution);
            }
          }
        }
        if ((mask & kGradValue) != 0) {
#pragma unroll
          for (int output_round = 0; output_round < 2; ++output_round) {
            const int output_tile = output_round * kWarps + warp;
            const int output_route =
                (output_tile / (kValueDim / kTensorTile)) * kTensorTile;
            const int output_feature =
                (output_tile % (kValueDim / kTensorTile)) * kTensorTile;
            wmma::fragment<
                wmma::accumulator,
                kTensorTile,
                kTensorTile,
                kTensorTile,
                float>
                value_accumulator;
            wmma::fill_fragment(value_accumulator, 0.0f);
#pragma unroll
            for (int query_offset = 0;
                 query_offset < kRows;
                 query_offset += kTensorTile) {
              wmma::fragment<
                  wmma::matrix_a,
                  kTensorTile,
                  kTensorTile,
                  kTensorTile,
                  __half,
                  wmma::row_major>
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
                  probability,
                  half_phase.route.probability.probability_high +
                      output_route * kProbabilityStride + query_offset,
                  kProbabilityStride);
              wmma::load_matrix_sync(
                  gradient,
                  half_phase.grad +
                      query_offset * kInputStride + output_feature,
                  kInputStride);
              wmma::mma_sync(
                  value_accumulator,
                  probability,
                  gradient,
                  value_accumulator);
              wmma::load_matrix_sync(
                  probability,
                  half_phase.route.probability.probability_low +
                      output_route * kProbabilityStride + query_offset,
                  kProbabilityStride);
              wmma::mma_sync(
                  value_accumulator,
                  probability,
                  gradient,
                  value_accumulator);
            }
            float* const warp_output =
                float_phase.mma_output + warp * kTensorTile * kTensorTile;
            wmma::store_matrix_sync(
                warp_output,
                value_accumulator,
                kTensorTile,
                wmma::mem_row_major);
            __syncwarp();
            for (int output = lane;
                 output < kTensorTile * kTensorTile;
                 output += kWarpSize) {
              const int route_offset =
                  output_route + output / kTensorTile;
              const int feature =
                  output_feature + output % kTensorTile;
              const int route = route_start + route_offset;
              if (route >= 1 && route < seq_len) {
                const int64_t target =
                    ((static_cast<int64_t>(batch) * seq_len + route) *
                         num_value_heads +
                     value_head) * kValueDim + feature;
                atomicAdd(dv + target, warp_output[output]);
              }
            }
            __syncwarp();
          }
        }
        __syncthreads();

        if constexpr (TensorSymbolMask != 0) {
          constexpr int kSymbolStride = 40;
          __half* const symbol_tile =
              reinterpret_cast<__half*>(float_phase.utility);
          constexpr int kProbabilityElements = kRouteCount * kProbabilityStride;
          if ((mask & kGradValue) == 0) {
            for (int index = thread;
                 index < kProbabilityElements;
                 index += blockDim.x) {
              half_phase.route.probability.probability_high[index] =
                  __float2half_rn(0.0f);
              half_phase.route.probability.probability_low[index] =
                  __float2half_rn(0.0f);
            }
            __syncthreads();
          }

          for (int cell = thread;
               cell < kRows * kDiagonals;
               cell += blockDim.x) {
            const int diagonal_offset = cell / kRows;
            const int row_offset = cell - diagonal_offset * kRows;
            const int candidate_row = row_start + row_offset;
            const bool active = row_offset < row_count &&
                diagonal_offset < tile_width &&
                diagonal_start + diagonal_offset <= candidate_row;
            if (active) {
              const int route_offset =
                  row_offset - diagonal_offset + kDiagonals - 1;
              const float credit = score_tile[
                  diagonal_offset * kScoreStride + row_offset];
              const int credit_index = route_offset * kProbabilityStride + row_offset;
              const __half high = __float2half_rn(credit);
              half_phase.route.probability.probability_high[credit_index] =
                  high;
              half_phase.route.probability.probability_low[credit_index] =
                  __float2half_rn(credit - __half2float(high));
            }
          }
          __syncthreads();

          // Signs are exactly representable in FP16. Splitting each FP32
          // credit into high and residual halves keeps Tensor Core
          // contractions close to the scalar FP32 accumulation.
          if constexpr ((TensorSymbolMask & kGradKey) != 0) {
            if ((mask & kGradKey) != 0) {
              for (int index = thread;
                   index < kRows * kSymbolStride;
                   index += blockDim.x) {
                const int row_offset = index / kSymbolStride;
                const int bit = index - row_offset * kSymbolStride;
                __half sign = __float2half_rn(0.0f);
                if (row_offset < row_count && bit < symbol_dim) {
                  const uint32_t word = static_cast<uint32_t>(
                      packed_query[
                          series_offset + row_start + row_offset]);
                  sign = __int2half_rn(sign_from_bit(word, bit));
                }
                symbol_tile[index] = sign;
              }
              __syncthreads();

              const int bit_tile_count =
                  (symbol_dim + kTensorTile - 1) / kTensorTile;
              const int output_tile_count =
                  (kRouteCount / kTensorTile) * bit_tile_count;
              const bool owns_output = warp < output_tile_count;
              const int output_route = owns_output
                  ? (warp / bit_tile_count) * kTensorTile
                  : 0;
              const int output_bit = owns_output
                  ? (warp % bit_tile_count) * kTensorTile
                  : 0;
              wmma::fragment<
                  wmma::accumulator,
                  kTensorTile,
                  kTensorTile,
                  kTensorTile,
                  float>
                  key_accumulator;
              wmma::fill_fragment(key_accumulator, 0.0f);
              if (owns_output) {
#pragma unroll
                for (int query_start = 0;
                     query_start < kRows;
                     query_start += kTensorTile) {
                  wmma::fragment<
                      wmma::matrix_a,
                      kTensorTile,
                      kTensorTile,
                      kTensorTile,
                      __half,
                      wmma::row_major>
                      credit;
                  wmma::fragment<
                      wmma::matrix_b,
                      kTensorTile,
                      kTensorTile,
                      kTensorTile,
                      __half,
                      wmma::row_major>
                      query_sign;
                  wmma::load_matrix_sync(
                      query_sign,
                      symbol_tile + query_start * kSymbolStride + output_bit,
                      kSymbolStride);
                  wmma::load_matrix_sync(
                      credit,
                      half_phase.route.probability.probability_high +
                          output_route * kProbabilityStride + query_start,
                      kProbabilityStride);
                  wmma::mma_sync(
                      key_accumulator,
                      credit,
                      query_sign,
                      key_accumulator);
                  wmma::load_matrix_sync(
                      credit,
                      half_phase.route.probability.probability_low +
                          output_route * kProbabilityStride + query_start,
                      kProbabilityStride);
                  wmma::mma_sync(
                      key_accumulator,
                      credit,
                      query_sign,
                      key_accumulator);
                }
              }
              __syncthreads();
              float* const warp_output =
                  float_phase.mma_output + warp * kTensorTile * kTensorTile;
              if (owns_output) {
                wmma::store_matrix_sync(
                    warp_output,
                    key_accumulator,
                    kTensorTile,
                    wmma::mem_row_major);
              }
              __syncthreads();
              if (owns_output) {
                for (int output = lane;
                     output < kTensorTile * kTensorTile;
                     output += kWarpSize) {
                  const int route_offset =
                      output_route + output / kTensorTile;
                  const int bit = output_bit + output % kTensorTile;
                  const int key_position = route_start + route_offset - 1;
                  if (bit < symbol_dim && key_position >= 0 &&
                      key_position < seq_len) {
                    const int64_t target =
                        ((static_cast<int64_t>(batch) * seq_len +
                          key_position) * num_heads + head) * symbol_dim + bit;
                    atomicAdd(
                        dk + target,
                        symbol_scale * warp_output[output]);
                  }
                }
              }
              __syncthreads();
            }
          }

          if constexpr ((TensorSymbolMask & kGradQuery) != 0) {
            if ((mask & kGradQuery) != 0) {
              for (int index = thread;
                   index < kRouteCount * kSymbolStride;
                   index += blockDim.x) {
                const int route_offset = index / kSymbolStride;
                const int bit = index - route_offset * kSymbolStride;
                const int key_position = route_start + route_offset - 1;
                __half sign = __float2half_rn(0.0f);
                if (bit < symbol_dim && key_position >= 0 &&
                    key_position < seq_len) {
                  const uint32_t word = static_cast<uint32_t>(
                      packed_key[series_offset + key_position]);
                  sign = __int2half_rn(sign_from_bit(word, bit));
                }
                symbol_tile[index] = sign;
              }
              __syncthreads();

              const int bit_tile_count =
                  (symbol_dim + kTensorTile - 1) / kTensorTile;
              const int output_tile_count =
                  (kRows / kTensorTile) * bit_tile_count;
              const bool owns_output = warp < output_tile_count;
              const int output_row = owns_output
                  ? (warp / bit_tile_count) * kTensorTile
                  : 0;
              const int output_bit = owns_output
                  ? (warp % bit_tile_count) * kTensorTile
                  : 0;
              wmma::fragment<
                  wmma::accumulator,
                  kTensorTile,
                  kTensorTile,
                  kTensorTile,
                  float>
                  query_accumulator;
              wmma::fill_fragment(query_accumulator, 0.0f);
              if (owns_output) {
#pragma unroll
                for (int route_offset = 0;
                     route_offset < kRouteCount;
                     route_offset += kTensorTile) {
                  wmma::fragment<
                      wmma::matrix_a,
                      kTensorTile,
                      kTensorTile,
                      kTensorTile,
                      __half,
                      wmma::col_major>
                      credit;
                  wmma::fragment<
                      wmma::matrix_b,
                      kTensorTile,
                      kTensorTile,
                      kTensorTile,
                      __half,
                      wmma::row_major>
                      key_sign;
                  wmma::load_matrix_sync(
                      key_sign,
                      symbol_tile + route_offset * kSymbolStride + output_bit,
                      kSymbolStride);
                  wmma::load_matrix_sync(
                      credit,
                      half_phase.route.probability.probability_high +
                          route_offset * kProbabilityStride + output_row,
                      kProbabilityStride);
                  wmma::mma_sync(
                      query_accumulator,
                      credit,
                      key_sign,
                      query_accumulator);
                  wmma::load_matrix_sync(
                      credit,
                      half_phase.route.probability.probability_low +
                          route_offset * kProbabilityStride + output_row,
                      kProbabilityStride);
                  wmma::mma_sync(
                      query_accumulator,
                      credit,
                      key_sign,
                      query_accumulator);
                }
              }
              __syncthreads();
              float* const warp_output =
                  float_phase.mma_output + warp * kTensorTile * kTensorTile;
              if (owns_output) {
                wmma::store_matrix_sync(
                    warp_output,
                    query_accumulator,
                    kTensorTile,
                    wmma::mem_row_major);
              }
              __syncthreads();
              if (owns_output) {
                for (int output = lane;
                     output < kTensorTile * kTensorTile;
                     output += kWarpSize) {
                  const int row_offset = output_row + output / kTensorTile;
                  const int bit = output_bit + output % kTensorTile;
                  if (row_offset < row_count && bit < symbol_dim) {
                    const int64_t target =
                        ((static_cast<int64_t>(batch) * seq_len +
                          row_start + row_offset) * num_heads + head) *
                            symbol_dim + bit;
                    atomicAdd(
                        dq + target,
                        symbol_scale * warp_output[output]);
                  }
                }
              }
              __syncthreads();
            }
          }
        }
      }
    }
  }
#endif
}


__global__ void finish_kernel(
    const c10::Half* __restrict__ query,
    const c10::Half* __restrict__ key,
    const c10::Half* __restrict__ value,
    float* __restrict__ dq,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int64_t query_elements,
    int64_t key_elements,
    int64_t value_elements) {
  const int64_t total =
      query_elements + key_elements + value_elements;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (index < query_elements) {
      dq[index] *=
          softsign_derivative(read_float(query, index));
    } else if (index < query_elements + key_elements) {
      const int64_t key_index = index - query_elements;
      dk[key_index] *=
          softsign_derivative(read_float(key, key_index));
    } else {
      const int64_t value_index =
          index - query_elements - key_elements;
      dv[value_index] *=
          softsign_derivative(read_float(value, value_index));
    }
  }
}

}  // namespace


template <int TensorSymbolMask, bool SpecializedReplay>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward_fp16(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& dy,
    const torch::Tensor& packed_q,
    const torch::Tensor& packed_k,
    const torch::Tensor& seed,
    const torch::Tensor& stats,
    float scale,
    float dropout_p,
    float mismatch,
    int mask) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  const int batch_size = query.size(0);
  const int seq_len = query.size(1);
  const int num_heads = query.size(2);
  const int symbol_dim = query.size(3);
  const int num_value_heads = value.size(2);
  const int series_count = batch_size * num_heads;
  const auto float_options = query.options().dtype(torch::kFloat32);
  torch::Tensor dq = (mask & kGradQuery) != 0
      ? torch::zeros(query.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor dk = (mask & kGradKey) != 0
      ? torch::zeros(key.sizes(), float_options)
      : torch::empty({0}, float_options);
  torch::Tensor dv = (mask & kGradValue) != 0
      ? torch::zeros(value.sizes(), float_options)
      : torch::empty({0}, float_options);
  if (seq_len <= 1) {
    return std::make_tuple(dq, dk, dv);
  }

  const int row_block_count = (seq_len + kRows - 1) / kRows;
  const int diagonal_tile_count =
      (seq_len - 1 + kDiagonals - 1) / kDiagonals;
  const int task_pair_count =
      series_count * ((diagonal_tile_count + 1) / 2);
  int blocks_per_sm = 1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      backward_kernel<
          TensorSymbolMask,
          SpecializedReplay>,
      kThreads,
      0));
  const int multiprocessors =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int resident_blocks =
      std::max(1, blocks_per_sm * multiprocessors);
  const int grid_count = std::min(
      task_pair_count, resident_blocks);
  torch::Tensor checkpoints = torch::empty(
      {grid_count, row_block_count, kDiagonals}, float_options);
  torch::Tensor prior = torch::empty({seq_len}, float_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  initialize_prior_kernel<<<
      (seq_len + kThreads - 1) / kThreads,
      kThreads,
      0,
      stream>>>(prior.data_ptr<float>(), seq_len);

  backward_kernel<
      TensorSymbolMask,
      SpecializedReplay><<<
      grid_count, kThreads, 0, stream>>>(
      dy.data_ptr<c10::Half>(),
      value.data_ptr<c10::Half>(),
      packed_q.data_ptr<int32_t>(),
      packed_k.data_ptr<int32_t>(),
      seed.data_ptr<int64_t>(),
      prior.data_ptr<float>(),
      stats.data_ptr<float>(),
      checkpoints.data_ptr<float>(),
      dq.numel() != 0 ? dq.data_ptr<float>() : nullptr,
      dk.numel() != 0 ? dk.data_ptr<float>() : nullptr,
      dv.numel() != 0 ? dv.data_ptr<float>() : nullptr,
      series_count,
      seq_len,
      num_heads,
      num_value_heads,
      symbol_dim,
      row_block_count,
      diagonal_tile_count,
      mismatch / static_cast<float>(symbol_dim),
      0.5f * mismatch / static_cast<float>(symbol_dim),
      scale,
      dropout_p,
      1.0f / (1.0f - dropout_p),
      mask);

  const int64_t query_elements = dq.numel();
  const int64_t key_elements = dk.numel();
  const int64_t value_elements = dv.numel();
  const int64_t final_elements =
      query_elements + key_elements + value_elements;
  if (final_elements != 0) {
    finish_kernel<<<
        std::min<int64_t>(
            65535, (final_elements + kThreads - 1) / kThreads),
        kThreads,
        0,
        stream>>>(
        query.data_ptr<c10::Half>(),
        key.data_ptr<c10::Half>(),
        value.data_ptr<c10::Half>(),
        dq.numel() != 0 ? dq.data_ptr<float>() : nullptr,
        dk.numel() != 0 ? dk.data_ptr<float>() : nullptr,
        dv.numel() != 0 ? dv.data_ptr<float>() : nullptr,
        query_elements,
        key_elements,
        value_elements);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(dq, dk, dv);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward_fp16_dispatch(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& dy,
    const torch::Tensor& packed_q,
    const torch::Tensor& packed_k,
    const torch::Tensor& seed,
    const torch::Tensor& stats,
    float scale,
    float dropout_p,
    float mismatch,
    int mask) {
  const int symbol_dim = query.size(3);
  const bool needs_query = (mask & kGradQuery) != 0;
  const bool needs_key = (mask & kGradKey) != 0;
  if (symbol_dim >= 16 && needs_query && needs_key) {
    return backward_fp16<
        kGradQuery | kGradKey,
        true>(
        query,
        key,
        value,
        dy,
        packed_q,
        packed_k,
        seed,
        stats,
        scale,
        dropout_p,
        mismatch,
        mask);
  }
  if (symbol_dim >= 8 && needs_key) {
    return backward_fp16<kGradKey, true>(
        query,
        key,
        value,
        dy,
        packed_q,
        packed_k,
        seed,
        stats,
        scale,
        dropout_p,
        mismatch,
        mask);
  }
  if (symbol_dim >= 16 && needs_query) {
    return backward_fp16<kGradQuery, false>(
        query,
        key,
        value,
        dy,
        packed_q,
        packed_k,
        seed,
        stats,
        scale,
        dropout_p,
        mismatch,
        mask);
  }
  return backward_fp16<0, false>(
      query,
      key,
      value,
      dy,
      packed_q,
      packed_k,
      seed,
      stats,
      scale,
      dropout_p,
      mismatch,
      mask);
}





std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_backward_fp16_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& dy,
    const torch::Tensor& packed_q,
    const torch::Tensor& packed_k,
    const torch::Tensor& seed,
    float scale,
    float dropout_p,
    float mismatch,
    int mask) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  if (at::cuda::getCurrentDeviceProperties()->major < 7) {
    return rosa_soft_backward_cuda(
        query,
        key,
        value,
        dy,
        packed_q,
        packed_k,
        seed,
        scale,
        dropout_p,
        mismatch,
        mask);
  }
  torch::Tensor stats = rosa_soft_stats_fp16_cuda(
      value,
      dy,
      packed_q,
      packed_k,
      seed,
      query.size(3),
      scale,
      dropout_p,
      mismatch);
  return backward_fp16_dispatch(
      query,
      key,
      value,
      dy,
      packed_q,
      packed_k,
      seed,
      stats,
      scale,
      dropout_p,
      mismatch,
      mask);
}
