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
#include "../soft.h"


using namespace rosa_soft::cuda;


namespace rosa::soft {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kProbabilityStride = 32;

__device__ __forceinline__ int tile_index(int route, int row) {
  // Permute aligned 8-half segments for both diagonal stores and ldmatrix.
  return route * kProbabilityStride + (row ^ ((route & 6) << 2));
}

// Documented mma.m16n8k8 registers, independent of WMMA's opaque layout.
struct Mma {
  float x[8] = {};

  template <bool Transpose>
  __device__ __forceinline__ static void load(unsigned (&r)[2], const __half* p) {
#if __CUDA_ARCH__ >= 750
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    if constexpr (Transpose) {
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                   : "=r"(r[0]), "=r"(r[1]) : "r"(a));
    } else {
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                   : "=r"(r[0]), "=r"(r[1]) : "r"(a));
    }
#endif
  }

  template <int K, bool Transpose = false>
  __device__ __forceinline__ void mul(
      const __half* high, const __half* low, const __half* b, int origin, int sb) {
#if __CUDA_ARCH__ >= 750
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int k = 0; k < K; k += 8) {
      unsigned br[2];
      load<true>(br, b + (k + lane % 8) * sb + ((lane / 8) % 2) * 8);
      // Swizzle the full coordinate, including the transposed tile's origin.
      const int a_index = Transpose
          ? tile_index(k + lane % 8, origin + ((lane / 8) % 2) * 8)
          : tile_index(origin + lane % 16, k);
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        const __half* a = part == 0 ? high : low;
        unsigned ar[2];
        load<Transpose>(ar, a + a_index);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
              : "+f"(x[4 * n]), "+f"(x[4 * n + 1]), "+f"(x[4 * n + 2]), "+f"(x[4 * n + 3])
              : "r"(ar[0]), "r"(ar[1]), "r"(br[n]));
        }
      }
    }
#endif
  }

  __device__ __forceinline__ void reorder() {
    const bool upper = (threadIdx.x & 4) != 0;
    // Swap lane bit 2 with register bit 0: four contiguous 8-float rows/warp.
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const float a = __shfl_xor_sync(0xffffffffu, upper ? x[i] : x[i + 1], 4);
      x[i] = upper ? a : x[i];
      x[i + 1] = upper ? x[i + 1] : a;
    }
  }
  __device__ __forceinline__ static int row(int i) {
    return (threadIdx.x % 32) / 8 * 2 + i % 2 + (i % 4 / 2) * 8;
  }
  __device__ __forceinline__ static int col(int i) {
    return (threadIdx.x % 4) * 2 + (threadIdx.x % 8) / 4 + (i / 4) * 8;
  }
};

constexpr int kRows = 32;
constexpr int kDiagonals = 32;
constexpr int kRouteCount = 64;
constexpr int kValueDim = 64;
// Physical input strides keep WMMA loads off repeated shared-memory banks.
constexpr int kUtilityStride = 72;
constexpr int kInputStride = 72;
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
  __half high[kRouteCount * kProbabilityStride];
  __half low[kRouteCount * kProbabilityStride];
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

static_assert(kRows == 32 && kRouteCount == 64, "Probability layout is 64x32");

template <int TensorMask>
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
#if __CUDA_ARCH__ >= 750
  constexpr bool Split = (TensorMask & kGradKey) != 0;
  __shared__ HalfPhaseStorage half_phase;
  __shared__ float utility_tile[kRows * kUtilityStride];
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

      float dv_carry[8] = {};
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
        if constexpr (Split) {
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
              utility_tile +
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
          const float gate = Split
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
              const int index = tile_index(route_offset, lane);
              const float dropped_probability = probability * dropout_scale;
              const __half high = __float2half_rn(dropped_probability);
              half_phase.route.probability.high[
                  index] = high;
              half_phase.route.probability.low[
                  index] = __float2half_rn(
                      dropped_probability - __half2float(high));
            }
            if ((mask & (kGradQuery | kGradKey)) != 0) {
              direct_vjp = scale * probability *
                  (dropout_scale * utility_tile[
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

        if constexpr ((TensorMask & kGradQuery) == 0) {
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

        if constexpr ((TensorMask & kGradKey) == 0) {
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
            const int output_tile = (1 - output_round) * kWarps + warp;
            const int output_route =
                (output_tile / (kValueDim / kTensorTile)) * kTensorTile;
            const int output_feature =
                (output_tile % (kValueDim / kTensorTile)) * kTensorTile;
            Mma acc;
            acc.mul<kRows>(
                half_phase.route.probability.high,
                half_phase.route.probability.low,
                half_phase.grad + output_feature, output_route, kInputStride);
            const bool defer = output_round == 1 && row_block > first_row_block;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              if (output_round == 0) acc.x[i] += dv_carry[i];
              if (defer) dv_carry[i] = acc.x[i];
            }
            // Permutation commutes with accumulation. Only reorder results
            // that leave the CTA, not the lower half retained for the next block.
            if (!defer) {
              acc.reorder();
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                const int route = route_start + output_route + Mma::row(i);
                const int feature = output_feature + Mma::col(i);
                if (route >= 1 && route < seq_len) {
                  const int64_t target =
                      ((static_cast<int64_t>(batch) * seq_len + route) * num_value_heads +
                       value_head) * kValueDim + feature;
                  atomicAdd(dv + target, acc.x[i]);
                }
              }
            }
          }
        }
        __syncthreads();

        if constexpr (TensorMask != 0) {
          constexpr int kSymbolStride = 40;
          __half* const symbol_tile =
              reinterpret_cast<__half*>(utility_tile);
          constexpr int kProbabilityElements = kRouteCount * kProbabilityStride;
          if ((mask & kGradValue) == 0) {
            for (int index = thread;
                 index < kProbabilityElements;
                 index += blockDim.x) {
              half_phase.route.probability.high[index] =
                  __float2half_rn(0.0f);
              half_phase.route.probability.low[index] =
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
              const int credit_index = tile_index(route_offset, row_offset);
              const __half high = __float2half_rn(credit);
              half_phase.route.probability.high[credit_index] =
                  high;
              half_phase.route.probability.low[credit_index] =
                  __float2half_rn(credit - __half2float(high));
            }
          }
          __syncthreads();

          // Signs are exactly representable in FP16. Splitting each FP32
          // credit into high and residual halves keeps Tensor Core
          // contractions close to the scalar FP32 accumulation.
          if constexpr ((TensorMask & kGradKey) != 0) {
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
              if (owns_output) {
                Mma acc;
                acc.mul<kRows>(
                    half_phase.route.probability.high,
                    half_phase.route.probability.low,
                    symbol_tile + output_bit, output_route, kSymbolStride);
                acc.reorder();
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                  const int position = route_start + output_route + Mma::row(i) - 1;
                  const int bit = output_bit + Mma::col(i);
                  if (bit < symbol_dim && position >= 0 && position < seq_len) {
                    const int64_t target =
                        ((static_cast<int64_t>(batch) * seq_len + position) *
                         num_heads + head) * symbol_dim + bit;
                    atomicAdd(dk + target, symbol_scale * acc.x[i]);
                  }
                }
              }
              // All readers must finish before the next symbol tile overwrites it.
              __syncthreads();
            }
          }

          if constexpr ((TensorMask & kGradQuery) != 0) {
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
              if (owns_output) {
                Mma acc;
                acc.mul<kRouteCount, true>(
                    half_phase.route.probability.high,
                    half_phase.route.probability.low,
                    symbol_tile + output_bit, output_row, kSymbolStride);
                acc.reorder();
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                  const int position = row_start + output_row + Mma::row(i);
                  const int bit = output_bit + Mma::col(i);
                  if (bit < symbol_dim && position < seq_len) {
                    const int64_t target =
                        ((static_cast<int64_t>(batch) * seq_len + position) *
                         num_heads + head) * symbol_dim + bit;
                    atomicAdd(dq + target, symbol_scale * acc.x[i]);
                  }
                }
              }
              // All readers must finish before the next symbol tile overwrites it.
              __syncthreads();
            }
          }
        }
      }
    }
  }
#endif
}


template <int TensorMask>
Grads launch(const Input& x, const Args& a, int mask,
             const Tensor& stats, const Tensor& prior) {
  const int b = x.q.size(0), t = x.q.size(1), h = x.q.size(2), d = x.q.size(3);
  const int hv = x.v.size(2), s = b * h;
  auto grad = gradients(x, mask);
  auto& [dq, dk, dv] = grad;
  if (t <= 1) return grad;
  const int rows = (t + kRows - 1) / kRows;
  const int diagonals = (t - 1 + kDiagonals - 1) / kDiagonals;
  const int pairs = s * ((diagonals + 1) / 2);
  int blocks = 1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, backward_kernel<TensorMask>, kThreads, 0));
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = std::min(pairs, std::max(1, blocks * sms));
  auto checkpoints = torch::empty({grid, rows, kDiagonals},
                                  x.q.options().dtype(torch::kFloat32));
  backward_kernel<TensorMask><<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.dy.data_ptr<c10::Half>(), x.v.data_ptr<c10::Half>(),
      x.pq.data_ptr<int32_t>(), x.pk.data_ptr<int32_t>(), x.seed.data_ptr<int64_t>(),
      prior.data_ptr<float>(), stats.data_ptr<float>(), checkpoints.data_ptr<float>(),
      dq.numel() ? dq.data_ptr<float>() : nullptr,
      dk.numel() ? dk.data_ptr<float>() : nullptr,
      dv.numel() ? dv.data_ptr<float>() : nullptr, s, t, h, hv, d, rows, diagonals,
      a.mismatch / float(d), .5f * a.mismatch / float(d), a.scale, a.dropout,
      1.0f / (1.0f - a.dropout), mask);
  finish(x, grad);
  return grad;
}
}  // namespace

Grads backward_fp16(const Input& x, const Args& a, int mask, const Tensor& prior) {
  auto stats = stats_fp16(x, a, prior);
  const int d = x.q.size(3);
  if (d >= 16 && (mask & 3) == 3) return launch<3>(x, a, mask, stats, prior);
  if (d >= 8 && (mask & 2)) return launch<2>(x, a, mask, stats, prior);
  if (d >= 16 && (mask & 1)) return launch<1>(x, a, mask, stats, prior);
  return launch<0>(x, a, mask, stats, prior);
}
}  // namespace rosa::soft
