#include "rosa_soft_vjp_common.cuh"

using namespace rosa_soft::cuda;

namespace {

constexpr int kRouteTile = 32;
constexpr int kSuffixTile = 32;
constexpr int kValueSharedStride = kRouteTile + 1;


template <int QueryTile>
struct TileTraits {
  static_assert(QueryTile == 16 || QueryTile == 32);
  static constexpr int kWorkspaceSuffixCapacity =
      QueryTile >= 32 ? 4 : 8;
  static constexpr int kValueTile = QueryTile >= 32 ? 32 : 64;
};


template <int QueryTile>
struct SharedView {
  int32_t* query_words;
  int32_t* key_words;
  float* gates;
  float* value_signs;
  float* grad_output;
  float* route_workspace;
  float* row_maximum;
  float* row_normalizer;
  float* row_expected_utility;
};


template <int QueryTile>
__host__ __device__ size_t shared_word_count() {
  constexpr int kWorkspaceSuffixCapacity =
      TileTraits<QueryTile>::kWorkspaceSuffixCapacity;
  constexpr int kValueTile = TileTraits<QueryTile>::kValueTile;
  constexpr int kQueryWords = QueryTile + kSuffixTile - 1;
  constexpr int kKeyWords = kRouteTile + kSuffixTile - 1;
  constexpr int kGateDiagonals = QueryTile + kRouteTile - 1;
  constexpr int kGateWords = kGateDiagonals * kQueryWords;
  constexpr int kValueWords = kValueSharedStride * kValueTile;
  constexpr int kGradOutputWords = QueryTile * kValueTile;
  constexpr int kRouteWorkspaceWords =
      kWorkspaceSuffixCapacity * QueryTile * kRouteTile;
  return kQueryWords + kKeyWords + kGateWords + kValueWords +
      kGradOutputWords + kRouteWorkspaceWords + 3 * QueryTile;
}


template <int QueryTile>
__device__ __forceinline__ SharedView<QueryTile> bind_shared(
    int32_t* storage) {
  constexpr int kWorkspaceSuffixCapacity =
      TileTraits<QueryTile>::kWorkspaceSuffixCapacity;
  constexpr int kValueTile = TileTraits<QueryTile>::kValueTile;
  constexpr int kQueryWords = QueryTile + kSuffixTile - 1;
  constexpr int kKeyWords = kRouteTile + kSuffixTile - 1;
  constexpr int kGateDiagonals = QueryTile + kRouteTile - 1;
  int32_t* query_words = storage;
  int32_t* key_words = query_words + kQueryWords;
  float* cursor = reinterpret_cast<float*>(key_words + kKeyWords);
  float* gates = cursor;
  cursor += kGateDiagonals * kQueryWords;
  float* value_signs = cursor;
  cursor += kValueSharedStride * kValueTile;
  float* grad_output = cursor;
  cursor += QueryTile * kValueTile;
  float* route_workspace = cursor;
  cursor += kWorkspaceSuffixCapacity * QueryTile * kRouteTile;
  float* row_maximum = cursor;
  cursor += QueryTile;
  float* row_normalizer = cursor;
  cursor += QueryTile;
  float* row_expected_utility = cursor;
  return {
      query_words,
      key_words,
      gates,
      value_signs,
      grad_output,
      route_workspace,
      row_maximum,
      row_normalizer,
      row_expected_utility};
}


template <int QueryTile>
__device__ __forceinline__ void stage_gate_tile(
    const int32_t* __restrict__ packed_query,
    const int32_t* __restrict__ packed_key,
    SharedView<QueryTile> shared,
    int query_base,
    int key_base,
    int seq_len,
    int symbol_dim,
    int suffix_count,
    float mismatch_scale,
    float inverse_symbol_dim) {
  constexpr int kQueryWords = QueryTile + kSuffixTile - 1;
  constexpr int kKeyWords = kRouteTile + kSuffixTile - 1;
  constexpr int kGateDiagonals = QueryTile + kRouteTile - 1;
  __syncthreads();
  for (int index = threadIdx.x; index < kQueryWords; index += blockDim.x) {
    const int position = query_base + index;
    shared.query_words[index] =
        position >= 0 && position < seq_len ? packed_query[position] : 0;
  }
  for (int index = threadIdx.x; index < kKeyWords; index += blockDim.x) {
    const int position = key_base + index;
    shared.key_words[index] =
        position >= 0 && position < seq_len ? packed_key[position] : 0;
  }
  __syncthreads();
  constexpr int kGateCount = kGateDiagonals * kQueryWords;
  for (int index = threadIdx.x; index < kGateCount; index += blockDim.x) {
    const int diagonal = index / kQueryWords;
    const int query_index = index - diagonal * kQueryWords;
    const int query_minus_key = diagonal - (kRouteTile - 1);
    const int key_index = query_index - query_minus_key;
    const int query_position = query_base + query_index;
    const int key_position = key_base + key_index;
    const int first_query_row = max(
        0,
        max(query_minus_key, query_index - (kSuffixTile - 1)));
    const int last_query_row = min(
        QueryTile - 1,
        min(
            query_minus_key + kRouteTile - 1,
            query_index - (kSuffixTile - 1) + suffix_count - 1));
    shared.gates[index] =
        first_query_row <= last_query_row &&
            query_position >= 0 && query_position < seq_len &&
            key_index >= 0 && key_index < kKeyWords &&
            key_position >= 0 && key_position < seq_len
        ? local_match_gate(
              static_cast<uint32_t>(shared.query_words[query_index]),
              static_cast<uint32_t>(shared.key_words[key_index]),
              symbol_dim,
              mismatch_scale,
              inverse_symbol_dim)
        : 0.0f;
  }
  __syncthreads();
}


template <typename scalar_t, int QueryTile>
__device__ __forceinline__ void stage_value_tile(
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ grad_output,
    SharedView<QueryTile> shared,
    int batch,
    int head,
    int value_head,
    int row_start,
    int route_start,
    int value_start,
    int seq_len,
    int num_heads,
    int num_value_heads,
    int value_dim,
    int value_count,
    bool need_value_signs) {
  constexpr int kValueTile = TileTraits<QueryTile>::kValueTile;
  __syncthreads();
  const int grad_count = QueryTile * value_count;
  for (int index = threadIdx.x; index < grad_count; index += blockDim.x) {
    const int row_offset = index / value_count;
    const int value_offset = index - row_offset * value_count;
    const int row = row_start + row_offset;
    float item = 0.0f;
    if (row < seq_len) {
      const int64_t source =
          ((static_cast<int64_t>(batch) * seq_len + row) * num_heads + head) *
              value_dim +
          value_start + value_offset;
      item = read_float(grad_output, source);
    }
    shared.grad_output[row_offset * kValueTile + value_offset] = item;
  }
  if (need_value_signs) {
    const int value_items = kRouteTile * value_count;
    for (int index = threadIdx.x; index < value_items; index += blockDim.x) {
      const int route_offset = index / value_count;
      const int value_offset = index - route_offset * value_count;
      const int route_position = route_start + route_offset;
      float item = 0.0f;
      if (route_position < seq_len) {
        const int64_t source =
            ((static_cast<int64_t>(batch) * seq_len + route_position) *
                 num_value_heads +
             value_head) *
                value_dim +
            value_start + value_offset;
        item = read_float(value, source) > 0.0f ? 1.0f : -1.0f;
      }
      shared.value_signs[
          value_offset * kValueSharedStride + route_offset] = item;
    }
  }
  __syncthreads();
}


template <int QueryTile>
__device__ __forceinline__ int tile_suffix_steps(
    int row_end,
    int route_start,
    int max_suffix_length) {
  const int route_end = min(row_end, route_start + kRouteTile - 1);
  return min(max_suffix_length, min(row_end + 1, route_end));
}


template <typename scalar_t, int QueryTile>
__global__ __launch_bounds__(QueryTile * kWarpSize, 1)
void tiled_streaming_vjp_kernel(
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
  constexpr int kRowsPerWarp = 1;
  constexpr int kValueTile = TileTraits<QueryTile>::kValueTile;
  constexpr int kQueryWords = QueryTile + kSuffixTile - 1;
  constexpr int kKeyWords = kRouteTile + kSuffixTile - 1;
  constexpr int kGateDiagonals = QueryTile + kRouteTile - 1;
  constexpr int kGateCount = kGateDiagonals * kQueryWords;
  constexpr int kDirectSuffixLimit = 4;
  extern __shared__ int32_t shared_storage[];
  const SharedView<QueryTile> shared =
      bind_shared<QueryTile>(shared_storage);
  const int tiles_per_head = (seq_len + QueryTile - 1) / QueryTile;
  const int tile_index = blockIdx.x % tiles_per_head;
  const int head = (blockIdx.x / tiles_per_head) % num_heads;
  const int batch = blockIdx.x / (tiles_per_head * num_heads);
  const int row_start = tile_index * QueryTile;
  const int row_end = min(seq_len - 1, row_start + QueryTile - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int value_head = head / (num_heads / num_value_heads);
  const int32_t* packed_query =
      packed_query_symbols +
      (static_cast<int64_t>(batch) * num_heads + head) * seq_len;
  const int32_t* packed_key =
      packed_key_symbols +
      (static_cast<int64_t>(batch) * num_heads + head) * seq_len;
  const float inverse_symbol_dim = 1.0f / static_cast<float>(symbol_dim);
  const bool needs_qk = (gradient_mask & (kGradQuery | kGradKey)) != 0;
  const bool needs_key = (gradient_mask & kGradKey) != 0;
  const bool needs_value = (gradient_mask & kGradValue) != 0;

  SoftmaxStats stats[kRowsPerWarp];
#pragma unroll
  for (int owned = 0; owned < kRowsPerWarp; ++owned) {
    const int row_offset = warp + owned * QueryTile;
    const int row = row_start + row_offset;
    stats[owned] = row < seq_len && lane == 0
        ? SoftmaxStats{kNullScore * scale, 1.0f, 0.0f}
        : SoftmaxStats{-FLT_MAX, 0.0f, 0.0f};
  }

  // Pass one computes exact online softmax statistics and E_p[dL/dp].
  for (int route_start = 1; route_start <= row_end; route_start += kRouteTile) {
    float products[kRowsPerWarp];
    float raw_scores[kRowsPerWarp];
#pragma unroll
    for (int owned = 0; owned < kRowsPerWarp; ++owned) {
      products[owned] = 1.0f;
      raw_scores[owned] = 0.0f;
    }
    const int maximum_steps =
        tile_suffix_steps<QueryTile>(row_end, route_start, max_suffix_length);
    for (int suffix_start = 0;
         suffix_start < maximum_steps;
         suffix_start += kSuffixTile) {
      const int query_base =
          row_start - suffix_start - (kSuffixTile - 1);
      const int key_base =
          route_start - 1 - suffix_start - (kSuffixTile - 1);
      const int suffix_count =
          min(kSuffixTile, maximum_steps - suffix_start);
      stage_gate_tile<QueryTile>(
          packed_query,
          packed_key,
          shared,
          query_base,
          key_base,
          seq_len,
          symbol_dim,
          suffix_count,
          mismatch_scale,
          inverse_symbol_dim);
#pragma unroll
      for (int owned = 0; owned < kRowsPerWarp; ++owned) {
        const int row_offset = warp + owned * QueryTile;
        const int row = row_start + row_offset;
        const int route_position = route_start + lane;
        const int suffix_steps = row < seq_len && route_position <= row
            ? min(max_suffix_length, min(row + 1, route_position))
            : 0;
#pragma unroll 4
        for (int local_suffix = 0;
             local_suffix < suffix_count;
             ++local_suffix) {
          if (suffix_start + local_suffix < suffix_steps) {
            const int query_index =
                row_offset + kSuffixTile - 1 - local_suffix;
            const int diagonal =
                row_offset - lane + (kRouteTile - 1);
            const float gate =
                shared.gates[diagonal * kQueryWords + query_index];
            products[owned] *= gate;
            raw_scores[owned] += products[owned];
          }
        }
      }
    }

    float utilities[kRowsPerWarp];
#pragma unroll
    for (int owned = 0; owned < kRowsPerWarp; ++owned) {
      utilities[owned] = 0.0f;
    }
    if (needs_qk) {
      for (int value_start = 0; value_start < value_dim; value_start += kValueTile) {
        const int value_count = min(kValueTile, value_dim - value_start);
        stage_value_tile<scalar_t, QueryTile>(
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
#pragma unroll
        for (int owned = 0; owned < kRowsPerWarp; ++owned) {
          const int row_offset = warp + owned * QueryTile;
          const int row = row_start + row_offset;
          const int route_position = route_start + lane;
          if (row < seq_len && route_position <= row) {
            for (int value_offset = 0;
                 value_offset < value_count;
                 ++value_offset) {
              utilities[owned] +=
                  shared.grad_output[row_offset * kValueTile + value_offset] *
                  shared.value_signs[
                      value_offset * kValueSharedStride + lane];
            }
          }
        }
      }
    }

#pragma unroll
    for (int owned = 0; owned < kRowsPerWarp; ++owned) {
      const int row_offset = warp + owned * QueryTile;
      const int row = row_start + row_offset;
      const int route_position = route_start + lane;
      if (row < seq_len && route_position <= row) {
        const ScoreTransform transformed = transform_score(raw_scores[owned]);
        const float dropout_scale = attention_dropout_scale(
            dropout_seed,
            dropout_p,
            inverse_keep_probability,
            batch,
            head,
            row,
            route_position);
        stats[owned] = append_item(
            stats[owned],
            transformed.route_score * scale - logf(static_cast<float>(row)),
            dropout_scale * utilities[owned]);
      }
    }
  }

#pragma unroll
  for (int owned = 0; owned < kRowsPerWarp; ++owned) {
    const int row_offset = warp + owned * QueryTile;
    const int row = row_start + row_offset;
    const SoftmaxStats reduced = warp_reduce_stats(stats[owned]);
    if (lane == 0 && row < seq_len) {
      shared.row_maximum[row_offset] = reduced.maximum;
      shared.row_normalizer[row_offset] = reduced.normalizer;
      shared.row_expected_utility[row_offset] =
          reduced.utility_numerator / reduced.normalizer;
    }
  }
  __syncthreads();

  // Pass two recomputes each score tile, then immediately consumes its VJP.
  for (int route_start = 1; route_start <= row_end; route_start += kRouteTile) {
    float products[kRowsPerWarp];
    float raw_scores[kRowsPerWarp];
#pragma unroll
    for (int owned = 0; owned < kRowsPerWarp; ++owned) {
      products[owned] = 1.0f;
      raw_scores[owned] = 0.0f;
    }
    const int maximum_steps =
        tile_suffix_steps<QueryTile>(row_end, route_start, max_suffix_length);
    for (int suffix_start = 0;
         suffix_start < maximum_steps;
         suffix_start += kSuffixTile) {
      const int query_base =
          row_start - suffix_start - (kSuffixTile - 1);
      const int key_base =
          route_start - 1 - suffix_start - (kSuffixTile - 1);
      const int suffix_count =
          min(kSuffixTile, maximum_steps - suffix_start);
      stage_gate_tile<QueryTile>(
          packed_query,
          packed_key,
          shared,
          query_base,
          key_base,
          seq_len,
          symbol_dim,
          suffix_count,
          mismatch_scale,
          inverse_symbol_dim);
#pragma unroll
      for (int owned = 0; owned < kRowsPerWarp; ++owned) {
        const int row_offset = warp + owned * QueryTile;
        const int row = row_start + row_offset;
        const int route_position = route_start + lane;
        const int suffix_steps = row < seq_len && route_position <= row
            ? min(max_suffix_length, min(row + 1, route_position))
            : 0;
#pragma unroll 4
        for (int local_suffix = 0;
             local_suffix < suffix_count;
             ++local_suffix) {
          if (suffix_start + local_suffix < suffix_steps) {
            const int query_index =
                row_offset + kSuffixTile - 1 - local_suffix;
            const int diagonal =
                row_offset - lane + (kRouteTile - 1);
            const float gate =
                shared.gates[diagonal * kQueryWords + query_index];
            products[owned] *= gate;
            raw_scores[owned] += products[owned];
          }
        }
      }
    }

    float probabilities[kRowsPerWarp];
    float dropout_scales[kRowsPerWarp];
    float utilities[kRowsPerWarp];
#pragma unroll
    for (int owned = 0; owned < kRowsPerWarp; ++owned) {
      const int row_offset = warp + owned * QueryTile;
      const int row = row_start + row_offset;
      const int route_position = route_start + lane;
      probabilities[owned] = 0.0f;
      dropout_scales[owned] = 0.0f;
      utilities[owned] = 0.0f;
      if (row < seq_len && route_position <= row) {
        const ScoreTransform transformed = transform_score(raw_scores[owned]);
        probabilities[owned] = __expf(
            transformed.route_score * scale -
            logf(static_cast<float>(row)) -
            shared.row_maximum[row_offset]) /
            shared.row_normalizer[row_offset];
        dropout_scales[owned] = attention_dropout_scale(
            dropout_seed,
            dropout_p,
            inverse_keep_probability,
            batch,
            head,
            row,
            route_position);
      }
      shared.route_workspace[row_offset * kRouteTile + lane] =
          probabilities[owned] * dropout_scales[owned];
    }

    if (needs_qk || needs_value) {
      for (int value_start = 0; value_start < value_dim; value_start += kValueTile) {
        const int value_count = min(kValueTile, value_dim - value_start);
        stage_value_tile<scalar_t, QueryTile>(
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
#pragma unroll
          for (int owned = 0; owned < kRowsPerWarp; ++owned) {
            const int row_offset = warp + owned * QueryTile;
            const int row = row_start + row_offset;
            const int route_position = route_start + lane;
            if (row < seq_len && route_position <= row) {
              for (int value_offset = 0;
                   value_offset < value_count;
                   ++value_offset) {
                utilities[owned] +=
                    shared.grad_output[row_offset * kValueTile + value_offset] *
                    shared.value_signs[
                        value_offset * kValueSharedStride + lane];
              }
            }
          }
        }
        if (needs_value) {
          const int item_count = kRouteTile * value_count;
          for (int index = threadIdx.x; index < item_count; index += blockDim.x) {
            const int route_offset = index / value_count;
            const int value_offset = index - route_offset * value_count;
            const int route_position = route_start + route_offset;
            if (route_position <= row_end) {
              float contribution = 0.0f;
#pragma unroll
              for (int row_offset = 0;
                   row_offset < QueryTile;
                   ++row_offset) {
                contribution +=
                    shared.route_workspace[
                        row_offset * kRouteTile + route_offset] *
                    shared.grad_output[
                        row_offset * kValueTile + value_offset];
              }
              const int64_t target =
                  ((static_cast<int64_t>(batch) * seq_len + route_position) *
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

    if (needs_qk) {
      float raw_score_vjp[kRowsPerWarp];
      float prefixes[kRowsPerWarp];
      float tails[kRowsPerWarp];
#pragma unroll
      for (int owned = 0; owned < kRowsPerWarp; ++owned) {
        const int row_offset = warp + owned * QueryTile;
        const int row = row_start + row_offset;
        const int route_position = route_start + lane;
        raw_score_vjp[owned] = 0.0f;
        if (row < seq_len && route_position <= row) {
          const ScoreTransform transformed = transform_score(raw_scores[owned]);
          const float route_score_vjp =
              scale * probabilities[owned] *
              (dropout_scales[owned] * utilities[owned] -
               shared.row_expected_utility[row_offset]);
          raw_score_vjp[owned] =
              route_score_vjp * transformed.raw_vjp_multiplier;
        }
        prefixes[owned] = 1.0f;
        tails[owned] = raw_scores[owned];
      }

      for (int suffix_start = 0;
           suffix_start < maximum_steps;
           suffix_start += kSuffixTile) {
        __syncthreads();
        const int query_base =
            row_start - suffix_start - (kSuffixTile - 1);
        const int key_base =
            route_start - 1 - suffix_start - (kSuffixTile - 1);
        const int suffix_count =
            min(kSuffixTile, maximum_steps - suffix_start);
        // Tiny tails cost less to contract directly than to clear and fill
        // the shared local-gate adjoint.
        const bool use_direct_suffix_vjp =
            suffix_count == 1 ||
            (suffix_count <= kDirectSuffixLimit &&
             suffix_count * symbol_dim <= 16);
        if (!use_direct_suffix_vjp) {
          for (int index = threadIdx.x;
               index < kGateCount;
               index += blockDim.x) {
            shared.route_workspace[index] = 0.0f;
          }
          __syncthreads();
        }
        if (maximum_steps > kSuffixTile) {
          stage_gate_tile<QueryTile>(
              packed_query,
              packed_key,
              shared,
              query_base,
              key_base,
              seq_len,
              symbol_dim,
              suffix_count,
              mismatch_scale,
              inverse_symbol_dim);
        }
#pragma unroll
        for (int owned = 0; owned < kRowsPerWarp; ++owned) {
          const int row_offset = warp + owned * QueryTile;
          const int row = row_start + row_offset;
          const int route_position = route_start + lane;
          const int suffix_steps = row < seq_len && route_position <= row
              ? min(max_suffix_length, min(row + 1, route_position))
              : 0;
          const int diagonal = row_offset - lane + (kRouteTile - 1);
#pragma unroll 4
          for (int local_suffix = 0;
               local_suffix < suffix_count;
               ++local_suffix) {
            const int suffix_offset = suffix_start + local_suffix;
            const int query_index =
                row_offset + kSuffixTile - 1 - local_suffix;
            float local_scale = 0.0f;
            if (suffix_offset < suffix_steps) {
              prefixes[owned] *=
                  shared.gates[diagonal * kQueryWords + query_index];
              local_scale =
                  mismatch_scale * inverse_symbol_dim *
                  raw_score_vjp[owned] * fmaxf(tails[owned], 0.0f);
              tails[owned] -= prefixes[owned];
            }
            if (use_direct_suffix_vjp) {
              shared.route_workspace[
                  (local_suffix * QueryTile + row_offset) * kRouteTile +
                  lane] = local_scale;
            } else if (local_scale != 0.0f) {
              // Every route-suffix term that reaches this local match gate
              // shares the same Q/K Jacobian.
              atomicAdd(
                  &shared.route_workspace[
                      diagonal * kQueryWords + query_index],
                  local_scale);
            }
          }
        }
        __syncthreads();
        if (use_direct_suffix_vjp) {
          if ((gradient_mask & kGradQuery) != 0) {
            const int query_items =
                suffix_count * QueryTile * symbol_dim;
            for (int index = threadIdx.x;
                 index < query_items;
                 index += blockDim.x) {
              const int suffix = index / (QueryTile * symbol_dim);
              const int within_suffix =
                  index - suffix * QueryTile * symbol_dim;
              const int row_offset = within_suffix / symbol_dim;
              const int bit = within_suffix - row_offset * symbol_dim;
              const int query_index =
                  row_offset + kSuffixTile - 1 - suffix;
              const int query_position = query_base + query_index;
              if (query_position >= 0 && query_position < seq_len) {
                float contribution = 0.0f;
#pragma unroll
                for (int route_offset = 0;
                     route_offset < kRouteTile;
                     ++route_offset) {
                  const float local_scale = shared.route_workspace[
                      (suffix * QueryTile + row_offset) * kRouteTile +
                      route_offset];
                  const int key_index =
                      route_offset + kSuffixTile - 1 - suffix;
                  const uint32_t key_word =
                      static_cast<uint32_t>(shared.key_words[key_index]);
                  contribution +=
                      0.5f * local_scale *
                      static_cast<float>(sign_from_bit(key_word, bit));
                }
                if (contribution != 0.0f) {
                  const int64_t query_target =
                      ((static_cast<int64_t>(batch) * seq_len +
                        query_position) *
                           num_heads +
                       head) *
                          symbol_dim +
                      bit;
                  atomicAdd(&grad_query[query_target], contribution);
                }
              }
            }
          }
          if (needs_key) {
            const int key_items =
                suffix_count * kRouteTile * symbol_dim;
            for (int index = threadIdx.x;
                 index < key_items;
                 index += blockDim.x) {
              const int suffix = index / (kRouteTile * symbol_dim);
              const int within_suffix =
                  index - suffix * kRouteTile * symbol_dim;
              const int route_offset = within_suffix / symbol_dim;
              const int bit = within_suffix - route_offset * symbol_dim;
              const int key_index =
                  route_offset + kSuffixTile - 1 - suffix;
              const int key_position = key_base + key_index;
              if (key_position >= 0 && key_position < seq_len) {
                float contribution = 0.0f;
#pragma unroll
                for (int row_offset = 0;
                     row_offset < QueryTile;
                     ++row_offset) {
                  const float local_scale = shared.route_workspace[
                      (suffix * QueryTile + row_offset) * kRouteTile +
                      route_offset];
                  const int query_index =
                      row_offset + kSuffixTile - 1 - suffix;
                  const uint32_t query_word =
                      static_cast<uint32_t>(shared.query_words[query_index]);
                  contribution +=
                      0.5f * local_scale *
                      static_cast<float>(sign_from_bit(query_word, bit));
                }
                if (contribution != 0.0f) {
                  const int64_t target =
                      ((static_cast<int64_t>(batch) * num_heads + head) *
                           symbol_dim +
                       bit) *
                          seq_len +
                      key_position;
                  atomicAdd(&grad_key[target], contribution);
                }
              }
            }
          }
        } else {
          if ((gradient_mask & kGradQuery) != 0) {
            const int first_query_index = kSuffixTile - suffix_count;
            const int active_query_words = QueryTile + suffix_count - 1;
            const int query_items = active_query_words * symbol_dim;
            for (int index = threadIdx.x;
                 index < query_items;
                 index += blockDim.x) {
              const int local_query_index = index / symbol_dim;
              const int bit = index - local_query_index * symbol_dim;
              const int query_index = first_query_index + local_query_index;
              const int query_position = query_base + query_index;
              if (query_position >= 0 && query_position < seq_len) {
                float contribution = 0.0f;
                if (suffix_count == kSuffixTile) {
#pragma unroll
                  for (int diagonal = 0;
                       diagonal < kGateDiagonals;
                       ++diagonal) {
                    const int query_minus_key = diagonal - (kRouteTile - 1);
                    const int key_index = query_index - query_minus_key;
                    if (key_index >= 0 && key_index < kKeyWords) {
                      const float local_scale = shared.route_workspace[
                          diagonal * kQueryWords + query_index];
                      const uint32_t key_word =
                          static_cast<uint32_t>(shared.key_words[key_index]);
                      contribution +=
                          0.5f * local_scale *
                          static_cast<float>(sign_from_bit(key_word, bit));
                    }
                  }
                } else {
                  const int first_row_offset =
                      max(0, query_index - (kSuffixTile - 1));
                  const int last_row_offset = min(
                      QueryTile - 1,
                      query_index - (kSuffixTile - 1) + suffix_count - 1);
                  const int first_diagonal = first_row_offset;
                  const int last_diagonal =
                      last_row_offset + (kRouteTile - 1);
                  for (int diagonal = first_diagonal;
                       diagonal <= last_diagonal;
                       ++diagonal) {
                    const int query_minus_key = diagonal - (kRouteTile - 1);
                    const int key_index = query_index - query_minus_key;
                    if (key_index >= 0 && key_index < kKeyWords) {
                      const float local_scale = shared.route_workspace[
                          diagonal * kQueryWords + query_index];
                      const uint32_t key_word =
                          static_cast<uint32_t>(shared.key_words[key_index]);
                      contribution +=
                          0.5f * local_scale *
                          static_cast<float>(sign_from_bit(key_word, bit));
                    }
                  }
                }
                if (contribution != 0.0f) {
                  const int64_t query_target =
                      ((static_cast<int64_t>(batch) * seq_len + query_position) *
                           num_heads +
                       head) *
                          symbol_dim +
                      bit;
                  atomicAdd(&grad_query[query_target], contribution);
                }
              }
            }
          }
          if (needs_key) {
            const int first_key_index = kSuffixTile - suffix_count;
            const int active_key_words = kRouteTile + suffix_count - 1;
            const int key_items = active_key_words * symbol_dim;
            for (int index = threadIdx.x;
                 index < key_items;
                 index += blockDim.x) {
              const int local_key_index = index / symbol_dim;
              const int bit = index - local_key_index * symbol_dim;
              const int key_index = first_key_index + local_key_index;
              const int key_position = key_base + key_index;
              if (key_position >= 0 && key_position < seq_len) {
                float contribution = 0.0f;
                if (suffix_count == kSuffixTile) {
#pragma unroll
                  for (int diagonal = 0;
                       diagonal < kGateDiagonals;
                       ++diagonal) {
                    const int query_minus_key = diagonal - (kRouteTile - 1);
                    const int query_index = key_index + query_minus_key;
                    if (query_index >= 0 && query_index < kQueryWords) {
                      const float local_scale = shared.route_workspace[
                          diagonal * kQueryWords + query_index];
                      const uint32_t query_word =
                          static_cast<uint32_t>(shared.query_words[query_index]);
                      contribution +=
                          0.5f * local_scale *
                          static_cast<float>(sign_from_bit(query_word, bit));
                    }
                  }
                } else {
                  const int first_route_offset =
                      max(0, key_index - (kSuffixTile - 1));
                  const int last_route_offset = min(
                      kRouteTile - 1,
                      key_index - (kSuffixTile - 1) + suffix_count - 1);
                  const int first_diagonal =
                      (kRouteTile - 1) - last_route_offset;
                  const int last_diagonal =
                      QueryTile - 1 + (kRouteTile - 1) - first_route_offset;
                  for (int diagonal = first_diagonal;
                       diagonal <= last_diagonal;
                       ++diagonal) {
                    const int query_minus_key = diagonal - (kRouteTile - 1);
                    const int query_index = key_index + query_minus_key;
                    if (query_index >= 0 && query_index < kQueryWords) {
                      const float local_scale = shared.route_workspace[
                          diagonal * kQueryWords + query_index];
                      const uint32_t query_word =
                          static_cast<uint32_t>(shared.query_words[query_index]);
                      contribution +=
                          0.5f * local_scale *
                          static_cast<float>(sign_from_bit(query_word, bit));
                    }
                  }
                }
                if (contribution != 0.0f) {
                  const int64_t target =
                      ((static_cast<int64_t>(batch) * num_heads + head) *
                           symbol_dim +
                       bit) *
                          seq_len +
                      key_position;
                  atomicAdd(&grad_key[target], contribution);
                }
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }
}


template <typename scalar_t, int QueryTile>
void launch_streaming_kernel(
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
  const int tiles_per_head = (seq_len + QueryTile - 1) / QueryTile;
  const int64_t block_count64 =
      static_cast<int64_t>(batch_size) * num_heads * tiles_per_head;
  TORCH_CHECK(
      block_count64 <= std::numeric_limits<int>::max(),
      "streaming VJP grid is too large");
  const size_t shared_bytes =
      shared_word_count<QueryTile>() * sizeof(int32_t);
  TORCH_CHECK(shared_bytes <= 48 * 1024, "streaming VJP shared tile is too large");
  tiled_streaming_vjp_kernel<scalar_t, QueryTile><<<
      static_cast<int>(block_count64),
      QueryTile * kWarpSize,
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
rosa_soft_streaming_vjp_cuda(
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
    int query_tile_size) {
  TORCH_CHECK(
      query_tile_size == 16 || query_tile_size == 32,
      "streaming VJP query tile size must be 16 or 32");
  const c10::cuda::CUDAGuard device_guard(query.device());
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
            {batch_size, num_heads, symbol_dim, seq_len},
            float_options)
      : torch::empty({0}, float_options);
  torch::Tensor grad_value = (gradient_mask & kGradValue) != 0
      ? torch::zeros(value.sizes(), float_options)
      : torch::empty({0}, float_options);
  const float inverse_keep_probability = 1.0f / (1.0f - dropout_p);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_ROSA_FLOAT_TYPES(
      query.scalar_type(),
      "rosa_soft_streaming_vjp_cuda",
      [&] {
        if (query_tile_size == 16) {
          launch_streaming_kernel<scalar_t, 16>(
              value, grad_output, packed_query_symbols, packed_key_symbols,
              grad_query, grad_key_accumulator, grad_value, batch_size,
              seq_len, num_heads, symbol_dim, num_value_heads, value_dim,
              static_cast<int>(max_suffix_length), scale, dropout_p,
              inverse_keep_probability, mismatch_scale, dropout_seed,
              gradient_mask, stream);
        } else {
          launch_streaming_kernel<scalar_t, 32>(
              value, grad_output, packed_query_symbols, packed_key_symbols,
              grad_query, grad_key_accumulator, grad_value, batch_size,
              seq_len, num_heads, symbol_dim, num_value_heads, value_dim,
              static_cast<int>(max_suffix_length), scale, dropout_p,
              inverse_keep_probability, mismatch_scale, dropout_seed,
              gradient_mask, stream);
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
