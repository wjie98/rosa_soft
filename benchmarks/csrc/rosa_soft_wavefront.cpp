#include <cmath>
#include <cstdint>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


torch::Tensor rosa_soft_wavefront_scores_cuda(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    float mismatch_scale,
    int plan);

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
    int compute_utility);

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
    int compute_utility);

torch::Tensor rosa_soft_wavefront_log_gate_vjp_cuda(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& raw_scores,
    const torch::Tensor& raw_score_vjp,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    float mismatch_scale);

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
    int execution_plan);

torch::Tensor rosa_soft_unbounded_group_scores_cuda(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t group_start,
    int64_t group_width,
    float mismatch_scale);

torch::Tensor rosa_soft_unbounded_replay_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t group_size,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int compute_utility);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_unbounded_replay_vjp_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t group_size,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_grouped_checkpoint_reverse_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    const torch::Tensor& row_stats,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_grouped_checkpoint_reverse_tensor_symbols_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    const torch::Tensor& row_stats,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask,
    int tensor_symbol_mask,
    int specialized_replay);

torch::Tensor rosa_soft_macro_checkpoint_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int symbol_dim,
    int macro_diagonals,
    int slab_size,
    float scale,
    float dropout_p,
    float mismatch_scale);

namespace {

torch::Tensor unbounded_group_scores(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t group_start,
    int64_t group_width,
    double mismatch_scale) {
  TORCH_CHECK(
      packed_query_symbols.is_cuda() &&
          packed_query_symbols.is_contiguous(),
      "packed_query_symbols must be contiguous CUDA");
  TORCH_CHECK(
      packed_key_symbols.is_cuda() && packed_key_symbols.is_contiguous(),
      "packed_key_symbols must be contiguous CUDA");
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32,
      "packed symbols must be int32");
  TORCH_CHECK(
      packed_query_symbols.dim() == 3 &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes(),
      "packed symbols must have matching [B,H,T] shapes");
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(group_start >= 1);
  TORCH_CHECK(group_width >= 1 && group_width <= 1024);
  TORCH_CHECK(
      group_start < packed_query_symbols.size(2),
      "group_start must name a nonempty causal diagonal");
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  const c10::cuda::CUDAGuard device_guard(
      packed_query_symbols.device());
  return rosa_soft_unbounded_group_scores_cuda(
      packed_query_symbols,
      packed_key_symbols,
      symbol_dim,
      group_start,
      std::min<int64_t>(
          group_width,
          packed_query_symbols.size(2) - group_start),
      static_cast<float>(mismatch_scale));
}


torch::Tensor unbounded_replay_stats(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t group_size,
    double scale,
    double dropout_p,
    double mismatch_scale) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous());
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous());
  TORCH_CHECK(value.dim() == 4 && grad_output.dim() == 4);
  TORCH_CHECK(value.scalar_type() == grad_output.scalar_type());
  TORCH_CHECK(
      value.scalar_type() == torch::kFloat32 ||
          value.scalar_type() == torch::kFloat16 ||
          value.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(
      packed_query_symbols.is_cuda() &&
          packed_query_symbols.is_contiguous());
  TORCH_CHECK(
      packed_key_symbols.is_cuda() && packed_key_symbols.is_contiguous());
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32);
  TORCH_CHECK(
      packed_query_symbols.dim() == 3 &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes());
  const int64_t batch_size = packed_query_symbols.size(0);
  const int64_t num_heads = packed_query_symbols.size(1);
  const int64_t seq_len = packed_query_symbols.size(2);
  TORCH_CHECK(value.size(0) == batch_size && value.size(1) == seq_len);
  TORCH_CHECK(value.size(2) > 0 && num_heads % value.size(2) == 0);
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {batch_size, seq_len, num_heads, value.size(3)}));
  TORCH_CHECK(dropout_seed.is_cuda() && dropout_seed.is_contiguous());
  TORCH_CHECK(dropout_seed.scalar_type() == torch::kInt64);
  TORCH_CHECK(dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0));
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(
      group_size >= 1 && group_size <= seq_len,
      "group_size must be in [1, sequence_length]");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0);
  TORCH_CHECK(std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  const c10::cuda::CUDAGuard device_guard(value.device());
  return rosa_soft_unbounded_replay_stats_cuda(
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      symbol_dim,
      group_size,
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      1);
}


torch::Tensor macro_checkpoint_stats(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t macro_diagonals,
    int64_t slab_size,
    double scale,
    double dropout_p,
    double mismatch_scale) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&value, "value"},
           {&grad_output, "grad_output"},
           {&packed_query_symbols, "packed_query_symbols"},
           {&packed_key_symbols, "packed_key_symbols"},
           {&dropout_seed, "dropout_seed"}}) {
    TORCH_CHECK(item.first->is_cuda(), item.second, " must be CUDA");
    TORCH_CHECK(
        item.first->is_contiguous(), item.second, " must be contiguous");
    TORCH_CHECK(item.first->device() == value.device());
  }
  TORCH_CHECK(
      value.scalar_type() == torch::kFloat16 &&
          grad_output.scalar_type() == torch::kFloat16,
      "macro checkpoint stats currently requires float16");
  TORCH_CHECK(value.dim() == 4 && value.size(3) == 64);
  TORCH_CHECK(grad_output.dim() == 4 && grad_output.size(3) == 64);
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32);
  TORCH_CHECK(
      packed_query_symbols.dim() == 3 &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes());
  const int64_t batch_size = packed_query_symbols.size(0);
  const int64_t num_heads = packed_query_symbols.size(1);
  const int64_t seq_len = packed_query_symbols.size(2);
  TORCH_CHECK(
      value.size(0) == batch_size && value.size(1) == seq_len &&
          value.size(2) > 0 && num_heads % value.size(2) == 0);
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {batch_size, seq_len, num_heads, 64}));
  TORCH_CHECK(dropout_seed.scalar_type() == torch::kInt64);
  TORCH_CHECK(dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0));
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(
      macro_diagonals == 32 || macro_diagonals == 64 ||
          macro_diagonals == 96,
      "macro_diagonals must be 32, 64, or 96");
  TORCH_CHECK(slab_size >= 1 && slab_size <= seq_len);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0);
  TORCH_CHECK(
      std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  const c10::cuda::CUDAGuard device_guard(value.device());
  return rosa_soft_macro_checkpoint_stats_cuda(
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      static_cast<int>(symbol_dim),
      static_cast<int>(macro_diagonals),
      static_cast<int>(slab_size),
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
unbounded_replay_vjp(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t group_size,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t gradient_mask) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&query, "query"},
           {&key, "key"},
           {&value, "value"},
           {&grad_output, "grad_output"},
           {&packed_query_symbols, "packed_query_symbols"},
           {&packed_key_symbols, "packed_key_symbols"},
           {&dropout_seed, "dropout_seed"}}) {
    TORCH_CHECK(item.first->is_cuda(), item.second, " must be CUDA");
    TORCH_CHECK(item.first->is_contiguous(), item.second, " must be contiguous");
    TORCH_CHECK(item.first->device() == query.device());
  }
  TORCH_CHECK(query.dim() == 4 && key.sizes() == query.sizes());
  TORCH_CHECK(value.dim() == 4 && grad_output.dim() == 4);
  TORCH_CHECK(query.size(3) >= 1 && query.size(3) <= 32);
  TORCH_CHECK(value.size(0) == query.size(0));
  TORCH_CHECK(value.size(1) == query.size(1));
  TORCH_CHECK(value.size(2) > 0 && query.size(2) % value.size(2) == 0);
  TORCH_CHECK(value.size(3) > 0);
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(1), query.size(2), value.size(3)}));
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type() &&
          query.scalar_type() == grad_output.scalar_type());
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat32 ||
          query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32);
  TORCH_CHECK(
      packed_query_symbols.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(2), query.size(1)}));
  TORCH_CHECK(packed_key_symbols.sizes() == packed_query_symbols.sizes());
  TORCH_CHECK(dropout_seed.scalar_type() == torch::kInt64);
  TORCH_CHECK(dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0));
  TORCH_CHECK(
      group_size >= 1 && group_size <= query.size(1),
      "group_size must be in [1, sequence_length]");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0);
  TORCH_CHECK(std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(gradient_mask >= 1 && gradient_mask <= 7);
  const c10::cuda::CUDAGuard device_guard(query.device());
  return rosa_soft_unbounded_replay_vjp_cuda(
      query,
      key,
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      group_size,
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      static_cast<int>(gradient_mask));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
grouped_checkpoint_reverse(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    const torch::Tensor& row_stats,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t gradient_mask) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&query, "query"},
           {&key, "key"},
           {&value, "value"},
           {&grad_output, "grad_output"},
           {&packed_query_symbols, "packed_query_symbols"},
           {&packed_key_symbols, "packed_key_symbols"},
           {&dropout_seed, "dropout_seed"},
           {&row_stats, "row_stats"}}) {
    TORCH_CHECK(item.first->is_cuda(), item.second, " must be CUDA");
    TORCH_CHECK(
        item.first->is_contiguous(), item.second, " must be contiguous");
    TORCH_CHECK(item.first->device() == query.device());
  }
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 &&
          key.scalar_type() == torch::kFloat16 &&
          value.scalar_type() == torch::kFloat16 &&
          grad_output.scalar_type() == torch::kFloat16,
      "grouped checkpoint prototype currently requires float16");
  TORCH_CHECK(query.dim() == 4 && key.sizes() == query.sizes());
  TORCH_CHECK(query.size(3) >= 1 && query.size(3) <= 32);
  TORCH_CHECK(
      value.dim() == 4 && value.size(0) == query.size(0) &&
          value.size(1) == query.size(1) && value.size(2) > 0 &&
          query.size(2) % value.size(2) == 0 && value.size(3) == 64,
      "grouped checkpoint prototype requires value width 64");
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(1), query.size(2), 64}));
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_query_symbols.sizes() == torch::IntArrayRef(
              {query.size(0), query.size(2), query.size(1)}) &&
          packed_key_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes());
  TORCH_CHECK(
      row_stats.scalar_type() == torch::kFloat32 &&
          row_stats.sizes() == torch::IntArrayRef(
              {query.size(0), query.size(2), query.size(1), 3}),
      "row_stats must be contiguous float32 [B,H,T,3]");
  TORCH_CHECK(
      dropout_seed.scalar_type() == torch::kInt64 &&
          dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0));
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0);
  TORCH_CHECK(
      std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0);
  TORCH_CHECK(
      std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(gradient_mask >= 1 && gradient_mask <= 7);
  const c10::cuda::CUDAGuard device_guard(query.device());
  return rosa_soft_grouped_checkpoint_reverse_cuda(
      query,
      key,
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      row_stats,
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      static_cast<int>(gradient_mask));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
grouped_checkpoint_reverse_tensor_symbols(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    const torch::Tensor& row_stats,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t gradient_mask,
    int64_t tensor_symbol_mask,
    bool specialized_replay) {
  TORCH_CHECK(
      query.is_cuda() && query.is_contiguous() &&
          key.is_cuda() && key.is_contiguous() &&
          value.is_cuda() && value.is_contiguous() &&
          grad_output.is_cuda() && grad_output.is_contiguous() &&
          packed_query_symbols.is_cuda() &&
          packed_query_symbols.is_contiguous() &&
          packed_key_symbols.is_cuda() && packed_key_symbols.is_contiguous() &&
          dropout_seed.is_cuda() && dropout_seed.is_contiguous() &&
          row_stats.is_cuda() && row_stats.is_contiguous(),
      "tensor-symbol reverse inputs must be contiguous CUDA tensors");
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 &&
          key.scalar_type() == torch::kFloat16 &&
          value.scalar_type() == torch::kFloat16 &&
          grad_output.scalar_type() == torch::kFloat16 &&
          value.size(3) == 64,
      "tensor-symbol reverse requires float16 and value width 64");
  TORCH_CHECK(gradient_mask >= 1 && gradient_mask <= 7);
  TORCH_CHECK(
      tensor_symbol_mask >= 0 && tensor_symbol_mask <= 3,
      "tensor_symbol_mask must be in [0, 3]");
  TORCH_CHECK(
      !specialized_replay || tensor_symbol_mask == 0 ||
          tensor_symbol_mask == 2 || tensor_symbol_mask == 3,
      "unsupported tensor-symbol mask for specialized replay");
  const c10::cuda::CUDAGuard device_guard(query.device());
  return rosa_soft_grouped_checkpoint_reverse_tensor_symbols_cuda(
      query,
      key,
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      row_stats,
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      static_cast<int>(gradient_mask),
      static_cast<int>(tensor_symbol_mask),
      specialized_replay ? 1 : 0);
}


torch::Tensor wavefront_scores(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    double mismatch_scale,
    int64_t plan) {
  TORCH_CHECK(
      packed_query_symbols.is_cuda(),
      "packed_query_symbols must be CUDA");
  TORCH_CHECK(
      packed_key_symbols.is_cuda(),
      "packed_key_symbols must be CUDA");
  TORCH_CHECK(
      packed_query_symbols.is_contiguous(),
      "packed_query_symbols must be contiguous");
  TORCH_CHECK(
      packed_key_symbols.is_contiguous(),
      "packed_key_symbols must be contiguous");
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32,
      "packed_query_symbols must be int32");
  TORCH_CHECK(
      packed_key_symbols.scalar_type() == torch::kInt32,
      "packed_key_symbols must be int32");
  TORCH_CHECK(
      packed_query_symbols.dim() == 3,
      "packed symbols must have shape [B,H,T]");
  TORCH_CHECK(
      packed_key_symbols.sizes() == packed_query_symbols.sizes(),
      "packed query and key symbols must have identical shapes");
  TORCH_CHECK(
      packed_query_symbols.size(0) > 0 &&
          packed_query_symbols.size(1) > 0 &&
          packed_query_symbols.size(2) > 0,
      "packed symbols must be nonempty");
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(max_suffix_length >= 1);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(plan == 0 || plan == 1, "plan must be 0 or 1");
  const c10::cuda::CUDAGuard device_guard(packed_query_symbols.device());
  return rosa_soft_wavefront_scores_cuda(
      packed_query_symbols,
      packed_key_symbols,
      symbol_dim,
      std::min<int64_t>(
          max_suffix_length,
          packed_query_symbols.size(2)),
      static_cast<float>(mismatch_scale),
      static_cast<int>(plan));
}


torch::Tensor wavefront_stats(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t utility_plan) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous());
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous());
  TORCH_CHECK(value.dim() == 4, "value must have shape [B,T,Hv,Dv]");
  TORCH_CHECK(
      grad_output.dim() == 4,
      "grad_output must have shape [B,T,H,Dv]");
  TORCH_CHECK(
      value.scalar_type() == grad_output.scalar_type(),
      "value and grad_output must have the same dtype");
  TORCH_CHECK(
      value.scalar_type() == torch::kFloat32 ||
          value.scalar_type() == torch::kFloat16 ||
          value.scalar_type() == torch::kBFloat16,
      "value must be float32, float16, or bfloat16");
  TORCH_CHECK(
      packed_query_symbols.is_cuda() &&
          packed_query_symbols.is_contiguous());
  TORCH_CHECK(
      packed_key_symbols.is_cuda() && packed_key_symbols.is_contiguous());
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32);
  TORCH_CHECK(
      packed_query_symbols.dim() == 3 &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes());
  const int64_t batch_size = packed_query_symbols.size(0);
  const int64_t num_heads = packed_query_symbols.size(1);
  const int64_t seq_len = packed_query_symbols.size(2);
  TORCH_CHECK(
      value.size(0) == batch_size && value.size(1) == seq_len);
  TORCH_CHECK(value.size(2) > 0 && num_heads % value.size(2) == 0);
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {batch_size, seq_len, num_heads, value.size(3)}));
  TORCH_CHECK(dropout_seed.is_cuda() && dropout_seed.is_contiguous());
  TORCH_CHECK(dropout_seed.scalar_type() == torch::kInt64);
  TORCH_CHECK(dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0));
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(max_suffix_length >= 1);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0);
  TORCH_CHECK(std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(
      utility_plan == 0 || utility_plan == 1,
      "utility_plan must be 0 (tiled) or 1 (scalar)");
  const c10::cuda::CUDAGuard device_guard(value.device());
  return rosa_soft_wavefront_stats_cuda(
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      symbol_dim,
      std::min<int64_t>(max_suffix_length, seq_len),
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      utility_plan == 0 ? 1 : 2);
}


torch::Tensor persistent_wavefront_stats(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t utility_plan) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous());
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous());
  TORCH_CHECK(value.dim() == 4 && grad_output.dim() == 4);
  TORCH_CHECK(value.scalar_type() == grad_output.scalar_type());
  TORCH_CHECK(
      value.scalar_type() == torch::kFloat32 ||
          value.scalar_type() == torch::kFloat16 ||
          value.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(
      packed_query_symbols.is_cuda() &&
          packed_query_symbols.is_contiguous());
  TORCH_CHECK(
      packed_key_symbols.is_cuda() && packed_key_symbols.is_contiguous());
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32);
  TORCH_CHECK(
      packed_query_symbols.dim() == 3 &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes());
  const int64_t batch_size = packed_query_symbols.size(0);
  const int64_t num_heads = packed_query_symbols.size(1);
  const int64_t seq_len = packed_query_symbols.size(2);
  TORCH_CHECK(
      value.size(0) == batch_size && value.size(1) == seq_len);
  TORCH_CHECK(value.size(2) > 0 && num_heads % value.size(2) == 0);
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {batch_size, seq_len, num_heads, value.size(3)}));
  TORCH_CHECK(dropout_seed.is_cuda() && dropout_seed.is_contiguous());
  TORCH_CHECK(dropout_seed.scalar_type() == torch::kInt64);
  TORCH_CHECK(dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0));
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(max_suffix_length >= 1);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0);
  TORCH_CHECK(std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(
      utility_plan == 0 || utility_plan == 1,
      "utility_plan must be 0 (tiled) or 1 (scalar)");
  const c10::cuda::CUDAGuard device_guard(value.device());
  return rosa_soft_persistent_wavefront_stats_cuda(
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      symbol_dim,
      std::min<int64_t>(max_suffix_length, seq_len),
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      utility_plan == 0 ? 1 : 2);
}


torch::Tensor wavefront_log_gate_vjp(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& raw_scores,
    const torch::Tensor& raw_score_vjp,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    double mismatch_scale) {
  TORCH_CHECK(
      packed_query_symbols.is_cuda() &&
          packed_query_symbols.is_contiguous());
  TORCH_CHECK(
      packed_key_symbols.is_cuda() && packed_key_symbols.is_contiguous());
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32);
  TORCH_CHECK(
      packed_query_symbols.dim() == 3 &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes());
  const int64_t batch_size = packed_query_symbols.size(0);
  const int64_t num_heads = packed_query_symbols.size(1);
  const int64_t seq_len = packed_query_symbols.size(2);
  TORCH_CHECK(raw_scores.is_cuda() && raw_scores.is_contiguous());
  TORCH_CHECK(raw_score_vjp.is_cuda() && raw_score_vjp.is_contiguous());
  TORCH_CHECK(raw_scores.scalar_type() == torch::kFloat32);
  TORCH_CHECK(raw_score_vjp.scalar_type() == torch::kFloat32);
  TORCH_CHECK(
      raw_scores.sizes() == torch::IntArrayRef(
          {batch_size, num_heads, seq_len, seq_len}));
  TORCH_CHECK(raw_score_vjp.sizes() == raw_scores.sizes());
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(max_suffix_length >= 1);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  const c10::cuda::CUDAGuard device_guard(raw_scores.device());
  return rosa_soft_wavefront_log_gate_vjp_cuda(
      packed_query_symbols,
      packed_key_symbols,
      raw_scores,
      raw_score_vjp,
      symbol_dim,
      std::min<int64_t>(max_suffix_length, seq_len),
      static_cast<float>(mismatch_scale));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> wavefront_vjp(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t max_suffix_length,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t gradient_mask,
    int64_t execution_plan) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&query, "query"},
           {&key, "key"},
           {&value, "value"},
           {&grad_output, "grad_output"},
           {&packed_query_symbols, "packed_query_symbols"},
           {&packed_key_symbols, "packed_key_symbols"},
           {&dropout_seed, "dropout_seed"}}) {
    TORCH_CHECK(item.first->is_cuda(), item.second, " must be CUDA");
    TORCH_CHECK(item.first->is_contiguous(), item.second, " must be contiguous");
    TORCH_CHECK(item.first->device() == query.device());
  }
  TORCH_CHECK(query.dim() == 4 && key.sizes() == query.sizes());
  TORCH_CHECK(value.dim() == 4 && grad_output.dim() == 4);
  TORCH_CHECK(query.size(3) >= 1 && query.size(3) <= 32);
  TORCH_CHECK(value.size(0) == query.size(0));
  TORCH_CHECK(value.size(1) == query.size(1));
  TORCH_CHECK(value.size(2) > 0 && query.size(2) % value.size(2) == 0);
  TORCH_CHECK(value.size(3) > 0);
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(1), query.size(2), value.size(3)}));
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type() &&
          query.scalar_type() == grad_output.scalar_type());
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat32 ||
          query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32);
  TORCH_CHECK(
      packed_query_symbols.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(2), query.size(1)}));
  TORCH_CHECK(packed_key_symbols.sizes() == packed_query_symbols.sizes());
  TORCH_CHECK(dropout_seed.scalar_type() == torch::kInt64);
  TORCH_CHECK(dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0));
  TORCH_CHECK(max_suffix_length >= 1);
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0);
  TORCH_CHECK(std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(gradient_mask >= 1 && gradient_mask <= 7);
  TORCH_CHECK(
      execution_plan == 0 || execution_plan == 1,
      "execution_plan must be 0 (multi-launch) or 1 (persistent)");
  const c10::cuda::CUDAGuard device_guard(query.device());
  return rosa_soft_wavefront_vjp_cuda(
      query,
      key,
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      std::min<int64_t>(max_suffix_length, query.size(1)),
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      static_cast<int>(gradient_mask),
      static_cast<int>(execution_plan));
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "unbounded_group_scores",
      &unbounded_group_scores,
      "Unbounded suffix scores for one replay diagonal group (CUDA)");
  module.def(
      "unbounded_replay_stats",
      &unbounded_replay_stats,
      "Linear-workspace unbounded replay softmax statistics (CUDA)");
  module.def(
      "macro_checkpoint_stats",
      &macro_checkpoint_stats,
      "Macro-diagonal checkpoint softmax statistics (CUDA)");
  module.def(
      "unbounded_replay_vjp",
      &unbounded_replay_vjp,
      "Linear-workspace unbounded diagonal-replay VJP (CUDA)");
  module.def(
      "grouped_checkpoint_reverse",
      &grouped_checkpoint_reverse,
      "Grouped checkpoint/replay reverse VJP (CUDA)");
  module.def(
      "grouped_checkpoint_reverse_tensor_symbols",
      &grouped_checkpoint_reverse_tensor_symbols,
      "Grouped reverse with Tensor Core Q/K contractions (CUDA)");
  module.def(
      "wavefront_scores",
      &wavefront_scores,
      "Debug block-wavefront finite suffix scores (CUDA)");
  module.def(
      "wavefront_stats",
      &wavefront_stats,
      "Block-wavefront online softmax statistics (CUDA)");
  module.def(
      "persistent_wavefront_stats",
      &persistent_wavefront_stats,
      "Cooperative persistent block-wavefront statistics (CUDA)");
  module.def(
      "wavefront_log_gate_vjp",
      &wavefront_log_gate_vjp,
      "Debug reverse-wavefront log-gate VJP (CUDA)");
  module.def(
      "wavefront_vjp",
      &wavefront_vjp,
      "Atomics-free multi-launch block-wavefront VJP (CUDA)");
}
