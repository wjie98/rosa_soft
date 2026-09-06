#include <cmath>
#include <cstdint>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_macro_wavefront_scores_cuda(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t tile_size,
    float mismatch_scale,
    int execution_plan);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_macro_wavefront_stats_cuda(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t tile_size,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int execution_plan);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_macro_wavefront_vjp_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t tile_size,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask,
    int execution_plan);


namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
macro_wavefront_scores(
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t symbol_dim,
    int64_t tile_size,
    double mismatch_scale,
    int64_t execution_plan) {
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
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32,
      "packed symbols must be int32");
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
  TORCH_CHECK(
      packed_query_symbols.device() == packed_key_symbols.device(),
      "packed query and key symbols must be on the same device");
  TORCH_CHECK(
      symbol_dim >= 1 && symbol_dim <= 32,
      "symbol_dim must be in [1, 32]");
  TORCH_CHECK(
      tile_size == 0 || tile_size == 32 || tile_size == 64 ||
          tile_size == 96 || tile_size == 128,
      "tile_size must be 0, 32, 64, 96, or 128");
  TORCH_CHECK(
      std::isfinite(mismatch_scale) && mismatch_scale > 0.0,
      "mismatch_scale must be finite and positive");
  TORCH_CHECK(
      execution_plan >= 0 && execution_plan <= 4,
      "execution_plan must be 0 (multi-launch), 1 (ready-queue persistent), "
      "2 (barrier persistent), 3 (folded diagonals), or 4 "
      "(persistent row streams)");

  const c10::cuda::CUDAGuard device_guard(
      packed_query_symbols.device());
  return rosa_soft_macro_wavefront_scores_cuda(
      packed_query_symbols,
      packed_key_symbols,
      symbol_dim,
      tile_size,
      static_cast<float>(mismatch_scale),
      static_cast<int>(execution_plan));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
macro_wavefront_stats(
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t symbol_dim,
    int64_t tile_size,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t execution_plan) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&value, "value"},
           {&grad_output, "grad_output"},
           {&packed_query_symbols, "packed_query_symbols"},
           {&packed_key_symbols, "packed_key_symbols"},
           {&dropout_seed, "dropout_seed"}}) {
    TORCH_CHECK(item.first->is_cuda(), item.second, " must be CUDA");
    TORCH_CHECK(
        item.first->is_contiguous(), item.second, " must be contiguous");
    TORCH_CHECK(
        item.first->device() == value.device(),
        item.second,
        " must be on the value device");
  }
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
      "value and grad_output must be float32, float16, or bfloat16");
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32,
      "packed symbols must be int32");
  TORCH_CHECK(
      packed_query_symbols.dim() == 3 &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes(),
      "packed symbols must have matching [B,H,T] shapes");
  const int64_t batch_size = packed_query_symbols.size(0);
  const int64_t num_heads = packed_query_symbols.size(1);
  const int64_t seq_len = packed_query_symbols.size(2);
  TORCH_CHECK(
      batch_size > 0 && num_heads > 0 && seq_len > 0,
      "packed symbols must be nonempty");
  TORCH_CHECK(
      value.size(0) == batch_size && value.size(1) == seq_len,
      "value batch and sequence dimensions must match packed symbols");
  TORCH_CHECK(
      value.size(2) > 0 && num_heads % value.size(2) == 0,
      "query heads must be divisible by value heads");
  TORCH_CHECK(value.size(3) > 0, "value dimension must be positive");
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {batch_size, seq_len, num_heads, value.size(3)}),
      "grad_output must have shape [B,T,H,Dv]");
  TORCH_CHECK(
      dropout_seed.scalar_type() == torch::kInt64,
      "dropout_seed must be int64");
  TORCH_CHECK(
      dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0),
      "dropout_seed must contain one value exactly when dropout is enabled");
  TORCH_CHECK(
      symbol_dim >= 1 && symbol_dim <= 32,
      "symbol_dim must be in [1, 32]");
  TORCH_CHECK(
      tile_size == 0 || tile_size == 32 || tile_size == 64 ||
          tile_size == 96 || tile_size == 128,
      "tile_size must be 0, 32, 64, 96, or 128");
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and positive");
  TORCH_CHECK(
      std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0,
      "dropout_p must be in [0, 1)");
  TORCH_CHECK(
      std::isfinite(mismatch_scale) && mismatch_scale > 0.0,
      "mismatch_scale must be finite and positive");
  TORCH_CHECK(
      execution_plan >= 0 && execution_plan <= 4,
      "execution_plan must be 0 (multi-launch), 1 (ready-queue persistent), "
      "2 (barrier persistent), 3 (folded diagonals), or 4 "
      "(persistent row streams)");

  const c10::cuda::CUDAGuard device_guard(value.device());
  return rosa_soft_macro_wavefront_stats_cuda(
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      symbol_dim,
      tile_size,
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      static_cast<int>(execution_plan));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
macro_wavefront_vjp(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    int64_t tile_size,
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
    TORCH_CHECK(
        item.first->is_contiguous(), item.second, " must be contiguous");
    TORCH_CHECK(
        item.first->device() == query.device(),
        item.second,
        " must be on the query device");
  }
  TORCH_CHECK(
      query.dim() == 4 && key.sizes() == query.sizes(),
      "query and key must have matching [B,T,H,D] shapes");
  TORCH_CHECK(value.dim() == 4, "value must have shape [B,T,Hv,Dv]");
  TORCH_CHECK(
      query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0,
      "query must be nonempty");
  TORCH_CHECK(
      query.size(3) >= 1 && query.size(3) <= 32,
      "query symbol dimension must be in [1, 32]");
  TORCH_CHECK(
      value.size(0) == query.size(0) && value.size(1) == query.size(1),
      "value batch and sequence dimensions must match query");
  TORCH_CHECK(
      value.size(2) > 0 && query.size(2) % value.size(2) == 0,
      "query heads must be divisible by value heads");
  TORCH_CHECK(value.size(3) > 0, "value dimension must be positive");
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(1), query.size(2), value.size(3)}),
      "grad_output must have shape [B,T,H,Dv]");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type() &&
          query.scalar_type() == grad_output.scalar_type(),
      "query, key, value, and grad_output must have the same dtype");
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat32 ||
          query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16,
      "inputs must be float32, float16, or bfloat16");
  TORCH_CHECK(
      packed_query_symbols.scalar_type() == torch::kInt32 &&
          packed_key_symbols.scalar_type() == torch::kInt32,
      "packed symbols must be int32");
  TORCH_CHECK(
      packed_query_symbols.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(2), query.size(1)}) &&
          packed_key_symbols.sizes() == packed_query_symbols.sizes(),
      "packed symbols must have matching [B,H,T] shapes");
  TORCH_CHECK(
      dropout_seed.scalar_type() == torch::kInt64 &&
          dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0),
      "dropout_seed must be int64 and contain one value only when enabled");
  TORCH_CHECK(
      tile_size == 0 || tile_size == 32 || tile_size == 64 ||
          tile_size == 96 || tile_size == 128,
      "tile_size must be 0, 32, 64, 96, or 128");
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and positive");
  TORCH_CHECK(
      std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0,
      "dropout_p must be in [0, 1)");
  TORCH_CHECK(
      std::isfinite(mismatch_scale) && mismatch_scale > 0.0,
      "mismatch_scale must be finite and positive");
  TORCH_CHECK(
      gradient_mask >= 1 && gradient_mask <= 7,
      "gradient_mask must be in [1, 7]");
  TORCH_CHECK(
      execution_plan >= 0 && execution_plan <= 4,
      "execution_plan must be 0 (multi-launch), 1 (ready-queue persistent), "
      "2 (barrier persistent), 3 (folded diagonals), or 4 "
      "(persistent row streams)");

  const c10::cuda::CUDAGuard device_guard(query.device());
  return rosa_soft_macro_wavefront_vjp_cuda(
      query,
      key,
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      tile_size,
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      static_cast<int>(gradient_mask),
      static_cast<int>(execution_plan));
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "macro_wavefront_scores",
      &macro_wavefront_scores,
      "Exact unlimited macro-tile wavefront scores and checkpoints (CUDA)");
  module.def(
      "macro_wavefront_stats",
      &macro_wavefront_stats,
      "Exact unlimited macro-tile online statistics and checkpoints (CUDA)");
  module.def(
      "macro_wavefront_vjp",
      &macro_wavefront_vjp,
      "Exact unlimited macro-tile replay VJP (CUDA)");
}
