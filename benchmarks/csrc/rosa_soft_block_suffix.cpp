#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


torch::Tensor rosa_block_suffix_scores_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int method);

torch::Tensor rosa_block_suffix_tail_scores_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int route_start,
    int active_queries,
    int method);

torch::Tensor rosa_block_suffix_hybrid_scores_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int symbol_dim,
    int max_suffix_length,
    float mismatch_scale,
    int tile_start,
    int tail_queries,
    int method);


namespace {

torch::Tensor block_suffix_scores(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    double mismatch_scale,
    int64_t method) {
  TORCH_CHECK(packed_query.is_cuda(), "packed_query must be CUDA");
  TORCH_CHECK(packed_key.is_cuda(), "packed_key must be CUDA");
  TORCH_CHECK(packed_query.is_contiguous(), "packed_query must be contiguous");
  TORCH_CHECK(packed_key.is_contiguous(), "packed_key must be contiguous");
  TORCH_CHECK(packed_query.device() == packed_key.device());
  TORCH_CHECK(packed_query.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_key.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_query.dim() == 3, "packed inputs must have shape [B,H,T]");
  TORCH_CHECK(packed_key.sizes() == packed_query.sizes());
  TORCH_CHECK(packed_query.size(0) > 0);
  TORCH_CHECK(packed_query.size(1) > 0);
  TORCH_CHECK(packed_query.size(2) > 0);
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(max_suffix_length >= 1);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(method >= 0 && method <= 4);
  const c10::cuda::CUDAGuard device_guard(packed_query.device());
  return rosa_block_suffix_scores_cuda(
      packed_query,
      packed_key,
      static_cast<int>(symbol_dim),
      static_cast<int>(std::min<int64_t>(
          max_suffix_length,
          packed_query.size(2))),
      static_cast<float>(mismatch_scale),
      static_cast<int>(method));
}


torch::Tensor block_suffix_tail_scores(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    double mismatch_scale,
    int64_t route_start,
    int64_t active_queries,
    int64_t method) {
  TORCH_CHECK(packed_query.is_cuda(), "packed_query must be CUDA");
  TORCH_CHECK(packed_key.is_cuda(), "packed_key must be CUDA");
  TORCH_CHECK(packed_query.is_contiguous(), "packed_query must be contiguous");
  TORCH_CHECK(packed_key.is_contiguous(), "packed_key must be contiguous");
  TORCH_CHECK(packed_query.device() == packed_key.device());
  TORCH_CHECK(packed_query.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_key.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_query.dim() == 3, "packed inputs must have shape [B,H,T]");
  TORCH_CHECK(packed_key.sizes() == packed_query.sizes());
  TORCH_CHECK(packed_query.size(0) > 0 && packed_query.size(1) > 0);
  TORCH_CHECK(packed_query.size(2) >= 128);
  TORCH_CHECK(packed_query.size(2) <= std::numeric_limits<int>::max());
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(max_suffix_length >= 1);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(active_queries >= 1 && active_queries <= 32);
  TORCH_CHECK(route_start >= std::max<int64_t>(
      max_suffix_length + 1,
      64 - active_queries));
  TORCH_CHECK(route_start + 64 <= packed_query.size(2));
  TORCH_CHECK(method >= 0 && method <= 4);
  TORCH_CHECK(method != 4 || max_suffix_length == 32);
  const c10::cuda::CUDAGuard device_guard(packed_query.device());
  return rosa_block_suffix_tail_scores_cuda(
      packed_query,
      packed_key,
      static_cast<int>(symbol_dim),
      static_cast<int>(max_suffix_length),
      static_cast<float>(mismatch_scale),
      static_cast<int>(route_start),
      static_cast<int>(active_queries),
      static_cast<int>(method));
}


torch::Tensor block_suffix_hybrid_scores(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int64_t symbol_dim,
    int64_t max_suffix_length,
    double mismatch_scale,
    int64_t tile_start,
    int64_t tail_queries,
    int64_t method) {
  TORCH_CHECK(packed_query.is_cuda(), "packed_query must be CUDA");
  TORCH_CHECK(packed_key.is_cuda(), "packed_key must be CUDA");
  TORCH_CHECK(packed_query.is_contiguous(), "packed_query must be contiguous");
  TORCH_CHECK(packed_key.is_contiguous(), "packed_key must be contiguous");
  TORCH_CHECK(packed_query.device() == packed_key.device());
  TORCH_CHECK(packed_query.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_key.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_query.dim() == 3, "packed inputs must have shape [B,H,T]");
  TORCH_CHECK(packed_key.sizes() == packed_query.sizes());
  TORCH_CHECK(packed_query.size(0) > 0 && packed_query.size(1) > 0);
  TORCH_CHECK(packed_query.size(2) <= std::numeric_limits<int>::max());
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(max_suffix_length == 32);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(tail_queries >= 1 && tail_queries <= 32);
  TORCH_CHECK(tile_start >= max_suffix_length + 1);
  TORCH_CHECK(tile_start + 64 <= packed_query.size(2));
  TORCH_CHECK(method >= 0 && method <= 3);
  const c10::cuda::CUDAGuard device_guard(packed_query.device());
  return rosa_block_suffix_hybrid_scores_cuda(
      packed_query,
      packed_key,
      static_cast<int>(symbol_dim),
      static_cast<int>(max_suffix_length),
      static_cast<float>(mismatch_scale),
      static_cast<int>(tile_start),
      static_cast<int>(tail_queries),
      static_cast<int>(method));
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "block_suffix_scores",
      &block_suffix_scores,
      "Exact RosaSoft suffix-score execution methods (CUDA)");
  module.def(
      "block_suffix_tail_scores",
      &block_suffix_tail_scores,
      "Exact RosaSoft causal-tail suffix-score methods (CUDA)");
  module.def(
      "block_suffix_hybrid_scores",
      &block_suffix_hybrid_scores,
      "Exact RosaSoft full-tile hybrid suffix-score methods (CUDA)");
}
