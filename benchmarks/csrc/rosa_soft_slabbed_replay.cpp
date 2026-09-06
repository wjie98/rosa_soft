#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_slabbed_replay_vjp_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    float scale,
    float dropout_p,
    float mismatch_scale,
    int gradient_mask,
    int execution_plan,
    int value_tile_size,
    int diagonal_tile_size,
    int slab_size);


namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
slabbed_replay_vjp(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& dropout_seed,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t gradient_mask,
    int64_t execution_plan,
    int64_t value_tile_size,
    int64_t diagonal_tile_size,
    int64_t slab_size) {
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
  TORCH_CHECK(
      query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0,
      "query must have nonempty B, T, and H dimensions");
  TORCH_CHECK(
      query.size(3) >= 1 && query.size(3) <= 32,
      "symbol width must be in [1, 32]");
  TORCH_CHECK(value.dim() == 4, "value must have shape [B,T,Hv,Dv]");
  TORCH_CHECK(
      value.size(0) == query.size(0) && value.size(1) == query.size(1) &&
          value.size(2) > 0 && query.size(2) % value.size(2) == 0 &&
          value.size(3) > 0,
      "value shape is incompatible with query");
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
      "floating tensors must be float32, float16, or bfloat16");
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
      "dropout_seed must be one int64 scalar iff dropout is enabled");
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and positive");
  TORCH_CHECK(
      std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p < 1.0,
      "dropout_p must be finite and in [0, 1)");
  TORCH_CHECK(
      std::isfinite(mismatch_scale) && mismatch_scale > 0.0,
      "mismatch_scale must be finite and positive");
  TORCH_CHECK(
      gradient_mask >= 1 && gradient_mask <= 7,
      "gradient_mask must be in [1, 7]");
  TORCH_CHECK(
      execution_plan >= 0 && execution_plan <= 15,
      "execution_plan must be in [0, 15]");
  TORCH_CHECK(
      value_tile_size == 16 || value_tile_size == 32,
      "value_tile_size must be 16 or 32");
  TORCH_CHECK(
      diagonal_tile_size == 16 || diagonal_tile_size == 32,
      "diagonal_tile_size must be 16 or 32");
  TORCH_CHECK(
      slab_size >= 32 && slab_size % 32 == 0 &&
          slab_size <= std::numeric_limits<int>::max(),
      "slab_size must be a positive int32 multiple of 32");
  TORCH_CHECK(
      query.numel() <= std::numeric_limits<int64_t>::max() / 2,
      "input is too large for slabbed replay indexing");

  const c10::cuda::CUDAGuard device_guard(query.device());
  return rosa_soft_slabbed_replay_vjp_cuda(
      query,
      key,
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      dropout_seed,
      static_cast<float>(scale),
      static_cast<float>(dropout_p),
      static_cast<float>(mismatch_scale),
      static_cast<int>(gradient_mask),
      static_cast<int>(execution_plan),
      static_cast<int>(value_tile_size),
      static_cast<int>(diagonal_tile_size),
      static_cast<int>(slab_size));
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "slabbed_replay_vjp",
      &slabbed_replay_vjp,
      "Exact unlimited RosaSoft VJP with fixed linear-workspace slabs (CUDA)");
}
