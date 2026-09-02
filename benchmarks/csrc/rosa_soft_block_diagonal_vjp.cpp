#include <cmath>
#include <cstdint>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_soft_block_diagonal_vjp_cuda(
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
    int plan);


namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> block_diagonal_vjp(
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
    int64_t plan) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&query, "query"},
           {&key, "key"},
           {&value, "value"},
           {&grad_output, "grad_output"},
           {&packed_query_symbols, "packed_query_symbols"},
           {&packed_key_symbols, "packed_key_symbols"},
           {&dropout_seed, "dropout_seed"}}) {
    check_cuda_contiguous(*item.first, item.second);
    TORCH_CHECK(item.first->device() == query.device());
  }
  TORCH_CHECK(query.dim() == 4, "query must have shape [B,T,H,D]");
  TORCH_CHECK(key.sizes() == query.sizes());
  TORCH_CHECK(value.dim() == 4, "value must have shape [B,T,Hv,Dv]");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0);
  TORCH_CHECK(query.size(2) > 0 && query.size(3) > 0);
  TORCH_CHECK(query.size(3) <= 32);
  TORCH_CHECK(value.size(0) == query.size(0));
  TORCH_CHECK(value.size(1) == query.size(1));
  TORCH_CHECK(value.size(2) > 0 && query.size(2) % value.size(2) == 0);
  TORCH_CHECK(value.size(3) > 0);
  TORCH_CHECK(query.scalar_type() == key.scalar_type());
  TORCH_CHECK(query.scalar_type() == value.scalar_type());
  TORCH_CHECK(query.scalar_type() == grad_output.scalar_type());
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat32 ||
          query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(1), query.size(2), value.size(3)}));
  TORCH_CHECK(packed_query_symbols.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_key_symbols.scalar_type() == torch::kInt32);
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
  TORCH_CHECK(plan >= 0 && plan <= 3);
  const c10::cuda::CUDAGuard device_guard(query.device());
  return rosa_soft_block_diagonal_vjp_cuda(
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
      static_cast<int>(plan));
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "block_diagonal_vjp",
      &block_diagonal_vjp,
      "Block-diagonal exact RosaSoft VJP study (CUDA)");
}
