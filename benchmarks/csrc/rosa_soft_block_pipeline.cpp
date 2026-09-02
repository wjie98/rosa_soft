#include <cmath>
#include <cstdint>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


torch::Tensor rosa_block_pipeline_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& grad_output,
    const torch::Tensor& value,
    int symbol_dim,
    float mismatch_scale,
    int method);


namespace {

torch::Tensor block_pipeline(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& grad_output,
    const torch::Tensor& value,
    int64_t symbol_dim,
    double mismatch_scale,
    int64_t method) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &packed_query, "packed_query"},
           {&packed_key, "packed_key"},
           {&grad_output, "grad_output"},
           {&value, "value"}}) {
    TORCH_CHECK(item.first->is_cuda(), item.second, " must be CUDA");
    TORCH_CHECK(item.first->is_contiguous(), item.second, " must be contiguous");
    TORCH_CHECK(item.first->device() == packed_query.device());
  }
  TORCH_CHECK(packed_query.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_key.scalar_type() == torch::kInt32);
  TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32);
  TORCH_CHECK(value.scalar_type() == torch::kFloat32);
  TORCH_CHECK(packed_query.dim() == 2, "packed inputs must have shape [S,T]");
  TORCH_CHECK(packed_key.sizes() == packed_query.sizes());
  TORCH_CHECK(packed_query.size(0) > 0);
  TORCH_CHECK(packed_query.size(1) >= 128);
  TORCH_CHECK(packed_query.size(1) % 64 == 0);
  TORCH_CHECK(
      grad_output.sizes() == torch::IntArrayRef(
          {packed_query.size(0), 64, 64}));
  TORCH_CHECK(
      value.sizes() == torch::IntArrayRef(
          {packed_query.size(0), packed_query.size(1), 64}));
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(method >= 0 && method <= 1);
  const c10::cuda::CUDAGuard device_guard(packed_query.device());
  return rosa_block_pipeline_cuda(
      packed_query,
      packed_key,
      grad_output,
      value,
      static_cast<int>(symbol_dim),
      static_cast<float>(mismatch_scale),
      static_cast<int>(method));
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "block_pipeline",
      &block_pipeline,
      "Warp-specialized block score/utility pipeline (CUDA)");
}
