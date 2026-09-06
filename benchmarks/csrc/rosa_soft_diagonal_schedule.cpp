#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


torch::Tensor rosa_diagonal_schedule_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int64_t symbol_dim,
    double mismatch_scale,
    int64_t method,
    int64_t worker_blocks);


torch::Tensor diagonal_schedule(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    int64_t symbol_dim,
    double mismatch_scale,
    int64_t method,
    int64_t worker_blocks) {
  TORCH_CHECK(
      packed_query.is_cuda() && packed_key.is_cuda(),
      "packed symbols must be CUDA tensors");
  TORCH_CHECK(
      packed_query.is_contiguous() && packed_key.is_contiguous(),
      "packed symbols must be contiguous");
  TORCH_CHECK(
      packed_query.scalar_type() == torch::kInt32 &&
          packed_key.scalar_type() == torch::kInt32,
      "packed symbols must be int32");
  TORCH_CHECK(
      packed_query.dim() == 3 &&
          packed_key.sizes() == packed_query.sizes(),
      "packed symbols must have matching [B,H,T] shapes");
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 32);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(method >= 0 && method <= 4);
  TORCH_CHECK(worker_blocks >= 0);
  const c10::cuda::CUDAGuard guard(packed_query.device());
  return rosa_diagonal_schedule_cuda(
      packed_query,
      packed_key,
      symbol_dim,
      mismatch_scale,
      method,
      worker_blocks);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("diagonal_schedule", &diagonal_schedule);
}
