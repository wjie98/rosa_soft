#include <cmath>
#include <cstdint>

#include <torch/extension.h>


torch::Tensor rosa_gate_matrix_cuda(
    const torch::Tensor& query_codes,
    const torch::Tensor& key_codes,
    int symbol_bits,
    float mismatch_scale,
    int method);

torch::Tensor rosa_batched_matmul_cuda(
    const torch::Tensor& left,
    const torch::Tensor& right,
    int method);

torch::Tensor rosa_suffix_scores_cuda(
    const torch::Tensor& gates,
    int max_suffix_length,
    int method);


namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}


torch::Tensor gate_matrix(
    const torch::Tensor& query_codes,
    const torch::Tensor& key_codes,
    int64_t symbol_bits,
    double mismatch_scale,
    int64_t method) {
  check_cuda_contiguous(query_codes, "query_codes");
  check_cuda_contiguous(key_codes, "key_codes");
  TORCH_CHECK(query_codes.device() == key_codes.device());
  TORCH_CHECK(query_codes.scalar_type() == torch::kInt32);
  TORCH_CHECK(key_codes.scalar_type() == torch::kInt32);
  TORCH_CHECK(query_codes.dim() == 1 && key_codes.dim() == 1);
  TORCH_CHECK(query_codes.numel() > 0 && key_codes.numel() > 0);
  TORCH_CHECK(symbol_bits >= 1 && symbol_bits <= 32);
  TORCH_CHECK(std::isfinite(mismatch_scale) && mismatch_scale > 0.0);
  TORCH_CHECK(method >= 0 && method <= 3);
  return rosa_gate_matrix_cuda(
      query_codes,
      key_codes,
      static_cast<int>(symbol_bits),
      static_cast<float>(mismatch_scale),
      static_cast<int>(method));
}


torch::Tensor batched_matmul(
    const torch::Tensor& left,
    const torch::Tensor& right,
    int64_t method) {
  check_cuda_contiguous(left, "left");
  check_cuda_contiguous(right, "right");
  TORCH_CHECK(left.device() == right.device());
  TORCH_CHECK(left.scalar_type() == torch::kFloat32);
  TORCH_CHECK(right.scalar_type() == torch::kFloat32);
  TORCH_CHECK(left.dim() == 3 && right.dim() == 3);
  TORCH_CHECK(left.size(0) == right.size(0));
  TORCH_CHECK(left.size(2) == right.size(1));
  TORCH_CHECK(left.size(0) > 0 && left.size(1) > 0);
  TORCH_CHECK(left.size(2) > 0 && right.size(2) > 0);
  TORCH_CHECK(method >= 0 && method <= 5);
  return rosa_batched_matmul_cuda(left, right, static_cast<int>(method));
}


torch::Tensor suffix_scores(
    const torch::Tensor& gates,
    int64_t max_suffix_length,
    int64_t method) {
  check_cuda_contiguous(gates, "gates");
  TORCH_CHECK(gates.scalar_type() == torch::kFloat32);
  TORCH_CHECK(gates.dim() == 2);
  TORCH_CHECK(gates.size(0) > 0 && gates.size(1) > 0);
  TORCH_CHECK(max_suffix_length >= 1 && max_suffix_length <= 32);
  TORCH_CHECK(method == 0 || method == 1);
  return rosa_suffix_scores_cuda(
      gates,
      static_cast<int>(max_suffix_length),
      static_cast<int>(method));
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gate_matrix", &gate_matrix, "RosaSoft gate tile methods");
  module.def(
      "batched_matmul",
      &batched_matmul,
      "RosaSoft Tensor-Core contraction methods");
  module.def(
      "suffix_scores",
      &suffix_scores,
      "RosaSoft diagonal suffix scan methods");
}
