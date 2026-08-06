#include <algorithm>

#include <torch/extension.h>

torch::Tensor sam_bitflip_descriptor_vjp_cuda(
    const torch::Tensor& base_routes,
    const torch::Tensor& query_offsets,
    const torch::Tensor& query_changes,
    const torch::Tensor& delete_offsets,
    const torch::Tensor& delete_changes,
    const torch::Tensor& override_offsets,
    const torch::Tensor& overrides,
    const torch::Tensor& query_codes,
    const torch::Tensor& key_codes,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    int64_t bit_width);

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor descriptor_vjp(
    const torch::Tensor& base_routes,
    const torch::Tensor& query_offsets,
    const torch::Tensor& query_changes,
    const torch::Tensor& delete_offsets,
    const torch::Tensor& delete_changes,
    const torch::Tensor& override_offsets,
    const torch::Tensor& overrides,
    const torch::Tensor& query_codes,
    const torch::Tensor& key_codes,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    int64_t bit_width) {
  check_cuda_contiguous(base_routes, "base_routes");
  check_cuda_contiguous(query_offsets, "query_offsets");
  check_cuda_contiguous(query_changes, "query_changes");
  check_cuda_contiguous(delete_offsets, "delete_offsets");
  check_cuda_contiguous(delete_changes, "delete_changes");
  check_cuda_contiguous(override_offsets, "override_offsets");
  check_cuda_contiguous(overrides, "overrides");
  check_cuda_contiguous(query_codes, "query_codes");
  check_cuda_contiguous(key_codes, "key_codes");
  check_cuda_contiguous(value, "value");
  check_cuda_contiguous(grad_output, "grad_output");

  TORCH_CHECK(base_routes.scalar_type() == torch::kInt32);
  TORCH_CHECK(query_offsets.scalar_type() == torch::kInt64);
  TORCH_CHECK(delete_offsets.scalar_type() == torch::kInt64);
  TORCH_CHECK(override_offsets.scalar_type() == torch::kInt64);
  TORCH_CHECK(query_changes.scalar_type() == torch::kInt32);
  TORCH_CHECK(delete_changes.scalar_type() == torch::kInt32);
  TORCH_CHECK(overrides.scalar_type() == torch::kInt32);
  TORCH_CHECK(query_codes.scalar_type() == torch::kUInt8);
  TORCH_CHECK(key_codes.scalar_type() == torch::kUInt8);
  TORCH_CHECK(value.is_floating_point());
  TORCH_CHECK(value.scalar_type() == grad_output.scalar_type());
  TORCH_CHECK(value.dim() == 2 && grad_output.sizes() == value.sizes());
  TORCH_CHECK(query_offsets.dim() == 1);
  TORCH_CHECK(delete_offsets.dim() == 1);
  TORCH_CHECK(override_offsets.dim() == 1);
  TORCH_CHECK(base_routes.dim() == 1 && base_routes.size(0) == value.size(0));
  TORCH_CHECK(query_codes.sizes() == base_routes.sizes());
  TORCH_CHECK(key_codes.sizes() == base_routes.sizes());
  TORCH_CHECK(query_changes.dim() == 2 && query_changes.size(1) == 6);
  TORCH_CHECK(delete_changes.dim() == 2 && delete_changes.size(1) == 6);
  TORCH_CHECK(overrides.dim() == 2 && overrides.size(1) == 10);
  TORCH_CHECK(bit_width >= 1 && bit_width <= 8);
  const auto device = value.device();
  for (const auto& tensor : {
           base_routes,
           query_offsets,
           query_changes,
           delete_offsets,
           delete_changes,
           override_offsets,
           overrides,
           query_codes,
           key_codes,
           grad_output}) {
    TORCH_CHECK(tensor.device() == device, "all inputs must share one device");
  }
  const int64_t key_length = delete_offsets.numel() - 1;
  const int64_t query_flip_count = query_offsets.numel() - 1;
  TORCH_CHECK(key_length >= 0 && query_flip_count >= 0);
  TORCH_CHECK(key_length == std::max<int64_t>(value.size(0) - 1, 0));
  TORCH_CHECK(query_flip_count == key_length * bit_width);
  TORCH_CHECK(override_offsets.numel() == query_offsets.numel());

  return sam_bitflip_descriptor_vjp_cuda(
      base_routes,
      query_offsets,
      query_changes,
      delete_offsets,
      delete_changes,
      override_offsets,
      overrides,
      query_codes,
      key_codes,
      value,
      grad_output,
      bit_width);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("descriptor_vjp", &descriptor_vjp);
}
