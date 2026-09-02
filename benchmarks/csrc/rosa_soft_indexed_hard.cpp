#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor> rosa_pack_sign_bits_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key);

std::tuple<torch::Tensor, torch::Tensor> rosa_build_occurrence_index_cuda(
    const torch::Tensor& packed_key,
    int symbol_dim);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_indexed_hard_forward_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& value,
    const torch::Tensor& offsets,
    const torch::Tensor& occurrences,
    int max_suffix_length,
    int method);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rosa_diagonal_hard_forward_cuda(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& value,
    int max_suffix_length);


namespace {

void check_packed_pair(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key) {
  TORCH_CHECK(packed_query.is_cuda(), "packed_query must be CUDA");
  TORCH_CHECK(packed_key.is_cuda(), "packed_key must be CUDA");
  TORCH_CHECK(
      packed_query.device() == packed_key.device(),
      "packed_query and packed_key must use the same device");
  TORCH_CHECK(
      packed_query.scalar_type() == torch::kInt32,
      "packed_query must have dtype torch.int32");
  TORCH_CHECK(
      packed_key.scalar_type() == torch::kInt32,
      "packed_key must have dtype torch.int32");
  TORCH_CHECK(
      packed_query.dim() == 3,
      "packed inputs must have shape [batch, heads, tokens]");
  TORCH_CHECK(
      packed_query.sizes() == packed_key.sizes(),
      "packed query/key shapes must match");
  TORCH_CHECK(
      packed_query.is_contiguous() && packed_key.is_contiguous(),
      "packed query/key must be contiguous");
  TORCH_CHECK(
      packed_query.size(0) > 0 &&
          packed_query.size(1) > 0 &&
          packed_query.size(2) > 0,
      "packed dimensions must be nonempty");
  TORCH_CHECK(
      packed_query.size(2) <= std::numeric_limits<int>::max(),
      "sequence length exceeds the CUDA int32 indexing limit");
}


std::tuple<torch::Tensor, torch::Tensor> pack_sign_bits(
    const torch::Tensor& query,
    const torch::Tensor& key) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(query.device() == key.device(), "query/key devices must match");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtypes must match");
  TORCH_CHECK(query.is_floating_point(), "query/key must be floating-point");
  TORCH_CHECK(query.dim() == 4, "query/key must have shape [B,T,H,D]");
  TORCH_CHECK(query.sizes() == key.sizes(), "query/key shapes must match");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous(), "query/key must be contiguous");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0);
  TORCH_CHECK(query.size(3) >= 1 && query.size(3) <= 8, "symbol width must be in [1, 8]");
  TORCH_CHECK(query.size(1) <= std::numeric_limits<int>::max());
  const c10::cuda::CUDAGuard device_guard(query.device());
  return rosa_pack_sign_bits_cuda(query, key);
}


std::tuple<torch::Tensor, torch::Tensor> build_occurrence_index(
    const torch::Tensor& packed_key,
    int64_t symbol_dim) {
  TORCH_CHECK(packed_key.is_cuda(), "packed_key must be CUDA");
  TORCH_CHECK(packed_key.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_key.dim() == 3, "packed_key must have shape [B,H,T]");
  TORCH_CHECK(packed_key.is_contiguous(), "packed_key must be contiguous");
  TORCH_CHECK(packed_key.size(0) > 0 && packed_key.size(1) > 0 && packed_key.size(2) > 0);
  TORCH_CHECK(packed_key.size(2) <= std::numeric_limits<int>::max());
  TORCH_CHECK(symbol_dim >= 1 && symbol_dim <= 8, "symbol_dim must be in [1, 8]");
  const c10::cuda::CUDAGuard device_guard(packed_key.device());
  return rosa_build_occurrence_index_cuda(
      packed_key,
      static_cast<int>(symbol_dim));
}


void check_value(
    const torch::Tensor& value,
    const torch::Tensor& packed_query) {
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(value.device() == packed_query.device(), "value device must match packed inputs");
  TORCH_CHECK(value.is_floating_point(), "value must be floating-point");
  TORCH_CHECK(value.dim() == 4, "value must have shape [B,T,Hv,Dv]");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  TORCH_CHECK(value.size(0) == packed_query.size(0));
  TORCH_CHECK(value.size(1) == packed_query.size(2));
  TORCH_CHECK(value.size(2) > 0 && value.size(3) > 0);
  TORCH_CHECK(
      packed_query.size(1) % value.size(2) == 0,
      "query heads must be divisible by value heads");
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
indexed_hard_forward(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& value,
    const torch::Tensor& offsets,
    const torch::Tensor& occurrences,
    int64_t max_suffix_length,
    int64_t method) {
  check_packed_pair(packed_query, packed_key);
  check_value(value, packed_query);
  TORCH_CHECK(offsets.is_cuda() && occurrences.is_cuda());
  TORCH_CHECK(offsets.device() == packed_query.device());
  TORCH_CHECK(occurrences.device() == packed_query.device());
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(occurrences.scalar_type() == torch::kInt32);
  TORCH_CHECK(offsets.is_contiguous() && occurrences.is_contiguous());
  TORCH_CHECK(
      offsets.sizes() == torch::IntArrayRef(
          {packed_query.size(0), packed_query.size(1), 257}),
      "offsets must have shape [B,H,257]");
  TORCH_CHECK(
      occurrences.sizes() == packed_query.sizes(),
      "occurrences must have shape [B,H,T]");
  TORCH_CHECK(max_suffix_length >= 1);
  TORCH_CHECK(method >= 0 && method <= 2, "method must be 0, 1, or 2");
  const c10::cuda::CUDAGuard device_guard(packed_query.device());
  return rosa_indexed_hard_forward_cuda(
      packed_query,
      packed_key,
      value,
      offsets,
      occurrences,
      static_cast<int>(std::min<int64_t>(
          max_suffix_length,
          packed_query.size(2))),
      static_cast<int>(method));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
diagonal_hard_forward(
    const torch::Tensor& packed_query,
    const torch::Tensor& packed_key,
    const torch::Tensor& value,
    int64_t max_suffix_length) {
  check_packed_pair(packed_query, packed_key);
  check_value(value, packed_query);
  TORCH_CHECK(max_suffix_length >= 1);
  const c10::cuda::CUDAGuard device_guard(packed_query.device());
  return rosa_diagonal_hard_forward_cuda(
      packed_query,
      packed_key,
      value,
      static_cast<int>(std::min<int64_t>(
          max_suffix_length,
          packed_query.size(2))));
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack_sign_bits", &pack_sign_bits);
  module.def("build_occurrence_index", &build_occurrence_index);
  module.def("indexed_hard_forward", &indexed_hard_forward);
  module.def("diagonal_hard_forward", &diagonal_hard_forward);
}
