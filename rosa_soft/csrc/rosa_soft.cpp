#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>


std::vector<torch::Tensor> soft_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t max_suffix_length);

std::vector<torch::Tensor> soft_backward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor query_symbols,
    torch::Tensor key_symbols,
    torch::Tensor rng_seed,
    int64_t max_suffix_length,
    float inverse_route_temperature,
    float mismatch_penalty);


namespace {

bool is_supported_dtype(c10::ScalarType dtype) {
  return dtype == torch::kFloat32 ||
      dtype == torch::kFloat16 ||
      dtype == torch::kBFloat16;
}

float positive_normal_float(double value, const char* name) {
  const float converted = static_cast<float>(value);
  TORCH_CHECK(
      converted > 0.0f && std::isnormal(converted),
      name,
      " must be representable as a positive normal float32 value");
  return converted;
}

void check_common(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    int64_t max_suffix_length) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(
      query.device() == key.device(),
      "query and key must be on the same CUDA device");
  TORCH_CHECK(
      query.device() == value.device(),
      "query and value must be on the same CUDA device");
  TORCH_CHECK(
      is_supported_dtype(query.scalar_type()),
      "query must be float32, float16, or bfloat16");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(query.scalar_type() == value.scalar_type(), "query/value dtype mismatch");
  TORCH_CHECK(query.dim() == 4, "query must have shape (B, T, H, D)");
  TORCH_CHECK(key.dim() == 4, "key must have shape (B, T, H, D)");
  TORCH_CHECK(value.dim() == 4, "value must have shape (B, T, H_v, D_v)");
  TORCH_CHECK(query.size(0) > 0, "batch size must be positive");
  TORCH_CHECK(query.size(1) > 0, "sequence length must be positive");
  TORCH_CHECK(query.size(2) > 0, "query head count must be positive");
  TORCH_CHECK(value.size(2) > 0, "value head count must be positive");
  TORCH_CHECK(value.size(3) > 0, "value dimension must be positive");
  TORCH_CHECK(query.size(0) == key.size(0), "query/key batch mismatch");
  TORCH_CHECK(query.size(1) == key.size(1), "query/key sequence mismatch");
  TORCH_CHECK(query.size(2) == key.size(2), "query/key head mismatch");
  TORCH_CHECK(query.size(3) == key.size(3), "query/key bit dimension mismatch");
  TORCH_CHECK(query.size(0) == value.size(0), "query/value batch mismatch");
  TORCH_CHECK(query.size(1) == value.size(1), "query/value sequence mismatch");
  TORCH_CHECK(
      query.size(2) % value.size(2) == 0,
      "query heads must be divisible by value heads");
  TORCH_CHECK(
      query.size(3) > 0 && query.size(3) <= 32,
      "query/key bit dimension must be in [1, 32]");
  TORCH_CHECK(
      query.numel() / query.size(3) <=
          std::numeric_limits<int>::max(),
      "B * T * H must fit in int32");
  TORCH_CHECK(
      value.size(3) <= std::numeric_limits<int>::max(),
      "value dimension must fit in int32");
  TORCH_CHECK(max_suffix_length >= 1, "max_suffix_length must be >= 1");
}

void check_symbols(
    const torch::Tensor& bits,
    const torch::Tensor& query,
    const char* name) {
  TORCH_CHECK(bits.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      bits.device() == query.device(),
      name,
      " must be on the same CUDA device as query");
  TORCH_CHECK(bits.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(bits.dim() == 3, name, " must have shape (B, T, H)");
  TORCH_CHECK(bits.size(0) == query.size(0), name, " batch mismatch");
  TORCH_CHECK(bits.size(1) == query.size(1), name, " sequence mismatch");
  TORCH_CHECK(bits.size(2) == query.size(2), name, " head mismatch");
}

void check_backward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& query_symbols,
    const torch::Tensor& key_symbols,
    const torch::Tensor& rng_seed,
    int64_t max_suffix_length,
    double route_temperature,
    double mismatch_penalty) {
  check_common(query, key, value, max_suffix_length);
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(
      grad_output.device() == query.device(),
      "grad_output must be on the same CUDA device as query");
  TORCH_CHECK(
      grad_output.scalar_type() == query.scalar_type(),
      "grad_output dtype mismatch");
  TORCH_CHECK(
      grad_output.sizes() ==
          torch::IntArrayRef(
              {query.size(0), query.size(1), query.size(2), value.size(3)}),
      "grad_output shape mismatch");
  check_symbols(query_symbols, query, "query_symbols");
  check_symbols(key_symbols, query, "key_symbols");
  TORCH_CHECK(rng_seed.is_cuda(), "rng_seed must be a CUDA tensor");
  TORCH_CHECK(
      rng_seed.device() == query.device(),
      "rng_seed must be on the same CUDA device as query");
  TORCH_CHECK(rng_seed.scalar_type() == torch::kInt64, "rng_seed must be int64");
  TORCH_CHECK(rng_seed.numel() == 1, "rng_seed must contain one value");
  TORCH_CHECK(
      std::isfinite(route_temperature) && route_temperature > 0.0,
      "route_temperature must be finite and > 0");
  TORCH_CHECK(
      std::isfinite(mismatch_penalty) && mismatch_penalty > 0.0,
      "mismatch_penalty must be finite and > 0");
}

}  // namespace


std::vector<torch::Tensor> soft_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t max_suffix_length) {
  check_common(query, key, value, max_suffix_length);
  const int64_t normalized_window =
      std::min<int64_t>(max_suffix_length, query.size(1));
  return soft_forward_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      normalized_window);
}


std::vector<torch::Tensor> soft_backward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor query_symbols,
    torch::Tensor key_symbols,
    torch::Tensor rng_seed,
    int64_t max_suffix_length,
    double route_temperature,
    double mismatch_penalty) {
  check_backward(
      query,
      key,
      value,
      grad_output,
      query_symbols,
      key_symbols,
      rng_seed,
      max_suffix_length,
      route_temperature,
      mismatch_penalty);
  const int64_t normalized_window =
      std::min<int64_t>(max_suffix_length, query.size(1));
  const float inverse_route_temperature =
      positive_normal_float(1.0 / route_temperature, "inverse route_temperature");
  TORCH_CHECK(
      inverse_route_temperature <=
          std::numeric_limits<float>::max() /
              static_cast<float>(normalized_window),
      "max_suffix_length / route_temperature must fit in float32");
  const float mismatch_penalty_f =
      positive_normal_float(mismatch_penalty, "mismatch_penalty");
  return soft_backward_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      grad_output.contiguous(),
      query_symbols.contiguous(),
      key_symbols.contiguous(),
      rng_seed.contiguous(),
      normalized_window,
      inverse_route_temperature,
      mismatch_penalty_f);
}


TORCH_LIBRARY_IMPL(rosa_soft, CUDA, m) {
  m.impl("soft_forward", &soft_forward);
  m.impl("soft_backward", &soft_backward);
}
