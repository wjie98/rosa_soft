#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>


std::vector<torch::Tensor> hard_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor payload,
    int64_t max_suffix_length);

std::vector<torch::Tensor> surrogate_vjp_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor payload,
    torch::Tensor grad_output,
    torch::Tensor packed_query_symbols,
    torch::Tensor packed_key_symbols,
    torch::Tensor rng_seed,
    int64_t max_suffix_length,
    float inverse_route_temperature,
    float mismatch_penalty);


namespace {

bool is_supported_logit_dtype(c10::ScalarType dtype) {
  return dtype == torch::kFloat32 ||
      dtype == torch::kFloat16 ||
      dtype == torch::kBFloat16;
}

float to_positive_normal_float(double value, const char* name) {
  const float converted = static_cast<float>(value);
  TORCH_CHECK(
      converted > 0.0f && std::isnormal(converted),
      name,
      " must be representable as a positive normal float32 value");
  return converted;
}

void check_common_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& payload,
    int64_t max_suffix_length) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(payload.is_cuda(), "payload must be a CUDA tensor");
  TORCH_CHECK(
      query.device() == key.device(),
      "query and key must be on the same CUDA device");
  TORCH_CHECK(
      query.device() == payload.device(),
      "query and payload must be on the same CUDA device");
  TORCH_CHECK(
      is_supported_logit_dtype(query.scalar_type()),
      "query must be float32, float16, or bfloat16");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(
      query.scalar_type() == payload.scalar_type(),
      "query/payload dtype mismatch");
  TORCH_CHECK(query.dim() == 4, "query must have shape (B, T, H, D)");
  TORCH_CHECK(key.dim() == 4, "key must have shape (B, T, H, D)");
  TORCH_CHECK(
      payload.dim() == 4,
      "payload must have shape (B, T, H_p, D_p)");
  TORCH_CHECK(query.size(0) > 0, "batch size must be positive");
  TORCH_CHECK(query.size(1) > 0, "sequence length must be positive");
  TORCH_CHECK(query.size(2) > 0, "query head count must be positive");
  TORCH_CHECK(payload.size(2) > 0, "payload head count must be positive");
  TORCH_CHECK(payload.size(3) > 0, "payload dimension must be positive");
  TORCH_CHECK(query.size(0) == key.size(0), "query/key batch mismatch");
  TORCH_CHECK(query.size(1) == key.size(1), "query/key sequence mismatch");
  TORCH_CHECK(query.size(2) == key.size(2), "query/key head mismatch");
  TORCH_CHECK(query.size(3) == key.size(3), "query/key bit dimension mismatch");
  TORCH_CHECK(
      query.size(0) == payload.size(0),
      "query/payload batch mismatch");
  TORCH_CHECK(
      query.size(1) == payload.size(1),
      "query/payload sequence mismatch");
  TORCH_CHECK(
      query.size(2) % payload.size(2) == 0,
      "query heads must be divisible by payload heads");
  TORCH_CHECK(
      query.size(3) > 0 && query.size(3) <= 32,
      "query/key bit dimension must be in [1, 32]");
  TORCH_CHECK(
      query.numel() / query.size(3) <=
          std::numeric_limits<int>::max(),
      "B * T * H must fit in int32");
  TORCH_CHECK(
      payload.size(3) <= std::numeric_limits<int>::max(),
      "payload dimension must fit in int32");
  TORCH_CHECK(max_suffix_length >= 1, "max_suffix_length must be >= 1");
}

void check_packed_symbols(
    const torch::Tensor& packed_symbols,
    const torch::Tensor& query,
    const char* name) {
  TORCH_CHECK(packed_symbols.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      packed_symbols.device() == query.device(),
      name,
      " must be on the same CUDA device as query");
  TORCH_CHECK(
      packed_symbols.scalar_type() == torch::kInt32,
      name,
      " must be int32");
  TORCH_CHECK(
      packed_symbols.dim() == 3,
      name,
      " must have shape (B, T, H)");
  TORCH_CHECK(
      packed_symbols.size(0) == query.size(0),
      name,
      " batch mismatch");
  TORCH_CHECK(
      packed_symbols.size(1) == query.size(1),
      name,
      " sequence mismatch");
  TORCH_CHECK(
      packed_symbols.size(2) == query.size(2),
      name,
      " head mismatch");
}

void check_surrogate_vjp_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& payload,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    const torch::Tensor& rng_seed,
    int64_t max_suffix_length,
    double route_temperature,
    double mismatch_penalty) {
  check_common_inputs(query, key, payload, max_suffix_length);
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
              {
                  query.size(0),
                  query.size(1),
                  query.size(2),
                  payload.size(3)}),
      "grad_output shape mismatch");
  check_packed_symbols(
      packed_query_symbols,
      query,
      "packed_query_symbols");
  check_packed_symbols(
      packed_key_symbols,
      query,
      "packed_key_symbols");
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


std::vector<torch::Tensor> hard_forward_op(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor payload,
    int64_t max_suffix_length) {
  check_common_inputs(query, key, payload, max_suffix_length);
  const int64_t effective_max_suffix_length =
      std::min<int64_t>(max_suffix_length, query.size(1));
  return hard_forward_cuda(
      query.contiguous(),
      key.contiguous(),
      payload.contiguous(),
      effective_max_suffix_length);
}


std::vector<torch::Tensor> surrogate_vjp_op(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor payload,
    torch::Tensor grad_output,
    torch::Tensor packed_query_symbols,
    torch::Tensor packed_key_symbols,
    torch::Tensor rng_seed,
    int64_t max_suffix_length,
    double route_temperature,
    double mismatch_penalty) {
  check_surrogate_vjp_inputs(
      query,
      key,
      payload,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      rng_seed,
      max_suffix_length,
      route_temperature,
      mismatch_penalty);
  const int64_t effective_max_suffix_length =
      std::min<int64_t>(max_suffix_length, query.size(1));
  const float inverse_route_temperature =
      to_positive_normal_float(
          1.0 / route_temperature,
          "inverse route_temperature");
  TORCH_CHECK(
      inverse_route_temperature <=
          std::numeric_limits<float>::max() /
              static_cast<float>(effective_max_suffix_length),
      "max_suffix_length / route_temperature must fit in float32");
  const float mismatch_penalty_f =
      to_positive_normal_float(mismatch_penalty, "mismatch_penalty");
  return surrogate_vjp_cuda(
      query.contiguous(),
      key.contiguous(),
      payload.contiguous(),
      grad_output.contiguous(),
      packed_query_symbols.contiguous(),
      packed_key_symbols.contiguous(),
      rng_seed.contiguous(),
      effective_max_suffix_length,
      inverse_route_temperature,
      mismatch_penalty_f);
}


TORCH_LIBRARY_IMPL(rosa_soft, CUDA, m) {
  m.impl("soft_forward", &hard_forward_op);
  m.impl("soft_backward", &surrogate_vjp_op);
}
