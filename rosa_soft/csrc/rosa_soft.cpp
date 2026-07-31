#include <ATen/Context.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hard_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t max_suffix_length);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
hard_forward_varlen_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens,
    int64_t max_suffix_length);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> surrogate_vjp_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor packed_query_symbols,
    torch::Tensor packed_key_symbols,
    torch::Tensor dropout_seed,
    int64_t max_suffix_length,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_scale,
    int gradient_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
surrogate_vjp_varlen_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens,
    torch::Tensor grad_output,
    torch::Tensor packed_query_symbols,
    torch::Tensor packed_key_symbols,
    torch::Tensor dropout_seed,
    int64_t max_suffix_length,
    float scale,
    float dropout_p,
    float inverse_keep_probability,
    float mismatch_scale,
    int gradient_mask);


namespace {

constexpr int64_t kKernelIndexSafetyMargin = 128;

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

struct SurrogateScalars {
  float scale;
  float dropout_p;
  float inverse_keep_probability;
  float mismatch_scale;
};

SurrogateScalars check_surrogate_scalars(
    int64_t effective_max_suffix_length,
    double scale,
    double dropout_p,
    double mismatch_scale) {
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and > 0");
  TORCH_CHECK(
      std::isfinite(dropout_p) &&
          dropout_p >= 0.0 &&
          dropout_p <= 1.0 - std::ldexp(1.0, -24),
      "dropout_p must be finite and in [0, 1 - 2^-24]");
  TORCH_CHECK(
      std::isfinite(mismatch_scale) && mismatch_scale > 0.0,
      "mismatch_scale must be finite and > 0");

  const float scale_f =
      to_positive_normal_float(scale, "scale");
  const float dropout_p_f = static_cast<float>(dropout_p);
  const float inverse_keep_probability_f =
      1.0f / (1.0f - dropout_p_f);
  const double float_max =
      static_cast<double>(std::numeric_limits<float>::max());
  TORCH_CHECK(
      scale <=
          float_max /
              static_cast<double>(effective_max_suffix_length),
      "max_suffix_length * scale must fit in float32");

  const float mismatch_scale_f =
      to_positive_normal_float(mismatch_scale, "mismatch_scale");
  const double scaled_horizon =
      static_cast<double>(effective_max_suffix_length) *
      scale;
  TORCH_CHECK(
      mismatch_scale <= float_max / scaled_horizon,
      "mismatch_scale * max_suffix_length * scale "
      "must fit in float32");
  return {
      scale_f,
      dropout_p_f,
      inverse_keep_probability_f,
      mismatch_scale_f,
  };
}

void check_common_inputs(
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
      is_supported_logit_dtype(query.scalar_type()),
      "query must be float32, float16, or bfloat16");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(
      query.scalar_type() == value.scalar_type(),
      "query/value dtype mismatch");
  TORCH_CHECK(query.dim() == 4, "query must have shape (B, T, H, D)");
  TORCH_CHECK(key.dim() == 4, "key must have shape (B, T, H, D)");
  TORCH_CHECK(
      value.dim() == 4,
      "value must have shape (B, T, H_p, D_p)");
  TORCH_CHECK(query.size(0) > 0, "batch size must be positive");
  TORCH_CHECK(query.size(1) > 0, "sequence length must be positive");
  TORCH_CHECK(query.size(2) > 0, "query head count must be positive");
  TORCH_CHECK(value.size(2) > 0, "value head count must be positive");
  TORCH_CHECK(value.size(3) > 0, "value dimension must be positive");
  TORCH_CHECK(query.size(0) == key.size(0), "query/key batch mismatch");
  TORCH_CHECK(query.size(1) == key.size(1), "query/key sequence mismatch");
  TORCH_CHECK(query.size(2) == key.size(2), "query/key head mismatch");
  TORCH_CHECK(query.size(3) == key.size(3), "query/key bit dimension mismatch");
  TORCH_CHECK(
      query.size(0) == value.size(0),
      "query/value batch mismatch");
  TORCH_CHECK(
      query.size(1) == value.size(1),
      "query/value sequence mismatch");
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
      query.size(1) <=
          std::numeric_limits<int>::max() -
              kKernelIndexSafetyMargin,
      "sequence length is too large for the CUDA kernel index stride");
  TORCH_CHECK(
      value.size(3) <=
          std::numeric_limits<int>::max() -
              kKernelIndexSafetyMargin,
      "value dimension is too large for the CUDA kernel index stride");
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
      " must have shape (B, H, T)");
  TORCH_CHECK(
      packed_symbols.size(0) == query.size(0),
      name,
      " batch mismatch");
  TORCH_CHECK(
      packed_symbols.size(1) == query.size(2),
      name,
      " head mismatch");
  TORCH_CHECK(
      packed_symbols.size(2) == query.size(1),
      name,
      " sequence mismatch");
}

void check_dropout_seed(
    const torch::Tensor& dropout_seed,
    const torch::Tensor& query,
    double dropout_p) {
  TORCH_CHECK(
      dropout_seed.is_cuda(),
      "dropout_seed must be a CUDA tensor");
  TORCH_CHECK(
      dropout_seed.device() == query.device(),
      "dropout_seed must be on the same CUDA device as query");
  TORCH_CHECK(
      dropout_seed.scalar_type() == torch::kInt64,
      "dropout_seed must have dtype int64");
  TORCH_CHECK(
      dropout_seed.numel() == (dropout_p > 0.0 ? 1 : 0),
      "dropout_seed must be scalar exactly when dropout_p > 0");
}

void check_surrogate_vjp_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    const torch::Tensor& packed_query_symbols,
    const torch::Tensor& packed_key_symbols,
    int64_t max_suffix_length) {
  check_common_inputs(query, key, value, max_suffix_length);
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
                  value.size(3)}),
      "grad_output shape mismatch");
  check_packed_symbols(
      packed_query_symbols,
      query,
      "packed_query_symbols");
  check_packed_symbols(
      packed_key_symbols,
      query,
      "packed_key_symbols");
}

void check_gradient_mask(int64_t gradient_mask) {
  TORCH_CHECK(
      gradient_mask >= 1 && gradient_mask <= 7,
      "gradient_mask must be an integer bit mask in [1, 7]");
}

void check_varlen_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& cu_seqlens,
    int64_t max_suffix_length) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(cu_seqlens.is_cuda(), "cu_seqlens must be a CUDA tensor");
  TORCH_CHECK(
      query.device() == key.device() &&
          query.device() == value.device() &&
          query.device() == cu_seqlens.device(),
      "all inputs and cu_seqlens must be on the same CUDA device");
  TORCH_CHECK(
      is_supported_logit_dtype(query.scalar_type()),
      "query must be float32, float16, or bfloat16");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(
      query.scalar_type() == value.scalar_type(),
      "query/value dtype mismatch");
  TORCH_CHECK(query.dim() == 3, "query must have shape (N, H, D)");
  TORCH_CHECK(key.dim() == 3, "key must have shape (N, H, D)");
  TORCH_CHECK(value.dim() == 3, "value must have shape (N, H_p, D_p)");
  TORCH_CHECK(query.size(0) > 0, "packed token count must be positive");
  TORCH_CHECK(query.size(1) > 0, "query head count must be positive");
  TORCH_CHECK(value.size(1) > 0, "value head count must be positive");
  TORCH_CHECK(value.size(2) > 0, "value dimension must be positive");
  TORCH_CHECK(query.size(0) == key.size(0), "query/key token mismatch");
  TORCH_CHECK(query.size(0) == value.size(0), "query/value token mismatch");
  TORCH_CHECK(query.size(1) == key.size(1), "query/key head mismatch");
  TORCH_CHECK(query.size(2) == key.size(2), "query/key bit dimension mismatch");
  TORCH_CHECK(
      query.size(1) % value.size(1) == 0,
      "query heads must be divisible by value heads");
  TORCH_CHECK(
      query.size(2) > 0 && query.size(2) <= 32,
      "query/key bit dimension must be in [1, 32]");
  TORCH_CHECK(
      query.size(0) * query.size(1) <=
          std::numeric_limits<int>::max(),
      "N * H must fit in int32");
  TORCH_CHECK(
      query.size(0) <=
          std::numeric_limits<int>::max() -
              kKernelIndexSafetyMargin,
      "packed token count is too large for CUDA kernel indexing");
  TORCH_CHECK(
      value.size(2) <=
          std::numeric_limits<int>::max() -
              kKernelIndexSafetyMargin,
      "value dimension is too large for the CUDA kernel index stride");
  TORCH_CHECK(
      cu_seqlens.scalar_type() == torch::kInt32,
      "cu_seqlens must be int32");
  TORCH_CHECK(
      cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
      "cu_seqlens must be one-dimensional with at least two entries");
  TORCH_CHECK(
      cu_seqlens.numel() - 1 <=
          std::numeric_limits<int>::max() -
              kKernelIndexSafetyMargin,
      "number of packed sequences is too large for CUDA indexing");
  TORCH_CHECK(max_suffix_length >= 1, "max_suffix_length must be >= 1");
}

void check_varlen_packed_symbols(
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
      packed_symbols.dim() == 2,
      name,
      " must have shape (H, N)");
  TORCH_CHECK(
      packed_symbols.size(0) == query.size(1) &&
          packed_symbols.size(1) == query.size(0),
      name,
      " shape mismatch");
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hard_forward_op(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t max_suffix_length) {
  check_common_inputs(query, key, value, max_suffix_length);
  const int64_t effective_max_suffix_length =
      std::min<int64_t>(max_suffix_length, query.size(1));
  return hard_forward_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      effective_max_suffix_length);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
hard_forward_varlen_op(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens,
    int64_t max_suffix_length) {
  check_varlen_inputs(
      query,
      key,
      value,
      cu_seqlens,
      max_suffix_length);
  const int64_t effective_max_suffix_length =
      std::min<int64_t>(max_suffix_length, query.size(0));
  return hard_forward_varlen_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      cu_seqlens.contiguous(),
      effective_max_suffix_length);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
surrogate_vjp_masked_op(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor packed_query_symbols,
    torch::Tensor packed_key_symbols,
    torch::Tensor dropout_seed,
    int64_t max_suffix_length,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t gradient_mask) {
  check_surrogate_vjp_inputs(
      query,
      key,
      value,
      grad_output,
      packed_query_symbols,
      packed_key_symbols,
      max_suffix_length);
  const int64_t effective_max_suffix_length =
      std::min<int64_t>(max_suffix_length, query.size(1));
  const auto surrogate_scalars = check_surrogate_scalars(
      effective_max_suffix_length,
      scale,
      dropout_p,
      mismatch_scale);
  check_dropout_seed(dropout_seed, query, dropout_p);
  check_gradient_mask(gradient_mask);
  at::globalContext().alertNotDeterministic(
      "rosa_soft::surrogate_vjp_masked");
  return surrogate_vjp_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      grad_output.contiguous(),
      packed_query_symbols.contiguous(),
      packed_key_symbols.contiguous(),
      dropout_seed.contiguous(),
      effective_max_suffix_length,
      surrogate_scalars.scale,
      surrogate_scalars.dropout_p,
      surrogate_scalars.inverse_keep_probability,
      surrogate_scalars.mismatch_scale,
      static_cast<int>(gradient_mask));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
surrogate_vjp_varlen_masked_op(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens,
    torch::Tensor grad_output,
    torch::Tensor packed_query_symbols,
    torch::Tensor packed_key_symbols,
    torch::Tensor dropout_seed,
    int64_t max_suffix_length,
    double scale,
    double dropout_p,
    double mismatch_scale,
    int64_t gradient_mask) {
  check_varlen_inputs(
      query,
      key,
      value,
      cu_seqlens,
      max_suffix_length);
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
                  value.size(2)}),
      "grad_output shape mismatch");
  check_varlen_packed_symbols(
      packed_query_symbols,
      query,
      "packed_query_symbols");
  check_varlen_packed_symbols(
      packed_key_symbols,
      query,
      "packed_key_symbols");
  const int64_t effective_max_suffix_length =
      std::min<int64_t>(max_suffix_length, query.size(0));
  const auto surrogate_scalars = check_surrogate_scalars(
      effective_max_suffix_length,
      scale,
      dropout_p,
      mismatch_scale);
  check_dropout_seed(dropout_seed, query, dropout_p);
  check_gradient_mask(gradient_mask);
  at::globalContext().alertNotDeterministic(
      "rosa_soft::surrogate_vjp_varlen_masked");
  return surrogate_vjp_varlen_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      cu_seqlens.contiguous(),
      grad_output.contiguous(),
      packed_query_symbols.contiguous(),
      packed_key_symbols.contiguous(),
      dropout_seed.contiguous(),
      effective_max_suffix_length,
      surrogate_scalars.scale,
      surrogate_scalars.dropout_p,
      surrogate_scalars.inverse_keep_probability,
      surrogate_scalars.mismatch_scale,
      static_cast<int>(gradient_mask));
}


TORCH_LIBRARY_IMPL(rosa_soft, CUDA, m) {
  m.impl("hard_forward", &hard_forward_op);
  m.impl("hard_forward_varlen", &hard_forward_varlen_op);
  m.impl("surrogate_vjp_masked", &surrogate_vjp_masked_op);
  m.impl(
      "surrogate_vjp_varlen_masked",
      &surrogate_vjp_varlen_masked_op);
}
