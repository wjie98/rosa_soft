#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

torch::Tensor rosa_anchor_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon);

std::vector<torch::Tensor> rosa_anchor_forward_with_stats_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon);

std::vector<torch::Tensor> rosa_anchor_forward_with_bits_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon);

std::vector<torch::Tensor> rosa_anchor_backward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon,
    double qk_damper_strength);

std::vector<torch::Tensor> rosa_anchor_backward_with_bits_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor q_bits,
    torch::Tensor k_bits,
    torch::Tensor row_max,
    torch::Tensor row_denom,
    double scale,
    int64_t suffix_window,
    double match_lambda,
    double sink_threshold,
    double tie_strength,
    double logit_epsilon,
    double qk_damper_strength);

namespace {

constexpr double kLambdaPower = 1.5;
constexpr double kSinkThreshold = 0.5;
constexpr double kTieStrength = 0.25;

bool is_supported_dtype(c10::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

double match_lambda(double scale, int64_t suffix_window) {
  return std::log(static_cast<double>(suffix_window)) +
      kLambdaPower * std::log1p(std::abs(scale));
}

void check_common(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    int64_t suffix_window) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(is_supported_dtype(query.scalar_type()), "query must be float32, float16, or bfloat16");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(query.scalar_type() == value.scalar_type(), "query/value dtype mismatch");
  TORCH_CHECK(query.dim() == 4, "query must have shape (B, T, H, D)");
  TORCH_CHECK(key.dim() == 4, "key must have shape (B, T, H, D)");
  TORCH_CHECK(value.dim() == 4, "value must have shape (B, T, H_v, D_v)");
  TORCH_CHECK(query.size(0) == key.size(0), "query/key batch mismatch");
  TORCH_CHECK(query.size(1) == key.size(1), "query/key sequence mismatch");
  TORCH_CHECK(query.size(2) == key.size(2), "query/key head mismatch");
  TORCH_CHECK(query.size(3) == key.size(3), "query/key bit dimension mismatch");
  TORCH_CHECK(query.size(0) == value.size(0), "query/value batch mismatch");
  TORCH_CHECK(query.size(1) == value.size(1), "query/value sequence mismatch");
  TORCH_CHECK(query.size(2) % value.size(2) == 0, "query heads must be divisible by value heads");
  TORCH_CHECK(query.size(3) > 0 && query.size(3) <= 32, "query/key bit dimension must be in [1, 32]");
  TORCH_CHECK(suffix_window >= 1, "suffix_window must be >= 1");
}

void check_logit_epsilon(double logit_epsilon) {
  TORCH_CHECK(logit_epsilon >= 0.0, "logit_epsilon must be >= 0");
}

void check_qk_damper(double qk_damper_strength) {
  TORCH_CHECK(
      qk_damper_strength >= 0.0 && qk_damper_strength <= 1.0,
      "qk_damper_strength must be in [0, 1]");
}

void check_backward_common(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& grad_output,
    int64_t suffix_window,
    double logit_epsilon,
    double qk_damper_strength) {
  check_common(query, key, value, suffix_window);
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(grad_output.scalar_type() == query.scalar_type(), "grad_output dtype mismatch");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output must have shape (B, T, H, D_v)");
  TORCH_CHECK(grad_output.size(0) == query.size(0), "grad_output batch mismatch");
  TORCH_CHECK(grad_output.size(1) == query.size(1), "grad_output sequence mismatch");
  TORCH_CHECK(grad_output.size(2) == query.size(2), "grad_output head mismatch");
  TORCH_CHECK(grad_output.size(3) == value.size(3), "grad_output value dimension mismatch");
  TORCH_CHECK(suffix_window <= 512, "RosaAnchor CUDA backward supports suffix_window <= 512");
  check_logit_epsilon(logit_epsilon);
  check_qk_damper(qk_damper_strength);
}

void check_bits(
    const torch::Tensor& q_bits,
    const torch::Tensor& k_bits,
    const torch::Tensor& query) {
  TORCH_CHECK(q_bits.is_cuda(), "q_bits must be a CUDA tensor");
  TORCH_CHECK(k_bits.is_cuda(), "k_bits must be a CUDA tensor");
  TORCH_CHECK(q_bits.scalar_type() == torch::kInt32, "q_bits must be int32");
  TORCH_CHECK(k_bits.scalar_type() == torch::kInt32, "k_bits must be int32");
  TORCH_CHECK(q_bits.dim() == 3, "q_bits must have shape (B, T, H)");
  TORCH_CHECK(k_bits.dim() == 3, "k_bits must have shape (B, T, H)");
  TORCH_CHECK(q_bits.size(0) == query.size(0), "q_bits batch mismatch");
  TORCH_CHECK(q_bits.size(1) == query.size(1), "q_bits sequence mismatch");
  TORCH_CHECK(q_bits.size(2) == query.size(2), "q_bits head mismatch");
  TORCH_CHECK(k_bits.sizes() == q_bits.sizes(), "k_bits/q_bits shape mismatch");
}

void check_row_stats(
    const torch::Tensor& row_max,
    const torch::Tensor& row_denom,
    const torch::Tensor& query) {
  TORCH_CHECK(row_max.is_cuda(), "row_max must be a CUDA tensor");
  TORCH_CHECK(row_denom.is_cuda(), "row_denom must be a CUDA tensor");
  TORCH_CHECK(row_max.scalar_type() == torch::kFloat32, "row_max must be float32");
  TORCH_CHECK(row_denom.scalar_type() == torch::kFloat32, "row_denom must be float32");
  TORCH_CHECK(row_max.dim() == 3, "row_max must have shape (B, H, T)");
  TORCH_CHECK(row_denom.dim() == 3, "row_denom must have shape (B, H, T)");
  TORCH_CHECK(row_max.size(0) == query.size(0), "row_max batch mismatch");
  TORCH_CHECK(row_max.size(1) == query.size(2), "row_max head mismatch");
  TORCH_CHECK(row_max.size(2) == query.size(1), "row_max sequence mismatch");
  TORCH_CHECK(row_denom.sizes() == row_max.sizes(), "row_denom/row_max shape mismatch");
}

}  // namespace

torch::Tensor rosa_anchor_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double logit_epsilon) {
  check_common(query, key, value, suffix_window);
  check_logit_epsilon(logit_epsilon);

  return rosa_anchor_forward_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      scale,
      suffix_window,
      match_lambda(scale, suffix_window),
      kSinkThreshold,
      kTieStrength,
      logit_epsilon);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rosa_anchor_forward_with_stats(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double logit_epsilon) {
  check_common(query, key, value, suffix_window);
  check_logit_epsilon(logit_epsilon);

  auto result = rosa_anchor_forward_with_stats_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      scale,
      suffix_window,
      match_lambda(scale, suffix_window),
      kSinkThreshold,
      kTieStrength,
      logit_epsilon);
  TORCH_CHECK(
      result.size() == 6,
      "rosa_anchor_forward_with_stats_cuda must return output, stats, bits, and row stats");
  return std::make_tuple(result[0], result[1], result[2], result[3], result[4], result[5]);
}

std::vector<torch::Tensor> rosa_anchor_forward_with_bits(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double scale,
    int64_t suffix_window,
    double logit_epsilon) {
  check_common(query, key, value, suffix_window);
  check_logit_epsilon(logit_epsilon);

  return rosa_anchor_forward_with_bits_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      scale,
      suffix_window,
      match_lambda(scale, suffix_window),
      kSinkThreshold,
      kTieStrength,
      logit_epsilon);
}

std::vector<torch::Tensor> rosa_anchor_backward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    double scale,
    int64_t suffix_window,
    double logit_epsilon,
    double qk_damper_strength) {
  check_backward_common(
      query, key, value, grad_output, suffix_window, logit_epsilon, qk_damper_strength);

  return rosa_anchor_backward_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      grad_output.contiguous(),
      scale,
      suffix_window,
      match_lambda(scale, suffix_window),
      kSinkThreshold,
      kTieStrength,
      logit_epsilon,
      qk_damper_strength);
}

std::vector<torch::Tensor> rosa_anchor_backward_with_bits(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor q_bits,
    torch::Tensor k_bits,
    double scale,
    int64_t suffix_window,
    double logit_epsilon,
    double qk_damper_strength) {
  check_backward_common(
      query, key, value, grad_output, suffix_window, logit_epsilon, qk_damper_strength);
  check_bits(q_bits, k_bits, query);

  return rosa_anchor_backward_with_bits_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      grad_output.contiguous(),
      q_bits.contiguous(),
      k_bits.contiguous(),
      torch::Tensor(),
      torch::Tensor(),
      scale,
      suffix_window,
      match_lambda(scale, suffix_window),
      kSinkThreshold,
      kTieStrength,
      logit_epsilon,
      qk_damper_strength);
}

std::vector<torch::Tensor> rosa_anchor_backward_with_bits_and_stats(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor grad_output,
    torch::Tensor q_bits,
    torch::Tensor k_bits,
    torch::Tensor row_max,
    torch::Tensor row_denom,
    double scale,
    int64_t suffix_window,
    double logit_epsilon,
    double qk_damper_strength) {
  check_backward_common(
      query, key, value, grad_output, suffix_window, logit_epsilon, qk_damper_strength);
  check_bits(q_bits, k_bits, query);
  check_row_stats(row_max, row_denom, query);

  return rosa_anchor_backward_with_bits_cuda(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      grad_output.contiguous(),
      q_bits.contiguous(),
      k_bits.contiguous(),
      row_max.contiguous(),
      row_denom.contiguous(),
      scale,
      suffix_window,
      match_lambda(scale, suffix_window),
      kSinkThreshold,
      kTieStrength,
      logit_epsilon,
      qk_damper_strength);
}

TORCH_LIBRARY_IMPL(rosa_soft, CUDA, m) {
  m.impl("rosa_anchor_forward", &rosa_anchor_forward);
  m.impl("rosa_anchor_forward_with_stats", &rosa_anchor_forward_with_stats);
  m.impl("rosa_anchor_forward_with_bits", &rosa_anchor_forward_with_bits);
  m.impl("rosa_anchor_backward", &rosa_anchor_backward);
  m.impl("rosa_anchor_backward_with_bits", &rosa_anchor_backward_with_bits);
  m.impl("rosa_anchor_backward_with_bits_and_stats", &rosa_anchor_backward_with_bits_and_stats);
}
