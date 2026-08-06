#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t, typename acc_t>
__device__ acc_t hard_value(
    const scalar_t* value,
    int32_t route,
    int64_t feature,
    int64_t feature_size) {
  if (route == 0) {
    return acc_t{0};
  }
  return value[static_cast<int64_t>(route) * feature_size + feature] >
          scalar_t{0}
      ? acc_t{1}
      : acc_t{-1};
}

template <typename acc_t>
__device__ acc_t block_sum(acc_t value) {
  __shared__ acc_t values[kThreads];
  values[threadIdx.x] = value;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      values[threadIdx.x] += values[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return values[0];
}

template <typename scalar_t, typename acc_t>
__global__ void query_vjp_kernel(
    const int32_t* base_routes,
    const int64_t* offsets,
    const int32_t* changes,
    const uint8_t* query_codes,
    const scalar_t* value,
    const scalar_t* grad_output,
    int64_t feature_size,
    int32_t bit_width,
    scalar_t* output) {
  const int32_t flip = blockIdx.x;
  acc_t partial = acc_t{0};
  for (int64_t descriptor = offsets[flip];
       descriptor < offsets[flip + 1];
       ++descriptor) {
    const int32_t* change = changes + descriptor * 6;
    for (int32_t row = change[0]; row < change[1]; ++row) {
      const int32_t offset = row - change[0];
      const int32_t route = change[4] + offset * change[5];
      const int32_t base_route = base_routes[row];
      for (int64_t feature = threadIdx.x;
           feature < feature_size;
           feature += blockDim.x) {
        const acc_t difference =
            hard_value<scalar_t, acc_t>(
                value, route, feature, feature_size) -
            hard_value<scalar_t, acc_t>(
                value, base_route, feature, feature_size);
        partial += difference * static_cast<acc_t>(
            grad_output[static_cast<int64_t>(row) * feature_size + feature]);
      }
    }
  }
  const acc_t delta = block_sum(partial);
  if (threadIdx.x == 0) {
    const int32_t position = flip / bit_width + 1;
    const int32_t bit = flip % bit_width;
    const acc_t sign = (query_codes[position] & (uint8_t{1} << bit)) != 0
        ? acc_t{1}
        : acc_t{-1};
    output[flip] = static_cast<scalar_t>(-sign * delta);
  }
}

template <typename scalar_t, typename acc_t>
__global__ void delete_vjp_kernel(
    const int32_t* base_routes,
    const int64_t* offsets,
    const int32_t* changes,
    const scalar_t* value,
    const scalar_t* grad_output,
    int64_t feature_size,
    scalar_t* delete_delta) {
  const int32_t key_position = blockIdx.x;
  acc_t partial = acc_t{0};
  for (int64_t descriptor = offsets[key_position];
       descriptor < offsets[key_position + 1];
       ++descriptor) {
    const int32_t* change = changes + descriptor * 6;
    for (int32_t row = change[0]; row < change[1]; ++row) {
      const int32_t offset = row - change[0];
      const int32_t route = change[4] + offset * change[5];
      const int32_t base_route = base_routes[row];
      for (int64_t feature = threadIdx.x;
           feature < feature_size;
           feature += blockDim.x) {
        const acc_t difference =
            hard_value<scalar_t, acc_t>(
                value, route, feature, feature_size) -
            hard_value<scalar_t, acc_t>(
                value, base_route, feature, feature_size);
        partial += difference * static_cast<acc_t>(
            grad_output[static_cast<int64_t>(row) * feature_size + feature]);
      }
    }
  }
  const acc_t delta = block_sum(partial);
  if (threadIdx.x == 0) {
    delete_delta[key_position] = static_cast<scalar_t>(delta);
  }
}

template <typename scalar_t, typename acc_t>
__global__ void key_vjp_kernel(
    const int64_t* offsets,
    const int32_t* overrides,
    const uint8_t* key_codes,
    const scalar_t* value,
    const scalar_t* grad_output,
    const scalar_t* delete_delta,
    int64_t feature_size,
    int32_t bit_width,
    int32_t query_flip_count,
    scalar_t* output) {
  const int32_t local_flip = blockIdx.x;
  acc_t partial = acc_t{0};
  for (int64_t descriptor = offsets[local_flip];
       descriptor < offsets[local_flip + 1];
       ++descriptor) {
    const int32_t* change = overrides + descriptor * 10;
    for (int32_t row = change[0]; row < change[1]; ++row) {
      const int32_t offset = row - change[0];
      const int32_t from_route = change[4] + offset * change[5];
      const int32_t to_route = change[8] + offset * change[9];
      for (int64_t feature = threadIdx.x;
           feature < feature_size;
           feature += blockDim.x) {
        const acc_t difference =
            hard_value<scalar_t, acc_t>(
                value, to_route, feature, feature_size) -
            hard_value<scalar_t, acc_t>(
                value, from_route, feature, feature_size);
        partial += difference * static_cast<acc_t>(
            grad_output[static_cast<int64_t>(row) * feature_size + feature]);
      }
    }
  }
  const acc_t correction = block_sum(partial);
  if (threadIdx.x == 0) {
    const int32_t key_position = local_flip / bit_width;
    const int32_t bit = local_flip % bit_width;
    const acc_t sign = (key_codes[key_position] & (uint8_t{1} << bit)) != 0
        ? acc_t{1}
        : acc_t{-1};
    const acc_t delta =
        static_cast<acc_t>(delete_delta[key_position]) + correction;
    output[query_flip_count + local_flip] =
        static_cast<scalar_t>(-sign * delta);
  }
}

}  // namespace

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
    int64_t bit_width) {
  const at::cuda::CUDAGuard device_guard(value.device());
  const int64_t query_flip_count = query_offsets.numel() - 1;
  const int64_t key_length = delete_offsets.numel() - 1;
  auto output = torch::empty(
      {2 * query_flip_count}, value.options());
  auto delete_delta = torch::empty({key_length}, value.options());
  if (query_flip_count == 0) {
    return output;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      value.scalar_type(),
      "sam_bitflip_descriptor_vjp",
      [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        query_vjp_kernel<scalar_t, acc_t>
            <<<query_flip_count, kThreads, 0, stream>>>(
                base_routes.data_ptr<int32_t>(),
                query_offsets.data_ptr<int64_t>(),
                query_changes.data_ptr<int32_t>(),
                query_codes.data_ptr<uint8_t>(),
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                value.size(1),
                static_cast<int32_t>(bit_width),
                output.data_ptr<scalar_t>());
        delete_vjp_kernel<scalar_t, acc_t>
            <<<key_length, kThreads, 0, stream>>>(
                base_routes.data_ptr<int32_t>(),
                delete_offsets.data_ptr<int64_t>(),
                delete_changes.data_ptr<int32_t>(),
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                value.size(1),
                delete_delta.data_ptr<scalar_t>());
        key_vjp_kernel<scalar_t, acc_t>
            <<<query_flip_count, kThreads, 0, stream>>>(
                override_offsets.data_ptr<int64_t>(),
                overrides.data_ptr<int32_t>(),
                key_codes.data_ptr<uint8_t>(),
                value.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                delete_delta.data_ptr<scalar_t>(),
                value.size(1),
                static_cast<int32_t>(bit_width),
                static_cast<int32_t>(query_flip_count),
                output.data_ptr<scalar_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
