#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cassert>

void rwkv7_albatross_cuda_forward_w0_fp16_dither_seq(int B, int T, int C, int H, __half* s0, __half* r, __half* w, __half* k, __half* v, __half* a, __half* b, __half* y, int* elapsed_t);
void rwkv7_albatross_cuda_forward_w0_fp16_dither_one(int B, int C, int H, __half* s0, __half* r, __half* w, __half* k, __half* v, __half* a, __half* b, __half* y, int* elapsed_t);

template<typename T> struct CudaType { using type = T; };
template<> struct CudaType<at::BFloat16> { using type = __nv_bfloat16; };
template<> struct CudaType<at::Half> { using type = __half; };

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    do { \
        switch (TYPE) { \
            case at::ScalarType::Half:  { using scalar_t = at::Half;  (__VA_ARGS__)(); break; } \
            default: TORCH_CHECK(false, #NAME, " not implemented for '", toString(TYPE), "'"); \
        } \
    } while (false)

    
void rwkv7_albatross_forward_w0_fp16_dither_seq(at::Tensor& s0, at::Tensor& r, at::Tensor& w, at::Tensor& k, at::Tensor& v, at::Tensor& a, at::Tensor& b, at::Tensor& y, at::Tensor& elapsed_t) {
    int B = r.size(0);
    int T = r.size(1);
    int H = r.size(2);
    int head_size = r.size(3);
    int C = H * head_size;

    DISPATCH_FLOATING_TYPES(
        r.scalar_type(),
        "rwkv7_albatross_forward_w0_fp16_dither_seq",
        [&] {
            using F = typename CudaType<scalar_t>::type;
            rwkv7_albatross_cuda_forward_w0_fp16_dither_seq(
                B, T, C, H,
                reinterpret_cast<F*>(s0.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(r.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(w.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(k.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(v.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(a.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(b.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(y.data_ptr<scalar_t>()),
                elapsed_t.data_ptr<int>()
            );
        }
    );
}


void rwkv7_albatross_forward_w0_fp16_dither_one(at::Tensor& s0, at::Tensor& r, at::Tensor& w, at::Tensor& k, at::Tensor& v, at::Tensor& a, at::Tensor& b, at::Tensor& y, at::Tensor& elapsed_t) {
    int B = r.size(0);
    int H = r.size(1);
    int head_size = r.size(2);
    int C = H * head_size;

    DISPATCH_FLOATING_TYPES(
        r.scalar_type(),
        "rwkv7_albatross_forward_w0_fp16_dither_one",
        [&] {
            using F = typename CudaType<scalar_t>::type;
            rwkv7_albatross_cuda_forward_w0_fp16_dither_one(
                B, C, H,
                reinterpret_cast<F*>(s0.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(r.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(w.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(k.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(v.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(a.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(b.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(y.data_ptr<scalar_t>()),
                elapsed_t.data_ptr<int>()
            );
        }
    );
}

TORCH_LIBRARY_IMPL(rwkv_cuda, CUDA, m) {
    m.impl("rwkv7_albatross_forward_w0_fp16_dither_seq", &rwkv7_albatross_forward_w0_fp16_dither_seq);
    m.impl("rwkv7_albatross_forward_w0_fp16_dither_one", &rwkv7_albatross_forward_w0_fp16_dither_one);
}
