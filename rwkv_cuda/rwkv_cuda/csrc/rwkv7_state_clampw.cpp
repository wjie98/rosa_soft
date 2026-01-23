#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cassert>

template<typename F> void rwkv7_state_clampw_cuda_forward(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, float* s0, F* r, F* w, F* k, F* v, F* a, F* b, F* y, float* s, float* sa);
template<typename F> void rwkv7_state_clampw_cuda_backward(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, F* r, F* w, F* k, F* v, F* a, F* b, F* dy, float* s, float* sa, float* ds0, F* dr, F* dw, F* dk, F* dv, F* da, F* db);

template<typename T> struct CudaType { using type = T; };
template<> struct CudaType<at::BFloat16> { using type = __nv_bfloat16; };
template<> struct CudaType<at::Half> { using type = __half; };

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    do { \
        switch (TYPE) { \
            case at::ScalarType::Float:  { using scalar_t = float;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Half:  { using scalar_t = at::Half;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::BFloat16:  { using scalar_t = at::BFloat16;  (__VA_ARGS__)(); break; } \
            default: TORCH_CHECK(false, #NAME, " not implemented for '", toString(TYPE), "'"); \
        } \
    } while (false)

    
void rwkv7_state_clampw_forward(
    at::Tensor& s0,
    at::Tensor& r, at::Tensor& w, at::Tensor& k, at::Tensor& v, at::Tensor& a, at::Tensor& b,
    at::Tensor& y, at::Tensor& s, at::Tensor& sa
) {
    int batch_size = r.size(0);
    int seq_len = r.size(1);
    int num_heads = r.size(2);
    int head_size = r.size(3);
    int chunk_len = 16;

    assert(seq_len % chunk_len == 0);

    DISPATCH_FLOATING_TYPES(
        r.scalar_type(),
        "rwkv7_state_clampw_forward",
        [&] {
            using F = typename CudaType<scalar_t>::type;
            rwkv7_state_clampw_cuda_forward<F>(
                batch_size, seq_len, num_heads, head_size, chunk_len,
                s0.data_ptr<float>(),
                reinterpret_cast<F*>(r.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(w.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(k.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(v.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(a.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(b.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(y.data_ptr<scalar_t>()),
                s.data_ptr<float>(),
                sa.data_ptr<float>()
            );
        }
    );
}


void rwkv7_state_clampw_backward(
    at::Tensor& r, at::Tensor& w, at::Tensor& k, at::Tensor& v, at::Tensor& a, at::Tensor& b,
    at::Tensor& dy, at::Tensor& s, at::Tensor& sa, at::Tensor& ds0,
    at::Tensor& dr, at::Tensor& dw, at::Tensor& dk, at::Tensor& dv, at::Tensor& da, at::Tensor& db
) {
    int batch_size = r.size(0);
    int seq_len = r.size(1);
    int num_heads = r.size(2);
    int head_size = r.size(3);
    int chunk_len = 16;

    assert(seq_len % chunk_len == 0);

    DISPATCH_FLOATING_TYPES(
        r.scalar_type(),
        "rwkv7_state_clampw_backward",
        [&] {
            using F = typename CudaType<scalar_t>::type;
            rwkv7_state_clampw_cuda_backward<F>(
                batch_size, seq_len, num_heads, head_size, chunk_len,
                reinterpret_cast<F*>(r.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(w.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(k.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(v.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(a.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(b.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(dy.data_ptr<scalar_t>()),
                s.data_ptr<float>(),
                sa.data_ptr<float>(),
                ds0.data_ptr<float>(),
                reinterpret_cast<F*>(dr.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(dw.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(dk.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(dv.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(da.data_ptr<scalar_t>()),
                reinterpret_cast<F*>(db.data_ptr<scalar_t>())
            );
        }
    );
}

TORCH_LIBRARY_IMPL(rwkv_cuda, CUDA, m) {
    m.impl("rwkv7_state_clampw_forward", &rwkv7_state_clampw_forward);
    m.impl("rwkv7_state_clampw_backward", &rwkv7_state_clampw_backward);
}
