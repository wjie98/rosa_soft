#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

#define DISPATCH_HEAD_SIZE(H_VAL, ...) \
    do { \
        switch (H_VAL) { \
            case 2:  { const int HEAD_SIZE = 2;  (__VA_ARGS__)(); break; } \
            case 4:  { const int HEAD_SIZE = 4;  (__VA_ARGS__)(); break; } \
            case 8:  { const int HEAD_SIZE = 8;  (__VA_ARGS__)(); break; } \
            case 16: { const int HEAD_SIZE = 16; (__VA_ARGS__)(); break; } \
            case 32: { const int HEAD_SIZE = 32; (__VA_ARGS__)(); break; } \
            case 64: { const int HEAD_SIZE = 64; (__VA_ARGS__)(); break; } \
            default: throw std::runtime_error("Unsupported HEAD_SIZE"); \
        } \
    } while (false)

#define DISPATCH_CHUNK_LEN(C_VAL, ...) \
    do { \
        switch (C_VAL) { \
            case 16: { const int CHUNK_LEN = 16; (__VA_ARGS__)(); break; } \
            default: throw std::runtime_error("Unsupported CHUNK_LEN"); \
        } \
    } while (false)

__device__ __forceinline__ float to_float(float u) { return u; }
__device__ __forceinline__ float to_float(__half u) { return __half2float(u); }
__device__ __forceinline__ float to_float(__nv_bfloat16 u) { return __bfloat162float(u); }

template<typename T> __device__ __forceinline__ T from_float(float u);
template<> __device__ __forceinline__ float from_float<float>(float u) { return u; }
template<> __device__ __forceinline__ __half from_float<__half>(float u) { return __float2half_rn(u); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float u) { return __float2bfloat16_rn(u); }

using i64 = long long int;
constexpr float W_SCALE = -0.6065306597f; // -exp(-0.5)

//###################################################################################################### 

template<typename F, int HEAD_SIZE, int CHUNK_LEN> __launch_bounds__(HEAD_SIZE, 2)
__global__ void rwkv7_state_clampw_forward_kernel(
    int T, int H, float* __restrict__ s0_,
    F* __restrict__ r_, F* __restrict__ w_, F* __restrict__ k_, F* __restrict__ v_,
    F* __restrict__ a_, F* __restrict__ b_, F* __restrict__ y_,
    float* __restrict__ s__, float* __restrict__ sa_
) {
    const int N = HEAD_SIZE;
    const int _CHUNK_LEN_ = CHUNK_LEN;

    const int bb=blockIdx.y, hh=blockIdx.x, i=threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);
    s0_ += i64(bb*H+hh) * i64(N*N) + i64(i*N);
    float state[N];
#pragma unroll
    for (int j=0; j<N; ++j) {
        state[j] = s0_[j];
    }
    __shared__ float r[N];
    __shared__ float w[N];
    __shared__ float k[N];
    __shared__ float a[N];
    __shared__ float b[N];

    for (int t = 0; t < T; ++t)
    {
        const int idx = ((bb*T+t)*H+hh)*N+i;

        __syncthreads();
        r[i] = to_float(r_[idx]);
        w[i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[idx]))));
        k[i] = to_float(k_[idx]);
        a[i] = to_float(a_[idx]);
        b[i] = to_float(b_[idx]);
        __syncthreads();

        float sa = 0.0f;
#pragma unroll
        for (int j=0; j<N; ++j) {
            sa += state[j] * a[j];
        }
        sa_[idx] = sa;

        float vi = to_float(v_[idx]);
        float y=0.0f;
#pragma unroll
        for (int j=0; j<N; ++j) {
            float s = state[j];
            s = s * w[j] + (sa * b[j] + k[j] * vi);
            y += s * r[j];
            state[j] = s;
        }

        y_[idx] = from_float<F>(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (t/_CHUNK_LEN_)*N*N + i;
#pragma unroll
            for (int j=0; j<N; ++j) {
                s_[base+j*N] = state[j];
            }
        }
    }
}

template<typename F>
void rwkv7_state_clampw_cuda_forward(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, float* s0, F* r, F* w, F* k, F* v, F* a, F* b, F* y, float* s, float* sa)
{
    DISPATCH_HEAD_SIZE(head_size, [&] {
        DISPATCH_CHUNK_LEN(chunk_len, [&] {
            rwkv7_state_clampw_forward_kernel<F, HEAD_SIZE, CHUNK_LEN><<<dim3(num_heads, batch_size), dim3(HEAD_SIZE)>>>(seq_len, num_heads, s0, r, w, k, v, a, b, y, s, sa);
        });
    });
}

//###################################################################################################### 

template<typename F, int HEAD_SIZE, int CHUNK_LEN>
__global__ void rwkv7_state_clampw_backward_kernel(
    int T, int H,
    F* __restrict__ r_, F* __restrict__ w_, F* __restrict__ k_, F* __restrict__ v_,
    F* __restrict__ a_, F* __restrict__ b_, F* __restrict__ dy_, float* __restrict__ s__, float* __restrict__ sa_, float* __restrict__ ds0_,
    F* __restrict__ dr_, F* __restrict__ dw_, F* __restrict__ dk_, F* __restrict__ dv_, F* __restrict__ da_, F* __restrict__ db_
) {
    const int N = HEAD_SIZE;
    const int _CHUNK_LEN_ = CHUNK_LEN;

    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);
    ds0_ += i64(bb*H+hh) * i64(N*N) + i64(i*N);

    float stateT[N] = {0}, dstate[N] = {0}, dstateT[N] = {0};
    __shared__ float r[N], w[N], k[N], v[N], a[N], b[N], dy[N], sa[N], dSb_shared[N];
    float ri, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--)
    {
        int idx = bb*T*H*N + t*H*N + hh * N + i;

        __syncthreads();
        r[i] = ri = to_float(r_[idx]);
        float w_sig = 1.0f / (1.0f + __expf(-to_float(w_[idx])));
        w[i] = wi = __expf(W_SCALE * w_sig);
        k[i] = ki = to_float(k_[idx]);
        v[i] = to_float(v_[idx]);
        a[i] = ai = to_float(a_[idx]);
        b[i] = bi = to_float(b_[idx]);
        dy[i] = dyi = to_float(dy_[idx]);
        sa[i] = sa_[idx];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (t/_CHUNK_LEN_)*N*N + i*N;
            const float4* s4 = (const float4*)(s_ + base);
#pragma unroll
            for (int j4 = 0; j4 < N/4; j4++) {
                float4 q = s4[j4];
                const int j = j4<<2;
                stateT[j+0] = q.x;
                stateT[j+1] = q.y;
                stateT[j+2] = q.z;
                stateT[j+3] = q.w;
            }
        }

        float dr = 0;
#pragma unroll
        for (int j = 0; j < N; j++) {
            dr += stateT[j] * dy[j];
        }
        dr_[idx] = from_float<F>(dr);

        float iwi = 1.0f / wi;
#pragma unroll
        for (int j = 0; j < N; j++) {
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * r[j];
            dstateT[j] += ri * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < N; j++) {
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * v[j];
            dv += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            db += dstateT[j] * sa[j];
        }
        dw_[idx] = from_float<F>(W_SCALE * dw * wi * w_sig * (1.0f - w_sig));

        dk_[idx] = from_float<F>(dk);
        dv_[idx] = from_float<F>(dv);
        db_[idx] = from_float<F>(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < N; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[idx] = from_float<F>(da);

#pragma unroll
        for (int j = 0; j < N; j++) {
            dstate[j] = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
        }
    }
#pragma unroll    
    for (int j = 0; j < N; j++) {
        ds0_[j] = dstate[j];
    }
}

template<typename F>
void rwkv7_state_clampw_cuda_backward(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, F* r, F* w, F* k, F* v, F* a, F* b, F* dy, float* s, float* sa, float* ds0, F* dr, F* dw, F* dk, F* dv, F* da, F* db)
{
    DISPATCH_HEAD_SIZE(head_size, [&] {
        DISPATCH_CHUNK_LEN(chunk_len, [&] {
            rwkv7_state_clampw_backward_kernel<F, HEAD_SIZE, CHUNK_LEN><<<dim3(num_heads, batch_size), dim3(HEAD_SIZE)>>>(seq_len, num_heads, r, w, k, v, a, b, dy, s, sa, ds0, dr, dw, dk, dv, da, db);
        });
    });
}


//###################################################################################################### 

template void rwkv7_state_clampw_cuda_forward<float>(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, float* s0, float* r, float* w, float* k, float* v, float* a, float* b, float* y, float* s, float* sa);
template void rwkv7_state_clampw_cuda_forward<__half>(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, float* s0, __half* r, __half* w, __half* k, __half* v, __half* a, __half* b, __half* y, float* s, float* sa);
template void rwkv7_state_clampw_cuda_forward<__nv_bfloat16>(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, float* s0, __nv_bfloat16* r, __nv_bfloat16* w, __nv_bfloat16* k, __nv_bfloat16* v, __nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* y, float* s, float* sa);

template void rwkv7_state_clampw_cuda_backward<float>(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, float* r, float* w, float* k, float* v, float* a, float* b, float* dy, float* s, float* sa, float* ds0, float* dr, float* dw, float* dk, float* dv, float* da, float* db);
template void rwkv7_state_clampw_cuda_backward<__half>(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, __half* r, __half* w, __half* k, __half* v, __half* a, __half* b, __half* dy, float* s, float* sa, float* ds0, __half* dr, __half* dw, __half* dk, __half* dv, __half* da, __half* db);
template void rwkv7_state_clampw_cuda_backward<__nv_bfloat16>(int batch_size, int seq_len, int num_heads, int head_size, int chunk_len, __nv_bfloat16* r, __nv_bfloat16* w, __nv_bfloat16* k, __nv_bfloat16* v, __nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* dy, float* s, float* sa, float* ds0, __nv_bfloat16* dr, __nv_bfloat16* dw, __nv_bfloat16* dk, __nv_bfloat16* dv, __nv_bfloat16* da, __nv_bfloat16* db);

