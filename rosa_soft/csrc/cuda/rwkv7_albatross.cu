#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <cuda_fp16.h>
#include <cassert>
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

__device__ __forceinline__ float to_float(float u) { return u; }
__device__ __forceinline__ float to_float(__half u) { return __half2float(u); }
// __device__ __forceinline__ float to_float(__nv_bfloat16 u) { return __bfloat162float(u); }

template<typename T> __device__ __forceinline__ T from_float(float u);
template<> __device__ __forceinline__ float from_float<float>(float u) { return u; }
template<> __device__ __forceinline__ __half from_float<__half>(float u) { return __float2half_rn(u); }
// template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float u) { return __float2bfloat16_rn(u); }


constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f, nlog2_e = -1.4426950408889634f;
constexpr int ro1 = (int)2654435769, ro2 = (int)1779033704, ro3 = (int)3144134277;

#define rotator(_A, _B, _C) (two_to_neg_41 * float(ro1 * (_A) + ro2 * (_B) + ro3 * (_C)))
#define rotator1(_A) (two_to_neg_41 * float(ro1 * (_A)))


template<typename F, int HEAD_SIZE>
__global__ void rwkv7_albatross_kernel_forward_w0_fp16_dither_seq(
    const int B, const int T, const int C, const int H,
    F* __restrict__ _state, const F* __restrict__ const _r, const F* __restrict__ const _w, const F* __restrict__ const _k, const F* __restrict__ const _v, const F* __restrict__ const _a, const F* __restrict__ const _b,
    F* __restrict__ const _y, const int* __restrict__ const _elapsed_t
) {
    const int _N_ = HEAD_SIZE;

    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);

    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            state_smem[row][(row % 32) ^ col] = ((__half2 *)&state_vec)[j1];
        }
    }

    __syncthreads();
    
    __half2 state[_N_ / 2];
    
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        state[j] = state_smem[i][(i % 32) ^ j];
    }
    
    __shared__ __half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    for (int _t = 0; _t < T; _t++)
    {
        const int t = bbb * T * C + h * _N_ + i + _t * C;
        __syncthreads();
        
        ((F *)w)[i] = from_float<F>(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * to_float(_w[t])))) - 1.0f + rotator1(_elapsed_t[bbb] + _t));
        ((F *)k)[i] = _k[t];
        ((F *)a)[i] = _a[t];
        ((F *)b)[i] = _b[t];
        ((F *)r)[i] = _r[t];
        __syncthreads();
        
        __half2 sa2 = {0., 0.};
        
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
        {
            sa2 += a[j] * state[j];
        }

        __half sa = sa2.x + sa2.y;
        sa2 = {sa, sa};

        __half vv = _v[t];
        __half2 vv2 = {vv, vv};
        __half2 y2 = {0., 0.};
        
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
        {
            __half2 &s = state[j];
            s += s * w[j] + k[j] * vv2 + sa2 * b[j];
            y2 += s * r[j];
        }
        
        _y[t] = y2.x + y2.y;
    }
    
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        state_smem[i][(i % 32) ^ j] = state[j];
    }

    __syncthreads();
    
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((__half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}

template<typename F, int HEAD_SIZE>
__global__ void rwkv7_albatross_kernel_forward_w0_fp16_dither_one(
    const int B, const int C, const int H,
    F* __restrict__ _state, const F* __restrict__ const _r, const F* __restrict__ const _w, const F* __restrict__ const _k, const F* __restrict__ const _v, const F* __restrict__ const _a, const F* __restrict__ const _b,
    F* __restrict__ const _y, const int* __restrict__ const _elapsed_t
) {
    const int _N_ = HEAD_SIZE;

    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);

    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            state_smem[row][(row % 32) ^ col] = ((__half2 *)&state_vec)[j1];
        }
    }

    __syncthreads();
    
    __half2 state[_N_ / 2];
    
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        state[j] = state_smem[i][(i % 32) ^ j];
    }
    
    __shared__ __half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    const int t = bbb * C + h * _N_ + i;

    ((F *)w)[i] = from_float<F>(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * to_float(_w[t])))) - 1.0f + rotator1(_elapsed_t[bbb]));
    ((F *)k)[i] = _k[t];
    ((F *)a)[i] = _a[t];
    ((F *)b)[i] = _b[t];
    ((F *)r)[i] = _r[t];
    
    __syncthreads();
    
    __half2 sa2 = {0., 0.};

    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        sa2 += a[j] * state[j];
    }

    __half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};

    __half vv = _v[t];
    __half2 vv2 = {vv, vv};
    __half2 y2 = {0., 0.};

    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        __half2 &s = state[j];
        s += s * w[j] + k[j] * vv2 + sa2 * b[j];
        y2 += s * r[j];
    }

    _y[t] = y2.x + y2.y;


    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        state_smem[i][(i % 32) ^ j] = state[j];
    }

    __syncthreads();

    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((__half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}


void rwkv7_albatross_cuda_forward_w0_fp16_dither_seq(int B, int T, int C, int H, __half* s0, __half* r, __half* w, __half* k, __half* v, __half* a, __half* b, __half* y, int* elapsed_t)
{
    const int head_size = C / H;
    DISPATCH_HEAD_SIZE(head_size, [&] {
        assert(H * HEAD_SIZE == C);
        rwkv7_albatross_kernel_forward_w0_fp16_dither_seq<__half, HEAD_SIZE><<<B * H, HEAD_SIZE>>>(B, T, C, H, s0, r, w, k, v, a, b, y, elapsed_t);
    });
}


void rwkv7_albatross_cuda_forward_w0_fp16_dither_one(int B, int C, int H, __half* s0, __half* r, __half* w, __half* k, __half* v, __half* a, __half* b, __half* y, int* elapsed_t)
{
    const int head_size = C / H;
    DISPATCH_HEAD_SIZE(head_size, [&] {
        assert(H * HEAD_SIZE == C);
        rwkv7_albatross_kernel_forward_w0_fp16_dither_one<__half, HEAD_SIZE><<<B * H, HEAD_SIZE>>>(B, C, H, s0, r, w, k, v, a, b, y, elapsed_t);
    });
}
