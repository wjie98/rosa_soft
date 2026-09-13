#include "bitflip.cuh"
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>

namespace rosa::bitflip {
namespace {
constexpr U position_mask = (1u << 20) - 1;
constexpr int table_width = 480;

__device__ int3 unpack(U x) {
  return make_int3(x & position_mask, (x >> 20) & position_mask, x >> 40);
}
__device__ U pack(int3 x) {
  return U(x.x) | (U(x.y) << 20) | (U(x.z) << 40);
}

// Union of the original winner's key and query coverage, in owner order.
struct Domain {
  int a, b, c, n;
  __device__ int lower(int p) const {
    return min(max(p - a, 0), b - a) + min(max(p - c, 0), n - b + a);
  }
  __device__ int index(int p) const {
    if (p >= a && p < b) return p - a;
    if (p >= c && p < c + n - b + a) return b - a + p - c;
    return -1;
  }
  __device__ int at(int p) const { return p < b - a ? a + p : c + p - b + a; }
};
template <class P> struct Summary { P a, b, c, e; };
template <class P> struct Maximum {
  __device__ Summary<P> operator()(Summary<P> x, Summary<P> y) const {
    return {max(x.a,y.a), max(x.b,y.b), max(x.c,y.c), max(x.e,y.e)};
  }
};
template <class P> struct Prefix {
  Summary<P> value{0,0,0,0};
  __device__ Summary<P> operator()(Summary<P> x) {
    auto old = value;
    value = Maximum<P>{}(value,x);
    return old;
  }
};
template <class P> __device__ P compact(int n, int j) {
  return n > 0 ? (P(n) << (sizeof(P)*4)) | P(j+1) : 0;
}
template <class P> __device__ unsigned endpoint(P x) {
  if constexpr (sizeof(P)==4) return x & 65535u;
  else return unsigned(x);
}

__global__ void scan(const unsigned* q, U* state, U* meta, int t, int rows,
                     int start, int count) {
  int delta = blockIdx.x*blockDim.x+threadIdx.x+1, s = blockIdx.y;
  if (delta >= t) return;
  q += int64_t(s)*t;
  int3 m = start ? unpack(state[int64_t(s)*t+delta])
                 : make_int3(delta-1,delta-1,delta-1);
  for (int r=0; r<count; ++r) {
    int i=start+r, j=i-delta;
    if (j>=0) {
      if (q[i]!=q[j]) m=make_int3(i,m.x,m.y);
      meta[(int64_t(s)*rows+r)*t+j]=pack(m);
    }
  }
  state[int64_t(s)*t+delta]=pack(m);
}

template <class P>
__global__ void row(const unsigned* q, const U* meta, const U* route,
                    const unsigned* v, const float* dy, P* summary, U* repairs,
                    unsigned* marks, int* lists, float* grad, int t, int d,
                    int dv, int rows, int start, int count, int shift,
                    int repeat, bool all) {
  int s=blockIdx.x/count, r=blockIdx.x%count, i=start+r, tid=threadIdx.x;
  if (!i) return;
  int64_t slot=int64_t(s)*rows+r;
  U old=route[int64_t(s)*t+i];
  unsigned from=unsigned(old), epoch=start/rows+1;
  int length=old>>32, end=int(from)-1;
  Domain dom{end-length+1,end+1,max(end+1,i-length+1),length+min(length,i-end)};
  int payload=all && from && dom.index(from)<0 ? int(from) : -1;
  bool planar=dom.n>=32;
  P* a=summary+slot*4*t;
  P* b=a+t; P* c=b+t; P* e=c+t;
  U* repair=repairs+slot*t*d;
  unsigned* mark=marks+slot*t;
  int* list=lists+slot*t;
  using Scan=cub::BlockScan<Summary<P>,256,cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ int used;
  __shared__ unsigned active;
  if (!tid) { used=0; active=0; }
  for (int p=tid; p<dom.n; p+=blockDim.x) a[p]=b[p]=c[p]=e[p]=0;
  q+=int64_t(s)*t;
  dy+=(int64_t(s)*t+i)*dv;
  grad+=int64_t(s)*t*d;
  int words=(dv+31)/32, width=min(dv,table_width);
  v+=int64_t(s/repeat)*t*words;
  extern __shared__ float table[];
  for (int p=tid; p<((width+3)/4)*17; p+=blockDim.x) {
    int group=p/17, code=p%17;
    unsigned previous=from ? v[int64_t(from)*words+group/8]>>((group%8)*4) : 0;
    float sum=0;
    for (int z=0; z<4 && group*4+z<width; ++z) {
      int delta=(code<16 ? ((code>>z)&1 ? 1 : -1) : 0)
                -(from ? ((previous>>z)&1 ? 1 : -1) : 0);
      if (delta) sum+=delta*dy[group*4+z];
    }
    table[p]=sum;
  }
  auto credit=[=] __device__(unsigned to, int flip) {
    float sum=0;
    for (int w=0; w<words; ++w) {
      unsigned x=to ? v[int64_t(to)*words+w] : 0;
      if (to && flip>=0 && flip/32==w) x^=1u<<(flip%32);
      if (w<table_width/32) {
#pragma unroll
        for (int z=0; z<8; ++z)
          if ((w*8+z)*4<dv)
            sum+=table[(w*8+z)*17+(to ? ((x>>(z*4))&15) : 16)];
        continue;
      }
      unsigned y=from ? v[int64_t(from)*words+w] : 0;
      unsigned bits=to && from ? x^y : 0xffffffffu>>(32-min(32,dv-w*32));
      while (bits) {
        int z=__ffs(bits)-1;
        int delta=(to ? ((x>>z)&1 ? 1 : -1) : 0)-(from ? ((y>>z)&1 ? 1 : -1) : 0);
        sum+=delta*dy[w*32+z];
        bits&=bits-1;
      }
    }
    return sum;
  };
  __syncthreads();
  auto emit=[=] __device__(P* field, int key, P value) {
    int at=dom.lower(key);
    if (at<dom.n) atomicMax(field+at,value);
  };
  int* used_ptr=&used;
  unsigned seen=0;
  auto claim=[=,&seen] __device__(int p, int bit, U next) {
    bool outside=dom.index(p)<0;
    if (!next || (outside && next<=old)) return;
    seen|=1u<<bit;
    U word=(U(epoch)<<(2*shift))|((next>>32)<<shift)|unsigned(next);
    atomicMax(repair+(planar ? int64_t(bit)*t+p : int64_t(p)*d+bit),word);
    if (outside && p!=payload && atomicExch(mark+p,epoch)!=epoch)
      list[atomicAdd(used_ptr,1)]=p;
  };
  for (int j=tid; j<i; j+=blockDim.x) {
    int delta=i-j;
    int3 m=unpack(meta[slot*t+j]);
    int n=i-m.x;
    if (n && dom.n) {
      int at=dom.lower(j-n+1);
      if (at) atomicMax(a+dom.n-at,compact<P>(n,j));
      emit(b,j-n+1,P(j+1));
      emit(c,j+1,compact<P>(n,j));
      emit(e,max(j+1,i-n),P(j+1));
    }
    if (m.x>=delta) {
      unsigned x=q[m.x], diff=x^q[m.x-delta];
      if (diff && !(diff&(diff-1))) {
        int bit=__ffs(diff)-1, p=m.x-delta;
        int left=max(m.y,p), right=m.x+delta<=i ? m.x+delta : m.y;
        if (p==m.y && p>=delta && q[p-delta]==x) left=m.z;
        claim(m.x,bit,priority(i-right,j));
        claim(p,bit,priority(i-left,j));
      }
    }
  }
  for (int step=16; step; step/=2) seen|=__shfl_xor_sync(0xffffffffu,seen,step);
  if (!(tid&31) && seen) atomicOr(&active,seen);
  __syncthreads();
  Prefix<P> prefix;
  for (int start=0; start<dom.n; start+=blockDim.x) {
    int p=start+tid;
    Summary<P> item=p<dom.n ? Summary<P>{a[p],b[p],c[p],e[p]} : Summary<P>{0,0,0,0};
    Summary<P> out;
    Scan(temp).InclusiveScan(item,out,Maximum<P>{},prefix);
    if (p<dom.n) { a[p]=out.a; b[p]=out.b; c[p]=out.c; e[p]=out.e; }
    __syncthreads();
  }
  int total=dom.n+(payload>=0)+used;
  for (int pos=tid; pos<total; pos+=blockDim.x) {
    int p=pos<dom.n ? dom.at(pos) : (payload>=0 && pos==dom.n ? payload : list[pos-dom.n-(payload>=0)]);
    int at=dom.index(p), cap=i-p;
    P base=compact<P>(length,end);
    if (at>=0) {
      base=a[dom.n-1-at];
      base=max(base,compact<P>(int(b[at])-1-p,int(b[at])-1));
      P original=c[at];
      base=max(base,compact<P>(min(int(original>>(sizeof(P)*4)),cap),int(endpoint(original))-1));
      if (e[at]) base=max(base,compact<P>(cap,int(e[at])-1));
    }
    unsigned base_to=endpoint(base);
    float shared=0;
    if (base_to!=from && !(all && base_to && base_to==unsigned(p))) shared=credit(base_to,-1);
    auto winner=[=] __device__(int bit) {
      if (!(active&(1u<<bit))) return base;
      U word=repair[planar ? int64_t(bit)*t+p : int64_t(p)*d+bit];
      U mask=(1ull<<shift)-1;
      P next=(word>>(2*shift))==epoch ? compact<P>((word>>shift)&mask,int(word&mask)-1) : 0;
      return max(base,next);
    };
    for (int bit=0; bit<d; ++bit) {
      P next=winner(bit);
      unsigned to=endpoint(next);
      if (to==from && !(all && to && to==unsigned(p))) continue;
      float delta=to==base_to && !(all && to && to==unsigned(p)) ? shared
                  : credit(to,all && to && to==unsigned(p) ? bit : -1);
      if (delta) atomicAdd(grad+int64_t(bit)*t+p,delta);
    }
  }
}
}  // namespace

Tensor joint_credit(const Tensor& q, const Tensor& v, const Tensor& dy,
                    const Tensor& route, int d, int rows, bool all) {
  int s=q.size(0), t=q.size(1), dv=dy.size(2);
  c10::cuda::CUDAGuard guard(q.device());
  auto grad=torch::zeros({s,d,t},dy.options());
  if (t<=1) return grad;
  at::globalContext().alertNotDeterministic("rosa_bitflip joint backward");
  auto stream=at::cuda::getCurrentCUDAStream();
  rows=std::min(rows,t);
  auto state=torch::empty({s,t},route.options());
  auto meta=torch::empty({s,rows,t},route.options());
  bool narrow=t<65536;
  auto summary=torch::empty({s,rows,4,t},narrow ? q.options() : route.options());
  auto repair=torch::zeros({s,rows,t,d},route.options());
  auto mark=torch::zeros({s,rows,t},q.options());
  auto list=torch::empty_like(mark);
  int shift=32-__builtin_clz(unsigned(t));
  for (int start=0; start<t; start+=rows) {
    int count=std::min(rows,t-start);
    scan<<<dim3((t+254)/256,s),256,0,stream>>>(reinterpret_cast<const unsigned*>(q.data_ptr<int>()),
        reinterpret_cast<U*>(state.data_ptr<int64_t>()),reinterpret_cast<U*>(meta.data_ptr<int64_t>()),
        t,rows,start,count);
    int bytes=((std::min(dv,table_width)+3)/4)*17*sizeof(float);
    auto launch=[&](auto tag) {
      using P=decltype(tag);
      row<P><<<s*count,256,bytes,stream>>>(reinterpret_cast<const unsigned*>(q.data_ptr<int>()),
          reinterpret_cast<const U*>(meta.data_ptr<int64_t>()),reinterpret_cast<const U*>(route.data_ptr<int64_t>()),
          reinterpret_cast<const unsigned*>(v.data_ptr<int>()),dy.data_ptr<float>(),static_cast<P*>(summary.data_ptr()),
          reinterpret_cast<U*>(repair.data_ptr<int64_t>()),reinterpret_cast<unsigned*>(mark.data_ptr<int>()),
          list.data_ptr<int>(),grad.data_ptr<float>(),t,d,dv,rows,start,count,shift,s/v.size(0),all);
    };
    if (narrow) launch(unsigned(0)); else launch(U(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return grad;
}
}  // namespace rosa::bitflip
