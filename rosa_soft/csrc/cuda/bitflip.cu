#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <type_traits>

#include "bitflip.cuh"

namespace rosa::bitflip {
namespace {
constexpr int credit_width = 480;
template <bool Narrow>
__global__ void dp(const int* q, const int* k, const int* in, int* out, U* meta,
                   int* ends, int* counts, int t, int rows, int start, int count,
                   int tiles) {
  int a = blockIdx.x / tiles, block = blockIdx.x % tiles, warp = threadIdx.x / 32,
      lane = threadIdx.x & 31;
  int delta = block * 8 + warp + 1, j0 = start - 1 - delta;
  int64_t offset = static_cast<int64_t>(a) * 3 * t;
  int last = j0 >= 0 ? j0 - in[offset + j0] : -1;
  int b = j0 >= 0 ? in[offset + 2 * t + j0] : -1;
  int second = b >= 0 ? j0 - in[offset + t + j0] : -1;
  __shared__ int tile[3][8][36];
  for (int x = 0; x < count; x += 32) {
    int row = x + lane, i = start + row, j = i - delta;
    bool valid = row < count && j >= 0;
    unsigned diff = valid ? static_cast<unsigned>(q[a * t + i] ^ k[a * t + j]) : 0;
    int own = diff && !(diff & (diff - 1)) ? __ffs(diff) - 1 : -1;
    unsigned mask = __ballot_sync(0xffffffffu, valid && diff);
    unsigned prefix = mask & (0xffffffffu >> (31 - lane));
    int high = prefix ? 31 - __clz(prefix) : 0;
    unsigned rest = prefix ? prefix & ~(1u << high) : 0;
    int first = prefix ? start + x + high - delta : last;
    int prev = rest ? start + x + 31 - __clz(rest) - delta : (prefix ? last : second);
    int tag = __shfl_sync(0xffffffffu, own, high);
    if (!prefix)
      tag = b;
    tile[0][warp][lane] = j - first;
    tile[1][warp][lane] = tag >= 0 ? j - prev : 0;
    tile[2][warp][lane] = tag;
    last = __shfl_sync(0xffffffffu, first, 31);
    second = __shfl_sync(0xffffffffu, prev, 31);
    b = __shfl_sync(0xffffffffu, tag, 31);
    __syncthreads();
    int y = threadIdx.x / 8, z = threadIdx.x % 8, rr = x + y;
    int jj = start + rr - (block * 8 + z + 1);
    int n = tile[0][z][y], r = tile[1][z][y], bit = tile[2][z][y];
    bool keep = rr < count && jj >= 0;
    int pos = jj;
    {
      keep = keep && (n > 0 || bit >= 0);
      unsigned all = __ballot_sync(0xffffffffu, keep),
               group = (all >> ((lane / 8) * 8)) & 255;
      int leader = (lane / 8) * 8 + (group ? __ffs(group) - 1 : 0), at = 0;
      if (group && lane == leader)
        at = atomicAdd(counts + a * rows + rr, __popc(group));
      at = __shfl_sync(0xffffffffu, at, leader);
      pos = at + __popc(group & ((1u << z) - 1));
    }
    int64_t row0 = static_cast<int64_t>(a) * rows + rr;
    if (keep) {
      if constexpr (Narrow)
        meta[row0 * t + pos] = (static_cast<U>(bit + 1) << 48) |
                              (static_cast<U>(n) << 32) | (static_cast<U>(r) << 16) | jj;
      else {
        meta[row0 * t + pos] = record(n, r, bit);
        ends[row0 * t + pos] = jj;
      }
    }
    if (rr == count - 1 && jj >= 0) {
      out[offset + jj] = n;
      out[offset + t + jj] = r;
      out[offset + 2 * t + jj] = bit;
    }
    __syncthreads();
  }
}

template <bool Narrow>
struct Record {
  const U* data;
  const int* ends;
  int64_t row;
  int t;
  __device__ U word(int p) const { return data[row * t + p]; }
  __device__ int n(int p) const {
    if constexpr (Narrow) return (word(p) >> 32) & 65535;
    return word(p) >> 34;
  }
  __device__ int r(int p) const {
    if constexpr (Narrow) return (word(p) >> 16) & 65535;
    return (word(p) >> 6) & ((1u << 28) - 1);
  }
  __device__ int bit(int p) const {
    if constexpr (Narrow) return static_cast<int>(word(p) >> 48) - 1;
    return static_cast<int>(word(p) & 63) - 1;
  }
  __device__ int j(int p) const {
    if constexpr (Narrow) return word(p) & 65535;
    return ends[row * t + p];
  }
};

template <class P>
struct Summary {
  P q_clip_end, k_left, k_clip_end, k_right;
};
template <class P>
struct Maximum {
  __device__ Summary<P> operator()(Summary<P> x, Summary<P> y) const {
    return {max(x.q_clip_end, y.q_clip_end), max(x.k_left, y.k_left),
            max(x.k_clip_end, y.k_clip_end), max(x.k_right, y.k_right)};
  }
};
template <class P>
struct Prefix {
  Summary<P> sum{0, 0, 0, 0};
  __device__ Summary<P> operator()(Summary<P> x) {
    auto old = sum;
    sum = Maximum<P>{}(sum, x);
    return old;
  }
};
template <class P>
struct ScalarPrefix {
  P sum = 0;
  __device__ P operator()(P x) {
    P old = sum;
    sum = max(sum, x);
    return old;
  }
};
template <class P>
struct RowSummary {
  P* data;
  int stride, origin[4], shift;
  bool latest = false;
  __device__ P pack(U x) const {
    if constexpr (sizeof(P) == 8)
      return x;
    return static_cast<P>(((x >> 32) << shift) | static_cast<unsigned>(x));
  }
  __device__ U wide(P x) const {
    if constexpr (sizeof(P) == 8)
      return x;
    return (static_cast<U>(x >> shift) << 32) | (x & ((1u << shift) - 1));
  }
  __device__ P* field(int p, int f) const { return data + f * stride + p; }
  __device__ P get(int p, int f) const { return *field(p - origin[f], f); }
  __device__ Summary<P> load(int p) const {
    return {*field(p, 0), *field(p, 1), *field(p, 2), *field(p, 3)};
  }
  __device__ void store(int p, Summary<P> x) const {
    *field(p, 0) = x.q_clip_end;
    *field(p, 1) = x.k_left;
    *field(p, 2) = x.k_clip_end;
    *field(p, 3) = x.k_right;
  }
};

template <class P>
__device__ U baseline(RowSummary<P> s, int i, int p, int side, U old) {
  int length = old >> 32, v_idx = static_cast<unsigned>(old);
  if (s.latest) {
    if (!side) return priority(min(length, i - p), v_idx - 1);
    if (p < v_idx - length || p >= v_idx) return old;
    return max(s.wide(s.get(p, 1)), priority(v_idx - 1 - p, v_idx - 1));
  }
  if (!side) {
    if (p == i)
      return 0;
    if (v_idx == i)
      return priority(min(length, i - p), v_idx - 1);
    if (p < i - length + 1)
      return old;
    int j = static_cast<int>(s.get(p, 0));
    return j ? priority(i - p, j - 1) : old;
  }
  if (p < v_idx - length || p >= v_idx)
    return old;
  int j = static_cast<int>(s.get(p, 2));
  return max(max(s.wide(s.get(p, 1)), s.wide(s.get(i - p - 1, 3))),
             priority(j - 1 - p, j - 1));
}

// C: direct, full LUT, bounded LUT with exact tail, or complete-group full LUT.
template <class M, class P, int N, int C>
__global__ __launch_bounds__(N, N == 256 && sizeof(P) == 4 ? 3 : 1)
void row(U* meta, int* ends, const int* counts,
                    const U* original, P* summaries, U* work, void* events, M* excluded,
                    float* dq, float* dk, float* common, int t, int d, int rows,
                    int start, int count, const void* upstream, const unsigned* packed,
                    int dv, bool half, U epoch, int repeat, const int* q, const int* k) {
  int a = blockIdx.x / count, rr = blockIdx.x % count, i = start + rr,
      thread = threadIdx.x;
  int64_t row = static_cast<int64_t>(a) * rows + rr;
  int records = counts[a * rows + rr];
  U old = original[a * t + i];
  int length = old >> 32, v_idx = static_cast<unsigned>(old);
  // Both launches use the same predicate; exactly one owns this row.
  if ((N == 32) != (records <= 512 && length <= 32))
    return;
  if (!records && !old)
    return;
  // A latest terminal match needs only the left prefix for destructive edits.
  bool latest = old && (length == 1 || v_idx == i);
  if (old && !latest && q[a * t + i] != k[a * t + i - 1]) {
    bool later = false;
    for (int p = v_idx + thread; p < i - 1; p += N)
      later |= q[a * t + i] == k[a * t + p];
    latest = !__syncthreads_or(later);
  }
  Record<sizeof(P) == 4> m{meta, ends, row, t};
  const unsigned* values =
      packed + static_cast<int64_t>(a / repeat) * t * ((dv + 31) / 32);
  extern __shared__ float lut[];
  auto upstream_value = [=] __device__(int z) {
    int64_t at = (static_cast<int64_t>(a) * t + i) * dv + z;
    return half ? static_cast<float>(static_cast<const at::Half*>(upstream)[at])
                : static_cast<const float*>(upstream)[at];
  };
  auto grad = [=] __device__(float* x, int p, int bit) {
    return x + (static_cast<int64_t>(a) * d + bit) * t + p;
  };
  auto credit = [=] __device__(int to) {
    if (to == v_idx)
      return 0.f;
    float sum = 0;
    int words = (dv + 31) / 32;
    for (int w = 0; w < words; ++w) {
      unsigned x = to ? values[to * words + w] : 0;
      if (C == 1 || C == 3 || (C == 2 && w < credit_width / 32)) {
#pragma unroll
        for (int z = 0; z < 8; ++z)
          if (C == 3 || (w * 8 + z) * 4 < dv)
            sum += lut[(w * 8 + z) * 17 + (to ? ((x >> (z * 4)) & 15) : 16)];
      } else {
        unsigned y = v_idx ? values[v_idx * words + w] : 0;
        unsigned bits =
            (to && v_idx) ? x ^ y : (0xffffffffu >> (32 - min(32, dv - w * 32)));
        while (bits) {
          int z = __ffs(bits) - 1;
          int delta = (to ? ((x >> z) & 1 ? 1 : -1) : 0) -
                      (v_idx ? ((y >> z) & 1 ? 1 : -1) : 0);
          sum += delta * upstream_value(w * 32 + z);
          bits &= bits - 1;
        }
      }
    }
    return sum;
  };
  if constexpr (C != 0) {
    // The original V is fixed for this row: subtract symbols before multiplying dY.
    int width = C == 2 ? min(dv, credit_width) : dv;
    for (int p = thread; p < ((width + 3) / 4) * 17; p += N) {
      int group = p / 17, code = p % 17;
      float sum = 0;
      unsigned old =
          v_idx ? (values[v_idx * ((dv + 31) / 32) + group / 8] >> ((group % 8) * 4))
                : 0;
      for (int z = 0; z < 4 && group * 4 + z < dv; ++z) {
        int delta = (code < 16 ? ((code >> z) & 1 ? 1 : -1) : 0) -
                    (v_idx ? ((old >> z) & 1 ? 1 : -1) : 0);
        if (delta)
          sum += delta * upstream_value(group * 4 + z);
      }
      lut[p] = sum;
    }
    __syncthreads();
  }
  using Scan = cub::BlockScan<Summary<P>, N>;
  using Fold = cub::BlockReduce<Summary<P>, N>;
  using Scan1 = cub::BlockScan<P, N>;
  using Fold1 = cub::BlockReduce<P, N>;
  // Capacities select storage only; longer matches use complete global buffers.
  constexpr int qcap = sizeof(P) == 4 && N == 256 ? 512 : 0;
  constexpr int list_size = N == 256 ? 256 : 0;
  struct Local {
    unsigned q[qcap ? qcap : 1];
    int list[list_size ? list_size : 1];
    unsigned used;
  };
  __shared__ union {
    typename Scan::TempStorage scan;
    typename Fold::TempStorage fold;
    typename Scan1::TempStorage scan1;
    typename Fold1::TempStorage fold1;
    Local local;
  } temp;
  int size = length, qsize = latest || v_idx == i ? min(length, 1) : length,
      shift = 32 - __clz(static_cast<unsigned>(t));
  P* data = summaries + row * (t + 1) * 4;
  int stride = t + 1;
  constexpr int capacity = sizeof(P) == 4 ? (N == 32 ? 32 : 512) : 0;
  __shared__ P cache[capacity ? 4 * capacity : 1];
  P* shared_data = cache;
  if (capacity && size <= capacity) {
    data = cache;
    stride = capacity;
  }
  RowSummary<P> s{data,
                  stride,
                  {i - qsize + 1, v_idx - size, v_idx - size, i - v_idx},
                  shift, latest};
  constexpr bool reuse = sizeof(P) == 4;
  U* w = work + row * (t + 1) * (reuse ? 1 : 2);
  M* changed = excluded + row * (t + 1) * 2;
  __shared__ int bucket[32], offset[33];
  __shared__ unsigned active;
  if (thread < 32)
    bucket[thread] = 0;
  for (int p = thread; p < size; p += N) {
    if (latest) *s.field(p, 1) = 0;
    else s.store(p, {0, 0, 0, 0});
  }
  __syncthreads();
  Summary<P> lower{0, 0, 0, 0};
  auto emit = [&] __device__(int bin, int f, P x) {
    if (latest && f != 1) return;
    if (bin <= s.origin[f]) {
      if (f == 0)
        lower.q_clip_end = max(lower.q_clip_end, x);
      if (f == 1)
        lower.k_left = max(lower.k_left, x);
      if (f == 2)
        lower.k_clip_end = max(lower.k_clip_end, x);
      if (f == 3)
        lower.k_right = max(lower.k_right, x);
    } else if (bin - s.origin[f] < (f ? size : qsize)) {
      auto at = s.field(bin - s.origin[f], f);
      if (f == 1)
        *at = x;
      else if constexpr (sizeof(P) == 4) {
        if (capacity && size <= capacity)
          atomicMax(shared_data + f * capacity + bin - s.origin[f], x);
        else {
          // Keep this global RED separate from the generic shared/global pointer.
          P* global = summaries + row * (t + 1) * 4 + f * (t + 1) + bin - s.origin[f];
          asm volatile("red.relaxed.gpu.global.max.u32 [%0], %1;"
                       : : "l"(global), "r"(x) : "memory");
        }
      }
      else
        atomicMax(at, x);
    }
  };
  U pending_record = N == 256 && thread < records ? m.word(thread) : 0;
  for (int x = 0; x < records; x += N) {
    int p = x + thread, j, n, b;
    if constexpr (N == 256) {
      U word = pending_record;
      pending_record = p + N < records ?
          *reinterpret_cast<const volatile U*>(m.data + m.row * t + p + N) : 0;
      if constexpr (sizeof(P) == 4) {
        j = p < records ? static_cast<int>(word & 65535) : -1;
        n = (word >> 32) & 65535;
        b = static_cast<int>(word >> 48) - 1;
      } else {
        j = p < records ? m.j(p) : -1;
        n = word >> 34;
        b = static_cast<int>(word & 63) - 1;
      }
    } else {
      j = p < records ? m.j(p) : -1;
      n = p < records ? m.n(p) : 0;
      b = p < records ? m.bit(p) : -1;
    }
    if (n) {
      int first = j - n + 1;
      P p0 = s.pack(priority(n, j));
      emit(i - n, 0, static_cast<P>(j + 1));
      emit(j + 1, 1, p0);
      emit(first, 2, static_cast<P>(j + 1));
      emit(i - first, 3, p0);
    }
    unsigned peers = __match_any_sync(0xffffffffu, b);
    if (b >= 0 && (thread & 31) == __ffs(peers) - 1)
      atomicAdd(bucket + b, __popc(peers));
  }
  __syncthreads();
  if (latest) {
    P folded = Fold1(temp.fold1).Reduce(lower.k_left, cub::Max());
    if (!thread && size) *s.field(0, 1) = folded;
  } else {
    auto folded = Fold(temp.fold).Reduce(lower, Maximum<P>{});
    if (!thread && size) s.store(0, folded);
  }
  __syncthreads();
  if (!thread) {
    // Counts become scatter cursors; offsets remain immutable bucket boundaries.
    offset[0] = 0;
    active = 0;
    for (int b = 0; b < d; ++b) {
      int n = bucket[b];
      offset[b + 1] = offset[b] + n;
      if (n)
        active |= 1u << b;
      bucket[b] = offset[b];
    }
  }
  __syncthreads();
  bool direct = reuse && active && !(active & (active - 1));
  bool qlocal = qcap && direct && length < qcap &&
                2 * offset[d] >= i + length + 1;
  if (active && !direct)
    for (int x = 0; x < records; x += N) {
      int p = x + thread, b = p < records ? m.bit(p) : -1;
      unsigned peers = __match_any_sync(0xffffffffu, b);
      if (b >= 0) {
        int first = __ffs(peers) - 1, at = 0, lane = thread & 31;
        if (lane == first)
          at = atomicAdd(bucket + b, __popc(peers));
        at = __shfl_sync(peers, at, first);
        int e = at + __popc(peers & ((1u << lane) - 1));
        if constexpr (sizeof(P) == 4)
          static_cast<U*>(events)[row * t + e] = m.word(p);
        else
          static_cast<int*>(events)[row * t + e] = p;
      }
    }
  __syncthreads();
  // A single active bit needs no scatter. Preserve its source and use events for Q winners.
  U* qw = reuse ? (direct ? static_cast<U*>(events) + row * t : meta + row * t) : w;
  U* kw = w + (reuse ? 0 : t + 1);
  auto winner = [=] __device__(int side, int p) {
    return side ? kw + p : qw + (reuse ? p - (i - length) : p);
  };
  const U* source = (direct ? meta : static_cast<U*>(events)) + row * t;
  auto event = [=] __device__(int e) {
    if constexpr (sizeof(P) == 4) {
      U x = source[e];
      return make_int3(x >> 48 ? static_cast<int>(x & 65535) : -1,
                       (x >> 32) & 65535, (x >> 16) & 65535);
    } else {
      int p = static_cast<int*>(events)[row * t + e];
      return make_int3(m.j(p), m.n(p), m.r(p));
    }
  };
  auto read_event = [=] __device__(int e, U word) {
    if constexpr (sizeof(P) == 4 && N == 256)
      return make_int3(word >> 48 ? static_cast<int>(word & 65535) : -1,
                       (word >> 32) & 65535, (word >> 16) & 65535);
    else
      return event(e);
  };
  auto prefetch_event = [=] __device__(int e, int right) {
    if constexpr (sizeof(P) == 4 && N == 256)
      return e < right ? *reinterpret_cast<const volatile U*>(source + e) : U(0);
    else
      return U(0);
  };
  if constexpr (reuse)
    if (active && !qlocal && !thread) qw[size] = 0;
  Prefix<P> prefix;
  ScalarPrefix<P> prefix1;
  for (int x = 0; x < size; x += N) {
    int p = x + thread;
    Summary<P> result;
    P result1;
    if (latest) {
      P item = p < size ? *s.field(p, 1) : 0;
      Scan1(temp.scan1).InclusiveScan(item, result1, cub::Max(), prefix1);
    } else {
      Summary<P> item = p < size ? s.load(p) : Summary<P>{0, 0, 0, 0};
      Scan(temp.scan).InclusiveScan(item, result, Maximum<P>{}, prefix);
    }
    if (p < size) {
      if (latest) *s.field(p, 1) = result1;
      else s.store(p, result);
      // The scan barrier also publishes masks; rows without repairs never read them.
      if (active) {
        if constexpr (reuse) if (!qlocal) qw[p] = 0;
        if (p < qsize)
          changed[p] = 0;
        changed[t + 1 + p] = 0;
      }
    }
    __syncthreads();
  }
  bool owners = direct && 2 * offset[d] >= i + length + 1;
  unsigned* used_ptr = &temp.local.used;
  unsigned& used = temp.local.used;
  int* list = temp.local.list;
  unsigned* local_q = temp.local.q;
  // CUB no longer uses temp after the last prefix-scan publication barrier.
  if (qlocal)
    for (int p = thread; p <= length; p += N) local_q[p] = 0;
  // L=0 implies n=0 for every candidate: K positions are already unique.
  bool track = owners && list_size && length;
  auto claim = [=] __device__(int side, int p, U encoded) {
    if (qlocal && !side) {
      // A single active bit shares one epoch; only its priority needs storage.
      atomicMax(local_q + p - (i - length),
                static_cast<unsigned>(encoded & ((1ull << (2 * shift)) - 1)));
      return;
    }
    U* at = winner(side, p);
    if (track && side) {
      U old = atomicMax(at, encoded);
      if ((old >> (2 * shift)) < (encoded >> (2 * shift))) {
        unsigned slot = atomicAdd(used_ptr, 1u);
        if (slot < list_size) list[slot] = p;
      }
    } else {
      // Keep non-returning reduction separate from first-touch atomicMax.
      asm volatile("red.relaxed.gpu.global.max.u64 [%0], %1;"
                   : : "l"(at), "l"(encoded) : "memory");
    }
  };
  for (unsigned pending = active; pending; pending &= pending - 1) {
    int bit = __ffs(pending) - 1, left = direct ? 0 : offset[bit],
        right = direct ? records : offset[bit + 1];
    if (!thread) used = 0;
    __syncthreads();
    U pending_event = prefetch_event(left + thread, right);
    for (int e = left + thread; e < right; e += N) {
      U word = pending_event;
      pending_event = prefetch_event(e + N, right);
      int3 x = read_event(e, word);
      int j = x.x, n = x.y, r = x.z;
      if (j < 0) continue;
      U next = priority(r, j);
      for (int side = 0; side < 2; ++side) {
        int p = (side ? j : i) - n;
        U encoded = stamp(epoch + side * d + bit + 1, r, j, shift);
        if (!side || next > baseline(s, i, p, side, old)) {
          claim(side, p, encoded);
        }
      }
    }
    __syncthreads();
    if (owners) {
      for (int side = 0; side < 2; ++side) {
        int begin = side ? 0 : i - length, end = side ? i : i + 1;
        U tag = epoch + side * d + bit + 1;
        bool listed = side && track && used <= list_size;
        int count = listed ? used : end - begin;
        for (int at = thread; at < count; at += N) {
          int p = listed ? list[at] : begin + at;
          U word;
          if (qlocal && !side) {
            word = local_q[p - begin];
            if (word) word |= tag << (2 * shift);
          } else word = *winner(side, p);
          if ((word >> (2 * shift)) != tag) continue;
          int first = side ? v_idx - size : i - qsize + 1;
          int limit = side ? v_idx : i + 1;
          if (p >= first && p < limit)
            changed[side * (t + 1) + p - first] |= 1u << bit;
          int to = word & ((1ull << shift) - 1);
          float delta = credit(to);
          if (delta != 0.f)
            atomicAdd(grad(side ? dk : dq, p, bit), delta);
        }
      }
    } else {
      U pending_event = prefetch_event(left + thread, right);
      for (int e = left + thread; e < right; e += N) {
        U word = pending_event;
        pending_event = prefetch_event(e + N, right);
        int3 x = read_event(e, word);
        int j = x.x, n = x.y, r = x.z;
        if (j < 0) continue;
        float delta = 0;
        bool ready = false;
        for (int side = 0; side < 2; ++side) {
          int p = (side ? j : i) - n;
          U encoded = stamp(epoch + side * d + bit + 1, r, j, shift);
          if (encoded == *winner(side, p)) {
            int first = side ? v_idx - size : i - qsize + 1, limit = side ? v_idx : i + 1;
            // One winner per (side,position,bit); bit iterations synchronize.
            if (p >= first && p < limit)
              changed[side * (t + 1) + p - first] |= 1u << bit;
            if (!ready) {
              delta = credit(j + 1);
              ready = true;
            }
            if (delta != 0.f)
              atomicAdd(grad(side ? dk : dq, p, bit), delta);
          }
        }
      }
    }
    __syncthreads();
  }
  // Repairs own their complete original-to-new delta, never a large correction.
  for (int side = 0; side < 2; ++side) {
    int limit = side ? v_idx : i + 1, first = limit - (side ? size : qsize);
    for (int p = first + thread; p < limit; p += N) {
      unsigned mask = active ? changed[side * (t + 1) + p - first] : 0;
      U next = baseline(s, i, p, side, old);
      float delta = credit(static_cast<unsigned>(next));
      if (delta == 0.f)
        continue;
      if (!mask)
        atomicAdd(common + (static_cast<int64_t>(a) * 2 + side) * t + p, delta);
      else
        for (unsigned bits = (0xffffffffu >> (32 - d)) & ~mask; bits; bits &= bits - 1)
          atomicAdd(grad(side ? dk : dq, p, __ffs(bits) - 1), delta);
    }
  }
}
}  // namespace

Credit credit(const Tensor& q, const Tensor& k, const Tensor& value,
              const Tensor& upstream, const Tensor& route, int d, int rows) {
  c10::cuda::CUDAGuard guard(q.device());
  int s = q.size(0), t = q.size(1), dv = value.size(3);
  bool half = upstream.scalar_type() == torch::kFloat16, narrow = t < 65536;
  auto dy = (half ? upstream : upstream.to(torch::kFloat32))
                .permute({0, 2, 1, 3})
                .reshape({s, t, dv})
                .contiguous();
  auto integer = q.options(), wide = q.options().dtype(torch::kInt64),
       fp = value.options().dtype(torch::kFloat32);
  auto dq = torch::zeros({s, d, t}, fp), dk = torch::zeros_like(dq),
       common = torch::zeros({s, 2, t}, fp);
  if (!t)
    return {dq, dk, common};
  rows = std::min(rows, t);
  if (t > 1)
    at::globalContext().alertNotDeterministic("rosa_bitflip backward");
  auto in = torch::zeros({s, 3, t}, integer), out = torch::empty_like(in);
  in.select(1, 2).fill_(-1);
  auto meta = torch::empty({s, rows, t}, wide),
       ends = narrow ? q : torch::empty({s, rows, t}, integer),
       counts = torch::empty({s, rows}, integer);
  auto summary = torch::empty({s, rows, 4, t + 1}, narrow ? integer : wide);
  auto work = torch::empty({s, rows, narrow ? 1 : 2, t + 1}, wide),
       events = torch::empty({s, rows, t}, narrow ? wide : integer);
  auto excluded = torch::empty({s, rows, 2, t + 1},
                               d <= 8 ? integer.dtype(torch::kUInt8) : integer);
  auto packed = pack_values(value);
  auto stream = at::cuda::getCurrentCUDAStream();
  int shift = 32 - __builtin_clz(static_cast<unsigned>(t));
  // High bits order successive edits; reset before their unsigned order wraps.
  U cycle = (~0ull >> (2 * shift)) / (2 * d);
  for (int start = 0; start < t; start += rows) {
    int count = std::min(rows, t - start), stop = start + count;
    U band = (start / rows) % cycle, epoch = band * (2 * d);
    if (!band)
      C10_CUDA_CHECK(cudaMemsetAsync(work.data_ptr(), 0, work.nbytes(), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr(), 0, counts.nbytes(), stream));
    int tiles = (stop - 1 + 7) / 8;
    auto scan = [&](auto small) {
      dp<decltype(small)::value><<<tiles * s, 256, 0, stream>>>(
          q.data_ptr<int>(), k.data_ptr<int>(), in.data_ptr<int>(), out.data_ptr<int>(),
          reinterpret_cast<U*>(meta.data_ptr()), ends.data_ptr<int>(),
          counts.data_ptr<int>(), t, rows, start, count, tiles);
    };
    if (tiles) {
      if (narrow) scan(std::true_type{});
      else scan(std::false_type{});
    }
    auto launch = [&](auto mask, auto word, auto block, auto lookup) {
      using M = decltype(mask);
      using P = decltype(word);
      constexpr int NT = decltype(block)::value;
      constexpr int C = decltype(lookup)::value;
      int width = C == 2 ? min(dv, credit_width) : dv;
      int bytes = C ? ((width + 3) / 4) * 17 * sizeof(float) : 0;
      row<M, P, NT, C><<<s * count, NT, bytes, stream>>>(
          reinterpret_cast<U*>(meta.data_ptr()), ends.data_ptr<int>(),
          counts.data_ptr<int>(), reinterpret_cast<const U*>(route.data_ptr<int64_t>()),
          reinterpret_cast<P*>(summary.data_ptr()),
          reinterpret_cast<U*>(work.data_ptr<int64_t>()), events.data_ptr(),
          reinterpret_cast<M*>(excluded.data_ptr()), dq.data_ptr<float>(),
          dk.data_ptr<float>(), common.data_ptr<float>(), t, d, rows, start, count,
          dy.data_ptr(), reinterpret_cast<const unsigned*>(packed.data_ptr<int>()), dv,
          half, epoch, upstream.size(2) / value.size(2), q.data_ptr<int>(),
          k.data_ptr<int>());
    };
    auto select = [&](auto mask) {
      auto size = [&](auto nt) {
        auto type = [&](auto p) {
          if (dv <= 512) {
            if (((dv + 3) / 4) % 8 == 0)
              launch(mask, p, nt, std::integral_constant<int, 3>{});
            else
              launch(mask, p, nt, std::integral_constant<int, 1>{});
          } else if constexpr (decltype(nt)::value == 256)
            launch(mask, p, nt, std::integral_constant<int, 2>{});
          else
            launch(mask, p, nt, std::integral_constant<int, 0>{});
        };
        if (narrow)
          type(unsigned{});
        else
          type(U{});
      };
      size(std::integral_constant<int, 32>{});
      size(std::integral_constant<int, 256>{});
    };
    if (d <= 8)
      select(uint8_t{});
    else
      select(unsigned{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::swap(in, out);
  }
  return {dq, dk, common};
}

}  // namespace rosa::bitflip
