#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "sam.h"

namespace {

void check(
    const torch::Tensor& x,
    const char* name,
    int64_t n,
    int64_t h) {
  TORCH_CHECK(x.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({n, h}),
              name, " must have shape [N,H]");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

}  // namespace

class RosaSam : public torch::CustomClassHolder {
 public:
  RosaSam(int64_t h, int64_t d) : h_(h) {
    TORCH_CHECK(h > 0, "num_heads must be positive");
    TORCH_CHECK(d >= 1 && d <= 32, "symbol_bits must be in [1,32]");
    mask_ = d == 32 ? ~uint32_t{0} : (uint32_t{1} << d) - 1;
  }

  torch::Tensor update(
      const torch::Tensor& cu,
      const torch::Tensor& q,
      const torch::Tensor& k) {
    TORCH_CHECK(cu.device().is_cpu() && cu.scalar_type() == torch::kInt64 &&
                    cu.dim() == 1 && cu.numel() >= 2 && cu.is_contiguous(),
                "cu_seqlens must be contiguous CPU int64");
    const auto* off = cu.data_ptr<int64_t>();
    const int64_t b = cu.numel() - 1;
    const int64_t n = off[b];
    TORCH_CHECK(off[0] == 0 && n >= 0, "invalid cu_seqlens bounds");
    for (int64_t i = 0; i < b; ++i) {
      TORCH_CHECK(off[i] <= off[i + 1], "cu_seqlens must be nondecreasing");
      TORCH_CHECK(off[i + 1] - off[i] <= std::numeric_limits<int>::max(),
                  "sequence exceeds int32 range");
    }
    check(q, "q", n, h_);
    check(k, "k", n, h_);

    if (b_ < 0) {
      b_ = b;
      sam_.resize(static_cast<size_t>(b * h_));
    }
    TORCH_CHECK(b == b_,
                "sequence count changed; call reset() before update()");
    for (int64_t s = 0; s < b; ++s) {
      const size_t add = static_cast<size_t>(off[s + 1] - off[s]);
      for (int64_t a = 0; a < h_; ++a) sam_[s * h_ + a].reserve(add);
    }

    auto y = torch::empty({n, h_}, q.options().dtype(torch::kInt64));
    const auto* qp = q.data_ptr<int32_t>();
    const auto* kp = k.data_ptr<int32_t>();
    auto* yp = y.data_ptr<int64_t>();
    at::parallel_for(0, b * h_, 1, [&](int64_t begin, int64_t end) {
      for (int64_t x = begin; x < end; ++x) {
        const int64_t s = x / h_;
        const int64_t a = x - s * h_;
        auto& m = sam_[x];
        for (int64_t i = off[s]; i < off[s + 1]; ++i) {
          const int64_t j = i * h_ + a;
          yp[j] = m.step(static_cast<uint32_t>(qp[j]) & mask_,
                         static_cast<uint32_t>(kp[j]) & mask_);
        }
      }
    });
    return y;
  }

  void reset() {
    sam_.clear();
    b_ = -1;
  }

 private:
  int64_t h_;
  uint32_t mask_;
  int64_t b_ = -1;
  std::vector<rosa_soft::Sam> sam_;
};

TORCH_LIBRARY_FRAGMENT(rosa_soft, m) {
  m.class_<RosaSam>("RosaSam")
      .def(torch::init<int64_t, int64_t>())
      .def("update", &RosaSam::update)
      .def("reset", &RosaSam::reset);
}
