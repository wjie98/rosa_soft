#include "cuda/bitflip.cuh"

#include <Python.h>

extern "C" {
PyObject* PyInit__bitflip() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_bitflip", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
}

namespace {
using Tensor = torch::Tensor;

void check_chunks(const Tensor& q, const Tensor& v, int64_t chunks) {
  TORCH_CHECK(chunks > 0, "chunks must be positive");
  TORCH_CHECK(q.dim() == 4 && v.dim() == 4 &&
              q.size(2) % chunks == 0 && v.size(2) % chunks == 0,
              "chunks must divide H and Hv (complete GQA groups)");
}

std::tuple<Tensor, Tensor, Tensor, Tensor> forward(const Tensor& q, const Tensor& k,
                                                   const Tensor& v, int64_t rows,
                                                   int64_t chunks) {
  TORCH_CHECK(rows >= 1 && rows <= 256, "rows must be in [1,256]");
  check_chunks(q, v, chunks);
  return rosa::bitflip::forward(q.contiguous(), k.contiguous(), v.contiguous());
}

std::tuple<Tensor, Tensor, Tensor> backward(const Tensor& q, const Tensor& k,
                                            const Tensor& v, const Tensor& dy,
                                            const Tensor& pq, const Tensor& pk,
                                            const Tensor& route, int64_t d,
                                            int64_t rows, int64_t mask) {
  TORCH_CHECK(mask >= 1 && mask <= 7, "gradient mask must be in [1,7]");
  TORCH_CHECK(d >= 1 && d <= 32 && rows >= 1 && rows <= 256, "invalid D/rows");
  return rosa::bitflip::backward(q.contiguous(), k.contiguous(), v.contiguous(),
                                dy.contiguous(), pq, pk, route, d, rows, mask);
}

auto joint_forward(const Tensor& x, const Tensor& v, int64_t rows, bool all, int64_t chunks) {
  TORCH_CHECK(rows >= 1 && rows <= 256, "rows must be in [1,256]");
  check_chunks(x, v, chunks);
  TORCH_CHECK(x.dim() == 4 && x.size(1) < (1 << 20) && x.size(0)*x.size(2) <= 65535,
              "joint bitflip index limit exceeded");
  TORCH_CHECK(!all || (x.sizes() == v.sizes() && x.device() == v.device() &&
                      x.scalar_type() == v.scalar_type()), "QKV requires matching inputs");
  auto q = x.contiguous();
  return rosa::bitflip::forward(q, q, all ? q : v.contiguous());
}

auto joint_backward(const Tensor& x, const Tensor& v, const Tensor& dy,
                    const Tensor& q, const Tensor& route, int64_t rows,
                    bool all, int64_t mask) {
  TORCH_CHECK(rows >= 1 && rows <= 256 && mask >= 1 && mask <= (all ? 1 : 3),
              "invalid joint rows/gradient mask");
  return rosa::bitflip::joint_backward(x.contiguous(), v.contiguous(), dy.contiguous(),
                                       q, route, rows, all, mask);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(rosa_soft, m) {
  m.def(
      "bitflip_forward(Tensor q, Tensor k, Tensor v, int rows, int chunks=1)"
      " -> (Tensor y, Tensor pq, Tensor pk, Tensor route)");
  m.def(
      "bitflip_backward(Tensor q, Tensor k, Tensor v, Tensor dy,"
      " Tensor pq, Tensor pk, Tensor route, int d, int rows, int mask)"
      " -> (Tensor dq, Tensor dk, Tensor dv)");
  m.def("joint_forward(Tensor x, Tensor v, int rows, bool all, int chunks=1)"
        " -> (Tensor y, Tensor q, Tensor k, Tensor route)");
  m.def("joint_backward(Tensor x, Tensor v, Tensor dy, Tensor q, Tensor route,"
        " int rows, bool all, int mask) -> (Tensor dx, Tensor dv)");
}

TORCH_LIBRARY_IMPL(rosa_soft, CUDA, m) {
  m.impl("bitflip_forward", &forward);
  m.impl("bitflip_backward", &backward);
  m.impl("joint_forward", &joint_forward);
  m.impl("joint_backward", &joint_backward);
}
