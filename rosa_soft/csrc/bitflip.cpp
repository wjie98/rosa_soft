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

std::tuple<Tensor, Tensor, Tensor, Tensor> forward(const Tensor& q, const Tensor& k,
                                                   const Tensor& v, int64_t rows) {
  TORCH_CHECK(rows >= 1 && rows <= 256, "rows must be in [1,256]");
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
}  // namespace

TORCH_LIBRARY_FRAGMENT(rosa_soft, m) {
  m.def(
      "bitflip_forward(Tensor q, Tensor k, Tensor v, int rows)"
      " -> (Tensor y, Tensor pq, Tensor pk, Tensor route)");
  m.def(
      "bitflip_backward(Tensor q, Tensor k, Tensor v, Tensor dy,"
      " Tensor pq, Tensor pk, Tensor route, int d, int rows, int mask)"
      " -> (Tensor dq, Tensor dk, Tensor dv)");
}

TORCH_LIBRARY_IMPL(rosa_soft, CUDA, m) {
  m.impl("bitflip_forward", &forward);
  m.impl("bitflip_backward", &backward);
}
