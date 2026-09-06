#include <Python.h>
#include <torch/extension.h>

extern "C" {
PyObject* PyInit__C() {
  static PyModuleDef module = {
      PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
}

#ifdef ROSA_WITH_CUDA
TORCH_LIBRARY(rosa_soft, m) {
  m.def("forward(Tensor q, Tensor k, Tensor v, Tensor cu)"
        " -> (Tensor y, Tensor pq, Tensor pk)");
  m.def("backward(Tensor q, Tensor k, Tensor v, Tensor dy, Tensor pq, Tensor pk,"
        " Tensor seed, Tensor cu, float scale, float dropout_p,"
        " float mismatch_scale, int mask) -> (Tensor dq, Tensor dk, Tensor dv)");
}
#endif
