#include <torch/extension.h>
#include <Python.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}


TORCH_LIBRARY(rwkv_cuda, m) {
    m.def("rwkv7_clampw_forward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor s, Tensor(a!) sa) -> ()");
    m.def("rwkv7_clampw_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dr, Tensor(a!) dw, Tensor(a!) dk, Tensor(a!) dv, Tensor(a!) da, Tensor(a!) db) -> ()");

    m.def("rwkv7_state_clampw_forward(Tensor s0, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor s, Tensor(a!) sa) -> ()");
    m.def("rwkv7_state_clampw_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor sa, Tensor(a!) ds0, Tensor(a!) dr, Tensor(a!) dw, Tensor(a!) dk, Tensor(a!) dv, Tensor(a!) da, Tensor(a!) db) -> ()");

    m.def("rwkv7_statepassing_clampw_forward(Tensor s0, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(a!) sT, Tensor s, Tensor(a!) sa) -> ()");
    m.def("rwkv7_statepassing_clampw_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor(a!) dsT, Tensor s, Tensor sa, Tensor(a!) ds0, Tensor(a!) dr, Tensor(a!) dw, Tensor(a!) dk, Tensor(a!) dv, Tensor(a!) da, Tensor(a!) db) -> ()");

    m.def("rwkv7_albatross_forward_w0_fp16_dither_seq(Tensor(a!) s0, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()");
    m.def("rwkv7_albatross_forward_w0_fp16_dither_one(Tensor(a!) s0, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()");
}
