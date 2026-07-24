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

#include <torch/extension.h>

#ifdef ROSA_WITH_CUDA
TORCH_LIBRARY(rosa_soft, m) {
    m.def("soft_forward(Tensor query_logits, Tensor key_logits, Tensor payload_logits, int max_suffix_length) -> Tensor[]");
    m.def("soft_backward(Tensor query_logits, Tensor key_logits, Tensor payload_logits, Tensor grad_output, Tensor query_symbols, Tensor key_symbols, Tensor rng_seed, int max_suffix_length, float route_temperature, float mismatch_penalty) -> Tensor[]");

#ifdef ROSA_WITH_RWKV7
    // Forward kernels write y and their saved-state workspaces.
    m.def("rwkv7_clampw_forward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("rwkv7_state_clampw_forward(Tensor s0, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("rwkv7_statepassing_clampw_forward(Tensor s0, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(b!) sT, Tensor(c!) s, Tensor(d!) sa) -> ()");

    // Backward kernels read saved state and write every gradient output.
    m.def("rwkv7_clampw_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dr, Tensor(b!) dw, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) da, Tensor(f!) db) -> ()");
    m.def("rwkv7_state_clampw_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor sa, Tensor(a!) ds0, Tensor(b!) dr, Tensor(c!) dw, Tensor(d!) dk, Tensor(e!) dv, Tensor(f!) da, Tensor(g!) db) -> ()");
    m.def("rwkv7_statepassing_clampw_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor dsT, Tensor s, Tensor sa, Tensor(a!) ds0, Tensor(b!) dr, Tensor(c!) dw, Tensor(d!) dk, Tensor(e!) dv, Tensor(f!) da, Tensor(g!) db) -> ()");

    // Albatross updates state in place and writes y; elapsed_t is read-only.
    m.def("rwkv7_albatross_forward_w0_fp16_dither(Tensor(a!) s0, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(b!) y, Tensor elapsed_t) -> ()");
#endif
}
#endif
