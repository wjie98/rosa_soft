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
    m.def("hard_forward(Tensor query, Tensor key, Tensor value, int max_suffix_length) -> (Tensor output, Tensor packed_query_symbols, Tensor packed_key_symbols)");
    m.def("hard_forward_varlen(Tensor query, Tensor key, Tensor value, Tensor cu_seqlens, int max_suffix_length) -> (Tensor output, Tensor packed_query_symbols, Tensor packed_key_symbols)");
    m.def("surrogate_vjp_masked(Tensor query, Tensor key, Tensor value, Tensor grad_output, Tensor packed_query_symbols, Tensor packed_key_symbols, Tensor dropout_seed, int max_suffix_length, float scale, float dropout_p, float mismatch_scale, int gradient_mask) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)");
    m.def("surrogate_vjp_varlen_masked(Tensor query, Tensor key, Tensor value, Tensor cu_seqlens, Tensor grad_output, Tensor packed_query_symbols, Tensor packed_key_symbols, Tensor dropout_seed, int max_suffix_length, float scale, float dropout_p, float mismatch_scale, int gradient_mask) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)");
}
#endif
