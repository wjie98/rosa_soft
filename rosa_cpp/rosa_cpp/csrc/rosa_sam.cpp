#include <map>
#include <tuple>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>

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


template<typename K, typename V, typename P>
struct rosa_state_t {
    P endpos;
    P length;
    P suffix_link;
    std::map<K, P> transitions;

    rosa_state_t() : endpos(-1), length(0), suffix_link(-1) {}
};

template<typename K, typename V, typename P>
class rosa_sam {
public:
    rosa_sam() : last_q_(0), last_k_(0) {}
    const std::vector<V>& values() const { return values_; }

    V append(K q, K k, V v, V u) {
        P i = -1;
        return append(q, k, v, u, i);
    }

    V append(K q, K k, V v, V u, P& i) {
        update_key_value_(k, v);

        i = update_query_(q, last_q_);
        auto ret = i != -1 ? values_[i + 1] : u;

        update_endpos_();
        return ret;
    }

    void clear() {
        values_.clear();
        states_.clear();
        last_q_ = 0;
        last_k_ = 0;
    }

private:
    P update_key_value_(K k, K v) {
        if (states_.empty()) states_.emplace_back();

        // P i = values_.size();
        values_.emplace_back(v);

        P r = states_.size();
        states_.emplace_back();
        states_[r].length = states_[last_k_].length + 1;

        P p = last_k_;
        while (p != -1) {
            if (states_[p].transitions.count(k)) {
                break;
            }

            states_[p].transitions[k] = r;
            p = states_[p].suffix_link;
        }

        if (p == -1) {
            states_[r].suffix_link = 0;
        } else {
            P c = states_[p].transitions[k];

            if (states_[p].length + 1 == states_[c].length) {
                states_[r].suffix_link = c;
            } else {
                P u = states_.size();
                states_.emplace_back(states_[c]);
                states_[u].length = states_[p].length + 1;
                states_[c].suffix_link = u;
                states_[r].suffix_link = u;

                while (p != -1) {
                    auto it = states_[p].transitions.find(k);
                    if (it != states_[p].transitions.end() && it->second == c) {
                        states_[p].transitions[k] = u;
                        p = states_[p].suffix_link;
                    } else {
                        break;
                    }
                }
            }
        }

        last_k_ = r;
        return r;
    }

    P update_endpos_() {
        P i = values_.size() - 1;
        P r = last_k_;
        while (r != -1 && states_[r].endpos < i) {
            states_[r].endpos = i;
            r = states_[r].suffix_link;
        }
        return i;
    }

    P update_query_(K q, P& s) {
        P j = -1;
        P r = last_q_;
        while (r != -1 && !states_[r].transitions.count(q)) {
            r = states_[r].suffix_link;
        }

        if (r == -1) {
            s = 0;
        } else {
            r = s = states_[r].transitions[q];
            while (r != -1) {
                if (states_[r].length > 0 && states_[r].endpos >= 0) {
                    j = states_[r].endpos;
                    break;
                }
                r = states_[r].suffix_link;
            }
        }
        
        return j;
    }

    std::vector<V> values_;
    std::vector<rosa_state_t<K, V, P>> states_;
    P last_q_;
    P last_k_;
};


#include <torch/extension.h>

template<typename K, typename V, typename P>
torch::Tensor& torch_rosa_sam_init(torch::Tensor& ctx) {
    int64_t B = ctx.numel();
    auto ctx_a = ctx.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_sam<K, V, P>*)ctx_a[i];
        
        if (r) {
            delete r;
        }

        r = new rosa_sam<K, V, P>();

        ctx_a[i] = (int64_t)r;
    }

    return ctx;
}

template<typename K, typename V, typename P>
torch::Tensor& torch_rosa_sam_free(torch::Tensor& ctx) {
    int64_t B = ctx.numel();
    auto ctx_a = ctx.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_sam<K, V, P>*)ctx_a[i];
        
        if (r) {
            delete r;
        }

        ctx_a[i] = 0;
    }

    return ctx;
}

template<typename K, typename V, typename P>
torch::Tensor torch_rosa_sam_update(const torch::Tensor& ctx, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t u) {
    assert(ctx.numel() == q.size(0));

    int64_t B = q.size(0);
    int64_t N = q.size(1);

    auto out = torch::empty({B, N}, v.options());

    auto q_a = q.accessor<K, 2>();
    auto k_a = k.accessor<K, 2>();
    auto v_a = v.accessor<V, 2>();

    auto ctx_a = ctx.accessor<int64_t, 1>();
    auto out_a = out.accessor<V, 2>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_sam<K, V, P>*)ctx_a[i];
        if (!r) {
            throw std::runtime_error("rosa context is null");
        }

        for (int64_t t = 0; t < N; ++t) {
            out_a[i][t] = r->append(q_a[i][t], k_a[i][t], v_a[i][t], static_cast<V>(u));
        }
    }

    return out;
}


TORCH_LIBRARY(rosa_cpp, m) {
    m.def("rosa_sam_k8v8_init(Tensor ctx) -> Tensor");
    m.def("rosa_sam_k8v8_free(Tensor ctx) -> Tensor");
    m.def("rosa_sam_k8v8_update(Tensor ctx, Tensor q, Tensor k, Tensor v, int u) -> Tensor");

    m.def("rosa_sam_k8v64_init(Tensor ctx) -> Tensor");
    m.def("rosa_sam_k8v64_free(Tensor ctx) -> Tensor");
    m.def("rosa_sam_k8v64_update(Tensor ctx, Tensor q, Tensor k, Tensor v, int u) -> Tensor");

    m.def("rosa_sam_k64v64_init(Tensor ctx) -> Tensor");
    m.def("rosa_sam_k64v64_free(Tensor ctx) -> Tensor");
    m.def("rosa_sam_k64v64_update(Tensor ctx, Tensor q, Tensor k, Tensor v, int u) -> Tensor");
}

TORCH_LIBRARY_IMPL(rosa_cpp, CPU, m) {
    m.impl("rosa_sam_k8v8_init", &torch_rosa_sam_init<uint8_t, uint8_t, int64_t>);
    m.impl("rosa_sam_k8v8_free", &torch_rosa_sam_free<uint8_t, uint8_t, int64_t>);
    m.impl("rosa_sam_k8v8_update", &torch_rosa_sam_update<uint8_t, uint8_t, int64_t>);

    m.impl("rosa_sam_k8v64_init", &torch_rosa_sam_init<uint8_t, int64_t, int64_t>);
    m.impl("rosa_sam_k8v64_free", &torch_rosa_sam_free<uint8_t, int64_t, int64_t>);
    m.impl("rosa_sam_k8v64_update", &torch_rosa_sam_update<uint8_t, int64_t, int64_t>);

    m.impl("rosa_sam_k64v64_init", &torch_rosa_sam_init<int64_t, int64_t, int64_t>);
    m.impl("rosa_sam_k64v64_free", &torch_rosa_sam_free<int64_t, int64_t, int64_t>);
    m.impl("rosa_sam_k64v64_update", &torch_rosa_sam_update<int64_t, int64_t, int64_t>);
}
