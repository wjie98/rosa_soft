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
    P length;
    P endpos;
    P suffix_link;
    std::map<K, P> transitions;

    rosa_state_t() : endpos(-1), length(0), suffix_link(-1) {}
};

template<typename K, typename V, typename P>
class rosa_sam {
public:
    rosa_sam() : last_q_state_(0), last_k_state_(0), last_q_bits_(0), last_k_bits_(0) {}

    V update(K xq, K xk, V xv, K mq, K mk, V u, P& endpos, P& length) {
        xq = (xq & mq) | (last_q_bits_ & ~mq);
        xk = (xk & mk) | (last_k_bits_ & ~mk);

        last_q_bits_ = xq;
        last_k_bits_ = xk;

        P i = update_query_(xq, last_q_state_);
        update_key_value_(xk, xv);

        endpos = i;
        length = states_[last_q_state_].length;

        return i != -1 ? values_[i + 1] : u;
    }

private:
    P update_key_value_(K k, K v) {
        if (states_.empty()) states_.emplace_back();

        P i = values_.size();
        values_.emplace_back(v);

        P r = states_.size();
        states_.emplace_back();
        states_[r].length = states_[last_k_state_].length + 1;

        P p = last_k_state_;
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
                        it->second = u;
                        p = states_[p].suffix_link;
                    } else {
                        break;
                    }
                }
            }
        }

        last_k_state_ = r;

        while (r != -1) {
            states_[r].endpos = i;
            r = states_[r].suffix_link;
        }
        
        return r;
    }

    P update_query_(K q, P& s) {
        if (states_.empty()) states_.emplace_back();

        P j = -1;

        P r = s;
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
    
    P last_q_state_;
    P last_k_state_;

    K last_q_bits_;
    K last_k_bits_;
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> torch_rosa_sam_update(const torch::Tensor& ctx, const torch::Tensor& xq, const torch::Tensor& xk, const torch::Tensor& xv, const torch::Tensor& mq, const torch::Tensor& mk, int64_t u) {
    assert(ctx.numel() == xq.size(0));

    int64_t B = xq.size(0);
    int64_t N = xq.size(1);

    auto xq_a = xq.accessor<K, 2>();
    auto xk_a = xk.accessor<K, 2>();
    auto xv_a = xv.accessor<V, 2>();
    auto mq_a = mq.accessor<K, 2>();
    auto mk_a = mk.accessor<K, 2>();

    auto out = torch::empty({B, N}, xv.options());

    auto ctx_a = ctx.accessor<int64_t, 1>();
    auto out_a = out.accessor<V, 2>();

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(xv.device());
    
    auto endpos = torch::empty({B, N}, options);
    auto endpos_a = endpos.accessor<V, 2>();

    auto length = torch::empty({B, N}, options);
    auto length_a = length.accessor<V, 2>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_sam<K, V, P>*)ctx_a[i];
        if (!r) {
            throw std::runtime_error("rosa context is null");
        }

        for (int64_t t = 0; t < N; ++t) {
            P _endpos = -1;
            P _length = 0;
            
            out_a[i][t] = r->update(xq_a[i][t], xk_a[i][t], xv_a[i][t], mq_a[i][t], mk_a[i][t], static_cast<V>(u), _endpos, _length);

            endpos_a[i][t] = _endpos;
            length_a[i][t] = _length;
        }
    }

    return {out, endpos, length};
}


TORCH_LIBRARY(rosa_soft, m) {
    m.def("rosa_sam_init(Tensor(a!) ctx) -> Tensor");
    m.def("rosa_sam_free(Tensor(a!) ctx) -> Tensor");
    m.def("rosa_sam_update(Tensor ctx, Tensor xq, Tensor xk, Tensor xv, Tensor mq, Tensor mk, int u) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(rosa_soft, CPU, m) {
    m.impl("rosa_sam_init", &torch_rosa_sam_init<int64_t, int64_t, int64_t>);
    m.impl("rosa_sam_free", &torch_rosa_sam_free<int64_t, int64_t, int64_t>);
    m.impl("rosa_sam_update", &torch_rosa_sam_update<int64_t, int64_t, int64_t>);
}
