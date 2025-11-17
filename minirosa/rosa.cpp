#include <map>
#include <tuple>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>


template<typename T, typename P>
struct rosa_state_t {
    P endpos;
    P length;
    P suffix_link;
    std::map<T, P> transitions;

    rosa_state_t() : endpos(-1), length(0), suffix_link(-1) {}
};

template<typename T, typename P>
class rosa_sam {
public:
    rosa_sam() : last_q_(0), last_k_(0) {}
    const std::vector<T>& values() const { return values_; }

    T append(T q, T k, T v, T u) {
        P i = -1;
        return append(q, k, v, u, i);
    }

    T append(T q, T k, T v, T u, P& i) {
        update_key_value_(k, v);

        i = update_query_(q, last_q_);
        T ret = i != -1 ? values_[i + 1] : u;

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
    P update_key_value_(T k, T v) {
        if (states_.empty()) states_.emplace_back();

        P i = values_.size();
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

    P update_query_(T q, P& s) {
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

    std::vector<T> values_;
    std::vector<rosa_state_t<T, P>> states_;
    P last_q_;
    P last_k_;
};


#include <torch/extension.h>

template<typename T, typename P>
torch::Tensor& torch_rosa_sam_init(torch::Tensor& ctx) {
    int64_t B = ctx.numel();
    auto ctx_a = ctx.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_sam<T, P>*)ctx_a[i];
        
        if (r) {
            delete r;
        }

        r = new rosa_sam<T, P>();

        ctx_a[i] = (int64_t)r;
    }

    return ctx;
}

template<typename T, typename P>
torch::Tensor& torch_rosa_sam_free(torch::Tensor& ctx) {
    int64_t B = ctx.numel();
    auto ctx_a = ctx.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_sam<T, P>*)ctx_a[i];
        
        if (r) {
            delete r;
        }

        ctx_a[i] = 0;
    }

    return ctx;
}

template<typename T, typename P>
torch::Tensor torch_rosa_sam_update(const torch::Tensor& ctx, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t u) {
    assert(ctx.numel() == q.size(0));

    int64_t B = q.size(0);
    int64_t N = q.size(1);

    auto out = torch::empty_like(q);

    auto q_a = q.accessor<T, 2>();
    auto k_a = k.accessor<T, 2>();
    auto v_a = v.accessor<T, 2>();

    auto ctx_a = ctx.accessor<int64_t, 1>();
    auto out_a = out.accessor<T, 2>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_sam<T, P>*)ctx_a[i];
        if (!r) {
            throw std::runtime_error("rosa context is null");
        }

        for (int64_t t = 0; t < N; ++t) {
            out_a[i][t] = r->append(q_a[i][t], k_a[i][t], v_a[i][t], static_cast<T>(u));
        }
    }

    return out;
}


TORCH_LIBRARY(torch_rosa, m) {
    m.def("rosa_sam_init(Tensor ctx) -> Tensor");
    m.def("rosa_sam_free(Tensor ctx) -> Tensor");
    m.def("rosa_sam_update(Tensor ctx, Tensor q, Tensor k, Tensor v, int u) -> Tensor");

    m.def("rosa_sam_8bits_init(Tensor ctx) -> Tensor");
    m.def("rosa_sam_8bits_free(Tensor ctx) -> Tensor");
    m.def("rosa_sam_8bits_update(Tensor ctx, Tensor q, Tensor k, Tensor v, int u) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_rosa, CPU, m) {
    m.impl("rosa_sam_init", &torch_rosa_sam_init<int64_t, int64_t>);
    m.impl("rosa_sam_free", &torch_rosa_sam_free<int64_t, int64_t>);
    m.impl("rosa_sam_update", &torch_rosa_sam_update<int64_t, int64_t>);

    m.impl("rosa_sam_8bits_init", &torch_rosa_sam_init<uint8_t, int64_t>);
    m.impl("rosa_sam_8bits_free", &torch_rosa_sam_free<uint8_t, int64_t>);
    m.impl("rosa_sam_8bits_update", &torch_rosa_sam_update<uint8_t, int64_t>);
}
