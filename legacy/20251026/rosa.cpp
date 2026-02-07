#include <map>
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
class rosa_seq {
public:
    rosa_seq() : last_(0) {}
    rosa_seq(size_t n) : last_(0) { this->reserve(n); }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& v() const { return v_; }

    T* extend(const T* x_ptr, const T* v_ptr, size_t n, T* out, T u) {
        for (P i = 0; i < n; ++i) {
            out[i] = append(x_ptr[i], v_ptr[i], u);
        }
        return out;
    }

    T append(T x, T v, T u) {
        P i = add_xv_(x, v);
        return i >= 0 ? v_[i] : u;
    }

    void clear() {
        x_.clear(); v_.clear();
        states_.clear(); last_ = 0;
    }

    void reserve(size_t n) {
        x_.reserve(n); v_.reserve(n);
        states_.reserve(2 * n + 1);
    }

private:

    P add_xv_(T x, T v) {
        if (states_.empty()) states_.emplace_back();

        P i = x_.size();
        x_.emplace_back(x);
        v_.emplace_back(v);

        P r = states_.size();
        states_.emplace_back();
        states_[r].length = states_[last_].length + 1;

        P p = last_;
        while (p != -1) {
            if (states_[p].transitions.count(x)) {
                break;
            }

            states_[p].transitions[x] = r;
            p = states_[p].suffix_link;
        }

        if (p == -1) {
            states_[r].suffix_link = 0;
        } else {
            P c = states_[p].transitions[x];

            if (states_[p].length + 1 == states_[c].length) {
                states_[r].suffix_link = c;
            } else {
                P u = states_.size();
                states_.emplace_back(states_[c]);
                states_[u].length = states_[p].length + 1;
                states_[c].suffix_link = u;
                states_[r].suffix_link = u;

                while (p != -1) {
                    auto it = states_[p].transitions.find(x);
                    if (it != states_[p].transitions.end() && it->second == c) {
                        states_[p].transitions[x] = u;
                        p = states_[p].suffix_link;
                    } else {
                        break;
                    }
                }
            }
        }

        P j = -1;

        last_ = r;
        while (r != -1) {
            if (states_[r].length > 0 && states_[r].endpos >= 0) {
                j = states_[r].endpos + 1;
                break;
            }
            r = states_[r].suffix_link;
        }

        r = last_;
        while (r != -1 && states_[r].endpos < i) {
            states_[r].endpos = i;
            r = states_[r].suffix_link;
        }
        
        return j;
    }

    std::vector<T> x_;
    std::vector<T> v_;
    std::vector<rosa_state_t<T, P>> states_;
    P last_;
};

template<typename T, typename P>
class rosa_qkv {
public:
    rosa_qkv() : last_q_(0), last_k_(0) {}
    rosa_qkv(size_t n) : last_q_(0), last_k_(0) { this->reserve(n); }

    const std::vector<T>& q() const { return q_; }
    const std::vector<T>& k() const { return k_; }
    const std::vector<T>& v() const { return v_; }

    T* extend(const T* q_ptr, const T* k_ptr, const T* v_ptr, size_t n, T* out) {
        for (P i = 0; i < n; ++i) {
            out[i] = append(q_ptr[i], k_ptr[i], v_ptr[i]);
        }
        return out;
    }

    T append(T q, T k, T v) {
        P i = add_qkv_(q, k, v);
        return i >= 0 ? v_[i] : v_.back();
    }

    void clear() {
        q_.clear(); k_.clear(); v_.clear();
        states_.clear(); last_q_ = 0; last_k_ = 0;
    }

    void reserve(size_t n) {
        q_.reserve(n); k_.reserve(n); v_.reserve(n);
        states_.reserve(2 * n + 1);
    }

private:

    P add_qkv_(T q, T k, T v) {
        if (states_.empty()) states_.emplace_back();

        P i = q_.size();
        q_.emplace_back(q);
        k_.emplace_back(k);
        v_.emplace_back(v);

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
        while (r != -1 && states_[r].endpos < i) {
            states_[r].endpos = i;
            r = states_[r].suffix_link;
        }

        P j = -1;

        p = last_q_;
        while (p != -1 && !states_[p].transitions.count(q)) {
            p = states_[p].suffix_link;
        }

        if (p == -1) {
            last_q_ = 0;
        } else {
            last_q_ = states_[p].transitions[q];

            p = last_q_;
            while (p != -1) {
                if (states_[p].length > 0 && states_[p].endpos >= 0) {
                    j = states_[p].endpos;
                    break;
                }
                p = states_[p].suffix_link;
            }
        }
        
        return j;
    }

    std::vector<T> q_;
    std::vector<T> k_;
    std::vector<T> v_;

    std::vector<rosa_state_t<T, P>> states_;
    P last_q_;
    P last_k_;
};

#include <torch/extension.h>

template<typename T, typename P>
torch::Tensor& torch_rosa_seq_init(torch::Tensor& objs, int64_t n) {
    int64_t B = objs.numel();
    auto objs_a = objs.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_seq<T, P>*)objs_a[i];
        if (r) {
            delete r;
        }

        r = new rosa_seq<T, P>();
        
        if (n > 0) {
            r->reserve(n);
        }

        objs_a[i] = (int64_t)r;
    }
    return objs;
}

template<typename T, typename P>
torch::Tensor& torch_rosa_seq_free(torch::Tensor& objs) {
    int64_t B = objs.numel();
    auto objs_a = objs.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_seq<T, P>*)objs_a[i];
        if (r) {
            delete r;
        }
        objs_a[i] = 0;
    }
    return objs;
}

template<typename T, typename P>
torch::Tensor torch_rosa_seq_append(const torch::Tensor& ctx, const torch::Tensor& src, const torch::Tensor& val, int64_t u) {
    int64_t B = std::min(ctx.numel(), src.numel());

    auto out = torch::empty_like(src);

    auto ctx_a = ctx.accessor<int64_t, 1>();
    auto src_a = src.accessor<T, 1>();
    auto val_a = val.accessor<T, 1>();
    auto out_a = out.accessor<T, 1>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_seq<T, P>*)ctx_a[i];
        if (!r) {
            throw std::runtime_error("rosa context is null");
        }
        out_a[i] = r->append(src_a[i], val_a[i], u);
    }

    return out;
}

template<typename T, typename P>
torch::Tensor torch_rosa_seq_extend(const torch::Tensor& ctx, const torch::Tensor& src, const torch::Tensor& val, int64_t u) {
    int64_t B = std::min(ctx.numel(), src.size(0));
    int64_t N = src.size(1);

    auto out = torch::empty_like(src);

    auto ctx_a = ctx.accessor<int64_t, 1>();
    auto src_a = src.accessor<T, 2>();
    auto val_a = val.accessor<T, 2>();
    auto out_a = out.accessor<T, 2>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_seq<T, P>*)ctx_a[i];
        if (!r) {
            throw std::runtime_error("rosa context is null");
        }

        for (int64_t t = 0; t < N; ++t) {
            out_a[i][t] = r->append(src_a[i][t], val_a[i][t], u);
        }
    }

    return out;
}


template<typename T, typename P>
torch::Tensor torch_rosa_seq_run(const torch::Tensor& src, const torch::Tensor& val, int64_t u) {
    int64_t B = src.size(0);
    int64_t N = src.size(1);

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto ctx = torch::zeros({B}, options);

    torch_rosa_seq_init<T, P>(ctx, N);
    
    auto out = torch_rosa_seq_extend<T, P>(ctx, src, val, u);

    torch_rosa_seq_free<T, P>(ctx);
    return out;
}


template<typename T, typename P>
torch::Tensor& torch_rosa_qkv_init(torch::Tensor& objs, int64_t n) {
    int64_t B = objs.numel();
    auto objs_a = objs.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_qkv<T, P>*)objs_a[i];
        if (r) {
            delete r;
        }

        r = new rosa_qkv<T, P>();

        if (n > 0) {
            r->reserve(n);
        }

        objs_a[i] = (int64_t)r;
    }
    return objs;
}

template<typename T, typename P>
torch::Tensor& torch_rosa_qkv_free(torch::Tensor& objs) {
    int64_t B = objs.numel();
    auto objs_a = objs.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_qkv<T, P>*)objs_a[i];
        if (r) {
            delete r;
        }
        objs_a[i] = 0;
    }
    return objs;
}

template<typename T, typename P>
torch::Tensor torch_rosa_qkv_append(const torch::Tensor& ctx, const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
    int64_t B = std::min(ctx.numel(), query.numel());

    auto out = torch::empty_like(query);

    auto ctx_a = ctx.accessor<int64_t, 1>();

    auto query_a = query.accessor<T, 1>();
    auto key_a = key.accessor<T, 1>();
    auto value_a = value.accessor<T, 1>();
    auto out_a = out.accessor<T, 1>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_qkv<T, P>*)ctx_a[i];
        if (!r) {
            throw std::runtime_error("rosa context is null");
        }
        out_a[i] = r->append(query_a[i], key_a[i], value_a[i]);
    }

    return out;
}

template<typename T, typename P>
torch::Tensor torch_rosa_qkv_extend(const torch::Tensor& ctx, const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
    int64_t B = std::min(ctx.numel(), query.size(0));
    int64_t N = query.size(1);

    auto out = torch::empty_like(query);

    auto ctx_a = ctx.accessor<int64_t, 1>();
    auto query_a = query.accessor<T, 2>();
    auto key_a = key.accessor<T, 2>();
    auto value_a = value.accessor<T, 2>();
    auto out_a = out.accessor<T, 2>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_qkv<T, P>*)ctx_a[i];
        if (!r) {
            throw std::runtime_error("rosa context is null");
        }

        for (int64_t t = 0; t < N; ++t) {
            out_a[i][t] = r->append(query_a[i][t], key_a[i][t], value_a[i][t]);
        }
    }

    return out;
}

template<typename T, typename P>
torch::Tensor torch_rosa_qkv_run(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
    int64_t B = query.size(0);
    int64_t N = query.size(1);

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto ctx = torch::zeros({B}, options);

    torch_rosa_qkv_init<T, P>(ctx, N);
    
    auto out = torch_rosa_qkv_extend<T, P>(ctx, query, key, value);

    torch_rosa_qkv_free<T, P>(ctx);
    return out;
}


TORCH_LIBRARY(torch_rosa, m) {
    m.def("rosa_seq_run(Tensor x, Tensor v, int u) -> Tensor");
    m.def("rosa_seq_init(Tensor ctx, int n) -> Tensor");
    m.def("rosa_seq_free(Tensor ctx) -> Tensor");
    m.def("rosa_seq_append(Tensor ctx, Tensor x, Tensor v, int u) -> Tensor");
    m.def("rosa_seq_extend(Tensor ctx, Tensor x, Tensor v, int u) -> Tensor");

    m.def("rosa_qkv_run(Tensor q, Tensor k, Tensor v) -> Tensor");
    m.def("rosa_qkv_init(Tensor ctx, int n) -> Tensor");
    m.def("rosa_qkv_free(Tensor ctx) -> Tensor");
    m.def("rosa_qkv_append(Tensor ctx, Tensor q, Tensor k, Tensor v) -> Tensor");
    m.def("rosa_qkv_extend(Tensor ctx, Tensor q, Tensor k, Tensor v) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_rosa, CPU, m) {
    m.impl("rosa_seq_run", &torch_rosa_seq_run<int64_t, int64_t>);
    m.impl("rosa_seq_init", &torch_rosa_seq_init<int64_t, int64_t>);
    m.impl("rosa_seq_free", &torch_rosa_seq_free<int64_t, int64_t>);
    m.impl("rosa_seq_append", &torch_rosa_seq_append<int64_t, int64_t>);
    m.impl("rosa_seq_extend", &torch_rosa_seq_extend<int64_t, int64_t>);

    m.impl("rosa_qkv_run", &torch_rosa_qkv_run<int64_t, int64_t>);
    m.impl("rosa_qkv_init", &torch_rosa_qkv_init<int64_t, int64_t>);
    m.impl("rosa_qkv_free", &torch_rosa_qkv_free<int64_t, int64_t>);
    m.impl("rosa_qkv_append", &torch_rosa_qkv_append<int64_t, int64_t>);
    m.impl("rosa_qkv_extend", &torch_rosa_qkv_extend<int64_t, int64_t>);
}
