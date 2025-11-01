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

    T append_for_seq(T x, T v, T u) {
        P i = -1;
        return append_for_seq(x, v, u, i);
    }

    T append_for_seq(T x, T v, T u, P& i) {
        update_key_value_(x, v);

        i = update_query_(x, last_q_);
        T ret = i != -1 ? values_[i + 1] : u;

        update_endpos_();
        return ret;
    }

    T append_for_qkv(T q, T k, T v, T u) {
        P i = -1;
        return append_for_qkv(q, k, v, u, i);
    }

    T append_for_qkv(T q, T k, T v, T u, P& i) {
        update_key_value_(k, v);
        update_endpos_();

        i = update_query_(q, last_q_);
        return i != -1 ? values_[i] : u;
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



template<typename T, typename P>
void rosa_seq_fwd_inner(T* y_ptr, const T* x_ptr, const T* v_ptr, size_t num_tokens, T u) {
    rosa_sam<T, P> r;
    for (size_t i = 0; i < num_tokens; ++i) {
        y_ptr[i] = r.append_for_seq(x_ptr[i], v_ptr[i], u);
    }
}

template<typename T, typename P>
void rosa_seq_fwd(T* y_ptr, const T* x_ptr, const T* v_ptr, size_t num_tokens, size_t batch_size, T u) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; ++i) {
        rosa_seq_fwd_inner<T, P>(y_ptr + i * num_tokens, x_ptr + i * num_tokens, v_ptr + i * num_tokens, num_tokens, u);
    }
}

template<typename T, typename P>
void rosa_qkv_fwd_inner(T* o_ptr, const T* q_ptr, const T* k_ptr, const T* v_ptr, size_t num_tokens, T u) {
    rosa_sam<T, P> r;
    for (size_t i = 0; i < num_tokens; ++i) {
        o_ptr[i] = r.append_for_qkv(q_ptr[i], k_ptr[i], v_ptr[i], u);
    }
}

template<typename T, typename P>
void rosa_qkv_fwd(T* o_ptr, const T* q_ptr, const T* k_ptr, const T* v_ptr, size_t num_tokens, size_t batch_size, T u) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; ++i) {
        rosa_qkv_fwd_inner<T, P>(o_ptr + i * num_tokens, q_ptr + i * num_tokens, k_ptr + i * num_tokens, v_ptr + i * num_tokens, num_tokens, u);
    }
}

template<typename T, typename F>
F rosa_phi_from_idx(T y, const F* dy_ptr, const F* e0_ptr, const F* e1_ptr, size_t num_bits) {
    F phi = 0;
    for (size_t i = 0; i < num_bits; ++i) {
        if ((y >> i) & 1) {
            phi += dy_ptr[i] * e1_ptr[i];
        } else {
            phi += dy_ptr[i] * e0_ptr[i];
        }
    }
    return phi;
}

template<typename T, typename P, typename F>
void rosa_seq_bwd_x_inner(
    F* dx_ptr, const F* dy_ptr,
    const F* x0_ptr, const F* e0_ptr, const F* e1_ptr,
    const T* x_ptr, const T* v_ptr, const T* y_ptr,
    size_t num_tokens, size_t num_x_bits, size_t num_v_bits,
    size_t start, size_t end,
    T u, F tau
) {
    F base_phi = 0.0;
    for (size_t i = 0; i < num_tokens; ++i) {
        base_phi += rosa_phi_from_idx(y_ptr[i], dy_ptr + i * num_v_bits, e0_ptr, e1_ptr, num_v_bits);
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t s = start; s < end; ++s) {
        for (size_t b = 0; b < num_x_bits; ++b) {
            rosa_sam<T, P> r;

            F phi_pos = base_phi;
            F phi_neg = 0;
            for (size_t i = 0; i < num_tokens; ++i) {
                T x = x_ptr[i];
                T v = v_ptr[i];
                
                if (i == s) {
                    x ^= (1 << b);
                }

                T y = r.append_for_seq(x, v, u);
                phi_neg += rosa_phi_from_idx(y, dy_ptr + i * num_v_bits, e0_ptr, e1_ptr, num_v_bits);
            }

            F xkb = x0_ptr[s * num_x_bits + b];

            F mag = xkb >= 0 ? xkb : -xkb;
            mag = mag > tau ? mag: tau;

            F g = (phi_pos - phi_neg) / (mag * 2); // TODO: check this
            dx_ptr[s * num_x_bits + b] = xkb > 0 ? g : -g;
        }
    }
}

template<typename T, typename P, typename F>
void rosa_seq_bwd_v_inner(
    F* dv_ptr, const F* dy_ptr,
    const F* e0_ptr, const F* e1_ptr,
    const T* x_ptr, const T* v_ptr,
    size_t num_tokens, size_t num_v_bits, T u, F tau
) {
    rosa_sam<T, P> r;
    for (size_t i = 0; i < num_tokens; ++i) {
        T x = x_ptr[i];
        T v = v_ptr[i];
        T p = -1;
        T y = r.append_for_seq(x, v, u, p);

        if (p == -1) {
            continue;
        }

        for (size_t b = 0; b < num_v_bits; ++b) {
            if ((y >> b) & 1) {
                dv_ptr[p * num_v_bits + b] += dy_ptr[i * num_v_bits + b] * e1_ptr[b];
            } else {
                dv_ptr[p * num_v_bits + b] += dy_ptr[i * num_v_bits + b] * e0_ptr[b];
            }
        }
    }
}

template<typename T, typename P, typename F>
void rosa_seq_bwd(
    F* dx_ptr, F* dv_ptr, const F* dy_ptr,
    const F* x0_ptr, const F* e0_ptr, const F* e1_ptr,
    const T* x_ptr, const T* v_ptr, const T* y_ptr,
    size_t num_tokens, size_t num_x_bits, size_t num_v_bits,
    size_t batch_size, size_t num_heads,
    T u, F tau, size_t BLOCK_SIZE
) {
    #pragma omp parallel for collapse(1) schedule(dynamic)
    for (size_t b = 0; b < batch_size * num_heads; ++b) {
        rosa_seq_bwd_v_inner<T, P, F>(
            dv_ptr + b * num_tokens * num_v_bits,
            dy_ptr + b * num_tokens * num_v_bits,
            e0_ptr + (b % num_heads) * num_v_bits,
            e1_ptr + (b % num_heads) * num_v_bits,
            x_ptr + b * num_tokens,
            v_ptr + b * num_tokens,
            num_tokens, num_v_bits, u, tau
        );
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t b = 0; b < batch_size * num_heads; ++b) {
        for (size_t s = 0; s < num_tokens; s += BLOCK_SIZE) {
            size_t t = s + BLOCK_SIZE;
            t = t > num_tokens ? num_tokens : t;
            rosa_seq_bwd_x_inner<T, P, F>(
                dx_ptr + b * num_tokens * num_x_bits,
                dy_ptr + b * num_tokens * num_v_bits,
                x0_ptr + b * num_tokens * num_x_bits,
                e0_ptr + (b % num_heads) * num_v_bits,
                e1_ptr + (b % num_heads) * num_v_bits,
                x_ptr + b * num_tokens,
                v_ptr + b * num_tokens,
                y_ptr + b * num_tokens,
                num_tokens, num_x_bits, num_v_bits,
                s, t, u, tau
            );
        }
    }
}


template<typename T, typename P, typename F>
void rosa_qkv_bwd_qk_inner(
    F* dq_ptr, F* dk_ptr, const F* dy_ptr,
    const F* q0_ptr, const F* k0_ptr, const F* e0_ptr, const F* e1_ptr,
    const T* q_ptr, const T* k_ptr, const T* v_ptr, const T* y_ptr,
    size_t num_tokens, size_t num_x_bits, size_t num_v_bits,
    size_t start, size_t end,
    T u, F tau
) {
    F base_phi = 0.0;
    for (size_t i = 0; i < num_tokens; ++i) {
        base_phi += rosa_phi_from_idx(y_ptr[i], dy_ptr + i * num_v_bits, e0_ptr, e1_ptr, num_v_bits);
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t s = start; s < end; ++s) {
        for (size_t b = 0; b < num_x_bits; ++b) {
            rosa_sam<T, P> r;

            F phi_pos = base_phi;
            F phi_neg = 0;
            for (size_t i = 0; i < num_tokens; ++i) {
                T q = q_ptr[i];
                T k = k_ptr[i];
                T v = v_ptr[i];
                
                if (i == s) {
                    q ^= (1 << b);
                }

                T y = r.append_for_qkv(q, k, v, u);
                phi_neg += rosa_phi_from_idx(y, dy_ptr + i * num_v_bits, e0_ptr, e1_ptr, num_v_bits);
            }

            F xkb = q0_ptr[s * num_x_bits + b];

            F mag = xkb >= 0 ? xkb : -xkb;
            mag = mag > tau ? mag: tau;

            F g = (phi_pos - phi_neg) / (mag * 2); // TODO: check this
            dq_ptr[s * num_x_bits + b] = xkb > 0 ? g : -g;
        }
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t s = start; s < end; ++s) {
        for (size_t b = 0; b < num_x_bits; ++b) {
            rosa_sam<T, P> r;

            F phi_pos = base_phi;
            F phi_neg = 0;
            for (size_t i = 0; i < num_tokens; ++i) {
                T q = q_ptr[i];
                T k = k_ptr[i];
                T v = v_ptr[i];
                
                if (i == s) {
                    k ^= (1 << b);
                }

                T y = r.append_for_qkv(q, k, v, u);
                phi_neg += rosa_phi_from_idx(y, dy_ptr + i * num_v_bits, e0_ptr, e1_ptr, num_v_bits);
            }

            F xkb = k0_ptr[s * num_x_bits + b];

            F mag = xkb >= 0 ? xkb : -xkb;
            mag = mag > tau ? mag: tau;

            F g = (phi_pos - phi_neg) / (mag * 2); // TODO: check this
            dk_ptr[s * num_x_bits + b] = xkb > 0 ? g : -g;
        }
    }
}


template<typename T, typename P, typename F>
void rosa_qkv_bwd_v_inner(
    F* dv_ptr, const F* dy_ptr,
    const F* e0_ptr, const F* e1_ptr,
    const T* q_ptr, const T* k_ptr, const T* v_ptr,
    size_t num_tokens, size_t num_v_bits, T u, F tau
) {
    rosa_sam<T, P> r;
    for (size_t i = 0; i < num_tokens; ++i) {
        T q = q_ptr[i];
        T k = k_ptr[i];
        T v = v_ptr[i];
        T p = -1;
        T y = r.append_for_qkv(q, k, v, u, p);

        if (p == -1) {
            continue;
        }

        for (size_t b = 0; b < num_v_bits; ++b) {
            if ((y >> b) & 1) {
                dv_ptr[p * num_v_bits + b] += dy_ptr[i * num_v_bits + b] * e1_ptr[b];
            } else {
                dv_ptr[p * num_v_bits + b] += dy_ptr[i * num_v_bits + b] * e0_ptr[b];
            }
        }
    }
}


template<typename T, typename P, typename F>
void rosa_qkv_bwd(
    F* dq_ptr, F* dk_ptr, F* dv_ptr, const F* dy_ptr,
    const F* q0_ptr, const F* k0_ptr, const F* e0_ptr, const F* e1_ptr,
    const T* q_ptr, const T* k_ptr, const T* v_ptr, const T* y_ptr,
    size_t num_tokens, size_t num_x_bits, size_t num_v_bits,
    size_t batch_size, size_t num_heads,
    T u, F tau, size_t BLOCK_SIZE
) {
    #pragma omp parallel for collapse(1) schedule(dynamic)
    for (size_t b = 0; b < batch_size * num_heads; ++b) {
        rosa_qkv_bwd_v_inner<T, P, F>(
            dv_ptr + b * num_tokens * num_v_bits,
            dy_ptr + b * num_tokens * num_v_bits,
            e0_ptr + (b % num_heads) * num_v_bits,
            e1_ptr + (b % num_heads) * num_v_bits,
            q_ptr + b * num_tokens,
            k_ptr + b * num_tokens,
            v_ptr + b * num_tokens,
            num_tokens, num_v_bits, u, tau
        );
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t b = 0; b < batch_size * num_heads; ++b) {
        for (size_t s = 0; s < num_tokens; s += BLOCK_SIZE) {
            size_t t = s + BLOCK_SIZE;
            t = t > num_tokens ? num_tokens : t;
            rosa_qkv_bwd_qk_inner<T, P, F>(
                dq_ptr + b * num_tokens * num_x_bits,
                dk_ptr + b * num_tokens * num_x_bits,
                dy_ptr + b * num_tokens * num_v_bits,
                q0_ptr + b * num_tokens * num_x_bits,
                k0_ptr + b * num_tokens * num_x_bits,
                e0_ptr + (b % num_heads) * num_v_bits,
                e1_ptr + (b % num_heads) * num_v_bits,
                q_ptr + b * num_tokens,
                k_ptr + b * num_tokens,
                v_ptr + b * num_tokens,
                y_ptr + b * num_tokens,
                num_tokens, num_x_bits, num_v_bits,
                s, t, u, tau
            );
        }
    }
}

#include <torch/extension.h>


template<typename T, typename P>
torch::Tensor torch_rosa_seq_fwd(const torch::Tensor& x, const torch::Tensor& v, int64_t u) {
    assert(x.dim() == 2);
    assert(v.dim() == 2);

    assert(x.is_contiguous());
    assert(v.is_contiguous());

    int64_t B = x.size(0);
    int64_t N = x.size(1);

    auto y = torch::empty_like(v);

    rosa_seq_fwd<T, P>(
        y.data_ptr<T>(),
        x.data_ptr<T>(),
        v.data_ptr<T>(),
        N, B, u
    );

    return y;
}

template<typename T, typename P>
torch::Tensor torch_rosa_qkv_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t u) {
    assert(q.dim() == 2);
    assert(k.dim() == 2);
    assert(v.dim() == 2);

    assert(q.is_contiguous());
    assert(k.is_contiguous());
    assert(v.is_contiguous());

    int64_t B = q.size(0);
    int64_t N = q.size(1);

    auto y = torch::empty_like(v);

    rosa_qkv_fwd<T, P>(
        y.data_ptr<T>(),
        q.data_ptr<T>(),
        k.data_ptr<T>(),
        v.data_ptr<T>(),
        N, B, u
    );

    return y;
}

template<typename T, typename P, typename F>
std::tuple<torch::Tensor, torch::Tensor> torch_rosa_seq_bwd(
    const torch::Tensor& dy,
    const torch::Tensor& x0, const torch::Tensor& e0, const torch::Tensor& e1,
    const torch::Tensor& x, const torch::Tensor& v, const torch::Tensor& y,
    int64_t u, double tau
) {
    size_t BLOCK_SIZE = 64;

    assert(dy.dim() == 4);
    assert(x0.dim() == 4);
    assert(e0.dim() == 1);
    assert(e1.dim() == 1);

    assert(x.dim() == 3);
    assert(v.dim() == 3);
    assert(y.dim() == 3);

    int64_t B = x.size(0);
    int64_t H = x.size(1);
    int64_t N = x.size(2);

    int64_t num_x_bits = x0.size(3);
    int64_t num_v_bits = dy.size(3);

    auto dx = torch::empty({B, H, N, num_x_bits}, dy.options());
    auto dv = torch::empty({B, H, N, num_v_bits}, dy.options());

    rosa_seq_bwd<T, P, F>(
        dx.data_ptr<F>(),
        dv.data_ptr<F>(),
        dy.data_ptr<F>(),
        x0.data_ptr<F>(),
        e0.data_ptr<F>(),
        e1.data_ptr<F>(),
        x.data_ptr<T>(),
        v.data_ptr<T>(),
        y.data_ptr<T>(),
        N, num_x_bits, num_v_bits,
        B, H, u, tau, BLOCK_SIZE
    );

    return {dx, dv};
}


template<typename T, typename P, typename F>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> torch_rosa_qkv_bwd(
    const torch::Tensor& dy, const torch::Tensor& q0, const torch::Tensor& k0,
    const torch::Tensor& e0, const torch::Tensor& e1,
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& y,
    int64_t u, double tau
) {
    size_t BLOCK_SIZE = 64;

    assert(dy.dim() == 4);
    assert(q0.dim() == 4);
    assert(k0.dim() == 4);
    assert(e0.dim() == 1);
    assert(e1.dim() == 1);

    assert(q.dim() == 3);
    assert(k.dim() == 3);
    assert(v.dim() == 3);
    assert(y.dim() == 3);

    int64_t B = q.size(0);
    int64_t H = q.size(1);
    int64_t N = q.size(2);

    int64_t num_x_bits = q0.size(3);
    int64_t num_v_bits = dy.size(3);

    auto dq = torch::empty({B, H, N, num_x_bits}, dy.options());
    auto dk = torch::empty({B, H, N, num_x_bits}, dy.options());
    auto dv = torch::empty({B, H, N, num_v_bits}, dy.options());

    rosa_qkv_bwd<T, P, F>(
        dq.data_ptr<F>(),
        dk.data_ptr<F>(),
        dv.data_ptr<F>(),
        dy.data_ptr<F>(),
        q0.data_ptr<F>(),
        k0.data_ptr<F>(),
        e0.data_ptr<F>(),
        e1.data_ptr<F>(),
        q.data_ptr<T>(),
        k.data_ptr<T>(),
        v.data_ptr<T>(),
        y.data_ptr<T>(),
        N, num_x_bits, num_v_bits,
        B, H, u, tau, BLOCK_SIZE
    );

    return {dq, dk, dv};
}


TORCH_LIBRARY(torch_rosa, m) {
    m.def("rosa_seq_fwd(Tensor x, Tensor v, int u) -> Tensor");
    m.def("rosa_seq_bwd(Tensor dy, Tensor x0, Tensor e0, Tensor e1, Tensor x, Tensor v, Tensor y, int u, float tau) -> (Tensor, Tensor)");
    m.def("rosa_qkv_fwd(Tensor q, Tensor k, Tensor v, int u) -> Tensor");
    m.def("rosa_qkv_bwd(Tensor dy, Tensor q0, Tensor k0, Tensor e0, Tensor e1, Tensor q, Tensor k, Tensor v, Tensor y, int u, float tau) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torch_rosa, CPU, m) {
    m.impl("rosa_seq_fwd", &torch_rosa_seq_fwd<int64_t, int64_t>);
    m.impl("rosa_seq_bwd", &torch_rosa_seq_bwd<int64_t, int64_t, float>);
    m.impl("rosa_qkv_fwd", &torch_rosa_qkv_fwd<int64_t, int64_t>);
    m.impl("rosa_qkv_bwd", &torch_rosa_qkv_bwd<int64_t, int64_t, float>);
}
