#include <map>
#include <tuple>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <iostream>

#if __cplusplus < 202002L
    #ifdef _MSC_VER
        #include <intrin.h>
        #define POPCOUNT(x) __popcnt(x)
    #else
        #define POPCOUNT(x) __builtin_popcount(x)
    #endif
#else
    #include <bit>
    #define POPCOUNT(x) std::popcount(static_cast<unsigned int>(x))
#endif

template<typename K, typename V, typename P>
struct rosa_trit_state_t {
    P length;
    P endpos;
    P suffix_link;
    std::vector<std::pair<K, V>> transitions;

    rosa_trit_state_t() : endpos(-1), length(0), suffix_link(-1) {}
};


template<typename K, typename V, typename P>
class rosa_trit {
public:
    rosa_trit() : last_q_(0), last_k_(0) {}
    const std::vector<V>& values() const { return values_; }

    V append(K q, K m, K k, V v, V u) {
        P i = update_query_(q, m, last_q_);
        update_key_value_(k, v);
        return i != -1 ? values_[i + 1] : u;
    }

    P probe(K q, K m, K k, V v, V u, P& endpos, P& length) {
        P i = update_query_(q, m, last_q_);
        update_key_value_(k, v);

        endpos = i;
        length = states_[last_q_].length;
        
        return i != -1 ? values_[i + 1] : u;
    }

private:
    P update_key_value_(K k, K v) {
        if (states_.empty()) states_.emplace_back();

        P i = values_.size();
        values_.emplace_back(v);

        P r = states_.size();
        states_.emplace_back();
        states_[r].length = states_[last_k_].length + 1;

        P p = last_k_;
        while (p != -1) {
            auto it = std::find_if(
                states_[p].transitions.begin(), states_[p].transitions.end(),
                [&](const auto& x) { return x.first == k; }
            );

            if (it != states_[p].transitions.end()) {
                break;
            }

            states_[p].transitions.emplace_back(k, r);
            p = states_[p].suffix_link;
        }

        if (p == -1) {
            states_[r].suffix_link = 0;
        } else {
            P c = std::find_if(
                states_[p].transitions.begin(), states_[p].transitions.end(),
                [&](const auto& x) { return x.first == k; }
            )->second;

            if (states_[p].length + 1 == states_[c].length) {
                states_[r].suffix_link = c;
            } else {
                P u = states_.size();
                states_.emplace_back(states_[c]);
                states_[u].length = states_[p].length + 1;
                states_[c].suffix_link = u;
                states_[r].suffix_link = u;

                while (p != -1) {
                    auto it = std::find_if(
                        states_[p].transitions.begin(), states_[p].transitions.end(),
                        [&](const auto& x) { return x.first == k; }
                    );

                    if (it != states_[p].transitions.end() && it->second == c) {
                        it->second = u;
                        p = states_[p].suffix_link;
                    } else {
                        break;
                    }
                }
            }
        }

        last_k_ = r;

        while (r != -1) {
            states_[r].endpos = i;
            r = states_[r].suffix_link;
        }
        
        return r;
    }

    P update_query_(K q, K m, P& s) {
        if (states_.empty()) states_.emplace_back();

        P j = -1;

        P r = s;
        while (r != -1) {
            auto it = std::find_if(
                states_[r].transitions.begin(), states_[r].transitions.end(),
                [&](const auto& x) { return (x.first & m) == (q & m); }
            );

            if (it != states_[r].transitions.end()) {
                break;
            }

            r = states_[r].suffix_link;
        }

        if (r == -1) {
            s = 0;
        } else {
            int x = -1;
            for (auto it = states_[r].transitions.rbegin(); it != states_[r].transitions.rend(); ++it) {
                if ((it->first & m) != (q & m)) {
                    continue;
                }

                int y = POPCOUNT(it->first ^ q);
                if (x < 0 || y < x) {
                    x = y;
                    s = it->second;
                }
            }

            r = s;
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
    std::vector<rosa_trit_state_t<K, V, P>> states_;
    P last_q_;
    P last_k_;
};


#include <torch/extension.h>

template<typename K, typename V, typename P>
torch::Tensor& torch_rosa_trit_init(torch::Tensor& ctx) {
    int64_t B = ctx.numel();
    auto ctx_a = ctx.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_trit<K, V, P>*)ctx_a[i];
        
        if (r) {
            delete r;
        }

        r = new rosa_trit<K, V, P>();

        ctx_a[i] = (int64_t)r;
    }

    return ctx;
}

template<typename K, typename V, typename P>
torch::Tensor& torch_rosa_trit_free(torch::Tensor& ctx) {
    int64_t B = ctx.numel();
    auto ctx_a = ctx.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_trit<K, V, P>*)ctx_a[i];
        
        if (r) {
            delete r;
        }

        ctx_a[i] = 0;
    }

    return ctx;
}

template<typename K, typename V, typename P>
torch::Tensor torch_rosa_trit_update(const torch::Tensor& ctx, const torch::Tensor& q, const torch::Tensor& m, const torch::Tensor& k, const torch::Tensor& v, int64_t u) {
    assert(ctx.numel() == q.size(0));

    int64_t B = q.size(0);
    int64_t N = q.size(1);

    auto q_a = q.accessor<K, 2>();
    auto m_a = m.accessor<K, 2>();
    auto k_a = k.accessor<K, 2>();
    auto v_a = v.accessor<V, 2>();

    auto out = torch::empty({B, N}, v.options());

    auto ctx_a = ctx.accessor<int64_t, 1>();
    auto out_a = out.accessor<V, 2>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_trit<K, V, P>*)ctx_a[i];
        if (!r) {
            throw std::runtime_error("rosa context is null");
        }

        for (int64_t t = 0; t < N; ++t) {
            out_a[i][t] = r->append(q_a[i][t], m_a[i][t], k_a[i][t], v_a[i][t], static_cast<V>(u));
        }
    }

    return out;
}


template<typename K, typename V, typename P>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> torch_rosa_trit_inspect(const torch::Tensor& ctx, const torch::Tensor& q, const torch::Tensor& m, const torch::Tensor& k, const torch::Tensor& v, int64_t u) {
    assert(ctx.numel() == q.size(0));

    int64_t B = q.size(0);
    int64_t N = q.size(1);

    auto q_a = q.accessor<K, 2>();
    auto m_a = m.accessor<K, 2>();
    auto k_a = k.accessor<K, 2>();
    auto v_a = v.accessor<V, 2>();

    auto out = torch::empty({B, N}, v.options());

    auto ctx_a = ctx.accessor<int64_t, 1>();
    auto out_a = out.accessor<V, 2>();

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(v.device());
    
    auto endpos = torch::empty({B, N}, options);
    auto endpos_a = endpos.accessor<V, 2>();

    auto length = torch::empty({B, N}, options);
    auto length_a = length.accessor<V, 2>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        rosa_trit<K, V, P> r;
        for (int64_t t = 0; t < N; ++t) {
            P _endpos = -1;
            P _length = 0;
            out_a[i][t] = r.probe(q_a[i][t], m_a[i][t], k_a[i][t], v_a[i][t], static_cast<V>(u), _endpos, _length);

            endpos_a[i][t] = _endpos;
            length_a[i][t] = _length;
        }
    }

    return {out, endpos, length};
}

template<typename K, typename V, typename P>
torch::Tensor torch_rosa_trit_forward(const torch::Tensor& q, const torch::Tensor& m, const torch::Tensor& k, const torch::Tensor& v, int64_t u) {
    int64_t B = q.size(0);
    int64_t N = q.size(1);

    auto q_a = q.accessor<K, 2>();
    auto m_a = m.accessor<K, 2>();
    auto k_a = k.accessor<K, 2>();
    auto v_a = v.accessor<V, 2>();

    auto out = torch::empty({B, N}, v.options());
    auto out_a = out.accessor<V, 2>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        rosa_trit<K, V, P> r;
        for (int64_t t = 0; t < N; ++t) {
            out_a[i][t] = r.append(q_a[i][t], m_a[i][t], k_a[i][t], v_a[i][t], static_cast<V>(u));
        }
    }

    return out;
}


TORCH_LIBRARY_IMPL(rosa_cpp, CPU, m) {
    m.impl("rosa_trit_init", &torch_rosa_trit_init<int64_t, int64_t, int64_t>);
    m.impl("rosa_trit_free", &torch_rosa_trit_free<int64_t, int64_t, int64_t>);
    m.impl("rosa_trit_update", &torch_rosa_trit_update<int64_t, int64_t, int64_t>);
    m.impl("rosa_trit_inspect", &torch_rosa_trit_inspect<int64_t, int64_t, int64_t>);

    m.impl("rosa_trit_forward", &torch_rosa_trit_forward<int64_t, int64_t, int64_t>);
}
