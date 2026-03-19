#include <map>
#include <tuple>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>


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

using rosa_cache = std::vector<rosa_sam<int64_t, int64_t, int64_t>>;

template<typename K, typename V>
void rosa_cache_update_kernel(
    const int64_t B, const int64_t nqk, const int64_t nvv, const V u,
    rosa_cache** cache, const int64_t* batch,
    const K* query, const K* key, const V* value,
    const K* query_trigger, const K* key_trigger,
    V* output, int64_t* endpos, int64_t* length
) {
    std::vector<int64_t> offset(B + 1);
    for (int64_t b = 0; b < B; ++b) {
        if (!cache[b]) {
            throw std::runtime_error("rosa cache is not initialized");
        } else if (cache[b]->size() != nqk) {
            throw std::runtime_error("rosa cache size does not match nqk");
        }

        offset[b + 1] = offset[b] + batch[b];
    }

    const int64_t n_rep = nqk / nvv;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t h = 0; h < nqk; ++h) {
        for (int64_t b = 0; b < B; ++b) {
            auto& r = cache[b]->at(h);
            for (int64_t t = 0; t < batch[b]; ++t) {
                int64_t qk_idx = h * offset[B] + offset[b] + t;
                int64_t vv_idx = h / n_rep * offset[B] + offset[b] + t;

                uint64_t xq = static_cast<uint64_t>(query[qk_idx]);
                uint64_t xk = static_cast<uint64_t>(key[qk_idx]);
                uint64_t xv = static_cast<uint64_t>(value[vv_idx]);

                uint64_t mq = -1;
                if (query_trigger) {
                    mq = static_cast<uint64_t>(query_trigger[qk_idx]);
                }

                uint64_t mk = -1;
                if (key_trigger) {
                    mk = static_cast<uint64_t>(key_trigger[qk_idx]);
                }

                uint64_t uv = static_cast<uint64_t>(u);

                int64_t pos = -1;
                int64_t len = 0;

                V out = r.update(xq, xk, xv, mq, mk, uv, pos, len);

                output[qk_idx] = static_cast<V>(out);
                endpos[qk_idx] = pos;
                length[qk_idx] = len;
            }
        }
    }
}


#include <torch/extension.h>

#define DISPATCH_KEY_TYPES(TYPE, NAME, ...) \
    do { \
        switch (TYPE) { \
            case at::ScalarType::Char:  { using key_t = int8_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Byte:  { using key_t = uint8_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Short:  { using key_t = int16_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::UInt16:  { using key_t = uint16_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Int:  { using key_t = int32_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::UInt32:  { using key_t = uint32_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Long:  { using key_t = int64_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::UInt64:  { using key_t = uint64_t;  (__VA_ARGS__)(); break; } \
            default: TORCH_CHECK(false, #NAME, " not implemented for '", toString(TYPE), "'"); \
        } \
    } while (false)

#define DISPATCH_VALUE_TYPES(TYPE, NAME, ...) \
    do { \
        switch (TYPE) { \
            case at::ScalarType::Char:  { using value_t = int8_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Byte:  { using value_t = uint8_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Short:  { using value_t = int16_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::UInt16:  { using value_t = uint16_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Int:  { using value_t = int32_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::UInt32:  { using value_t = uint32_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::Long:  { using value_t = int64_t;  (__VA_ARGS__)(); break; } \
            case at::ScalarType::UInt64:  { using value_t = uint64_t;  (__VA_ARGS__)(); break; } \
            default: TORCH_CHECK(false, #NAME, " not implemented for '", toString(TYPE), "'"); \
        } \
    } while (false)

#define CHECK_QK_TENSOR(x, nqk, ntk) \
    do { \
        TORCH_CHECK(x.dim() == 2, #x, " must be 2D tensor"); \
        TORCH_CHECK(x.size(0) == nqk, #x, " must have size(0) equal to nqk"); \
        TORCH_CHECK(x.size(1) == ntk, #x, " must have size(1) equal to ntk"); \
        TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous"); \
    } while (false)

#define CHECK_VV_TENSOR(x, nvv, ntk) \
    do { \
        TORCH_CHECK(x.dim() == 2, #x, " must be 2D tensor"); \
        TORCH_CHECK(x.size(0) == nvv, #x, " must have size(0) equal to nvv"); \
        TORCH_CHECK(x.size(1) == ntk, #x, " must have size(1) equal to ntk"); \
        TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous"); \
    } while (false)


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> torch_rosa_cache_update(
    const torch::Tensor& cache,
    const torch::Tensor& batch,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const c10::optional<torch::Tensor>& query_trigger,
    const c10::optional<torch::Tensor>& key_trigger,
    int64_t u
) {
    auto cache_ = cache.accessor<int64_t, 1>();
    auto batch_ = batch.accessor<int64_t, 1>();

    int64_t B = cache.numel();
    int64_t nqk = query.size(0);
    int64_t nvv = value.size(0);

    if (batch.numel() != cache.numel()) {
        throw std::runtime_error("batch size must be equal to cache size");
    }

    if (query.size(0) % value.size(0) != 0) {
        throw std::runtime_error("query heads must be divisible by value heads");
    }

    int64_t ntk = 0; // total tokens
    std::vector<rosa_cache*> cache_vec(B); // cache pointer vector
    std::vector<int64_t> batch_vec(B); // batch size vector
    for (int64_t i = 0; i < B; ++i) {
        ntk += batch_[i];
        cache_vec[i] = (rosa_cache*)cache_[i];
        batch_vec[i] = batch_[i];
    }

    CHECK_QK_TENSOR(query, nqk, ntk);
    CHECK_QK_TENSOR(key, nqk, ntk);
    CHECK_VV_TENSOR(value, nvv, ntk);

    if (query_trigger.has_value()) { CHECK_QK_TENSOR(query_trigger.value(), nqk, ntk); }
    if (key_trigger.has_value()) { CHECK_QK_TENSOR(key_trigger.value(), nqk, ntk); }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(value.device());

    auto output = torch::empty({nqk, ntk}, value.options());
    auto endpos = torch::empty({nqk, ntk}, options);
    auto length = torch::empty({nqk, ntk}, options);

    DISPATCH_KEY_TYPES(
        key.scalar_type(),
        "rosa_cache_update_",
        [&] {
            DISPATCH_VALUE_TYPES(
                value.scalar_type(),
                "rosa_cache_update_",
                [&] {
                    const key_t* query_ptr = query.data_ptr<key_t>();
                    const key_t* key_ptr = key.data_ptr<key_t>();
                    const value_t* value_ptr = value.data_ptr<value_t>();

                    const key_t* query_trigger_ptr = nullptr;
                    if (query_trigger.has_value()) {
                        query_trigger_ptr = query_trigger.value().data_ptr<key_t>();
                    }
                    
                    const key_t* key_trigger_ptr = nullptr;
                    if (key_trigger.has_value()) {
                        key_trigger_ptr = key_trigger.value().data_ptr<key_t>();
                    }

                    value_t* output_ptr = output.data_ptr<value_t>();
                    int64_t* endpos_ptr = endpos.data_ptr<int64_t>();
                    int64_t* length_ptr = length.data_ptr<int64_t>();

                    rosa_cache_update_kernel<key_t, value_t>(
                        B, nqk, nvv, static_cast<value_t>(u),
                        cache_vec.data(), batch_vec.data(),
                        query_ptr, key_ptr, value_ptr,
                        query_trigger_ptr, key_trigger_ptr,
                        output_ptr, endpos_ptr, length_ptr
                    );
                }
            );
        }
    );

    // for (int64_t i = 0; i < B; ++i) {
    //     cache_[i] = (int64_t)cache_vec[i];
    // }

    return {output, endpos, length};
}

void torch_rosa_cache_create(torch::Tensor& cache, int64_t num_heads) {
    auto cache_ = cache.accessor<int64_t, 1>();
    int64_t B = cache.numel();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_cache*)cache_[i];
        if (!r) {
            r = new rosa_cache();
            r->resize(num_heads);
        } else if (r->size() != num_heads) {
            throw std::runtime_error("cache size must be equal to num_heads");
        }
        cache_[i] = (int64_t)r;
    }
}

void torch_rosa_cache_delete(torch::Tensor& cache) {
    auto cache_ = cache.accessor<int64_t, 1>();
    int64_t B = cache.numel();

    #pragma omp parallel for
    for (int64_t i = 0; i < B; ++i) {
        auto r = (rosa_cache*)cache_[i];
        if (r) {
            delete r;
        }
        cache_[i] = 0;
    }
}


TORCH_LIBRARY_IMPL(rosa_soft, CPU, m) {
    m.impl("rosa_cache_update", &torch_rosa_cache_update);
    m.impl("rosa_cache_create", &torch_rosa_cache_create);
    m.impl("rosa_cache_delete", &torch_rosa_cache_delete);
}
