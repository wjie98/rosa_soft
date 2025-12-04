#include <map>
#include <tuple>
#include <utility>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>


#ifdef _MSC_VER
#include <intrin.h>
template <typename T> int popcount(T n) { return (int)__popcnt64(n); }
#else
template <typename T> int popcount(T n) { return __builtin_popcountll(n); }
#endif


struct rosa_gss_config {
    size_t reservoir_cap;
    size_t suffix_weight;
    size_t walk_depth;

    rosa_gss_config() : reservoir_cap(16), suffix_weight(2), walk_depth(1) {}
};


template<typename K, typename V, typename P = int64_t>
struct rosa_gss_state_t {
    P length;
    P suffix_link;
    std::map<K, P> transitions;

    std::vector<P> reservoir; 
    size_t total_occurrences;
    size_t write_ptr;

    rosa_gss_state_t() 
        : length(0), suffix_link(-1), total_occurrences(0), write_ptr(0) {}

    void add_pos(P pos, size_t capacity, std::mt19937& rng) {
        size_t recency_zone = capacity / 2;
        recency_zone = recency_zone < 1 ? 1 : recency_zone;

        size_t global_zone_size = capacity - recency_zone;
        global_zone_size = global_zone_size < 0 ? 0 : global_zone_size;

        capacity = recency_zone + global_zone_size;
        if (reservoir.capacity() < capacity) {
            reservoir.reserve(capacity);
        }

        ++ total_occurrences;

        if (reservoir.size() < recency_zone) {
            reservoir.push_back(pos);
        } else {
            reservoir[write_ptr % recency_zone] = pos;
            
            if (global_zone_size > 0) {
                if (reservoir.size() < recency_zone + global_zone_size) {
                    reservoir.push_back(pos);
                } else {
                    std::uniform_int_distribution<size_t> dist(0, total_occurrences - 1 - recency_zone);
                    size_t k = dist(rng);
                    if (k < global_zone_size) {
                        reservoir[recency_zone + k] = pos;
                    }
                }
            }
        }

        ++ write_ptr;
    }

    bool find_query_endpos(P& ret, size_t capacity) const {
        if (write_ptr > 0) {
            size_t recency_zone = capacity / 2;
            recency_zone = recency_zone < 1 ? 1 : recency_zone;
            ret = reservoir[(write_ptr - 1) % recency_zone];
            return true;
        }

        return false;
    }

    template<typename F>
    void update_endpos_table(std::map<P, F>& table, size_t dist) {
        for (auto it = reservoir.begin(); it != reservoir.end(); ++it) {
            F quality = length + std::exp(-static_cast<F>(dist));
            auto p = table.find(*it);
            if (p == table.end()) {
                table[*it] = quality;
            } else if (p->second < quality) {
                p->second = quality;
            }
        }
    }
};


template<typename K, typename V, typename P = int64_t, typename F = double>
class rosa_gss {
public:
    rosa_gss() : last_k_(0), last_q_(0) {}

    rosa_gss(const rosa_gss_config& config) : last_k_(0), last_q_(0), config_(config) {}

    V append(K q, K k, V v, V u) {
        P j = update_query_(q, last_q_);
        update_key_value_(k, v, rng_);
        return j != -1 ? values_[j+1] : u;
    }

    V append_with_sampling(
        K q, K k, V v, V u,
        std::vector<P>& indices,
        std::vector<F>& quality,
        size_t num_samples, F tau,
        size_t index_offset
    ) {
        P j = random_walk_query_(indices, quality, q, last_q_, num_samples,  tau, rng_, index_offset);
        update_key_value_(k, v, rng_);
        return j != -1 ? values_[j+1] : u;
    }

private:
    std::vector<V> values_;
    std::vector<rosa_gss_state_t<K, V, P>> states_;
    
    P last_k_;
    P last_q_;

    std::mt19937 rng_;
    rosa_gss_config config_;

    P random_walk_query_(
        std::vector<P>& indices,
        std::vector<F>& quality,
        K q, P& s,
        size_t num_samples, F tau,
        std::mt19937& rng, size_t index_offset
    ) {
        if (states_.empty()) states_.emplace_back();

        std::map<P, P> candidates;

        P r = s;
        P cur_depth = 0;
        P walk_dist = 0;

        // while (r != -1 && (cur_depth < config_.walk_depth || candidates.size() < num_samples)) {
        while (r != -1 && cur_depth < config_.walk_depth) {
            for (auto it = states_[r].transitions.begin(); it != states_[r].transitions.end(); ++it) {
                P dist = popcount(q ^ it->first) + walk_dist;

                auto p = candidates.find(it->second);
                if (p == candidates.end()) {
                    candidates[it->second] = dist;
                } else if (p->second > dist) {
                    p->second = dist;
                }
            }
            r = states_[r].suffix_link;
            walk_dist += config_.suffix_weight;
            ++ cur_depth;
        }

        std::map<P, F> duplicated_endpos;

        if (candidates.size() <= num_samples) {
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {
                states_[it->first].update_endpos_table(duplicated_endpos, it->second);
            }
        } else {
            std::uniform_real_distribution<F> uniform(0.0, 1.0);

            std::vector<std::pair<F, P>> scores;
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {
                F logit = -static_cast<F>(it->second) / tau;

                F u = uniform(rng);
                u = std::max(u, 1e-7); // avoid log(0)
                F gumbel_noise = -std::log(-std::log(u));

                scores.emplace_back(logit + gumbel_noise, it->first);
            }

            std::partial_sort(scores.begin(), scores.begin() + num_samples, scores.end(), std::greater<>());
            for (size_t k = 0; k < num_samples; ++k) {
                auto it = candidates.find(scores[k].second);
                states_[it->first].update_endpos_table(duplicated_endpos, it->second);
            }
        }

        P j = update_query_(q, s);
        if (j != -1) {
            auto it = duplicated_endpos.find(j);
            if (it != duplicated_endpos.end()) {
                duplicated_endpos.erase(it);
            }
        }

        std::vector<std::pair<P, F>> sampled_endpos;
        sampled_endpos.reserve(duplicated_endpos.size());

        for (auto it = duplicated_endpos.begin(); it != duplicated_endpos.end(); ++it) {
            sampled_endpos.emplace_back(it->first, it->second);
        }

        std::shuffle(sampled_endpos.begin(), sampled_endpos.end(), rng);

        if (j != -1) {
            indices.emplace_back(j + index_offset);
            quality.emplace_back(static_cast<F>(states_[j].length + 1));
        }

        for (size_t k = 0; k < sampled_endpos.size() && k < num_samples; ++k) {
            indices.emplace_back(sampled_endpos[k].first + index_offset);
            quality.emplace_back(sampled_endpos[k].second);
        }

        return j;
    }

    P update_key_value_(K k, V v, std::mt19937& rng) {
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

        while (r != -1) {
            states_[r].add_pos(i, config_.reservoir_cap, rng);
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
                if (states_[r].length > 0 && states_[r].find_query_endpos(j, config_.reservoir_cap)) {
                    break;
                }
                r = states_[r].suffix_link;
            }
        }
        
        return j;
    }
};


#include <torch/extension.h>

template<typename K, typename V, typename P>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> torch_rosa_gss_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t u,
    int64_t num_samples, double tau
) {
    int64_t B = q.size(0);
    int64_t N = q.size(1);

    auto indices_options = torch::TensorOptions().dtype(torch::kInt64).device(q.device());
    auto quality_options = torch::TensorOptions().dtype(torch::kFloat).device(q.device());

    auto q_a = q.accessor<K, 2>();
    auto k_a = k.accessor<K, 2>();
    auto v_a = v.accessor<V, 2>();

    auto out = torch::empty({B, N}, v.options());
    auto out_a = out.accessor<V, 2>();
    
    std::vector<std::vector<P>> indices_vec(B);
    std::vector<std::vector<double>> quality_vec(B);

    auto indptr = torch::empty({B * N + 1}, indices_options);
    auto indptr_a = indptr.accessor<int64_t, 1>();
    
    indptr_a[0] = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < B; ++i) {
        rosa_gss<K, V, P, double> r;
        
        auto& indices = indices_vec[i];
        auto& quality = quality_vec[i];

        for (int64_t t = 0; t < N; ++t) {
            size_t s = indices.size();
            
            out_a[i][t] = r.append_with_sampling(
                q_a[i][t], k_a[i][t], v_a[i][t], static_cast<V>(u),
                indices, quality,
                num_samples, tau,
                i * N
            );
            
            indptr_a[i * N + t + 1] = static_cast<int64_t>(indices.size() - s);
        }
    }

    indptr.cumsum_(0);

    int64_t num_values = 0;
    for (size_t i = 0; i < indices_vec.size(); ++i) {
        num_values += indices_vec[i].size();
    }

    auto indices = torch::empty({num_values}, indices_options);
    auto quality = torch::empty({num_values}, quality_options);

    auto indices_a = indices.accessor<int64_t, 1>();
    auto quality_a = quality.accessor<float, 1>();

    int64_t write_ptr = 0;
    for (int64_t i = 0; i < B; ++i) {
        for (int64_t j = 0; j < indices_vec[i].size(); ++j) {
            indices_a[write_ptr] = static_cast<int64_t>(indices_vec[i][j]);
            quality_a[write_ptr] = static_cast<float>(quality_vec[i][j]);
            ++ write_ptr;
        }
    }

    return {out, indptr, indices, quality};
}


TORCH_LIBRARY_IMPL(rosa_cpp, CPU, m) {
    m.impl("rosa_gss_forward", &torch_rosa_gss_forward<int64_t, int64_t, int64_t>);
}