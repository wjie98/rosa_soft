#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct SamStats {
    int64_t states = 0;
    int64_t edges = 0;
    int64_t values = 0;
};

enum class Backend {
    Compact,
    Map,
};

class ISam {
public:
    virtual ~ISam() = default;
    virtual uint8_t update(uint8_t query, uint8_t key, uint8_t value, int64_t& endpos) = 0;
    virtual SamStats stats() const = 0;
};

struct MapState {
    int64_t maxlen = 0;
    int64_t endpos = -1;
    int64_t suffix_link = -1;
    std::map<uint8_t, int64_t> transitions;
};

class MapSam final : public ISam {
public:
    uint8_t update(uint8_t query, uint8_t key, uint8_t value, int64_t& endpos) override {
        const int64_t pos = update_query(query);
        update_key_value(key, value);
        endpos = pos;
        return pos >= 0 ? values_[static_cast<size_t>(pos + 1)] : uint8_t{0};
    }

    SamStats stats() const override {
        int64_t edges = 0;
        for (const auto& state : states_) {
            edges += static_cast<int64_t>(state.transitions.size());
        }
        return {
            static_cast<int64_t>(states_.size()),
            edges,
            static_cast<int64_t>(values_.size()),
        };
    }

private:
    void ensure_root() {
        if (states_.empty()) {
            states_.emplace_back();
        }
    }

    void update_key_value(uint8_t key, uint8_t value) {
        ensure_root();

        const int64_t value_pos = static_cast<int64_t>(values_.size());
        values_.push_back(value);

        const int64_t next_state = static_cast<int64_t>(states_.size());
        states_.emplace_back();
        states_[static_cast<size_t>(next_state)].maxlen =
            states_[static_cast<size_t>(last_key_state_)].maxlen + 1;

        int64_t p = last_key_state_;
        while (p != -1) {
            auto& transitions = states_[static_cast<size_t>(p)].transitions;
            if (transitions.count(key) != 0) {
                break;
            }
            transitions[key] = next_state;
            p = states_[static_cast<size_t>(p)].suffix_link;
        }

        if (p == -1) {
            states_[static_cast<size_t>(next_state)].suffix_link = 0;
        } else {
            const int64_t child = states_[static_cast<size_t>(p)].transitions[key];
            if (states_[static_cast<size_t>(p)].maxlen + 1 == states_[static_cast<size_t>(child)].maxlen) {
                states_[static_cast<size_t>(next_state)].suffix_link = child;
            } else {
                const int64_t clone = static_cast<int64_t>(states_.size());
                states_.push_back(states_[static_cast<size_t>(child)]);
                states_[static_cast<size_t>(clone)].maxlen = states_[static_cast<size_t>(p)].maxlen + 1;
                states_[static_cast<size_t>(child)].suffix_link = clone;
                states_[static_cast<size_t>(next_state)].suffix_link = clone;

                while (p != -1) {
                    auto it = states_[static_cast<size_t>(p)].transitions.find(key);
                    if (it != states_[static_cast<size_t>(p)].transitions.end() && it->second == child) {
                        it->second = clone;
                        p = states_[static_cast<size_t>(p)].suffix_link;
                    } else {
                        break;
                    }
                }
            }
        }

        last_key_state_ = next_state;

        int64_t r = next_state;
        while (r != -1) {
            states_[static_cast<size_t>(r)].endpos = value_pos;
            r = states_[static_cast<size_t>(r)].suffix_link;
        }
    }

    int64_t update_query(uint8_t query) {
        ensure_root();

        int64_t r = last_query_state_;
        while (r != -1) {
            auto it = states_[static_cast<size_t>(r)].transitions.find(query);
            if (it != states_[static_cast<size_t>(r)].transitions.end()) {
                r = it->second;
                break;
            }
            r = states_[static_cast<size_t>(r)].suffix_link;
        }

        if (r == -1) {
            last_query_state_ = 0;
            return -1;
        }

        last_query_state_ = r;

        while (r != -1) {
            const auto& state = states_[static_cast<size_t>(r)];
            if (state.maxlen > 0 && state.endpos >= 0) {
                return state.endpos;
            }
            r = state.suffix_link;
        }

        return -1;
    }

    std::vector<uint8_t> values_;
    std::vector<MapState> states_;
    int64_t last_query_state_ = 0;
    int64_t last_key_state_ = 0;
};

struct CompactState {
    int32_t maxlen = 0;
    int32_t endpos = -1;
    int32_t suffix_link = -1;
    int32_t first_edge = -1;
};

struct CompactEdge {
    uint8_t key = 0;
    int32_t next_state = -1;
    int32_t next_edge = -1;
};

class CompactSam final : public ISam {
public:
    uint8_t update(uint8_t query, uint8_t key, uint8_t value, int64_t& endpos) override {
        const int32_t pos = update_query(query);
        update_key_value(key, value);
        endpos = static_cast<int64_t>(pos);
        return pos >= 0 ? values_[static_cast<size_t>(pos + 1)] : uint8_t{0};
    }

    SamStats stats() const override {
        return {
            static_cast<int64_t>(states_.size()),
            static_cast<int64_t>(edges_.size()),
            static_cast<int64_t>(values_.size()),
        };
    }

private:
    void ensure_root() {
        if (states_.empty()) {
            states_.emplace_back();
        }
    }

    int32_t find_transition(int32_t state, uint8_t key) const {
        int32_t edge = states_[static_cast<size_t>(state)].first_edge;
        while (edge != -1) {
            const CompactEdge& e = edges_[static_cast<size_t>(edge)];
            if (e.key == key) {
                return e.next_state;
            }
            edge = e.next_edge;
        }
        return -1;
    }

    void set_transition(int32_t state, uint8_t key, int32_t next_state) {
        int32_t edge = states_[static_cast<size_t>(state)].first_edge;
        while (edge != -1) {
            CompactEdge& e = edges_[static_cast<size_t>(edge)];
            if (e.key == key) {
                e.next_state = next_state;
                return;
            }
            edge = e.next_edge;
        }

        const int32_t new_edge = static_cast<int32_t>(edges_.size());
        edges_.push_back({
            key,
            next_state,
            states_[static_cast<size_t>(state)].first_edge,
        });
        states_[static_cast<size_t>(state)].first_edge = new_edge;
    }

    int32_t copy_edges(int32_t first_edge) {
        if (first_edge == -1) {
            return -1;
        }

        int32_t new_first = -1;
        int32_t prev = -1;
        int32_t edge = first_edge;
        while (edge != -1) {
            const CompactEdge& src = edges_[static_cast<size_t>(edge)];
            const uint8_t key = src.key;
            const int32_t next_state = src.next_state;
            const int32_t next_edge = src.next_edge;
            const int32_t dst = static_cast<int32_t>(edges_.size());
            edges_.push_back({key, next_state, -1});
            if (prev == -1) {
                new_first = dst;
            } else {
                edges_[static_cast<size_t>(prev)].next_edge = dst;
            }
            prev = dst;
            edge = next_edge;
        }
        return new_first;
    }

    void update_key_value(uint8_t key, uint8_t value) {
        ensure_root();
        constexpr size_t kMaxInt32 = static_cast<size_t>(std::numeric_limits<int32_t>::max());
        TORCH_CHECK(states_.size() < kMaxInt32, "ROSA SAM state count exceeded int32 range");
        TORCH_CHECK(values_.size() < kMaxInt32, "ROSA SAM value count exceeded int32 range");

        const int32_t value_pos = static_cast<int32_t>(values_.size());
        values_.push_back(value);

        const int32_t next_state = static_cast<int32_t>(states_.size());
        states_.emplace_back();
        states_[static_cast<size_t>(next_state)].maxlen =
            states_[static_cast<size_t>(last_key_state_)].maxlen + 1;

        int32_t p = last_key_state_;
        while (p != -1) {
            if (find_transition(p, key) != -1) {
                break;
            }
            set_transition(p, key, next_state);
            p = states_[static_cast<size_t>(p)].suffix_link;
        }

        if (p == -1) {
            states_[static_cast<size_t>(next_state)].suffix_link = 0;
        } else {
            const int32_t child = find_transition(p, key);
            if (states_[static_cast<size_t>(p)].maxlen + 1 == states_[static_cast<size_t>(child)].maxlen) {
                states_[static_cast<size_t>(next_state)].suffix_link = child;
            } else {
                const int32_t clone = static_cast<int32_t>(states_.size());
                states_.push_back(states_[static_cast<size_t>(child)]);
                states_[static_cast<size_t>(clone)].maxlen = states_[static_cast<size_t>(p)].maxlen + 1;
                states_[static_cast<size_t>(clone)].first_edge =
                    copy_edges(states_[static_cast<size_t>(child)].first_edge);
                states_[static_cast<size_t>(child)].suffix_link = clone;
                states_[static_cast<size_t>(next_state)].suffix_link = clone;

                while (p != -1) {
                    const int32_t dst = find_transition(p, key);
                    if (dst == child) {
                        set_transition(p, key, clone);
                        p = states_[static_cast<size_t>(p)].suffix_link;
                    } else {
                        break;
                    }
                }
            }
        }

        last_key_state_ = next_state;

        int32_t r = next_state;
        while (r != -1) {
            states_[static_cast<size_t>(r)].endpos = value_pos;
            r = states_[static_cast<size_t>(r)].suffix_link;
        }
    }

    int32_t update_query(uint8_t query) {
        ensure_root();

        int32_t r = last_query_state_;
        int32_t next = find_transition(r, query);
        while (r != -1 && next == -1) {
            r = states_[static_cast<size_t>(r)].suffix_link;
            next = r == -1 ? -1 : find_transition(r, query);
        }

        if (r == -1) {
            last_query_state_ = 0;
            return -1;
        }

        r = next;
        last_query_state_ = r;

        while (r != -1) {
            const CompactState& state = states_[static_cast<size_t>(r)];
            if (state.maxlen > 0 && state.endpos >= 0) {
                return state.endpos;
            }
            r = state.suffix_link;
        }

        return -1;
    }

    std::vector<uint8_t> values_;
    std::vector<CompactState> states_;
    std::vector<CompactEdge> edges_;
    int32_t last_query_state_ = 0;
    int32_t last_key_state_ = 0;
};

Backend parse_backend(const std::string& backend) {
    if (backend == "compact") {
        return Backend::Compact;
    }
    if (backend == "map") {
        return Backend::Map;
    }
    TORCH_CHECK(false, "unknown RosaRuntime backend: ", backend, " (expected 'compact' or 'map')");
}

const char* backend_name(Backend backend) {
    return backend == Backend::Compact ? "compact" : "map";
}

std::unique_ptr<ISam> make_sam(Backend backend) {
    if (backend == Backend::Compact) {
        return std::make_unique<CompactSam>();
    }
    return std::make_unique<MapSam>();
}

template <typename scalar_t>
std::vector<int64_t> read_cu_seqlens(const torch::Tensor& cu_seqlens) {
    const auto* ptr = cu_seqlens.data_ptr<scalar_t>();
    std::vector<int64_t> result(static_cast<size_t>(cu_seqlens.numel()));
    for (int64_t i = 0; i < cu_seqlens.numel(); ++i) {
        result[static_cast<size_t>(i)] = static_cast<int64_t>(ptr[i]);
    }
    return result;
}

std::vector<int64_t> cu_seqlens_to_vec(const torch::Tensor& cu_seqlens) {
    TORCH_CHECK(cu_seqlens.device().is_cpu(), "cu_seqlens must be a CPU tensor");
    TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be a 1D tensor");
    TORCH_CHECK(cu_seqlens.numel() >= 2, "cu_seqlens must contain at least two entries");
    TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");

    std::vector<int64_t> result;
    if (cu_seqlens.scalar_type() == torch::kInt32) {
        result = read_cu_seqlens<int32_t>(cu_seqlens);
    } else if (cu_seqlens.scalar_type() == torch::kInt64) {
        result = read_cu_seqlens<int64_t>(cu_seqlens);
    } else {
        TORCH_CHECK(false, "cu_seqlens must be int32 or int64");
    }

    TORCH_CHECK(result.front() == 0, "cu_seqlens[0] must be 0");
    for (size_t i = 1; i < result.size(); ++i) {
        TORCH_CHECK(result[i] >= result[i - 1], "cu_seqlens must be monotonic");
    }
    return result;
}

void check_packed_tensor(const torch::Tensor& x, const char* name, int64_t total, int64_t heads) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.scalar_type() == torch::kUInt8, name, " must have dtype torch.uint8");
    TORCH_CHECK(x.dim() == 2, name, " must be shaped [total_tokens, heads]");
    TORCH_CHECK(x.size(0) == total, name, " has wrong total token dimension");
    TORCH_CHECK(x.size(1) == heads, name, " has wrong head dimension");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

} // namespace

class RosaRuntime : public torch::CustomClassHolder {
public:
    RosaRuntime(int64_t num_heads, int64_t num_value_heads, int64_t qk_bits, int64_t value_bits, std::string backend)
        : num_heads_(num_heads),
          num_value_heads_(num_value_heads),
          qk_bits_(qk_bits),
          value_bits_(value_bits),
          backend_(parse_backend(backend)) {
        TORCH_CHECK(num_heads_ > 0, "num_heads must be positive");
        TORCH_CHECK(num_value_heads_ > 0, "num_value_heads must be positive");
        TORCH_CHECK(num_heads_ % num_value_heads_ == 0, "num_heads must be divisible by num_value_heads");
        TORCH_CHECK(qk_bits_ > 0 && qk_bits_ <= 8, "qk_bits must be in [1, 8]");
        TORCH_CHECK(value_bits_ > 0 && value_bits_ <= 8, "value_bits must be in [1, 8]");
    }

    std::tuple<torch::Tensor, torch::Tensor> update_packed(
        const torch::Tensor& cu_seqlens,
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value) {
        TORCH_CHECK(!closed_, "RosaRuntime is closed");

        const std::vector<int64_t> offsets = cu_seqlens_to_vec(cu_seqlens);
        const int64_t batch = static_cast<int64_t>(offsets.size()) - 1;
        const int64_t total = offsets.back();
        TORCH_CHECK(total >= 0, "total token count must be non-negative");

        check_packed_tensor(query, "query", total, num_heads_);
        check_packed_tensor(key, "key", total, num_heads_);
        check_packed_tensor(value, "value", total, num_value_heads_);

        ensure_cache(batch);

        auto output = torch::empty({total, num_heads_}, value.options());
        auto endpos = torch::empty({total, num_heads_}, value.options().dtype(torch::kInt64));

        const uint8_t* query_ptr = query.data_ptr<uint8_t>();
        const uint8_t* key_ptr = key.data_ptr<uint8_t>();
        const uint8_t* value_ptr = value.data_ptr<uint8_t>();
        uint8_t* output_ptr = output.data_ptr<uint8_t>();
        int64_t* endpos_ptr = endpos.data_ptr<int64_t>();

        const int64_t group_size = num_heads_ / num_value_heads_;

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < num_heads_; ++h) {
                ISam& sam = *cache_[static_cast<size_t>(b * num_heads_ + h)];
                const int64_t value_head = h / group_size;
                for (int64_t t = offsets[static_cast<size_t>(b)]; t < offsets[static_cast<size_t>(b + 1)]; ++t) {
                    int64_t pos = -1;
                    const uint8_t out = sam.update(
                        query_ptr[t * num_heads_ + h],
                        key_ptr[t * num_heads_ + h],
                        value_ptr[t * num_value_heads_ + value_head],
                        pos);
                    output_ptr[t * num_heads_ + h] = out;
                    endpos_ptr[t * num_heads_ + h] = pos;
                }
            }
        }

        return {output, endpos};
    }

    void close() {
        cache_.clear();
        batch_size_ = -1;
        closed_ = true;
    }

    int64_t num_heads() const {
        return num_heads_;
    }

    int64_t num_value_heads() const {
        return num_value_heads_;
    }

    int64_t qk_bits() const {
        return qk_bits_;
    }

    int64_t value_bits() const {
        return value_bits_;
    }

    std::string backend() const {
        return backend_name(backend_);
    }

    std::tuple<int64_t, int64_t, int64_t> stats() const {
        int64_t states = 0;
        int64_t edges = 0;
        int64_t values = 0;
        for (const auto& sam : cache_) {
            const SamStats s = sam->stats();
            states += s.states;
            edges += s.edges;
            values += s.values;
        }
        return {states, edges, values};
    }

private:
    void ensure_cache(int64_t batch) {
        if (batch_size_ == -1) {
            batch_size_ = batch;
            cache_.reserve(static_cast<size_t>(batch_size_ * num_heads_));
            for (int64_t i = 0; i < batch_size_ * num_heads_; ++i) {
                cache_.push_back(make_sam(backend_));
            }
            return;
        }
        TORCH_CHECK(batch == batch_size_, "RosaRuntime batch size is fixed after the first update");
    }

    int64_t num_heads_;
    int64_t num_value_heads_;
    int64_t qk_bits_;
    int64_t value_bits_;
    int64_t batch_size_ = -1;
    Backend backend_;
    bool closed_ = false;
    std::vector<std::unique_ptr<ISam>> cache_;
};

TORCH_LIBRARY_FRAGMENT(rosa_soft, m) {
    m.class_<RosaRuntime>("RosaRuntime")
        .def(torch::init<int64_t, int64_t, int64_t, int64_t, std::string>())
        .def("update_packed", &RosaRuntime::update_packed)
        .def("close", &RosaRuntime::close)
        .def("num_heads", &RosaRuntime::num_heads)
        .def("num_value_heads", &RosaRuntime::num_value_heads)
        .def("qk_bits", &RosaRuntime::qk_bits)
        .def("value_bits", &RosaRuntime::value_bits)
        .def("backend", &RosaRuntime::backend)
        .def("stats", &RosaRuntime::stats);
}
