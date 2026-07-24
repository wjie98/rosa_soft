#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

struct SamStats {
  int64_t states = 0;
  int64_t edges = 0;
  int64_t values = 0;
};

struct SamState {
  int32_t max_length = 0;
  int32_t latest_end = -1;
  int32_t suffix_link = -1;
  int32_t first_edge = -1;
};

struct SamEdge {
  uint8_t symbol = 0;
  int32_t next_state = -1;
  int32_t next_edge = -1;
};

class BoundedSuffixAutomaton {
 public:
  explicit BoundedSuffixAutomaton(int32_t max_suffix_length)
      : max_suffix_length_(max_suffix_length) {
    states_.emplace_back();
  }

  uint8_t update(
      uint8_t query,
      uint8_t key,
      uint8_t value,
      int64_t& end_position) {
    const int32_t matched_end = match_query(query);
    extend_key(key, value);
    end_position = static_cast<int64_t>(matched_end);
    return matched_end >= 0
        ? values_[static_cast<size_t>(matched_end + 1)]
        : uint8_t{0};
  }

  SamStats stats() const {
    return {
        static_cast<int64_t>(states_.size()),
        static_cast<int64_t>(edges_.size()),
        static_cast<int64_t>(values_.size()),
    };
  }

 private:
  int32_t find_transition(int32_t state, uint8_t symbol) const {
    int32_t edge = states_[static_cast<size_t>(state)].first_edge;
    while (edge != -1) {
      const SamEdge& candidate = edges_[static_cast<size_t>(edge)];
      if (candidate.symbol == symbol) {
        return candidate.next_state;
      }
      edge = candidate.next_edge;
    }
    return -1;
  }

  void set_transition(int32_t state, uint8_t symbol, int32_t next_state) {
    int32_t edge = states_[static_cast<size_t>(state)].first_edge;
    while (edge != -1) {
      SamEdge& candidate = edges_[static_cast<size_t>(edge)];
      if (candidate.symbol == symbol) {
        candidate.next_state = next_state;
        return;
      }
      edge = candidate.next_edge;
    }

    const int32_t new_edge = static_cast<int32_t>(edges_.size());
    edges_.push_back(
        {symbol, next_state, states_[static_cast<size_t>(state)].first_edge});
    states_[static_cast<size_t>(state)].first_edge = new_edge;
  }

  int32_t copy_edges(int32_t source_edge) {
    int32_t first_copy = -1;
    int32_t previous_copy = -1;
    while (source_edge != -1) {
      const SamEdge source = edges_[static_cast<size_t>(source_edge)];
      const int32_t copied_edge = static_cast<int32_t>(edges_.size());
      edges_.push_back({source.symbol, source.next_state, -1});
      if (previous_copy == -1) {
        first_copy = copied_edge;
      } else {
        edges_[static_cast<size_t>(previous_copy)].next_edge = copied_edge;
      }
      previous_copy = copied_edge;
      source_edge = source.next_edge;
    }
    return first_copy;
  }

  int32_t match_query(uint8_t symbol) {
    int32_t state = query_state_;
    int32_t next = find_transition(state, symbol);
    while (state != 0 && next == -1) {
      state = states_[static_cast<size_t>(state)].suffix_link;
      query_length_ = std::min(
          query_length_,
          states_[static_cast<size_t>(state)].max_length);
      next = find_transition(state, symbol);
    }

    if (next == -1) {
      query_state_ = 0;
      query_length_ = 0;
      return -1;
    }

    query_state_ = next;
    query_length_ =
        std::min(query_length_ + 1, max_suffix_length_);
    while (states_[static_cast<size_t>(query_state_)].suffix_link != -1) {
      const int32_t parent =
          states_[static_cast<size_t>(query_state_)].suffix_link;
      if (states_[static_cast<size_t>(parent)].max_length < query_length_) {
        break;
      }
      query_state_ = parent;
    }
    return states_[static_cast<size_t>(query_state_)].latest_end;
  }

  void extend_key(uint8_t symbol, uint8_t value) {
    constexpr size_t kMaxInt32 =
        static_cast<size_t>(std::numeric_limits<int32_t>::max());
    TORCH_CHECK(
        states_.size() < kMaxInt32,
        "ROSA SAM state count exceeded int32 range");
    TORCH_CHECK(
        values_.size() < kMaxInt32,
        "ROSA SAM value count exceeded int32 range");

    const int32_t value_position = static_cast<int32_t>(values_.size());
    values_.push_back(value);

    const int32_t next_state = static_cast<int32_t>(states_.size());
    states_.emplace_back();
    states_[static_cast<size_t>(next_state)].max_length =
        states_[static_cast<size_t>(last_key_state_)].max_length + 1;

    int32_t parent = last_key_state_;
    while (
        parent != -1 &&
        find_transition(parent, symbol) == -1) {
      set_transition(parent, symbol, next_state);
      parent = states_[static_cast<size_t>(parent)].suffix_link;
    }

    if (parent == -1) {
      states_[static_cast<size_t>(next_state)].suffix_link = 0;
    } else {
      const int32_t child = find_transition(parent, symbol);
      if (
          states_[static_cast<size_t>(parent)].max_length + 1 ==
          states_[static_cast<size_t>(child)].max_length) {
        states_[static_cast<size_t>(next_state)].suffix_link = child;
      } else {
        TORCH_CHECK(
            states_.size() < kMaxInt32,
            "ROSA SAM state count exceeded int32 range");
        const int32_t clone = static_cast<int32_t>(states_.size());
        states_.push_back(states_[static_cast<size_t>(child)]);
        states_[static_cast<size_t>(clone)].max_length =
            states_[static_cast<size_t>(parent)].max_length + 1;
        states_[static_cast<size_t>(clone)].first_edge =
            copy_edges(states_[static_cast<size_t>(child)].first_edge);
        states_[static_cast<size_t>(child)].suffix_link = clone;
        states_[static_cast<size_t>(next_state)].suffix_link = clone;

        while (
            parent != -1 &&
            find_transition(parent, symbol) == child) {
          set_transition(parent, symbol, clone);
          parent = states_[static_cast<size_t>(parent)].suffix_link;
        }
      }
    }

    last_key_state_ = next_state;
    recent_keys_.push_back(symbol);
    if (
        recent_keys_.size() >
        static_cast<size_t>(max_suffix_length_)) {
      recent_keys_.pop_front();
    }
    update_recent_suffixes(value_position);
  }

  void update_recent_suffixes(int32_t end_position) {
    // Queries are capped at W. The state for the newest W-symbol suffix and
    // its suffix-link ancestors cover every end-position class they can use.
    int32_t state = 0;
    for (const uint8_t symbol : recent_keys_) {
      state = find_transition(state, symbol);
      TORCH_INTERNAL_ASSERT(state != -1);
    }
    while (state != 0) {
      states_[static_cast<size_t>(state)].latest_end = end_position;
      state = states_[static_cast<size_t>(state)].suffix_link;
    }
  }

  const int32_t max_suffix_length_;
  std::vector<uint8_t> values_;
  std::vector<SamState> states_;
  std::vector<SamEdge> edges_;
  std::deque<uint8_t> recent_keys_;
  int32_t query_state_ = 0;
  int32_t query_length_ = 0;
  int32_t last_key_state_ = 0;
};

template <typename scalar_t>
std::vector<int64_t> read_cu_seqlens(const torch::Tensor& cu_seqlens) {
  const auto* values = cu_seqlens.data_ptr<scalar_t>();
  std::vector<int64_t> result(
      static_cast<size_t>(cu_seqlens.numel()));
  for (int64_t index = 0; index < cu_seqlens.numel(); ++index) {
    result[static_cast<size_t>(index)] =
        static_cast<int64_t>(values[index]);
  }
  return result;
}

std::vector<int64_t> cu_seqlens_to_vector(
    const torch::Tensor& cu_seqlens) {
  TORCH_CHECK(
      cu_seqlens.device().is_cpu(),
      "cu_seqlens must be a CPU tensor");
  TORCH_CHECK(
      cu_seqlens.dim() == 1,
      "cu_seqlens must be a 1D tensor");
  TORCH_CHECK(
      cu_seqlens.numel() >= 2,
      "cu_seqlens must contain at least two entries");
  TORCH_CHECK(
      cu_seqlens.is_contiguous(),
      "cu_seqlens must be contiguous");

  std::vector<int64_t> offsets;
  if (cu_seqlens.scalar_type() == torch::kInt32) {
    offsets = read_cu_seqlens<int32_t>(cu_seqlens);
  } else if (cu_seqlens.scalar_type() == torch::kInt64) {
    offsets = read_cu_seqlens<int64_t>(cu_seqlens);
  } else {
    TORCH_CHECK(false, "cu_seqlens must be int32 or int64");
  }

  TORCH_CHECK(offsets.front() == 0, "cu_seqlens[0] must be 0");
  for (size_t index = 1; index < offsets.size(); ++index) {
    TORCH_CHECK(
        offsets[index] >= offsets[index - 1],
        "cu_seqlens must be monotonic");
  }
  return offsets;
}

void check_packed_tensor(
    const torch::Tensor& tensor,
    const char* name,
    int64_t total_tokens,
    int64_t heads) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kUInt8,
      name,
      " must have dtype torch.uint8");
  TORCH_CHECK(
      tensor.dim() == 2,
      name,
      " must be shaped [total_tokens, heads]");
  TORCH_CHECK(
      tensor.size(0) == total_tokens,
      name,
      " has wrong total token dimension");
  TORCH_CHECK(
      tensor.size(1) == heads,
      name,
      " has wrong head dimension");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

class RosaRuntime : public torch::CustomClassHolder {
 public:
  RosaRuntime(
      int64_t num_heads,
      int64_t num_value_heads,
      int64_t qk_bits,
      int64_t value_bits,
      int64_t max_suffix_length)
      : num_heads_(num_heads),
        num_value_heads_(num_value_heads),
        qk_bits_(qk_bits),
        value_bits_(value_bits),
        max_suffix_length_(max_suffix_length) {
    TORCH_CHECK(num_heads_ > 0, "num_heads must be positive");
    TORCH_CHECK(
        num_value_heads_ > 0,
        "num_value_heads must be positive");
    TORCH_CHECK(
        num_heads_ % num_value_heads_ == 0,
        "num_heads must be divisible by num_value_heads");
    TORCH_CHECK(
        qk_bits_ > 0 && qk_bits_ <= 8,
        "qk_bits must be in [1, 8]");
    TORCH_CHECK(
        value_bits_ > 0 && value_bits_ <= 8,
        "value_bits must be in [1, 8]");
    TORCH_CHECK(
        max_suffix_length_ > 0 &&
            max_suffix_length_ <= std::numeric_limits<int32_t>::max(),
        "max_suffix_length must be in [1, int32_max]");
  }

  std::tuple<torch::Tensor, torch::Tensor> update_packed(
      const torch::Tensor& cu_seqlens,
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(!closed_, "RosaRuntime is closed");

    const std::vector<int64_t> offsets =
        cu_seqlens_to_vector(cu_seqlens);
    const int64_t batch =
        static_cast<int64_t>(offsets.size()) - 1;
    const int64_t total_tokens = offsets.back();
    TORCH_CHECK(
        total_tokens >= 0,
        "total token count must be non-negative");

    check_packed_tensor(
        query,
        "query",
        total_tokens,
        num_heads_);
    check_packed_tensor(
        key,
        "key",
        total_tokens,
        num_heads_);
    check_packed_tensor(
        value,
        "value",
        total_tokens,
        num_value_heads_);
    ensure_cache(batch);

    auto output = torch::empty({total_tokens, num_heads_}, value.options());
    auto end_positions = torch::empty(
        {total_tokens, num_heads_},
        value.options().dtype(torch::kInt64));

    const uint8_t* query_data = query.data_ptr<uint8_t>();
    const uint8_t* key_data = key.data_ptr<uint8_t>();
    const uint8_t* value_data = value.data_ptr<uint8_t>();
    uint8_t* output_data = output.data_ptr<uint8_t>();
    int64_t* end_position_data = end_positions.data_ptr<int64_t>();
    const int64_t query_heads_per_value_head =
        num_heads_ / num_value_heads_;
    const uint8_t qk_mask = static_cast<uint8_t>(
        (uint16_t{1} << qk_bits_) - 1);
    const uint8_t value_mask = static_cast<uint8_t>(
        (uint16_t{1} << value_bits_) - 1);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      for (int64_t head = 0; head < num_heads_; ++head) {
        BoundedSuffixAutomaton& automaton =
            *cache_[static_cast<size_t>(
                batch_index * num_heads_ + head)];
        const int64_t value_head =
            head / query_heads_per_value_head;
        const int64_t begin =
            offsets[static_cast<size_t>(batch_index)];
        const int64_t end =
            offsets[static_cast<size_t>(batch_index + 1)];
        for (int64_t token = begin; token < end; ++token) {
          int64_t end_position = -1;
          const uint8_t result = automaton.update(
              query_data[token * num_heads_ + head] & qk_mask,
              key_data[token * num_heads_ + head] & qk_mask,
              value_data[
                  token * num_value_heads_ + value_head] & value_mask,
              end_position);
          output_data[token * num_heads_ + head] = result;
          end_position_data[token * num_heads_ + head] =
              end_position;
        }
      }
    }

    return {output, end_positions};
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
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

  int64_t max_suffix_length() const {
    return max_suffix_length_;
  }

  std::tuple<int64_t, int64_t, int64_t> stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t states = 0;
    int64_t edges = 0;
    int64_t values = 0;
    for (const auto& automaton : cache_) {
      const SamStats current = automaton->stats();
      states += current.states;
      edges += current.edges;
      values += current.values;
    }
    return {states, edges, values};
  }

 private:
  void ensure_cache(int64_t batch) {
    if (batch_size_ == -1) {
      batch_size_ = batch;
      cache_.reserve(
          static_cast<size_t>(batch_size_ * num_heads_));
      for (
          int64_t index = 0;
          index < batch_size_ * num_heads_;
          ++index) {
        cache_.push_back(
            std::make_unique<BoundedSuffixAutomaton>(
                static_cast<int32_t>(max_suffix_length_)));
      }
      return;
    }
    TORCH_CHECK(
        batch == batch_size_,
        "RosaRuntime batch size is fixed after the first update");
  }

  const int64_t num_heads_;
  const int64_t num_value_heads_;
  const int64_t qk_bits_;
  const int64_t value_bits_;
  const int64_t max_suffix_length_;
  int64_t batch_size_ = -1;
  bool closed_ = false;
  std::vector<std::unique_ptr<BoundedSuffixAutomaton>> cache_;
  mutable std::mutex mutex_;
};

TORCH_LIBRARY_FRAGMENT(rosa_soft, m) {
  m.class_<RosaRuntime>("RosaRuntime")
      .def(torch::init<
          int64_t,
          int64_t,
          int64_t,
          int64_t,
          int64_t>())
      .def("update_packed", &RosaRuntime::update_packed)
      .def("close", &RosaRuntime::close)
      .def("num_heads", &RosaRuntime::num_heads)
      .def("num_value_heads", &RosaRuntime::num_value_heads)
      .def("qk_bits", &RosaRuntime::qk_bits)
      .def("value_bits", &RosaRuntime::value_bits)
      .def("max_suffix_length", &RosaRuntime::max_suffix_length)
      .def("stats", &RosaRuntime::stats);
}
