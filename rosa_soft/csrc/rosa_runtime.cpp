#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

struct AutomatonStats {
  int64_t states = 0;
  int64_t edges = 0;
  int64_t logical_bytes = 0;
};

struct AutomatonState {
  int32_t max_length = 0;
  int32_t latest_end = -1;
  int32_t suffix_link = -1;
  int32_t first_edge = -1;
};

struct AutomatonEdge {
  uint8_t symbol = 0;
  int32_t next_state = -1;
  int32_t next_edge = -1;
};

class SuffixAutomaton {
 public:
  explicit SuffixAutomaton(int32_t max_suffix_length)
      : max_suffix_length_(max_suffix_length) {
    states_.emplace_back();
  }

  int32_t update(uint8_t query, uint8_t key) {
    const int32_t matched_end = match_query(query);
    extend_key(key);
    return matched_end;
  }

  AutomatonStats stats() const {
    return {
        static_cast<int64_t>(states_.size()),
        static_cast<int64_t>(edges_.size()),
        static_cast<int64_t>(
            states_.size() * sizeof(AutomatonState) +
            edges_.size() * sizeof(AutomatonEdge) +
            recent_keys_.size() * sizeof(uint8_t)),
    };
  }

 private:
  int32_t find_transition_edge(int32_t state, uint8_t symbol) const {
    int32_t edge = states_[static_cast<size_t>(state)].first_edge;
    while (edge != -1) {
      const AutomatonEdge& candidate = edges_[static_cast<size_t>(edge)];
      if (candidate.symbol == symbol) {
        return edge;
      }
      edge = candidate.next_edge;
    }
    return -1;
  }

  int32_t find_transition(int32_t state, uint8_t symbol) const {
    const int32_t edge = find_transition_edge(state, symbol);
    return edge == -1
        ? -1
        : edges_[static_cast<size_t>(edge)].next_state;
  }

  void add_transition(int32_t state, uint8_t symbol, int32_t next_state) {
    const int32_t new_edge = static_cast<int32_t>(edges_.size());
    edges_.push_back(
        {symbol, next_state, states_[static_cast<size_t>(state)].first_edge});
    states_[static_cast<size_t>(state)].first_edge = new_edge;
  }

  int32_t copy_edges(int32_t source_edge) {
    int32_t first_copy = -1;
    int32_t previous_copy = -1;
    while (source_edge != -1) {
      const AutomatonEdge source = edges_[static_cast<size_t>(source_edge)];
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

  void extend_key(uint8_t symbol) {
    constexpr size_t kMaxInt32 =
        static_cast<size_t>(std::numeric_limits<int32_t>::max());
    TORCH_CHECK(
        states_.size() < kMaxInt32,
        "ROSA suffix-automaton state count exceeded int32 range");
    TORCH_CHECK(
        key_count_ < std::numeric_limits<int32_t>::max(),
        "ROSA suffix-automaton key count exceeded int32 range");
    const int32_t end_position = key_count_++;

    const int32_t next_state = static_cast<int32_t>(states_.size());
    states_.emplace_back();
    states_[static_cast<size_t>(next_state)].max_length =
        states_[static_cast<size_t>(last_key_state_)].max_length + 1;

    int32_t parent = last_key_state_;
    int32_t child = -1;
    while (parent != -1) {
      const int32_t edge = find_transition_edge(parent, symbol);
      if (edge != -1) {
        child = edges_[static_cast<size_t>(edge)].next_state;
        break;
      }
      add_transition(parent, symbol, next_state);
      parent = states_[static_cast<size_t>(parent)].suffix_link;
    }

    if (parent == -1) {
      states_[static_cast<size_t>(next_state)].suffix_link = 0;
    } else {
      if (
          states_[static_cast<size_t>(parent)].max_length + 1 ==
          states_[static_cast<size_t>(child)].max_length) {
        states_[static_cast<size_t>(next_state)].suffix_link = child;
      } else {
        TORCH_CHECK(
            states_.size() < kMaxInt32,
            "ROSA suffix-automaton state count exceeded int32 range");
        const int32_t clone = static_cast<int32_t>(states_.size());
        states_.push_back(states_[static_cast<size_t>(child)]);
        states_[static_cast<size_t>(clone)].max_length =
            states_[static_cast<size_t>(parent)].max_length + 1;
        states_[static_cast<size_t>(clone)].first_edge =
            copy_edges(states_[static_cast<size_t>(child)].first_edge);
        states_[static_cast<size_t>(child)].suffix_link = clone;
        states_[static_cast<size_t>(next_state)].suffix_link = clone;

        while (parent != -1) {
          const int32_t edge = find_transition_edge(parent, symbol);
          if (
              edge == -1 ||
              edges_[static_cast<size_t>(edge)].next_state != child) {
            break;
          }
          edges_[static_cast<size_t>(edge)].next_state = clone;
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
    update_recent_suffixes(end_position);
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
  std::vector<AutomatonState> states_;
  std::vector<AutomatonEdge> edges_;
  std::deque<uint8_t> recent_keys_;
  int32_t query_state_ = 0;
  int32_t query_length_ = 0;
  int32_t last_key_state_ = 0;
  int32_t key_count_ = 0;
};

template <typename scalar_t>
std::vector<int64_t> read_cu_seqlens(const torch::Tensor& cu_seqlens) {
  const auto* entries = cu_seqlens.data_ptr<scalar_t>();
  std::vector<int64_t> result(
      static_cast<size_t>(cu_seqlens.numel()));
  for (int64_t index = 0; index < cu_seqlens.numel(); ++index) {
    result[static_cast<size_t>(index)] =
        static_cast<int64_t>(entries[index]);
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
      int64_t num_payload_heads,
      int64_t qk_bits,
      int64_t payload_bits,
      int64_t max_suffix_length)
      : num_heads_(num_heads),
        num_payload_heads_(num_payload_heads),
        qk_bits_(qk_bits),
        payload_bits_(payload_bits),
        max_suffix_length_(max_suffix_length) {
    TORCH_CHECK(num_heads_ > 0, "num_heads must be positive");
    TORCH_CHECK(
        num_payload_heads_ > 0,
        "num_payload_heads must be positive");
    TORCH_CHECK(
        num_heads_ % num_payload_heads_ == 0,
        "num_heads must be divisible by num_payload_heads");
    TORCH_CHECK(
        qk_bits_ > 0 && qk_bits_ <= 8,
        "qk_bits must be in [1, 8]");
    TORCH_CHECK(
        payload_bits_ > 0 && payload_bits_ <= 8,
        "payload_bits must be in [1, 8]");
    TORCH_CHECK(
        max_suffix_length_ > 0 &&
            max_suffix_length_ <= std::numeric_limits<int32_t>::max(),
        "max_suffix_length must be in [1, int32_max]");
  }

  std::tuple<torch::Tensor, torch::Tensor> update_packed(
      const torch::Tensor& cu_seqlens,
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& payload) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(!closed_, "RosaRuntime is closed");

    const std::vector<int64_t> offsets =
        cu_seqlens_to_vector(cu_seqlens);
    const int64_t slot_count =
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
        payload,
        "payload",
        total_tokens,
        num_payload_heads_);
    ensure_automata(slot_count);

    auto output = torch::empty({total_tokens, num_heads_}, payload.options());
    auto matched_key_end_positions = torch::empty(
        {total_tokens, num_heads_},
        payload.options().dtype(torch::kInt64));

    const uint8_t* query_data = query.data_ptr<uint8_t>();
    const uint8_t* key_data = key.data_ptr<uint8_t>();
    const uint8_t* payload_data = payload.data_ptr<uint8_t>();
    uint8_t* output_data = output.data_ptr<uint8_t>();
    int64_t* matched_key_end_position_data =
        matched_key_end_positions.data_ptr<int64_t>();
    const int64_t query_heads_per_payload_head =
        num_heads_ / num_payload_heads_;
    const uint8_t qk_mask = static_cast<uint8_t>(
        (uint16_t{1} << qk_bits_) - 1);
    const uint8_t payload_mask = static_cast<uint8_t>(
        (uint16_t{1} << payload_bits_) - 1);

    for (int64_t slot = 0; slot < slot_count; ++slot) {
      const int64_t begin = offsets[static_cast<size_t>(slot)];
      const int64_t end = offsets[static_cast<size_t>(slot + 1)];
      const size_t token_count = static_cast<size_t>(end - begin);
      TORCH_CHECK(
          token_count <=
              static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "ROSA sequence length exceeded int32 range");
      for (
          int64_t payload_head = 0;
          payload_head < num_payload_heads_;
          ++payload_head) {
        const auto& history = payload_histories_[static_cast<size_t>(
            slot * num_payload_heads_ + payload_head)];
        TORCH_CHECK(
            history.size() <=
                static_cast<size_t>(std::numeric_limits<int32_t>::max()) -
                    token_count,
            "ROSA payload history exceeded int32 range");
      }
    }

    std::exception_ptr parallel_error;
    std::mutex parallel_error_mutex;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t slot = 0; slot < slot_count; ++slot) {
      for (
          int64_t payload_head = 0;
          payload_head < num_payload_heads_;
          ++payload_head) {
        try {
          const int64_t begin = offsets[static_cast<size_t>(slot)];
          const int64_t end = offsets[static_cast<size_t>(slot + 1)];
          auto& history = payload_histories_[static_cast<size_t>(
              slot * num_payload_heads_ + payload_head)];
          history.reserve(
              history.size() + static_cast<size_t>(end - begin));
          for (int64_t token = begin; token < end; ++token) {
            history.push_back(
                payload_data[
                    token * num_payload_heads_ + payload_head] &
                payload_mask);
          }
        } catch (...) {
          std::lock_guard<std::mutex> error_lock(parallel_error_mutex);
          if (!parallel_error) {
            parallel_error = std::current_exception();
          }
        }
      }
    }
    if (parallel_error) {
      std::rethrow_exception(parallel_error);
    }

    parallel_error = nullptr;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t slot = 0; slot < slot_count; ++slot) {
      for (int64_t head = 0; head < num_heads_; ++head) {
        try {
          SuffixAutomaton& automaton =
              *automata_[static_cast<size_t>(slot * num_heads_ + head)];
          const int64_t payload_head =
              head / query_heads_per_payload_head;
          const auto& payload_history =
              payload_histories_[static_cast<size_t>(
                  slot * num_payload_heads_ + payload_head)];
          const int64_t begin = offsets[static_cast<size_t>(slot)];
          const int64_t end = offsets[static_cast<size_t>(slot + 1)];
          for (int64_t token = begin; token < end; ++token) {
            const int32_t matched_end = automaton.update(
                query_data[token * num_heads_ + head] & qk_mask,
                key_data[token * num_heads_ + head] & qk_mask);
            uint8_t result = 0;
            if (matched_end >= 0) {
              const size_t successor =
                  static_cast<size_t>(matched_end) + 1;
              TORCH_INTERNAL_ASSERT(successor < payload_history.size());
              result = payload_history[successor];
            }
            output_data[token * num_heads_ + head] = result;
            matched_key_end_position_data[token * num_heads_ + head] =
                static_cast<int64_t>(matched_end);
          }
        } catch (...) {
          std::lock_guard<std::mutex> error_lock(parallel_error_mutex);
          if (!parallel_error) {
            parallel_error = std::current_exception();
          }
        }
      }
    }
    if (parallel_error) {
      std::rethrow_exception(parallel_error);
    }

    return {output, matched_key_end_positions};
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    automata_.clear();
    payload_histories_.clear();
    slot_count_ = -1;
    closed_ = true;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(!closed_, "RosaRuntime is closed");
    automata_.clear();
    payload_histories_.clear();
    slot_count_ = -1;
  }

  std::tuple<
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t>
  stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t states = 0;
    int64_t edges = 0;
    int64_t payload_symbols = 0;
    int64_t logical_bytes = 0;
    for (const auto& automaton : automata_) {
      const AutomatonStats current = automaton->stats();
      states += current.states;
      edges += current.edges;
      logical_bytes += current.logical_bytes;
    }
    for (const auto& history : payload_histories_) {
      payload_symbols += static_cast<int64_t>(history.size());
      logical_bytes += static_cast<int64_t>(
          history.size() * sizeof(uint8_t));
    }
    return {
        states,
        edges,
        payload_symbols,
        static_cast<int64_t>(automata_.size()),
        slot_count_ < 0 ? 0 : slot_count_,
        logical_bytes,
    };
  }

 private:
  void ensure_automata(int64_t slot_count) {
    if (slot_count_ == -1) {
      slot_count_ = slot_count;
      automata_.reserve(
          static_cast<size_t>(slot_count_ * num_heads_));
      for (
          int64_t index = 0;
          index < slot_count_ * num_heads_;
          ++index) {
        automata_.push_back(
            std::make_unique<SuffixAutomaton>(
                static_cast<int32_t>(max_suffix_length_)));
      }
      payload_histories_.resize(
          static_cast<size_t>(
              slot_count_ * num_payload_heads_));
      return;
    }
    TORCH_CHECK(
        slot_count == slot_count_,
        "RosaRuntime slot count is fixed after the first update");
  }

  const int64_t num_heads_;
  const int64_t num_payload_heads_;
  const int64_t qk_bits_;
  const int64_t payload_bits_;
  const int64_t max_suffix_length_;
  int64_t slot_count_ = -1;
  bool closed_ = false;
  std::vector<std::unique_ptr<SuffixAutomaton>> automata_;
  std::vector<std::vector<uint8_t>> payload_histories_;
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
      .def("reset", &RosaRuntime::reset)
      .def("close", &RosaRuntime::close)
      .def("stats", &RosaRuntime::stats);
}
