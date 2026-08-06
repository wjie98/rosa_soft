#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

uint64_t elapsed_ns(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - start)
          .count());
}

template <typename T>
uint64_t vector_bytes(const std::vector<T>& values) {
  return static_cast<uint64_t>(values.capacity()) * sizeof(T);
}

struct State {
  int32_t max_length = 0;
  int32_t suffix_link = -1;
  std::vector<std::pair<uint8_t, int32_t>> transitions;
};

struct Trace {
  int32_t state = 0;
  int32_t length = 0;
  int32_t latest_end = -1;

  bool operator==(const Trace& other) const {
    return state == other.state && length == other.length &&
        latest_end == other.latest_end;
  }
};

class SuffixAutomaton {
 public:
  explicit SuffixAutomaton(const std::vector<uint8_t>& symbols) {
    const auto started = Clock::now();
    states_.emplace_back();
    int32_t last = 0;
    for (uint8_t symbol : symbols) {
      last = extend(last, symbol);
      prefix_states_.push_back(last);
    }
    children_.resize(states_.size());
    for (int32_t state = 1; state < static_cast<int32_t>(states_.size()); ++state) {
      children_[states_[state].suffix_link].push_back(state);
    }
    build_tree_order();
    build_ancestors();
    build_latest_ends();
    build_ns_ = elapsed_ns(started);
  }

  int32_t transition(int32_t state, uint8_t symbol) const {
    for (const auto& edge : states_[state].transitions) {
      if (edge.first == symbol) {
        return edge.second;
      }
    }
    return -1;
  }

  int32_t normalize(int32_t state, int32_t length) const {
    if (length <= 0) {
      return 0;
    }
    while (state != 0) {
      const int32_t parent = states_[state].suffix_link;
      if (states_[parent].max_length < length) {
        break;
      }
      state = parent;
    }
    return state;
  }

  int32_t lca(int32_t left, int32_t right) const {
    if (depth_[left] < depth_[right]) {
      std::swap(left, right);
    }
    int32_t difference = depth_[left] - depth_[right];
    int32_t level = 0;
    while (difference != 0) {
      if ((difference & 1) != 0) {
        left = ancestors_[level][left];
      }
      difference >>= 1;
      ++level;
    }
    if (left == right) {
      return left;
    }
    for (int32_t current = static_cast<int32_t>(ancestors_.size()) - 1;
         current >= 0;
         --current) {
      const int32_t left_parent = ancestors_[current][left];
      const int32_t right_parent = ancestors_[current][right];
      if (left_parent != right_parent) {
        left = left_parent;
        right = right_parent;
      }
    }
    return ancestors_[0][left];
  }

  int32_t common_suffix_length(const Trace& trace, int32_t key_end) const {
    if (trace.length == 0 || key_end < 0) {
      return 0;
    }
    const int32_t ancestor = lca(trace.state, prefix_states_[key_end]);
    return std::min(
        {trace.length, states_[ancestor].max_length, key_end + 1});
  }

  int32_t state_count() const {
    return static_cast<int32_t>(states_.size());
  }

  int32_t edge_count() const {
    int32_t result = 0;
    for (const State& state : states_) {
      result += static_cast<int32_t>(state.transitions.size());
    }
    return result;
  }

  const State& state(int32_t index) const { return states_[index]; }
  int32_t prefix_state(int32_t end) const { return prefix_states_[end]; }
  int32_t prefix_count() const { return static_cast<int32_t>(prefix_states_.size()); }
  int32_t tin(int32_t state) const { return tin_[state]; }
  int32_t tout(int32_t state) const { return tout_[state]; }
  int32_t latest_end(int32_t state) const { return latest_ends_[state]; }
  uint64_t build_ns() const { return build_ns_; }

  uint64_t allocated_bytes() const {
    uint64_t result = vector_bytes(states_);
    for (const State& state : states_) {
      result += vector_bytes(state.transitions);
    }
    result += vector_bytes(prefix_states_) + vector_bytes(children_) +
        vector_bytes(tin_) + vector_bytes(tout_) + vector_bytes(depth_) +
        vector_bytes(ancestors_) + vector_bytes(latest_ends_);
    for (const auto& children : children_) {
      result += vector_bytes(children);
    }
    for (const auto& ancestors : ancestors_) {
      result += vector_bytes(ancestors);
    }
    return result;
  }

 private:
  void set_transition(int32_t state, uint8_t symbol, int32_t target) {
    for (auto& edge : states_[state].transitions) {
      if (edge.first == symbol) {
        edge.second = target;
        return;
      }
    }
    states_[state].transitions.emplace_back(symbol, target);
  }

  int32_t extend(int32_t last, uint8_t symbol) {
    const int32_t current = static_cast<int32_t>(states_.size());
    states_.emplace_back();
    states_[current].max_length = states_[last].max_length + 1;
    int32_t parent = last;
    while (parent >= 0 && transition(parent, symbol) < 0) {
      set_transition(parent, symbol, current);
      parent = states_[parent].suffix_link;
    }
    if (parent < 0) {
      states_[current].suffix_link = 0;
      return current;
    }
    const int32_t child = transition(parent, symbol);
    if (states_[parent].max_length + 1 == states_[child].max_length) {
      states_[current].suffix_link = child;
      return current;
    }
    const int32_t clone = static_cast<int32_t>(states_.size());
    states_.push_back(states_[child]);
    states_[clone].max_length = states_[parent].max_length + 1;
    while (parent >= 0 && transition(parent, symbol) == child) {
      set_transition(parent, symbol, clone);
      parent = states_[parent].suffix_link;
    }
    states_[child].suffix_link = clone;
    states_[current].suffix_link = clone;
    return current;
  }

  void build_tree_order() {
    const int32_t count = state_count();
    tin_.assign(count, 0);
    tout_.assign(count, 0);
    depth_.assign(count, 0);
    int32_t clock = 0;
    std::vector<std::pair<int32_t, bool>> stack{{0, false}};
    while (!stack.empty()) {
      const auto [state, leaving] = stack.back();
      stack.pop_back();
      if (leaving) {
        tout_[state] = clock;
        continue;
      }
      tin_[state] = clock++;
      stack.emplace_back(state, true);
      for (auto child = children_[state].rbegin();
           child != children_[state].rend();
           ++child) {
        depth_[*child] = depth_[state] + 1;
        stack.emplace_back(*child, false);
      }
    }
  }

  void build_ancestors() {
    int32_t levels = 1;
    while ((int64_t{1} << levels) < state_count()) {
      ++levels;
    }
    ancestors_.assign(levels, std::vector<int32_t>(state_count(), 0));
    for (int32_t state = 1; state < state_count(); ++state) {
      ancestors_[0][state] = states_[state].suffix_link;
    }
    for (int32_t level = 1; level < levels; ++level) {
      for (int32_t state = 0; state < state_count(); ++state) {
        ancestors_[level][state] =
            ancestors_[level - 1][ancestors_[level - 1][state]];
      }
    }
  }

  void build_latest_ends() {
    latest_ends_.assign(state_count(), -1);
    for (int32_t end = 0; end < prefix_count(); ++end) {
      latest_ends_[prefix_states_[end]] = end;
    }
    std::vector<int32_t> order(state_count() - 1);
    std::iota(order.begin(), order.end(), 1);
    std::sort(order.begin(), order.end(), [&](int32_t left, int32_t right) {
      return states_[left].max_length > states_[right].max_length;
    });
    for (int32_t state : order) {
      const int32_t parent = states_[state].suffix_link;
      latest_ends_[parent] = std::max(latest_ends_[parent], latest_ends_[state]);
    }
  }

  std::vector<State> states_;
  std::vector<int32_t> prefix_states_;
  std::vector<std::vector<int32_t>> children_;
  std::vector<int32_t> tin_;
  std::vector<int32_t> tout_;
  std::vector<int32_t> depth_;
  std::vector<std::vector<int32_t>> ancestors_;
  std::vector<int32_t> latest_ends_;
  uint64_t build_ns_ = 0;
};

struct WaveletNode {
  int32_t low = 0;
  int32_t high = 0;
  int32_t left = -1;
  int32_t right = -1;
  std::vector<int32_t> prefix_ones;
  std::vector<int32_t> zero_positions;
  std::vector<int32_t> one_positions;
};

class WaveletRangePredecessor {
 public:
  WaveletRangePredecessor(
      const std::vector<int32_t>& values,
      int32_t value_count) {
    const auto started = Clock::now();
    int32_t high = 1;
    while (high < std::max(value_count, 1)) {
      high <<= 1;
    }
    root_ = build(values, 0, high);
    build_ns_ = elapsed_ns(started);
  }

  int32_t latest(
      int32_t prefix_size,
      int32_t value_low,
      int32_t value_high) const {
    if (prefix_size <= 0 || value_low >= value_high) {
      return -1;
    }
    return latest_node(root_, prefix_size, value_low, value_high);
  }

  uint64_t build_ns() const { return build_ns_; }

  uint64_t allocated_bytes() const {
    uint64_t result = vector_bytes(nodes_);
    for (const WaveletNode& node : nodes_) {
      result += vector_bytes(node.prefix_ones) +
          vector_bytes(node.zero_positions) +
          vector_bytes(node.one_positions);
    }
    return result;
  }

 private:
  int32_t build(
      const std::vector<int32_t>& values,
      int32_t low,
      int32_t high) {
    const int32_t node_index = static_cast<int32_t>(nodes_.size());
    nodes_.push_back(WaveletNode{});
    nodes_[node_index].low = low;
    nodes_[node_index].high = high;
    nodes_[node_index].prefix_ones.push_back(0);
    if (high - low == 1 || values.empty()) {
      nodes_[node_index].prefix_ones.resize(values.size() + 1, 0);
      return node_index;
    }
    const int32_t middle = low + (high - low) / 2;
    std::vector<int32_t> zeros;
    std::vector<int32_t> ones;
    zeros.reserve(values.size());
    ones.reserve(values.size());
    int32_t one_count = 0;
    for (int32_t position = 0;
         position < static_cast<int32_t>(values.size());
         ++position) {
      if (values[position] < middle) {
        zeros.push_back(values[position]);
        nodes_[node_index].zero_positions.push_back(position);
      } else {
        ones.push_back(values[position]);
        nodes_[node_index].one_positions.push_back(position);
        ++one_count;
      }
      nodes_[node_index].prefix_ones.push_back(one_count);
    }
    if (!zeros.empty()) {
      const int32_t child = build(zeros, low, middle);
      nodes_[node_index].left = child;
    }
    if (!ones.empty()) {
      const int32_t child = build(ones, middle, high);
      nodes_[node_index].right = child;
    }
    return node_index;
  }

  int32_t latest_node(
      int32_t node_index,
      int32_t prefix_size,
      int32_t value_low,
      int32_t value_high) const {
    if (node_index < 0 || prefix_size <= 0) {
      return -1;
    }
    const WaveletNode& node = nodes_[node_index];
    if (value_high <= node.low || node.high <= value_low) {
      return -1;
    }
    if (value_low <= node.low && node.high <= value_high) {
      return prefix_size - 1;
    }
    const int32_t ones = node.prefix_ones[prefix_size];
    const int32_t zeros = prefix_size - ones;
    int32_t left = latest_node(node.left, zeros, value_low, value_high);
    if (left >= 0) {
      left = node.zero_positions[left];
    }
    int32_t right = latest_node(node.right, ones, value_low, value_high);
    if (right >= 0) {
      right = node.one_positions[right];
    }
    return std::max(left, right);
  }

  int32_t root_ = -1;
  std::vector<WaveletNode> nodes_;
  uint64_t build_ns_ = 0;
};

class EndPositionIndex {
 public:
  explicit EndPositionIndex(const SuffixAutomaton& automaton)
      : automaton_(automaton),
        wavelet_(terminal_euler(automaton), automaton.state_count()) {
    const auto certificate_started = Clock::now();
    const int32_t count = automaton.state_count();
    counts_.assign(count, 0);
    minimums_.assign(count, -1);
    maximums_.assign(count, -1);
    gaps_.assign(count, 0);
    for (int32_t end = 0; end < automaton.prefix_count(); ++end) {
      const int32_t state = automaton.prefix_state(end);
      counts_[state] = 1;
      minimums_[state] = end;
      maximums_[state] = end;
    }
    std::vector<int32_t> order(count - 1);
    std::iota(order.begin(), order.end(), 1);
    std::sort(order.begin(), order.end(), [&](int32_t left, int32_t right) {
      return automaton.state(left).max_length >
          automaton.state(right).max_length;
    });
    for (int32_t state : order) {
      if (counts_[state] == 0) {
        continue;
      }
      const int32_t parent = automaton.state(state).suffix_link;
      if (counts_[parent] == 0) {
        counts_[parent] = counts_[state];
        minimums_[parent] = minimums_[state];
        maximums_[parent] = maximums_[state];
        gaps_[parent] = gaps_[state];
      } else {
        gaps_[parent] = std::gcd(
            std::gcd(gaps_[parent], gaps_[state]),
            std::abs(minimums_[parent] - minimums_[state]));
        counts_[parent] += counts_[state];
        minimums_[parent] = std::min(minimums_[parent], minimums_[state]);
        maximums_[parent] = std::max(maximums_[parent], maximums_[state]);
      }
    }
    steps_.assign(count, -1);
    for (int32_t state = 0; state < count; ++state) {
      if (counts_[state] == 1) {
        steps_[state] = 0;
      } else if (counts_[state] > 1 && gaps_[state] > 0) {
        const int32_t span = maximums_[state] - minimums_[state];
        if (span / gaps_[state] + 1 == counts_[state]) {
          steps_[state] = gaps_[state];
        }
      }
    }
    build_ns_ = wavelet_.build_ns() + elapsed_ns(certificate_started);
  }

  int32_t predecessor(int32_t state, int32_t bound) const {
    ++queries;
    if (bound < 0 || counts_[state] == 0) {
      return -1;
    }
    const int32_t step = steps_[state];
    if (step >= 0) {
      ++arithmetic_hits;
      if (bound < minimums_[state]) {
        return -1;
      }
      if (step == 0 || bound >= maximums_[state]) {
        return maximums_[state];
      }
      return minimums_[state] +
          ((bound - minimums_[state]) / step) * step;
    }
    ++wavelet_fallbacks;
    const int32_t prefix = std::min(bound + 1, automaton_.prefix_count());
    return wavelet_.latest(
        prefix, automaton_.tin(state), automaton_.tout(state));
  }

  int32_t minimum(int32_t state) const { return minimums_[state]; }
  uint64_t build_ns() const { return build_ns_; }

  uint64_t allocated_bytes() const {
    return wavelet_.allocated_bytes() + vector_bytes(counts_) +
        vector_bytes(minimums_) + vector_bytes(maximums_) +
        vector_bytes(gaps_) + vector_bytes(steps_);
  }

  mutable uint64_t queries = 0;
  mutable uint64_t arithmetic_hits = 0;
  mutable uint64_t wavelet_fallbacks = 0;

 private:
  static std::vector<int32_t> terminal_euler(
      const SuffixAutomaton& automaton) {
    std::vector<int32_t> result;
    result.reserve(automaton.prefix_count());
    for (int32_t end = 0; end < automaton.prefix_count(); ++end) {
      result.push_back(automaton.tin(automaton.prefix_state(end)));
    }
    return result;
  }

  const SuffixAutomaton& automaton_;
  WaveletRangePredecessor wavelet_;
  std::vector<int32_t> counts_;
  std::vector<int32_t> minimums_;
  std::vector<int32_t> maximums_;
  std::vector<int32_t> gaps_;
  std::vector<int32_t> steps_;
  uint64_t build_ns_ = 0;
};

Trace advance(
    const SuffixAutomaton& automaton,
    const EndPositionIndex& endpos,
    Trace previous,
    uint8_t symbol,
    int32_t bound) {
  int32_t state = previous.state;
  int32_t length = previous.length;
  while (true) {
    int32_t next = automaton.transition(state, symbol);
    if (next >= 0) {
      const int32_t next_length = length + 1;
      next = automaton.normalize(next, next_length);
      const int32_t latest = endpos.predecessor(next, bound);
      if (latest >= 0) {
        return {next, next_length, latest};
      }
    }
    if (state == 0) {
      return {};
    }
    state = automaton.state(state).suffix_link;
    length = std::min(length, automaton.state(state).max_length);
  }
}

Trace advance_full(
    const SuffixAutomaton& automaton,
    Trace previous,
    uint8_t symbol) {
  int32_t state = previous.state;
  int32_t length = previous.length;
  int32_t next = automaton.transition(state, symbol);
  while (state != 0 && next < 0) {
    state = automaton.state(state).suffix_link;
    length = std::min(length, automaton.state(state).max_length);
    next = automaton.transition(state, symbol);
  }
  if (next < 0) {
    return {};
  }
  const int32_t next_length = length + 1;
  next = automaton.normalize(next, next_length);
  return {next, next_length, automaton.latest_end(next)};
}

struct Run {
  int32_t start = 0;
  int32_t stop = 0;
  int32_t length_offset = 0;
  int32_t route_offset = 0;
};

struct HeapEntry {
  int32_t length_offset = 0;
  int32_t route_offset = 0;
  int32_t stop = 0;

  bool operator<(const HeapEntry& other) const {
    return std::tie(length_offset, route_offset, stop) <
        std::tie(other.length_offset, other.route_offset, other.stop);
  }
};

struct Branch {
  Trace trace;
  uint16_t bits = 0;
};

struct Winner {
  int32_t length = 0;
  int32_t route = 0;

  bool operator==(const Winner& other) const {
    return length == other.length && route == other.route;
  }

  bool operator<(const Winner& other) const {
    return std::tie(length, route) < std::tie(other.length, other.route);
  }
};

struct RowWinner {
  int32_t row = 0;
  Winner winner;
};

struct AffineChange {
  int32_t start = 0;
  int32_t stop = 0;
  int32_t length_start = 0;
  int32_t length_step = 0;
  int32_t route_start = 0;
  int32_t route_step = 0;

  Winner value(int32_t row) const {
    const int32_t offset = row - start;
    return {
        length_start + offset * length_step,
        route_start + offset * route_step};
  }
};

struct WinnerOverride {
  int32_t start = 0;
  int32_t stop = 0;
  int32_t from_length_start = 0;
  int32_t from_length_step = 0;
  int32_t from_route_start = 0;
  int32_t from_route_step = 0;
  int32_t to_length_start = 0;
  int32_t to_length_step = 0;
  int32_t to_route_start = 0;
  int32_t to_route_step = 0;

  Winner from(int32_t row) const {
    const int32_t offset = row - start;
    return {
        from_length_start + offset * from_length_step,
        from_route_start + offset * from_route_step};
  }

  Winner to(int32_t row) const {
    const int32_t offset = row - start;
    return {
        to_length_start + offset * to_length_step,
        to_route_start + offset * to_route_step};
  }
};

struct RowOverride {
  int32_t row = 0;
  Winner from;
  Winner to;
};

std::vector<AffineChange> compress_winners(
    const std::vector<RowWinner>& rows) {
  std::vector<AffineChange> result;
  size_t index = 0;
  while (index < rows.size()) {
    const RowWinner& first = rows[index];
    size_t stop = index + 1;
    int32_t length_step = 0;
    int32_t route_step = 0;
    if (stop < rows.size() && rows[stop].row == first.row + 1) {
      length_step = rows[stop].winner.length - first.winner.length;
      route_step = rows[stop].winner.route - first.winner.route;
      ++stop;
      while (stop < rows.size() &&
             rows[stop].row == rows[stop - 1].row + 1 &&
             rows[stop].winner.length - rows[stop - 1].winner.length ==
                 length_step &&
             rows[stop].winner.route - rows[stop - 1].winner.route ==
                 route_step) {
        ++stop;
      }
    }
    result.push_back(
        {first.row,
         rows[stop - 1].row + 1,
         first.winner.length,
         length_step,
         first.winner.route,
         route_step});
    index = stop;
  }
  return result;
}

std::vector<WinnerOverride> compress_overrides(
    const std::vector<RowOverride>& rows) {
  std::vector<WinnerOverride> result;
  size_t index = 0;
  while (index < rows.size()) {
    const RowOverride& first = rows[index];
    size_t stop = index + 1;
    int32_t from_length_step = 0;
    int32_t from_route_step = 0;
    int32_t to_length_step = 0;
    int32_t to_route_step = 0;
    if (stop < rows.size() && rows[stop].row == first.row + 1) {
      from_length_step = rows[stop].from.length - first.from.length;
      from_route_step = rows[stop].from.route - first.from.route;
      to_length_step = rows[stop].to.length - first.to.length;
      to_route_step = rows[stop].to.route - first.to.route;
      ++stop;
      while (stop < rows.size() &&
             rows[stop].row == rows[stop - 1].row + 1 &&
             rows[stop].from.length - rows[stop - 1].from.length ==
                 from_length_step &&
             rows[stop].from.route - rows[stop - 1].from.route ==
                 from_route_step &&
             rows[stop].to.length - rows[stop - 1].to.length ==
                 to_length_step &&
             rows[stop].to.route - rows[stop - 1].to.route ==
                 to_route_step) {
        ++stop;
      }
    }
    result.push_back(
        {first.row,
         rows[stop - 1].row + 1,
         first.from.length,
         from_length_step,
         first.from.route,
         from_route_step,
         first.to.length,
         to_length_step,
         first.to.route,
         to_route_step});
    index = stop;
  }
  return result;
}

struct FactorizedResult {
  int32_t sequence_length = 0;
  int32_t bit_width = 0;
  std::vector<int32_t> base_routes;
  std::vector<int32_t> base_lengths;
  std::vector<std::vector<AffineChange>> query_changes;
  std::vector<std::vector<AffineChange>> key_delete_changes;
  std::vector<std::vector<WinnerOverride>> key_overrides;
};

struct SolverStats {
  uint64_t query_advances = 0;
  uint64_t query_merged = 0;
  uint64_t replacement_probes = 0;
  uint64_t replacement_rows = 0;
  uint64_t replacement_left_queries = 0;
  uint64_t virtual_runs = 0;
  uint64_t virtual_rows = 0;
  uint64_t trace_build_ns = 0;
  uint64_t output_initialize_ns = 0;
  uint64_t query_solve_ns = 0;
  uint64_t key_solve_ns = 0;
  uint64_t total_solve_ns = 0;
  uint64_t query_descriptors = 0;
  uint64_t delete_descriptors = 0;
  uint64_t override_descriptors = 0;
  uint64_t factorized_bytes = 0;
};

constexpr int64_t kStatCount = 31;

class Solver {
 public:
  Solver(
      const uint8_t* query,
      const uint8_t* key,
      int32_t sequence_length,
      int32_t bit_width)
      : query_(query, query + sequence_length),
        key_(key, key + sequence_length),
        length_(sequence_length),
        key_length_(std::max(sequence_length - 1, 0)),
        bit_width_(bit_width),
        automaton_(usable_key()),
        endpos_(automaton_),
        reverse_automaton_(reverse_key()) {
    const auto trace_started = Clock::now();
    build_traces();
    for (int32_t position = 1; position < length_; ++position) {
      query_positions_[query_[position]].push_back(position);
    }
    stats_.trace_build_ns = elapsed_ns(trace_started);
  }

  void solve(
      int64_t* routes,
      int64_t* lengths,
      uint64_t* output_stats,
      int64_t stats_count) {
    const auto solve_started = Clock::now();
    const FactorizedResult result = solve_factorized();
    const int64_t flip_count =
        int64_t{2} * key_length_ * bit_width_;
    const auto initialize_started = Clock::now();
    materialize(result, routes, lengths);
    stats_.output_initialize_ns = elapsed_ns(initialize_started);
    stats_.total_solve_ns = elapsed_ns(solve_started);
    write_stats(output_stats, stats_count, flip_count);
  }

  FactorizedResult solve_factorized() {
    const auto solve_started = Clock::now();
    FactorizedResult result;
    result.sequence_length = length_;
    result.bit_width = bit_width_;
    result.base_routes = base_routes_;
    result.base_lengths = base_lengths_;

    const auto query_started = Clock::now();
    result.query_changes = solve_query_changes();
    stats_.query_solve_ns = elapsed_ns(query_started);
    const auto key_started = Clock::now();
    solve_key_factorized(result);
    stats_.key_solve_ns = elapsed_ns(key_started);
    for (const auto& changes : result.query_changes) {
      stats_.query_descriptors += changes.size();
    }
    for (const auto& changes : result.key_delete_changes) {
      stats_.delete_descriptors += changes.size();
    }
    for (const auto& changes : result.key_overrides) {
      stats_.override_descriptors += changes.size();
    }
    stats_.factorized_bytes =
        static_cast<uint64_t>(2) * length_ * sizeof(int32_t) +
        static_cast<uint64_t>(result.query_changes.size() + 1) *
            sizeof(int64_t) +
        stats_.query_descriptors * sizeof(AffineChange) +
        static_cast<uint64_t>(result.key_delete_changes.size() + 1) *
            sizeof(int64_t) +
        stats_.delete_descriptors * sizeof(AffineChange) +
        static_cast<uint64_t>(result.key_overrides.size() + 1) *
            sizeof(int64_t) +
        stats_.override_descriptors * sizeof(WinnerOverride);
    stats_.total_solve_ns = elapsed_ns(solve_started);
    return result;
  }

  void copy_stats(
      uint64_t* output,
      int64_t stats_count,
      uint64_t output_bytes = 0) const {
    write_stats(
        output,
        stats_count,
        int64_t{2} * key_length_ * bit_width_,
        output_bytes);
  }

 private:
  uint64_t working_bytes() const {
    uint64_t result = vector_bytes(query_) + vector_bytes(key_) +
        vector_bytes(base_traces_) + vector_bytes(full_query_traces_) +
        vector_bytes(reverse_query_traces_) + vector_bytes(base_routes_) +
        vector_bytes(base_lengths_);
    for (const auto& positions : query_positions_) {
      result += vector_bytes(positions);
    }
    return result;
  }

  void materialize(
      const FactorizedResult& result,
      int64_t* routes,
      int64_t* lengths) const {
    const int64_t query_flip_count = int64_t{key_length_} * bit_width_;
    const int64_t flip_count = 2 * query_flip_count;
    for (int64_t flip = 0; flip < flip_count; ++flip) {
      std::copy(
          result.base_routes.begin(),
          result.base_routes.end(),
          routes + flip * length_);
      std::copy(
          result.base_lengths.begin(),
          result.base_lengths.end(),
          lengths + flip * length_);
    }
    const auto apply_change = [&](
                                  int64_t flip,
                                  const AffineChange& change) {
      for (int32_t row = change.start; row < change.stop; ++row) {
        const Winner winner = change.value(row);
        routes[flip * length_ + row] = winner.route;
        lengths[flip * length_ + row] = winner.length;
      }
    };
    for (int64_t flip = 0; flip < query_flip_count; ++flip) {
      for (const AffineChange& change : result.query_changes[flip]) {
        apply_change(flip, change);
      }
    }
    for (int64_t local_flip = 0;
         local_flip < query_flip_count;
         ++local_flip) {
      const int64_t flip = query_flip_count + local_flip;
      const int32_t key_position =
          static_cast<int32_t>(local_flip / bit_width_);
      for (const AffineChange& change :
           result.key_delete_changes[key_position]) {
        apply_change(flip, change);
      }
      for (const WinnerOverride& change : result.key_overrides[local_flip]) {
        for (int32_t row = change.start; row < change.stop; ++row) {
          const Winner winner = change.to(row);
          routes[flip * length_ + row] = winner.route;
          lengths[flip * length_ + row] = winner.length;
        }
      }
    }
  }

  void write_stats(
      uint64_t* output,
      int64_t count,
      int64_t flip_count,
      uint64_t output_bytes = UINT64_MAX) const {
    const std::array<uint64_t, kStatCount> values = {
        static_cast<uint64_t>(automaton_.state_count()),
        static_cast<uint64_t>(automaton_.edge_count()),
        static_cast<uint64_t>(reverse_automaton_.state_count()),
        static_cast<uint64_t>(reverse_automaton_.edge_count()),
        endpos_.queries,
        endpos_.arithmetic_hits,
        endpos_.wavelet_fallbacks,
        stats_.query_advances,
        stats_.query_merged,
        stats_.replacement_probes,
        stats_.replacement_rows,
        stats_.virtual_runs,
        stats_.virtual_rows,
        automaton_.build_ns(),
        reverse_automaton_.build_ns(),
        endpos_.build_ns(),
        stats_.trace_build_ns,
        stats_.output_initialize_ns,
        stats_.query_solve_ns,
        stats_.key_solve_ns,
        stats_.total_solve_ns,
        automaton_.allocated_bytes(),
        reverse_automaton_.allocated_bytes(),
        endpos_.allocated_bytes(),
        working_bytes(),
        output_bytes == UINT64_MAX
            ? static_cast<uint64_t>(flip_count) * length_ * sizeof(int64_t) * 2
            : output_bytes,
        stats_.query_descriptors,
        stats_.delete_descriptors,
        stats_.override_descriptors,
        stats_.factorized_bytes,
        stats_.replacement_left_queries,
    };
    std::copy(values.begin(), values.begin() + std::min(count, kStatCount), output);
  }

  std::vector<uint8_t> usable_key() const {
    return std::vector<uint8_t>(key_.begin(), key_.begin() + key_length_);
  }

  std::vector<uint8_t> reverse_key() const {
    std::vector<uint8_t> result = usable_key();
    std::reverse(result.begin(), result.end());
    return result;
  }

  void build_traces() {
    base_traces_.assign(length_, Trace{});
    for (int32_t row = 1; row < length_; ++row) {
      base_traces_[row] = advance(
          automaton_, endpos_, base_traces_[row - 1], query_[row], row - 1);
    }
    base_routes_.resize(length_, 0);
    base_lengths_.resize(length_, 0);
    for (int32_t row = 0; row < length_; ++row) {
      base_lengths_[row] = base_traces_[row].length;
      base_routes_[row] = base_traces_[row].length > 0
          ? base_traces_[row].latest_end + 1
          : 0;
    }

    full_query_traces_.assign(length_, Trace{});
    for (int32_t row = 1; row < length_; ++row) {
      full_query_traces_[row] = advance_full(
          automaton_, full_query_traces_[row - 1], query_[row]);
    }
    std::vector<uint8_t> reverse_query(query_.begin() + std::min(1, length_), query_.end());
    std::reverse(reverse_query.begin(), reverse_query.end());
    Trace previous;
    for (uint8_t symbol : reverse_query) {
      previous = advance_full(reverse_automaton_, previous, symbol);
      reverse_query_traces_.push_back(previous);
    }
  }

  int64_t query_flip_index(int32_t position, int32_t bit) const {
    return int64_t{position - 1} * bit_width_ + bit;
  }

  static void merge_branch(
      std::vector<Branch>& branches,
      const Trace& trace,
      uint16_t bits) {
    for (Branch& branch : branches) {
      if (branch.trace == trace) {
        branch.bits |= bits;
        return;
      }
    }
    branches.push_back({trace, bits});
  }

  std::vector<std::vector<AffineChange>> solve_query_changes() {
    const int64_t query_flip_count = int64_t{key_length_} * bit_width_;
    std::vector<std::vector<RowWinner>> updates(query_flip_count);
    for (int32_t position = 1; position < length_; ++position) {
      std::vector<Branch> branches;
      for (int32_t bit = 0; bit < bit_width_; ++bit) {
        const Trace trace = advance(
            automaton_,
            endpos_,
            base_traces_[position - 1],
            query_[position] ^ (uint8_t{1} << bit),
            position - 1);
        ++stats_.query_advances;
        if (!(trace == base_traces_[position])) {
          updates[query_flip_index(position, bit)].push_back(
              {position,
               {trace.length, trace.length > 0 ? trace.latest_end + 1 : 0}});
          merge_branch(branches, trace, uint16_t{1} << bit);
        }
      }
      for (int32_t row = position + 1;
           row < length_ && !branches.empty();
           ++row) {
        std::vector<Branch> next;
        for (const Branch& branch : branches) {
          stats_.query_merged +=
              static_cast<uint64_t>(__builtin_popcount(branch.bits) - 1);
          const Trace trace = advance(
              automaton_, endpos_, branch.trace, query_[row], row - 1);
          ++stats_.query_advances;
          if (trace == base_traces_[row]) {
            continue;
          }
          for (int32_t bit = 0; bit < bit_width_; ++bit) {
            if ((branch.bits & (uint16_t{1} << bit)) != 0) {
              updates[query_flip_index(position, bit)].push_back(
                  {row,
                   {trace.length,
                    trace.length > 0 ? trace.latest_end + 1 : 0}});
            }
          }
          merge_branch(next, trace, branch.bits);
        }
        branches.swap(next);
      }
    }

    std::vector<std::vector<AffineChange>> result(query_flip_count);
    for (int64_t flip = 0; flip < query_flip_count; ++flip) {
      result[flip] = compress_winners(updates[flip]);
    }
    return result;
  }

  int32_t left_context(int32_t query_position, int32_t key_position) const {
    if (query_position <= 0 || key_position <= 0) {
      return 0;
    }
    return automaton_.common_suffix_length(
        full_query_traces_[query_position - 1], key_position - 1);
  }

  int32_t right_context(int32_t query_position, int32_t key_position) const {
    if (query_position + 1 >= length_ || key_position + 1 >= key_length_) {
      return 0;
    }
    return reverse_automaton_.common_suffix_length(
        reverse_query_traces_[key_length_ - query_position - 1],
        key_length_ - key_position - 2);
  }

  std::vector<WinnerOverride> apply_runs_factorized(
      const std::vector<Run>& runs,
      const std::vector<Winner>& baseline) {
    std::vector<RowOverride> rows;
    if (runs.empty()) {
      return {};
    }
    std::priority_queue<HeapEntry> heap;
    int32_t run_index = 0;
    int32_t row = runs.front().start;
    while (run_index < static_cast<int32_t>(runs.size()) || !heap.empty()) {
      if (heap.empty() && run_index < static_cast<int32_t>(runs.size())) {
        row = std::max(row, runs[run_index].start);
      }
      while (run_index < static_cast<int32_t>(runs.size()) &&
             runs[run_index].start <= row) {
        const Run& run = runs[run_index++];
        heap.push({run.length_offset, run.route_offset, run.stop});
      }
      while (!heap.empty() && heap.top().stop < row) {
        heap.pop();
      }
      if (heap.empty()) {
        continue;
      }
      ++stats_.virtual_rows;
      const Winner candidate = {
          heap.top().length_offset + row,
          heap.top().route_offset + row};
      if (baseline[row] < candidate) {
        rows.push_back({row, baseline[row], candidate});
      }
      ++row;
    }

    return compress_overrides(rows);
  }

  std::vector<std::vector<AffineChange>> solve_delete_batch() {
    std::vector<std::vector<RowWinner>> updates(key_length_);
    for (int32_t row = 1; row < length_; ++row) {
      const Trace& trace = base_traces_[row];
      const int32_t maximum_length = trace.length;
      const int32_t base_route = base_routes_[row];
      if (maximum_length == 0 || base_route == 0) {
        continue;
      }

      std::vector<int32_t> states(maximum_length + 1, 0);
      std::vector<int32_t> minimums(maximum_length + 1, -1);
      std::vector<int32_t> latest(maximum_length + 1, -1);
      int32_t state = trace.state;
      int32_t previous_state = -1;
      for (int32_t length = maximum_length; length >= 1; --length) {
        while (state != 0) {
          const int32_t parent = automaton_.state(state).suffix_link;
          if (automaton_.state(parent).max_length < length) {
            break;
          }
          state = parent;
        }
        states[length] = state;
        minimums[length] = endpos_.minimum(state);
        if (state != previous_state) {
          latest[length] = endpos_.predecessor(state, row - 1);
          previous_state = state;
        } else {
          latest[length] = latest[length + 1];
        }
      }

      const int32_t first_key = base_route - maximum_length;
      const int32_t last_key = base_route - 1;
      std::vector<int32_t> answers(
          static_cast<size_t>(last_key - first_key + 1), maximum_length);
      int32_t covered_left = 1;
      int32_t covered_right = 0;
      for (int32_t length = 1; length <= maximum_length; ++length) {
        ++stats_.replacement_probes;
        const int32_t invalid_left = latest[length] - length + 1;
        const int32_t invalid_right = minimums[length];
        if (invalid_left > invalid_right) {
          continue;
        }
        if (covered_left <= covered_right &&
            (invalid_left > covered_left || invalid_right < covered_right)) {
          throw std::runtime_error("replacement invalid intervals are not nested");
        }
        const int32_t left_stop = covered_left <= covered_right
            ? covered_left - 1
            : invalid_right;
        for (int32_t key_position = std::max(first_key, invalid_left);
             key_position <= std::min(last_key, left_stop);
             ++key_position) {
          answers[key_position - first_key] = length - 1;
        }
        if (covered_left <= covered_right) {
          for (int32_t key_position =
                   std::max(first_key, covered_right + 1);
               key_position <= std::min(last_key, invalid_right);
               ++key_position) {
            answers[key_position - first_key] = length - 1;
          }
        }
        covered_left = invalid_left;
        covered_right = invalid_right;
      }

      for (int32_t key_position = first_key;
           key_position <= last_key;
           ++key_position) {
        ++stats_.replacement_rows;
        const int32_t low = answers[key_position - first_key];
        Winner winner;
        if (low > 0) {
          int32_t end = latest[low];
          if (end < key_position + low) {
            ++stats_.replacement_left_queries;
            end = endpos_.predecessor(states[low], key_position - 1);
          }
          winner = {low, end + 1};
        }
        if (!(winner == Winner{maximum_length, base_route})) {
          updates[key_position].push_back({row, winner});
        }
      }
    }

    std::vector<std::vector<AffineChange>> result(key_length_);
    for (int32_t key_position = 0;
         key_position < key_length_;
         ++key_position) {
      result[key_position] = compress_winners(updates[key_position]);
    }
    return result;
  }

  void solve_key_factorized(FactorizedResult& result) {
    result.key_delete_changes = solve_delete_batch();
    result.key_overrides.resize(
        static_cast<size_t>(key_length_) * bit_width_);
    std::vector<Winner> replacement(length_);
    for (int32_t key_position = 0; key_position < key_length_; ++key_position) {
      std::vector<std::vector<Run>> run_batches(bit_width_);
      bool has_runs = false;
      for (int32_t bit = 0; bit < bit_width_; ++bit) {
        const uint8_t target = key_[key_position] ^ (uint8_t{1} << bit);
        const auto& positions = query_positions_[target];
        auto center = std::upper_bound(
            positions.begin(), positions.end(), key_position);
        auto& runs = run_batches[bit];
        runs.reserve(static_cast<size_t>(positions.end() - center));
        for (; center != positions.end(); ++center) {
          const int32_t query_position = *center;
          const int32_t left = left_context(query_position, key_position);
          const int32_t right = right_context(query_position, key_position);
          runs.push_back(
              {query_position,
               query_position + right,
               left + 1 - query_position,
               key_position + 1 - query_position});
        }
        stats_.virtual_runs += runs.size();
        has_runs = has_runs || !runs.empty();
      }
      if (!has_runs) {
        continue;
      }

      for (int32_t row = 0; row < length_; ++row) {
        replacement[row] = {base_lengths_[row], base_routes_[row]};
      }
      for (const AffineChange& change :
           result.key_delete_changes[key_position]) {
        for (int32_t row = change.start; row < change.stop; ++row) {
          replacement[row] = change.value(row);
        }
      }

      for (int32_t bit = 0; bit < bit_width_; ++bit) {
        const auto overrides = apply_runs_factorized(
            run_batches[bit], replacement);
        result.key_overrides[
            static_cast<size_t>(key_position) * bit_width_ + bit] =
            overrides;
      }
    }
  }

  std::vector<uint8_t> query_;
  std::vector<uint8_t> key_;
  int32_t length_ = 0;
  int32_t key_length_ = 0;
  int32_t bit_width_ = 0;
  SuffixAutomaton automaton_;
  EndPositionIndex endpos_;
  SuffixAutomaton reverse_automaton_;
  std::vector<Trace> base_traces_;
  std::vector<Trace> full_query_traces_;
  std::vector<Trace> reverse_query_traces_;
  std::vector<int32_t> base_routes_;
  std::vector<int32_t> base_lengths_;
  std::array<std::vector<int32_t>, 256> query_positions_;
  SolverStats stats_;
};

struct FactorizedHandle {
  FactorizedResult result;
};

template <typename Change>
int64_t flattened_size(const std::vector<std::vector<Change>>& groups) {
  int64_t result = 0;
  for (const auto& group : groups) {
    result += static_cast<int64_t>(group.size());
  }
  return result;
}

template <typename Change>
void copy_offsets(
    const std::vector<std::vector<Change>>& groups,
    int64_t* offsets) {
  int64_t total = 0;
  offsets[0] = 0;
  for (size_t group = 0; group < groups.size(); ++group) {
    total += static_cast<int64_t>(groups[group].size());
    offsets[group + 1] = total;
  }
}

void copy_affine_changes(
    const std::vector<std::vector<AffineChange>>& groups,
    int32_t* output) {
  for (const auto& group : groups) {
    for (const AffineChange& change : group) {
      *output++ = change.start;
      *output++ = change.stop;
      *output++ = change.length_start;
      *output++ = change.length_step;
      *output++ = change.route_start;
      *output++ = change.route_step;
    }
  }
}

void copy_overrides(
    const std::vector<std::vector<WinnerOverride>>& groups,
    int32_t* output) {
  for (const auto& group : groups) {
    for (const WinnerOverride& change : group) {
      *output++ = change.start;
      *output++ = change.stop;
      *output++ = change.from_length_start;
      *output++ = change.from_length_step;
      *output++ = change.from_route_start;
      *output++ = change.from_route_step;
      *output++ = change.to_length_start;
      *output++ = change.to_length_step;
      *output++ = change.to_route_start;
      *output++ = change.to_route_step;
    }
  }
}

}  // namespace

extern "C" void* rosa_sam_bitflip_factorized_create(
    const uint8_t* query,
    const uint8_t* key,
    int64_t sequence_length,
    int64_t bit_width,
    uint64_t* stats,
    int64_t stats_count,
    int32_t* error) {
  if (error != nullptr) {
    *error = 0;
  }
  try {
    if (sequence_length < 0 || sequence_length > INT32_MAX ||
        bit_width < 1 || bit_width > 8 || stats_count < 0 ||
        (stats_count > 0 && stats == nullptr)) {
      if (error != nullptr) {
        *error = 1;
      }
      return nullptr;
    }
    auto handle = new FactorizedHandle;
    handle->result.sequence_length = static_cast<int32_t>(sequence_length);
    handle->result.bit_width = static_cast<int32_t>(bit_width);
    if (sequence_length == 0) {
      if (stats != nullptr) {
        std::fill(
            stats,
            stats + std::min(stats_count, kStatCount),
            uint64_t{0});
      }
      return handle;
    }
    if (query == nullptr || key == nullptr) {
      delete handle;
      if (error != nullptr) {
        *error = 2;
      }
      return nullptr;
    }
    Solver solver(
        query,
        key,
        static_cast<int32_t>(sequence_length),
        static_cast<int32_t>(bit_width));
    handle->result = solver.solve_factorized();
    if (stats != nullptr) {
      solver.copy_stats(stats, stats_count);
    }
    return handle;
  } catch (...) {
    if (error != nullptr) {
      *error = 3;
    }
    return nullptr;
  }
}

extern "C" int32_t rosa_sam_bitflip_factorized_sizes(
    const void* opaque,
    int64_t* sizes,
    int64_t size_count) {
  if (opaque == nullptr || sizes == nullptr || size_count < 7) {
    return 1;
  }
  const auto& result = static_cast<const FactorizedHandle*>(opaque)->result;
  const int64_t key_length = std::max(result.sequence_length - 1, 0);
  sizes[0] = result.sequence_length;
  sizes[1] = result.bit_width;
  sizes[2] = key_length * result.bit_width;
  sizes[3] = key_length;
  sizes[4] = flattened_size(result.query_changes);
  sizes[5] = flattened_size(result.key_delete_changes);
  sizes[6] = flattened_size(result.key_overrides);
  return 0;
}

extern "C" int32_t rosa_sam_bitflip_factorized_copy(
    const void* opaque,
    int32_t* base_routes,
    int32_t* base_lengths,
    int64_t* query_offsets,
    int32_t* query_changes,
    int64_t* delete_offsets,
    int32_t* delete_changes,
    int64_t* override_offsets,
    int32_t* overrides) {
  if (opaque == nullptr) {
    return 1;
  }
  const auto& result = static_cast<const FactorizedHandle*>(opaque)->result;
  const bool missing_base = result.sequence_length > 0 &&
      (base_routes == nullptr || base_lengths == nullptr);
  const bool missing_query = query_offsets == nullptr ||
      (flattened_size(result.query_changes) > 0 && query_changes == nullptr);
  const bool missing_delete = delete_offsets == nullptr ||
      (flattened_size(result.key_delete_changes) > 0 &&
       delete_changes == nullptr);
  const bool missing_override = override_offsets == nullptr ||
      (flattened_size(result.key_overrides) > 0 && overrides == nullptr);
  if (missing_base || missing_query || missing_delete || missing_override) {
    return 2;
  }
  std::copy(result.base_routes.begin(), result.base_routes.end(), base_routes);
  std::copy(result.base_lengths.begin(), result.base_lengths.end(), base_lengths);
  copy_offsets(result.query_changes, query_offsets);
  copy_affine_changes(result.query_changes, query_changes);
  copy_offsets(result.key_delete_changes, delete_offsets);
  copy_affine_changes(result.key_delete_changes, delete_changes);
  copy_offsets(result.key_overrides, override_offsets);
  copy_overrides(result.key_overrides, overrides);
  return 0;
}

extern "C" void rosa_sam_bitflip_factorized_destroy(void* opaque) {
  delete static_cast<FactorizedHandle*>(opaque);
}

extern "C" int32_t rosa_sam_bitflip_routes(
    const uint8_t* query,
    const uint8_t* key,
    int64_t sequence_length,
    int64_t bit_width,
    int64_t* routes,
    int64_t* lengths,
    uint64_t* stats,
    int64_t stats_count) {
  try {
    if (sequence_length < 0 || sequence_length > INT32_MAX ||
        bit_width < 1 || bit_width > 8 || stats_count < 13) {
      return 1;
    }
    if (sequence_length == 0) {
      if (stats == nullptr) {
        return 2;
      }
      std::fill(
          stats,
          stats + std::min(stats_count, kStatCount),
          uint64_t{0});
      return 0;
    }
    if (query == nullptr || key == nullptr || stats == nullptr ||
        (sequence_length > 1 && (routes == nullptr || lengths == nullptr))) {
      return 2;
    }
    Solver solver(
        query,
        key,
        static_cast<int32_t>(sequence_length),
        static_cast<int32_t>(bit_width));
    solver.solve(routes, lengths, stats, stats_count);
    return 0;
  } catch (...) {
    return 3;
  }
}
