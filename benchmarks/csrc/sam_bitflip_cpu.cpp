#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

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

  int32_t state_for_length(int32_t state, int32_t length) const {
    if (length <= 0) {
      return 0;
    }
    for (int32_t level = static_cast<int32_t>(ancestors_.size()) - 1;
         level >= 0;
         --level) {
      const int32_t ancestor = ancestors_[level][state];
      if (ancestor != 0 && states_[ancestor].max_length >= length) {
        state = ancestor;
      }
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
    int32_t high = 1;
    while (high < std::max(value_count, 1)) {
      high <<= 1;
    }
    root_ = build(values, 0, high);
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
};

class EndPositionIndex {
 public:
  explicit EndPositionIndex(const SuffixAutomaton& automaton)
      : automaton_(automaton),
        wavelet_(terminal_euler(automaton), automaton.state_count()) {
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

struct SolverStats {
  uint64_t query_advances = 0;
  uint64_t query_merged = 0;
  uint64_t replacement_probes = 0;
  uint64_t replacement_rows = 0;
  uint64_t virtual_runs = 0;
  uint64_t virtual_rows = 0;
};

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
    build_traces();
    for (int32_t position = 1; position < length_; ++position) {
      query_positions_[query_[position]].push_back(position);
    }
  }

  void solve(int64_t* routes, int64_t* lengths, uint64_t* output_stats) {
    const int64_t flip_count =
        int64_t{2} * key_length_ * bit_width_;
    for (int64_t flip = 0; flip < flip_count; ++flip) {
      std::copy(
          base_routes_.begin(),
          base_routes_.end(),
          routes + flip * length_);
      std::copy(
          base_lengths_.begin(),
          base_lengths_.end(),
          lengths + flip * length_);
    }
    solve_query_flips(routes, lengths);
    solve_key_flips(routes, lengths);
    output_stats[0] = automaton_.state_count();
    output_stats[1] = automaton_.edge_count();
    output_stats[2] = reverse_automaton_.state_count();
    output_stats[3] = reverse_automaton_.edge_count();
    output_stats[4] = endpos_.queries;
    output_stats[5] = endpos_.arithmetic_hits;
    output_stats[6] = endpos_.wavelet_fallbacks;
    output_stats[7] = stats_.query_advances;
    output_stats[8] = stats_.query_merged;
    output_stats[9] = stats_.replacement_probes;
    output_stats[10] = stats_.replacement_rows;
    output_stats[11] = stats_.virtual_runs;
    output_stats[12] = stats_.virtual_rows;
  }

 private:
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

  int64_t key_flip_index(int32_t position, int32_t bit) const {
    return int64_t{key_length_} * bit_width_ +
        int64_t{position} * bit_width_ + bit;
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

  void write_trace(
      int64_t flip,
      int32_t row,
      const Trace& trace,
      int64_t* routes,
      int64_t* lengths) const {
    const int64_t index = flip * length_ + row;
    lengths[index] = trace.length;
    routes[index] = trace.length > 0 ? trace.latest_end + 1 : 0;
  }

  void solve_query_flips(int64_t* routes, int64_t* lengths) {
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
        write_trace(
            query_flip_index(position, bit),
            position,
            trace,
            routes,
            lengths);
        if (!(trace == base_traces_[position])) {
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
          for (int32_t bit = 0; bit < bit_width_; ++bit) {
            if ((branch.bits & (uint16_t{1} << bit)) != 0) {
              write_trace(
                  query_flip_index(position, bit),
                  row,
                  trace,
                  routes,
                  lengths);
            }
          }
          if (!(trace == base_traces_[row])) {
            merge_branch(next, trace, branch.bits);
          }
        }
        branches.swap(next);
      }
    }
  }

  bool feasible_avoiding(
      const Trace& trace,
      int32_t bound,
      int32_t key_position,
      int32_t length) {
    ++stats_.replacement_probes;
    const int32_t state = automaton_.state_for_length(trace.state, length);
    const int32_t left = endpos_.predecessor(
        state, std::min(bound, key_position - 1));
    if (left >= 0) {
      return true;
    }
    const int32_t latest = endpos_.predecessor(state, bound);
    return latest >= key_position + length;
  }

  std::pair<int32_t, int32_t> best_avoiding(
      const Trace& trace,
      int32_t bound,
      int32_t key_position) {
    int32_t low = 0;
    int32_t high = trace.length;
    while (low < high) {
      const int32_t middle = low + (high - low + 1) / 2;
      if (feasible_avoiding(trace, bound, key_position, middle)) {
        low = middle;
      } else {
        high = middle - 1;
      }
    }
    if (low == 0) {
      return {0, 0};
    }
    const int32_t state = automaton_.state_for_length(trace.state, low);
    const int32_t left = endpos_.predecessor(
        state, std::min(bound, key_position - 1));
    const int32_t latest = endpos_.predecessor(state, bound);
    const int32_t end = latest >= key_position + low ? latest : left;
    return {low, end + 1};
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

  void apply_runs(
      int64_t flip,
      const std::vector<Run>& runs,
      int64_t* routes,
      int64_t* lengths) {
    if (runs.empty()) {
      return;
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
      const int64_t index = flip * length_ + row;
      const int32_t candidate_length = heap.top().length_offset + row;
      const int32_t candidate_route = heap.top().route_offset + row;
      if (std::tie(candidate_length, candidate_route) >
          std::tie(lengths[index], routes[index])) {
        lengths[index] = candidate_length;
        routes[index] = candidate_route;
      }
      ++row;
    }
  }

  void solve_key_flips(int64_t* routes, int64_t* lengths) {
    std::vector<int64_t> replacement_routes(base_routes_.begin(), base_routes_.end());
    std::vector<int64_t> replacement_lengths(base_lengths_.begin(), base_lengths_.end());
    for (int32_t key_position = 0; key_position < key_length_; ++key_position) {
      std::copy(base_routes_.begin(), base_routes_.end(), replacement_routes.begin());
      std::copy(base_lengths_.begin(), base_lengths_.end(), replacement_lengths.begin());
      for (int32_t row = 1; row < length_; ++row) {
        const int32_t route = base_routes_[row];
        const int32_t length = base_lengths_[row];
        if (route == 0 || length == 0 ||
            route - length > key_position || route - 1 < key_position) {
          continue;
        }
        ++stats_.replacement_rows;
        const auto [new_length, new_route] = best_avoiding(
            base_traces_[row], row - 1, key_position);
        replacement_lengths[row] = new_length;
        replacement_routes[row] = new_route;
      }

      for (int32_t bit = 0; bit < bit_width_; ++bit) {
        const int64_t flip = key_flip_index(key_position, bit);
        std::copy(
            replacement_routes.begin(),
            replacement_routes.end(),
            routes + flip * length_);
        std::copy(
            replacement_lengths.begin(),
            replacement_lengths.end(),
            lengths + flip * length_);
        const uint8_t target = key_[key_position] ^ (uint8_t{1} << bit);
        const auto& positions = query_positions_[target];
        auto center = std::upper_bound(
            positions.begin(), positions.end(), key_position);
        std::vector<Run> runs;
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
        apply_runs(flip, runs, routes, lengths);
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

}  // namespace

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
      std::fill(stats, stats + 13, uint64_t{0});
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
    solver.solve(routes, lengths, stats);
    return 0;
  } catch (...) {
    return 3;
  }
}
