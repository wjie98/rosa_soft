#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#ifndef ROSA_RUNTIME_ACCESS
#define ROSA_RUNTIME_ACCESS(...) do { } while (false)
#endif

#define ROSA_TRACE_STATE(index, write, kind)                              \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTraceState,                                                   \
      0,                                                                 \
      static_cast<int64_t>(index) * sizeof(AutomatonState),             \
      sizeof(AutomatonState),                                            \
      write,                                                             \
      kind)
#define ROSA_TRACE_EDGE_SYMBOL(index, write, kind)                        \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTraceEdgeSymbol,                                              \
      0,                                                                 \
      static_cast<int64_t>(index) * sizeof(uint8_t),                     \
      sizeof(uint8_t),                                                    \
      write,                                                             \
      kind)
#define ROSA_TRACE_EDGE_TARGET(index, write, kind)                        \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTraceEdgeTarget,                                              \
      0,                                                                 \
      static_cast<int64_t>(index) * sizeof(int32_t),                     \
      sizeof(int32_t),                                                    \
      write,                                                             \
      kind)
#define ROSA_TRACE_EDGE_NEXT(index, write, kind)                          \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTraceEdgeNext,                                                \
      0,                                                                 \
      static_cast<int64_t>(index) * sizeof(int32_t),                     \
      sizeof(int32_t),                                                    \
      write,                                                             \
      kind)
#define ROSA_TRACE_TRANSITION_META(index, write, kind)                    \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTraceTransitionMeta,                                          \
      0,                                                                 \
      static_cast<int64_t>(index) * sizeof(TransitionIndex),             \
      sizeof(TransitionIndex),                                           \
      write,                                                             \
      kind)
#define ROSA_TRACE_TRANSITION_SLOTS(owner, offset, width, write, kind)    \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTraceTransitionSlots,                                         \
      owner,                                                             \
      offset,                                                            \
      width,                                                             \
      write,                                                             \
      kind)
#define ROSA_TRACE_ROOT(offset, width, write, kind)                       \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTraceRootTable, 0, offset, width, write, kind)
#define ROSA_TRACE_LCT(index, write, kind)                                \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTraceLatestEndTree,                                           \
      0,                                                                 \
      static_cast<int64_t>(index) * sizeof(Node),                        \
      sizeof(Node),                                                       \
      write,                                                             \
      kind)
#define ROSA_TRACE_PAYLOAD(index, write, kind)                            \
  ROSA_RUNTIME_ACCESS(                                                   \
      kRosaTracePayload,                                                 \
      0,                                                                 \
      static_cast<int64_t>(index) * sizeof(uint8_t),                     \
      sizeof(uint8_t),                                                    \
      write,                                                             \
      kind)

namespace {

struct AutomatonStats {
  int64_t states = 0;
  int64_t edges = 0;
  int64_t logical_bytes = 0;
};

struct AutomatonComplexityStats {
  int64_t updates = 0;
  int64_t transition_probes = 0;
  int64_t query_suffix_link_steps = 0;
  int64_t extension_suffix_link_steps = 0;
  int64_t latest_end_updates = 0;
  int64_t clones = 0;
  int64_t copied_edges = 0;
  int64_t max_query_suffix_link_steps = 0;
  int64_t max_latest_end_chain = 0;
  int64_t latest_path_updates = 0;
  int64_t latest_tree_rotations = 0;
  int64_t latest_tree_activations = 0;
  int64_t compressed_run_symbols = 0;
  int64_t compressed_materializations = 0;
  int64_t max_compressed_run_length = 0;
  int64_t root_table_lookups = 0;
  int64_t root_table_activations = 0;
  int64_t transition_index_lookups = 0;
  int64_t transition_index_activations = 0;
};

struct AutomatonState {
  int32_t max_length = 0;
  int32_t latest_end = -1;
  int32_t suffix_link = -1;
  int32_t first_edge = -1;
};

class AutomatonEdges {
 public:
  size_t size() const {
    return symbols_.size();
  }

  bool empty() const {
    return symbols_.empty();
  }

  void reserve(size_t size) {
    symbols_.reserve(size);
    next_states_.reserve(size);
    next_edges_.reserve(size);
  }

  void push_back(uint8_t symbol, int32_t next_state, int32_t next_edge) {
    ROSA_TRACE_EDGE_SYMBOL(symbols_.size(), true, kRosaTraceAppend);
    ROSA_TRACE_EDGE_TARGET(symbols_.size(), true, kRosaTraceAppend);
    ROSA_TRACE_EDGE_NEXT(symbols_.size(), true, kRosaTraceAppend);
    symbols_.push_back(symbol);
    next_states_.push_back(next_state);
    next_edges_.push_back(next_edge);
  }

  uint8_t symbol(int32_t edge) const {
    ROSA_TRACE_EDGE_SYMBOL(edge, false, kRosaTraceTransition);
    return symbols_[static_cast<size_t>(edge)];
  }

  int32_t next_state(int32_t edge) const {
    ROSA_TRACE_EDGE_TARGET(edge, false, kRosaTraceTransition);
    return next_states_[static_cast<size_t>(edge)];
  }

  int32_t next_edge(int32_t edge) const {
    ROSA_TRACE_EDGE_NEXT(edge, false, kRosaTraceTransition);
    return next_edges_[static_cast<size_t>(edge)];
  }

  void set_next_state(int32_t edge, int32_t next_state) {
    ROSA_TRACE_EDGE_TARGET(edge, true, kRosaTraceClone);
    next_states_[static_cast<size_t>(edge)] = next_state;
  }

  void set_next_edge(int32_t edge, int32_t next_edge) {
    ROSA_TRACE_EDGE_NEXT(edge, true, kRosaTraceClone);
    next_edges_[static_cast<size_t>(edge)] = next_edge;
  }

  int64_t logical_bytes() const {
    return static_cast<int64_t>(
        size() * (sizeof(uint8_t) + 2 * sizeof(int32_t)));
  }

 private:
  std::vector<uint8_t> symbols_;
  std::vector<int32_t> next_states_;
  std::vector<int32_t> next_edges_;
};

struct TransitionIndex {
  int32_t first_edge = -1;
  int32_t edge_count = 0;
  bool direct = false;
  std::vector<int32_t> slots;

  int64_t logical_bytes() const {
    return static_cast<int64_t>(
        2 * sizeof(int32_t) + sizeof(bool) +
        slots.size() * sizeof(int32_t));
  }
};

class LatestEndTree {
 public:
  struct Node {
    int32_t left = -1;
    int32_t right = -1;
    int32_t parent = -1;
    int32_t value = -1;
    int32_t lazy_value = -1;
  };

  LatestEndTree(
      const std::vector<AutomatonState>& states,
      int64_t* rotation_counter) : rotation_counter_(rotation_counter) {
    nodes_.resize(states.size());
    stack_.reserve(states.size());
    for (size_t index = 0; index < states.size(); ++index) {
      ROSA_TRACE_STATE(index, false, kRosaTraceRebuild);
      ROSA_TRACE_LCT(index, true, kRosaTraceRebuild);
      nodes_[index].value = states[index].latest_end;
      nodes_[index].parent = states[index].suffix_link;
    }
  }

  int32_t add_node(int32_t value) {
    TORCH_INTERNAL_ASSERT(
        nodes_.size() <
        static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    const int32_t index = static_cast<int32_t>(nodes_.size());
    nodes_.push_back(Node{});
    ROSA_TRACE_LCT(index, true, kRosaTraceAppend);
    nodes_.back().value = value;
    return index;
  }

  int32_t point_query(int32_t node) {
    access(node);
    ROSA_TRACE_LCT(node, false, kRosaTraceLatest);
    return nodes_[static_cast<size_t>(node)].value;
  }

  void assign_root_path(int32_t node, int32_t value) {
    access(node);
    apply_value(node, value);
  }

  void link(int32_t child, int32_t parent) {
    TORCH_INTERNAL_ASSERT(child >= 0 && parent >= 0);
    access(child);
    ROSA_TRACE_LCT(child, false, kRosaTraceLatest);
    TORCH_INTERNAL_ASSERT(nodes_[static_cast<size_t>(child)].left == -1);
    ROSA_TRACE_LCT(child, true, kRosaTraceLatest);
    nodes_[static_cast<size_t>(child)].parent = parent;
  }

  void reparent(int32_t child, int32_t parent) {
    access(child);
    ROSA_TRACE_LCT(child, false, kRosaTraceLatest);
    const int32_t ancestors = nodes_[static_cast<size_t>(child)].left;
    TORCH_INTERNAL_ASSERT(ancestors != -1);
    ROSA_TRACE_LCT(child, true, kRosaTraceLatest);
    nodes_[static_cast<size_t>(child)].left = -1;
    ROSA_TRACE_LCT(ancestors, true, kRosaTraceLatest);
    nodes_[static_cast<size_t>(ancestors)].parent = -1;
    nodes_[static_cast<size_t>(child)].parent = parent;
  }

  int64_t logical_bytes() const {
    return static_cast<int64_t>(nodes_.size() * sizeof(Node));
  }

 private:
  bool is_aux_root(int32_t node) const {
    ROSA_TRACE_LCT(node, false, kRosaTraceLatest);
    const int32_t parent = nodes_[static_cast<size_t>(node)].parent;
    if (parent == -1) {
      return true;
    }
    ROSA_TRACE_LCT(parent, false, kRosaTraceLatest);
    const Node& parent_node = nodes_[static_cast<size_t>(parent)];
    return parent_node.left != node && parent_node.right != node;
  }

  void apply_value(int32_t node, int32_t value) {
    if (node == -1) {
      return;
    }
    ROSA_TRACE_LCT(node, true, kRosaTraceLatest);
    Node& target = nodes_[static_cast<size_t>(node)];
    target.value = value;
    target.lazy_value = value;
  }

  void push(int32_t node) {
    ROSA_TRACE_LCT(node, false, kRosaTraceLatest);
    Node& target = nodes_[static_cast<size_t>(node)];
    if (target.lazy_value < 0) {
      return;
    }
    apply_value(target.left, target.lazy_value);
    apply_value(target.right, target.lazy_value);
    ROSA_TRACE_LCT(node, true, kRosaTraceLatest);
    target.lazy_value = -1;
  }

  void push_path(int32_t node) {
    stack_.clear();
    int32_t current = node;
    stack_.push_back(current);
    while (!is_aux_root(current)) {
      ROSA_TRACE_LCT(current, false, kRosaTraceLatest);
      current = nodes_[static_cast<size_t>(current)].parent;
      stack_.push_back(current);
    }
    for (auto iterator = stack_.rbegin(); iterator != stack_.rend(); ++iterator) {
      push(*iterator);
    }
  }

  void rotate(int32_t node) {
    ROSA_TRACE_LCT(node, false, kRosaTraceLatest);
    const int32_t parent = nodes_[static_cast<size_t>(node)].parent;
    ROSA_TRACE_LCT(parent, false, kRosaTraceLatest);
    const int32_t grandparent = nodes_[static_cast<size_t>(parent)].parent;
    const bool node_is_right =
        nodes_[static_cast<size_t>(parent)].right == node;
    const int32_t middle = node_is_right
        ? nodes_[static_cast<size_t>(node)].left
        : nodes_[static_cast<size_t>(node)].right;
    if (!is_aux_root(parent)) {
      ROSA_TRACE_LCT(grandparent, true, kRosaTraceLatest);
      Node& grandparent_node = nodes_[static_cast<size_t>(grandparent)];
      if (grandparent_node.left == parent) {
        grandparent_node.left = node;
      } else {
        grandparent_node.right = node;
      }
    }
    ROSA_TRACE_LCT(node, true, kRosaTraceLatest);
    nodes_[static_cast<size_t>(node)].parent = grandparent;
    ROSA_TRACE_LCT(parent, true, kRosaTraceLatest);
    if (node_is_right) {
      nodes_[static_cast<size_t>(node)].left = parent;
      nodes_[static_cast<size_t>(parent)].right = middle;
    } else {
      nodes_[static_cast<size_t>(node)].right = parent;
      nodes_[static_cast<size_t>(parent)].left = middle;
    }
    nodes_[static_cast<size_t>(parent)].parent = node;
    if (middle != -1) {
      ROSA_TRACE_LCT(middle, true, kRosaTraceLatest);
      nodes_[static_cast<size_t>(middle)].parent = parent;
    }
    ++*rotation_counter_;
  }

  void splay(int32_t node) {
    push_path(node);
    while (!is_aux_root(node)) {
      ROSA_TRACE_LCT(node, false, kRosaTraceLatest);
      const int32_t parent = nodes_[static_cast<size_t>(node)].parent;
      if (!is_aux_root(parent)) {
        ROSA_TRACE_LCT(parent, false, kRosaTraceLatest);
        const int32_t grandparent =
            nodes_[static_cast<size_t>(parent)].parent;
        ROSA_TRACE_LCT(grandparent, false, kRosaTraceLatest);
        const bool zig_zig =
            (nodes_[static_cast<size_t>(parent)].left == node) ==
            (nodes_[static_cast<size_t>(grandparent)].left == parent);
        rotate(zig_zig ? parent : node);
      }
      rotate(node);
    }
  }

  void access(int32_t node) {
    int32_t preferred = -1;
    for (int32_t current = node; current != -1;) {
      splay(current);
      ROSA_TRACE_LCT(current, false, kRosaTraceLatest);
      ROSA_TRACE_LCT(current, true, kRosaTraceLatest);
      Node& current_node = nodes_[static_cast<size_t>(current)];
      current_node.right = preferred;
      if (preferred != -1) {
        ROSA_TRACE_LCT(preferred, true, kRosaTraceLatest);
        nodes_[static_cast<size_t>(preferred)].parent = current;
      }
      preferred = current;
      current = current_node.parent;
    }
    splay(node);
  }

  std::vector<Node> nodes_;
  std::vector<int32_t> stack_;
  int64_t* rotation_counter_;
};

class SuffixAutomaton {
 public:
  SuffixAutomaton() {
    states_.emplace_back();
    ROSA_TRACE_STATE(0, true, kRosaTraceAppend);
  }

  int32_t update(uint8_t query, uint8_t key) {
    ++complexity_.updates;
    if (compressed_run_active_) {
      return update_compressed_run(query, key);
    }
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
            edges_.logical_bytes() +
            (latest_end_tree_ == nullptr
                 ? 0
                 : latest_end_tree_->logical_bytes()) +
            (root_transition_edges_ == nullptr
                 ? 0
                 : sizeof(*root_transition_edges_)) +
            transition_index_bytes()),
    };
  }

  AutomatonComplexityStats complexity_stats() const {
    return complexity_;
  }

 private:
  static bool has_transition_index(int32_t first_edge) {
    return first_edge < -1;
  }

  static size_t transition_index_id(int32_t first_edge) {
    TORCH_INTERNAL_ASSERT(has_transition_index(first_edge));
    return static_cast<size_t>(-first_edge - 2);
  }

  int32_t transition_head(int32_t state) const {
    ROSA_TRACE_STATE(state, false, kRosaTraceTransition);
    const int32_t first_edge =
        states_[static_cast<size_t>(state)].first_edge;
    if (has_transition_index(first_edge)) {
      ROSA_TRACE_TRANSITION_META(
          transition_index_id(first_edge),
          false,
          kRosaTraceTransition);
    }
    return has_transition_index(first_edge)
        ? transition_indices_[transition_index_id(first_edge)].first_edge
        : first_edge;
  }

  int64_t transition_index_bytes() const {
    int64_t bytes = 0;
    for (const TransitionIndex& index : transition_indices_) {
      bytes += index.logical_bytes();
    }
    return bytes;
  }

  size_t transition_slot(uint8_t symbol, size_t mask) const {
    return (static_cast<size_t>(symbol) * size_t{0x9e3779b1U}) & mask;
  }

  void insert_indexed_transition(
      TransitionIndex& index,
      int32_t edge,
      bool count_edge) {
    [[maybe_unused]] const int64_t index_id = static_cast<int64_t>(
        &index - transition_indices_.data());
    ROSA_TRACE_TRANSITION_META(index_id, true, kRosaTraceIndex);
    const uint8_t symbol = edges_.symbol(edge);
    if (index.direct) {
      ROSA_TRACE_TRANSITION_SLOTS(
          index_id,
          static_cast<int64_t>(symbol) * sizeof(int32_t),
          sizeof(int32_t),
          true,
          kRosaTraceIndex);
      index.slots[symbol] = edge;
      if (count_edge) {
        ++index.edge_count;
      }
      return;
    }

    constexpr int32_t kDirectTransitionThreshold = 65;
    const int32_t resulting_count =
        index.edge_count + (count_edge ? 1 : 0);
    if (
        index.slots.empty() ||
        resulting_count * 2 > static_cast<int32_t>(index.slots.size())) {
      const bool use_direct = resulting_count >= kDirectTransitionThreshold;
      const size_t new_size = use_direct
          ? size_t{256}
          : (index.slots.empty() ? size_t{32} : index.slots.size() * 2);
      if (!index.slots.empty()) {
        ROSA_TRACE_TRANSITION_SLOTS(
            index_id,
            0,
            static_cast<int64_t>(index.slots.size() * sizeof(int32_t)),
            false,
            kRosaTraceIndex);
      }
      std::vector<int32_t> old_slots = std::move(index.slots);
      index.slots.assign(new_size, -1);
      ROSA_TRACE_TRANSITION_SLOTS(
          index_id,
          0,
          static_cast<int64_t>(new_size * sizeof(int32_t)),
          true,
          kRosaTraceAppend);
      index.direct = use_direct;
      for (const int32_t old_edge : old_slots) {
        if (old_edge != -1) {
          insert_indexed_transition(index, old_edge, false);
        }
      }
    }
    if (index.direct) {
      ROSA_TRACE_TRANSITION_SLOTS(
          index_id,
          static_cast<int64_t>(symbol) * sizeof(int32_t),
          sizeof(int32_t),
          true,
          kRosaTraceIndex);
      index.slots[symbol] = edge;
      if (count_edge) {
        ++index.edge_count;
      }
      return;
    }
    const size_t mask = index.slots.size() - 1;
    size_t slot = transition_slot(symbol, mask);
    while (true) {
      ROSA_TRACE_TRANSITION_SLOTS(
          index_id,
          static_cast<int64_t>(slot * sizeof(int32_t)),
          sizeof(int32_t),
          false,
          kRosaTraceIndex);
      if (index.slots[slot] == -1) {
        break;
      }
      slot = (slot + 1) & mask;
    }
    ROSA_TRACE_TRANSITION_SLOTS(
        index_id,
        static_cast<int64_t>(slot * sizeof(int32_t)),
        sizeof(int32_t),
        true,
        kRosaTraceIndex);
    index.slots[slot] = edge;
    if (count_edge) {
      ++index.edge_count;
    }
  }

  void maybe_index_transitions(int32_t state) {
    constexpr int32_t kTransitionIndexThreshold = 16;
    ROSA_TRACE_STATE(state, false, kRosaTraceIndex);
    AutomatonState& target = states_[static_cast<size_t>(state)];
    if (state == 0 || has_transition_index(target.first_edge)) {
      return;
    }
    int32_t edge_count = 0;
    for (int32_t edge = target.first_edge; edge != -1;) {
      ++edge_count;
      if (edge_count >= kTransitionIndexThreshold) {
        break;
      }
      edge = edges_.next_edge(edge);
    }
    if (edge_count < kTransitionIndexThreshold) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        transition_indices_.size() <
        static_cast<size_t>(std::numeric_limits<int32_t>::max() - 1));
    const int32_t index_id =
        static_cast<int32_t>(transition_indices_.size());
    transition_indices_.emplace_back();
    ROSA_TRACE_TRANSITION_META(index_id, true, kRosaTraceAppend);
    TransitionIndex& index = transition_indices_.back();
    index.first_edge = target.first_edge;
    for (int32_t edge = index.first_edge; edge != -1;) {
      insert_indexed_transition(index, edge, true);
      edge = edges_.next_edge(edge);
    }
    ROSA_TRACE_STATE(state, true, kRosaTraceIndex);
    target.first_edge = -index_id - 2;
    ++complexity_.transition_index_activations;
  }

  int32_t update_compressed_run(uint8_t query, uint8_t key) {
    if (!compressed_run_initialized_) {
      compressed_run_initialized_ = true;
      compressed_symbol_ = key;
      compressed_run_length_ = 1;
      ++complexity_.compressed_run_symbols;
      complexity_.max_compressed_run_length = 1;
      return -1;
    }

    int32_t matched_end = -1;
    if (query == compressed_symbol_) {
      compressed_query_length_ = std::min(
          compressed_query_length_ + 1,
          compressed_run_length_);
      matched_end = compressed_run_length_ - 1;
    } else {
      compressed_query_length_ = 0;
    }

    if (key == compressed_symbol_) {
      TORCH_CHECK(
          compressed_run_length_ < std::numeric_limits<int32_t>::max(),
          "ROSA compressed run length exceeded int32 range");
      ++compressed_run_length_;
      ++complexity_.compressed_run_symbols;
      complexity_.max_compressed_run_length = std::max(
          complexity_.max_compressed_run_length,
          static_cast<int64_t>(compressed_run_length_));
      return matched_end;
    }

    materialize_compressed_run();
    extend_key(key);
    return matched_end;
  }

  void materialize_compressed_run() {
    TORCH_INTERNAL_ASSERT(compressed_run_active_);
    TORCH_INTERNAL_ASSERT(compressed_run_initialized_);
    TORCH_INTERNAL_ASSERT(compressed_run_length_ > 0);
    TORCH_INTERNAL_ASSERT(states_.size() == 1);
    TORCH_INTERNAL_ASSERT(edges_.empty());

    const int32_t run_length = compressed_run_length_;
    states_.reserve(static_cast<size_t>(run_length) + 2);
    edges_.reserve(static_cast<size_t>(run_length) + 2);
    ROSA_TRACE_STATE(0, true, kRosaTraceMaterialize);
    states_[0].latest_end = run_length - 1;
    for (int32_t length = 1; length <= run_length; ++length) {
      const int32_t state = static_cast<int32_t>(states_.size());
      states_.emplace_back();
      ROSA_TRACE_STATE(state, true, kRosaTraceMaterialize);
      AutomatonState& current = states_.back();
      current.max_length = length;
      current.latest_end = run_length - 1;
      current.suffix_link = length - 1;
      add_transition(length - 1, compressed_symbol_, state);
    }
    query_state_ = compressed_query_length_;
    last_key_state_ = run_length;
    key_count_ = run_length;
    compressed_run_active_ = false;
    ++complexity_.compressed_materializations;
  }

  int32_t find_transition_edge(int32_t state, uint8_t symbol) const {
    if (state == 0 && root_transition_edges_ != nullptr) {
      ++complexity_.transition_probes;
      ++complexity_.root_table_lookups;
      ROSA_TRACE_ROOT(
          static_cast<int64_t>(symbol) * sizeof(int32_t),
          sizeof(int32_t),
          false,
          kRosaTraceTransition);
      return (*root_transition_edges_)[symbol];
    }
    ROSA_TRACE_STATE(state, false, kRosaTraceTransition);
    const int32_t first_edge =
        states_[static_cast<size_t>(state)].first_edge;
    if (has_transition_index(first_edge)) {
      ++complexity_.transition_index_lookups;
      const size_t index_id = transition_index_id(first_edge);
      ROSA_TRACE_TRANSITION_META(
          index_id,
          false,
          kRosaTraceTransition);
      const TransitionIndex& index =
          transition_indices_[index_id];
      if (index.direct) {
        ++complexity_.transition_probes;
        ROSA_TRACE_TRANSITION_SLOTS(
            index_id,
            static_cast<int64_t>(symbol) * sizeof(int32_t),
            sizeof(int32_t),
            false,
            kRosaTraceTransition);
        return index.slots[symbol];
      }
      const size_t mask = index.slots.size() - 1;
      size_t slot = transition_slot(symbol, mask);
      while (true) {
        ++complexity_.transition_probes;
        ROSA_TRACE_TRANSITION_SLOTS(
            index_id,
            static_cast<int64_t>(slot * sizeof(int32_t)),
            sizeof(int32_t),
            false,
            kRosaTraceTransition);
        const int32_t edge = index.slots[slot];
        if (edge == -1) {
          return -1;
        }
        if (edges_.symbol(edge) == symbol) {
          return edge;
        }
        slot = (slot + 1) & mask;
      }
    }

    int32_t edge = first_edge;
    while (edge != -1) {
      ++complexity_.transition_probes;
      if (edges_.symbol(edge) == symbol) {
        return edge;
      }
      edge = edges_.next_edge(edge);
    }
    return -1;
  }

  int32_t find_transition(int32_t state, uint8_t symbol) const {
    const int32_t edge = find_transition_edge(state, symbol);
    return edge == -1
        ? -1
        : edges_.next_state(edge);
  }

  void add_transition(int32_t state, uint8_t symbol, int32_t next_state) {
    TORCH_CHECK(
        edges_.size() <
            static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "ROSA suffix-automaton edge count exceeded int32 range");
    const int32_t new_edge = static_cast<int32_t>(edges_.size());
    const int32_t previous_edge = transition_head(state);
    edges_.push_back(symbol, next_state, previous_edge);
    ROSA_TRACE_STATE(state, false, kRosaTraceExtension);
    AutomatonState& target = states_[static_cast<size_t>(state)];
    if (has_transition_index(target.first_edge)) {
      const size_t index_id = transition_index_id(target.first_edge);
      ROSA_TRACE_TRANSITION_META(index_id, true, kRosaTraceIndex);
      TransitionIndex& index =
          transition_indices_[index_id];
      index.first_edge = new_edge;
      insert_indexed_transition(index, new_edge, true);
    } else {
      ROSA_TRACE_STATE(state, true, kRosaTraceExtension);
      target.first_edge = new_edge;
      maybe_index_transitions(state);
    }
    if (state != 0) {
      return;
    }

    ++root_transition_count_;
    if (root_transition_edges_ != nullptr) {
      ROSA_TRACE_ROOT(
          static_cast<int64_t>(symbol) * sizeof(int32_t),
          sizeof(int32_t),
          true,
          kRosaTraceIndex);
      (*root_transition_edges_)[symbol] = new_edge;
      return;
    }
    constexpr int32_t kRootTableThreshold = 16;
    if (root_transition_count_ < kRootTableThreshold) {
      return;
    }
    root_transition_edges_ =
        std::make_unique<std::array<int32_t, 256>>();
    root_transition_edges_->fill(-1);
    ROSA_TRACE_ROOT(
        0,
        sizeof(*root_transition_edges_),
        true,
        kRosaTraceAppend);
    int32_t edge = transition_head(0);
    while (edge != -1) {
      const uint8_t edge_symbol = edges_.symbol(edge);
      ROSA_TRACE_ROOT(
          static_cast<int64_t>(edge_symbol) * sizeof(int32_t),
          sizeof(int32_t),
          true,
          kRosaTraceIndex);
      (*root_transition_edges_)[edge_symbol] = edge;
      edge = edges_.next_edge(edge);
    }
    ++complexity_.root_table_activations;
  }

  int32_t copy_edges(int32_t source_edge) {
    int32_t first_copy = -1;
    int32_t previous_copy = -1;
    while (source_edge != -1) {
      TORCH_CHECK(
          edges_.size() <
              static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "ROSA suffix-automaton edge count exceeded int32 range");
      const uint8_t source_symbol = edges_.symbol(source_edge);
      const int32_t source_next_state = edges_.next_state(source_edge);
      const int32_t next_source_edge = edges_.next_edge(source_edge);
      const int32_t copied_edge = static_cast<int32_t>(edges_.size());
      edges_.push_back(source_symbol, source_next_state, -1);
      ++complexity_.copied_edges;
      if (previous_copy == -1) {
        first_copy = copied_edge;
      } else {
        edges_.set_next_edge(previous_copy, copied_edge);
      }
      previous_copy = copied_edge;
      source_edge = next_source_edge;
    }
    return first_copy;
  }

  int32_t match_query(uint8_t symbol) {
    int32_t state = query_state_;
    int32_t next = find_transition(state, symbol);
    int64_t suffix_link_steps = 0;
    while (state != 0 && next == -1) {
      ROSA_TRACE_STATE(state, false, kRosaTraceQuery);
      state = states_[static_cast<size_t>(state)].suffix_link;
      ++suffix_link_steps;
      next = find_transition(state, symbol);
    }
    complexity_.query_suffix_link_steps += suffix_link_steps;
    complexity_.max_query_suffix_link_steps = std::max(
        complexity_.max_query_suffix_link_steps,
        suffix_link_steps);

    if (next == -1) {
      query_state_ = 0;
      return -1;
    }

    query_state_ = next;
    if (latest_end_tree_ == nullptr) {
      ROSA_TRACE_STATE(query_state_, false, kRosaTraceQuery);
      return states_[static_cast<size_t>(query_state_)].latest_end;
    }
    return latest_end_tree_->point_query(query_state_);
  }

  void link_latest_node(int32_t child, int32_t parent) {
    if (latest_end_tree_ != nullptr) {
      latest_end_tree_->link(child, parent);
    }
  }

  void maybe_activate_latest_end_tree(int64_t latest_end_chain) {
    constexpr int64_t kMinChainLength = 64;
    constexpr int64_t kWritesPerState = 16;
    if (
        latest_end_tree_ != nullptr ||
        latest_end_chain < kMinChainLength ||
        complexity_.latest_end_updates <
            kWritesPerState * static_cast<int64_t>(states_.size())) {
      return;
    }
    latest_end_tree_ = std::make_unique<LatestEndTree>(
        states_,
        &complexity_.latest_tree_rotations);
    ++complexity_.latest_tree_activations;
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
    ROSA_TRACE_STATE(next_state, true, kRosaTraceAppend);
    if (latest_end_tree_ != nullptr) {
      TORCH_INTERNAL_ASSERT(latest_end_tree_->add_node(-1) == next_state);
    }
    ROSA_TRACE_STATE(last_key_state_, false, kRosaTraceExtension);
    ROSA_TRACE_STATE(next_state, true, kRosaTraceExtension);
    states_[static_cast<size_t>(next_state)].max_length =
        states_[static_cast<size_t>(last_key_state_)].max_length + 1;

    int32_t parent = last_key_state_;
    int32_t child = -1;
    while (parent != -1) {
      ++complexity_.extension_suffix_link_steps;
      const int32_t edge = find_transition_edge(parent, symbol);
      if (edge != -1) {
        child = edges_.next_state(edge);
        break;
      }
      add_transition(parent, symbol, next_state);
      ROSA_TRACE_STATE(parent, false, kRosaTraceExtension);
      parent = states_[static_cast<size_t>(parent)].suffix_link;
    }

    if (parent == -1) {
      ROSA_TRACE_STATE(next_state, true, kRosaTraceExtension);
      states_[static_cast<size_t>(next_state)].suffix_link = 0;
      link_latest_node(next_state, 0);
    } else {
      ROSA_TRACE_STATE(parent, false, kRosaTraceExtension);
      ROSA_TRACE_STATE(child, false, kRosaTraceExtension);
      if (
          states_[static_cast<size_t>(parent)].max_length + 1 ==
          states_[static_cast<size_t>(child)].max_length) {
        ROSA_TRACE_STATE(next_state, true, kRosaTraceExtension);
        states_[static_cast<size_t>(next_state)].suffix_link = child;
        link_latest_node(next_state, child);
      } else {
        TORCH_CHECK(
            states_.size() < kMaxInt32,
            "ROSA suffix-automaton state count exceeded int32 range");
        const int32_t clone = static_cast<int32_t>(states_.size());
        ROSA_TRACE_STATE(child, false, kRosaTraceClone);
        states_.push_back(states_[static_cast<size_t>(child)]);
        ROSA_TRACE_STATE(clone, true, kRosaTraceClone);
        int32_t clone_latest_end;
        if (latest_end_tree_ == nullptr) {
          ROSA_TRACE_STATE(child, false, kRosaTraceClone);
          clone_latest_end =
              states_[static_cast<size_t>(child)].latest_end;
        } else {
          clone_latest_end = latest_end_tree_->point_query(child);
        }
        ROSA_TRACE_STATE(clone, true, kRosaTraceClone);
        states_[static_cast<size_t>(clone)].latest_end = clone_latest_end;
        if (latest_end_tree_ != nullptr) {
          TORCH_INTERNAL_ASSERT(
              latest_end_tree_->add_node(clone_latest_end) == clone);
        }
        ++complexity_.clones;
        ROSA_TRACE_STATE(parent, false, kRosaTraceClone);
        ROSA_TRACE_STATE(clone, true, kRosaTraceClone);
        states_[static_cast<size_t>(clone)].max_length =
            states_[static_cast<size_t>(parent)].max_length + 1;
        ROSA_TRACE_STATE(clone, true, kRosaTraceClone);
        states_[static_cast<size_t>(clone)].first_edge =
            copy_edges(transition_head(child));
        maybe_index_transitions(clone);
        ROSA_TRACE_STATE(child, false, kRosaTraceClone);
        const int32_t old_suffix_link =
            states_[static_cast<size_t>(child)].suffix_link;
        ROSA_TRACE_STATE(child, true, kRosaTraceClone);
        ROSA_TRACE_STATE(next_state, true, kRosaTraceClone);
        states_[static_cast<size_t>(child)].suffix_link = clone;
        states_[static_cast<size_t>(next_state)].suffix_link = clone;
        if (latest_end_tree_ != nullptr) {
          latest_end_tree_->link(clone, old_suffix_link);
          latest_end_tree_->reparent(child, clone);
          latest_end_tree_->link(next_state, clone);
        }

        while (parent != -1) {
          ++complexity_.extension_suffix_link_steps;
          const int32_t edge = find_transition_edge(parent, symbol);
          if (
              edge == -1 ||
              edges_.next_state(edge) != child) {
            break;
          }
          edges_.set_next_state(edge, clone);
          ROSA_TRACE_STATE(parent, false, kRosaTraceClone);
          parent = states_[static_cast<size_t>(parent)].suffix_link;
        }
      }
    }

    last_key_state_ = next_state;
    if (latest_end_tree_ != nullptr) {
      latest_end_tree_->assign_root_path(next_state, end_position);
      ++complexity_.latest_path_updates;
      return;
    }

    int32_t state = next_state;
    int64_t latest_end_chain = 0;
    while (state != -1) {
      ROSA_TRACE_STATE(state, true, kRosaTraceLatest);
      states_[static_cast<size_t>(state)].latest_end = end_position;
      ++latest_end_chain;
      ROSA_TRACE_STATE(state, false, kRosaTraceLatest);
      state = states_[static_cast<size_t>(state)].suffix_link;
    }
    complexity_.latest_end_updates += latest_end_chain;
    complexity_.max_latest_end_chain = std::max(
        complexity_.max_latest_end_chain,
        latest_end_chain);
    maybe_activate_latest_end_tree(latest_end_chain);
  }

  std::vector<AutomatonState> states_;
  AutomatonEdges edges_;
  int32_t query_state_ = 0;
  int32_t last_key_state_ = 0;
  int32_t key_count_ = 0;
  bool compressed_run_active_ = true;
  bool compressed_run_initialized_ = false;
  uint8_t compressed_symbol_ = 0;
  int32_t compressed_run_length_ = 0;
  int32_t compressed_query_length_ = 0;
  std::unique_ptr<LatestEndTree> latest_end_tree_;
  int32_t root_transition_count_ = 0;
  std::unique_ptr<std::array<int32_t, 256>> root_transition_edges_;
  std::vector<TransitionIndex> transition_indices_;
  mutable AutomatonComplexityStats complexity_;
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

void check_matched_positions_tensor(
    const torch::Tensor& tensor,
    int64_t total_tokens,
    int64_t heads) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "matched_key_end_positions must be a CPU tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt64,
      "matched_key_end_positions must have dtype torch.int64");
  TORCH_CHECK(
      tensor.dim() == 2 &&
          tensor.size(0) == total_tokens &&
          tensor.size(1) == heads,
      "matched_key_end_positions must be shaped [total_tokens, heads]");
  TORCH_CHECK(
      tensor.is_contiguous(),
      "matched_key_end_positions must be contiguous");
}

}  // namespace

#ifndef ROSA_RUNTIME_CORE_ONLY
class RosaRuntime : public torch::CustomClassHolder {
 public:
  RosaRuntime(
      int64_t num_heads,
      int64_t num_payload_heads,
      int64_t qk_bits,
      int64_t payload_bits)
      : num_heads_(num_heads),
        num_payload_heads_(num_payload_heads),
        qk_bits_(qk_bits),
        payload_bits_(payload_bits) {
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
  }

  std::tuple<torch::Tensor, torch::Tensor> update_packed(
      const torch::Tensor& cu_seqlens,
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& payload) {
    auto output = torch::empty(
        {query.size(0), num_heads_},
        payload.options());
    auto matched_key_end_positions = torch::empty(
        {query.size(0), num_heads_},
        payload.options().dtype(torch::kInt64));
    return update_packed_out(
        cu_seqlens,
        query,
        key,
        payload,
        output,
        matched_key_end_positions);
  }

  std::tuple<torch::Tensor, torch::Tensor> update_packed_out(
      const torch::Tensor& cu_seqlens,
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& payload,
      const torch::Tensor& output,
      const torch::Tensor& matched_key_end_positions) {
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
    check_packed_tensor(
        output,
        "output",
        total_tokens,
        num_heads_);
    check_matched_positions_tensor(
        matched_key_end_positions,
        total_tokens,
        num_heads_);
    ensure_automata(slot_count);

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

  std::tuple<
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t>
  complexity_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    AutomatonComplexityStats total;
    for (const auto& automaton : automata_) {
      const AutomatonComplexityStats current =
          automaton->complexity_stats();
      total.updates += current.updates;
      total.transition_probes += current.transition_probes;
      total.query_suffix_link_steps += current.query_suffix_link_steps;
      total.extension_suffix_link_steps +=
          current.extension_suffix_link_steps;
      total.latest_end_updates += current.latest_end_updates;
      total.clones += current.clones;
      total.copied_edges += current.copied_edges;
      total.max_query_suffix_link_steps = std::max(
          total.max_query_suffix_link_steps,
          current.max_query_suffix_link_steps);
      total.max_latest_end_chain = std::max(
          total.max_latest_end_chain,
          current.max_latest_end_chain);
      total.latest_path_updates += current.latest_path_updates;
      total.latest_tree_rotations += current.latest_tree_rotations;
      total.latest_tree_activations += current.latest_tree_activations;
      total.compressed_run_symbols += current.compressed_run_symbols;
      total.compressed_materializations +=
          current.compressed_materializations;
      total.max_compressed_run_length = std::max(
          total.max_compressed_run_length,
          current.max_compressed_run_length);
      total.root_table_lookups += current.root_table_lookups;
      total.root_table_activations += current.root_table_activations;
      total.transition_index_lookups += current.transition_index_lookups;
      total.transition_index_activations +=
          current.transition_index_activations;
    }
    return {
        total.updates,
        total.transition_probes,
        total.query_suffix_link_steps,
        total.extension_suffix_link_steps,
        total.latest_end_updates,
        total.clones,
        total.copied_edges,
        total.max_query_suffix_link_steps,
        total.max_latest_end_chain,
        total.latest_path_updates,
        total.latest_tree_rotations,
        total.latest_tree_activations,
        total.compressed_run_symbols,
        total.compressed_materializations,
        total.max_compressed_run_length,
        total.root_table_lookups,
        total.root_table_activations,
        total.transition_index_lookups,
        total.transition_index_activations,
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
        automata_.push_back(std::make_unique<SuffixAutomaton>());
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
          int64_t>())
      .def("update_packed", &RosaRuntime::update_packed)
      .def("update_packed_out", &RosaRuntime::update_packed_out)
      .def("reset", &RosaRuntime::reset)
      .def("close", &RosaRuntime::close)
      .def("stats", &RosaRuntime::stats)
      .def("complexity_stats", &RosaRuntime::complexity_stats);
}
#endif
