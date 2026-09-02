#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rosa_soft {

using SamStateId = int32_t;
using SamEdgeId = int32_t;
using SamPosition = int32_t;
using SamSymbol = uint32_t;

constexpr SamStateId kNoState = -1;
constexpr SamEdgeId kNoEdge = -1;
constexpr SamPosition kNoMatch = -1;

struct SamState {
  int32_t max_length = 0;
  SamStateId suffix_link = kNoState;
  SamPosition latest_end = kNoMatch;
  SamEdgeId first_edge = kNoEdge;
};

struct SamEdge {
  SamSymbol symbol = 0;
  SamStateId target = kNoState;
  SamEdgeId next = kNoEdge;
};

// Exact online ROSA routing state for one query/key head.
// Query matching happens before the key at the same position is appended.
class RosaSuffixAutomaton {
 public:
  RosaSuffixAutomaton() { states_.emplace_back(); }

  SamPosition match_then_append(SamSymbol query, SamSymbol key) {
    const SamPosition matched_end = match(query);
    append(key);
    return matched_end;
  }

  void reserve_additional(size_t token_count) {
    const size_t max_id =
        static_cast<size_t>(std::numeric_limits<int32_t>::max());
    if (token_count > (max_id - states_.size()) / 2) {
      throw std::length_error("ROSA SAM state count exceeds int32 range");
    }
    states_.reserve(states_.size() + 2 * token_count);
    if (token_count <= max_id - edges_.size()) {
      edges_.reserve(edges_.size() + token_count);
    }
  }

 private:
  SamEdgeId find_edge(SamStateId state, SamSymbol symbol) const {
    SamEdgeId edge = states_[static_cast<size_t>(state)].first_edge;
    while (edge != kNoEdge) {
      const SamEdge& candidate = edges_[static_cast<size_t>(edge)];
      if (candidate.symbol == symbol) {
        return edge;
      }
      edge = candidate.next;
    }
    return kNoEdge;
  }

  SamStateId find_transition(SamStateId state, SamSymbol symbol) const {
    const SamEdgeId edge = find_edge(state, symbol);
    return edge == kNoEdge
        ? kNoState
        : edges_[static_cast<size_t>(edge)].target;
  }

  void set_transition(
      SamStateId state,
      SamSymbol symbol,
      SamStateId target) {
    const SamEdgeId edge = find_edge(state, symbol);
    if (edge != kNoEdge) {
      edges_[static_cast<size_t>(edge)].target = target;
      return;
    }
    if (edges_.size() >=
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::length_error("ROSA SAM edge count exceeds int32 range");
    }
    const SamEdgeId new_edge = static_cast<SamEdgeId>(edges_.size());
    edges_.push_back({
        symbol,
        target,
        states_[static_cast<size_t>(state)].first_edge,
    });
    states_[static_cast<size_t>(state)].first_edge = new_edge;
  }

  SamEdgeId copy_edges(SamEdgeId source_edge) {
    SamEdgeId first_copy = kNoEdge;
    SamEdgeId previous_copy = kNoEdge;
    while (source_edge != kNoEdge) {
      if (edges_.size() >=
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("ROSA SAM edge count exceeds int32 range");
      }
      const SamEdge source = edges_[static_cast<size_t>(source_edge)];
      const SamEdgeId copy = static_cast<SamEdgeId>(edges_.size());
      edges_.push_back({source.symbol, source.target, kNoEdge});
      if (previous_copy == kNoEdge) {
        first_copy = copy;
      } else {
        edges_[static_cast<size_t>(previous_copy)].next = copy;
      }
      previous_copy = copy;
      source_edge = source.next;
    }
    return first_copy;
  }

  SamPosition match(SamSymbol symbol) {
    SamStateId state = query_state_;
    SamStateId next = find_transition(state, symbol);
    while (state != 0 && next == kNoState) {
      state = states_[static_cast<size_t>(state)].suffix_link;
      next = find_transition(state, symbol);
    }
    if (next == kNoState) {
      query_state_ = 0;
      return kNoMatch;
    }
    query_state_ = next;
    return states_[static_cast<size_t>(next)].latest_end;
  }

  void append(SamSymbol symbol) {
    const size_t max_id =
        static_cast<size_t>(std::numeric_limits<int32_t>::max());
    if (
        states_.size() >= max_id ||
        key_count_ == std::numeric_limits<int32_t>::max()) {
      throw std::length_error("ROSA SAM sequence exceeds int32 range");
    }

    const SamPosition end_position = key_count_++;
    const SamStateId next_state = static_cast<SamStateId>(states_.size());
    states_.emplace_back();
    states_[static_cast<size_t>(next_state)].max_length =
        states_[static_cast<size_t>(last_key_state_)].max_length + 1;

    SamStateId parent = last_key_state_;
    while (
        parent != kNoState &&
        find_transition(parent, symbol) == kNoState) {
      set_transition(parent, symbol, next_state);
      parent = states_[static_cast<size_t>(parent)].suffix_link;
    }

    if (parent == kNoState) {
      states_[static_cast<size_t>(next_state)].suffix_link = 0;
    } else {
      const SamStateId child = find_transition(parent, symbol);
      if (
          states_[static_cast<size_t>(parent)].max_length + 1 ==
          states_[static_cast<size_t>(child)].max_length) {
        states_[static_cast<size_t>(next_state)].suffix_link = child;
      } else {
        if (states_.size() >= max_id) {
          throw std::length_error("ROSA SAM state count exceeds int32 range");
        }
        const SamStateId clone = static_cast<SamStateId>(states_.size());
        states_.push_back(states_[static_cast<size_t>(child)]);
        states_[static_cast<size_t>(clone)].max_length =
            states_[static_cast<size_t>(parent)].max_length + 1;
        states_[static_cast<size_t>(clone)].first_edge =
            copy_edges(states_[static_cast<size_t>(child)].first_edge);
        states_[static_cast<size_t>(child)].suffix_link = clone;
        states_[static_cast<size_t>(next_state)].suffix_link = clone;

        while (
            parent != kNoState &&
            find_transition(parent, symbol) == child) {
          set_transition(parent, symbol, clone);
          parent = states_[static_cast<size_t>(parent)].suffix_link;
        }
      }
    }

    last_key_state_ = next_state;
    SamStateId state = next_state;
    while (state != kNoState) {
      states_[static_cast<size_t>(state)].latest_end = end_position;
      state = states_[static_cast<size_t>(state)].suffix_link;
    }
  }

  std::vector<SamState> states_;
  std::vector<SamEdge> edges_;
  SamStateId query_state_ = 0;
  SamStateId last_key_state_ = 0;
  SamPosition key_count_ = 0;
};

}  // namespace rosa_soft
