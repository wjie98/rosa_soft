#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rosa_soft {

constexpr int kNone = -1;

struct State {
  int len = 0;
  int link = kNone;
  int end = kNone;
  int edge = kNone;
};

struct Edge {
  uint32_t c = 0;
  int to = kNone;
  int next = kNone;
};

class Sam {
 public:
  Sam() { s_.emplace_back(); }

  int step(uint32_t q, uint32_t k) {
    const int end = match(q);
    extend(k);
    return end;
  }

  void reserve(size_t n) {
    const size_t limit = std::numeric_limits<int>::max();
    if (n > (limit - s_.size()) / 2) {
      throw std::length_error("ROSA SAM exceeds int32 state space");
    }
    // Small decode chunks must not force a reallocation at every token.
    auto grow = [limit](auto& v, size_t add) {
      const size_t need = v.size() + add;
      if (need > v.capacity())
        v.reserve(std::max(need, std::min(limit, 2 * v.capacity())));
    };
    grow(s_, 2 * n);
    grow(e_, n);
  }

 private:
  int edge(int p, uint32_t c) const {
    for (int i = s_[p].edge; i != kNone; i = e_[i].next) {
      if (e_[i].c == c) return i;
    }
    return kNone;
  }

  int go(int p, uint32_t c) const {
    const int i = edge(p, c);
    return i == kNone ? kNone : e_[i].to;
  }

  void set(int p, uint32_t c, int q) {
    const int i = edge(p, c);
    if (i != kNone) {
      e_[i].to = q;
      return;
    }
    if (e_.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::length_error("ROSA SAM exceeds int32 edge space");
    }
    const int next = s_[p].edge;
    s_[p].edge = static_cast<int>(e_.size());
    e_.push_back({c, q, next});
  }

  int copy_edges(int i) {
    int first = kNone;
    int prev = kNone;
    while (i != kNone) {
      if (e_.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("ROSA SAM exceeds int32 edge space");
      }
      const Edge x = e_[i];
      const int copy = static_cast<int>(e_.size());
      e_.push_back({x.c, x.to, kNone});
      if (prev == kNone) first = copy;
      else e_[prev].next = copy;
      prev = copy;
      i = x.next;
    }
    return first;
  }

  int match(uint32_t c) {
    int p = q_;
    int n = go(p, c);
    while (p && n == kNone) {
      p = s_[p].link;
      n = go(p, c);
    }
    q_ = n == kNone ? 0 : n;
    return n == kNone ? kNone : s_[n].end;
  }

  void extend(uint32_t c) {
    if (s_.size() >= static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n_ == std::numeric_limits<int>::max()) {
      throw std::length_error("ROSA SAM sequence exceeds int32 range");
    }
    const int end = n_++;
    const int cur = static_cast<int>(s_.size());
    s_.emplace_back();
    s_[cur].len = s_[last_].len + 1;

    int p = last_;
    while (p != kNone && go(p, c) == kNone) {
      set(p, c, cur);
      p = s_[p].link;
    }
    if (p == kNone) {
      s_[cur].link = 0;
    } else {
      const int x = go(p, c);
      if (s_[p].len + 1 == s_[x].len) {
        s_[cur].link = x;
      } else {
        const int clone = static_cast<int>(s_.size());
        s_.push_back(s_[x]);
        s_[clone].len = s_[p].len + 1;
        s_[clone].edge = copy_edges(s_[x].edge);
        s_[x].link = clone;
        s_[cur].link = clone;
        while (p != kNone && go(p, c) == x) {
          set(p, c, clone);
          p = s_[p].link;
        }
      }
    }
    last_ = cur;
    for (int x = cur; x != kNone; x = s_[x].link) s_[x].end = end;
  }

  std::vector<State> s_;
  std::vector<Edge> e_;
  int q_ = 0;
  int last_ = 0;
  int n_ = 0;
};

}  // namespace rosa_soft
