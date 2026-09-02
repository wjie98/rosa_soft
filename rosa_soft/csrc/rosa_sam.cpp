#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "rosa_sam_core.h"

namespace {

std::vector<int64_t> read_offsets(const torch::Tensor& cu_seqlens) {
  TORCH_CHECK(
      cu_seqlens.device().is_cpu(),
      "cu_seqlens must be on CPU");
  TORCH_CHECK(
      cu_seqlens.scalar_type() == torch::kInt32 ||
          cu_seqlens.scalar_type() == torch::kInt64,
      "cu_seqlens must have dtype int32 or int64");
  TORCH_CHECK(
      cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
      "cu_seqlens must be a 1D tensor with at least two entries");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");

  std::vector<int64_t> offsets(
      static_cast<size_t>(cu_seqlens.numel()));
  AT_DISPATCH_INTEGRAL_TYPES(
      cu_seqlens.scalar_type(),
      "rosa_sam_read_offsets",
      [&] {
        const auto* data = cu_seqlens.data_ptr<scalar_t>();
        for (int64_t index = 0; index < cu_seqlens.numel(); ++index) {
          offsets[static_cast<size_t>(index)] =
              static_cast<int64_t>(data[index]);
        }
      });
  TORCH_CHECK(offsets.front() == 0, "cu_seqlens must start at zero");
  for (size_t index = 1; index < offsets.size(); ++index) {
    TORCH_CHECK(
        offsets[index] >= offsets[index - 1],
        "cu_seqlens must be nondecreasing");
  }
  return offsets;
}

void check_symbols(
    const torch::Tensor& tensor,
    const char* name,
    int64_t total_tokens,
    int64_t num_heads) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt32,
      name,
      " must have dtype int32");
  TORCH_CHECK(
      tensor.dim() == 2 &&
          tensor.size(0) == total_tokens &&
          tensor.size(1) == num_heads,
      name,
      " must have shape [total_tokens, num_heads]");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

class RosaSam : public torch::CustomClassHolder {
 public:
  RosaSam(int64_t num_heads, int64_t symbol_bits)
      : num_heads_(num_heads) {
    TORCH_CHECK(num_heads_ > 0, "num_heads must be positive");
    TORCH_CHECK(
        symbol_bits >= 1 && symbol_bits <= 32,
        "symbol_bits must be in [1, 32]");
    symbol_mask_ = symbol_bits == 32
        ? std::numeric_limits<uint32_t>::max()
        : (uint32_t{1} << symbol_bits) - 1;
  }

  torch::Tensor update_packed(
      const torch::Tensor& cu_seqlens,
      const torch::Tensor& query,
      const torch::Tensor& key) {
    const std::vector<int64_t> offsets = read_offsets(cu_seqlens);
    const int64_t sequence_count =
        static_cast<int64_t>(offsets.size()) - 1;
    const int64_t total_tokens = offsets.back();
    TORCH_CHECK(total_tokens >= 0, "total token count must be non-negative");
    check_symbols(query, "query", total_tokens, num_heads_);
    check_symbols(key, "key", total_tokens, num_heads_);
    ensure_automata(sequence_count);

    for (int64_t sequence = 0; sequence < sequence_count; ++sequence) {
      const int64_t token_count =
          offsets[static_cast<size_t>(sequence + 1)] -
          offsets[static_cast<size_t>(sequence)];
      TORCH_CHECK(
          token_count <= std::numeric_limits<int32_t>::max(),
          "ROSA SAM chunk length exceeds int32 range");
      for (int64_t head = 0; head < num_heads_; ++head) {
        automata_[automaton_index(sequence, head)].reserve_additional(
            static_cast<size_t>(token_count));
      }
    }

    auto matched_key_end = torch::empty(
        {total_tokens, num_heads_},
        query.options().dtype(torch::kInt64));
    const int32_t* query_data = query.data_ptr<int32_t>();
    const int32_t* key_data = key.data_ptr<int32_t>();
    int64_t* output_data = matched_key_end.data_ptr<int64_t>();

    at::parallel_for(
        0,
        sequence_count * num_heads_,
        1,
        [&](int64_t begin, int64_t end) {
          for (int64_t task = begin; task < end; ++task) {
            const int64_t sequence = task / num_heads_;
            const int64_t head = task % num_heads_;
            rosa_soft::RosaSuffixAutomaton& automaton =
                automata_[automaton_index(sequence, head)];
            const int64_t token_begin =
                offsets[static_cast<size_t>(sequence)];
            const int64_t token_end =
                offsets[static_cast<size_t>(sequence + 1)];
            for (int64_t token = token_begin; token < token_end; ++token) {
              const int64_t index = token * num_heads_ + head;
              const uint32_t query_symbol =
                  static_cast<uint32_t>(query_data[index]) & symbol_mask_;
              const uint32_t key_symbol =
                  static_cast<uint32_t>(key_data[index]) & symbol_mask_;
              output_data[index] = static_cast<int64_t>(
                  automaton.match_then_append(query_symbol, key_symbol));
            }
          }
        });
    return matched_key_end;
  }

  void reset() {
    automata_.clear();
    sequence_count_ = -1;
  }

 private:
  size_t automaton_index(int64_t sequence, int64_t head) const {
    return static_cast<size_t>(sequence * num_heads_ + head);
  }

  void ensure_automata(int64_t sequence_count) {
    if (sequence_count_ == -1) {
      sequence_count_ = sequence_count;
      automata_.resize(
          static_cast<size_t>(sequence_count_ * num_heads_));
      return;
    }
    TORCH_CHECK(
        sequence_count == sequence_count_,
        "RosaSam sequence count is fixed after its first update; call reset() "
        "before changing it");
  }

  const int64_t num_heads_;
  uint32_t symbol_mask_ = 0;
  int64_t sequence_count_ = -1;
  std::vector<rosa_soft::RosaSuffixAutomaton> automata_;
};

TORCH_LIBRARY_FRAGMENT(rosa_soft, m) {
  m.class_<RosaSam>("RosaSam")
      .def(torch::init<int64_t, int64_t>())
      .def("update_packed", &RosaSam::update_packed)
      .def("reset", &RosaSam::reset);
}
