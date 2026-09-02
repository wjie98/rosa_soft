#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace rosa_runtime_page_trace {

enum Region : int64_t {
  kState = 0,
  kEdgeSymbol = 1,
  kEdgeTarget = 2,
  kEdgeNext = 3,
  kTransitionMeta = 4,
  kTransitionSlots = 5,
  kRootTable = 6,
  kLatestEndTree = 7,
  kPayload = 8,
};

enum Kind : int64_t {
  kTransition = 0,
  kQuery = 1,
  kExtension = 2,
  kLatest = 3,
  kClone = 4,
  kAppend = 5,
  kIndex = 6,
  kMaterialize = 7,
  kPayloadAccess = 8,
  kRebuild = 9,
};

constexpr int64_t kWriteFlag = 1;

struct Record {
  int64_t token;
  int64_t region;
  int64_t owner;
  int64_t page;
  int64_t flags;
  int64_t kind;
  int64_t repeats;
};

class Recorder {
 public:
  explicit Recorder(int64_t page_bytes) : page_bytes_(page_bytes) {}

  void set_token(int64_t token) {
    token_ = token;
  }

  void touch(
      int64_t region,
      int64_t owner,
      int64_t byte_offset,
      int64_t width,
      bool write,
      int64_t kind) {
    if (width <= 0) {
      return;
    }
    const int64_t first_page = byte_offset / page_bytes_;
    const int64_t last_page = (byte_offset + width - 1) / page_bytes_;
    const int64_t flags = write ? kWriteFlag : 0;
    for (int64_t page = first_page; page <= last_page; ++page) {
      append({token_, region, owner, page, flags, kind, 1});
    }
  }

  const std::vector<Record>& records() const {
    return records_;
  }

  int64_t page_touches() const {
    return page_touches_;
  }

 private:
  void append(const Record& record) {
    ++page_touches_;
    if (!records_.empty()) {
      Record& previous = records_.back();
      if (
          previous.token == record.token &&
          previous.region == record.region &&
          previous.owner == record.owner &&
          previous.page == record.page &&
          previous.flags == record.flags &&
          previous.kind == record.kind) {
        ++previous.repeats;
        return;
      }
    }
    records_.push_back(record);
  }

  int64_t page_bytes_;
  int64_t token_ = -1;
  int64_t page_touches_ = 0;
  std::vector<Record> records_;
};

thread_local Recorder* active_recorder = nullptr;

class RecorderScope {
 public:
  explicit RecorderScope(Recorder* recorder)
      : previous_(active_recorder) {
    active_recorder = recorder;
  }

  ~RecorderScope() {
    active_recorder = previous_;
  }

 private:
  Recorder* previous_;
};

inline void record_access(
    int64_t region,
    int64_t owner,
    int64_t byte_offset,
    int64_t width,
    bool write,
    int64_t kind) {
  if (active_recorder != nullptr) {
    active_recorder->touch(
        region,
        owner,
        byte_offset,
        width,
        write,
        kind);
  }
}

}  // namespace rosa_runtime_page_trace

#define kRosaTraceState rosa_runtime_page_trace::kState
#define kRosaTraceEdgeSymbol rosa_runtime_page_trace::kEdgeSymbol
#define kRosaTraceEdgeTarget rosa_runtime_page_trace::kEdgeTarget
#define kRosaTraceEdgeNext rosa_runtime_page_trace::kEdgeNext
#define kRosaTraceTransitionMeta rosa_runtime_page_trace::kTransitionMeta
#define kRosaTraceTransitionSlots rosa_runtime_page_trace::kTransitionSlots
#define kRosaTraceRootTable rosa_runtime_page_trace::kRootTable
#define kRosaTraceLatestEndTree rosa_runtime_page_trace::kLatestEndTree
#define kRosaTracePayload rosa_runtime_page_trace::kPayload

#define kRosaTraceTransition rosa_runtime_page_trace::kTransition
#define kRosaTraceQuery rosa_runtime_page_trace::kQuery
#define kRosaTraceExtension rosa_runtime_page_trace::kExtension
#define kRosaTraceLatest rosa_runtime_page_trace::kLatest
#define kRosaTraceClone rosa_runtime_page_trace::kClone
#define kRosaTraceAppend rosa_runtime_page_trace::kAppend
#define kRosaTraceIndex rosa_runtime_page_trace::kIndex
#define kRosaTraceMaterialize rosa_runtime_page_trace::kMaterialize
#define kRosaTracePayloadAccess rosa_runtime_page_trace::kPayloadAccess
#define kRosaTraceRebuild rosa_runtime_page_trace::kRebuild

#define ROSA_RUNTIME_ACCESS(region, owner, offset, width, write, kind)     \
  rosa_runtime_page_trace::record_access(                                \
      region, owner, offset, width, write, kind)
#define ROSA_RUNTIME_CORE_ONLY 1
#include "rosa_runtime.cpp"

namespace {

torch::Tensor records_to_tensor(
    const std::vector<rosa_runtime_page_trace::Record>& records) {
  auto result = torch::empty(
      {static_cast<int64_t>(records.size()), 7},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* output = result.data_ptr<int64_t>();
  for (size_t index = 0; index < records.size(); ++index) {
    const auto& record = records[index];
    const size_t offset = index * 7;
    output[offset] = record.token;
    output[offset + 1] = record.region;
    output[offset + 2] = record.owner;
    output[offset + 3] = record.page;
    output[offset + 4] = record.flags;
    output[offset + 5] = record.kind;
    output[offset + 6] = record.repeats;
  }
  return result;
}

torch::Tensor complexity_to_tensor(
    const AutomatonComplexityStats& stats) {
  return torch::tensor(
      {
          stats.updates,
          stats.transition_probes,
          stats.query_suffix_link_steps,
          stats.extension_suffix_link_steps,
          stats.latest_end_updates,
          stats.clones,
          stats.copied_edges,
          stats.max_query_suffix_link_steps,
          stats.max_latest_end_chain,
          stats.latest_path_updates,
          stats.latest_tree_rotations,
          stats.latest_tree_activations,
          stats.compressed_run_symbols,
          stats.compressed_materializations,
          stats.max_compressed_run_length,
          stats.root_table_lookups,
          stats.root_table_activations,
          stats.transition_index_lookups,
          stats.transition_index_activations,
      },
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
}

pybind11::dict trace_runtime(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& payload,
    int64_t qk_bits,
    int64_t payload_bits,
    int64_t chunk_size,
    int64_t canonical_page_bytes) {
  TORCH_CHECK(query.device().is_cpu(), "query must be on CPU");
  TORCH_CHECK(key.device().is_cpu(), "key must be on CPU");
  TORCH_CHECK(payload.device().is_cpu(), "payload must be on CPU");
  TORCH_CHECK(
      query.scalar_type() == torch::kUInt8 &&
          key.scalar_type() == torch::kUInt8 &&
          payload.scalar_type() == torch::kUInt8,
      "query, key, and payload must have dtype torch.uint8");
  TORCH_CHECK(
      query.dim() == 1 && key.dim() == 1 && payload.dim() == 1,
      "query, key, and payload must be one-dimensional");
  TORCH_CHECK(
      query.numel() == key.numel() && query.numel() == payload.numel(),
      "query, key, and payload lengths must match");
  TORCH_CHECK(
      query.is_contiguous() && key.is_contiguous() && payload.is_contiguous(),
      "query, key, and payload must be contiguous");
  TORCH_CHECK(qk_bits >= 1 && qk_bits <= 8, "qk_bits must be in [1, 8]");
  TORCH_CHECK(
      payload_bits >= 1 && payload_bits <= 8,
      "payload_bits must be in [1, 8]");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(
      canonical_page_bytes > 0 &&
          (canonical_page_bytes & (canonical_page_bytes - 1)) == 0,
      "canonical_page_bytes must be a positive power of two");
  TORCH_CHECK(
      query.numel() <= std::numeric_limits<int32_t>::max(),
      "trace length exceeds the production int32 limit");

  const int64_t tokens = query.numel();
  const uint8_t* query_data = query.data_ptr<uint8_t>();
  const uint8_t* key_data = key.data_ptr<uint8_t>();
  const uint8_t* payload_data = payload.data_ptr<uint8_t>();
  const uint8_t qk_mask = static_cast<uint8_t>(
      (uint16_t{1} << qk_bits) - 1);
  const uint8_t payload_mask = static_cast<uint8_t>(
      (uint16_t{1} << payload_bits) - 1);

  auto routes = torch::empty(
      {tokens},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto output = torch::zeros(
      {tokens},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  int64_t* route_data = routes.data_ptr<int64_t>();
  uint8_t* output_data = output.data_ptr<uint8_t>();

  rosa_runtime_page_trace::Recorder recorder(canonical_page_bytes);
  AutomatonStats automaton_stats;
  AutomatonComplexityStats complexity_stats;
  {
    rosa_runtime_page_trace::RecorderScope scope(&recorder);
    SuffixAutomaton automaton;
    std::vector<uint8_t> payload_history;
    for (int64_t begin = 0; begin < tokens; begin += chunk_size) {
      const int64_t end = std::min(tokens, begin + chunk_size);
      payload_history.reserve(
          payload_history.size() + static_cast<size_t>(end - begin));
      for (int64_t token = begin; token < end; ++token) {
        recorder.set_token(token);
        recorder.touch(
            rosa_runtime_page_trace::kPayload,
            0,
            token * static_cast<int64_t>(sizeof(uint8_t)),
            sizeof(uint8_t),
            true,
            rosa_runtime_page_trace::kAppend);
        payload_history.push_back(payload_data[token] & payload_mask);
      }
      for (int64_t token = begin; token < end; ++token) {
        recorder.set_token(token);
        const int32_t matched_end = automaton.update(
            query_data[token] & qk_mask,
            key_data[token] & qk_mask);
        route_data[token] = static_cast<int64_t>(matched_end);
        if (matched_end >= 0) {
          const size_t successor = static_cast<size_t>(matched_end) + 1;
          TORCH_INTERNAL_ASSERT(successor < payload_history.size());
          recorder.touch(
              rosa_runtime_page_trace::kPayload,
              0,
              static_cast<int64_t>(successor * sizeof(uint8_t)),
              sizeof(uint8_t),
              false,
              rosa_runtime_page_trace::kPayloadAccess);
          output_data[token] = payload_history[successor];
        }
      }
    }
    automaton_stats = automaton.stats();
    complexity_stats = automaton.complexity_stats();
  }

  auto summary = torch::tensor(
      {
          automaton_stats.states,
          automaton_stats.edges,
          automaton_stats.logical_bytes,
          recorder.page_touches(),
          static_cast<int64_t>(recorder.records().size()),
      },
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  pybind11::dict result;
  result["records"] = records_to_tensor(recorder.records());
  result["routes"] = routes;
  result["output"] = output;
  result["summary"] = summary;
  result["complexity"] = complexity_to_tensor(complexity_stats);
  result["canonical_page_bytes"] = canonical_page_bytes;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "trace_runtime",
      &trace_runtime,
      pybind11::arg("query"),
      pybind11::arg("key"),
      pybind11::arg("payload"),
      pybind11::arg("qk_bits"),
      pybind11::arg("payload_bits"),
      pybind11::arg("chunk_size"),
      pybind11::arg("canonical_page_bytes") = 4096);
}
