#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Routes {
  std::vector<uint32_t> route;
  std::vector<uint32_t> length;
  uint64_t comparisons = 0;
};

struct Options {
  uint32_t sequence_length = 4096;
  uint32_t bit_width = 8;
  std::string pattern = "random";
  double clarity = 0.0;
  uint32_t seed = 17;
  uint32_t repeats = 7;
};

uint32_t parse_u32(const char *value, const char *name) {
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed > UINT32_MAX) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return static_cast<uint32_t>(parsed);
}

double parse_double(const char *value, const char *name) {
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0') {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return parsed;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("every option requires one value");
    }
    const std::string name = argv[index];
    const char *value = argv[index + 1];
    if (name == "--sequence-length") {
      options.sequence_length = parse_u32(value, "sequence length");
    } else if (name == "--bit-width") {
      options.bit_width = parse_u32(value, "bit width");
    } else if (name == "--pattern") {
      options.pattern = value;
    } else if (name == "--clarity") {
      options.clarity = parse_double(value, "clarity");
    } else if (name == "--seed") {
      options.seed = parse_u32(value, "seed");
    } else if (name == "--repeats") {
      options.repeats = parse_u32(value, "repeats");
    } else {
      throw std::invalid_argument("unknown option: " + name);
    }
  }
  if (options.sequence_length == 0 || options.bit_width == 0 ||
      options.bit_width > 8 || options.repeats == 0 || options.clarity < 0.0 ||
      options.clarity > 1.0) {
    throw std::invalid_argument("invalid benchmark configuration");
  }
  return options;
}

std::pair<std::vector<uint8_t>, std::vector<uint8_t>> make_codes(
    const Options &options) {
  const uint32_t alphabet = uint32_t{1} << options.bit_width;
  std::mt19937 generator(options.seed);
  std::uniform_int_distribution<uint32_t> symbol(0, alphabet - 1);
  std::bernoulli_distribution select_target(options.clarity);
  std::vector<uint8_t> query(options.sequence_length);
  std::vector<uint8_t> key(options.sequence_length);
  for (uint32_t index = 0; index < options.sequence_length; ++index) {
    query[index] = static_cast<uint8_t>(symbol(generator));
    key[index] = static_cast<uint8_t>(symbol(generator));
  }
  if (options.pattern == "random") {
    return {query, key};
  }

  std::vector<uint8_t> target_key(options.sequence_length);
  if (options.pattern == "shift_random") {
    for (uint8_t &value : target_key) {
      value = static_cast<uint8_t>(symbol(generator));
    }
  } else if (options.pattern == "shift_motif") {
    const uint32_t motif_length = std::min<uint32_t>(16, std::max<uint32_t>(2, options.sequence_length / 8));
    std::vector<uint8_t> motif(motif_length);
    for (uint8_t &value : motif) {
      value = static_cast<uint8_t>(symbol(generator));
    }
    for (uint32_t index = 0; index < options.sequence_length; ++index) {
      target_key[index] = motif[index % motif_length];
    }
  } else if (options.pattern == "collapse") {
    std::fill(target_key.begin(), target_key.end(), uint8_t{0});
  } else if (options.pattern == "period4") {
    for (uint32_t index = 0; index < options.sequence_length; ++index) {
      target_key[index] = static_cast<uint8_t>(index % std::min<uint32_t>(4, alphabet));
    }
  } else {
    throw std::invalid_argument("unknown pattern: " + options.pattern);
  }
  std::vector<uint8_t> target_query = target_key;
  for (uint32_t index = 1; index < options.sequence_length; ++index) {
    target_query[index] = target_key[index - 1];
  }
  for (uint32_t index = 0; index < options.sequence_length; ++index) {
    if (select_target(generator)) {
      query[index] = target_query[index];
    }
    if (select_target(generator)) {
      key[index] = target_key[index];
    }
  }
  return {query, key};
}

Routes diagonal_routes(const std::vector<uint8_t> &query,
                       const std::vector<uint8_t> &key) {
  const uint32_t length = static_cast<uint32_t>(query.size());
  Routes result{std::vector<uint32_t>(length), std::vector<uint32_t>(length), 0};
  std::vector<uint32_t> previous(length, 0);
  std::vector<uint32_t> current(length, 0);
  for (uint32_t query_index = 1; query_index < length; ++query_index) {
    std::fill(current.begin(), current.begin() + query_index + 1, 0);
    uint32_t best_route = 0;
    uint32_t best_length = 0;
    for (uint32_t route = 1; route <= query_index; ++route) {
      ++result.comparisons;
      if (query[query_index] == key[route - 1]) {
        current[route] = previous[route - 1] + 1;
      }
      const uint32_t candidate = current[route];
      if (candidate > best_length ||
          (candidate == best_length && candidate > 0)) {
        best_route = route;
        best_length = candidate;
      }
    }
    result.route[query_index] = best_route;
    result.length[query_index] = best_length;
    previous.swap(current);
  }
  return result;
}

Routes direct_routes(const std::vector<uint8_t> &query,
                     const std::vector<uint8_t> &key) {
  const uint32_t length = static_cast<uint32_t>(query.size());
  Routes result{std::vector<uint32_t>(length), std::vector<uint32_t>(length), 0};
  for (uint32_t query_index = 1; query_index < length; ++query_index) {
    uint32_t best_route = 0;
    uint32_t best_length = 0;
    for (uint32_t route = 1; route <= query_index; ++route) {
      uint32_t candidate = 0;
      while (candidate < route) {
        ++result.comparisons;
        if (query[query_index - candidate] != key[route - 1 - candidate]) {
          break;
        }
        ++candidate;
      }
      if (candidate > best_length ||
          (candidate == best_length && candidate > 0)) {
        best_route = route;
        best_length = candidate;
      }
    }
    result.route[query_index] = best_route;
    result.length[query_index] = best_length;
  }
  return result;
}

template <typename Callable>
double median_ms(Callable &&callable, uint32_t repeats, Routes &last) {
  std::vector<double> samples;
  samples.reserve(repeats);
  for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    last = callable();
    const auto stop = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

uint64_t checksum(const Routes &routes) {
  uint64_t result = 0;
  for (size_t index = 0; index < routes.route.size(); ++index) {
    result += (index + 1) * (routes.route[index] + 3 * routes.length[index]);
  }
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto [query, key] = make_codes(options);
    const Routes direct_warmup = direct_routes(query, key);
    const Routes diagonal_warmup = diagonal_routes(query, key);
    if (direct_warmup.route != diagonal_warmup.route ||
        direct_warmup.length != diagonal_warmup.length) {
      throw std::runtime_error("hard-route implementations disagree");
    }
    Routes direct;
    Routes diagonal;
    const double direct_ms = median_ms(
        [&] { return direct_routes(query, key); }, options.repeats, direct);
    const double diagonal_ms = median_ms(
        [&] { return diagonal_routes(query, key); }, options.repeats, diagonal);
    const double direct_rate = direct.comparisons / (direct_ms * 1.0e6);
    const double diagonal_rate = diagonal.comparisons / (diagonal_ms * 1.0e6);
    std::cout << "{\n"
              << "  \"sequence_length\": " << options.sequence_length << ",\n"
              << "  \"bit_width\": " << options.bit_width << ",\n"
              << "  \"pattern\": \"" << options.pattern << "\",\n"
              << "  \"clarity\": " << options.clarity << ",\n"
              << "  \"direct_ms\": " << direct_ms << ",\n"
              << "  \"diagonal_ms\": " << diagonal_ms << ",\n"
              << "  \"direct_comparisons\": " << direct.comparisons << ",\n"
              << "  \"diagonal_comparisons\": " << diagonal.comparisons << ",\n"
              << "  \"direct_gcomparisons_per_second\": " << direct_rate << ",\n"
              << "  \"diagonal_gcomparisons_per_second\": " << diagonal_rate << ",\n"
              << "  \"checksum\": " << checksum(direct) << "\n"
              << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
