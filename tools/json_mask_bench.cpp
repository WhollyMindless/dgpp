// CPU-only GrammarState replay. No weights, CUDA execution or serving required.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "text/tokenizer.hpp"
#include "text/tool_grammar.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using namespace dgpp::text;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double micros(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::invalid_argument("usage: json_mask_bench TOKENIZER_JSON [ROWS=1] [ROUNDS=3]");
    const int rows = argc > 2 ? std::stoi(argv[2]) : 1;
    const int rounds = argc > 3 ? std::stoi(argv[3]) : 3;
    require(rows >= 1 && rows <= 4 && rounds >= 1 && rounds <= 100, "rows 1..4; rounds 1..100");
    auto tokenizer = Tokenizer::load(argv[1]);
    // Added-token aliases can end below the top of the dense vocabulary.
    const int64_t max_id =
        std::max(tokenizer.max_id(), static_cast<int64_t>(tokenizer.token_by_id_size()) - 1);
    require(max_id <= std::numeric_limits<int>::max() - 33, "tokenizer too large for mask staging");
    const int eos = static_cast<int>(max_id + 1);
    auto vocab = GrammarVocab::from_tokenizer(tokenizer, {eos}, eos + 1);
    const auto table_start = Clock::now();
    vocab.prepare_json();
    std::fprintf(stderr, "vocab=%d table_us=%.3f rows=%d rounds=%d\n", vocab.vocab_size(),
                 micros(table_start, Clock::now()), rows, rounds);

    const std::string plain =
        R"({"type":"object","properties":{"notes":{"type":"string"},"date":{"type":"string"},"tail":{"type":"string"}},"required":["notes","date","tail"],"additionalProperties":false})";
    const std::string mixed =
        R"({"type":"object","properties":{"notes":{"type":"string"},"date":{"type":"string","format":"date"},"tail":{"type":"string"}},"required":["notes","date","tail"],"additionalProperties":false})";
    const std::string active = R"({"type":"string","pattern":"^[A-Z]+$"})";
    std::string notes;
    for (int i = 0; i < 32; ++i) notes += "The report contains ordinary prose. ";
    const std::string document =
        "{\"notes\":\"" + notes + "\",\"date\":\"2024-02-29\",\"tail\":\"done\"}";
    const std::array<std::array<std::string, 3>, 4> cases = {
        {{"plain", plain, document},
         {"mixed", mixed, document},
         {"short", mixed, R"({"notes":"ok","date":"2024-02-29","tail":"done"})"},
         {"active", active, "\"" + std::string(128, 'A') + "\""}}};
    std::puts(
        "case,round,rows,tokens,construct_us,mask_stage_us,allows_advance_us,total_us,digest");
    for (const auto& c : cases) {
      auto ids = tokenizer.encode(c[2]);
      require(!ids.empty(), "empty tokenized document");
      require(tokenizer.decode(ids) == c[2], "tokenizer must round-trip the document");
      ids.push_back(eos);
      GrammarSpec spec;
      spec.mode = GrammarSpec::Mode::kJson;
      spec.json_schema = c[1];
      for (int round = 0; round < rounds; ++round) {
        uint64_t digest = 1469598103934665603ull;
        double mask_us = 0, advance_us = 0;
        const auto begin = Clock::now();
        GrammarState state(&vocab, spec, false);
        const double construct_us = micros(begin, Clock::now());
        std::array<TokenMask, 4> masks;
        const int stride = TokenMask::words_for(vocab.vocab_size()) + 1;
        std::vector<uint32_t> staged(static_cast<size_t>(stride * rows));
        for (size_t p = 0; p < ids.size(); ++p) {
          const auto mask_start = Clock::now();
          state.mask(&masks[0]);
          const int count = std::min(rows, static_cast<int>(ids.size() - p));
          if (count > 1) {
            GrammarState after = state;
            for (int t = 1; t < count; ++t) {
              after.advance(ids[p + t - 1]);
              after.mask(&masks[t]);
            }
          }
          for (int t = 0; t < count; ++t) {
            auto* row = staged.data() + t * stride;
            row[0] = masks[t].constrained() ? static_cast<uint32_t>(masks[t].allowed) : 0;
            if (masks[t].constrained())
              std::copy(masks[t].words.begin(), masks[t].words.end(), row + 1);
          }
          mask_us += micros(mask_start, Clock::now());
          require(masks[0].allows(ids[p]), "replay token outside mask");
          const auto advance_start = Clock::now();
          require(state.allows(ids[p]), "pointwise replay refused");
          state.advance(ids[p]);
          require(state.active(), "grammar died on accepted token");
          advance_us += micros(advance_start, Clock::now());
          for (int t = 0; t < count; ++t) {
            // Observe the actual staging writes outside the timed interval.
            const volatile uint32_t* row = staged.data() + t * stride;
            const uint32_t allowed = row[0];
            digest ^= allowed;
            digest *= 1099511628211ull;
            if (allowed != 0) {
              for (int w = 1; w < stride; ++w) {
                digest ^= row[w];
                digest *= 1099511628211ull;
              }
            }
          }
        }
        const double total = micros(begin, Clock::now());
        std::printf("%s,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%llu\n", c[0].c_str(), round, rows,
                    ids.size(), construct_us, mask_us, advance_us, total,
                    static_cast<unsigned long long>(digest));
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
