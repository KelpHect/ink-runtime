// Tokenizer ground-truth test: encode/decode must reproduce the HF reference
// ids in testdata/golden/meta.json exactly (apply_chat_template of one user
// turn with default thinking effort 0.9), plus round-trip and edge cases.
#include "io/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace ink;

static int fails = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      fails++;                                                    \
    }                                                             \
  } while (0)

static void print_ids(const char* label, const std::vector<int>& ids) {
  std::printf("%s [", label);
  for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? "," : "", ids[i]);
  std::printf("]\n");
}

static std::string find_model_dir() {
  if (const char* env = std::getenv("INK_MODEL_DIR"))
    if (std::filesystem::exists(std::filesystem::path(env) / "tokenizer.json"))
      return env;
#ifdef INK_MODEL_DIR
  if (std::filesystem::exists(std::filesystem::path(INK_MODEL_DIR) / "tokenizer.json"))
    return INK_MODEL_DIR;
#endif
  for (const char* cand : {"Inkling-2-Mini", "../Inkling-2-Mini",
                           "../../Inkling-2-Mini", "../../../Inkling-2-Mini"}) {
    if (std::filesystem::exists(std::filesystem::path(cand) / "tokenizer.json"))
      return cand;
  }
  return {};
}

int main(int argc, char** argv) {
  Tokenizer tok;
  std::string err;
  const std::string dir = find_model_dir();
  if (dir.empty()) {
    std::printf("SKIP: Inkling-2-Mini/tokenizer.json not found\n");
    return 2;
  }
  CHECK(tok.load(dir, &err));
  if (!tok.loaded()) {
    std::printf("load failed: %s\n", err.c_str());
    return 1;
  }

  // Debug aid: `ink_tokenizer --encode "text"` prints the ids.
  if (argc > 2 && std::string(argv[1]) == "--encode") {
    print_ids("ids", tok.encode(argv[2]));
    return 0;
  }
  // `ink_tokenizer --encode-hex <utf8-bytes-as-hex>` avoids argv codepage issues.
  if (argc > 2 && std::string(argv[1]) == "--encode-hex") {
    std::string s;
    auto hv = [](char c) {
      return c >= '0' && c <= '9' ? c - '0'
           : c >= 'a' && c <= 'f' ? c - 'a' + 10
           : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 0;
    };
    for (const char* p = argv[2]; p[0] && p[1]; p += 2)
      s.push_back(static_cast<char>(hv(p[0]) * 16 + hv(p[1])));
    print_ids("ids", tok.encode(s));
    return 0;
  }

  // ---- Ground truth: testdata/golden/meta.json prompt.input_ids ----
  const std::string kUser = "What is the capital of France? Answer briefly.";
  const std::string kRendered =
      "<|message_system|><|content_text|>Thinking effort level: "
      "0.9<|end_message|><|message_user|><|content_text|>" +
      kUser + "<|end_message|>";
  const std::vector<int> kGolden = {
      200002, 200004, 34182,   279,    154554, 150628, 25,
      220,    15,     13,      24,     200010, 200000, 200004,
      1775,   149884, 149876,  151451, 149879, 151194, 30,
      154710, 183263, 13,      200010};

  std::vector<int> ids = tok.encode_chat_user(kUser);
  if (ids != kGolden) {
    print_ids("got    ", ids);
    print_ids("expect ", kGolden);
    CHECK(false);
  }

  // encode() of the raw rendered text must give the same ids.
  std::vector<int> ids2 = tok.encode(kRendered);
  if (ids2 != kGolden) {
    print_ids("encode(rendered) got   ", ids2);
    print_ids("encode(rendered) expect", kGolden);
    CHECK(false);
  }

  // decode(golden ids) reproduces the rendered text exactly.
  CHECK(tok.decode(kGolden) == kRendered);

  // decode(golden generated_ids) == meta.json generated_text.
  CHECK(tok.decode({200001, 200008, 371, 150409, 171960, 149878, 150744, 151231}) ==
        "<|message_model|><|content_thinking|>The user asks a simple fact");

  // ---- Special-token literals ----
  CHECK(tok.special_id("<|begin_of_text|>") == 200028);
  CHECK(tok.special_id("<|end_message|>") == 200010);
  CHECK(tok.special_id("nope") == -1);
  {
    std::vector<int> one = tok.encode("<|begin_of_text|>");
    CHECK(one == std::vector<int>{200028});
    CHECK(tok.decode({200028}) == "<|begin_of_text|>");
    // Literals embedded in text still split out.
    std::vector<int> mixed = tok.encode("hi<|end_message|>");
    std::vector<int> want = tok.encode("hi");
    want.push_back(200010);
    CHECK(mixed == want);
  }

  // ---- Round-trips ----
  const char* cases[] = {
      "Hello, world!",
      "The quick brown fox jumps over the lazy dog.",
      "  multiple   spaces\tand\ttabs",
      "trailing space ",
      " leading space",
      "line1\nline2\r\nline3",
      "numbers: 0 1 12 123 1234 12345 3.14159",
      "don't can't it's I'M you're they'll we've he'd",
      "UPPER lower CamelCase ABCDEFGHIJK",
      "symbols !@#$%^&*()_+-=[]{}|;:'\",.<>/?`~",
      "unicode: h\xc3\xa9llo w\xc3\xb6rld caf\xc3\xa9 na\xc3\xafve "
      "\xc3\x9f\xc3\x9f",
      "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c",          // CJK
      "\xce\xb1\xce\xb2\xce\xb3 \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82",  // greek+cyrillic
      "emoji \xf0\x9f\x98\x80\xf0\x9f\x8e\x89 ok",
      "a<|end_message|>b<|begin_of_text|>c",
      "",
  };
  for (const char* c : cases) {
    const std::string s(c);
    std::vector<int> enc = tok.encode(s);
    std::string dec = tok.decode(enc);
    if (dec != s) {
      std::printf("round-trip mismatch for %s\n  dec=%s\n", s.c_str(), dec.c_str());
      print_ids("  ids", enc);
      CHECK(false);
    }
    // encode is deterministic
    CHECK(tok.encode(s) == enc);
  }

  // BPE sanity: numbers chunk in groups of <=3 (1234 -> "123","4"), and a
  // longer run merges to fewer ids than chars.
  {
    std::vector<int> d = tok.encode("1234");
    CHECK(!d.empty() && d.size() <= 2);
  }

  if (fails == 0) {
    std::printf("tokenizer: all checks passed (model=%s)\n", dir.c_str());
    return 0;
  }
  std::printf("tokenizer: %d check(s) failed\n", fails);
  return 1;
}
