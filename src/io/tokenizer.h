#pragma once

// BPE tokenizer for Inkling-2-Mini.
//
// Loads the HF "tokenizers" format tokenizer.json found in the model dir:
//   - model: byte-level BPE (vocab + merges, GPT-2 style byte<->unicode alphabet)
//   - pre_tokenizer: Split on the declared Unicode regex (o200k-style), then ByteLevel
//   - added_tokens: special tokens matched as literals (<|message_user|>, ...)
//   - decoder: ByteLevel (mapped chars -> raw bytes -> UTF-8)
//
// encode_chat_user() applies the fixed prompt wrapper used by the golden
// harness (testdata/golden/meta.json "rendered_text"): the chat_template.jinja
// rendering of a single user turn with the default thinking effort of 0.9.

#include <memory>
#include <string>
#include <vector>

namespace ink {

struct Tokenizer {
  Tokenizer();
  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // Loads <model_dir>/tokenizer.json. On failure returns false and fills *err.
  bool load(const std::string& model_dir, std::string* err);
  bool loaded() const;

  // Special-token literals in `text` (e.g. "<|begin_of_text|>") are emitted as
  // their added-token ids; the rest goes through pretokenize + byte-level BPE.
  std::vector<int> encode(const std::string& text) const;

  // encode() of "<|message_system|><|content_text|>Thinking effort level:
  // 0.9<|end_message|><|message_user|><|content_text|>" + user_text +
  // "<|end_message|>". Matches golden prompt.input_ids exactly.
  std::vector<int> encode_chat_user(const std::string& user_text) const;

  std::string decode(const std::vector<int>& ids) const;

  // Convenience: token id of a special literal, or -1 if unknown.
  int special_id(const std::string& literal) const;
  const std::string* special_literal(int id) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ink
