#pragma once
// Qwen3 文本 tokenizer（tokenizers-cpp C API）
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#include "tokenizers_cpp.h"

namespace qwen3tts {

class TextTokenizer {
public:
    bool load(const std::string& model_dir);
    // 文本 → token ids（add_special_token=false，与 HF 默认一致）
    std::vector<int> encode(const std::string& text) const;
    bool loaded() const { return tokenizer_ != nullptr; }

private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

}  // namespace qwen3tts
