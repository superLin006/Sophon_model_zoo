#pragma once
// Qwen3 tokenizer：基于现成 tokenizers-cpp（1_third_party，QwenLLM 同款）
// C 接口 tokenizers_decode 支持 skip_special_token，与 HF 行为一致
#include <string>
#include <vector>
#include <cstdint>

namespace asr {

class Qwen3Tokenizer {
public:
    Qwen3Tokenizer() = default;
    ~Qwen3Tokenizer();
    Qwen3Tokenizer(const Qwen3Tokenizer&) = delete;
    Qwen3Tokenizer& operator=(const Qwen3Tokenizer&) = delete;

    // 从模型目录加载 tokenizer.json
    bool load(const std::string& model_dir);

    // 解码（跳过特殊 token，如 <|im_end|> 等）
    std::string decode(const std::vector<int>& ids) const;

    // token → id（特殊 token 查询）
    int token_to_id(const std::string& token) const;

    size_t vocab_size() const { return vocab_size_; }
    bool loaded() const { return handle_ != nullptr; }

private:
    void* handle_ = nullptr;   // TokenizerHandle
    size_t vocab_size_ = 0;
};

}  // namespace asr
