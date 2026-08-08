// Qwen3 tokenizer：tokenizers-cpp（C API）包装
#include "tokenizer.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "tokenizers_c.h"

namespace asr {

Qwen3Tokenizer::~Qwen3Tokenizer() {
    if (handle_) { tokenizers_free(handle_); handle_ = nullptr; }
}

bool Qwen3Tokenizer::load(const std::string& model_dir) {
    std::string path = model_dir + "/tokenizer.json";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open tokenizer.json at %s\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    handle_ = tokenizers_new_from_str(json.data(), json.size());
    if (!handle_) {
        fprintf(stderr, "tokenizers_new_from_str failed\n");
        return false;
    }
    tokenizers_get_vocab_size(handle_, &vocab_size_);
    printf("Tokenizer loaded: vocab_size=%zu\n", vocab_size_);
    return true;
}

std::string Qwen3Tokenizer::decode(const std::vector<int>& ids) const {
    if (!handle_ || ids.empty()) return "";
    tokenizers_decode(handle_, (const uint32_t*)ids.data(), ids.size(), 1 /* skip_special_token */);
    const char* data = nullptr;
    size_t len = 0;
    tokenizers_get_decode_str(handle_, &data, &len);
    return std::string(data ? data : "", len);
}

int Qwen3Tokenizer::token_to_id(const std::string& token) const {
    if (!handle_) return -1;
    int32_t id = -1;
    tokenizers_token_to_id(handle_, token.data(), token.size(), &id);
    return id;
}

}  // namespace asr
