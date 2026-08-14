#include "tokenizer.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "tokenizers_cpp.h"

namespace qwen3tts {

bool TextTokenizer::load(const std::string& model_dir) {
    std::string path = model_dir + "/tokenizer.json";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "[tokenizer] cannot open %s\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json);
    if (!tokenizer_) {
        fprintf(stderr, "[tokenizer] FromBlobJSON failed\n");
        return false;
    }
    fprintf(stderr, "[tokenizer] loaded vocab=%zu\n", tokenizer_->GetVocabSize());
    return true;
}

std::vector<int> TextTokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    if (!tokenizer_) return out;
    auto ids = tokenizer_->Encode(text);
    out.assign(ids.begin(), ids.end());
    return out;
}

}  // namespace qwen3tts
