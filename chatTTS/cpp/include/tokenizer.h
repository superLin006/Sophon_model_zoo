#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// WordPiece BertTokenizer backed by vocab.txt (ChatTTS, 21178 tokens)
class BertTokenizer {
public:
    // vocab_path: path to vocab.txt exported from tokenizer.pt
    explicit BertTokenizer(const std::string& vocab_path);

    // Encode text → token ids, no [CLS]/[SEP] added
    std::vector<int> encode(const std::string& text) const;

    int token_to_id(const std::string& token) const;
    std::string id_to_token(int id) const;
    int vocab_size() const { return static_cast<int>(id2tok_.size()); }

private:
    std::vector<std::string>          id2tok_;
    std::unordered_map<std::string, int> tok2id_;

    // WordPiece segment a single word (already lowercased/cleaned)
    std::vector<int> wordpiece(const std::string& word) const;

    // Basic tokenization: CJK char splitting + whitespace
    static std::vector<std::string> basic_tokenize(const std::string& text);
    static bool is_cjk(uint32_t cp);
    static bool is_punctuation(uint32_t cp);
    static uint32_t decode_utf8(const char* s, int& out_bytes);
    static void     encode_utf8(uint32_t cp, std::string& out);
};
