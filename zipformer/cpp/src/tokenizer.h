#pragma once
#include <string>
#include <vector>

// Minimal vocabulary for Transducer greedy decoding.
// Format: one "symbol id" pair per line.
class Tokenizer {
public:
    bool Load(const std::string& path, std::string* error = nullptr);
    // Check that every ID in [0, vocab_size) has an entry.
    bool Covers(int vocab_size, std::string* error = nullptr) const;
    // Decode token IDs into text (skips blank and unk).
    std::string Decode(const std::vector<int>& ids, int blank_id,
                       int unk_id) const;

private:
    std::vector<std::string> pieces_;
};
