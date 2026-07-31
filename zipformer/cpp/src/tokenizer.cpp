#include "tokenizer.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

bool Tokenizer::Load(const std::string& path, std::string* error) {
    pieces_.clear();
    std::ifstream f(path.c_str());
    if (!f) {
        if (error) *error = "cannot open tokens: " + path;
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string symbol, idtext, extra;
        if (!(ss >> symbol >> idtext) || (ss >> extra)) {
            if (error) *error = "invalid token line " + std::to_string(lineno);
            return false;
        }
        char* end = nullptr;
        errno = 0;
        long id = std::strtol(idtext.c_str(), &end, 10);
        if (errno || !end || *end || id < 0 || id > 10000000) {
            if (error) *error = "invalid token id line " + std::to_string(lineno);
            return false;
        }
        if (static_cast<size_t>(id) >= pieces_.size())
            pieces_.resize(static_cast<size_t>(id) + 1);
        if (!pieces_[id].empty()) {
            if (error) *error = "duplicate token id " + std::to_string(id);
            return false;
        }
        pieces_[id] = symbol;
    }
    return !pieces_.empty();
}

bool Tokenizer::Covers(int vocab_size, std::string* error) const {
    if (vocab_size <= 0 || static_cast<size_t>(vocab_size) > pieces_.size()) {
        if (error) *error = "tokens do not cover vocab size";
        return false;
    }
    for (int i = 0; i < vocab_size; ++i) {
        if (pieces_[i].empty()) {
            if (error) *error = "missing token id " + std::to_string(i);
            return false;
        }
    }
    return true;
}

std::string Tokenizer::Decode(const std::vector<int>& ids, int blank,
                              int unk) const {
    std::string out;
    for (int id : ids) {
        if (id >= 0 && id != blank && id != unk &&
            static_cast<size_t>(id) < pieces_.size()) {
            out += pieces_[id];
        }
    }
    // Replace sentence-piece space markers.
    size_t pos = 0;
    while ((pos = out.find("▁", pos)) != std::string::npos) {
        out.replace(pos, 3, " ");
        ++pos;
    }
    pos = out.find_first_not_of(' ');
    return (pos == std::string::npos) ? std::string() : out.substr(pos);
}
