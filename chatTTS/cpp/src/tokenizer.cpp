#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// ── UTF-8 helpers ────────────────────────────────────────────────────────────

uint32_t BertTokenizer::decode_utf8(const char* s, int& out_bytes) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { out_bytes = 1; return c; }
    if ((c & 0xE0) == 0xC0) { out_bytes = 2; return ((c&0x1F)<<6)|((unsigned char)s[1]&0x3F); }
    if ((c & 0xF0) == 0xE0) { out_bytes = 3;
        return ((c&0x0F)<<12)|(((unsigned char)s[1]&0x3F)<<6)|((unsigned char)s[2]&0x3F); }
    if ((c & 0xF8) == 0xF0) { out_bytes = 4;
        return ((c&0x07)<<18)|(((unsigned char)s[1]&0x3F)<<12)|
               (((unsigned char)s[2]&0x3F)<<6)|((unsigned char)s[3]&0x3F); }
    out_bytes = 1; return 0xFFFD;
}

void BertTokenizer::encode_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) { out += (char)cp; }
    else if (cp < 0x800) { out += (char)(0xC0|(cp>>6)); out += (char)(0x80|(cp&0x3F)); }
    else if (cp < 0x10000) {
        out += (char)(0xE0|(cp>>12)); out += (char)(0x80|((cp>>6)&0x3F)); out += (char)(0x80|(cp&0x3F));
    } else {
        out += (char)(0xF0|(cp>>18)); out += (char)(0x80|((cp>>12)&0x3F));
        out += (char)(0x80|((cp>>6)&0x3F)); out += (char)(0x80|(cp&0x3F));
    }
}

bool BertTokenizer::is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF)   ||
           (cp >= 0x3400 && cp <= 0x4DBF)   ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF)   ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool BertTokenizer::is_punctuation(uint32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        return true;
    // Unicode general punctuation
    if (cp >= 0x2000 && cp <= 0x206F) return true;
    if (cp >= 0xFF00 && cp <= 0xFFEF) return true;
    return false;
}

// ── Constructor ──────────────────────────────────────────────────────────────

BertTokenizer::BertTokenizer(const std::string& vocab_path) {
    std::ifstream f(vocab_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open vocab file: " + vocab_path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        int id = static_cast<int>(id2tok_.size());
        tok2id_[line] = id;
        id2tok_.push_back(line);
    }
    // ChatTTS extended vocabulary (tokens 21128-21177), exactly matching tokenizer.pt
    static const char* kChatTTSTokens[] = {
        "[Sasr]","[Pasr]","[Easr]","[Stts]","[Ptts]","[Etts]",
        "[Sbreak]","[Pbreak]","[Ebreak]","[uv_break]","[v_break]",
        "[lbreak]","[llbreak]","[undefine]","[laugh]","[spk_emb]","[empty_spk]",
        "[music]","[pure]","[break_0]","[break_1]","[break_2]","[break_3]",
        "[break_4]","[break_5]","[break_6]","[break_7]",
        "[laugh_0]","[laugh_1]","[laugh_2]",
        "[oral_0]","[oral_1]","[oral_2]","[oral_3]","[oral_4]","[oral_5]",
        "[oral_6]","[oral_7]","[oral_8]","[oral_9]",
        "[speed_0]","[speed_1]","[speed_2]","[speed_3]","[speed_4]",
        "[speed_5]","[speed_6]","[speed_7]","[speed_8]","[speed_9]",
    };
    for (const char* tok : kChatTTSTokens) {
        int id = static_cast<int>(id2tok_.size());
        tok2id_[tok] = id;
        id2tok_.push_back(tok);
    }
}

int BertTokenizer::token_to_id(const std::string& token) const {
    auto it = tok2id_.find(token);
    return it != tok2id_.end() ? it->second : tok2id_.at("[UNK]");
}

std::string BertTokenizer::id_to_token(int id) const {
    if (id < 0 || id >= (int)id2tok_.size()) return "[UNK]";
    return id2tok_[id];
}

// ── Basic tokenization ───────────────────────────────────────────────────────

std::vector<std::string> BertTokenizer::basic_tokenize(const std::string& text) {
    // Insert spaces around CJK chars and punctuation, then split on whitespace
    std::string spaced;
    spaced.reserve(text.size() * 2);
    int i = 0;
    while (i < (int)text.size()) {
        int bytes = 0;
        uint32_t cp = decode_utf8(text.c_str() + i, bytes);
        if (is_cjk(cp) || is_punctuation(cp)) {
            spaced += ' ';
            encode_utf8(cp, spaced);
            spaced += ' ';
        } else {
            encode_utf8(cp, spaced);
        }
        i += bytes;
    }

    // Split on whitespace, lowercase ASCII
    std::vector<std::string> tokens;
    std::istringstream iss(spaced);
    std::string tok;
    while (iss >> tok) {
        // lowercase ASCII letters only (preserve non-ASCII as-is, e.g. Chinese)
        for (char& c : tok) {
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
        }
        tokens.push_back(tok);
    }
    return tokens;
}

// ── WordPiece ────────────────────────────────────────────────────────────────

std::vector<int> BertTokenizer::wordpiece(const std::string& word) const {
    // Greedy longest-match-first
    std::vector<int> ids;
    int start = 0;
    int len   = (int)word.size();

    while (start < len) {
        int end = len;
        int best_id = -1;
        while (start < end) {
            std::string substr = (start == 0 ? "" : "##") + word.substr(start, end - start);
            auto it = tok2id_.find(substr);
            if (it != tok2id_.end()) { best_id = it->second; break; }
            // Move end back by one UTF-8 character
            --end;
            while (end > start && (word[end] & 0xC0) == 0x80) --end;
        }
        if (best_id == -1) {
            // Unknown word → single [UNK]
            ids.clear();
            ids.push_back(token_to_id("[UNK]"));
            return ids;
        }
        ids.push_back(best_id);
        start = end;
    }
    return ids;
}

// ── encode ───────────────────────────────────────────────────────────────────

std::vector<int> BertTokenizer::encode(const std::string& text) const {
    // Handle ChatTTS special tokens like [spk_emb], [speed_5], [Stts], [Ptts], etc.
    // They appear verbatim in the input after speaker.decorate_code_prompts
    std::vector<int> all_ids;

    // Only treat [tag] as a special token if it's in vocab.
    // Otherwise tokenize it as plain text (including the brackets).
    int i = 0;
    while (i < (int)text.size()) {
        if (text[i] == '[') {
            // Try to find matching ']'
            int j = i + 1;
            while (j < (int)text.size() && text[j] != ']') ++j;
            if (j < (int)text.size()) {
                std::string tag = text.substr(i, j - i + 1);
                auto it = tok2id_.find(tag);
                if (it != tok2id_.end()) {
                    all_ids.push_back(it->second);
                    i = j + 1;
                    continue;
                }
                // Not a special token — fall through to tokenize as plain text
                // (collect this segment including the '[' and ']')
            }
        }
        // Collect plain segment until next '[' that starts a known special token
        int j = i;
        while (j < (int)text.size()) {
            if (text[j] == '[') {
                // Peek ahead to see if this is a known special token
                int k = j + 1;
                while (k < (int)text.size() && text[k] != ']') ++k;
                if (k < (int)text.size()) {
                    std::string tag = text.substr(j, k - j + 1);
                    if (tok2id_.count(tag)) break;  // known token — stop here
                }
            }
            ++j;
        }
        if (j > i) {
            std::string seg = text.substr(i, j - i);
            auto words = basic_tokenize(seg);
            for (const auto& w : words) {
                auto sub_ids = wordpiece(w);
                all_ids.insert(all_ids.end(), sub_ids.begin(), sub_ids.end());
            }
        }
        i = j;
    }
    return all_ids;
}
