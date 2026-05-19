#include "normalizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <cassert>

using json = nlohmann::json;

// ── UTF-8 helpers ────────────────────────────────────────────────────────────

uint32_t Normalizer::decode_utf8(const char* s, int& out_bytes) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { out_bytes = 1; return c; }
    if ((c & 0xE0) == 0xC0) {
        out_bytes = 2;
        return ((c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
    }
    if ((c & 0xF0) == 0xE0) {
        out_bytes = 3;
        return ((c & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F);
    }
    if ((c & 0xF8) == 0xF0) {
        out_bytes = 4;
        return ((c & 0x07) << 18) | (((unsigned char)s[1] & 0x3F) << 12) |
               (((unsigned char)s[2] & 0x3F) << 6) | ((unsigned char)s[3] & 0x3F);
    }
    out_bytes = 1; return 0xFFFD; // replacement char
}

void Normalizer::encode_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

// ── Constructor ──────────────────────────────────────────────────────────────

Normalizer::Normalizer(const std::string& homophones_map_path) {
    std::ifstream f(homophones_map_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open homophones_map: " + homophones_map_path);

    json j;
    f >> j;

    // JSON format: { "错字": "正字", ... }  — both are single UTF-8 chars
    for (auto& [key, val] : j.items()) {
        int kb = 0, vb = 0;
        uint32_t kcp = decode_utf8(key.c_str(), kb);
        uint32_t vcp = decode_utf8(val.get<std::string>().c_str(), vb);
        map_[kcp] = vcp;
    }
}

// ── Core processing ──────────────────────────────────────────────────────────

// Process plain text segment (no tags), apply homophones replacement
std::string Normalizer::process_segment(const std::string& seg,
                                        const std::unordered_map<uint32_t, uint32_t>& map) {
    std::string out;
    out.reserve(seg.size());
    int i = 0;
    while (i < (int)seg.size()) {
        int bytes = 0;
        uint32_t cp = decode_utf8(seg.c_str() + i, bytes);
        auto it = map.find(cp);
        encode_utf8(it != map.end() ? it->second : cp, out);
        i += bytes;
    }
    return out;
}

std::string Normalizer::normalize(const std::string& text) const {
    // Split on [tags], replace homophones only in plain-text segments
    std::string result;
    result.reserve(text.size());

    int i = 0;
    while (i < (int)text.size()) {
        if (text[i] == '[') {
            // find matching ']'
            int j = i + 1;
            while (j < (int)text.size() && text[j] != ']') ++j;
            // copy tag verbatim
            result += text.substr(i, j - i + 1);
            i = j + 1;
        } else {
            // collect plain-text until next '[' or end
            int j = i;
            while (j < (int)text.size() && text[j] != '[') ++j;
            result += process_segment(text.substr(i, j - i), map_);
            i = j;
        }
    }
    return result;
}
