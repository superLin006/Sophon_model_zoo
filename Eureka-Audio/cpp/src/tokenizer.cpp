// Qwen3 tokenizer 实现
// 使用 tiktoken-compatible BPE，从 tokenizer.json 加载
// 仅实现 Eureka-Audio 所需功能：encode / decode / apply_chat_template

#include "tokenizer.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <cassert>

// 使用 nlohmann/json 单头文件解析 tokenizer.json
// 如果环境里没有，可用 -DUSE_SIMPLE_JSON，改为手写简单解析
#ifdef USE_SIMPLE_JSON
// 简单 JSON 子集，仅支持 tokenizer.json 格式（不通用）
#include "simple_json.hpp"
namespace json_ns = simple_json;
#else
#include "nlohmann/json.hpp"
using json = nlohmann::json;
#endif

namespace eureka {

// ─────────────────────────────────────────────────────────────────────────────
// UTF-8 / tiktoken byte-to-unicode 映射
// tiktoken 用 256 个 unicode 码点表示 byte 值（用于 BPE）
// 规则：可打印 ASCII + 拉丁补充 → 自身；其余补到第 256 个可用码点
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<uint32_t> g_byte_encoder;  // [256] byte → unicode codepoint
static std::unordered_map<uint32_t, uint8_t>  g_byte_decoder;  // codepoint → byte

static void init_byte_encoder() {
    if (!g_byte_encoder.empty()) return;
    g_byte_encoder.resize(256, 0);
    std::vector<uint32_t> bs;
    // !..~ (0x21..0x7E = printable ASCII minus space)
    for (int i = '!'; i <= '~'; ++i)  bs.push_back(i);
    // ¡..¬ (0xA1..0xAC)
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    // ®..ÿ (0xAE..0xFF)
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    // space (0x20) also maps to itself in tiktoken (Ġ = 0x0120 in GPT2; in Qwen tiktoken space = Ġ)
    // Qwen3 tokenizer uses the same byte-level encoding as GPT-2 tiktoken
    uint32_t n = 256;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), (uint32_t)b) == bs.end())
            bs.push_back(b);  // 顺序占位
    }
    // 重新按 tiktoken 规则映射
    // 参考：https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    // 128 个"安全"字节直接映射，其余映射到 256+
    g_byte_encoder.assign(256, 0);
    std::vector<uint32_t> safe_bytes;
    for (int i = '!'; i <= '~'; ++i)    safe_bytes.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i)  safe_bytes.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i)  safe_bytes.push_back(i);

    for (uint32_t b : safe_bytes)
        g_byte_encoder[b] = b;

    uint32_t next = 256;
    for (int b = 0; b < 256; ++b) {
        if (std::find(safe_bytes.begin(), safe_bytes.end(), (uint32_t)b) == safe_bytes.end()) {
            g_byte_encoder[b] = next++;
        }
    }
    for (int b = 0; b < 256; ++b)
        g_byte_decoder[g_byte_encoder[b]] = (uint8_t)b;
}

// UTF-8 编码单个 codepoint
static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// 将 tokenizer.json 词表里的 token 字符串（tiktoken unicode 编码）→ 字节串
static std::string tok_to_bytes(const std::string& tok) {
    init_byte_encoder();
    // tok 是 UTF-8 字符串，先解码成 codepoints，每个 codepoint 通过 byte_decoder → byte
    std::string bytes;
    size_t i = 0;
    while (i < tok.size()) {
        uint32_t cp = 0;
        unsigned char c = tok[i];
        int len = 1;
        if      ((c & 0x80) == 0x00) { cp = c;            len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F;     len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F;     len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07;     len = 4; }
        for (int j = 1; j < len && i + j < tok.size(); ++j)
            cp = (cp << 6) | (((unsigned char)tok[i + j]) & 0x3F);
        i += len;

        auto it = g_byte_decoder.find(cp);
        if (it != g_byte_decoder.end())
            bytes += (char)it->second;
        // else: 特殊 token，跳过
    }
    return bytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// BPE
// ─────────────────────────────────────────────────────────────────────────────
// merge_rank_: token_pair_string → rank（由 tokenizer.json merges 数组初始化）
struct PairHash {
    size_t operator()(const std::pair<std::string,std::string>& p) const {
        size_t h1 = std::hash<std::string>{}(p.first);
        size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 * 2654435761u);
    }
};

static std::unordered_map<std::pair<std::string,std::string>, int, PairHash> g_merge_rank;
static bool g_merge_init = false;

std::vector<int> Qwen3Tokenizer::bpe_encode(const std::string& text) const {
    if (text.empty()) return {};

    init_byte_encoder();

    // 将输入 UTF-8 字节 → unicode 编码字符（tiktoken byte encoding）
    // 每个字节对应一个 unicode 字符
    std::vector<std::string> symbols;
    for (unsigned char b : text)
        symbols.push_back(codepoint_to_utf8(g_byte_encoder[b]));

    // BPE merge loop
    while (symbols.size() > 1) {
        // 找最低 rank 的相邻对
        int best_rank = INT_MAX;
        int best_pos  = -1;
        for (int i = 0; i + 1 < (int)symbols.size(); ++i) {
            auto it = g_merge_rank.find({symbols[i], symbols[i+1]});
            if (it != g_merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = i;
            }
        }
        if (best_pos < 0) break;

        // 合并
        std::string merged = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
        symbols[best_pos] = merged;
    }

    // 每个 symbol 查词表
    std::vector<int> ids;
    for (auto& sym : symbols) {
        auto it = vocab_.find(sym);
        if (it != vocab_.end())
            ids.push_back(it->second);
        // 未知 token：跳过（不应出现）
    }
    return ids;
}

// ─────────────────────────────────────────────────────────────────────────────
// load
// ─────────────────────────────────────────────────────────────────────────────
bool Qwen3Tokenizer::load(const std::string& model_dir) {
    std::string path = model_dir + "/tokenizer.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open tokenizer.json at %s\n", path.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (std::exception& e) {
        fprintf(stderr, "JSON parse error: %s\n", e.what());
        return false;
    }

    // 词表
    auto& model = j["model"];
    auto& vocab_j = model["vocab"];
    int max_id = 0;
    for (auto& [tok, id] : vocab_j.items()) {
        int tid = id.get<int>();
        vocab_[tok] = tid;
        if (tid > max_id) max_id = tid;
    }
    id_to_token_.resize(max_id + 1);
    for (auto& [tok, id] : vocab_j.items())
        id_to_token_[id.get<int>()] = tok;

    // 合并规则：支持 ["a","b"] 数组形式和 "a b" 字符串形式
    g_merge_rank.clear();
    auto& merges_j = model["merges"];
    for (int i = 0; i < (int)merges_j.size(); ++i) {
        std::string a, b;
        if (merges_j[i].is_array()) {
            a = merges_j[i][0].get<std::string>();
            b = merges_j[i][1].get<std::string>();
        } else {
            std::string line = merges_j[i].get<std::string>();
            auto sp = line.find(' ');
            if (sp == std::string::npos) continue;
            a = line.substr(0, sp);
            b = line.substr(sp + 1);
        }
        g_merge_rank[{a, b}] = i;
    }
    g_merge_init = true;

    // 特殊 token
    if (j.contains("added_tokens")) {
        for (auto& at : j["added_tokens"]) {
            std::string content = at["content"].get<std::string>();
            int id = at["id"].get<int>();
            vocab_[content] = id;
            if ((int)id_to_token_.size() <= id)
                id_to_token_.resize(id + 1);
            id_to_token_[id] = content;
        }
    }

    // 特殊 token IDs（Qwen3 默认）
    auto it_eos = vocab_.find("<|im_end|>");
    if (it_eos != vocab_.end()) eos_id_ = it_eos->second;
    auto it_bos = vocab_.find("<|im_start|>");
    if (it_bos != vocab_.end()) bos_id_ = it_bos->second;
    pad_id_ = bos_id_;

    loaded_ = true;
    printf("Tokenizer loaded: vocab_size=%d  eos=%d\n",
           (int)vocab_.size(), eos_id_);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// encode
// ─────────────────────────────────────────────────────────────────────────────
// 简化：不做 pre-tokenization（按空格切分等），直接整段 BPE
// 对系统提示词（纯文本+特殊标记）已经够用
std::vector<int> Qwen3Tokenizer::encode(const std::string& text) const {
    if (!loaded_) return {};

    // 先尝试作为特殊 token 整体查询
    auto it = vocab_.find(text);
    if (it != vocab_.end()) return {it->second};

    // 否则做 BPE
    return bpe_encode(text);
}

// ─────────────────────────────────────────────────────────────────────────────
// decode
// ─────────────────────────────────────────────────────────────────────────────
std::string Qwen3Tokenizer::decode(const std::vector<int>& ids) const {
    if (!loaded_) return "";
    init_byte_encoder();

    std::string bytes_result;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[id];
        if (tok.empty()) continue;
        // 特殊 token（如 <|im_end|>）：跳过
        if (tok.front() == '<' && tok.back() == '>') continue;
        // 将 token 中的 tiktoken unicode → bytes
        bytes_result += tok_to_bytes(tok);
    }
    return bytes_result;
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_chat_template
// Qwen3 格式（无 thinking 模式）：
//   <|im_start|>system\n{sys}<|im_end|>\n
//   <|im_start|>user\n{user_text}<|im_end|>\n
//   <|im_start|>assistant\n
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int> Qwen3Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string,std::string>>& messages) const {

    std::vector<int> ids;
    auto push_special = [&](const std::string& tok) {
        auto it = vocab_.find(tok);
        if (it != vocab_.end()) ids.push_back(it->second);
    };
    auto push_text = [&](const std::string& text) {
        auto enc = bpe_encode(text);
        ids.insert(ids.end(), enc.begin(), enc.end());
    };
    auto push_newline = [&]() {
        push_text("\n");
    };

    for (auto& [role, content] : messages) {
        push_special("<|im_start|>");
        push_text(role);
        push_newline();
        if (!content.empty()) push_text(content);
        push_special("<|im_end|>");
        push_newline();
    }
    // generation prompt
    push_special("<|im_start|>");
    push_text("assistant");
    push_newline();

    return ids;
}

std::string Qwen3Tokenizer::bytes_to_unicode_str(unsigned char c) {
    init_byte_encoder();
    return codepoint_to_utf8(g_byte_encoder[c]);
}

}  // namespace eureka
