/**
 * Qwen3 BM1684X 纯 C++ 推理 Demo
 *
 * 用法:
 *   ./qwen_demo <model_dir>
 *
 * model_dir 目录结构:
 *   model_dir/
 *     qwen3.bmodel          (或 config.json 中指定的名称)
 *     config/tokenizer.json
 *
 * 编译:
 *   见同目录 CMakeLists.txt（需要 aarch64 交叉编译工具链）
 */

#include "QwenEngine.h"
#include "tokenizers_cpp.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---- UTF-8 流式解码（累积到无替换字符为止再输出）----

static bool has_replacement_char(const std::string& s)
{
    // U+FFFD 的 UTF-8 编码为 EF BF BD
    for (size_t i = 0; i + 2 < s.size(); i++) {
        if ((unsigned char)s[i]     == 0xEF &&
            (unsigned char)s[i + 1] == 0xBF &&
            (unsigned char)s[i + 2] == 0xBD)
            return true;
    }
    return false;
}

// ---- 读取文件内容为字符串 ----

static std::string read_file(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    return {std::istreambuf_iterator<char>(ifs), {}};
}

// ---- 生成完整回复（打印流式 token，返回最终字符串）----

static std::string generate(QwenEngine& engine,
                             tokenizers::Tokenizer& tok,
                             const std::string& prompt,
                             int max_new_tokens,
                             double* out_prefill_ms = nullptr,
                             double* out_tps = nullptr)
{
    auto ids = tok.Encode(prompt, /*add_special_tokens=*/false);

    using clk = std::chrono::steady_clock;
    std::string full_text;
    std::string word_buf;

    // prefill
    auto t0    = clk::now();
    int  token = engine.forward_first(ids);
    auto t1    = clk::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (out_prefill_ms) *out_prefill_ms = prefill_ms;

    int EOS = tok.TokenToId("<|im_end|>");

    int tok_count = 0;
    auto td0 = clk::now();

    while (token != EOS && engine.token_length < engine.SEQLEN - 1
           && tok_count < max_new_tokens)
    {
        std::string piece = tok.IdToToken(token);
        word_buf += piece;

        // 积累到没有替换字符再输出（UTF-8 多字节）
        if (!has_replacement_char(word_buf)) {
            std::cout << word_buf << std::flush;
            full_text += word_buf;
            word_buf.clear();
        }

        token = engine.forward_next();
        tok_count++;
    }
    if (!word_buf.empty()) {
        std::cout << word_buf << std::flush;
        full_text += word_buf;
    }
    std::cout << "\n";

    auto td1 = clk::now();
    double decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
    if (out_tps && tok_count > 0) {
        *out_tps = tok_count / (decode_ms / 1000.0);
    }

    return full_text;
}

// ---- 构建 ChatML prompt ----

static std::string build_prompt(const std::string& system_prompt,
                                 const std::string& user_text,
                                 bool no_think = true)
{
    std::string p;
    if (!system_prompt.empty()) {
        p += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    p += "<|im_start|>user\n" + user_text + "<|im_end|>\n";
    p += "<|im_start|>assistant\n";
    if (no_think) {
        p += "<think>\n\n</think>\n\n";
    }
    return p;
}

// ---- benchmark ----

static void benchmark(QwenEngine& engine,
                      tokenizers::Tokenizer& tok,
                      const std::string& user_text,
                      const std::string& label,
                      int max_new_tokens = 256)
{
    // 每次推理前清 KV cache（如果 bmodel 支持）
    engine.clear_kv();
    // 不支持 prefill-KV 时需手动重置
    engine.history_length = 0;
    engine.token_length   = 0;

    std::string prompt = build_prompt("You are a helpful assistant.", user_text);

    std::cout << "\n=== [" << label << "] ===\n";
    std::cout << "User: " << user_text << "\n";
    std::cout << "Assistant: ";

    double prefill_ms = 0, tps = 0;
    generate(engine, tok, prompt, max_new_tokens, &prefill_ms, &tps);

    std::cout << "  Prefill : " << prefill_ms << " ms  (time to first token)\n";
    std::cout << "  Decode  : " << tps << " token/s\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [bmodel_filename]\n";
        std::cerr << "  model_dir: directory containing bmodel and config/tokenizer.json\n";
        return 1;
    }

    std::string model_dir = argv[1];
    if (!model_dir.empty() && model_dir.back() != '/') model_dir += '/';

    // bmodel 文件名：默认扫描 .bmodel，或通过第2参数指定
    std::string bmodel_path;
    if (argc >= 3) {
        bmodel_path = model_dir + argv[2];
    } else {
        for (auto& e : fs::directory_iterator(model_dir)) {
            if (e.path().extension() == ".bmodel") {
                bmodel_path = e.path().string();
                break;
            }
        }
    }
    if (bmodel_path.empty()) {
        std::cerr << "[error] No .bmodel found in " << model_dir << "\n";
        return 1;
    }

    std::string tokenizer_path = model_dir + "config/tokenizer.json";
    if (!fs::exists(tokenizer_path)) {
        std::cerr << "[error] tokenizer.json not found at " << tokenizer_path << "\n";
        return 1;
    }

    std::cout << "[info] bmodel: "    << bmodel_path    << "\n";
    std::cout << "[info] tokenizer: " << tokenizer_path << "\n";

    // ---- 初始化 tokenizer ----
    auto tok = tokenizers::Tokenizer::FromBlobJSON(read_file(tokenizer_path));

    // ---- 初始化推理引擎 ----
    QwenEngine engine;
    engine.generation_mode = "greedy";
    engine.temperature     = 1.0f;
    engine.top_p           = 1.0f;
    engine.top_k           = 1;
    engine.penalty         = 1.0f;

    try {
        engine.init({0}, bmodel_path);
    } catch (const std::exception& e) {
        std::cerr << "[error] Engine init failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[info] SEQLEN=" << engine.SEQLEN
              << " MAX_INPUT_LENGTH=" << engine.MAX_INPUT_LENGTH
              << " NUM_LAYERS=" << engine.NUM_LAYERS << "\n";

    // ---- benchmark ----
    benchmark(engine, *tok, "你好，请用一句话介绍一下自己。", "短回复");
    benchmark(engine, *tok, "请列举5种常见的编程语言，并分别说明它们的主要用途。", "中等回复");

    engine.deinit();
    return 0;
}
