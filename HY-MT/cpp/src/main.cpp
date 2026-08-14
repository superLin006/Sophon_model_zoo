#include "QwenEngine.h"
#include "tokenizers_cpp.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string read_file(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open file: " + path);
    return {std::istreambuf_iterator<char>(stream), {}};
}

static std::string build_prompt(const std::string& user_text)
{
    // Exact chat_template.jinja behavior for one user message with
    // add_generation_prompt=false (the setting used by the official README).
    return "<｜hy_begin▁of▁sentence｜><｜hy_User｜>" + user_text +
           "<｜hy_place▁holder▁no▁8｜>";
}

static void expand_escaped_newlines(std::string& text)
{
    size_t pos = 0;
    while ((pos = text.find("\\n", pos)) != std::string::npos) {
        text.replace(pos, 2, "\n");
        ++pos;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_dir> <translation_prompt> [max_new_tokens]\n";
        return 1;
    }
    std::string model_dir = argv[1];
    std::string user_text = argv[2];
    expand_escaped_newlines(user_text);
    int max_new_tokens = argc > 3 ? std::stoi(argv[3]) : 128;
    if (!model_dir.empty() && model_dir.back() != '/') model_dir += '/';

    std::string bmodel_path;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".bmodel") {
            bmodel_path = entry.path().string();
            break;
        }
    }
    if (bmodel_path.empty()) throw std::runtime_error("No bmodel in " + model_dir);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(
        read_file(model_dir + "config/tokenizer.json"));

    auto encoded = tokenizer->Encode(build_prompt(user_text));
    std::vector<int> input_ids(encoded.begin(), encoded.end());
    const int eos = tokenizer->TokenToId("<｜hy_place▁holder▁no▁2｜>");

    QwenEngine engine;
    engine.generation_mode = "greedy";
    engine.init({0}, bmodel_path);

    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    int token = engine.forward_first(input_ids);
    const auto first = clock::now();
    std::vector<int32_t> output_ids;
    while (token != eos && engine.token_length < engine.SEQLEN - 1 &&
           static_cast<int>(output_ids.size()) < max_new_tokens) {
        output_ids.push_back(token);
        token = engine.forward_next();
    }
    const auto end = clock::now();

    std::cout << "Translation: " << tokenizer->Decode(output_ids) << "\n";
    const double prefill_ms =
        std::chrono::duration<double, std::milli>(first - start).count();
    const double decode_s = std::chrono::duration<double>(end - first).count();
    std::cout << "Input tokens: " << input_ids.size()
              << ", output tokens: " << output_ids.size() << "\n";
    std::cout << "Prefill: " << prefill_ms << " ms, decode: "
              << (decode_s > 0 ? output_ids.size() / decode_s : 0) << " token/s\n";
    engine.deinit();
    return 0;
}
