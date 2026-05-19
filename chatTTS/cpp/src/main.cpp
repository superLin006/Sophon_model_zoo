#include "chattts.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cstring>
#include <vector>
#include <fstream>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --model-dir    <dir>   bmodels + assets directory (default: ../models)\n"
              << "  --text         <text>  text to synthesize\n"
              << "  --output       <file>  output wav file (default: output.wav)\n"
              << "  --spk-emb      <file>  speaker embedding binary (float32, 768 values)\n"
              << "  --speed        <1-9>   speaking speed (default: 5)\n"
              << "  --temp         <val>   sampling temperature (default: 0.0001)\n"
              << "  --tpu-id       <id>    TPU device id (default: 0)\n"
              << "  --max-tokens   <n>     max GPT decode steps (default: 2048)\n"
              << "  --stream                enable streaming mode\n"
              << "  --stream-batch <n>     GPT steps per decoder chunk (default: 24)\n";
}

int main(int argc, char* argv[]) {
    std::string model_dir  = "../models";
    std::string text       = "大家好，我是一个文本转语音模型，专为对话场景设计。";
    std::string output     = "output.wav";
    std::string spk_file;
    int   speed        = 5;
    float temp         = 0.0001f;
    int   tpu_id       = 0;
    int   max_token    = 2048;
    bool  stream_mode  = false;
    int   stream_batch = 24;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--model-dir")    && i+1 < argc) model_dir    = argv[++i];
        else if (!std::strcmp(argv[i], "--text")         && i+1 < argc) text         = argv[++i];
        else if (!std::strcmp(argv[i], "--output")       && i+1 < argc) output       = argv[++i];
        else if (!std::strcmp(argv[i], "--spk-emb")      && i+1 < argc) spk_file     = argv[++i];
        else if (!std::strcmp(argv[i], "--speed")        && i+1 < argc) speed        = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--temp")         && i+1 < argc) temp         = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--tpu-id")       && i+1 < argc) tpu_id       = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-tokens")   && i+1 < argc) max_token    = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--stream"))                      stream_mode  = true;
        else if (!std::strcmp(argv[i], "--stream-batch") && i+1 < argc) stream_batch = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--help"))  { print_usage(argv[0]); return 0; }
    }

    ChatTTSConfig cfg;
    cfg.gpt_model_path     = model_dir + "/chattts-llama_int4_1dev_1024_bm1684x.bmodel";
    cfg.decoder_model_path = model_dir + "/decoder_1-768-1024_bm1684x.bmodel";
    cfg.vocos_model_path   = model_dir + "/vocos_1-100-2048_bm1684x.bmodel";
    cfg.vocab_path         = model_dir + "/asset/tokenizer/vocab.txt";
    cfg.homophones_map_path= model_dir + "/asset/homophones_map.json";
    cfg.tpu_id             = tpu_id;

    std::cout << "[1/3] Loading models from: " << model_dir << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    ChatTTS tts(cfg);

    if (!spk_file.empty()) {
        if (!tts.load_speaker(spk_file)) {
            std::cerr << "Warning: failed to load speaker embedding: " << spk_file << std::endl;
        } else {
            std::cout << "Speaker loaded: " << spk_file << std::endl;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[1/3] Done. Load time: " << load_ms / 1000.0 << "s\n";

    std::cout << "[2/3] Inferring"
              << (stream_mode ? " [STREAM]" : "") << ": \""
              << text.substr(0, 60) << (text.size() > 60 ? "..." : "") << "\"\n";

    InferParams params;
    params.temperature   = temp;
    params.speed         = speed;
    params.max_new_token = max_token;

    std::vector<float> pcm;
    auto t2 = std::chrono::steady_clock::now();

    if (!stream_mode) {
        // ── Non-streaming ────────────────────────────────────────────────────
        pcm = tts.infer(text, params);
    } else {
        // ── Streaming ────────────────────────────────────────────────────────
        StreamParams sparams;
        sparams.stream_batch        = stream_batch;
        sparams.pass_first_n_batches = 2;

        bool first_chunk = true;
        auto t_start = t2;

        tts.infer_stream(text, params, sparams,
            [&](const std::vector<float>& chunk) {
                if (first_chunk) {
                    auto t_first = std::chrono::steady_clock::now();
                    double ttfa_ms = std::chrono::duration<double, std::milli>(
                                         t_first - t_start).count();
                    std::cout << "  First chunk  : " << ttfa_ms << "ms ("
                              << chunk.size() << " samples = "
                              << (double)chunk.size()/24000.0 << "s audio)\n";
                    first_chunk = false;
                }
                pcm.insert(pcm.end(), chunk.begin(), chunk.end());
            });
    }

    auto t3 = std::chrono::steady_clock::now();

    if (pcm.empty()) {
        std::cerr << "Error: inference returned empty audio.\n";
        return 1;
    }

    double infer_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double audio_sec = (double)pcm.size() / 24000.0;
    double rtf       = (infer_ms / 1000.0) / audio_sec;

    std::cout << "[2/3] Done.\n"
              << "  Audio length : " << audio_sec         << "s\n"
              << "  Infer time   : " << infer_ms / 1000.0 << "s\n"
              << "  RTF          : " << rtf                << "\n";

    std::cout << "[3/3] Saving to: " << output << std::endl;
    if (!ChatTTS::save_wav(output, pcm)) {
        std::cerr << "Error: failed to save WAV.\n";
        return 1;
    }
    std::cout << "[3/3] Done.\n";
    return 0;
}
