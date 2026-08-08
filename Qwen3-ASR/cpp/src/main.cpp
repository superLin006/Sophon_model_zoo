//
// Qwen3-ASR-0.6B BM1684X 推理 CLI（C++ / 纯 bmrt）
//
// 单个音频：
//   ./qwen3_asr_bm1684x \
//     --encoder_bmodel models/BM1684X/qwen3_asr_encoder_F16.bmodel \
//     --qwen3_bmodel   models/BM1684X/qwen3_asr_llm_w4bf16_seq512_bm1684x.bmodel \
//     --model_dir      . --audio test_data/test_zh.wav
//
// 批量目录（bmodel 只加载一次，循环推理）：
//   ./qwen3_asr_bm1684x ... --audio_dir test_data
//
// 流式模式（按 1s 块喂音频，打印中间结果，最后定稿）：
//   ./qwen3_asr_bm1684x ... --stream --stream_encoder models/BM1684X/qwen3_asr_encoder_w500_F16.bmodel \
//       --audio test_data/test_zh.wav
//

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <getopt.h>
#include <dirent.h>

#include "asr_engine.h"

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s\n"
        "  --encoder_bmodel <path>  encoder bmodel (required)\n"
        "  --qwen3_bmodel   <path>  LLM bmodel (required)\n"
        "  --model_dir      <path>  含 prefix/suffix_embeds.bin + mel_filters.npz + tokenizer.json (required)\n"
        "  --audio          <path>  单个 WAV 文件\n"
        "  --audio_dir      <path>  批量：目录下所有 *.wav（与 --audio 二选一）\n"
        "  --max_new_tokens <n>     默认 256\n"
        "  --device         <n>     设备 ID，默认 0\n",
        prog);
}

int main(int argc, char** argv) {
    std::string enc_path, qwen_path, model_dir, audio, audio_dir, stream_enc, text_ids;
    int max_new_tokens = 256, device = 0;
    bool stream_mode = false;

    static struct option long_opts[] = {
        {"encoder_bmodel",  required_argument, 0, 'e'},
        {"qwen3_bmodel",    required_argument, 0, 'q'},
        {"model_dir",       required_argument, 0, 'm'},
        {"audio",           required_argument, 0, 'a'},
        {"audio_dir",       required_argument, 0, 'd'},
        {"max_new_tokens",  required_argument, 0, 'n'},
        {"device",          required_argument, 0, 'v'},
        {"stream",          no_argument,       0, 's'},
        {"stream_encoder",  required_argument, 0, 'w'},
        {"text",            required_argument, 0, 't'},
        {0, 0, 0, 0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "e:q:m:a:d:n:v:sw:t:", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'e': enc_path = optarg; break;
            case 'q': qwen_path = optarg; break;
            case 'm': model_dir = optarg; break;
            case 'a': audio = optarg; break;
            case 'd': audio_dir = optarg; break;
            case 'n': max_new_tokens = atoi(optarg); break;
            case 'v': device = atoi(optarg); break;
            case 's': stream_mode = true; break;
            case 'w': stream_enc = optarg; break;
            case 't': text_ids = optarg; break;
            default: print_usage(argv[0]); return 1;
        }
    }
    if (enc_path.empty() || qwen_path.empty() || model_dir.empty() ||
        (audio.empty() && audio_dir.empty() && text_ids.empty())) {
        print_usage(argv[0]);
        return 1;
    }

    asr::AsrPipeline pipe;
    if (!pipe.init(enc_path, qwen_path, model_dir, device)) {
        fprintf(stderr, "[main] init failed\n");
        return 1;
    }

    // ── 纯文本模式（调试）：--text "1 2 3 4 5"（空格分隔 token id）──
    if (!text_ids.empty()) {
        std::vector<int> ids;
        size_t p = 0;
        while (p < text_ids.size()) {
            while (p < text_ids.size() && text_ids[p] == ' ') p++;
            size_t q = p;
            while (q < text_ids.size() && text_ids[q] != ' ') q++;
            if (q > p) ids.push_back(atoi(text_ids.substr(p, q - p).c_str()));
            p = q;
        }
        printf("[Text] input_ids=%zu tokens\n", ids.size());
        std::string out = pipe.text_generate(ids, max_new_tokens);
        printf("[Output] %s\n", out.c_str());
        pipe.deinit();
        return 0;
    }

    // ── 流式模式：按 1s（16000 samples）块喂音频，打印中间结果，最后定稿 ──
    if (stream_mode) {
        if (audio.empty() || stream_enc.empty()) {
            fprintf(stderr, "[main] --stream 需要 --audio 和 --stream_encoder\n");
            return 1;
        }
        if (!pipe.init_stream(stream_enc)) {
            fprintf(stderr, "[main] stream init failed\n");
            return 1;
        }
        std::vector<float> samples;
        if (!asr::load_wav_16k_mono(audio, samples)) {
            fprintf(stderr, "[main] load wav failed: %s\n", audio.c_str());
            return 1;
        }
        // 2s 更新频率：每 2s 音频的处理耗时 ~1.2-1.7s（生成 32 token 占大头）≤ 2s → 实时（RTF < 1）
        // 1s 块实测 RTF 1.2-1.9 超实时（每块固定 prefill+生成开销摊薄不足）
        const int CHUNK = 16000;   // 1s
        printf("=== [Stream] %s (%d samples, %.1fs) ===\n", audio.c_str(),
               (int)samples.size(), samples.size() / 16000.0);
        auto t0 = std::chrono::steady_clock::now();
        double audio_played = 0;
        for (size_t off = 0; off < samples.size(); off += CHUNK) {
            int n = std::min(CHUNK, (int)(samples.size() - off));
            double chunk_sec = n / 16000.0;
            pipe.stream_push(samples.data() + off, n);
            auto tu0 = std::chrono::steady_clock::now();
            std::string partial = pipe.stream_update(32);
            double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tu0).count();
            audio_played += chunk_sec;
            double t = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            printf("[%4.1fs] %s   (audio %.1fs | proc %.2fs | RTF %.2f)\n",
                   t, partial.empty() ? "(...)" : partial.c_str(),
                   audio_played, dt, dt / chunk_sec);
        }
        std::string final_text = pipe.stream_finish(256);
        printf("\n[Final] %s\n", final_text.c_str());
        return 0;
    }

    auto run_one = [&](const std::string& wav) {
        auto t0 = std::chrono::steady_clock::now();
        std::string text = pipe.transcribe(wav, max_new_tokens);
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        printf("\n========================================\n");
        printf("[Audio] %s\n", wav.c_str());
        printf("[Output] %s\n", text.empty() ? "(failed)" : text.c_str());
        printf("[Perf]   %.2fs\n", dt);
        printf("========================================\n");
    };

    if (!audio.empty()) {
        run_one(audio);
    } else {
        DIR* dir = opendir(audio_dir.c_str());
        if (!dir) { fprintf(stderr, "[main] cannot open dir %s\n", audio_dir.c_str()); return 1; }
        std::vector<std::string> files;
        struct dirent* ent;
        while ((ent = readdir(dir)) != nullptr) {
            std::string name = ent->d_name;
            if (name.size() > 4 && name.substr(name.size() - 4) == ".wav")
                files.push_back(audio_dir + "/" + name);
        }
        closedir(dir);
        std::sort(files.begin(), files.end());
        for (const auto& f : files) run_one(f);
    }

    pipe.deinit();
    return 0;
}
