#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <getopt.h>

#include "tts_engine.h"

static bool write_wav(const std::string& path, const std::vector<float>& pcm, int sr) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    int n = (int)pcm.size();
    int byte_rate = sr * 2;
    // WAV header
    auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w32(36 + n * 2);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(1); w32(sr); w32(byte_rate); w16(2); w16(16);
    fwrite("data", 1, 4, f); w32(n * 2);
    for (int i = 0; i < n; i++) {
        float v = pcm[i];
        if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    std::string talker_bmodel, cp_bmodel, codec_bmodel, cp_cache_bmodel, model_dir, text,
                speaker = "Vivian", language = "Chinese", out = "out.wav", batch_file;
    int device = 0, max_new_tokens = 512, seed = 42;
    bool do_sample = false, cp_only = false;
    static struct option opts[] = {
        {"talker_bmodel", required_argument, 0, 't'},
        {"cp_bmodel", required_argument, 0, 'c'},
        {"cp_cache_bmodel", required_argument, 0, 'k'},
        {"codec_bmodel", required_argument, 0, 'd'},
        {"model_dir", required_argument, 0, 'm'},
        {"text", required_argument, 0, 'x'},
        {"speaker", required_argument, 0, 's'},
        {"language", required_argument, 0, 'l'},
        {"out", required_argument, 0, 'o'},
        {"max_new_tokens", required_argument, 0, 'n'},
        {"max-new-tokens", required_argument, 0, 'n'},
        {"device", required_argument, 0, 'v'},
        {"sample", no_argument, 0, 'p'},
        {"cp_only", no_argument, 0, 'q'},
        {"seed", required_argument, 0, 'e'},
        {"batch", required_argument, 0, 'b'},
        {0, 0, 0, 0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "t:c:k:d:m:x:s:l:o:n:v:pqe:b:", opts, nullptr)) != -1) {
        switch (c) {
            case 't': talker_bmodel = optarg; break;
            case 'c': cp_bmodel = optarg; break;
            case 'k': cp_cache_bmodel = optarg; break;
            case 'd': codec_bmodel = optarg; break;
            case 'm': model_dir = optarg; break;
            case 'x': text = optarg; break;
            case 's': speaker = optarg; break;
            case 'l': language = optarg; break;
            case 'o': out = optarg; break;
            case 'n': max_new_tokens = atoi(optarg); break;
            case 'v': device = atoi(optarg); break;
            case 'p': do_sample = true; break;
            case 'q': cp_only = true; break;
            case 'e': seed = atoi(optarg); break;
            case 'b': batch_file = optarg; break;
            default: return 1;
        }
    }
    if (talker_bmodel.empty() || cp_bmodel.empty() || codec_bmodel.empty() ||
        model_dir.empty() || (!cp_only && text.empty() && batch_file.empty())) {
        fprintf(stderr, "usage: %s --talker_bmodel ... --cp_bmodel ... [--cp_cache_bmodel ...] --codec_bmodel ... --model_dir ... [--text ... | --cp_only]\n", argv[0]);
        return 1;
    }

    qwen3tts::TtsEngine eng;
    if (!eng.init(talker_bmodel, cp_bmodel, codec_bmodel, model_dir, device, cp_cache_bmodel)) {
        fprintf(stderr, "[main] init failed\n"); return 1;
    }

    if (cp_only) {
        // debug：不跑 talker，直接用固定 last_hidden 跑 CP 链路
        bool ok = eng.test_cp_only();
        eng.deinit();
        return ok ? 0 : 1;
    }

    // 批量模式：一次加载模型，连续合成多条（name|speaker|lang|text 每行）
    if (!batch_file.empty()) {
        FILE* bf = fopen(batch_file.c_str(), "r");
        if (!bf) { fprintf(stderr, "[main] batch file open failed: %s\n", batch_file.c_str()); return 1; }
        char line[4096];
        int n_ok = 0, n_fail = 0;
        while (fgets(line, sizeof(line), bf)) {
            line[strcspn(line, "\r\n")] = 0;
            if (!line[0] || line[0] == '#') continue;
            char name[128] = {}, spk[64] = {}, lang[64] = {}, txt[3000] = {};
            if (sscanf(line, "%127[^|]|%63[^|]|%63[^|]|%2999[^\n]", name, spk, lang, txt) != 4) {
                fprintf(stderr, "[main] bad line: %s\n", line); continue;
            }
            auto t0 = std::chrono::steady_clock::now();
            std::vector<float> pcm;
            int sr = 0;
            bool ok = eng.generate(txt, spk, lang, pcm, sr, max_new_tokens, do_sample, seed);
            auto t1 = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            std::string out_path = out + "/" + name + ".wav";
            if (ok && !pcm.empty()) {
                if (!write_wav(out_path, pcm, sr)) {
                    fprintf(stderr, "[BATCH] %s 写盘失败: %s\n", name, out_path.c_str());
                    n_fail++;
                } else {
                    printf("[BATCH] %s OK dur=%.2fs proc=%.2fs RTF=%.3f\n",
                           name, pcm.size() / (double)sr, dt, dt / (pcm.size() / (double)sr));
                    n_ok++;
                }
            } else {
                printf("[BATCH] %s FAILED proc=%.2fs\n", name, dt);
                n_fail++;
            }
            fflush(stdout);
        }
        fclose(bf);
        eng.deinit();
        printf("[BATCH] done: %d ok, %d failed\n", n_ok, n_fail);
        return 0;
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> pcm;
    int sr = 0;
    bool ok = eng.generate(text, speaker, language, pcm, sr, max_new_tokens, do_sample, seed);
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    if (!ok) { fprintf(stderr, "[main] generate failed\n"); return 1; }
    write_wav(out, pcm, sr);
    printf("[TTS] samples=%zu sr=%d dur=%.2fs proc=%.2fs RTF=%.3f\n",
           pcm.size(), sr, pcm.size() / (double)sr, dt, dt / (pcm.size() / (double)sr));
    eng.deinit();
    return 0;
}
