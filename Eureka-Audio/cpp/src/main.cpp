//
// Eureka-Audio BM1684X 推理 CLI（C++ / 纯 bmrt）
//
// 单个音频：
//   ./eureka_audio_bm1684x \
//     --whisper_bmodel models/BM1684X/whisper_encoder_b1_bf16.bmodel \
//     --qwen3_bmodel   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
//     --model_dir      .  --audio  qa_example.wav
//
// 批量目录（bmodel 只加载一次，循环推理，输出分段性能）：
//   ./eureka_audio_bm1684x ... --audio_dir intent_wav
//

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <getopt.h>
#include <dirent.h>

#include "eureka_audio.h"
#include "whisper_mel.h"
#include "tokenizer.h"

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s\n"
        "  --whisper_bmodel <path>  (required)\n"
        "  --qwen3_bmodel   <path>  (required)\n"
        "  --model_dir      <path>  prefix/suffix_embeds.bin + mel_filters.npz + tokenizer.json (required)\n"
        "  --audio          <path>  单个 WAV 文件\n"
        "  --audio_dir      <path>  批量：目录下所有 *.wav（与 --audio 二选一）\n"
        "  --prompt         <text>  系统提示词 (optional)\n"
        "  --max_new_tokens <n>     默认 64\n"
        "  --device         <n>     设备 ID，默认 0\n",
        prog);
}

static const char* DEFAULT_SYSTEM_PROMPT =
    "把用户的语音指令归类为以下动作之一："
    "open_whiteboard, close_window, set_volume, open_camera, set_pen, "
    "draw_shape, set_tool, save_file, screenshot。\n"
    "输出JSON：{\"action\":\"动作名\",\"params\":{}}。只输出JSON，不要解释。";

// 读 .bin embed 文件 → float vector，返回 token 数（/2048）
static int load_embeds(const std::string& path, std::vector<float>& out) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    out.resize(fsize / sizeof(float));
    if (fread(out.data(), sizeof(float), out.size(), fp) != out.size()) {
        fclose(fp); return -1;
    }
    fclose(fp);
    return (int)(out.size() / 2048);
}

// 对单个音频做一次完整推理，打印结果 + 分段性能
static void run_one(eureka::EurekaAudioPipeline& pipeline,
                    eureka::Qwen3Tokenizer& tokenizer,
                    eureka::WhisperMel& mel_proc,
                    const std::vector<float>& prefix_embeds, int prefix_len,
                    const std::vector<float>& suffix_embeds, int suffix_len,
                    const std::string& audio_path, int max_new_tokens) {
    // WAV → 实际 audio token 数 + mel chunks
    std::vector<float> wav_samples;
    int real_frames = -1;
    if (eureka::load_wav_16k_mono(audio_path, wav_samples))
        real_frames = (int)(wav_samples.size() / 1280);

    auto chunks = mel_proc.wav_to_mel_chunks(audio_path);
    if (chunks.empty()) {
        fprintf(stderr, "[%s] mel failed\n", audio_path.c_str());
        return;
    }
    int n_chunks = (int)chunks.size();
    std::vector<float> mel_buf;
    mel_buf.reserve((size_t)n_chunks * 128 * 3000);
    for (auto& ch : chunks) mel_buf.insert(mel_buf.end(), ch.begin(), ch.end());

    std::vector<int> out_ids = pipeline.generate(
        mel_buf.data(), n_chunks,
        prefix_embeds.data(), prefix_len,
        suffix_len > 0 ? suffix_embeds.data() : nullptr, suffix_len,
        max_new_tokens, tokenizer.eos_token_id(), real_frames);

    std::string resp = tokenizer.decode(out_ids);
    double dec_tps = pipeline.last_decode_tokens > 0
                   ? pipeline.last_decode_tokens / pipeline.last_decode_s : 0;
    printf("[%s]\n", audio_path.c_str());
    printf("  Output: %s\n", resp.c_str());
    printf("  Perf:   whisper=%.2fs  prefill=%.2fs(%dtok)  decode=%.2fs(%dtok, %.1f tok/s)\n",
           pipeline.last_whisper_s,
           pipeline.last_prefill_s, pipeline.last_prefill_len,
           pipeline.last_decode_s, pipeline.last_decode_tokens, dec_tps);
}

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    std::string whisper_bmodel, qwen3_bmodel, model_dir, audio_path, audio_dir, system_prompt;
    int max_new_tokens = 64;
    int device_id      = 0;
    system_prompt      = DEFAULT_SYSTEM_PROMPT;

    static struct option opts[] = {
        {"whisper_bmodel",  required_argument, 0, 'w'},
        {"qwen3_bmodel",    required_argument, 0, 'q'},
        {"model_dir",       required_argument, 0, 'm'},
        {"audio",           required_argument, 0, 'a'},
        {"audio_dir",       required_argument, 0, 'D'},
        {"prompt",          required_argument, 0, 'p'},
        {"max_new_tokens",  required_argument, 0, 'n'},
        {"device",          required_argument, 0, 'd'},
        {0, 0, 0, 0},
    };
    int c, idx;
    while ((c = getopt_long(argc, argv, "", opts, &idx)) != -1) {
        switch (c) {
            case 'w': whisper_bmodel = optarg; break;
            case 'q': qwen3_bmodel   = optarg; break;
            case 'm': model_dir      = optarg; break;
            case 'a': audio_path     = optarg; break;
            case 'D': audio_dir      = optarg; break;
            case 'p': system_prompt  = optarg; break;
            case 'n': max_new_tokens = atoi(optarg); break;
            case 'd': device_id      = atoi(optarg); break;
            default:  print_usage(argv[0]); return 1;
        }
    }
    if (whisper_bmodel.empty() || qwen3_bmodel.empty() || model_dir.empty() ||
        (audio_path.empty() && audio_dir.empty())) {
        print_usage(argv[0]);
        return 1;
    }

    // ── 初始化（bmodel + tokenizer + mel filter，只做一次）──────────────────
    printf("Initializing pipeline ...\n");
    auto t0 = std::chrono::steady_clock::now();
    eureka::EurekaAudioPipeline pipeline;
    if (!pipeline.init(whisper_bmodel, qwen3_bmodel, device_id)) {
        fprintf(stderr, "Pipeline init failed\n"); return 1;
    }
    eureka::Qwen3Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "Tokenizer load failed from %s\n", model_dir.c_str()); return 1;
    }
    eureka::WhisperMel mel_proc;
    if (!mel_proc.load_filters(model_dir + "/mel_filters.npz")) {
        fprintf(stderr, "Failed to load mel filters\n"); return 1;
    }
    std::vector<float> prefix_embeds, suffix_embeds;
    int prefix_len = load_embeds(model_dir + "/prefix_embeds.bin", prefix_embeds);
    if (prefix_len <= 0) {
        fprintf(stderr, "prefix_embeds.bin not found in %s\n", model_dir.c_str()); return 1;
    }
    int suffix_len = load_embeds(model_dir + "/suffix_embeds.bin", suffix_embeds);
    if (suffix_len < 0) suffix_len = 0;
    auto t_init = std::chrono::steady_clock::now();
    printf("Init done: %.2fs (prefix=%d suffix=%d tokens)\n\n",
           std::chrono::duration<double>(t_init - t0).count(), prefix_len, suffix_len);

    // ── 收集待处理音频列表 ──────────────────────────────────────────────────
    std::vector<std::string> audios;
    if (!audio_path.empty()) {
        audios.push_back(audio_path);
    } else {
        DIR* dir = opendir(audio_dir.c_str());
        if (!dir) { fprintf(stderr, "Cannot open dir %s\n", audio_dir.c_str()); return 1; }
        struct dirent* ent;
        while ((ent = readdir(dir)) != nullptr) {
            std::string n = ent->d_name;
            if (n.size() > 4 && n.substr(n.size() - 4) == ".wav")
                audios.push_back(audio_dir + "/" + n);
        }
        closedir(dir);
        std::sort(audios.begin(), audios.end());
    }

    // ── 循环推理（bmodel 常驻，逐个音频）─────────────────────────────────────
    double sum_whisper = 0, sum_prefill = 0, sum_decode = 0;
    int    sum_dec_tok = 0;
    for (auto& a : audios) {
        run_one(pipeline, tokenizer, mel_proc,
                prefix_embeds, prefix_len, suffix_embeds, suffix_len,
                a, max_new_tokens);
        sum_whisper += pipeline.last_whisper_s;
        sum_prefill += pipeline.last_prefill_s;
        sum_decode  += pipeline.last_decode_s;
        sum_dec_tok += pipeline.last_decode_tokens;
    }

    // ── 性能汇总（批量时）────────────────────────────────────────────────────
    if (audios.size() > 1) {
        int N = (int)audios.size();
        printf("\n==================== 性能汇总 (%d cases) ====================\n", N);
        printf("  whisper encoder : 平均 %.2fs/case\n", sum_whisper / N);
        printf("  qwen3 prefill   : 平均 %.2fs/case\n", sum_prefill / N);
        printf("  qwen3 decode    : 平均 %.2fs/case, 共 %d tok, %.1f tok/s\n",
               sum_decode / N, sum_dec_tok,
               sum_decode > 0 ? sum_dec_tok / sum_decode : 0);
        printf("  端到端          : 平均 %.2fs/case\n",
               (sum_whisper + sum_prefill + sum_decode) / N);
        printf("============================================================\n");
    }

    pipeline.deinit();
    return 0;
}
