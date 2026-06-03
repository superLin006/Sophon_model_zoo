#include "whisper_inference.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <algorithm>

#define LOG(msg)   std::cout << "[INFO] " << msg << std::endl
#define LOGE(msg)  std::cerr << "[ERROR] " << msg << std::endl
#define LOGD(msg)  if (debug_mode_) { std::cout << "[DEBUG] " << msg << std::endl; }

// ============================================================
// BMRuntime 辅助：把 host float 数组推理一次
// 封装 bmrt_launch_tensor_ex 的样板代码
// ============================================================

// 用 bmrt 做单次推理：inputs/outputs 已在 device 上分配
static bool bm_infer(void* rt, const bm_net_info_t* info,
                     std::vector<bm_tensor_t>& in_tensors,
                     std::vector<bm_tensor_t>& out_tensors) {
    bool ok = bmrt_launch_tensor_ex(rt, info->name,
                                    in_tensors.data(),  in_tensors.size(),
                                    out_tensors.data(), out_tensors.size(),
                                    true,   // need_perf_opt
                                    false); // sync
    if (!ok) return false;
    // 等待推理完成（bmrt_get_bm_handle 返回 handle，sp4 API 只有一个参数）
    bm_handle_t handle = (bm_handle_t)bmrt_get_bm_handle(rt);
    bm_thread_sync(handle);
    return true;
}

// ============================================================
// Constructor / Destructor
// ============================================================

WhisperInference::WhisperInference() {}

WhisperInference::~WhisperInference() { release(); }

// ============================================================
// init
// ============================================================

int WhisperInference::init(const char* model_dir, const char* precision) {
    if (initialized_) return 0;

    const char* dbg = std::getenv("WHISPER_DEBUG");
    debug_mode_ = (dbg && std::string(dbg) == "1");

    LOG("Initializing from: " << model_dir << "  precision: " << precision);
    std::string base(model_dir);
    std::string prec(precision);

    // BM handle（device 0）
    if (bm_dev_request(&bm_handle_, 0) != BM_SUCCESS) {
        LOGE("bm_dev_request failed"); return -1;
    }

    // 加载 bmodel
    if (!load_encoder(base + "/whisper_base_encoder_" + prec + ".bmodel")) return -1;
    if (!load_decoder(base + "/whisper_base_decoder_" + prec + ".bmodel")) return -1;

    // mel filters [80 x 201]
    mel_filters_.resize(N_MELS * MELS_FILTERS_SIZE);
    if (read_mel_filters((base + "/mel_80_filters.txt").c_str(),
                         mel_filters_.data(), N_MELS * MELS_FILTERS_SIZE) != 0) {
        LOGE("Failed to load mel filters"); return -1;
    }

    // positional embedding [448, 512]，从 npy 文件加载（纯二进制，跳过 128 字节 npy header）
    {
        std::string pe_path = base + "/positional_embedding.npy";
        FILE* fp = fopen(pe_path.c_str(), "rb");
        if (!fp) { LOGE("positional_embedding.npy not found: " << pe_path); return -1; }
        fseek(fp, 128, SEEK_SET);
        size_t sz = PADDING_SIZE * N_STATE;
        positional_embedding_.resize(sz);
        if (fread(positional_embedding_.data(), sizeof(float), sz, fp) != sz) {
            LOGE("positional_embedding read error"); fclose(fp); return -1;
        }
        fclose(fp);
        LOG("Positional embedding loaded: [" << PADDING_SIZE << ", " << N_STATE << "]");
    }

    // vocab
    vocab_.resize(VOCAB_NUM);
    if (read_vocab((base + "/vocab.txt").c_str(), vocab_.data()) != 0) {
        LOGE("Failed to load vocab"); return -1;
    }

    // KV cache 常驻 device，无需 host buffer（在 run_decoder 内绑定到 decoder KV input tensor）

    initialized_ = true;
    LOG("Init done");
    return 0;
}

bool WhisperInference::load_encoder(const std::string& path) {
    encoder_rt_ = bmrt_create(bm_handle_);
    if (!bmrt_load_bmodel(encoder_rt_, path.c_str())) {
        LOGE("Load encoder bmodel failed: " << path); return false;
    }
    // sp4 API: bmrt_get_network_names(rt, &names)
    const char** names = nullptr;
    bmrt_get_network_names(encoder_rt_, &names);
    encoder_info_ = bmrt_get_network_info(encoder_rt_, names[0]);
    LOG("Encoder loaded: " << path);
    return true;
}

bool WhisperInference::load_decoder(const std::string& path) {
    decoder_rt_ = bmrt_create(bm_handle_);
    if (!bmrt_load_bmodel(decoder_rt_, path.c_str())) {
        LOGE("Load decoder bmodel failed: " << path); return false;
    }
    const char** names = nullptr;
    bmrt_get_network_names(decoder_rt_, &names);
    decoder_info_ = bmrt_get_network_info(decoder_rt_, names[0]);
    LOG("Decoder loaded: " << path);
    return true;
}

// ============================================================
// run
// ============================================================

std::string WhisperInference::run(const char* audio_file, const char* language,
                                  TokenCallback callback) {
    if (!initialized_) { LOGE("Not initialized"); return ""; }

    LOG("Running on: " << audio_file);
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. 加载音频
    audio_buffer_t audio{};
    if (load_audio(audio_file, &audio) != 0) { LOGE("load_audio failed"); return ""; }

    // 2. Mel spectrogram
    std::vector<float> mel;
    audio_preprocess(&audio, mel_filters_.data(), mel);
    free_audio(&audio);
    // mel: [N_MELS, 3000]，需要扩展 batch 维 → [1, N_MELS, 3000]（内存连续，直接用）

    // 3. Encoder
    std::vector<float> audio_features;
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!run_encoder(mel, audio_features)) { LOGE("encoder failed"); return ""; }
    auto t2 = std::chrono::high_resolution_clock::now();

    // 4. Decoder
    int lang_token = WHISPER_LANG_EN;
    if (language && std::string(language) == "zh") lang_token = WHISPER_LANG_ZH;

    std::vector<int> tokens;
    if (!run_decoder(audio_features, lang_token, tokens, callback)) {
        LOGE("decoder failed"); return "";
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    std::string text = decode_tokens(tokens);

    auto enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto dec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto tot_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();

    LOG("Encoder: " << enc_ms << "ms  Decoder: " << dec_ms << "ms  Total: " << tot_ms << "ms"
        << "  Tokens: " << tokens.size());
    LOG("Result: " << text);
    return text;
}

// ============================================================
// Encoder inference
// ============================================================

bool WhisperInference::run_encoder(const std::vector<float>& mel,
                                   std::vector<float>& audio_features) {
    // 输入: mel [1, 80, 3000]
    // 输出: audio_features [1, 1500, 512]
    int in_count  = encoder_info_->input_num;
    int out_count = encoder_info_->output_num;

    std::vector<bm_tensor_t> ins(in_count), outs(out_count);

    // 分配输入 device mem 并上传
    bmrt_tensor(&ins[0], encoder_rt_, BM_FLOAT32,
                encoder_info_->stages[0].input_shapes[0]);
    bm_memcpy_s2d(bm_handle_, ins[0].device_mem,
                  (void*)mel.data());

    // 分配输出 device mem
    bmrt_tensor(&outs[0], encoder_rt_, BM_FLOAT32,
                encoder_info_->stages[0].output_shapes[0]);

    if (!bm_infer(encoder_rt_, encoder_info_, ins, outs)) {
        LOGE("encoder inference failed"); return false;
    }

    // 下载结果
    size_t out_sz = N_AUDIO_CTX * N_STATE;
    audio_features.resize(out_sz);
    bm_memcpy_d2s(bm_handle_, audio_features.data(), outs[0].device_mem);

    // 释放 device mem
    bm_free_device(bm_handle_, ins[0].device_mem);
    bm_free_device(bm_handle_, outs[0].device_mem);

    LOGD("Encoder done, audio_features[0..4]: "
         << audio_features[0] << " " << audio_features[1] << " "
         << audio_features[2] << " " << audio_features[3]);
    return true;
}

// ============================================================
// Decoder inference（与 MTK 逻辑完全一致）
// ============================================================

void WhisperInference::reset_kv_cache() {
    // KV cache 现常驻 device，清零在 run_decoder 分配 in_ts 后做（bm_memset device buffer）。
    // 这里只重置状态标志。
    cache_len_               = 0;
    cross_cache_initialized_ = false;
}

void WhisperInference::get_position_embedding(int pos, float* out) {
    // positional_embedding_: [PADDING_SIZE, N_STATE]
    if (pos >= 0 && pos < PADDING_SIZE) {
        std::memcpy(out,
                    positional_embedding_.data() + pos * N_STATE,
                    N_STATE * sizeof(float));
    } else {
        std::fill(out, out + N_STATE, 0.f);
    }
}

void WhisperInference::create_self_attn_mask(int cache_len, float* out) {
    // [1, 1, 1, PADDING_SIZE+1]：有效位=0，无效位=-1e9
    // 与 MTK create_self_attn_mask 完全一致
    int len = PADDING_SIZE + 1;
    std::fill(out, out + len, -1e9f);
    for (int i = 0; i < cache_len; i++) out[i] = 0.f;
    out[PADDING_SIZE] = 0.f;  // 当前 token 固定在末尾位
}

bool WhisperInference::run_decoder(const std::vector<float>& audio_features,
                                   int language_token,
                                   std::vector<int>& tokens,
                                   TokenCallback callback) {
    reset_kv_cache();
    tokens.clear();

    // 初始 SOT 序列：[SOT, lang, TRANSCRIBE, NO_TIMESTAMPS]
    std::vector<int> init_tokens = {
        WHISPER_SOT, language_token, WHISPER_TRANSCRIBE, WHISPER_NO_TIMESTAMPS
    };
    LOG("Initial tokens: [" << init_tokens[0] << ", " << init_tokens[1]
        << ", " << init_tokens[2] << ", " << init_tokens[3] << "]");

    // Decoder 输入数量：token(1) + audio_features(1) + pos_emb(1) + mask(1)
    //                   + past_self_k(6) + past_self_v(6) + cross_k(6) + cross_v(6) = 28
    // Decoder 输出数量：logits(1) + new_self_k(6) + new_self_v(6)
    //                   + new_cross_k(6) + new_cross_v(6) = 25
    int in_num  = decoder_info_->input_num;   // 28
    int out_num = decoder_info_->output_num;  // 25

    // 工作 buffer（KV 全部常驻 device，host 侧只保留 pos/mask/logits 小 buffer）
    std::vector<float> pos_emb(N_STATE);
    std::vector<float> self_attn_mask(PADDING_SIZE + 1);
    std::vector<float> logits(VOCAB_NUM);

    // 分配 device tensors（一次性，循环复用）
    std::vector<bm_tensor_t> in_ts(in_num), out_ts(out_num);
    for (int i = 0; i < in_num;  i++)
        bmrt_tensor(&in_ts[i],  decoder_rt_, BM_FLOAT32,
                    decoder_info_->stages[0].input_shapes[i]);
    for (int i = 0; i < out_num; i++)
        bmrt_tensor(&out_ts[i], decoder_rt_, BM_FLOAT32,
                    decoder_info_->stages[0].output_shapes[i]);
    // token 输入：ONNX 是 int64，BM1684 编译后降为 INT32，用 BM_INT32 传
    bm_free_device(bm_handle_, in_ts[0].device_mem);
    bmrt_tensor(&in_ts[0], decoder_rt_, BM_INT32,
                decoder_info_->stages[0].input_shapes[0]);

    // KV cache 常驻 device：past_self(in_ts[4..15]) + cross(in_ts[16..27]) 全程留 device。
    // 首步前清零这些 input buffer（原靠每步 host 上传清零的 host buffer 保证，现直接清 device）。
    {
        int z = 0;
        for (int i = 4; i < 28; i++)
            bm_memset_device_ext(bm_handle_, &z, 1, in_ts[i].device_mem);
    }

    auto run_one_step = [&](int token_id) -> bool {
        // --- 准备输入 ---
        // 0. token [1,1]，BM1684 bmodel 编译后为 INT32
        int32_t tok = (int32_t)token_id;
        bm_memcpy_s2d(bm_handle_, in_ts[0].device_mem, &tok);

        // 1. audio_features [1, 1500, 512]
        bm_memcpy_s2d(bm_handle_, in_ts[1].device_mem,
                      (void*)audio_features.data());

        // 2. pos_emb [1, 1, 512]
        get_position_embedding(cache_len_, pos_emb.data());
        bm_memcpy_s2d(bm_handle_, in_ts[2].device_mem, pos_emb.data());

        // 3. self_attn_mask [1, 1, 1, 449]
        create_self_attn_mask(cache_len_, self_attn_mask.data());
        bm_memcpy_s2d(bm_handle_, in_ts[3].device_mem, self_attn_mask.data());

        // 4-27. past_self_k/v(4-15) + cross_k/v(16-27)：全部常驻 device，不再每步上传。
        //   - past_self KV：上一步输出 new_self 已 d2d 写入这些 input buffer 的 cache_len 位置
        //   - cross KV：首步推理后一次性 d2d 写入，之后内容固定不变
        //   首步（cache_len_==0）时这些 input buffer 已被 reset_kv_cache 的 device memset 清零

        // --- 推理 ---
        if (!bm_infer(decoder_rt_, decoder_info_, in_ts, out_ts)) return false;

        // --- 处理输出 ---
        // 0. logits [1, 1, 51865]：仍需下载到 host 做 greedy argmax（小数据，无法避免）
        bm_memcpy_d2s(bm_handle_, logits.data(), out_ts[0].device_mem);

        // self KV：output new_self(out_ts[1..12]) d2d 写入 input past_self(in_ts[4..15])
        //   的 cache_len_ 偏移位置。device 内传输，不经过 host。
        //   past_self input 与 new_self output 是不同 device tensor，读写无冲突（同 Eureka 模式）。
        // 防御：past_self buffer 每层容量 PADDING_SIZE 个 token，cache_len_ 必须 < PADDING_SIZE
        if (cache_len_ >= PADDING_SIZE) {
            LOGE("cache_len " << cache_len_ << " >= PADDING_SIZE " << PADDING_SIZE
                 << "，KV cache 已满"); return false;
        }
        const size_t self_step_bytes = (size_t)N_STATE * sizeof(float);  // 单层单 token KV
        const size_t self_off        = (size_t)cache_len_ * N_STATE * sizeof(float);
        for (int l = 0; l < N_LAYER; l++) {
            // new_self_k_l (out_ts[1+l]) → past_self_k_l (in_ts[4+l]) @ cache_len_
            bm_memcpy_d2d_byte(bm_handle_, in_ts[4 + l].device_mem,  self_off,
                               out_ts[1 + l].device_mem, 0, self_step_bytes);
            // new_self_v_l (out_ts[7+l]) → past_self_v_l (in_ts[10+l]) @ cache_len_
            bm_memcpy_d2d_byte(bm_handle_, in_ts[10 + l].device_mem, self_off,
                               out_ts[7 + l].device_mem, 0, self_step_bytes);
        }

        // cross KV：只首步 d2d。output new_cross(out_ts[13..24]) → input cross(in_ts[16..27])
        //   整块复制，之后内容固定，后续步不再触碰。
        if (!cross_cache_initialized_) {
            const size_t cross_bytes = (size_t)N_AUDIO_CTX * N_STATE * sizeof(float);
            for (int l = 0; l < N_LAYER; l++) {
                bm_memcpy_d2d_byte(bm_handle_, in_ts[16 + l].device_mem, 0,
                                   out_ts[13 + l].device_mem, 0, cross_bytes);
                bm_memcpy_d2d_byte(bm_handle_, in_ts[22 + l].device_mem, 0,
                                   out_ts[19 + l].device_mem, 0, cross_bytes);
            }
            cross_cache_initialized_ = true;
        }
        cache_len_++;
        return true;
    };

    // Phase 1：处理 SOT 序列
    LOG("Phase 1: processing " << init_tokens.size() << " initial tokens...");
    for (int tok : init_tokens) {
        if (!run_one_step(tok)) { LOGE("decoder step failed"); goto cleanup; }
    }

    // Phase 2：自回归生成
    LOG("Phase 2: generating tokens (max " << (PADDING_SIZE - cache_len_) << ")...");
    {
        int max_iter = PADDING_SIZE - cache_len_;
        for (int iter = 0; iter < max_iter; iter++) {
            // Greedy：取前 WHISPER_EOT+1 个 token 中 logit 最大的
            int next_tok = 0;
            float max_val = -1e30f;
            for (int i = 0; i <= WHISPER_EOT; i++) {
                if (logits[i] > max_val) { max_val = logits[i]; next_tok = i; }
            }

            if (next_tok == WHISPER_EOT) {
                LOG("EOT at iter " << iter);
                break;
            }

            tokens.push_back(next_tok);

            if (callback) {
                std::string piece = decode_single_token(next_tok);
                if (!piece.empty()) callback(piece);
            }

            if (!run_one_step(next_tok)) { LOGE("decoder step failed"); goto cleanup; }

            if (cache_len_ >= PADDING_SIZE) {
                LOG("Reached max cache length");
                break;
            }
        }
    }

cleanup:
    // 释放 device mem
    for (auto& t : in_ts)  bm_free_device(bm_handle_, t.device_mem);
    for (auto& t : out_ts) bm_free_device(bm_handle_, t.device_mem);

    LOG("Decoder done, " << tokens.size() << " tokens");
    return true;
}

// ============================================================
// 文本解码
// ============================================================

std::string WhisperInference::decode_single_token(int id) {
    if (id < 0 || id >= VOCAB_NUM) return "";
    const std::string& tok = vocab_[id].token;
    if (tok.size() >= 2 && tok[0] == '<' && tok[1] == '|') return "";
    std::string s = base64_decode(tok);
    replace_substr(s, "\xc4\xa0", " ");  // BPE space Ġ → ASCII space
    return s;
}

std::string WhisperInference::decode_tokens(const std::vector<int>& tokens) {
    std::string text;
    for (int id : tokens) {
        if (id < 0 || id >= VOCAB_NUM) continue;
        const std::string& tok = vocab_[id].token;
        if (tok.size() >= 2 && tok[0] == '<' && tok[1] == '|') continue;
        std::string s = base64_decode(tok);
        text += s;
    }
    replace_substr(text, "\xc4\xa0", " ");
    replace_substr(text, "\n", "");
    return text;
}

// ============================================================
// release
// ============================================================

void WhisperInference::release() {
    if (encoder_rt_) { bmrt_destroy(encoder_rt_); encoder_rt_ = nullptr; }
    if (decoder_rt_) { bmrt_destroy(decoder_rt_); decoder_rt_ = nullptr; }
    if (bm_handle_)  { bm_dev_free(bm_handle_);   bm_handle_  = nullptr; }
    initialized_ = false;
    LOG("Released");
}
