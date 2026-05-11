#pragma once

#include <string>
#include <vector>
#include <functional>

#include "bmruntime_interface.h"
#include "utils/audio_utils.h"

// whisper-base 推理参数
#define N_AUDIO_CTX   1500
#define N_STATE        512
#define N_LAYER          6
#define N_HEAD           8
#define PADDING_SIZE   448
#define WHISPER_SOT      50258
#define WHISPER_EOT      50257
#define WHISPER_TRANSCRIBE 50359
#define WHISPER_NO_TIMESTAMPS 50363
#define WHISPER_LANG_ZH  50260
#define WHISPER_LANG_EN  50259

using TokenCallback = std::function<void(const std::string&)>;

class WhisperInference {
public:
    WhisperInference();
    ~WhisperInference();

    int  init(const char* model_dir, const char* precision = "F32");
    std::string run(const char* audio_file, const char* language,
                    TokenCallback callback = nullptr);
    void release();

private:
    // BMRuntime 资源
    bm_handle_t      bm_handle_   = nullptr;
    void*            encoder_rt_  = nullptr;
    void*            decoder_rt_  = nullptr;
    const bm_net_info_t* encoder_info_ = nullptr;
    const bm_net_info_t* decoder_info_ = nullptr;

    // 模型加载
    bool load_encoder(const std::string& path);
    bool load_decoder(const std::string& path);

    // 推理
    bool run_encoder(const std::vector<float>& mel,
                     std::vector<float>& audio_features);
    bool run_decoder(const std::vector<float>& audio_features,
                     int language_token,
                     std::vector<int>& tokens,
                     TokenCallback callback);

    // KV Cache 管理（与 MTK 完全一致）
    void reset_kv_cache();
    void get_position_embedding(int pos, float* out);   // 从 positional_embedding_ 切片
    void create_self_attn_mask(int cache_len, float* out); // [1,1,1,449]

    // 文本处理
    std::string decode_tokens(const std::vector<int>& tokens);
    std::string decode_single_token(int token_id);

    // 数据资源
    std::vector<float>     positional_embedding_;  // [448, 512]，从 bmodel 内取出或 npy 加载
    std::vector<VocabEntry> vocab_;
    std::vector<float>     mel_filters_;           // [80 x 201]

    // KV Cache buffers（与 MTK 对齐，固定 [N_LAYER, 1, PADDING_SIZE, N_STATE]）
    // 展平存储：[N_LAYER * PADDING_SIZE * N_STATE]，按 layer 连续
    std::vector<float> past_self_k_;    // [6, 1, 448, 512]
    std::vector<float> past_self_v_;
    std::vector<float> cross_k_;        // [6, 1, 1500, 512]，首步后固定
    std::vector<float> cross_v_;

    int  cache_len_              = 0;
    bool cross_cache_initialized_ = false;
    bool initialized_            = false;
    bool debug_mode_             = false;
};
