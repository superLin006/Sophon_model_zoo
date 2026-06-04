#pragma once

#include <string>
#include <vector>
#include <functional>

#include "bmruntime_interface.h"
#include "utils/audio_utils.h"

// 维度参数（n_state/n_layer/n_head/n_mels/vocab/n_audio_ctx/padding）
// 全部在 init() 内从 bmodel net_info 运行时读出，base / large-v3-turbo 通用，见成员变量区。
//
// token-id：SOT/EOT/语言 token 在语言块之前，base 与 turbo 一致，保留为宏。
// 但 transcribe / notimestamps 在语言块之后——turbo 比 base 多 1 种语言(100 vs 99)，
// 故这两个 id 会偏移 1，必须按 n_vocab 运行时推导（见 init()），不能写死。
#define WHISPER_SOT      50258
#define WHISPER_EOT      50257
#define WHISPER_LANG_EN  50259   // = SOT+1，语言块起点
#define WHISPER_LANG_ZH  50260

using TokenCallback = std::function<void(const std::string&)>;

class WhisperInference {
public:
    WhisperInference();
    ~WhisperInference();

    // model_name: bmodel 文件名前缀，"base" -> whisper_base_{encoder,decoder}_<prec>.bmodel
    //             "turbo" -> whisper_turbo_..., 维度自动从 bmodel 读取
    int  init(const char* model_dir, const char* precision = "F32",
              const char* model_name = "base");
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
    std::vector<float>     positional_embedding_;  // [padding_size_, n_state_]，npy 加载
    std::vector<VocabEntry> vocab_;
    std::vector<float>     mel_filters_;           // [n_mels_ x 201]

    // 运行时维度（init() 内从 bmodel net_info 读出，base / turbo 通用）
    int n_mels_      = 0;   // encoder mel 输入维度 (base 80 / turbo 128)
    int n_audio_ctx_ = 0;   // encoder 输出帧数 (1500)
    int n_state_     = 0;   // 隐藏维 (base 512 / turbo 1280)
    int n_layer_     = 0;   // decoder 层数 (base 6 / turbo 4)
    int padding_size_= 0;   // KV cache 最大长度 = n_text_ctx (448)
    int vocab_num_   = 0;   // 词表大小 (base 51865 / turbo 51866)
    // 语言块之后的特殊 token（按 vocab_num_ 推导，base/turbo 不同）
    int tok_transcribe_     = 0;
    int tok_no_timestamps_  = 0;

    // KV cache 常驻 device（在 run_decoder 内绑定到 decoder 的 KV input tensor，全程不下载 host）。
    // self KV 每步 output→input d2d 写增量，cross KV 首步 d2d 后固定。详见 run_decoder。
    int  cache_len_              = 0;
    bool cross_cache_initialized_ = false;
    bool initialized_            = false;
    bool debug_mode_             = false;
};
