#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "bmruntime_interface.h"
#include "qwen_mel.h"
#include "tokenizer.h"

namespace asr {

// ── 配置（与 bmodel 编译一致）────────────────────────────────────────────────
struct AsrConfig {
    int mel_frames     = 3000;   // encoder 输入 mel 帧（30s）
    int enc_tokens     = 390;    // encoder 输出帧（30 chunks × 13）
    int enc_d_model    = 1024;   // projector 输出维度（= LLM hidden）
    int hidden_size    = 1024;
    int num_layers     = 28;
    int num_kv_heads   = 8;
    int head_dim       = 128;
    int vocab_size     = 151936;
    int seq_length     = 2048;
    int audio_token_id = 151676;
    int eos_im_end     = 151643;   // <|im_end|>
    int eos_eot        = 151645;   // <|endoftext|>
};

// ── EncoderBmodel ────────────────────────────────────────────────────────────
// 封装 qwen3_asr_encoder_F16.bmodel：mel [1,128,3000] → audio_embeds [390,1024]
class EncoderBmodel {
public:
    EncoderBmodel() = default;
    ~EncoderBmodel();
    bool init(bm_handle_t handle, const std::string& bmodel_path,
              const char* net_name = "qwen3_asr_encoder");
    void release();
    // mel: [1,128,3000] row-major → out [390,1024] row-major
    bool run(const float* mel, std::vector<float>& out);
private:
    bm_handle_t handle_ = nullptr;
    void* p_bmrt_ = nullptr;
    const bm_net_info_t* net_ = nullptr;
    bool inited_ = false;
};

// ── Qwen3Bmodel ──────────────────────────────────────────────────────────────
// 封装 qwen3_asr_llm_w4bf16_*.bmodel（inputs_embeds 版）
// 网络：embedding_cache / block_i / block_cache_i / lm_head / greedy_head
// decode 修正（vs Eureka 的 off-by-one）：pos = token_length、KV 写回 slot token_length
class Qwen3Bmodel {
public:
    Qwen3Bmodel() = default;
    ~Qwen3Bmodel();

    bool init(bm_handle_t handle, const std::string& bmodel_path);
    void deinit();

    // prefill：embeds [1, S, H]（prefix + audio + suffix 拼好，尾部补零）→ 首 token
    int forward_first(const std::vector<float>& embeds_f32, int tlen);
    // decode：上一步 token → 下一步 token
    int forward_next();
    void clear_kv();

    int token_length  = 0;
    int SEQLEN        = 0;
    int MAX_INPUT_LEN = 0;
    int NUM_LAYERS    = 0;

private:
    void init_by_names();

    bm_handle_t handle_ = nullptr;
    void* p_bmrt_ = nullptr;

    const bm_net_info_t* net_embed_cache_ = nullptr;
    const bm_net_info_t* net_lm_          = nullptr;
    const bm_net_info_t* net_greedy_head_ = nullptr;
    std::vector<const bm_net_info_t*> net_blocks_;
    std::vector<const bm_net_info_t*> net_blocks_cache_;

    // KV cache 常驻 device：[1,SEQ,N_KV,HEAD] f32/层
    std::vector<bm_device_mem_t> past_key_dev_;
    std::vector<bm_device_mem_t> past_value_dev_;
    size_t kv_layer_bytes_ = 0;
    size_t kv_token_bytes_ = 0;
    int kv_per_token_ = 0;
    int hidden_size_  = 0;

    // decode 复用 IO device buffer
    bm_device_mem_t dec_hidden_;
    bm_device_mem_t dec_pos_;
    bm_device_mem_t dec_mask_;
    bm_device_mem_t dec_newk_;
    bm_device_mem_t dec_newv_;
    bm_device_mem_t dec_emb_out_;
    bool dec_io_ready_ = false;

    // prefill 复用 IO device buffer（ping-pong hidden + pos/mask 每层复用 + KV 免 host 中转）
    bm_device_mem_t pre_hidden_a_;
    bm_device_mem_t pre_hidden_b_;
    bm_device_mem_t pre_pos_;
    bm_device_mem_t pre_mask_;
    bm_device_mem_t pre_kv_;
    bm_device_mem_t pre_v_;
    bool pre_io_ready_ = false;

    float mask_value_f32_ = -1e9f;
    int cur_token_ = 0;
    bool inited_ = false;
};

// ── AsrPipeline ───────────────────────────────────────────────────────────────
// 顶层推理类：mel → encoder → 拼 embeds → LLM 生成 → 文本
class AsrPipeline {
public:
    bool init(const std::string& encoder_path, const std::string& qwen3_path,
              const std::string& model_dir, int device_id = 0);
    void deinit();

    // 单音频转写：返回 "language <NAME><asr_text>..." 原始文本
    // 失败返回空字符串
    std::string transcribe(const std::string& wav_path, int max_new_tokens = 256);

    // ── 流式推理（对齐官方：1s chunk + 5s 重编码窗口）────────────────────────
    bool init_stream(const std::string& encoder_w500_path);  // 加载窗口版 encoder（500 mel 帧）
    void stream_push(const float* samples, int n);           // 推入音频（16k mono，建议按 1s=16000 对齐）
    // 增量处理：返回中间转写（可修订）；每次 push 后调用
    std::string stream_update(int max_new_tokens = 32);
    // 定稿：完整转写
    std::string stream_finish(int max_new_tokens = 256);

private:
    bm_handle_t bm_handle_ = nullptr;    // LLM bmrt 的 handle
    bm_handle_t bm_handle_enc_ = nullptr; // encoder bmrt 独立 handle（实测共享 handle 时 encoder 输出 NaN）
    EncoderBmodel enc_;
    EncoderBmodel enc_w500_;   // 流式窗口 encoder（500 mel 帧 = 5s）
    Qwen3Bmodel  qwen3_;
    QwenMel mel_;
    Qwen3Tokenizer tok_;
    AsrConfig cfg_;

    // 流式状态
    bool stream_inited_ = false;
    std::vector<float> stream_samples_;      // 累积音频（16k mono）
    std::vector<float> stream_mel_log_;      // 累积 log10-mel（未裁剪，[T,128] t-major，增量追加）
    std::vector<float> stream_mel_;          // 累积 mel 缓存（[T, 128] t-major，重裁剪后）
    int stream_mel_frames_ = 0;              // 已计算 mel 帧数
    std::vector<float> stream_audio_tokens_; // 累积 audio embeds [N, 1024]
    int stream_audio_count_ = 0;             // 累积 audio token 数
    // 基于累积 audio tokens 生成转写（流式共用）
    std::string generate_from_audio(int max_new_tokens);
    // 用 mel 缓存重编码最近 win 帧窗口（win ≤ 500，起点对齐 100），更新累积 audio tokens
    int reencode_window(int win);

    // 计算 audio token 数（与 processor._get_audio_token_length 一致：3 次 stride2 conv）
    static int mel_frames_to_tokens(int t_real_mel);

    std::vector<float> prefix_embeds_;   // [plen, 1024]
    std::vector<float> suffix_embeds_;   // [slen, 1024]
    int plen_ = 0, slen_ = 0;
    bool loaded_ = false;

    // 构造 prefill 输入 embeds（prefix + audio + suffix，[1,S,H] 尾部补零）
    bool build_inputs_embeds(const std::vector<float>& audio_embeds, int alen,
                             std::vector<float>& embeds, int& tlen) const;
};

}  // namespace asr
