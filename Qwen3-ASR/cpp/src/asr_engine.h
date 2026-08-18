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
// 封装全量 encoder（对齐官方 Python/transformers 语义）：mel [1,128,3000]（30s）→ [390,1024]
// 离线/流式共用同一个网络（流式每次全量重编码，0.08s 固定开销，1s 块下实时）
class EncoderBmodel {
public:
    EncoderBmodel() = default;
    ~EncoderBmodel();
    // 从主 bmrt（单文件已加载）取指定网络，不独立加载 bmodel
    bool init(void* p_bmrt, const char* net_name = "qwen3_asr_encoder");
    void release();
    // mel: [1,128,3000] row-major → out [390,1024] row-major（shape 从网络读，自动适配）
    bool run(const float* mel, std::vector<float>& out);
private:
    bm_handle_t handle_ = nullptr;
    void* p_bmrt_ = nullptr;      // 主 bmrt（共享，不 destroy）
    const bm_net_info_t* net_ = nullptr;
    bool inited_ = false;
};

// ── Qwen3Bmodel ──────────────────────────────────────────────────────────────
// 封装标准 Qwen3 LLM bmodel（w4bf16/w4f16/w8bf16 等激活 dtype）
// 网络：embedding_cache / block_i / block_cache_i / lm_head / greedy_head
// decode 修正（vs Eureka 的 off-by-one）：pos = token_length、KV 写回 slot token_length
class Qwen3Bmodel {
public:
    Qwen3Bmodel() = default;
    ~Qwen3Bmodel();

    bool init(bm_handle_t handle, const std::string& bmodel_path);
    void deinit();

    // prefill：input_ids（含 audio 占位 token）+ audio embeds → 首 token
    // 流程：embedding 网络（token ids → 文本 embeds）→ 替换 audio 段 → block 循环
    int forward_first(const std::vector<int>& input_ids, int tlen,
                      const std::vector<float>& audio_embeds, int audio_start);
    // decode：上一步 token → 下一步 token
    int forward_next();
    void clear_kv();
    void* bmrt() const { return p_bmrt_; }


    int token_length  = 0;
    int SEQLEN        = 0;
    int MAX_INPUT_LEN = 0;
    int NUM_LAYERS    = 0;

private:
    void init_by_names();

    bm_handle_t handle_ = nullptr;
    void* p_bmrt_ = nullptr;

    const bm_net_info_t* net_embed_        = nullptr;  // prefill: token ids → embeds（标准 Qwen3 bmodel）
    const bm_net_info_t* net_embed_cache_ = nullptr;
    const bm_net_info_t* net_lm_          = nullptr;
    const bm_net_info_t* net_greedy_head_ = nullptr;
    std::vector<const bm_net_info_t*> net_blocks_;
    std::vector<const bm_net_info_t*> net_blocks_cache_;

    // KV cache 常驻 device：[1,SEQ,N_KV,HEAD]，dtype 按 block_cache 输入声明
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

    uint16_t mask_neg_inf_ = 0xC61C;   // attention mask -inf 值（按 BF16/F16 输入 dtype）
    int cur_token_ = 0;
    bool inited_ = false;

    // hidden f32 → bmodel activation dtype → lm_head → argmax
    int lm_head_argmax(const float* hidden_f32);
};

// ── AsrPipeline ───────────────────────────────────────────────────────────────
// 顶层推理类：mel → encoder → 拼 embeds → LLM 生成 → 文本
class AsrPipeline {
public:
    // 单文件 bmodel（encoder + LLM 合并）：一个 handle + 一个 bmrt，encoder 复用 LLM 的 bmrt
    bool init(const std::string& bmodel_path, const std::string& model_dir, int device_id = 0);
    void deinit();

    // 单音频转写：返回 "language <NAME><asr_text>..." 原始文本
    // 失败返回空字符串
    std::string transcribe(const std::string& wav_path, int max_new_tokens = 256);

    // 纯文本生成（调试用：绕过 encoder/audio，直接验证 LLM 链路）
    std::string text_generate(const std::vector<int>& input_ids, int max_new_tokens = 32);

    // ── 流式推理（全量重编码：每次 update 全量 mel 过 encoder + 全量 prefill）───
    bool init_stream();   // encoder 就是主 encoder，无额外初始化
    void stream_push(const float* samples, int n);           // 推入音频（16k mono，建议按 1s=16000 对齐）
    // 增量处理：返回中间转写（可修订）；每次 push 后调用
    std::string stream_update(int max_new_tokens = 32);
    // 定稿：完整转写（与 update 同一路径，行为一致）
    std::string stream_finish(int max_new_tokens = 256);

private:
    bm_handle_t bm_handle_ = nullptr;    // 唯一 handle（单文件 bmodel）
    EncoderBmodel enc_;                  // chunk encoder（离线/流式共用）
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
    std::string stream_partial_text_;        // 分段定稿历史（每段一行，支持无限持续流式）
    // 基于累积 audio tokens 生成转写（流式共用）
    std::string generate_from_audio(int max_new_tokens);
    // 计算 audio token 数（与 processor._get_audio_token_length 一致：3 次 stride2 conv）
    static int mel_frames_to_tokens(int t_real_mel);
    // mel（[T,128] t-major）→ pad 到 3000（复制尾帧）→ 全量 encoder → [390,1024] → 截取
    bool encode_full_mel(const std::vector<float>& mel, int T,
                         std::vector<float>& audio_out, int& alen);
    // 在 center 附近找静音窗口中心（mel_log 帧均值 < -7 = 静音），找不到返回 center
    static int find_silence_center(const std::vector<float>& mel_log, int T,
                                   int center, int search = 150);

    std::vector<int>   prefix_ids_;      // prefix token ids（audio_start 前）
    std::vector<int>   suffix_ids_;      // suffix token ids（audio_end 后）
    int plen_ = 0, slen_ = 0;
    bool loaded_ = false;

    // 构造 prefill 输入 input_ids（prefix_ids + audio_pad×alen + suffix_ids，S 尾部补 0）
    // 返回 audio 段起始位置（= plen_）
    int build_input_ids(int alen, std::vector<int>& ids_out, int& tlen) const;
};

}  // namespace asr
