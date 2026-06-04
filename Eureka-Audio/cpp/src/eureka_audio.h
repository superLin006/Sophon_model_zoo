#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "bmruntime_interface.h"

namespace eureka {

// ── 配置 ──────────────────────────────────────────────────────────────────────
struct EurekaConfig {
    // Whisper encoder
    int n_mel         = 128;
    int mel_chunk_len = 3000;   // 每块 30s
    int enc_out_t     = 1500;   // Whisper encoder 输出时间步/块
    int enc_d_model   = 1280;

    // MoE downsample（已融合在 whisper encoder bmodel 里）
    int downsample    = 4;
    int tokens_per_chunk = 375;  // enc_out_t / downsample

    // Qwen3 backbone
    int hidden_size   = 2048;
    int num_layers    = 28;
    int num_kv_heads  = 8;
    int head_dim      = 128;
    int vocab_size    = 151936;
    int seq_length    = 512;   // bmodel 编译时的 SEQ_LENGTH
};

// ── WhisperEncoderBmodel ──────────────────────────────────────────────────────
// 封装 whisper_encoder_b1_bf16.bmodel 推理
// 输入：mel float32 [B, 128, 3000]
// 输出：audio_embeds float32 [B*375, 2048]
class WhisperEncoderBmodel {
public:
    WhisperEncoderBmodel() = default;
    ~WhisperEncoderBmodel();

    // 独立 bmrt 实例，避免两个大 bmodel 共享 runtime 时的内存冲突
    bool init(bm_handle_t handle, const std::string& bmodel_path);
    void release();
    // mel: row-major float32, shape [n_chunks, 128, 3000]
    // out: row-major float32, shape [n_chunks * 375, 2048]
    bool run(const float* mel, int n_chunks, std::vector<float>& out);

private:
    bm_handle_t   handle_  = nullptr;
    void*         p_bmrt_  = nullptr;
    const bm_net_info_t* net_enc_     = nullptr;
    const bm_net_info_t* net_adaptor_ = nullptr;
    bool          inited_  = false;
};

// ── Qwen3EmbedsBmodel ─────────────────────────────────────────────────────────
// 封装 qwen3_1.7b_embeds_*.bmodel 推理（inputs_embeds 版本）
// 网络结构：embedding_embeds / embedding_cache / block_i / block_cache_i / lm_head / greedy_head
class Qwen3EmbedsBmodel {
public:
    Qwen3EmbedsBmodel() = default;
    ~Qwen3EmbedsBmodel();

    // 独立 bmrt 实例
    bool init(bm_handle_t handle, const std::string& bmodel_path);
    void deinit();

    // 第一次前向（prefill）：
    //   inputs_embeds: [1, MAX_INPUT_LEN, 2048] float32（含音频 embed 注入，尾部补零）
    //   token_len: 实际有效 token 数
    //   返回第一个生成 token
    int forward_first(const std::vector<float>& inputs_embeds_f32, int token_len);

    // 后续解码步：单 token → 下一个 token
    int forward_next();

    void clear_kv();

    int token_length  = 0;
    int SEQLEN        = 0;
    int MAX_INPUT_LEN = 0;
    int NUM_LAYERS    = 0;

private:
    void init_by_names();

    bm_handle_t  handle_  = nullptr;
    void*        p_bmrt_  = nullptr;

    const bm_net_info_t* net_embed_cache_  = nullptr;  // decode: 单 token 查表
    const bm_net_info_t* net_lm_           = nullptr;
    const bm_net_info_t* net_greedy_head_  = nullptr;
    std::vector<const bm_net_info_t*> net_blocks_;
    std::vector<const bm_net_info_t*> net_blocks_cache_;

    // KV cache 常驻 device（独立分配，非 net 内置 mem）：
    // 每层 [1,N_KV,SEQLEN,HEAD_DIM]，decode 时原地偏移写新 token KV，避免每步传整块 KV
    std::vector<bm_device_mem_t> past_key_dev_;    // [NUM_LAYERS] 各 SEQLEN*kv_per_token*4 字节
    std::vector<bm_device_mem_t> past_value_dev_;
    size_t kv_layer_bytes_ = 0;   // 一层完整 KV 字节数 = SEQLEN*kv_per_token*4
    size_t kv_token_bytes_ = 0;   // 一个 token 一层 KV 字节数 = kv_per_token*4（decode 偏移步长）
    int  kv_per_token_ = 0;       // N_KV_HEADS*HEAD_DIM = 1024
    int  hidden_size_  = 0;       // 2048

    // decode 复用的 IO device buffer（避免每步 224 次 malloc/free）
    bm_device_mem_t dec_hidden_;   // [1,1,H] block_cache 共用的 hidden in/out
    bm_device_mem_t dec_pos_;      // [1,1] position_id
    bm_device_mem_t dec_mask_;     // [1,1,1,SEQ+1] attention_mask
    bm_device_mem_t dec_newk_;     // [1,8,1,128] 新 KV 临时
    bm_device_mem_t dec_newv_;
    bm_device_mem_t dec_emb_out_;  // embedding_cache 输出 [1,1,H]
    bool dec_io_ready_ = false;

    float mask_value_f32_ = -1e9f;
    std::vector<int> visited_tokens_;
    int  cur_token_ = 0;     // 当前解码 token（prefill/decode 间传递）
    bool inited_ = false;
};

// ── EurekaAudioPipeline ───────────────────────────────────────────────────────
// 顶层推理类，对外接口：
//   init(whisper_bmodel, qwen3_bmodel)
//   generate(mel_data, n_chunks, token_ids_prefix, text_token_embeds) → tokens
class EurekaAudioPipeline {
public:
    EurekaAudioPipeline()  = default;
    ~EurekaAudioPipeline() = default;

    bool init(const std::string& whisper_bmodel_path,
              const std::string& qwen3_bmodel_path,
              int device_id = 0);
    void deinit();

    // 完整推理：
    //   mel:             float32 [n_chunks, 128, 3000]
    //   n_chunks:        音频块数
    //   prefix_embeds:   系统提示词 + 文本部分的 embed（已通过 embed_tokens 转好）
    //                    shape [prefix_len, 2048] float32
    //   suffix_embeds:   音频后的文本 token embed（通常空）
    //   max_new_tokens:  最大生成长度
    //   返回生成的 token id 列表（不含 prefix）
    std::vector<int> generate(
        const float*  mel,
        int           n_chunks,
        const float*  prefix_embeds,   // [prefix_len, 2048]
        int           prefix_len,
        const float*  suffix_embeds,   // [suffix_len, 2048]，可为 nullptr
        int           suffix_len,
        int           max_new_tokens = 256,
        int           eos_token_id   = 151645,
        int           real_audio_tokens = -1);  // 实际音频 token 数，-1=用全部

    // 辅助：将 float32 embed 转 bf16
    static std::vector<uint16_t> fp32_to_bf16_vec(const float* data, int n);
    // 辅助：从 bf16 转 fp32
    static float bf16_to_fp32(uint16_t bf16);

    EurekaConfig config;

    // 上一次 generate 的分段耗时（秒）和 token 数，供性能统计
    double last_whisper_s = 0;   // whisper encoder + audio_adaptor
    double last_prefill_s = 0;   // qwen3 prefill（全 28 层 + lm_head）
    double last_decode_s  = 0;   // qwen3 decode 循环
    int    last_decode_tokens = 0;
    int    last_prefill_len   = 0;   // prefill 的 token 数 (tlen)

private:
    bm_handle_t           bm_handle_ = nullptr;
    WhisperEncoderBmodel  whisper_enc_;
    Qwen3EmbedsBmodel     qwen3_;
    std::string           whisper_path_;   // 两阶段加载用
    std::string           qwen3_path_;

    // 将音频 embed 注入 prefix/suffix embed，拼成完整 inputs_embeds
    // 返回拼接后的 [MAX_INPUT_LEN, 2048] float32 tensor（左对齐，右侧补零）
    std::vector<float> build_inputs_embeds(
        const float* prefix_embeds, int prefix_len,
        const float* audio_embeds,  int n_audio_tokens,
        const float* suffix_embeds, int suffix_len);
};

}  // namespace eureka
