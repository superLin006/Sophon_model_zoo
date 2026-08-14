#pragma once
// Qwen3-TTS BM1684X 推理引擎（纯 bmrt）
// 三段：talker（28层自回归）+ code_predictor（嵌套15步）+ codec decoder。

#include <string>
#include <vector>
#include <cstdint>
#include <random>

#include "bmruntime_interface.h"
#include "tokenizer.h"

namespace qwen3tts {

struct TtsConfig {
    // 常量（来自 config.json）
    static constexpr int NUM_LAYERS = 28;
    static constexpr int HIDDEN = 1024;
    static constexpr int KV_HEADS = 8;
    static constexpr int HEAD_DIM = 128;
    static constexpr int SEQLEN = 192;      // talker prefill / KV 最大长度（192 帧=15.36s，更长效）
    static constexpr int VOCAB = 3072;       // talker codec 词表
    static constexpr int CP_LAYERS = 5;
    static constexpr int CP_VOCAB = 2048;

    // 文本 token
    static constexpr int IM_START = 151644;
    static constexpr int IM_END = 151645;
    static constexpr int ASSISTANT = 77091;
    static constexpr int TTS_PAD = 151671;
    static constexpr int TTS_BOS = 151672;
    static constexpr int TTS_EOS = 151673;

    // codec 控制 token
    static constexpr int CODEC_PAD = 2148;
    static constexpr int CODEC_BOS = 2149;
    static constexpr int CODEC_EOS = 2150;
    static constexpr int CODEC_THINK = 2154;
    static constexpr int CODEC_NOTHINK = 2155;
    static constexpr int CODEC_THINK_BOS = 2156;
    static constexpr int CODEC_THINK_EOS = 2157;
};

class TtsEngine {
public:
    TtsEngine() = default;
    ~TtsEngine();
    TtsEngine(const TtsEngine&) = delete;
    TtsEngine& operator=(const TtsEngine&) = delete;

    bool init(const std::string& talker_bmodel, const std::string& cp_bmodel,
              const std::string& codec_bmodel, const std::string& model_dir, int device = 0,
              const std::string& cp_cache_bmodel = "");
    void deinit();

    // 生成语音：返回 24kHz PCM（float32，范围约 [-1,1]），sample_rate=24000
    // do_sample=false 为 greedy（与 baseline 对齐）
    bool generate(const std::string& text, const std::string& speaker,
                  const std::string& language, std::vector<float>& pcm,
                  int& sample_rate, int max_new_tokens = 512, bool do_sample = false,
                  int seed = 42);

    // debug：跳过 talker，用固定 last_hidden 直接跑 CP 链路（验证 talker 是否污染 CP）
    bool test_cp_only();

private:
    bool build_prefill_embeds(const std::vector<int>& input_ids,
                              std::vector<float>& embeds, int& prefill_len);
    bool talker_prefill(const std::vector<float>& embeds, int prefill_len,
                        std::vector<float>& first_logits);
    bool talker_decode_step(int code0, const std::vector<int>& codes16,
                            int pos, std::vector<float>& next_logits);
    bool code_predictor_generate(int code0, const std::vector<float>& past_hidden,
                                 std::vector<int>& codes16);
    bool codec_decode(const std::vector<int>& codes, std::vector<float>& pcm, int& sr);

    int sample_logits(const std::vector<float>& logits, int vocab, bool do_sample,
                      int top_k, float top_p, float temperature);
    int argmax(const std::vector<float>& v) const;

    // 工具
    bool run_net(void* rt, const bm_net_info_t* net, const std::vector<const void*>& in_hosts,
                 const std::vector<void*>& out_hosts);
    std::vector<float> codec_embed(const std::vector<int>& ids);  // embedding_code lookup → ids*1024 f32
    std::vector<float> text_embed(const std::vector<int>& ids);  // embedding_text lookup → ids*1024 f32

    int speaker_id(const std::string& speaker) const;
    int language_id(const std::string& language) const;

    bm_handle_t handle_ = nullptr;
    void* talker_rt_ = nullptr;
    void* cp_rt_ = nullptr;
    void* cp_cache_rt_ = nullptr;  // 独立 cache bmodel（实验：bf16/f16 cache 不重新 combine）
    void* codec_rt_ = nullptr;

    const bm_net_info_t* net_embed_text_ = nullptr;
    const bm_net_info_t* net_embed_code_ = nullptr;
    const bm_net_info_t* net_codec_head_ = nullptr;
    std::vector<const bm_net_info_t*> talker_blocks_;
    std::vector<const bm_net_info_t*> talker_blocks_cache_;

    std::vector<const bm_net_info_t*> cp_blocks_;
    std::vector<const bm_net_info_t*> cp_blocks_cache_;
    std::vector<const bm_net_info_t*> cp_lm_head_;
    std::vector<const bm_net_info_t*> cp_embedding_;
    const bm_net_info_t* cp_lm_head_all_ = nullptr;    // 合并 15 head（index 输入，可为 null 回退单网络）
    const bm_net_info_t* cp_embedding_all_ = nullptr;  // 合并 15 embedding（index 输入）
    const bm_net_info_t* net_codec_ = nullptr;

    // KV cache 设备常驻（talker [8,SEQLEN,128]，CP [8,15,128]）。
    // decode 输入用 bmrt_tensor_with_device 零拷贝直连设备 KV，新 KV 用 d2d 写槽。
    // 性能：消除每帧 ~28MB 的 KV s2d + host 镜像 memcpy（v1 同款方案）。
    std::vector<bm_device_mem_t> talker_k_dev_, talker_v_dev_;
    std::vector<bm_device_mem_t> cp_k_dev_, cp_v_dev_;
    // 上一步 CP 生成的 embedding（[0]=code0，[1..15]=code1..code15），供 talker_decode_step 直接求和
    std::vector<std::vector<float>> last_cp_embs_;
    std::vector<float> tts_pad_emb_;  // text_embed(TTS_PAD) 固定结果缓存

    TextTokenizer tokenizer_;
    bool inited_ = false;
    int seed_ = 42;
    std::mt19937 rng_;                    // 采样 RNG：generate() 内按 seed 重新播种（可复现）
    std::string current_speaker_ = "Vivian";
    std::string current_lang_ = "Chinese";
    std::vector<float> prefill_last_hidden_;   // prefill 最后一 token 的 hidden（供 code_predictor）
};

}  // namespace qwen3tts
