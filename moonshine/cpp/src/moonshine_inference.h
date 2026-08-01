#pragma once

#include <string>
#include <vector>

#include "bmruntime_interface.h"

// Moonshine streaming-small BM1684X bmruntime 推理
// -------------------------------------------------
// 固定 10s 输入: WAV -> (补零 160000) -> 分帧 CMVN + asinh -> x_frames
// encoder [1,2000,80] -> encoder_out [1,500,620]
// decoder 23 入 21 出, 逐步贪心自回归(max 128 步), KV cache [10][128][512]
// token 解码: tokens.txt 按 id 直接拼接(▁ -> 空格), 跳过特殊 token
struct MoonshineStats {
    double feat_ms = 0;    // 特征提取耗时
    double infer_ms = 0;   // TPU 推理耗时(encoder + 全部 decoder 步, 含数据搬运)
    double audio_ms = 0;   // 实际音频时长
    int    steps = 0;      // decoder 解码步数(不含 eos 后的停止)
    double rtf = 0;        // (feat + infer) / audio
};

class MoonshineInference {
public:
    MoonshineInference();
    ~MoonshineInference();

    // model_dir: 目录内含 moonshine_encoder_<prec>.bmodel、
    //            moonshine_decoder_<prec>.bmodel、tokens.txt
    // precision: "F32" 或 "F16"
    int init(const char* model_dir, const char* precision);

    // 对 wav 做识别, 返回转写文本
    std::string run(const char* wav_path, MoonshineStats* stats = nullptr);

    // 调试: 设置后保存 x_frames.bin / encoder_out.bin / logits_%02d.bin /
    // tokens.txt 到该目录(与 /tmp/bmverify 的 compare.py 配套)
    void set_debug_dir(const char* dir) { debug_dir_ = dir ? dir : ""; }

    void release();

private:
    bool load_vocab(const std::string& path);
    std::string decode_tokens(const std::vector<int>& ids);
    bool run_encoder(const std::vector<float>& x_frames, std::vector<float>& enc_out);
    bool run_decoder(const std::vector<float>& enc_out, std::vector<int>& ids);

    bm_handle_t bm_handle_ = nullptr;
    void* enc_rt_ = nullptr;
    void* dec_rt_ = nullptr;
    const bm_net_info_t* enc_info_ = nullptr;
    const bm_net_info_t* dec_info_ = nullptr;

    std::vector<std::string> vocab_;   // id -> token 文本
    std::string debug_dir_;
    bool initialized_ = false;
};
