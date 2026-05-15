#pragma once
#include <string>
#include <vector>
#include <cstdint>

#include "bmruntime_interface.h"
#include "bmlib_runtime.h"

namespace vits_tts {

static const int  L_MAX        = 128;   // max padded sequence length
static const int  T_MEL_FIXED  = 256;   // bmodel fixed T_mel dimension (~3s @ 44100Hz)
static const int  Z_DIM        = 192;   // z_p channels
static const int  UPSAMPLE     = 512;   // samples per mel frame
static const int  SAMPLE_RATE  = 44100;

struct TTSResult {
    std::vector<float> audio;
    int    n_samples;
    double part_a_ms;   // TPU: enc_p + dp
    double part_b_ms;   // CPU: MAS
    double part_c_ms;   // TPU: flow + decoder
    double total_ms;
    double rtf;
};

class TTSInference {
public:
    TTSInference();
    ~TTSInference();

    // model_dir: directory with vits_part_a_*.bmodel and vits_part_c_*.bmodel
    // precision: "F32" or "F16"
    int init(const char* model_dir, const char* precision);
    void release();

    // tokens/tones: raw int64 arrays (with blank, no padding)
    // seq_len: actual sequence length
    TTSResult run(const int64_t* tokens, const int64_t* tones, int seq_len);

private:
    bool initialized_ = false;

    bm_handle_t bm_handle_ = nullptr;

    // Part A: enc_p + dp
    void* runtime_a_         = nullptr;
    const bm_net_info_t* net_a_ = nullptr;

    // Part C: flow + decoder
    void* runtime_c_         = nullptr;
    const bm_net_info_t* net_c_ = nullptr;
};

}  // namespace vits_tts
