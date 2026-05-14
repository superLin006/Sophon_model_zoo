#pragma once
#include <string>
#include <vector>
#include <cstdint>

// onnxruntime C API
#include "onnxruntime_c_api.h"
// bmruntime
#include "bmruntime_interface.h"
#include "bmlib_runtime.h"

namespace vits_tts {

// Fixed model constants
static const int  L_MAX         = 128;    // max padded sequence length
static const int  T_MEL_FIXED   = 256;    // bmodel fixed T_mel dimension
static const int  Z_DIM         = 192;    // z_hat channels
static const int  G_DIM         = 256;    // g_emb channels
static const int  UPSAMPLE      = 512;    // samples per mel frame
static const int  SAMPLE_RATE   = 44100;

struct TTSResult {
    std::vector<float> audio;   // valid audio samples
    int    n_samples;
    double part1_ms;            // onnx inference time
    double part2_ms;            // bmodel inference time
    double total_ms;
    double rtf;                 // real-time factor = total_ms / audio_duration_ms
};

class TTSInference {
public:
    TTSInference();
    ~TTSInference();

    // model_dir: directory containing model_part1.onnx and bmodel file
    // precision: "F32" or "F16"
    int init(const char* model_dir, const char* precision);
    void release();

    // tokens/tones: raw int64 arrays (with blank, no padding needed here)
    // seq_len: actual sequence length (without padding)
    TTSResult run(const int64_t* tokens, const int64_t* tones, int seq_len);

private:
    bool initialized_ = false;

    // onnxruntime
    const OrtApi*    ort_api_    = nullptr;
    OrtEnv*          ort_env_    = nullptr;
    OrtSession*      ort_session_= nullptr;
    OrtSessionOptions* ort_opts_ = nullptr;
    OrtMemoryInfo*   ort_mem_    = nullptr;

    // bmruntime
    bm_handle_t      bm_handle_  = nullptr;
    void*            runtime_    = nullptr;
    const bm_net_info_t* net_info_ = nullptr;
};

}  // namespace vits_tts
