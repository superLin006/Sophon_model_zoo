#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <mutex>

#include "bmruntime_interface.h"
#include "bmlib_runtime.h"

namespace vits_tts {

static const int  L_MAX        = 256;   // max padded sequence length
static const int  T_MEL_FIXED  = 1024;   // bmodel fixed T_mel dimension (~11.9s @ 44100Hz)
static const int  Z_DIM        = 192;   // z_p channels
static const int  UPSAMPLE     = 512;   // samples per mel frame
static const int  SAMPLE_RATE  = 44100;
static const int  STREAM_MEL_WINDOW  = 128;
static const int  STREAM_MEL_OVERLAP = 32;
static const int  STREAM_MEL_CONTEXT = 16;

struct TTSResult {
    std::vector<float> audio;
    int    n_samples = 0;
    int    streamed_chunks = 0;
    int    mel_frames = 0;
    bool   truncated = false;
    double part_a_ms = 0.0;   // TPU: enc_p + dp
    double part_b_ms = 0.0;   // CPU: MAS
    double part_c_ms = 0.0;   // TPU: flow + decoder
    double total_ms = 0.0;
    double rtf = 0.0;
    bool   cancelled = false;
};

struct TTSInputSegment {
    std::vector<int64_t> tokens;
    std::vector<int64_t> tones;
    std::string label;
};

using PcmChunkCallback = std::function<bool(
    const float* samples, int n_samples, int segment_index,
    int segment_count, int64_t sample_offset)>;

struct TTSStreamOptions {
    PcmChunkCallback on_audio;
    bool strict_mel_limit = true;
};

struct TTSStreamResult {
    std::vector<float> audio;
    int completed_segments = 0;
    int failed_segment = -1;
    int64_t n_samples = 0;
    double first_chunk_ms = 0.0;
    double total_ms = 0.0;
    bool success = false;
    bool cancelled = false;
    std::string error;
};

struct TTSWindowStreamOptions {
    PcmChunkCallback on_audio;
    bool strict_mel_limit = true;
};

struct TTSWindowStreamResult {
    std::vector<float> audio;
    int completed_chunks = 0;
    int64_t n_samples = 0;
    double first_chunk_ms = 0.0;
    double total_ms = 0.0;
    bool success = false;
    bool cancelled = false;
    std::string error;
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

    // Runs complete independent segments and emits each non-empty PCM segment synchronously.
    TTSStreamResult run_segments(const std::vector<TTSInputSegment>& segments,
                                 const TTSStreamOptions& options = TTSStreamOptions());

    TTSWindowStreamResult run_windowed(const int64_t* tokens,
                                       const int64_t* tones,
                                       int seq_len,
                                       const TTSWindowStreamOptions& options =
                                           TTSWindowStreamOptions());

private:
    TTSResult run_impl(const int64_t* tokens, const int64_t* tones, int seq_len,
                       const TTSWindowStreamOptions* window_options = nullptr);

    bool initialized_ = false;
    std::mutex run_mutex_;

    bm_handle_t bm_handle_ = nullptr;

    // single bmruntime holding all three bmodels
    void* runtime_ = nullptr;
    const bm_net_info_t* net_a_  = nullptr;  // enc_p + dp
    const bm_net_info_t* net_c1_ = nullptr;  // flow
    const bm_net_info_t* net_c2_ = nullptr;  // decoder
    const bm_net_info_t* net_c2_stream_ = nullptr;  // window decoder
};

}  // namespace vits_tts
