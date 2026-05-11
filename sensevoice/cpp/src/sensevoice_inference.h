#pragma once

#include <string>
#include <vector>

#include "bmruntime_interface.h"
#include "sensevoice_config.h"
#include "audio_frontend.h"
#include "tokenizer.h"

namespace sensevoice {

// Fixed model parameters
#define SV_FIXED_FRAMES  166
#define SV_INPUT_DIM     560
#define SV_VOCAB_SIZE    25055
#define SV_OUTPUT_FRAMES 170  // 166 + 4 prompt tokens

class SenseVoiceInference {
public:
    SenseVoiceInference();
    ~SenseVoiceInference();

    // model_dir: directory containing bmodel and tokens.txt
    // precision: "F32" or "F16"
    int  init(const char* model_dir, const char* precision = "F32");
    RecognitionResult run(const char* audio_file);
    void release();

private:
    bm_handle_t   bm_handle_ = nullptr;
    void*         runtime_   = nullptr;
    const bm_net_info_t* net_info_ = nullptr;

    AudioFrontend* frontend_  = nullptr;
    Tokenizer      tokenizer_;
    bool           initialized_ = false;

    // Pad/truncate LFR features to exactly SV_FIXED_FRAMES
    std::vector<float> pad_or_truncate(const std::vector<float>& lfr,
                                       int32_t num_frames) const;
};

}  // namespace sensevoice
