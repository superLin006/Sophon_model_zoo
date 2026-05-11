/* SenseVoice Configuration
 *
 * Configuration structures for SenseVoice MTK inference.
 */

#pragma once

#include <string>
#include <cstdint>

namespace sensevoice {

// Model configuration
struct ModelConfig {
    std::string model_path;           // Path to DLA file
    std::string tokens_path;          // Path to tokens.txt file

    // Model parameters (fixed for SenseVoice Small)
    int32_t vocab_size = 25055;
    int32_t input_feat_dim = 560;     // 80 * 7 (after LFR)
    int32_t encoder_output_dim = 512;

    // LFR parameters
    int32_t lfr_window_size = 7;      // lfr_m
    int32_t lfr_window_shift = 6;     // lfr_n

    // Feature parameters
    int32_t sample_rate = 16000;
    int32_t num_mel_bins = 80;
    int32_t frame_shift_ms = 10;
    int32_t frame_length_ms = 25;

    // Special token IDs
    int32_t blank_id = 0;
    int32_t sos_id = 1;
    int32_t eos_id = 2;
};

// Language configuration
enum class Language {
    Auto = 0,
    Chinese = 3,
    English = 4,
    Cantonese = 7,
    Japanese = 11,
    Korean = 12,
    NoSpeech = 13
};

// Text normalization configuration
enum class TextNorm {
    WithITN = 14,    // With inverse text normalization
    WithoutITN = 15  // Without inverse text normalization
};

// Inference configuration
struct InferenceConfig {
    Language language = Language::Auto;
    TextNorm text_norm = TextNorm::WithITN;  // Default: with punctuation
    bool use_greedy_search = true;  // Currently only greedy search is supported
};

// Audio configuration
struct AudioConfig {
    int32_t sample_rate = 16000;
    int32_t num_mel_bins = 80;
    int32_t frame_shift_ms = 10;
    int32_t frame_length_ms = 25;
    float dither = 0.0f;
    float preemph_coeff = 0.97f;
    std::string window_type = "hamming";
    bool snip_edges = true;
};

// Full configuration
struct SenseVoiceConfig {
    ModelConfig model;
    AudioConfig audio;
    InferenceConfig inference;
};

// Helper functions
inline int32_t GetLanguageId(Language lang) {
    return static_cast<int32_t>(lang);
}

inline int32_t GetTextNormId(TextNorm norm) {
    return static_cast<int32_t>(norm);
}

// Calculate number of output frames after LFR
inline int32_t CalcLfrOutputFrames(int32_t input_frames, int32_t window_size = 7, int32_t window_shift = 6) {
    if (input_frames < window_size) {
        return 0;
    }
    return (input_frames - window_size) / window_shift + 1;
}

// Calculate number of fbank frames from audio samples
inline int32_t CalcNumFrames(int64_t num_samples, int32_t sample_rate = 16000,
                             int32_t frame_shift_ms = 10, int32_t frame_length_ms = 25) {
    int32_t frame_shift = sample_rate * frame_shift_ms / 1000;
    int32_t frame_length = sample_rate * frame_length_ms / 1000;

    if (num_samples < frame_length) {
        return 0;
    }
    return (num_samples - frame_length) / frame_shift + 1;
}

}  // namespace sensevoice
