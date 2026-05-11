/* Audio Frontend for SenseVoice
 *
 * Computes Fbank features and applies LFR (Low Frame Rate) transformation.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "sensevoice_config.h"

namespace sensevoice {

class AudioFrontend {
public:
    explicit AudioFrontend(const AudioConfig& config);
    ~AudioFrontend();

    // Compute fbank features from audio samples
    // Input: audio samples (float, normalized to [-1, 1])
    // Output: fbank features [num_frames, num_mel_bins]
    std::vector<float> ComputeFbank(const std::vector<float>& samples);

    // Apply LFR transformation
    // Input: fbank features [num_frames, 80]
    // Output: LFR features [out_frames, 560]
    static std::vector<float> ApplyLFR(const std::vector<float>& fbank,
                                       int32_t num_frames,
                                       int32_t feat_dim = 80,
                                       int32_t window_size = 7,
                                       int32_t window_shift = 6);

    // Full pipeline: audio -> LFR features
    std::vector<float> Process(const std::vector<float>& samples,
                               int32_t* out_num_frames = nullptr);

    // Get number of mel bins
    int32_t NumMelBins() const { return config_.num_mel_bins; }

    // Get sample rate
    int32_t SampleRate() const { return config_.sample_rate; }

private:
    AudioConfig config_;

    // Internal state for fbank computation
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Utility: Load WAV file and return samples
bool LoadWavFile(const std::string& filename,
                 std::vector<float>* samples,
                 int32_t* sample_rate);

// Utility: Load raw PCM file (16-bit, mono)
bool LoadPcmFile(const std::string& filename,
                 std::vector<float>* samples,
                 int32_t expected_sample_rate = 16000);

}  // namespace sensevoice
