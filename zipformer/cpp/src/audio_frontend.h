#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace zipformer {
struct FbankConfig {
  int sample_rate = 16000;
  int mel_bins = 80;
  float frame_length_ms = 25.0f;
  float frame_shift_ms = 10.0f;
  float dither = 0.0f;
  float preemphasis = 0.97f;
  bool snip_edges = false;
  bool remove_dc = true;
  float low_freq = 20.0f;
  float high_freq = -400.0f;
  bool use_energy = false;
  bool use_log = true;
  bool use_power = true;
  std::string window = "povey";
};

class AudioFrontend {
 public:
  explicit AudioFrontend(const FbankConfig& config = FbankConfig());
  ~AudioFrontend();
  const FbankConfig& config() const { return config_; }
  // Returns row-major [frames, mel_bins]. The frame count is the fbank
  // implementation's NumFramesReady(), not a snip-edges formula.
  std::vector<float> Compute(const std::vector<float>& pcm) const;
  int last_num_frames() const { return last_num_frames_; }
 private:
  FbankConfig config_;
  mutable int last_num_frames_ = 0;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Appends exactly 1.03 seconds at 16 kHz, then computes fbank.
std::vector<float> ComputeStrictFbank(const std::vector<float>& pcm,
                                      int* frames = nullptr,
                                      const FbankConfig& config = FbankConfig());

// Split fbank frames into streaming chunks: each chunk is [segment * n_mels]
// with a stride of `offset` frames. The last chunk is zero-padded if needed.
inline std::vector<std::vector<float>> MakeFeatureChunks(
    const std::vector<float>& features, int num_frames, int n_mels,
    int segment, int offset) {
  std::vector<std::vector<float>> chunks;
  if (num_frames < 1 || n_mels < 1 || segment < 1 || offset < 1) return chunks;
  int count = std::max(1, (std::max(0, num_frames - segment) + offset - 1) / offset + 1);
  for (int p = 0; p < count; ++p) {
    std::vector<float> chunk(static_cast<size_t>(segment) * n_mels, 0.0f);
    for (int i = 0; i < segment && p * offset + i < num_frames; ++i) {
      std::copy(features.begin() + static_cast<size_t>(p * offset + i) * n_mels,
                features.begin() + static_cast<size_t>(p * offset + i + 1) * n_mels,
                chunk.begin() + static_cast<size_t>(i) * n_mels);
    }
    chunks.push_back(std::move(chunk));
  }
  return chunks;
}
}  // namespace zipformer
