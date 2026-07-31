#include "audio_frontend.h"
#include <kaldi-native-fbank/csrc/feature-fbank.h>
#include <kaldi-native-fbank/csrc/online-feature.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace zipformer {
static knf::FbankOptions Options(const FbankConfig& c) {
  knf::FbankOptions o;
  o.frame_opts.samp_freq = c.sample_rate;
  o.frame_opts.frame_length_ms = c.frame_length_ms;
  o.frame_opts.frame_shift_ms = c.frame_shift_ms;
  o.frame_opts.dither = c.dither;
  o.frame_opts.preemph_coeff = c.preemphasis;
  o.frame_opts.window_type = c.window;
  o.frame_opts.snip_edges = c.snip_edges;
  o.frame_opts.remove_dc_offset = c.remove_dc;
  o.mel_opts.num_bins = c.mel_bins;
  o.mel_opts.low_freq = c.low_freq;
  o.mel_opts.high_freq = c.high_freq;
  o.use_energy = c.use_energy;
  o.use_log_fbank = c.use_log;
  o.use_power = c.use_power;
  return o;
}
class AudioFrontend::Impl {
 public:
  explicit Impl(const FbankConfig& c) : opts(Options(c)), fbank(new knf::OnlineFbank(opts)) {}
  knf::FbankOptions opts;
  mutable std::unique_ptr<knf::OnlineFbank> fbank;
};
AudioFrontend::AudioFrontend(const FbankConfig& c) : config_(c), impl_(new Impl(c)) {}
AudioFrontend::~AudioFrontend() = default;
std::vector<float> AudioFrontend::Compute(const std::vector<float>& pcm) const {
  impl_->fbank.reset(new knf::OnlineFbank(impl_->opts));
  if (!pcm.empty()) impl_->fbank->AcceptWaveform(config_.sample_rate, pcm.data(), pcm.size());
  impl_->fbank->InputFinished();
  last_num_frames_ = impl_->fbank->NumFramesReady();
  std::vector<float> out(static_cast<size_t>(last_num_frames_) * config_.mel_bins);
  for (int i = 0; i < last_num_frames_; ++i) {
    const float* p = impl_->fbank->GetFrame(i);
    std::copy(p, p + config_.mel_bins, out.begin() + static_cast<size_t>(i) * config_.mel_bins);
  }
  return out;
}
std::vector<float> ComputeStrictFbank(const std::vector<float>& pcm, int* frames,
                                      const FbankConfig& c) {
  std::vector<float> padded = pcm;
  padded.insert(padded.end(), static_cast<size_t>(std::llround(c.sample_rate * 1.03)), 0.0f);
  AudioFrontend frontend(c);
  std::vector<float> out = frontend.Compute(padded);
  if (frames) *frames = frontend.last_num_frames();
  return out;
}
}  // namespace zipformer
