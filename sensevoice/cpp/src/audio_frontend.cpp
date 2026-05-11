/* Audio Frontend Implementation
 *
 * Uses kaldi-native-fbank for feature extraction.
 */

#include "audio_frontend.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace sensevoice {

// Implementation class using kaldi-native-fbank
class AudioFrontend::Impl {
public:
    explicit Impl(const AudioConfig& config) : config_(config) {
        // Configure fbank options
        knf::FbankOptions opts;
        opts.frame_opts.samp_freq = static_cast<float>(config.sample_rate);
        opts.frame_opts.frame_shift_ms = static_cast<float>(config.frame_shift_ms);
        opts.frame_opts.frame_length_ms = static_cast<float>(config.frame_length_ms);
        opts.frame_opts.dither = config.dither;
        opts.frame_opts.preemph_coeff = config.preemph_coeff;
        opts.frame_opts.window_type = config.window_type;
        opts.frame_opts.snip_edges = config.snip_edges;
        opts.frame_opts.remove_dc_offset = true;

        opts.mel_opts.num_bins = config.num_mel_bins;
        opts.mel_opts.low_freq = 20.0f;
        opts.mel_opts.high_freq = 0.0f;  // Nyquist

        opts.use_energy = false;
        opts.use_log_fbank = true;
        opts.use_power = true;

        fbank_ = std::make_unique<knf::OnlineFbank>(opts);
    }

    std::vector<float> ComputeFbank(const std::vector<float>& samples) {
        // Reset fbank state
        fbank_ = std::make_unique<knf::OnlineFbank>(GetOptions());

        // Accept waveform
        fbank_->AcceptWaveform(static_cast<float>(config_.sample_rate),
                               samples.data(),
                               static_cast<int32_t>(samples.size()));
        fbank_->InputFinished();

        // Get number of frames
        int32_t num_frames = fbank_->NumFramesReady();
        if (num_frames == 0) {
            return {};
        }

        // Extract features
        std::vector<float> features(num_frames * config_.num_mel_bins);
        for (int32_t i = 0; i < num_frames; ++i) {
            const float* frame = fbank_->GetFrame(i);
            std::copy(frame, frame + config_.num_mel_bins,
                      features.begin() + i * config_.num_mel_bins);
        }

        return features;
    }

private:
    knf::FbankOptions GetOptions() const {
        knf::FbankOptions opts;
        opts.frame_opts.samp_freq = static_cast<float>(config_.sample_rate);
        opts.frame_opts.frame_shift_ms = static_cast<float>(config_.frame_shift_ms);
        opts.frame_opts.frame_length_ms = static_cast<float>(config_.frame_length_ms);
        opts.frame_opts.dither = config_.dither;
        opts.frame_opts.preemph_coeff = config_.preemph_coeff;
        opts.frame_opts.window_type = config_.window_type;
        opts.frame_opts.snip_edges = config_.snip_edges;
        opts.frame_opts.remove_dc_offset = true;

        opts.mel_opts.num_bins = config_.num_mel_bins;
        opts.mel_opts.low_freq = 20.0f;
        opts.mel_opts.high_freq = 0.0f;

        opts.use_energy = false;
        opts.use_log_fbank = true;
        opts.use_power = true;

        return opts;
    }

    AudioConfig config_;
    std::unique_ptr<knf::OnlineFbank> fbank_;
};

AudioFrontend::AudioFrontend(const AudioConfig& config)
    : config_(config), impl_(std::make_unique<Impl>(config)) {
}

AudioFrontend::~AudioFrontend() = default;

std::vector<float> AudioFrontend::ComputeFbank(const std::vector<float>& samples) {
    return impl_->ComputeFbank(samples);
}

std::vector<float> AudioFrontend::ApplyLFR(const std::vector<float>& fbank,
                                           int32_t num_frames,
                                           int32_t feat_dim,
                                           int32_t window_size,
                                           int32_t window_shift) {
    if (num_frames < window_size) {
        return {};
    }

    int32_t out_num_frames = (num_frames - window_size) / window_shift + 1;
    int32_t out_feat_dim = feat_dim * window_size;

    std::vector<float> lfr_features(out_num_frames * out_feat_dim);

    const float* p_in = fbank.data();
    float* p_out = lfr_features.data();

    for (int32_t i = 0; i < out_num_frames; ++i) {
        // Copy window_size consecutive frames
        std::copy(p_in, p_in + out_feat_dim, p_out);
        p_out += out_feat_dim;
        p_in += window_shift * feat_dim;
    }

    return lfr_features;
}

std::vector<float> AudioFrontend::Process(const std::vector<float>& samples,
                                          int32_t* out_num_frames) {
    // Compute fbank features
    std::vector<float> fbank = ComputeFbank(samples);
    if (fbank.empty()) {
        if (out_num_frames) *out_num_frames = 0;
        return {};
    }

    int32_t num_fbank_frames = static_cast<int32_t>(fbank.size()) / config_.num_mel_bins;

    // Apply LFR
    std::vector<float> lfr = ApplyLFR(fbank, num_fbank_frames, config_.num_mel_bins);
    if (lfr.empty()) {
        if (out_num_frames) *out_num_frames = 0;
        return {};
    }

    int32_t num_lfr_frames = static_cast<int32_t>(lfr.size()) / (config_.num_mel_bins * 7);

    if (out_num_frames) {
        *out_num_frames = num_lfr_frames;
    }

    return lfr;
}

// WAV file header structure
#pragma pack(push, 1)
struct WavHeader {
    char riff[4];           // "RIFF"
    uint32_t file_size;
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;
    uint16_t audio_format;  // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};
#pragma pack(pop)

bool LoadWavFile(const std::string& filename,
                 std::vector<float>* samples,
                 int32_t* sample_rate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // Verify RIFF/WAVE format
    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        return false;
    }

    // Skip to data chunk
    char chunk_id[4];
    uint32_t chunk_size;

    while (file.read(chunk_id, 4)) {
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::strncmp(chunk_id, "data", 4) == 0) {
            break;
        }

        file.seekg(chunk_size, std::ios::cur);
    }

    if (file.eof()) {
        return false;
    }

    // Read audio data
    int32_t num_samples = chunk_size / (header.bits_per_sample / 8) / header.num_channels;
    samples->resize(num_samples);

    if (header.bits_per_sample == 16) {
        std::vector<int16_t> raw_samples(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(raw_samples.data()),
                  num_samples * header.num_channels * sizeof(int16_t));

        // Convert to float and take first channel
        for (int32_t i = 0; i < num_samples; ++i) {
            (*samples)[i] = raw_samples[i * header.num_channels] / 32768.0f;
        }
    } else if (header.bits_per_sample == 32) {
        std::vector<float> raw_samples(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(raw_samples.data()),
                  num_samples * header.num_channels * sizeof(float));

        // Take first channel
        for (int32_t i = 0; i < num_samples; ++i) {
            (*samples)[i] = raw_samples[i * header.num_channels];
        }
    } else {
        return false;
    }

    *sample_rate = static_cast<int32_t>(header.sample_rate);
    return true;
}

bool LoadPcmFile(const std::string& filename,
                 std::vector<float>* samples,
                 int32_t expected_sample_rate) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    int32_t num_samples = static_cast<int32_t>(size / sizeof(int16_t));
    std::vector<int16_t> raw_samples(num_samples);

    if (!file.read(reinterpret_cast<char*>(raw_samples.data()), size)) {
        return false;
    }

    // Convert to float
    samples->resize(num_samples);
    for (int32_t i = 0; i < num_samples; ++i) {
        (*samples)[i] = raw_samples[i] / 32768.0f;
    }

    return true;
}

}  // namespace sensevoice
