#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

inline int16_t pcm16_from_float(float value) {
    if (!std::isfinite(value)) {
        value = 0.0f;
    }
    value = std::max(-1.0f, std::min(1.0f, value));
    return static_cast<int16_t>(value * 32767.0f);
}

inline bool write_pcm_s16le(FILE* file, const float* data, int n_samples) {
    if (!file || !data || n_samples < 0) {
        return false;
    }
    std::vector<int16_t> pcm(static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        pcm[static_cast<size_t>(i)] = pcm16_from_float(data[i]);
    }
    return std::fwrite(pcm.data(), sizeof(int16_t), pcm.size(), file) == pcm.size();
}

class StreamingWavWriter {
public:
    StreamingWavWriter() = default;
    ~StreamingWavWriter() {
        if (file_) {
            std::fclose(file_);
        }
    }

    StreamingWavWriter(const StreamingWavWriter&) = delete;
    StreamingWavWriter& operator=(const StreamingWavWriter&) = delete;

    bool open(const char* path, int sample_rate = 44100) {
        if (file_ || !path || sample_rate <= 0) {
            return false;
        }
        file_ = std::fopen(path, "wb");
        if (!file_) {
            return false;
        }
        sample_rate_ = sample_rate;
        data_bytes_ = 0;
        if (!write_header(0)) {
            std::fclose(file_);
            file_ = nullptr;
            return false;
        }
        return true;
    }

    bool append(const float* data, int n_samples) {
        if (!file_ || !data || n_samples < 0) {
            return false;
        }
        const uint64_t bytes = static_cast<uint64_t>(n_samples) * sizeof(int16_t);
        if (bytes > std::numeric_limits<uint32_t>::max() - data_bytes_) {
            return false;
        }
        if (!write_pcm_s16le(file_, data, n_samples)) {
            return false;
        }
        data_bytes_ += static_cast<uint32_t>(bytes);
        return true;
    }

    bool finalize() {
        if (!file_) {
            return false;
        }
        bool ok = std::fflush(file_) == 0 &&
                  std::fseek(file_, 0, SEEK_SET) == 0 &&
                  write_header(data_bytes_) &&
                  std::fflush(file_) == 0;
        std::fclose(file_);
        file_ = nullptr;
        return ok;
    }

    bool is_open() const { return file_ != nullptr; }

private:
    bool write_header(uint32_t data_bytes) {
        const uint32_t riff_bytes = 36u + data_bytes;
        const uint32_t fmt_bytes = 16u;
        const uint16_t audio_format = 1u;
        const uint16_t channels = 1u;
        const uint32_t byte_rate = static_cast<uint32_t>(sample_rate_) * 2u;
        const uint16_t block_align = 2u;
        const uint16_t bits_per_sample = 16u;
        return std::fwrite("RIFF", 1, 4, file_) == 4 &&
               std::fwrite(&riff_bytes, sizeof(riff_bytes), 1, file_) == 1 &&
               std::fwrite("WAVEfmt ", 1, 8, file_) == 8 &&
               std::fwrite(&fmt_bytes, sizeof(fmt_bytes), 1, file_) == 1 &&
               std::fwrite(&audio_format, sizeof(audio_format), 1, file_) == 1 &&
               std::fwrite(&channels, sizeof(channels), 1, file_) == 1 &&
               std::fwrite(&sample_rate_, sizeof(sample_rate_), 1, file_) == 1 &&
               std::fwrite(&byte_rate, sizeof(byte_rate), 1, file_) == 1 &&
               std::fwrite(&block_align, sizeof(block_align), 1, file_) == 1 &&
               std::fwrite(&bits_per_sample, sizeof(bits_per_sample), 1, file_) == 1 &&
               std::fwrite("data", 1, 4, file_) == 4 &&
               std::fwrite(&data_bytes, sizeof(data_bytes), 1, file_) == 1;
    }

    FILE* file_ = nullptr;
    int sample_rate_ = 44100;
    uint32_t data_bytes_ = 0;
};

inline bool write_wav(const char* path, const float* data, int n_samples,
                      int sample_rate = 44100) {
    if (n_samples <= 0) {
        return false;
    }
    StreamingWavWriter writer;
    return writer.open(path, sample_rate) && writer.append(data, n_samples) &&
           writer.finalize();
}
