#pragma once

#include <string>
#include <vector>

namespace eureka {

// Whisper 128-bin mel 前处理
// 与 Python 侧 audio_to_mel() 完全对齐：
//   STFT: n_fft=400, hop=160, hann window
//   mel filter: 128 bins，从 mel_filters.npz 加载
//   log10 + clip(1e-10) + (max-8) clamp + (x+4)/4 归一化
//   输出切块：每块 3000 帧（≈30s），不足补 log10(1e-10) 归一化值

class WhisperMel {
public:
    WhisperMel() = default;

    // 从 mel_filters.npz 加载 128-bin mel 滤波器
    // npz 格式：key "mel_128"，shape [128, 201]，float32
    bool load_filters(const std::string& npz_path);

    // 从 WAV 文件加载并转 mel
    // 返回 chunks：每个 chunk float32 [128*3000]（行主序）
    // n_chunks = ceil(audio_len / 48000)  (30s chunks)
    std::vector<std::vector<float>> wav_to_mel_chunks(const std::string& wav_path) const;

    // 核心：waveform → mel chunks
    // samples: 16kHz mono float32
    std::vector<std::vector<float>> process(const std::vector<float>& samples) const;

private:
    static constexpr int N_FFT       = 400;
    static constexpr int HOP_LENGTH  = 160;
    static constexpr int N_MELS      = 128;
    static constexpr int CHUNK_FRAMES = 3000;  // mel 帧数/chunk
    static constexpr int SAMPLE_RATE  = 16000;

    std::vector<float> mel_filters_;  // [N_MELS, N_FFT/2+1] = [128, 201]
    bool filters_loaded_ = false;

    // 计算 log-mel spectrogram，shape [N_MELS, T]（T 由输入长度决定）
    std::vector<float> log_mel_spectrogram(const std::vector<float>& audio) const;

    // 将 [N_MELS, T] 切成若干 [N_MELS, CHUNK_FRAMES]，最后一块右侧补 pad_val
    std::vector<std::vector<float>> split_chunks(const std::vector<float>& mel, int T) const;

    // Hann 窗
    static std::vector<float> make_hann_window(int n);

    // 简单 npz 解析（仅支持无压缩 float32 数组）
    bool load_npz_array(const std::string& path, const std::string& key,
                        std::vector<float>& data, std::vector<int>& shape) const;
};

// 工具：加载 WAV 文件，返回 16kHz mono float32
bool load_wav_16k_mono(const std::string& path, std::vector<float>& samples);

}  // namespace eureka
