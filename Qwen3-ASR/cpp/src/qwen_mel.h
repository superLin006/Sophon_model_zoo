#pragma once

#include <string>
#include <vector>

namespace asr {

// Qwen3-ASR log-mel 前处理（与 Qwen3ASRFeatureExtractor 完全对齐）：
//   STFT: n_fft=400, hop=160, hann window, power=2, center=True(reflect pad)，丢最后一帧
//   mel filter: 128 bins slaney（从 mel_filters.npz 加载，dump 自原生 feature_extractor）
//   log10 + clip(1e-10) + (max-8) clamp + (x+4)/4 归一化
//   输出切块：每块 3000 帧（30s），不足**复制最后一帧**（非补零——encoder 非因果
//   窗口 attention，pad 帧参与注意力，复制尾部 ≈ 静音延续，污染温和）
class QwenMel {
public:
    QwenMel() = default;

    // 从 mel_filters.npz 加载 128-bin mel 滤波器（npz key "mel_filters"，shape [201,128]，需转置）
    bool load_filters(const std::string& npz_path);

    // 从 WAV 文件加载并转 mel → [1, 128, 3000] 单块（30s 上限）
    // 返回: mel 数据（128*3000 行主序）、实际 mel 帧数 T_real
    bool wav_to_mel(const std::string& wav_path, std::vector<float>& mel_out, int& t_real) const;

    // 核心：waveform → [128, 3000] mel（尾部复制 pad），返回实际帧数
    std::vector<float> process(const std::vector<float>& samples, int& t_real) const;

    // log-mel spectrogram，[T, N_MELS]（t-major；流式增量 mel 用）
    std::vector<float> log_mel_spectrogram(const std::vector<float>& audio, int& t_frames) const;

    // 增量 log10-mel（未裁剪）：只计算帧 [start_frame, n_frames) 的 FFT+filter+log10，
    // 追加到 mel_log（之前帧已缓存）。流式用：STFT 帧独立，新帧只依赖 audio 的 pad 上下文。
    void log_mel_frames(const std::vector<float>& audio, int start_frame,
                        std::vector<float>& mel_log, int& n_frames) const;

private:
    static constexpr int N_FFT        = 400;
    static constexpr int HOP_LENGTH   = 160;
    static constexpr int N_MELS       = 128;
    static constexpr int CHUNK_FRAMES = 3000;
    static constexpr int SAMPLE_RATE  = 16000;

    std::vector<float> mel_filters_;  // [N_MELS, N_FFT/2+1] = [128, 201]
    bool filters_loaded_ = false;

    // FFTW3（1_third_party/fftw，chatTTS/moonshine 同款）：plan + buffer 复用
    void* fftw_plan_ = nullptr;    // fftwf_plan（r2c, n_fft=400 → 201 bins）
    void* fftw_in_   = nullptr;    // float[n_fft]（加窗后的帧）
    void* fftw_out_  = nullptr;    // fftwf_complex[n_fft/2+1]
    bool fftw_ready_ = false;
    bool init_fftw();              // 创建 plan + buffer


    // Hann 窗
    static std::vector<float> make_hann_window(int n);

    // 最小 npz 解析（无压缩 float32）
    bool load_npz_array(const std::string& path, const std::string& key,
                        std::vector<float>& data, std::vector<int>& shape) const;
};

// 工具：加载 WAV → 16kHz mono float32
bool load_wav_16k_mono(const std::string& path, std::vector<float>& samples);

}  // namespace asr
