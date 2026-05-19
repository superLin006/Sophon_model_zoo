#pragma once
#include <vector>

// Inverse STFT (center padding), matches Python vocos_spectral_ops.ISTFT
// n_fft=1024, hop_length=256, win_length=1024, padding="center"
//
// Input: complex spectrogram as (mag, x, y) each [n_bins * T]
//        where complex = mag * (x + j*y),  n_bins = n_fft/2+1 = 513
// Output: PCM float32 samples
class ISTFT {
public:
    ISTFT(int n_fft = 1024, int hop_length = 256, int win_length = 1024);
    ~ISTFT();

    // mag/x/y each have length n_bins * T  (n_bins = n_fft/2+1)
    // Returns audio samples (length ≈ T * hop_length)
    std::vector<float> forward(const std::vector<float>& mag,
                               const std::vector<float>& x,
                               const std::vector<float>& y,
                               int T) const;

private:
    int n_fft_, hop_, win_;
    std::vector<float> window_;     // Hann window [win_length]
    std::vector<float> window_sq_;  // window^2   [win_length]
};
