#include "istft.h"
#include <fftw3.h>
#include <cmath>
#include <stdexcept>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ISTFT::ISTFT(int n_fft, int hop_length, int win_length)
    : n_fft_(n_fft), hop_(hop_length), win_(win_length) {

    // Hann window
    window_.resize(win_);
    window_sq_.resize(win_);
    for (int i = 0; i < win_; ++i) {
        float w = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / win_));
        window_[i]    = w;
        window_sq_[i] = w * w;
    }
}

ISTFT::~ISTFT() = default;

std::vector<float> ISTFT::forward(const std::vector<float>& mag,
                                  const std::vector<float>& x,
                                  const std::vector<float>& y,
                                  int T) const {
    // n_bins = n_fft/2 + 1 = 513
    const int n_bins = n_fft_ / 2 + 1;
    if ((int)mag.size() != n_bins * T)
        throw std::runtime_error("ISTFT: mag size mismatch");

    // center padding: pad = (win - hop) / 2
    const int pad = (win_ - hop_) / 2;

    // output_size before trimming
    const int output_size = (T - 1) * hop_ + win_;
    // allocate OLA buffers
    std::vector<float> y_ola(output_size, 0.0f);
    std::vector<float> env(output_size, 0.0f);

    // FFTW plan: irfft of size n_fft (complex → real)
    // We reuse one plan for all frames
    std::vector<fftwf_complex> freq(n_bins);
    std::vector<float>         frame_out(n_fft_);

    fftwf_plan plan = fftwf_plan_dft_c2r_1d(n_fft_,
                                             freq.data(),
                                             frame_out.data(),
                                             FFTW_ESTIMATE);
    if (!plan)
        throw std::runtime_error("ISTFT: fftw plan creation failed");

    for (int t = 0; t < T; ++t) {
        // Build complex spectrum: mag * (x + j*y)
        for (int k = 0; k < n_bins; ++k) {
            float m  = mag[k * T + t];
            float re = x  [k * T + t];
            float im = y  [k * T + t];
            freq[k][0] = m * re;
            freq[k][1] = m * im;
        }

        fftwf_execute(plan);

        // Apply window and accumulate (OLA)
        int pos = t * hop_;
        for (int i = 0; i < win_; ++i) {
            if (pos + i < output_size) {
                y_ola[pos + i] += frame_out[i] * window_[i] / n_fft_;
                env  [pos + i] += window_sq_[i];
            }
        }
    }

    fftwf_destroy_plan(plan);

    // Normalize by window envelope, trim padding
    // center padding: trim pad from each side
    const int out_len = output_size - 2 * pad;
    std::vector<float> audio(out_len);
    for (int i = 0; i < out_len; ++i) {
        int j = i + pad;
        float e = env[j];
        audio[i] = (e > 1e-11f) ? y_ola[j] / e : 0.0f;
    }
    return audio;
}
