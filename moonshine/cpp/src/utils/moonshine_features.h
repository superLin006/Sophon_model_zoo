#pragma once
// Moonshine 特征提取(方案 B, 与 python/test/test_onnx.py 的 numpy 公式一致)
// ---------------------------------------------------------------------
// 1. 音频 16kHz mono float32, 尾部补零到 160000 samples(10s)
// 2. 分帧 reshape [2000, 80](frame_len=80 = 5ms)
// 3. CMVN: mean = x.mean(-1); centered = x - mean
//          rms = sqrt(centered^2.mean(-1) + 1e-6); normed = centered / rms
// 4. asinh(exp(log_k) * normed) -> x_frames [1,2000,80] float32
// LOG_K = -0.4875200987, eps = 1e-6(均与 make_ref.py / export_onnx.py 一致)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace moonshine {

constexpr int N_SAMPLES = 160000;   // 10s @ 16kHz
constexpr int FRAME_LEN = 80;
constexpr int N_FRAMES = 2000;      // 160000 / 80
constexpr float LOG_K = -0.4875200987f;
constexpr float EPS = 1e-6f;

constexpr int N_LAYER = 10;
constexpr int MAX_DEC_LEN = 128;
constexpr int HID = 512;
constexpr int VOCAB = 32768;
constexpr int TOK_SOS = 1;
constexpr int TOK_EOS = 2;

// audio16k: 16kHz mono float32(可短于 10s)
// out: [N_FRAMES * FRAME_LEN] float32 = [1, 2000, 80] 行主序
inline void compute_x_frames(const std::vector<float>& audio16k, std::vector<float>& out) {
    // 补零到 160000 samples(与 Python `out = np.zeros(160000); out[:len]=audio` 一致)
    std::vector<float> buf(N_SAMPLES, 0.0f);
    size_t n = std::min(audio16k.size(), (size_t)N_SAMPLES);
    if (n) std::memcpy(buf.data(), audio16k.data(), n * sizeof(float));

    out.resize((size_t)N_FRAMES * FRAME_LEN);
    // numpy 公式实际按 float64 计算最后 astype(float32)(NEP50 下 np.exp(log_k)
    // 为 np.float64 标量), 故 C++ 用 double 累加、末尾一次转 float, 与其一致。
    const double k = std::exp((double)LOG_K);
    for (int f = 0; f < N_FRAMES; f++) {
        const float* x = buf.data() + (size_t)f * FRAME_LEN;
        double sum = 0.0;
        for (int j = 0; j < FRAME_LEN; j++) sum += x[j];
        const double mean = sum / FRAME_LEN;
        double sq = 0.0;
        for (int j = 0; j < FRAME_LEN; j++) { const double c = (double)x[j] - mean; sq += c * c; }
        const double rms = std::sqrt(sq / FRAME_LEN + (double)EPS);
        float* o = out.data() + (size_t)f * FRAME_LEN;
        for (int j = 0; j < FRAME_LEN; j++) {
            o[j] = (float)std::asinh(k * ((double)x[j] - mean) / rms);
        }
    }
}

}  // namespace moonshine
