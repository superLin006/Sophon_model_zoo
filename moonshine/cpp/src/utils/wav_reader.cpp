#include "wav_reader.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <cmath>

namespace {

uint32_t u32(const unsigned char* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24); }
uint16_t u16(const unsigned char* p) { return p[0] | (p[1] << 8); }

// 线性插值重采样: src 采样率 -> dst_rate
void resample_linear(const std::vector<float>& in, int src_rate, int dst_rate,
                     std::vector<float>& out) {
    if (src_rate == dst_rate) { out = in; return; }
    double ratio = (double)src_rate / dst_rate;   // 源位置 / 输出样本
    size_t n_out = (size_t)((in.size() * (double)dst_rate) / src_rate);
    // 与 torchaudio 对齐: 输出样本数 = floor(len * dst/src)
    out.resize(n_out);
    for (size_t i = 0; i < n_out; i++) {
        double pos = i * ratio;
        size_t i0 = (size_t)pos;
        double frac = pos - i0;
        double v0 = (i0 < in.size()) ? in[i0] : 0.0;
        double v1 = (i0 + 1 < in.size()) ? in[i0 + 1] : v0;
        out[i] = (float)(v0 + (v1 - v0) * frac);
    }
}

}  // namespace

bool ReadWavResample16k(const std::string& path, WavData* wav, std::string* error) {
    if (!wav) return false;
    auto err = [&](const std::string& s) { if (error) *error = s; return false; };

    std::ifstream f(path, std::ios::binary);
    if (!f) return err("cannot open WAV file: " + path);

    unsigned char h[12];
    f.read(reinterpret_cast<char*>(h), 12);
    if (f.gcount() != 12 || std::memcmp(h, "RIFF", 4) || std::memcmp(h + 8, "WAVE", 4))
        return err("not a RIFF/WAVE file");

    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<unsigned char> data;
    while (f) {
        unsigned char x[8];
        f.read(reinterpret_cast<char*>(x), 8);
        if (f.gcount() != 8) break;
        uint32_t n = u32(x + 4);
        if (!std::memcmp(x, "fmt ", 4)) {
            std::vector<unsigned char> b(n);
            f.read(reinterpret_cast<char*>(b.data()), n);
            if (n < 16) continue;
            fmt = u16(b.data());
            ch = u16(b.data() + 2);
            rate = u32(b.data() + 4);
            bits = u16(b.data() + 14);
            if (n & 1) f.get();
        } else if (!std::memcmp(x, "data", 4)) {
            data.resize(n);
            f.read(reinterpret_cast<char*>(data.data()), n);
            break;
        } else {
            f.seekg(n + (n & 1), std::ios::cur);
        }
    }

    if ((fmt != 1 && fmt != 3) || !ch || !rate || data.empty())
        return err("unsupported WAV format (need PCM16 or float32)");

    size_t ns = data.size() / (bits / 8) / ch;
    std::vector<float> mono(ns, 0.0f);
    if (fmt == 1 && bits == 16) {
        for (size_t i = 0; i < ns; i++) {
            double z = 0;
            for (int c = 0; c < ch; c++) {
                int16_t v = static_cast<int16_t>(u16(data.data() + 2 * (i * ch + c)));
                z += v / 32768.0;
            }
            mono[i] = (float)(z / ch);
        }
    } else if (fmt == 3 && bits == 32) {
        for (size_t i = 0; i < ns; i++) {
            double z = 0;
            for (int c = 0; c < ch; c++) {
                float v;
                std::memcpy(&v, data.data() + 4 * (i * ch + c), 4);
                z += v;
            }
            mono[i] = (float)(z / ch);
        }
    } else {
        return err("unsupported bit depth (need 16-bit PCM or 32-bit float)");
    }

    wav->sample_rate = (int)rate;
    wav->num_channels = ch;
    if (rate == 16000) {
        wav->samples = std::move(mono);
    } else {
        resample_linear(mono, (int)rate, 16000, wav->samples);
    }
    return true;
}
