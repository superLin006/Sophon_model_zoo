#include "wav.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>

static uint32_t u32(const unsigned char* p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
}
static uint16_t u16(const unsigned char* p) {
    return p[0] | (p[1] << 8);
}

bool ReadWav16(const std::string& path, Wav* wav, std::string* error) {
    if (!wav) return false;
    auto err = [&](const std::string& s) { if (error) *error = s; return false; };
    std::ifstream f(path, std::ios::binary);
    if (!f) return err("cannot open WAV");
    unsigned char h[12];
    f.read(reinterpret_cast<char*>(h), 12);
    if (f.gcount() != 12 || std::memcmp(h, "RIFF", 4) ||
        std::memcmp(h + 8, "WAVE", 4))
        return err("not RIFF/WAVE");

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

    if (fmt != 1 || !ch || rate != 16000 || bits != 16 || data.empty())
        return err("only PCM16 mono/multichannel 16k WAV is supported");

    wav->sample_rate = rate;
    wav->num_channels = ch;
    size_t ns = data.size() / (2 * ch);
    wav->samples.resize(ns);
    for (size_t i = 0; i < ns; ++i) {
        double z = 0;
        for (int c = 0; c < ch; ++c) {
            int16_t v = static_cast<int16_t>(u16(data.data() + 2 * (i * ch + c)));
            z += v / 32768.0;
        }
        wav->samples[i] = static_cast<float>(z / ch);
    }
    return true;
}
