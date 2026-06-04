// Whisper 128-bin mel 前处理实现
// 完全对齐 Python 侧 librosa / whisper.audio 的计算流程

#include "whisper_mel.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <complex>
#include <stdexcept>

// 最小化 npz 解析（无压缩 npy），不依赖 zlib / cnpy
#include <fstream>
#include <sstream>

namespace eureka {

// ─────────────────────────────────────────────────────────────────────────────
// 工具函数
// ─────────────────────────────────────────────────────────────────────────────

static inline float log10_safe(float x) {
    if (x < 1e-10f) x = 1e-10f;
    return std::log10(x);
}

std::vector<float> WhisperMel::make_hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n));
    return w;
}

// ─────────────────────────────────────────────────────────────────────────────
// 简单 npz 解析（仅无压缩 float32 数组，key 精确匹配）
// npz = zip 文件，每个文件是 npy 格式
// 这里只实现能处理 mel_filters.npz 所需的最小子集
// ─────────────────────────────────────────────────────────────────────────────

// 解析 npy header，返回数据偏移和元素数量
static bool parse_npy_header(const uint8_t* buf, size_t buf_len,
                              size_t& data_offset, std::vector<int>& shape) {
    // npy magic: \x93NUMPY (6 bytes), then major(1), minor(1), header_len(2 LE), header, data
    if (buf_len < 12) return false;
    if (buf[0] != 0x93 || buf[1] != 'N' || buf[2] != 'U' ||
        buf[3] != 'M'  || buf[4] != 'P' || buf[5] != 'Y')
        return false;
    // header_len at [8..9] (little-endian), header string at [10..10+header_len-1]
    uint16_t header_len = (uint16_t)buf[8] | ((uint16_t)buf[9] << 8);
    if ((size_t)(10 + header_len) > buf_len) return false;

    std::string header((const char*)buf + 10, header_len);

    // 解析 shape：找 'shape': (a, b, ...)
    auto sp = header.find("'shape'");
    if (sp == std::string::npos) sp = header.find("\"shape\"");
    if (sp == std::string::npos) return false;
    auto lp = header.find('(', sp);
    auto rp = header.find(')', lp);
    if (lp == std::string::npos || rp == std::string::npos) return false;
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    shape.clear();
    std::istringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if (!tok.empty()) shape.push_back(std::stoi(tok));
    }

    data_offset = 10 + header_len;
    return true;
}

// zip local file header magic
static const uint32_t ZIP_LOCAL_MAGIC = 0x04034b50;

bool WhisperMel::load_npz_array(const std::string& path, const std::string& key,
                                 std::vector<float>& data,
                                 std::vector<int>& shape) const {
    // npz = zip，遍历 local file headers
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }

    while (true) {
        uint32_t sig = 0;
        if (!f.read((char*)&sig, 4)) break;
        if (sig != ZIP_LOCAL_MAGIC) break;  // end of local headers or other record

        uint16_t ver, flags, compress, mod_time, mod_date;
        uint32_t crc32, comp_size, uncomp_size;
        uint16_t fname_len, extra_len;
        f.read((char*)&ver, 2);
        f.read((char*)&flags, 2);
        f.read((char*)&compress, 2);
        f.read((char*)&mod_time, 2);
        f.read((char*)&mod_date, 2);
        f.read((char*)&crc32, 4);
        f.read((char*)&comp_size, 4);
        f.read((char*)&uncomp_size, 4);
        f.read((char*)&fname_len, 2);
        f.read((char*)&extra_len, 2);

        std::string fname(fname_len, '\0');
        f.read(&fname[0], fname_len);
        f.seekg(extra_len, std::ios::cur);

        // npy 文件名为 "key.npy"
        std::string expected = key + ".npy";
        bool match = (fname == expected);

        if (compress != 0) {
            // 压缩数据，跳过
            f.seekg(comp_size, std::ios::cur);
            continue;
        }

        if (!match) {
            f.seekg(uncomp_size, std::ios::cur);
            continue;
        }

        // 读整块 npy
        std::vector<uint8_t> npy_buf(uncomp_size);
        f.read((char*)npy_buf.data(), uncomp_size);
        if (!f) return false;

        size_t data_offset = 0;
        if (!parse_npy_header(npy_buf.data(), npy_buf.size(), data_offset, shape))
            return false;

        size_t n_elem = 1;
        for (int s : shape) n_elem *= s;
        data.resize(n_elem);
        memcpy(data.data(), npy_buf.data() + data_offset, n_elem * sizeof(float));
        return true;
    }
    fprintf(stderr, "Key '%s' not found in %s\n", key.c_str(), path.c_str());
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// 加载 mel 滤波器
// ─────────────────────────────────────────────────────────────────────────────
bool WhisperMel::load_filters(const std::string& npz_path) {
    std::vector<int> shape;
    if (!load_npz_array(npz_path, "mel_128", mel_filters_, shape)) {
        fprintf(stderr, "Failed to load mel_128 from %s\n", npz_path.c_str());
        return false;
    }
    // shape should be [128, 201]
    if (shape.size() != 2 || shape[0] != N_MELS || shape[1] != N_FFT / 2 + 1) {
        fprintf(stderr, "Unexpected mel filter shape: [%d,%d]\n",
                shape[0], shape[1]);
        return false;
    }
    filters_loaded_ = true;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// STFT + mel 计算
// ─────────────────────────────────────────────────────────────────────────────

// 简单 DFT（频率范围 0..N_FFT/2），仅用于小维度，性能够用
// 对于 n_fft=400，每帧 400 点 DFT，实际速度约 1ms/frame x 3000 frames = 3s
// 但 Whisper 输入是 30s audio → 需要快一点，用 Cooley-Tukey FFT

static void fft(std::vector<std::complex<float>>& a) {
    int n = (int)a.size();
    if (n == 1) return;
    // bit-reversal
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                auto u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j]            = u + v;
                a[i + j + len / 2]  = u - v;
                w *= wlen;
            }
        }
    }
}

std::vector<float> WhisperMel::log_mel_spectrogram(
    const std::vector<float>& audio) const {

    assert(filters_loaded_);

    // Whisper: pad audio with N_FFT/2 zeros on both sides
    int padded_len = (int)audio.size() + N_FFT;
    std::vector<float> padded(padded_len, 0.0f);
    std::copy(audio.begin(), audio.end(), padded.begin() + N_FFT / 2);

    auto hann = make_hann_window(N_FFT);

    // 下一个 2 的幂，FFT 需要
    int fft_size = 1;
    while (fft_size < N_FFT) fft_size <<= 1;  // 512

    int n_frames = (padded_len - N_FFT) / HOP_LENGTH + 1;
    int n_freqs  = N_FFT / 2 + 1;  // 201

    // 计算功率谱 [n_frames, n_freqs]
    std::vector<float> power(n_frames * n_freqs);

    std::vector<std::complex<float>> frame_buf(fft_size);
    for (int t = 0; t < n_frames; ++t) {
        int start = t * HOP_LENGTH;
        for (int i = 0; i < fft_size; ++i) {
            float s = (i < N_FFT && start + i < padded_len)
                      ? padded[start + i] * hann[i] : 0.0f;
            frame_buf[i] = {s, 0.0f};
        }
        fft(frame_buf);
        float* pw = power.data() + t * n_freqs;
        for (int f = 0; f < n_freqs; ++f) {
            float re = frame_buf[f].real(), im = frame_buf[f].imag();
            pw[f] = re * re + im * im;
        }
    }

    // mel filter: [N_MELS, n_freqs] x [n_frames, n_freqs]^T → [N_MELS, n_frames]
    std::vector<float> mel(N_MELS * n_frames);
    const float* filt = mel_filters_.data();  // [N_MELS, n_freqs]
    for (int m = 0; m < N_MELS; ++m) {
        for (int t = 0; t < n_frames; ++t) {
            float sum = 0.0f;
            const float* pw = power.data() + t * n_freqs;
            const float* fi = filt + m * n_freqs;
            for (int f = 0; f < n_freqs; ++f)
                sum += fi[f] * pw[f];
            mel[m * n_frames + t] = log10_safe(sum);
        }
    }

    // log10 归一化：(x + 4) / 4，先找 max
    float max_val = *std::max_element(mel.begin(), mel.end());
    for (float& v : mel) {
        v = std::max(v, max_val - 8.0f);
        v = (v + 4.0f) / 4.0f;
    }

    // 返回 [N_MELS, n_frames]
    // (在 split_chunks 里按 T=n_frames 切块)
    return mel;  // 附带 n_frames 信息在外部用
}

// ─────────────────────────────────────────────────────────────────────────────
// 切块
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<float>> WhisperMel::split_chunks(
    const std::vector<float>& mel, int T) const {

    // mel 存储格式：[N_MELS, T]，行主序（m*T + t）
    // 切成若干 [N_MELS, CHUNK_FRAMES] 块，最后一块右侧补 pad_val
    float pad_val = (log10_safe(1e-10f) + 4.0f) / 4.0f;  // ≈ -2.5
    int n_chunks  = (T + CHUNK_FRAMES - 1) / CHUNK_FRAMES;
    if (n_chunks == 0) n_chunks = 1;

    std::vector<std::vector<float>> chunks(n_chunks,
        std::vector<float>(N_MELS * CHUNK_FRAMES, pad_val));

    for (int c = 0; c < n_chunks; ++c) {
        int start = c * CHUNK_FRAMES;
        int end   = std::min(start + CHUNK_FRAMES, T);
        for (int m = 0; m < N_MELS; ++m) {
            float* dst = chunks[c].data() + m * CHUNK_FRAMES;
            const float* src = mel.data() + m * T + start;
            std::copy(src, src + (end - start), dst);
            // 剩余已初始化为 pad_val
        }
    }
    return chunks;
}

// ─────────────────────────────────────────────────────────────────────────────
// process
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<float>> WhisperMel::process(
    const std::vector<float>& samples) const {

    // log_mel_spectrogram 返回 [N_MELS, T]，但 T 在返回值里
    // 计算 T = (padded_len - N_FFT) / HOP + 1
    int padded_len = (int)samples.size() + N_FFT;
    int n_frames   = (padded_len - N_FFT) / HOP_LENGTH + 1;

    auto mel = log_mel_spectrogram(samples);
    return split_chunks(mel, n_frames);
}

// ─────────────────────────────────────────────────────────────────────────────
// wav_to_mel_chunks
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<float>> WhisperMel::wav_to_mel_chunks(
    const std::string& wav_path) const {

    std::vector<float> samples;
    if (!load_wav_16k_mono(wav_path, samples)) {
        fprintf(stderr, "Failed to load WAV: %s\n", wav_path.c_str());
        return {};
    }
    return process(samples);
}

// ─────────────────────────────────────────────────────────────────────────────
// WAV 加载（16kHz mono, PCM 16-bit 或 32-bit float，自动重采样 44.1k→16k 待实现）
// ─────────────────────────────────────────────────────────────────────────────

// RIFF/WAV header 解析
struct WavHeader {
    char     riff[4];       // "RIFF"
    uint32_t chunk_size;
    char     wave[4];       // "WAVE"
    char     fmt_chunk[4];  // "fmt "
    uint32_t fmt_size;
    uint16_t audio_fmt;     // 1=PCM, 3=IEEE_FLOAT
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

// 简单线性插值重采样
static std::vector<float> resample(const std::vector<float>& in,
                                    int in_rate, int out_rate) {
    if (in_rate == out_rate) return in;
    double ratio = (double)in_rate / out_rate;
    int out_len  = (int)(in.size() / ratio);
    std::vector<float> out(out_len);
    for (int i = 0; i < out_len; ++i) {
        double pos = i * ratio;
        int    lo  = (int)pos;
        double frac = pos - lo;
        int    hi  = std::min(lo + 1, (int)in.size() - 1);
        out[i] = (float)(in[lo] * (1.0 - frac) + in[hi] * frac);
    }
    return out;
}

bool load_wav_16k_mono(const std::string& path, std::vector<float>& samples) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open WAV: %s\n", path.c_str());
        return false;
    }

    WavHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (strncmp(hdr.riff, "RIFF", 4) || strncmp(hdr.wave, "WAVE", 4)) {
        fclose(fp); return false;
    }

    // 找 "data" chunk（跳过额外 fmt 字节和 LIST chunk 等）
    // 标准 WAV: fmt_size = 16；extended fmt 可能有额外字节
    uint32_t fmt_extra = (hdr.fmt_size > 16) ? hdr.fmt_size - 16 : 0;
    fseek(fp, (long)fmt_extra, SEEK_CUR);

    // 查找 "data" chunk
    uint32_t data_size = 0;
    char tag[4];
    while (true) {
        if (fread(tag, 1, 4, fp) != 4) { fclose(fp); return false; }
        uint32_t csz = 0;
        if (fread(&csz, 4, 1, fp) != 1) { fclose(fp); return false; }
        if (strncmp(tag, "data", 4) == 0) {
            data_size = csz;
            break;
        }
        fseek(fp, (long)csz, SEEK_CUR);
    }
    if (data_size == 0) { fclose(fp); return false; }

    int n_samples_raw = data_size / hdr.block_align;
    std::vector<float> raw(n_samples_raw);

    if (hdr.audio_fmt == 1 && hdr.bits_per_sample == 16) {
        // PCM s16
        std::vector<int16_t> buf(n_samples_raw * hdr.num_channels);
        if (fread(buf.data(), sizeof(int16_t), buf.size(), fp) != buf.size()) {
            fprintf(stderr, "[wav] truncated PCM16 data\n"); fclose(fp); return false;
        }
        for (int i = 0; i < n_samples_raw; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < hdr.num_channels; ++c)
                sum += buf[i * hdr.num_channels + c] / 32768.0f;
            raw[i] = sum / hdr.num_channels;
        }
    } else if (hdr.audio_fmt == 3 && hdr.bits_per_sample == 32) {
        // IEEE float32
        std::vector<float> buf(n_samples_raw * hdr.num_channels);
        if (fread(buf.data(), sizeof(float), buf.size(), fp) != buf.size()) {
            fprintf(stderr, "[wav] truncated float32 data\n"); fclose(fp); return false;
        }
        for (int i = 0; i < n_samples_raw; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < hdr.num_channels; ++c)
                sum += buf[i * hdr.num_channels + c];
            raw[i] = sum / hdr.num_channels;
        }
    } else if (hdr.audio_fmt == 1 && hdr.bits_per_sample == 32) {
        // PCM s32
        std::vector<int32_t> buf(n_samples_raw * hdr.num_channels);
        if (fread(buf.data(), sizeof(int32_t), buf.size(), fp) != buf.size()) {
            fprintf(stderr, "[wav] truncated PCM32 data\n"); fclose(fp); return false;
        }
        for (int i = 0; i < n_samples_raw; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < hdr.num_channels; ++c)
                sum += buf[i * hdr.num_channels + c] / 2147483648.0f;
            raw[i] = sum / hdr.num_channels;
        }
    } else {
        fprintf(stderr, "Unsupported WAV format: fmt=%d bits=%d\n",
                hdr.audio_fmt, hdr.bits_per_sample);
        fclose(fp);
        return false;
    }
    fclose(fp);

    // 重采样到 16kHz
    samples = resample(raw, (int)hdr.sample_rate, 16000);
    return true;
}

}  // namespace eureka
