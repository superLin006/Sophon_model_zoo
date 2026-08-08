// Qwen3-ASR log-mel 前处理实现
// 公式对齐 transformers Qwen3ASRFeatureExtractor（torch.stft center=True + slaney mel + log10 压缩）

#include "qwen_mel.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <complex>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include "fftw3.h"

namespace asr {

static inline float log10_safe(float x) {
    if (x < 1e-10f) x = 1e-10f;
    return std::log10(x);
}

std::vector<float> QwenMel::make_hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n));
    return w;
}

// ── 简单 npz 解析（仅无压缩 float32 数组，key 精确匹配）──────────────────────

static bool parse_npy_header(const uint8_t* buf, size_t buf_len,
                             size_t& data_offset, std::vector<int>& shape) {
    if (buf_len < 12) return false;
    if (buf[0] != 0x93 || buf[1] != 'N' || buf[2] != 'U' ||
        buf[3] != 'M'  || buf[4] != 'P' || buf[5] != 'Y')
        return false;
    uint16_t header_len = (uint16_t)buf[8] | ((uint16_t)buf[9] << 8);
    if ((size_t)(10 + header_len) > buf_len) return false;
    std::string header((const char*)buf + 10, header_len);

    auto sp = header.find("'shape'");
    if (sp == std::string::npos) sp = header.find("\"shape\"");
    if (sp == std::string::npos) return false;
    sp = header.find('(', sp);
    if (sp == std::string::npos) return false;
    auto ep = header.find(')', sp);
    if (ep == std::string::npos) return false;
    std::string shape_str = header.substr(sp + 1, ep - sp - 1);
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(0, tok.find_first_not_of(" \t"));
        tok.erase(tok.find_last_not_of(" \t") + 1);
        if (tok.empty() || tok == " ") continue;
        shape.push_back(std::atoi(tok.c_str()));
    }
    data_offset = 10 + header_len;
    return !shape.empty();
}

bool QwenMel::load_npz_array(const std::string& path, const std::string& key,
                             std::vector<float>& data, std::vector<int>& shape) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[QwenMel] cannot open %s\n", path.c_str()); return false; }
    std::vector<uint8_t> zip((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // zip: Local File Header 签名 0x04034b50
    size_t pos = 0;
    while (pos + 30 <= zip.size()) {
        if (zip[pos] != 0x50 || zip[pos+1] != 0x4b || zip[pos+2] != 0x03 || zip[pos+3] != 0x04)
            break;
        uint16_t method = zip[pos+8] | (zip[pos+9] << 8);
        uint32_t csize  = (uint32_t)zip[pos+18] | ((uint32_t)zip[pos+19] << 8) |
                          ((uint32_t)zip[pos+20] << 16) | ((uint32_t)zip[pos+21] << 24);
        uint32_t usize  = (uint32_t)zip[pos+22] | ((uint32_t)zip[pos+23] << 8) |
                          ((uint32_t)zip[pos+24] << 16) | ((uint32_t)zip[pos+25] << 24);
        uint16_t nlen   = zip[pos+26] | (zip[pos+27] << 8);
        uint16_t elen   = zip[pos+28] | (zip[pos+29] << 8);
        // ZIP64 兼容：numpy 2.x 写的 npz 即使小文件也用 ZIP64 占位（0xFFFFFFFF），
        // 实际大小在 extra field（id=0x0001）里
        if (csize == 0xFFFFFFFFu || usize == 0xFFFFFFFFu) {
            size_t epos = pos + 30 + nlen;
            while (epos + 4 <= pos + 30 + nlen + elen) {
                uint16_t eid = zip[epos] | (zip[epos+1] << 8);
                uint16_t esz = zip[epos+2] | (zip[epos+3] << 8);
                if (eid == 1 && esz >= 16) {
                    uint64_t u = 0, c = 0;
                    memcpy(&u, &zip[epos+4], 8);
                    memcpy(&c, &zip[epos+12], 8);
                    if (usize == 0xFFFFFFFFu) usize = (uint32_t)u;
                    if (csize == 0xFFFFFFFFu) csize = (uint32_t)c;
                }
                epos += 4 + esz;
            }
        }
        std::string name((const char*)&zip[pos+30], nlen);
        size_t data_start = pos + 30 + nlen + elen;
        if (method == 0 && name == key + ".npy" && data_start + csize <= zip.size()) {
            const uint8_t* npy = &zip[data_start];
            size_t off = 0;
            if (!parse_npy_header(npy, csize, off, shape)) {
                fprintf(stderr, "[QwenMel] bad npy header for key %s\n", key.c_str());
                return false;
            }
            size_t n = 1;
            for (int d : shape) n *= (size_t)d;
            if (off + n * 4 > csize) return false;
            data.resize(n);
            memcpy(data.data(), npy + off, n * 4);
            return true;
        }
        pos = data_start + csize;
    }
    fprintf(stderr, "[QwenMel] key %s not found in %s\n", key.c_str(), path.c_str());
    return false;
}

bool QwenMel::load_filters(const std::string& npz_path) {
    std::vector<int> shape;
    if (!load_npz_array(npz_path, "mel_filters", mel_filters_, shape)) {
        fprintf(stderr, "[QwenMel] Failed to load mel_filters from %s\n", npz_path.c_str());
        return false;
    }
    // dump 为 [201, 128]（n_fft//2+1, n_mels）→ 转置为 [128, 201]
    if (shape.size() != 2 || shape[1] != N_MELS) {
        fprintf(stderr, "[QwenMel] unexpected filter shape [%d,%d]\n", shape[0], shape[1]);
        return false;
    }
    int rows = shape[0], cols = shape[1];
    std::vector<float> t(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            t[c * rows + r] = mel_filters_[r * cols + c];
    mel_filters_.swap(t);
    filters_loaded_ = true;
    printf("[QwenMel] filters loaded [%d, %d] (%d elems)\n", N_MELS, rows, (int)mel_filters_.size());
    return true;
}

// FFTW3（1_third_party/fftw，chatTTS 同款）：r2c plan，n_fft=400 → 201 bins（与 torch.stft 一致，无需 pad）
bool QwenMel::init_fftw() {
    if (fftw_ready_) return true;
    fftw_in_  = fftwf_malloc(sizeof(float) * N_FFT);
    fftw_out_ = fftwf_malloc(sizeof(float) * (N_FFT / 2 + 1) * 2);  // fftwf_complex = 2 float
    if (!fftw_in_ || !fftw_out_) return false;
    fftw_plan_ = fftwf_plan_dft_r2c_1d(N_FFT, (float*)fftw_in_, (fftwf_complex*)fftw_out_, FFTW_ESTIMATE);
    if (!fftw_plan_) return false;
    fftw_ready_ = true;
    return true;
}

// torch.stft(center=True, pad_mode='reflect', return_complex=True) 的 numpy 等价
// 输出幅度谱 power=2：[N_FFT/2+1, T]，丢弃最后一帧
// 帧数 = (len + 2*pad - n_fft)/hop + 1 = len/hop + 1（整数除法），丢弃最后一帧 → len/hop
// 注意：不能按 (len+2*pad)/hop + 1 算——那会多出几帧并越界读（读 heap 垃圾产生 NaN）
std::vector<float> QwenMel::log_mel_spectrogram(const std::vector<float>& audio,
                                                int& t_frames) const {
    assert(filters_loaded_);
    if (!fftw_ready_) const_cast<QwenMel*>(this)->init_fftw();

    int len = (int)audio.size();
    int pad = N_FFT / 2;
    int total = len + 2 * pad;
    int n_frames_full = (total - N_FFT) / HOP_LENGTH + 1;   // torch.stft 帧数
    int frames = n_frames_full - 1;                          // 丢弃最后一帧
    t_frames = frames;
    int n_bins = N_FFT / 2 + 1;

    // reflect pad（对齐 numpy/torch reflect 语义）：
    //   左 pad k 个 = a[k], a[k-1], ..., a[1]（不重复 a[0]）
    //   右 pad k 个 = a[n-2], a[n-3], ..., a[n-1-k]
    std::vector<float> padded(total);
    for (int i = 0; i < pad; ++i) {
        padded[i]             = audio[pad - i];           // 左：a[pad]..a[1]
        padded[len + pad + i] = audio[len - 2 - i];      // 右：a[n-2]..a[n-1-pad]
    }
    memcpy(&padded[pad], audio.data(), len * sizeof(float));

    auto window = make_hann_window(N_FFT);

    // 逐帧 FFT → 功率谱
    std::vector<float> mag((size_t)n_bins * frames);
    for (int t = 0; t < frames; ++t) {
        const float* frame = &padded[(size_t)t * HOP_LENGTH];
        float* in = (float*)fftw_in_;
        for (int i = 0; i < N_FFT; ++i) in[i] = frame[i] * window[i];
        fftwf_execute((fftwf_plan)fftw_plan_);
        const float* out = (const float*)fftw_out_;
        float* pw = mag.data() + (size_t)t * n_bins;
        for (int k = 0; k < n_bins; ++k) {
            float re = out[2 * k], im = out[2 * k + 1];
            pw[k] = re * re + im * im;
        }
    }

    // mel filter + log10 压缩
    std::vector<float> mel((size_t)N_MELS * frames);
    for (int t = 0; t < frames; ++t) {
        const float* col = &mag[(size_t)t * n_bins];
        for (int m = 0; m < N_MELS; ++m) {
            const float* filt = &mel_filters_[(size_t)m * n_bins];
            double acc = 0;
            for (int k = 0; k < n_bins; ++k) acc += (double)filt[k] * col[k];
            mel[(size_t)t * N_MELS + m] = log10_safe((float)acc);
        }
    }

    // max-8 动态裁剪（按 utterance 的全局 max）
    float gmax = -1e30f;
    for (float v : mel) if (v > gmax) gmax = v;
    for (float& v : mel) {
        if (v < gmax - 8.0f) v = gmax - 8.0f;
        v = (v + 4.0f) / 4.0f;
    }
    return mel;   // [T, N_MELS] 行主序（t-major）
}

// 增量 log10-mel（未裁剪）：只算帧 [start_frame, n_frames)，追加到 mel_log
// 与 log_mel_spectrogram 的 FFT+filter+log10 部分逐位一致（不含 max-8 裁剪与缩放）
void QwenMel::log_mel_frames(const std::vector<float>& audio, int start_frame,
                             std::vector<float>& mel_log, int& n_frames) const {
    assert(filters_loaded_);
    if (!fftw_ready_) const_cast<QwenMel*>(this)->init_fftw();

    int len = (int)audio.size();
    int pad = N_FFT / 2;
    int total = len + 2 * pad;
    n_frames = (total - N_FFT) / HOP_LENGTH + 1 - 1;   // 丢弃最后一帧
    if (n_frames < start_frame) n_frames = start_frame;
    int n_bins = N_FFT / 2 + 1;

    // reflect pad（全量做，O(n) 拷贝很快）
    std::vector<float> padded(total);
    for (int i = 0; i < pad; ++i) {
        padded[i]             = audio[pad - i];
        padded[len + pad + i] = audio[len - 2 - i];
    }
    memcpy(&padded[pad], audio.data(), len * sizeof(float));

    auto window = make_hann_window(N_FFT);
    mel_log.reserve((size_t)n_frames * N_MELS);
    // 功率谱每帧只算一次（201 bins），128 个 mel filter 共用（避免 127 次冗余乘法）
    std::vector<float> mag(n_bins);
    for (int t = start_frame; t < n_frames; ++t) {
        const float* frame = &padded[(size_t)t * HOP_LENGTH];
        float* in = (float*)fftw_in_;
        for (int i = 0; i < N_FFT; ++i) in[i] = frame[i] * window[i];
        fftwf_execute((fftwf_plan)fftw_plan_);
        const float* out = (const float*)fftw_out_;
        for (int k = 0; k < n_bins; ++k) {
            float re = out[2 * k], im = out[2 * k + 1];
            mag[k] = re * re + im * im;
        }
        for (int m = 0; m < N_MELS; ++m) {
            const float* filt = &mel_filters_[(size_t)m * n_bins];
            double acc = 0;
            for (int k = 0; k < n_bins; ++k) acc += (double)filt[k] * mag[k];
            mel_log.push_back(log10_safe((float)acc));
        }
    }
}

std::vector<float> QwenMel::process(const std::vector<float>& samples, int& t_real) const {
    assert(filters_loaded_);
    int frames = 0;
    auto mel = log_mel_spectrogram(samples, frames);   // [T, 128]
    t_real = frames;

    // 复制最后一帧 pad 到 CHUNK_FRAMES（[T,128] → [3000,128]）
    std::vector<float> out((size_t)CHUNK_FRAMES * N_MELS, 0.0f);
    int copy = std::min(frames, CHUNK_FRAMES);
    if (copy > 0) memcpy(out.data(), mel.data(), (size_t)copy * N_MELS * sizeof(float));
    for (int t = copy; t < CHUNK_FRAMES; ++t)
        memcpy(&out[(size_t)t * N_MELS], &out[(size_t)(copy-1) * N_MELS], N_MELS * sizeof(float));
    return out;   // [3000, 128] 行主序（mel 帧-major），encoder bmodel 输入 [1,128,3000] 需转置
}

bool QwenMel::wav_to_mel(const std::string& wav_path, std::vector<float>& mel_out, int& t_real) const {
    std::vector<float> samples;
    if (!load_wav_16k_mono(wav_path, samples)) return false;
    if (samples.empty()) return false;
    int total_ms = (int)(samples.size() * 1000 / SAMPLE_RATE);
    if (total_ms > 30000) {
        fprintf(stderr, "[QwenMel] audio too long: %d ms > 30000 ms\n", total_ms);
        return false;
    }
    mel_out = process(samples, t_real);
    return true;
}

// ── WAV 加载（16kHz mono float32）─────────────────────────────────────────────

bool load_wav_16k_mono(const std::string& path, std::vector<float>& samples) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    char hdr[12];
    if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr+8, "WAVE", 4) != 0) {
        fclose(fp); return false;
    }
    uint32_t sample_rate = 0;
    uint16_t channels = 0, bits = 0;
    bool got_fmt = false, got_data = false;
    uint32_t data_size = 0;
    while (!got_data) {
        char chunk[8];
        if (fread(chunk, 1, 8, fp) != 8) break;
        uint32_t csize = (uint32_t)(uint8_t)chunk[4] | ((uint32_t)(uint8_t)chunk[5] << 8) |
                         ((uint32_t)(uint8_t)chunk[6] << 16) | ((uint32_t)(uint8_t)chunk[7] << 24);
        if (memcmp(chunk, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (csize >= 16 && fread(fmt, 1, 16, fp) == 16) {
                channels = (uint16_t)fmt[2] | ((uint16_t)fmt[3] << 8);
                sample_rate = (uint32_t)fmt[4] | ((uint32_t)fmt[5] << 8) |
                              ((uint32_t)fmt[6] << 16) | ((uint32_t)fmt[7] << 24);
                bits = (uint16_t)fmt[14] | ((uint16_t)fmt[15] << 8);
                got_fmt = true;
            }
            if (csize > 16) fseek(fp, csize - 16, SEEK_CUR);
        } else if (memcmp(chunk, "data", 4) == 0) {
            data_size = csize;
            got_data = true;
        } else {
            fseek(fp, csize, SEEK_CUR);
        }
    }
    if (!got_fmt || !got_data || sample_rate != 16000 || channels != 1) {
        fclose(fp);
        if (sample_rate != 16000) fprintf(stderr, "[QwenMel] sample rate %u != 16000\n", sample_rate);
        return false;
    }
    int n = (int)(data_size / (bits / 8));
    samples.resize(n);
    if (bits == 16) {
        std::vector<int16_t> raw(n);
        if (fread(raw.data(), 2, n, fp) != (size_t)n) { fclose(fp); return false; }
        for (int i = 0; i < n; ++i) samples[i] = raw[i] / 32768.0f;
    } else if (bits == 32) {
        std::vector<float> raw(n);
        if (fread(raw.data(), 4, n, fp) != (size_t)n) { fclose(fp); return false; }
        memcpy(samples.data(), raw.data(), n * 4);
    } else {
        fclose(fp); return false;
    }
    fclose(fp);
    return true;
}

}  // namespace asr
