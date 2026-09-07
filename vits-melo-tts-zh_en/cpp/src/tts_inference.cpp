#include "tts_inference.h"
#include "wav_writer.h"

#include <cstring>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

namespace vits_tts {

TTSInference::TTSInference() = default;
TTSInference::~TTSInference() { release(); }

// ─── helpers ─────────────────────────────────────────────────────────────────

// Load one bmodel into an existing runtime; return the first network found.
static const bm_net_info_t* load_bmodel_into(void* runtime, const std::string& path) {
    if (!bmrt_load_bmodel(runtime, path.c_str())) {
        std::cerr << "[Error] bmrt_load_bmodel failed: " << path << "\n";
        return nullptr;
    }
    // Network names accumulate across loads; query count each time.
    int net_count = bmrt_get_network_number(runtime);
    if (net_count <= 0) {
        std::cerr << "[Error] No networks after loading: " << path << "\n";
        return nullptr;
    }
    // The most recently loaded network is appended last.
    const char** names = nullptr;
    bmrt_get_network_names(runtime, &names);
    if (!names || !names[net_count - 1]) {
        std::cerr << "[Error] Cannot get network name: " << path << "\n";
        free(names);
        return nullptr;
    }
    const char* last_name = names[net_count - 1];
    const bm_net_info_t* info = bmrt_get_network_info(runtime, last_name);
    if (!info) {
        std::cerr << "[Error] Cannot get network info: " << last_name << "\n";
        free(names);
        return nullptr;
    }
    std::cerr << "[Init] Loaded " << path << "  net=" << last_name << "\n";
    free(names);
    return info;
}

static bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}


int TTSInference::init(const char* model_dir, const char* precision) {
    std::string base(model_dir);
    std::string prec(precision);

    if (bm_dev_request(&bm_handle_, 0) != BM_SUCCESS) {
        std::cerr << "[Error] bm_dev_request failed\n"; return -1;
    }

    // One shared runtime — all three bmodels share the same TPU DDR allocator,
    // preventing address conflicts that occur when using separate runtimes.
    runtime_ = bmrt_create(bm_handle_);
    if (!runtime_) {
        std::cerr << "[Error] bmrt_create failed\n";
        release();
        return -1;
    }

    std::string path_a  = base + "/vits_part_a_"  + prec + ".bmodel";
    std::string path_c1 = base + "/vits_part_c1_" + prec + ".bmodel";
    std::string path_c2 = base + "/vits_part_c2_" + prec + ".bmodel";

    net_a_ = load_bmodel_into(runtime_, path_a);
    if (!net_a_) { release(); return -1; }
    net_c1_ = load_bmodel_into(runtime_, path_c1);
    if (!net_c1_) { release(); return -1; }
    net_c2_ = load_bmodel_into(runtime_, path_c2);
    if (!net_c2_) { release(); return -1; }

    std::string path_c2_stream = base + "/vits_part_c2_stream_W" +
                                 std::to_string(STREAM_MEL_WINDOW) + "_R" +
                                 std::to_string(STREAM_MEL_CONTEXT) + "_" + prec +
                                 ".bmodel";
    if (file_exists(path_c2_stream)) {
        net_c2_stream_ = load_bmodel_into(runtime_, path_c2_stream);
        if (!net_c2_stream_) { release(); return -1; }
    }

    initialized_ = true;
    return 0;
}

// ─── Part B: CPU MAS (Monotonic Alignment Search) ────────────────────────────
//
// Inputs:
//   dp_w    [1,1,L]      log-duration from DP
//   x_mask  [1,1,L]      text mask (1=valid, 0=pad)
//   length_scale  float  speed control
//
// Outputs:
//   attn    [1,1,T_mel,L]  alignment matrix (one-hot rows)
//   T_mel   actual mel frames
//
// Algorithm:
//   1. duration = ceil(exp(dp_w) * length_scale) * x_mask
//   2. T_mel    = sum(duration[0,0,:L])
//   3. attn     = duration_to_attn(duration, T_mel, L)

static void mas_cpu(const float* dp_w,   // [1,1,L]
                    const float* x_mask, // [1,1,L]
                    int   L,
                    float length_scale,
                    int   max_t_mel,
                    // outputs
                    std::vector<float>& attn,   // [1,1,T_mel,L]
                    int& T_mel,
                    bool& truncated) {
    // Compute per-phoneme integer duration
    std::vector<int> dur(L);
    for (int i = 0; i < L; ++i) {
        float mask = x_mask[i];  // x_mask [1,1,L] row-major, index i
        float w    = std::exp(dp_w[i]) * length_scale * mask;
        if (!std::isfinite(w) || w >= max_t_mel) {
            dur[i] = max_t_mel;
        } else {
            dur[i] = std::max(0, (int)std::ceil(w));
        }
    }

    T_mel = 0;
    truncated = false;
    for (int i = 0; i < L; ++i) {
        if (dur[i] > max_t_mel - T_mel) {
            T_mel = max_t_mel;
            truncated = true;
            break;
        }
        T_mel += dur[i];
    }
    if (T_mel <= 0) T_mel = 1;

    // Build attention matrix [1,1,T_mel,L]: row t has 1 at the phoneme mapped to frame t
    attn.assign(T_mel * L, 0.0f);
    int t = 0;
    for (int i = 0; i < L && t < T_mel; ++i) {
        for (int d = 0; d < dur[i] && t < T_mel; ++d, ++t) {
            attn[t * L + i] = 1.0f;
        }
    }
}

// Compute z_p = h @ attn^T
// h:        buffer [Z, h_stride] — row stride may be L_MAX, not L
// attn:     [T_mel, L] contiguous
// out:      [Z, T_mel]
static void matmul_ht(const float* h, int h_stride,
                      const float* attn,
                      int Z, int L, int T,
                      float* out) {
    std::memset(out, 0, Z * T * sizeof(float));
    for (int z = 0; z < Z; ++z) {
        for (int t = 0; t < T; ++t) {
            float acc = 0.0f;
            const float* h_row    = h    + (size_t)z * h_stride;
            const float* attn_row = attn + (size_t)t * L;
            for (int l = 0; l < L; ++l) acc += h_row[l] * attn_row[l];
            out[z * T + t] = acc;
        }
    }
}

// ─── run ─────────────────────────────────────────────────────────────────────

TTSResult TTSInference::run(const int64_t* tokens, const int64_t* tones, int seq_len) {
    std::lock_guard<std::mutex> lock(run_mutex_);
    return run_impl(tokens, tones, seq_len);
}

TTSResult TTSInference::run_impl(const int64_t* tokens, const int64_t* tones,
                                 int seq_len,
                                 const TTSWindowStreamOptions* window_options) {
    TTSResult res{};
    if (!initialized_) { std::cerr << "[Error] Not initialized\n"; return res; }
    if (!tokens || !tones || seq_len <= 0 || seq_len > L_MAX) {
        std::cerr << "[Error] Invalid input: seq_len=" << seq_len
                  << ", expected [1, " << L_MAX << "]\n";
        return res;
    }

    const float length_scale = 1.0f;  // speed: 1.0 = normal

    // =========================================================
    // Build padded inputs for Part A
    // =========================================================
    std::vector<int64_t> x_padded(L_MAX, 0), t_padded(L_MAX, 0);
    for (int i = 0; i < seq_len && i < L_MAX; ++i) {
        x_padded[i] = tokens[i];
        t_padded[i] = tones[i];
    }
    std::vector<int64_t> x_lengths = { (int64_t)seq_len };

    // =========================================================
    // Part A: TPU  →  dp_w[1,1,L], h[1,192,L], attn_mask, x_mask
    // =========================================================
    // Allocate device inputs
    bm_tensor_t in_a[3], out_a[4];

    // BMRuntime has no BM_INT64; TPU-MLIR compiles int64 ONNX inputs as int32.
    // Cast int64 → int32 before uploading.
    auto alloc_dev_int64 = [&](bm_tensor_t& t, int64_t* data, int d0, int d1) {
        int n = d0 * std::max(d1, 1);
        std::vector<int32_t> buf(n);
        for (int i = 0; i < n; ++i) buf[i] = (int32_t)data[i];
        size_t bytes = (size_t)n * sizeof(int32_t);
        bm_malloc_device_byte(bm_handle_, &t.device_mem, bytes);
        t.dtype   = BM_INT32;
        t.st_mode = BM_STORE_1N;
        t.shape.num_dims = (d1 > 0) ? 2 : 1;
        t.shape.dims[0] = d0; if (d1 > 0) t.shape.dims[1] = d1;
        bm_memcpy_s2d(bm_handle_, t.device_mem, buf.data());
    };

    alloc_dev_int64(in_a[0], x_padded.data(),  1, L_MAX);  // x    [1,128]
    alloc_dev_int64(in_a[1], x_lengths.data(), 1, 0);       // x_lengths [1] (scalar-ish)
    in_a[1].shape.num_dims = 1; in_a[1].shape.dims[0] = 1;
    alloc_dev_int64(in_a[2], t_padded.data(),  1, L_MAX);  // tones [1,128]

    // Output: dp_w[1,1,L], h[1,192,L], attn_mask[1,1,1,L], x_mask[1,1,L]
    const int out_a_sizes[] = { 1*1*L_MAX, 1*Z_DIM*L_MAX, 1*1*1*L_MAX, 1*1*L_MAX };
    for (int i = 0; i < 4; ++i) {
        bm_malloc_device_byte(bm_handle_, &out_a[i].device_mem,
                              out_a_sizes[i] * sizeof(float));
        out_a[i].dtype   = BM_FLOAT32;
        out_a[i].st_mode = BM_STORE_1N;
    }
    // set shapes for Part A outputs
    out_a[0].shape = {3, {1,1,L_MAX}};          // dp_w
    out_a[1].shape = {3, {1,Z_DIM,L_MAX}};      // h
    out_a[2].shape = {4, {1,1,1,L_MAX}};        // attn_mask
    out_a[3].shape = {3, {1,1,L_MAX}};          // x_mask

    auto t0 = Clock::now();
    bool ok_a = bmrt_launch_tensor_ex(runtime_, net_a_->name,
                                      in_a, 3, out_a, 4, true, false);
    if (!ok_a) { std::cerr << "[Error] Part A launch failed\n"; return res; }
    bm_thread_sync(bm_handle_);
    auto t1 = Clock::now();
    res.part_a_ms = Ms(t1 - t0).count();

    // Copy Part A outputs to host
    std::vector<float> dp_w(L_MAX), h(Z_DIM * L_MAX), x_mask_buf(L_MAX);
    bm_memcpy_d2s(bm_handle_, dp_w.data(),      out_a[0].device_mem);
    bm_memcpy_d2s(bm_handle_, h.data(),          out_a[1].device_mem);
    bm_memcpy_d2s(bm_handle_, x_mask_buf.data(), out_a[3].device_mem);

    for (int i = 0; i < 3; ++i) bm_free_device(bm_handle_, in_a[i].device_mem);
    for (int i = 0; i < 4; ++i) bm_free_device(bm_handle_, out_a[i].device_mem);

    std::cerr << "[PartA] TPU enc_p+dp=" << res.part_a_ms << "ms\n";

    // =========================================================
    // Part B: CPU MAS
    // =========================================================
    auto t2 = Clock::now();

    std::vector<float> attn_vec;
    int T_mel_actual = 0;
    bool truncated = false;
    // The decoder is compiled for T_MEL_FIXED; cap before allocating host buffers.
    mas_cpu(dp_w.data(), x_mask_buf.data(), seq_len, length_scale,
            T_MEL_FIXED, attn_vec, T_mel_actual, truncated);
    res.mel_frames = T_mel_actual;
    res.truncated = truncated;

    // z_p = h @ attn^T  ([Z_DIM,L] x [T_mel,L]^T -> [Z_DIM,T_mel])
    std::vector<float> z_p(Z_DIM * T_mel_actual);
    // h is stored as [Z_DIM, L_MAX] — stride is L_MAX, not seq_len
    matmul_ht(h.data(), L_MAX, attn_vec.data(), Z_DIM, seq_len, T_mel_actual, z_p.data());

    auto t3 = Clock::now();
    res.part_b_ms = Ms(t3 - t2).count();

    std::cerr << "[PartB] CPU MAS  T_mel=" << T_mel_actual
              << "  time=" << res.part_b_ms << "ms\n";
    if (truncated) {
        std::cerr << "[Warn] T_mel exceeded " << T_MEL_FIXED
                  << " frames; output was truncated\n";
    }

    // =========================================================
    // Part C1: TPU flow  z_p[1,192,T_MEL_FIXED] + y_mask[1,1,T_MEL_FIXED]
    //                  → z_hat[1,192,T_MEL_FIXED]
    // Part C2: TPU decoder  z_hat[1,192,T_MEL_FIXED]
    //                     → audio[1,1,T_MEL_FIXED*UPSAMPLE]
    // =========================================================
    int copy_t = std::min(T_mel_actual, T_MEL_FIXED);

    // Pad z_p [Z_DIM, T_mel_actual] → [Z_DIM, T_MEL_FIXED]
    std::vector<float> z_p_pad(Z_DIM * T_MEL_FIXED, 0.0f);
    for (int z = 0; z < Z_DIM; ++z) {
        std::copy(z_p.data() + z * T_mel_actual,
                  z_p.data() + z * T_mel_actual + copy_t,
                  z_p_pad.data() + z * T_MEL_FIXED);
    }

    // y_mask [1,1,T_MEL_FIXED]: 1 for valid frames, 0 for padding
    std::vector<float> y_mask(T_MEL_FIXED, 0.0f);
    for (int i = 0; i < copy_t; ++i) y_mask[i] = 1.0f;

    auto t4 = Clock::now();

    // ── C1: flow ──────────────────────────────────────────────
    bm_tensor_t in_c1[2], out_c1[1];

    bm_malloc_device_byte(bm_handle_, &in_c1[0].device_mem,
                          Z_DIM * T_MEL_FIXED * sizeof(float));
    in_c1[0].dtype   = BM_FLOAT32;
    in_c1[0].st_mode = BM_STORE_1N;
    in_c1[0].shape   = {3, {1, Z_DIM, T_MEL_FIXED}};
    bm_memcpy_s2d(bm_handle_, in_c1[0].device_mem, z_p_pad.data());

    bm_malloc_device_byte(bm_handle_, &in_c1[1].device_mem,
                          T_MEL_FIXED * sizeof(float));
    in_c1[1].dtype   = BM_FLOAT32;
    in_c1[1].st_mode = BM_STORE_1N;
    in_c1[1].shape   = {3, {1, 1, T_MEL_FIXED}};
    bm_memcpy_s2d(bm_handle_, in_c1[1].device_mem, y_mask.data());

    bm_malloc_device_byte(bm_handle_, &out_c1[0].device_mem,
                          Z_DIM * T_MEL_FIXED * sizeof(float));
    out_c1[0].dtype   = BM_FLOAT32;
    out_c1[0].st_mode = BM_STORE_1N;
    out_c1[0].shape   = {3, {1, Z_DIM, T_MEL_FIXED}};

    bool ok_c1 = bmrt_launch_tensor_ex(runtime_, net_c1_->name,
                                       in_c1, 2, out_c1, 1, true, false);
    if (!ok_c1) {
        bm_free_device(bm_handle_, in_c1[0].device_mem);
        bm_free_device(bm_handle_, in_c1[1].device_mem);
        bm_free_device(bm_handle_, out_c1[0].device_mem);
        std::cerr << "[Error] Part C1 (flow) launch failed\n";
        return res;
    }
    bm_thread_sync(bm_handle_);

    bm_free_device(bm_handle_, in_c1[0].device_mem);
    bm_free_device(bm_handle_, in_c1[1].device_mem);

    // ── C2: decoder ───────────────────────────────────────────
    std::vector<float> audio_full;
    if (window_options != nullptr) {
        if (net_c2_stream_ == nullptr) {
            bm_free_device(bm_handle_, out_c1[0].device_mem);
            std::cerr << "[Error] C2 stream bmodel is not loaded\n";
            return res;
        }

        std::vector<float> z_hat(Z_DIM * T_MEL_FIXED);
        if (bm_memcpy_d2s(bm_handle_, z_hat.data(), out_c1[0].device_mem) != BM_SUCCESS) {
            bm_free_device(bm_handle_, out_c1[0].device_mem);
            std::cerr << "[Error] Failed to download C1 stream input\n";
            return res;
        }

        const int input_frames = STREAM_MEL_WINDOW + 2 * STREAM_MEL_CONTEXT;
        const int step = STREAM_MEL_WINDOW - STREAM_MEL_OVERLAP;
        const int window_count = (copy_t + step - 1) / step;
        std::vector<float> pending_tail;
        bool cancelled = false;

        auto emit = [&](const std::vector<float>& chunk) {
            if (chunk.empty()) return true;
            const int64_t offset = static_cast<int64_t>(res.audio.size());
            res.audio.insert(res.audio.end(), chunk.begin(), chunk.end());
            ++res.streamed_chunks;
            if (window_options->on_audio && !window_options->on_audio(
                    chunk.data(), static_cast<int>(chunk.size()),
                    res.streamed_chunks - 1, window_count, offset)) {
                cancelled = true;
                return false;
            }
            return true;
        };

        for (int window_index = 0; window_index < window_count && !cancelled;
             ++window_index) {
            const int start = window_index * step;
            const int valid_frames = std::min(STREAM_MEL_WINDOW, copy_t - start);
            std::vector<float> input(Z_DIM * input_frames, 0.0f);
            const int source_begin = std::max(0, start - STREAM_MEL_CONTEXT);
            const int source_end = std::min(copy_t, start + STREAM_MEL_WINDOW +
                                                     STREAM_MEL_CONTEXT);
            const int destination_begin = source_begin - (start - STREAM_MEL_CONTEXT);
            for (int z = 0; z < Z_DIM; ++z) {
                std::copy(z_hat.data() + z * T_MEL_FIXED + source_begin,
                          z_hat.data() + z * T_MEL_FIXED + source_end,
                          input.data() + z * input_frames + destination_begin);
            }

            bm_tensor_t stream_input{}, stream_output{};
            const size_t input_bytes = input.size() * sizeof(float);
            const size_t output_bytes = static_cast<size_t>(input_frames) *
                                         UPSAMPLE * sizeof(float);
            if (bm_malloc_device_byte(bm_handle_, &stream_input.device_mem,
                                      input_bytes) != BM_SUCCESS ||
                bm_malloc_device_byte(bm_handle_, &stream_output.device_mem,
                                      output_bytes) != BM_SUCCESS) {
                if (stream_input.device_mem.u.device.device_addr != 0) {
                    bm_free_device(bm_handle_, stream_input.device_mem);
                }
                bm_free_device(bm_handle_, out_c1[0].device_mem);
                std::cerr << "[Error] Failed to allocate C2 stream tensors\n";
                return res;
            }
            stream_input.dtype = BM_FLOAT32;
            stream_input.st_mode = BM_STORE_1N;
            stream_input.shape = {3, {1, Z_DIM, input_frames}};
            stream_output.dtype = BM_FLOAT32;
            stream_output.st_mode = BM_STORE_1N;
            stream_output.shape = {3, {1, 1, input_frames * UPSAMPLE}};

            bool ok = bm_memcpy_s2d(bm_handle_, stream_input.device_mem,
                                    input.data()) == BM_SUCCESS;
            bm_tensor_t stream_inputs[] = {stream_input};
            bm_tensor_t stream_outputs[] = {stream_output};
            if (ok) {
                ok = bmrt_launch_tensor_ex(runtime_, net_c2_stream_->name,
                                           stream_inputs, 1, stream_outputs, 1,
                                           true, false) &&
                     bm_thread_sync(bm_handle_) == BM_SUCCESS;
            }
            std::vector<float> window_audio(static_cast<size_t>(valid_frames) *
                                             UPSAMPLE);
            if (ok) {
                std::vector<float> output(static_cast<size_t>(input_frames) *
                                          UPSAMPLE);
                ok = bm_memcpy_d2s(bm_handle_, output.data(),
                                   stream_output.device_mem) == BM_SUCCESS;
                if (ok) {
                    const size_t begin = static_cast<size_t>(STREAM_MEL_CONTEXT) *
                                          UPSAMPLE;
                    const size_t end = begin + window_audio.size();
                    if (!std::all_of(output.begin() + begin, output.begin() + end,
                                     [](float value) { return std::isfinite(value); })) {
                        ok = false;
                    } else {
                        std::copy(output.begin() + begin,
                                  output.begin() + end,
                                  window_audio.begin());
                    }
                }
            }
            bm_free_device(bm_handle_, stream_input.device_mem);
            bm_free_device(bm_handle_, stream_output.device_mem);
            if (!ok) {
                bm_free_device(bm_handle_, out_c1[0].device_mem);
                std::cerr << "[Error] C2 stream window failed\n";
                return res;
            }

            const bool last = window_index + 1 == window_count;
            if (window_index == 0 && last) {
                emit(window_audio);
                continue;
            }
            if (window_index == 0) {
                const size_t hold = std::min(window_audio.size(),
                                             static_cast<size_t>(STREAM_MEL_OVERLAP) *
                                                 UPSAMPLE);
                emit(std::vector<float>(window_audio.begin(),
                                        window_audio.end() - hold));
                pending_tail.assign(window_audio.end() - hold, window_audio.end());
                continue;
            }

            const size_t max_overlap = static_cast<size_t>(STREAM_MEL_OVERLAP) *
                                       UPSAMPLE;
            const size_t overlap = std::min(std::min(pending_tail.size(),
                                                      window_audio.size()),
                                             max_overlap);
            std::vector<float> chunk;
            if (pending_tail.size() > overlap) {
                chunk.insert(chunk.end(), pending_tail.begin(),
                             pending_tail.end() - overlap);
            }
            for (size_t i = 0; i < overlap; ++i) {
                const float alpha = static_cast<float>(i + 1) / overlap;
                chunk.push_back(pending_tail[pending_tail.size() - overlap + i] *
                                    (1.0f - alpha) +
                                window_audio[i] * alpha);
            }
                const size_t hold =
                    last ? 0 : static_cast<size_t>(
                        std::min(STREAM_MEL_OVERLAP,
                                 std::max(0, valid_frames - step))) * UPSAMPLE;
                chunk.insert(chunk.end(), window_audio.begin() + overlap,
                         window_audio.end() - hold);
            if (!last) {
                pending_tail.assign(window_audio.end() - hold, window_audio.end());
            }
            emit(chunk);
        }
        res.cancelled = cancelled;
        res.n_samples = static_cast<int>(res.audio.size());
    } else {
        bm_tensor_t in_c2[1], out_c2[1];
        in_c2[0] = out_c1[0];

        const int audio_out_n = T_MEL_FIXED * UPSAMPLE;
        if (bm_malloc_device_byte(bm_handle_, &out_c2[0].device_mem,
                                  audio_out_n * sizeof(float)) != BM_SUCCESS) {
            bm_free_device(bm_handle_, out_c1[0].device_mem);
            std::cerr << "[Error] Failed to allocate C2 output\n";
            return res;
        }
        out_c2[0].dtype = BM_FLOAT32;
        out_c2[0].st_mode = BM_STORE_1N;
        out_c2[0].shape = {3, {1, 1, audio_out_n}};

        bool ok_c2 = bmrt_launch_tensor_ex(runtime_, net_c2_->name,
                                           in_c2, 1, out_c2, 1, true, false);
        if (!ok_c2 || bm_thread_sync(bm_handle_) != BM_SUCCESS) {
            bm_free_device(bm_handle_, out_c1[0].device_mem);
            bm_free_device(bm_handle_, out_c2[0].device_mem);
            std::cerr << "[Error] Part C2 (decoder) launch failed\n";
            return res;
        }

        audio_full.resize(audio_out_n);
        if (bm_memcpy_d2s(bm_handle_, audio_full.data(), out_c2[0].device_mem) !=
            BM_SUCCESS) {
            bm_free_device(bm_handle_, out_c1[0].device_mem);
            bm_free_device(bm_handle_, out_c2[0].device_mem);
            std::cerr << "[Error] Failed to download C2 audio\n";
            return res;
        }
        bm_free_device(bm_handle_, out_c2[0].device_mem);
        const int valid_n = std::min(copy_t * UPSAMPLE, audio_out_n);
        res.audio.assign(audio_full.begin(), audio_full.begin() + valid_n);
        res.n_samples = valid_n;
    }

    bm_free_device(bm_handle_, out_c1[0].device_mem);

    auto t5 = Clock::now();
    res.part_c_ms = Ms(t5 - t4).count();
    const int valid_n = static_cast<int>(res.audio.size());
    res.n_samples = valid_n;

    res.total_ms = res.part_a_ms + res.part_b_ms + res.part_c_ms;
    double audio_ms = (double)valid_n * 1000.0 / SAMPLE_RATE;
    res.rtf = audio_ms > 0 ? res.total_ms / audio_ms : 0.0;

    std::cerr << "[PartC] TPU c1(flow)+c2(dec)=" << res.part_c_ms << "ms\n";
    std::cerr << "[Timing] A=" << res.part_a_ms << "ms"
              << "  B=" << res.part_b_ms << "ms"
              << "  C=" << res.part_c_ms << "ms"
              << "  Total=" << res.total_ms << "ms"
              << "  Audio=" << audio_ms << "ms"
              << "  RTF=" << res.rtf << "\n";

    return res;
}

TTSStreamResult TTSInference::run_segments(
    const std::vector<TTSInputSegment>& segments,
    const TTSStreamOptions& options) {
    TTSStreamResult stream;
    if (segments.empty()) {
        stream.error = "no input segments";
        return stream;
    }

    std::lock_guard<std::mutex> lock(run_mutex_);
    const auto started = Clock::now();
    int64_t sample_offset = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        if (segment.tokens.empty() || segment.tokens.size() != segment.tones.size() ||
            segment.tokens.size() > static_cast<size_t>(L_MAX)) {
            stream.failed_segment = static_cast<int>(i);
            stream.error = "invalid token/tone segment";
            break;
        }

        TTSResult result = run_impl(segment.tokens.data(), segment.tones.data(),
                                    static_cast<int>(segment.tokens.size()));
        if (result.audio.empty()) {
            stream.failed_segment = static_cast<int>(i);
            stream.error = "segment inference failed";
            break;
        }
        if (options.strict_mel_limit && result.truncated) {
            stream.failed_segment = static_cast<int>(i);
            stream.error = "segment exceeds fixed mel limit";
            break;
        }

        if (stream.completed_segments == 0) {
            stream.first_chunk_ms = Ms(Clock::now() - started).count();
        }

        stream.audio.insert(stream.audio.end(), result.audio.begin(), result.audio.end());
        sample_offset += result.n_samples;
        stream.n_samples = sample_offset;
        ++stream.completed_segments;

        if (options.on_audio && !options.on_audio(
                result.audio.data(), result.n_samples, static_cast<int>(i),
                static_cast<int>(segments.size()), sample_offset - result.n_samples)) {
            stream.cancelled = true;
            break;
        }
    }

    stream.total_ms = Ms(Clock::now() - started).count();
    stream.success = !stream.cancelled && stream.failed_segment < 0 &&
                     stream.completed_segments == static_cast<int>(segments.size());
    return stream;
}

TTSWindowStreamResult TTSInference::run_windowed(
    const int64_t* tokens, const int64_t* tones, int seq_len,
    const TTSWindowStreamOptions& options) {
    TTSWindowStreamResult stream;
    if (!initialized_) {
        stream.error = "not initialized";
        return stream;
    }
    if (!tokens || !tones || seq_len <= 0 || seq_len > L_MAX) {
        stream.error = "invalid input";
        return stream;
    }
    if (net_c2_stream_ == nullptr) {
        stream.error = "C2 stream bmodel is not loaded";
        return stream;
    }

    std::lock_guard<std::mutex> lock(run_mutex_);
    const auto started = Clock::now();
    TTSWindowStreamOptions impl_options = options;
    impl_options.on_audio = [&](const float* samples, int n, int chunk_index,
                                int chunk_count, int64_t sample_offset) {
        if (stream.completed_chunks == 0) {
            stream.first_chunk_ms = Ms(Clock::now() - started).count();
        }
        bool ok = !options.on_audio || options.on_audio(
            samples, n, chunk_index, chunk_count, sample_offset);
        if (ok) ++stream.completed_chunks;
        return ok;
    };

    TTSResult result = run_impl(tokens, tones, seq_len, &impl_options);
    stream.audio = std::move(result.audio);
    stream.n_samples = stream.audio.size();
    stream.cancelled = result.cancelled;
    stream.total_ms = Ms(Clock::now() - started).count();
    if (stream.completed_chunks == 0 && !stream.audio.empty()) {
        stream.completed_chunks = result.streamed_chunks;
        stream.first_chunk_ms = stream.total_ms;
    }
    if (result.truncated && options.strict_mel_limit) {
        stream.error = "input exceeds fixed mel limit";
    }
    stream.success = !stream.cancelled && stream.error.empty() &&
                     !stream.audio.empty() &&
                     stream.audio.size() ==
                         static_cast<size_t>(result.mel_frames * UPSAMPLE);
    if (!stream.success && !stream.cancelled && stream.error.empty()) {
        stream.error = "window inference failed";
    }
    return stream;
}


void TTSInference::release() {
    if (runtime_)   { bmrt_destroy(runtime_);   runtime_   = nullptr; }
    if (bm_handle_) { bm_dev_free(bm_handle_);  bm_handle_ = nullptr; }
    net_a_ = net_c1_ = net_c2_ = net_c2_stream_ = nullptr;
    initialized_ = false;
}

}  // namespace vits_tts
