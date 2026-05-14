#include "tts_inference.h"
#include "wav_writer.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// OrtApi helper macros
#define ORT_CHECK(api, expr) do { \
    OrtStatus* _s = (expr); \
    if (_s) { \
        const char* _msg = (api)->GetErrorMessage(_s); \
        std::cerr << "[ORT Error] " << _msg << "  at " << __FILE__ << ":" << __LINE__ << "\n"; \
        (api)->ReleaseStatus(_s); \
        return -1; \
    } \
} while(0)

namespace vits_tts {

TTSInference::TTSInference() = default;

TTSInference::~TTSInference() {
    release();
}

int TTSInference::init(const char* model_dir, const char* precision) {
    std::string base(model_dir);
    std::string prec(precision);

    // =========================================================
    // 1. Init OnnxRuntime (Part 1)
    // =========================================================
    ort_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_api_) {
        std::cerr << "[Error] OrtGetApiBase failed\n";
        return -1;
    }

    ORT_CHECK(ort_api_, ort_api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vits_tts", &ort_env_));
    ORT_CHECK(ort_api_, ort_api_->CreateSessionOptions(&ort_opts_));
    ORT_CHECK(ort_api_, ort_api_->SetIntraOpNumThreads(ort_opts_, 4));
    ORT_CHECK(ort_api_, ort_api_->SetSessionGraphOptimizationLevel(ort_opts_, ORT_ENABLE_ALL));

    std::string onnx_path = base + "/model_part1.onnx";
    ORT_CHECK(ort_api_, ort_api_->CreateSession(ort_env_, onnx_path.c_str(), ort_opts_, &ort_session_));

    ORT_CHECK(ort_api_, ort_api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_mem_));

    std::cout << "[Init] Loaded ONNX model: " << onnx_path << "\n";

    // =========================================================
    // 2. Init BMRuntime (Part 2)
    // =========================================================
    std::string bmodel_path = base + "/vits-melo-tts-zh_en_decoder_T256_" + prec + ".bmodel";

    if (bm_dev_request(&bm_handle_, 0) != BM_SUCCESS) {
        std::cerr << "[Error] bm_dev_request failed\n";
        return -1;
    }

    runtime_ = bmrt_create(bm_handle_);
    if (!runtime_) {
        std::cerr << "[Error] bmrt_create failed\n";
        return -1;
    }

    if (!bmrt_load_bmodel(runtime_, bmodel_path.c_str())) {
        std::cerr << "[Error] bmrt_load_bmodel failed: " << bmodel_path << "\n";
        return -1;
    }

    // Get network info (use first network)
    const char** net_names = nullptr;
    bmrt_get_network_names(runtime_, &net_names);
    if (!net_names || !net_names[0]) {
        std::cerr << "[Error] No networks in bmodel\n";
        return -1;
    }
    net_info_ = bmrt_get_network_info(runtime_, net_names[0]);
    std::cout << "[Init] Loaded bmodel: " << bmodel_path << "  net=" << net_names[0] << "\n";
    free(net_names);

    if (!net_info_) {
        std::cerr << "[Error] Cannot get network info\n";
        return -1;
    }

    initialized_ = true;
    return 0;
}

// Helper: read OrtValue as float pointer + total element count
static const float* get_ort_float_ptr(const OrtApi* api, OrtValue* val, size_t* count) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    api->GetTensorTypeAndShape(val, &info);
    api->GetTensorShapeElementCount(info, count);
    api->ReleaseTensorTypeAndShapeInfo(info);
    float* ptr = nullptr;
    api->GetTensorMutableData(val, (void**)&ptr);
    return ptr;
}

TTSResult TTSInference::run(const int64_t* tokens, const int64_t* tones, int seq_len) {
    TTSResult res{};
    res.n_samples = 0;

    if (!initialized_) {
        std::cerr << "[Error] Not initialized\n";
        return res;
    }

    // =========================================================
    // Build padded inputs for ONNX
    // =========================================================
    int64_t x_padded   [L_MAX] = {0};
    int64_t tone_padded[L_MAX] = {0};
    for (int i = 0; i < seq_len && i < L_MAX; ++i) {
        x_padded[i]    = tokens[i];
        tone_padded[i] = tones[i];
    }
    int64_t x_lengths[1]      = { (int64_t)seq_len };
    int64_t sid_val  [1]      = { 1 };    // MUST be 1 (speaker id)
    float   noise_scale[1]    = { 0.667f };
    float   length_scale[1]   = { 1.0f  };
    float   noise_scale_w[1]  = { 0.8f  };

    // Input shapes
    int64_t shape_1L[2]  = { 1, L_MAX };
    int64_t shape_1 [1]  = { 1 };

    // Create OrtValues
    OrtValue* ov_x            = nullptr;
    OrtValue* ov_x_lengths    = nullptr;
    OrtValue* ov_tones        = nullptr;
    OrtValue* ov_sid          = nullptr;
    OrtValue* ov_noise_scale  = nullptr;
    OrtValue* ov_length_scale = nullptr;
    OrtValue* ov_noise_scale_w= nullptr;

    auto create_int64 = [&](int64_t* data, int64_t* shape, size_t ndim, OrtValue** out) -> int {
        OrtStatus* s = ort_api_->CreateTensorWithDataAsOrtValue(
            ort_mem_, data, ndim > 1 ? shape[0]*shape[1]*8 : shape[0]*8,
            shape, ndim, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, out);
        if (s) {
            std::cerr << "[ORT Error] CreateTensor(int64) " << ort_api_->GetErrorMessage(s) << "\n";
            ort_api_->ReleaseStatus(s);
            return -1;
        }
        return 0;
    };
    auto create_float = [&](float* data, int64_t* shape, size_t ndim, OrtValue** out) -> int {
        OrtStatus* s = ort_api_->CreateTensorWithDataAsOrtValue(
            ort_mem_, data, ndim > 1 ? shape[0]*shape[1]*4 : shape[0]*4,
            shape, ndim, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, out);
        if (s) {
            std::cerr << "[ORT Error] CreateTensor(float) " << ort_api_->GetErrorMessage(s) << "\n";
            ort_api_->ReleaseStatus(s);
            return -1;
        }
        return 0;
    };

    if (create_int64(x_padded,      shape_1L, 2, &ov_x)            != 0) return res;
    if (create_int64(x_lengths,     shape_1,  1, &ov_x_lengths)    != 0) return res;
    if (create_int64(tone_padded,   shape_1L, 2, &ov_tones)        != 0) return res;
    if (create_int64(sid_val,       shape_1,  1, &ov_sid)          != 0) return res;
    if (create_float(noise_scale,   shape_1,  1, &ov_noise_scale)  != 0) return res;
    if (create_float(length_scale,  shape_1,  1, &ov_length_scale) != 0) return res;
    if (create_float(noise_scale_w, shape_1,  1, &ov_noise_scale_w)!= 0) return res;

    // Input/output name arrays
    const char* input_names[]  = {
        "x", "x_lengths", "tones", "sid",
        "noise_scale", "length_scale", "noise_scale_w"
    };
    const char* output_names[] = {
        "y",
        "/Mul_10_output_0",
        "/Unsqueeze_6_output_0"
    };

    OrtValue* inputs[] = {
        ov_x, ov_x_lengths, ov_tones, ov_sid,
        ov_noise_scale, ov_length_scale, ov_noise_scale_w
    };
    OrtValue* outputs[3] = { nullptr, nullptr, nullptr };

    // =========================================================
    // Part 1: ONNX inference
    // =========================================================
    auto t0 = Clock::now();
    OrtStatus* run_status = ort_api_->Run(
        ort_session_, nullptr,
        input_names,  inputs,  7,
        output_names, 3,       outputs
    );
    auto t1 = Clock::now();
    res.part1_ms = Ms(t1 - t0).count();

    // Release input OrtValues
    for (auto* v : inputs) ort_api_->ReleaseValue(v);

    if (run_status) {
        std::cerr << "[Error] OrtSession::Run failed: "
                  << ort_api_->GetErrorMessage(run_status) << "\n";
        ort_api_->ReleaseStatus(run_status);
        return res;
    }

    // Extract z_hat and g_emb
    size_t z_hat_count = 0, g_emb_count = 0;
    const float* z_hat_data = get_ort_float_ptr(ort_api_, outputs[1], &z_hat_count);
    const float* g_emb_data = get_ort_float_ptr(ort_api_, outputs[2], &g_emb_count);

    // Get T_mel_actual from z_hat shape
    OrtTensorTypeAndShapeInfo* z_info = nullptr;
    ort_api_->GetTensorTypeAndShape(outputs[1], &z_info);
    size_t z_ndim = 0;
    ort_api_->GetDimensionsCount(z_info, &z_ndim);
    std::vector<int64_t> z_dims(z_ndim);
    ort_api_->GetDimensions(z_info, z_dims.data(), z_ndim);
    ort_api_->ReleaseTensorTypeAndShapeInfo(z_info);

    int T_mel_actual = (int)z_dims[2];  // [1, 192, T_mel]
    std::cout << "[Part1] z_hat=" << z_dims[0] << "x" << z_dims[1] << "x" << z_dims[2]
              << "  g_emb_count=" << g_emb_count
              << "  part1=" << res.part1_ms << "ms\n";

    // =========================================================
    // Pad z_hat to [1, 192, T_MEL_FIXED]
    // =========================================================
    int z_hat_padded_size = 1 * Z_DIM * T_MEL_FIXED;
    std::vector<float> z_hat_padded(z_hat_padded_size, 0.0f);
    int copy_frames = std::min(T_mel_actual, T_MEL_FIXED);
    // z_hat layout: [1, 192, T_mel], row-major → copy each channel slice
    for (int c = 0; c < Z_DIM; ++c) {
        const float* src = z_hat_data + c * T_mel_actual;
        float*       dst = z_hat_padded.data() + c * T_MEL_FIXED;
        std::copy(src, src + copy_frames, dst);
    }

    // g_emb is already [1, 256, 1] — just copy
    std::vector<float> g_emb_buf(g_emb_count);
    std::copy(g_emb_data, g_emb_data + g_emb_count, g_emb_buf.begin());

    ort_api_->ReleaseValue(outputs[0]);
    ort_api_->ReleaseValue(outputs[1]);
    ort_api_->ReleaseValue(outputs[2]);

    // =========================================================
    // Part 2: BMRuntime decoder inference
    // =========================================================
    // Allocate device tensors
    bm_tensor_t in_tensor[2];
    bm_tensor_t out_tensor[1];

    // Input 0: z_hat_padded [1, 192, 256]
    {
        bm_shape_t sh;
        sh.num_dims = 3;
        sh.dims[0] = 1; sh.dims[1] = Z_DIM; sh.dims[2] = T_MEL_FIXED;
        if (bm_malloc_device_byte(bm_handle_, &in_tensor[0].device_mem,
                                  z_hat_padded_size * sizeof(float)) != BM_SUCCESS) {
            std::cerr << "[Error] bm_malloc z_hat failed\n";
            return res;
        }
        in_tensor[0].dtype   = BM_FLOAT32;
        in_tensor[0].shape   = sh;
        in_tensor[0].st_mode = BM_STORE_1N;
        bm_memcpy_s2d(bm_handle_, in_tensor[0].device_mem, z_hat_padded.data());
    }

    // Input 1: g_emb [1, 256, 1]
    {
        bm_shape_t sh;
        sh.num_dims = 3;
        sh.dims[0] = 1; sh.dims[1] = G_DIM; sh.dims[2] = 1;
        if (bm_malloc_device_byte(bm_handle_, &in_tensor[1].device_mem,
                                  g_emb_count * sizeof(float)) != BM_SUCCESS) {
            std::cerr << "[Error] bm_malloc g_emb failed\n";
            bm_free_device(bm_handle_, in_tensor[0].device_mem);
            return res;
        }
        in_tensor[1].dtype   = BM_FLOAT32;
        in_tensor[1].shape   = sh;
        in_tensor[1].st_mode = BM_STORE_1N;
        bm_memcpy_s2d(bm_handle_, in_tensor[1].device_mem, g_emb_buf.data());
    }

    // Output: audio [1, 1, T_MEL_FIXED * UPSAMPLE]
    int audio_out_size = 1 * 1 * T_MEL_FIXED * UPSAMPLE;
    {
        bm_shape_t sh;
        sh.num_dims = 3;
        sh.dims[0] = 1; sh.dims[1] = 1; sh.dims[2] = T_MEL_FIXED * UPSAMPLE;
        if (bm_malloc_device_byte(bm_handle_, &out_tensor[0].device_mem,
                                  audio_out_size * sizeof(float)) != BM_SUCCESS) {
            std::cerr << "[Error] bm_malloc audio_out failed\n";
            bm_free_device(bm_handle_, in_tensor[0].device_mem);
            bm_free_device(bm_handle_, in_tensor[1].device_mem);
            return res;
        }
        out_tensor[0].dtype   = BM_FLOAT32;
        out_tensor[0].shape   = sh;
        out_tensor[0].st_mode = BM_STORE_1N;
    }

    auto t2 = Clock::now();
    bool ok = bmrt_launch_tensor_ex(runtime_, net_info_->name,
                                    in_tensor,  2,
                                    out_tensor, 1,
                                    true, false);
    if (!ok) {
        std::cerr << "[Error] bmrt_launch_tensor_ex failed\n";
        bm_free_device(bm_handle_, in_tensor[0].device_mem);
        bm_free_device(bm_handle_, in_tensor[1].device_mem);
        bm_free_device(bm_handle_, out_tensor[0].device_mem);
        return res;
    }
    bm_thread_sync(bm_handle_);
    auto t3 = Clock::now();
    res.part2_ms = Ms(t3 - t2).count();

    // Copy output back to host
    std::vector<float> audio_full(audio_out_size);
    bm_memcpy_d2s(bm_handle_, audio_full.data(), out_tensor[0].device_mem);

    bm_free_device(bm_handle_, in_tensor[0].device_mem);
    bm_free_device(bm_handle_, in_tensor[1].device_mem);
    bm_free_device(bm_handle_, out_tensor[0].device_mem);

    // =========================================================
    // Trim to valid samples: T_mel_actual * UPSAMPLE
    // =========================================================
    int valid_samples = T_mel_actual * UPSAMPLE;
    if (valid_samples > audio_out_size) valid_samples = audio_out_size;

    res.audio.assign(audio_full.begin(), audio_full.begin() + valid_samples);
    res.n_samples = valid_samples;

    // Timing
    res.total_ms = res.part1_ms + res.part2_ms;
    double audio_duration_ms = (double)valid_samples * 1000.0 / SAMPLE_RATE;
    res.rtf = res.total_ms / audio_duration_ms;

    std::cout << "[Part2] bmodel=" << res.part2_ms << "ms\n";
    std::cout << "[Timing] Part1(ONNX)=" << res.part1_ms << "ms"
              << "  Part2(TPU)=" << res.part2_ms << "ms"
              << "  Total=" << res.total_ms << "ms"
              << "  Audio=" << audio_duration_ms << "ms"
              << "  RTF=" << res.rtf << "\n";

    return res;
}

void TTSInference::release() {
    if (ort_mem_)     { ort_api_->ReleaseMemoryInfo(ort_mem_);       ort_mem_     = nullptr; }
    if (ort_session_) { ort_api_->ReleaseSession(ort_session_);      ort_session_ = nullptr; }
    if (ort_opts_)    { ort_api_->ReleaseSessionOptions(ort_opts_);  ort_opts_    = nullptr; }
    if (ort_env_)     { ort_api_->ReleaseEnv(ort_env_);              ort_env_     = nullptr; }

    if (runtime_) {
        bmrt_destroy(runtime_);
        runtime_ = nullptr;
    }
    if (bm_handle_) {
        bm_dev_free(bm_handle_);
        bm_handle_ = nullptr;
    }
    initialized_ = false;
}

}  // namespace vits_tts
