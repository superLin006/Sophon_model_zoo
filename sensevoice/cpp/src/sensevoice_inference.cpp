#include "sensevoice_inference.h"

#include <cstring>
#include <iostream>
#include <algorithm>

namespace sensevoice {

SenseVoiceInference::SenseVoiceInference() = default;

SenseVoiceInference::~SenseVoiceInference() {
    release();
}

int SenseVoiceInference::init(const char* model_dir, const char* precision) {
    std::string base(model_dir);
    std::string prec(precision);

    // Load bmodel
    std::string bmodel_path = base + "/sensevoice_small_" + prec + ".bmodel";

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

    const char* net_name = "sensevoice_small";
    net_info_ = bmrt_get_network_info(runtime_, net_name);
    if (!net_info_) {
        // Try fallback: use first network
        const char** names = nullptr;
        int num_nets = 0;
        bmrt_get_network_names(runtime_, &names);
        // count names (null-terminated array)
        while (names && names[num_nets]) ++num_nets;
        if (num_nets > 0) {
            net_info_ = bmrt_get_network_info(runtime_, names[0]);
            std::cout << "[Init] Using network: " << names[0] << "\n";
        }
        free(names);
    }

    if (!net_info_) {
        std::cerr << "[Error] Cannot get network info\n";
        return -1;
    }

    // Load tokens
    std::string tokens_path = base + "/tokens.txt";
    if (!tokenizer_.Load(tokens_path)) {
        std::cerr << "[Error] Cannot load tokens: " << tokens_path << "\n";
        return -1;
    }

    // Init audio frontend
    AudioConfig audio_cfg;
    frontend_ = new AudioFrontend(audio_cfg);

    initialized_ = true;
    std::cout << "[Init] SenseVoice loaded (" << prec << ") from " << base << "\n";
    return 0;
}

std::vector<float> SenseVoiceInference::pad_or_truncate(const std::vector<float>& lfr,
                                                         int32_t num_frames) const {
    std::vector<float> out(SV_FIXED_FRAMES * SV_INPUT_DIM, 0.0f);
    int32_t copy_frames = std::min(num_frames, SV_FIXED_FRAMES);
    std::copy(lfr.begin(), lfr.begin() + copy_frames * SV_INPUT_DIM, out.begin());
    return out;
}

RecognitionResult SenseVoiceInference::run(const char* audio_file) {
    RecognitionResult empty;
    if (!initialized_) return empty;

    // Load audio
    std::vector<float> samples;
    int32_t sample_rate = 0;
    if (!LoadWavFile(audio_file, &samples, &sample_rate)) {
        std::cerr << "[Error] Cannot load audio: " << audio_file << "\n";
        return empty;
    }

    if (sample_rate != 16000) {
        std::cerr << "[Error] Expected 16kHz audio, got " << sample_rate << "Hz\n";
        return empty;
    }

    // Feature extraction: audio → Fbank80 → LFR → [num_frames, 560]
    int32_t num_lfr_frames = 0;
    std::vector<float> lfr = frontend_->Process(samples, &num_lfr_frames);
    if (lfr.empty()) {
        std::cerr << "[Error] Feature extraction failed\n";
        return empty;
    }

    std::cout << "[Run] LFR frames: " << num_lfr_frames
              << " (model expects " << SV_FIXED_FRAMES << ")\n";

    // Pad/truncate to fixed size
    std::vector<float> input = pad_or_truncate(lfr, num_lfr_frames);

    // Run inference
    // Input tensor: [1, 166, 560]
    bm_tensor_t input_tensor;
    bm_shape_t in_shape;
    in_shape.num_dims = 3;
    in_shape.dims[0] = 1;
    in_shape.dims[1] = SV_FIXED_FRAMES;
    in_shape.dims[2] = SV_INPUT_DIM;

    if (bm_malloc_device_byte(bm_handle_, &input_tensor.device_mem,
                              input.size() * sizeof(float)) != BM_SUCCESS) {
        std::cerr << "[Error] bm_malloc_device_byte (input) failed\n";
        return empty;
    }
    input_tensor.dtype = BM_FLOAT32;
    input_tensor.shape = in_shape;
    input_tensor.st_mode = BM_STORE_1N;

    bm_memcpy_s2d(bm_handle_, input_tensor.device_mem, input.data());

    // Output tensor: [1, 170, 25055]
    bm_tensor_t output_tensor;
    bm_shape_t out_shape;
    out_shape.num_dims = 3;
    out_shape.dims[0] = 1;
    out_shape.dims[1] = SV_OUTPUT_FRAMES;
    out_shape.dims[2] = SV_VOCAB_SIZE;

    size_t out_size = SV_OUTPUT_FRAMES * SV_VOCAB_SIZE * sizeof(float);
    if (bm_malloc_device_byte(bm_handle_, &output_tensor.device_mem, out_size) != BM_SUCCESS) {
        std::cerr << "[Error] bm_malloc_device_byte (output) failed\n";
        bm_free_device(bm_handle_, input_tensor.device_mem);
        return empty;
    }
    output_tensor.dtype = BM_FLOAT32;
    output_tensor.shape = out_shape;
    output_tensor.st_mode = BM_STORE_1N;

    bm_tensor_t* inputs  = &input_tensor;
    bm_tensor_t* outputs = &output_tensor;

    bool ok = bmrt_launch_tensor_ex(runtime_, net_info_->name,
                                    inputs,  1,
                                    outputs, 1,
                                    true, false);
    if (!ok) {
        std::cerr << "[Error] bmrt_launch_tensor_ex failed\n";
        bm_free_device(bm_handle_, input_tensor.device_mem);
        bm_free_device(bm_handle_, output_tensor.device_mem);
        return empty;
    }
    bm_thread_sync(bm_handle_);

    // Copy output back to host
    std::vector<float> logits(SV_OUTPUT_FRAMES * SV_VOCAB_SIZE);
    bm_memcpy_d2s(bm_handle_, logits.data(), output_tensor.device_mem);

    bm_free_device(bm_handle_, input_tensor.device_mem);
    bm_free_device(bm_handle_, output_tensor.device_mem);

    // CTC decode: skip first 4 prompt frames
    RecognitionResult result = tokenizer_.Decode(logits.data(),
                                                 SV_OUTPUT_FRAMES,
                                                 SV_VOCAB_SIZE);
    return result;
}

void SenseVoiceInference::release() {
    if (frontend_) {
        delete frontend_;
        frontend_ = nullptr;
    }
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

}  // namespace sensevoice
