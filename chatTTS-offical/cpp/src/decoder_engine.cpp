#include "decoder_engine.h"
#include <bmruntime_interface.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <limits>

struct DecoderEngine::Impl {
    bm_handle_t  handle   = nullptr;
    void*        bmrt     = nullptr;
    std::string  net_name;
    int          max_T    = 0;   // max time steps bmodel accepts
    int          hidden   = 768;
    int          n_mels   = 100;
};

DecoderEngine::DecoderEngine(const std::string& bmodel_path, int tpu_id)
    : impl_(std::make_unique<Impl>()) {

    if (bm_dev_request(&impl_->handle, tpu_id) != BM_SUCCESS)
        throw std::runtime_error("DecoderEngine: bm_dev_request failed");

    impl_->bmrt = bmrt_create(impl_->handle);
    if (!impl_->bmrt)
        throw std::runtime_error("DecoderEngine: bmrt_create failed");

    if (!bmrt_load_bmodel(impl_->bmrt, bmodel_path.c_str()))
        throw std::runtime_error("DecoderEngine: load bmodel failed: " + bmodel_path);

    // Get first network name
    const char** names = nullptr;
    int n_nets = 0;
    bmrt_get_network_names(impl_->bmrt, &names);
    // find count
    while (names && names[n_nets]) ++n_nets;
    if (n_nets == 0) throw std::runtime_error("DecoderEngine: no network in bmodel");
    impl_->net_name = names[0];
    free(names);

    // Query input shape to get max_T
    const bm_net_info_t* info = bmrt_get_network_info(impl_->bmrt, impl_->net_name.c_str());
    if (!info) throw std::runtime_error("DecoderEngine: get network info failed");
    // shapes live in stages[0]
    impl_->hidden = info->stages[0].input_shapes[0].dims[1];
    impl_->max_T  = info->stages[0].input_shapes[0].dims[2];
    impl_->n_mels = info->stages[0].output_shapes[0].dims[1];
}

DecoderEngine::~DecoderEngine() {
    if (impl_->bmrt)   bmrt_destroy(impl_->bmrt);
    if (impl_->handle) bm_dev_free(impl_->handle);
}

int DecoderEngine::input_T() const { return impl_->max_T; }

std::vector<float> DecoderEngine::infer(const std::vector<uint16_t>& hiddens_f16,
                                         int hidden_size, int T) {
    const int max_T   = impl_->max_T;
    const int n_mels  = impl_->n_mels;
    const int out_T   = max_T * 2;   // decoder upsamples 2x

    // Build input float32 tensor [1, hidden_size, max_T]
    // hiddens_f16 layout: T vectors of hidden_size (row-major)
    // We need layout [1, hidden_size, T] → transpose + pad to max_T
    std::vector<float> input_buf(hidden_size * max_T, 0.0f);
    int use_T = std::min(T, max_T);
    for (int t = 0; t < use_T; ++t) {
        for (int h = 0; h < hidden_size; ++h) {
            // f16 → f32
            uint16_t u16 = hiddens_f16[t * hidden_size + h];
            // IEEE 754 half → float
            uint32_t sign     = (u16 >> 15) & 1;
            uint32_t exponent = (u16 >> 10) & 0x1F;
            uint32_t mantissa = u16 & 0x3FF;
            float val;
            if (exponent == 0) {
                val = ldexp((float)mantissa, -24);
            } else if (exponent == 31) {
                val = mantissa ? std::numeric_limits<float>::quiet_NaN()
                               : std::numeric_limits<float>::infinity();
            } else {
                val = ldexp((float)(mantissa | 0x400), (int)exponent - 25);
            }
            if (sign) val = -val;
            // [h, t] in [hidden_size, max_T]
            input_buf[h * max_T + t] = val;
        }
    }

    // Setup bmruntime tensors
    bm_tensor_t in_tensor, out_tensor;
    bm_shape_t in_shape  = {{3}, {1, (unsigned)hidden_size, (unsigned)max_T}};
    bm_shape_t out_shape = {{4}, {1, (unsigned)n_mels, (unsigned)out_T, 1}};

    bmrt_tensor(&in_tensor,  impl_->bmrt, BM_FLOAT32, in_shape);
    bmrt_tensor(&out_tensor, impl_->bmrt, BM_FLOAT32, out_shape);

    bm_memcpy_s2d(impl_->handle, in_tensor.device_mem,
                  (void*)input_buf.data());

    bool ok = bmrt_launch_tensor(impl_->bmrt, impl_->net_name.c_str(),
                                  &in_tensor, 1, &out_tensor, 1);
    bm_thread_sync(impl_->handle);

    std::vector<float> mel_out;
    if (ok) {
        mel_out.resize(n_mels * out_T);
        bm_memcpy_d2s(impl_->handle, mel_out.data(), out_tensor.device_mem);
    }

    bm_free_device(impl_->handle, in_tensor.device_mem);
    bm_free_device(impl_->handle, out_tensor.device_mem);

    return mel_out;
}
