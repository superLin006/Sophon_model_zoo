#include "decoder_engine.h"
#include <bmruntime_interface.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <limits>

struct DecoderEngine::Impl {
    bm_handle_t  handle    = nullptr;
    void*        bmrt      = nullptr;
    bool         owns_hdl  = true;   // false when using shared handle
    bool         owns_bmrt = true;   // false when bmrt provided externally
    std::string  net_name;
    int          max_T    = 0;
    int          hidden   = 768;
    int          n_mels   = 100;

    // Persistent device buffers — allocated once, reused every infer() call
    bm_device_mem_t in_dm  = {};   // [1, hidden, max_T] f32
    bm_device_mem_t out_dm = {};   // [1, n_mels, max_T*2, 1] f32
    bool bufs_ready = false;

    void init(const std::string& bmodel_path);
};

void DecoderEngine::Impl::init(const std::string& bmodel_path) {
    if (!bmrt_load_bmodel(bmrt, bmodel_path.c_str()))
        throw std::runtime_error("DecoderEngine: load bmodel failed: " + bmodel_path);

    const char** names = nullptr;
    int n_nets = 0;
    bmrt_get_network_names(bmrt, &names);
    while (names && names[n_nets]) ++n_nets;
    if (n_nets == 0) {
        free(names);
        throw std::runtime_error("DecoderEngine: no network in bmodel");
    }
    net_name = names[0];
    free(names);

    const bm_net_info_t* info = bmrt_get_network_info(bmrt, net_name.c_str());
    if (!info) throw std::runtime_error("DecoderEngine: get network info failed");
    hidden = info->stages[0].input_shapes[0].dims[1];
    max_T  = info->stages[0].input_shapes[0].dims[2];
    n_mels = info->stages[0].output_shapes[0].dims[1];

    size_t in_bytes  = (size_t)hidden * max_T * sizeof(float);
    size_t out_bytes = (size_t)n_mels * max_T * 2 * sizeof(float);
    if (bm_malloc_device_byte(handle, &in_dm,  in_bytes)  != BM_SUCCESS ||
        bm_malloc_device_byte(handle, &out_dm, out_bytes) != BM_SUCCESS)
        throw std::runtime_error("DecoderEngine: pre-alloc device buffers failed");
    bufs_ready = true;
}

DecoderEngine::DecoderEngine(const std::string& bmodel_path, int tpu_id)
    : impl_(std::make_unique<Impl>()) {
    impl_->owns_hdl  = true;
    impl_->owns_bmrt = true;
    if (bm_dev_request(&impl_->handle, tpu_id) != BM_SUCCESS)
        throw std::runtime_error("DecoderEngine: bm_dev_request failed");
    impl_->bmrt = bmrt_create(impl_->handle);
    if (!impl_->bmrt) throw std::runtime_error("DecoderEngine: bmrt_create failed");
    impl_->init(bmodel_path);
}

DecoderEngine::DecoderEngine(const std::string& bmodel_path, void* bm_handle, void* /*ignored*/)
    : impl_(std::make_unique<Impl>()) {
    // Shared-handle: use external handle (for shared allocator), create own bmrt
    impl_->owns_hdl = false;
    impl_->handle   = static_cast<bm_handle_t>(bm_handle);
    impl_->bmrt     = bmrt_create(impl_->handle);
    if (!impl_->bmrt) throw std::runtime_error("DecoderEngine: bmrt_create failed");
    impl_->owns_bmrt = true;
    impl_->init(bmodel_path);
}

DecoderEngine::~DecoderEngine() {
    if (impl_->bufs_ready) {
        bm_free_device(impl_->handle, impl_->in_dm);
        bm_free_device(impl_->handle, impl_->out_dm);
    }
    if (impl_->owns_bmrt && impl_->bmrt) bmrt_destroy(impl_->bmrt);
    if (impl_->owns_hdl  && impl_->handle) bm_dev_free(impl_->handle);
}

int DecoderEngine::input_T() const { return impl_->max_T; }

std::vector<float> DecoderEngine::infer(const std::vector<uint16_t>& hiddens_f16,
                                         int hidden_size, int T) {
    const int max_T  = impl_->max_T;
    const int n_mels = impl_->n_mels;
    const int out_T  = max_T * 2;

    std::vector<float> input_buf(hidden_size * max_T, 0.0f);
    int use_T = std::min(T, max_T);
    for (int t = 0; t < use_T; ++t) {
        for (int h = 0; h < hidden_size; ++h) {
            uint16_t u16 = hiddens_f16[t * hidden_size + h];
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
            input_buf[h * max_T + t] = val;
        }
    }

    bm_tensor_t in_tensor, out_tensor;
    bm_shape_t in_shape  = {{3}, {1, (unsigned)hidden_size, (unsigned)max_T}};
    bm_shape_t out_shape = {{4}, {1, (unsigned)n_mels, (unsigned)out_T, 1}};

    bmrt_tensor_with_device(&in_tensor,  impl_->in_dm,  BM_FLOAT32, in_shape);
    bmrt_tensor_with_device(&out_tensor, impl_->out_dm, BM_FLOAT32, out_shape);

    bm_memcpy_s2d(impl_->handle, impl_->in_dm, (void*)input_buf.data());

    bool ok = bmrt_launch_tensor_ex(impl_->bmrt, impl_->net_name.c_str(),
                                     &in_tensor, 1, &out_tensor, 1, true, false);
    bm_thread_sync(impl_->handle);

    std::vector<float> mel_out;
    if (ok) {
        mel_out.resize(n_mels * out_T);
        bm_memcpy_d2s(impl_->handle, mel_out.data(), impl_->out_dm);
    }

    return mel_out;
}
