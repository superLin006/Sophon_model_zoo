#include "vocos_engine.h"
#include <bmruntime_interface.h>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <cmath>

struct VocosEngine::Impl {
    bm_handle_t handle   = nullptr;
    void*       bmrt     = nullptr;
    std::string net_name;
    int         max_T    = 0;
    int         n_mels   = 100;
    int         n_bins   = 513;  // n_fft/2+1 = 513
};

VocosEngine::VocosEngine(const std::string& bmodel_path, int tpu_id)
    : impl_(std::make_unique<Impl>()) {

    if (bm_dev_request(&impl_->handle, tpu_id) != BM_SUCCESS)
        throw std::runtime_error("VocosEngine: bm_dev_request failed");

    impl_->bmrt = bmrt_create(impl_->handle);
    if (!impl_->bmrt)
        throw std::runtime_error("VocosEngine: bmrt_create failed");

    if (!bmrt_load_bmodel(impl_->bmrt, bmodel_path.c_str()))
        throw std::runtime_error("VocosEngine: load bmodel failed: " + bmodel_path);

    const char** names = nullptr;
    bmrt_get_network_names(impl_->bmrt, &names);
    if (!names || !names[0]) throw std::runtime_error("VocosEngine: no network");
    impl_->net_name = names[0];
    free(names);

    const bm_net_info_t* info = bmrt_get_network_info(impl_->bmrt, impl_->net_name.c_str());
    if (!info) throw std::runtime_error("VocosEngine: get network info failed");
    // input shape [1, n_mels, T] — live in stages[0]
    impl_->n_mels = info->stages[0].input_shapes[0].dims[1];
    impl_->max_T  = info->stages[0].input_shapes[0].dims[2];
    // output[0] shape [1, n_bins, T]
    impl_->n_bins = info->stages[0].output_shapes[0].dims[1];
}

VocosEngine::~VocosEngine() {
    if (impl_->bmrt)   bmrt_destroy(impl_->bmrt);
    if (impl_->handle) bm_dev_free(impl_->handle);
}

int VocosEngine::input_T() const { return impl_->max_T; }

VocosOutput VocosEngine::infer(const std::vector<float>& mel, int n_mels, int T) {
    const int max_T  = impl_->max_T;
    const int n_bins = impl_->n_bins;

    // Pad mel to [1, n_mels, max_T]
    std::vector<float> in_buf(n_mels * max_T, 0.0f);
    int use_T = std::min(T, max_T);
    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < use_T; ++t) {
            in_buf[m * max_T + t] = mel[m * T + t];
        }
    }

    bm_tensor_t in_tensor;
    bm_shape_t in_shape = {{3}, {1, (unsigned)n_mels, (unsigned)max_T}};
    bmrt_tensor(&in_tensor, impl_->bmrt, BM_FLOAT32, in_shape);
    bm_memcpy_s2d(impl_->handle, in_tensor.device_mem, (void*)in_buf.data());

    // 3 outputs: mag, x, y  each [1, n_bins, max_T]
    bm_tensor_t out_tensors[3];
    bm_shape_t out_shape = {{3}, {1, (unsigned)n_bins, (unsigned)max_T}};
    for (int i = 0; i < 3; ++i)
        bmrt_tensor(&out_tensors[i], impl_->bmrt, BM_FLOAT32, out_shape);

    bool ok = bmrt_launch_tensor(impl_->bmrt, impl_->net_name.c_str(),
                                  &in_tensor, 1, out_tensors, 3);
    bm_thread_sync(impl_->handle);

    VocosOutput result;
    if (ok) {
        result.T   = use_T;
        int sz     = n_bins * max_T;
        result.mag.resize(sz);
        result.x.resize(sz);
        result.y.resize(sz);
        bm_memcpy_d2s(impl_->handle, result.mag.data(), out_tensors[0].device_mem);
        bm_memcpy_d2s(impl_->handle, result.x.data(),   out_tensors[1].device_mem);
        bm_memcpy_d2s(impl_->handle, result.y.data(),   out_tensors[2].device_mem);
    }

    bm_free_device(impl_->handle, in_tensor.device_mem);
    for (int i = 0; i < 3; ++i)
        bm_free_device(impl_->handle, out_tensors[i].device_mem);

    return result;
}
