#include "vocos_engine.h"
#include <bmruntime_interface.h>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <cmath>
#include <algorithm>

struct VocosEngine::Impl {
    bm_handle_t handle    = nullptr;
    void*       bmrt      = nullptr;
    bool        owns_hdl  = true;
    bool        owns_bmrt = true;
    std::string net_name;
    int         max_T    = 0;
    int         n_mels   = 100;
    int         n_bins   = 513;

    // Persistent device buffers — allocated once, reused every infer() call
    bm_device_mem_t in_dm     = {};   // [1, n_mels, max_T] f32
    bm_device_mem_t out_dm[3] = {};   // [1, n_bins, max_T] f32 × 3
    bool bufs_ready = false;

    void init(const std::string& bmodel_path);
};

void VocosEngine::Impl::init(const std::string& bmodel_path) {
    if (!bmrt_load_bmodel(bmrt, bmodel_path.c_str()))
        throw std::runtime_error("VocosEngine: load bmodel failed: " + bmodel_path);

    const char** names = nullptr;
    int n_nets = 0;
    bmrt_get_network_names(bmrt, &names);
    while (names && names[n_nets]) ++n_nets;
    if (n_nets == 0) {
        free(names);
        throw std::runtime_error("VocosEngine: no network in bmodel");
    }
    net_name = names[0];
    free(names);

    const bm_net_info_t* info = bmrt_get_network_info(bmrt, net_name.c_str());
    if (!info) throw std::runtime_error("VocosEngine: get network info failed");
    n_mels = info->stages[0].input_shapes[0].dims[1];
    max_T  = info->stages[0].input_shapes[0].dims[2];
    n_bins = info->stages[0].output_shapes[0].dims[1];

    size_t in_bytes  = (size_t)n_mels * max_T * sizeof(float);
    size_t out_bytes = (size_t)n_bins  * max_T * sizeof(float);
    bool ok = bm_malloc_device_byte(handle, &in_dm, in_bytes) == BM_SUCCESS;
    for (int i = 0; i < 3 && ok; ++i)
        ok = bm_malloc_device_byte(handle, &out_dm[i], out_bytes) == BM_SUCCESS;
    if (!ok)
        throw std::runtime_error("VocosEngine: pre-alloc device buffers failed");
    bufs_ready = true;
}

VocosEngine::VocosEngine(const std::string& bmodel_path, int tpu_id)
    : impl_(std::make_unique<Impl>()) {
    impl_->owns_hdl  = true;
    impl_->owns_bmrt = true;
    if (bm_dev_request(&impl_->handle, tpu_id) != BM_SUCCESS)
        throw std::runtime_error("VocosEngine: bm_dev_request failed");
    impl_->bmrt = bmrt_create(impl_->handle);
    if (!impl_->bmrt) throw std::runtime_error("VocosEngine: bmrt_create failed");
    impl_->init(bmodel_path);
}

VocosEngine::VocosEngine(const std::string& bmodel_path, void* bm_handle, void* /*ignored*/)
    : impl_(std::make_unique<Impl>()) {
    impl_->owns_hdl  = false;
    impl_->owns_bmrt = true;  // we create our own bmrt using the shared handle
    impl_->handle    = static_cast<bm_handle_t>(bm_handle);
    impl_->bmrt      = bmrt_create(impl_->handle);
    if (!impl_->bmrt) throw std::runtime_error("VocosEngine: bmrt_create failed");
    impl_->init(bmodel_path);
}

VocosEngine::~VocosEngine() {
    if (impl_->bufs_ready) {
        bm_free_device(impl_->handle, impl_->in_dm);
        for (int i = 0; i < 3; ++i)
            bm_free_device(impl_->handle, impl_->out_dm[i]);
    }
    if (impl_->owns_bmrt && impl_->bmrt) bmrt_destroy(impl_->bmrt);
    if (impl_->owns_hdl  && impl_->handle) bm_dev_free(impl_->handle);
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
    bmrt_tensor_with_device(&in_tensor, impl_->in_dm, BM_FLOAT32, in_shape);
    bm_memcpy_s2d(impl_->handle, impl_->in_dm, (void*)in_buf.data());

    bm_tensor_t out_tensors[3];
    bm_shape_t out_shape = {{3}, {1, (unsigned)n_bins, (unsigned)max_T}};
    for (int i = 0; i < 3; ++i)
        bmrt_tensor_with_device(&out_tensors[i], impl_->out_dm[i], BM_FLOAT32, out_shape);

    bool ok = bmrt_launch_tensor_ex(impl_->bmrt, impl_->net_name.c_str(),
                                     &in_tensor, 1, out_tensors, 3, true, false);
    bm_thread_sync(impl_->handle);

    VocosOutput result;
    if (ok) {
        result.T = use_T;
        // Download full bmodel output [n_bins, max_T], then slice to [n_bins, use_T]
        std::vector<float> tmp(n_bins * max_T);

        bm_memcpy_d2s(impl_->handle, tmp.data(), impl_->out_dm[0]);
        result.mag.resize(n_bins * use_T);
        for (int b = 0; b < n_bins; ++b)
            std::copy(tmp.begin() + b * max_T, tmp.begin() + b * max_T + use_T,
                      result.mag.begin() + b * use_T);

        bm_memcpy_d2s(impl_->handle, tmp.data(), impl_->out_dm[1]);
        result.x.resize(n_bins * use_T);
        for (int b = 0; b < n_bins; ++b)
            std::copy(tmp.begin() + b * max_T, tmp.begin() + b * max_T + use_T,
                      result.x.begin() + b * use_T);

        bm_memcpy_d2s(impl_->handle, tmp.data(), impl_->out_dm[2]);
        result.y.resize(n_bins * use_T);
        for (int b = 0; b < n_bins; ++b)
            std::copy(tmp.begin() + b * max_T, tmp.begin() + b * max_T + use_T,
                      result.y.begin() + b * use_T);
    }

    return result;
}
