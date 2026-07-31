#include "bmruntime_backend.h"
#ifdef ZIPFORMER_WITH_BM
#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>

namespace {

void SetError(std::string* e, const std::string& s) {
    if (e) *e = s;
}

bool MulOverflow(size_t a, size_t b, size_t* result) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return false;
    *result = a * b;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

BmRuntime::BmRuntime() {
    for (int i = 0; i < 36; ++i)
        state_in_[i] = state_out_[i] = -1;
}

BmRuntime::~BmRuntime() { cleanup(); }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

size_t BmRuntime::type_bytes(const std::string& dtype) {
    if (dtype == "float32" || dtype == "int32") return 4;
    if (dtype == "float16") return 2;
    if (dtype == "int8" || dtype == "uint8") return 1;
    if (dtype == "int64") return 8;
    return 0;
}

bool BmRuntime::dtype(const std::string& d, bm_data_type_t* out) {
    if (!out) return false;
    if (d == "float32") *out = BM_FLOAT32;
    else if (d == "float16") *out = BM_FLOAT16;
    else if (d == "int32" || d == "int64") *out = BM_INT32;
    else if (d == "int8") *out = BM_INT8;
    else if (d == "uint8") *out = BM_UINT8;
    else return false;
    return true;
}

bool BmRuntime::shape(const TensorSpec& t, bm_shape_t* out) {
    if (!out || t.shape.empty() || t.shape.size() > BM_MAX_DIMS_NUM)
        return false;
    out->num_dims = static_cast<int>(t.shape.size());
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (t.shape[i] <= 0 ||
            static_cast<uint64_t>(t.shape[i]) > UINT_MAX)
            return false;
        out->dims[i] = static_cast<unsigned>(t.shape[i]);
    }
    return true;
}

size_t BmRuntime::elements(const TensorSpec& t) {
    size_t n = 1;
    for (int64_t x : t.shape) {
        if (x <= 0 || !MulOverflow(n, static_cast<size_t>(x), &n)) return 0;
    }
    return n;
}

void BmRuntime::release(Buffer* buf) {
    if (buf && buf->owned) {
        if (handle_) bm_free_device(handle_, buf->mem);
        buf->owned = false;
        buf->bytes = 0;
    }
}

void BmRuntime::cleanup() {
    for (int s = 0; s < 2; ++s)
        for (int i = 0; i < 35; ++i)
            release(&state_[s][i]);

    Buffer* bufs[] = {&enc_x_, &enc_out_, &dec_in_, &dec_out_,
                      &join_enc_, &join_dec_, &join_out_};
    for (Buffer* b : bufs) release(b);

    for (void*& r : runtime_) {
        if (r) { bmrt_destroy(r); r = nullptr; }
    }
    for (int i = 0; i < 3; ++i) {
        info_[i] = nullptr;
        graph_[i].clear();
    }
    if (handle_) { bm_dev_free(handle_); handle_ = nullptr; }
    validated_ = false;
    state_count_ = state_slot_ = 0;
}

bool BmRuntime::alloc(Buffer* buf, size_t n, std::string* error) {
    if (!buf || !handle_ || !n) {
        SetError(error, "invalid device allocation");
        return false;
    }
    if (bm_malloc_device_byte(handle_, &buf->mem, n) != BM_SUCCESS) {
        SetError(error, "device allocation failed");
        return false;
    }
    buf->bytes = n;
    buf->owned = true;
    return true;
}

const NetworkSpec* BmRuntime::net(const char* name) const {
    for (const auto& n : manifest_.networks)
        if (n.name == name) return &n;
    return nullptr;
}

// ---------------------------------------------------------------------------
// Load & Validate
// ---------------------------------------------------------------------------

bool BmRuntime::Load(const std::string& enc_path,
                     const std::string& dec_path,
                     const std::string& joi_path, std::string* error) {
    if (handle_ || runtime_[0] || runtime_[1] || runtime_[2]) {
        SetError(error, "Load called on an already loaded runtime");
        return false;
    }
    if (bm_dev_request(&handle_, 0) != BM_SUCCESS) {
        SetError(error, "bm_dev_request failed");
        cleanup();
        return false;
    }

    const std::string paths[3] = {enc_path, dec_path, joi_path};
    const char* expected_names[3] = {"zipformer_encoder",
                                     "zipformer_decoder",
                                     "zipformer_joiner"};

    for (int k = 0; k < 3; ++k) {
        runtime_[k] = bmrt_create(handle_);
        if (!runtime_[k] ||
            !bmrt_load_bmodel(runtime_[k], paths[k].c_str())) {
            SetError(error, "bmodel load failed");
            cleanup();
            return false;
        }

        int count = bmrt_get_network_number(runtime_[k]);
        if (count != 1) {
            SetError(error, "each bmodel must contain exactly one graph");
            cleanup();
            return false;
        }

        const char** names = nullptr;
        bmrt_get_network_names(runtime_[k], &names);
        if (!names || !names[0]) {
            if (names) free(names);
            SetError(error, "network name unavailable");
            cleanup();
            return false;
        }
        graph_[k] = names[0];
        free(names);

        if (graph_[k] != expected_names[k]) {
            SetError(error, "graph name does not match logical network");
            cleanup();
            return false;
        }

        info_[k] =
            bmrt_get_network_info(runtime_[k], graph_[k].c_str());
        if (!info_[k]) {
            SetError(error, "network info unavailable");
            cleanup();
            return false;
        }
    }
    return true;
}

bool BmRuntime::Validate(const Manifest& m, std::string* error) {
    if (validated_) {
        SetError(error, "Validate may only be called once");
        return false;
    }
    if (!handle_) {
        SetError(error, "runtime is not loaded");
        return false;
    }
    if (m.networks.size() != 3) {
        SetError(error, "manifest must contain encoder, decoder and joiner");
        return false;
    }

    manifest_ = m;
    const NetworkSpec* networks[3] = {net("encoder"), net("decoder"),
                                       net("joiner")};
    if (!networks[0] || !networks[1] || !networks[2]) {
        SetError(error, "manifest network names incomplete");
        return false;
    }

    // Fixed protocol constants.
    if (m.context != 2 || m.segment != 103 || m.offset != 96 ||
        m.n_mels != 80 || m.sample_rate != 16000 || m.vocab_size != 6254) {
        SetError(error, "manifest protocol constants mismatch");
        return false;
    }

    // Per-network tensor validation against bmodel metadata.
    for (int k = 0; k < 3; ++k) {
        const NetworkSpec& net = *networks[k];
        const bm_net_info_t* inf = info_[k];

        if (static_cast<int>(net.inputs.size()) != inf->input_num ||
            static_cast<int>(net.outputs.size()) != inf->output_num) {
            SetError(error, net.name + " tensor count mismatch");
            return false;
        }

        auto CheckDir = [&](const std::vector<TensorSpec>& tensors,
                            const char* const* names,
                            const bm_shape_t* shapes,
                            const bm_data_type_t* dtypes, int count,
                            bool is_output, const size_t* maxbytes) {
            std::vector<bool> seen(static_cast<size_t>(count), false);

            for (const TensorSpec& t : tensors) {
                if (t.index < 0 || t.index >= count || seen[t.index]) {
                    SetError(error, net.name + " tensor index error: " + t.name);
                    return false;
                }
                // Output names may carry structural suffixes from TPU-MLIR.
                if (is_output) {
                    std::string rname = names[t.index];
                    bool match =
                        (rname == t.name ||
                         rname == t.name + "_Transpose" ||
                         rname == t.name + "_Unsqueeze" ||
                         rname == t.name + "_Squeeze" ||
                         rname == t.name + "_Gemm" ||
                         rname == t.name + "_Transpose_f32" ||
                         rname == t.name + "_Unsqueeze_f32" ||
                         rname == t.name + "_Squeeze_f32" ||
                         rname == t.name + "_Gemm_f32");
                    if (!match) {
                        SetError(error, net.name + " output name mismatch: " +
                                            t.name);
                        return false;
                    }
                } else {
                    if (std::string(names[t.index]) != t.name) {
                        SetError(error, net.name + " input name mismatch: " +
                                            t.name);
                        return false;
                    }
                }
                seen[t.index] = true;

                // Dtype check.  Decoder token_ids is the only
                // logical-int64 / runtime-int32 mismatch.
                bm_data_type_t logical_dt, runtime_dt;
                if (!dtype(t.dtype, &logical_dt) ||
                    !dtype(t.runtime_dtype, &runtime_dt)) {
                    SetError(error, "unsupported dtype: " + t.name);
                    return false;
                }
                if (is_output || !(net.name == "decoder" &&
                                   t.name == "token_ids")) {
                    if (t.dtype != t.runtime_dtype) {
                        SetError(error,
                                 "logical/runtime dtype mismatch: " + t.name);
                        return false;
                    }
                } else {
                    if (t.dtype != "int64" || t.runtime_dtype != "int32") {
                        SetError(error,
                                 "decoder token must be logical int64 / "
                                 "runtime int32");
                        return false;
                    }
                }

                // Shape check.
                bm_shape_t s;
                if (!shape(t, &s) || !bmrt_shape_is_same(&s, &shapes[t.index]) ||
                    runtime_dt != dtypes[t.index]) {
                    SetError(error,
                             "runtime tensor shape/dtype mismatch: " + t.name);
                    return false;
                }

                // Byte-size check.
                size_t byte_size;
                if (!MulOverflow(elements(t), type_bytes(t.runtime_dtype),
                                 &byte_size) ||
                    !byte_size) {
                    SetError(error, "invalid tensor byte size: " + t.name);
                    return false;
                }
                if (maxbytes[t.index] < byte_size) {
                    SetError(error, "runtime max bytes smaller than manifest "
                                    "bytes: " +
                                        t.name);
                    return false;
                }
            }
            for (bool x : seen)
                if (!x) {
                    SetError(error, "manifest tensor indices are not contiguous");
                    return false;
                }
            return true;
        };

        if (!CheckDir(net.inputs, inf->input_names,
                      inf->stages[0].input_shapes, inf->input_dtypes,
                      inf->input_num, false, inf->max_input_bytes) ||
            !CheckDir(net.outputs, inf->output_names,
                      inf->stages[0].output_shapes, inf->output_dtypes,
                      inf->output_num, true, inf->max_output_bytes))
            return false;
    }

    // --- Encoder state validation ---
    for (const TensorSpec& t : networks[0]->inputs)
        if (t.name != "x" && t.name.find("cached_") != 0) {
            SetError(error, "encoder state must be cached_ tensors");
            return false;
        }
    if (networks[0]->inputs.size() != 36 ||
        networks[0]->outputs.size() != 36) {
        SetError(error, "encoder must have exactly 35 states plus x/output");
        return false;
    }
    state_count_ = 35;

    // Pair every cached_* input with its new_cached_* output.
    int slot = 0;
    for (const auto& in_ref : networks[0]->inputs) {
        if (in_ref.name == "x") continue;
        if (in_ref.name.find("cached_") != 0 ||
            in_ref.name.find("new_") == 0 || slot >= 35) {
            SetError(error, "invalid encoder cached state set");
            return false;
        }
        const TensorSpec* out = nullptr;
        for (const auto& t : networks[0]->outputs) {
            if (t.name == "new_" + in_ref.name) {
                if (out) {
                    SetError(error, "duplicate new_cached state");
                    return false;
                }
                out = &t;
            }
        }
        if (!out || out->shape != in_ref.shape ||
            out->runtime_dtype != in_ref.runtime_dtype ||
            out->bytes != in_ref.bytes) {
            SetError(error, "cached/new_cached state mismatch: " + in_ref.name);
            return false;
        }
        state_in_[in_ref.index] = slot;
        state_out_[out->index] = slot;
        ++slot;
    }
    if (slot != 35) {
        SetError(error, "encoder must have exactly 35 cached states");
        return false;
    }
    for (const auto& t : networks[0]->outputs)
        if (t.name != "encoder_out" && state_out_[t.index] < 0) {
            SetError(error, "unmapped encoder state output");
            return false;
        }

    if (!alloc_buffers(error)) return false;
    validated_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// Device buffer management
// ---------------------------------------------------------------------------

bool BmRuntime::alloc_buffers(std::string* error) {
    const NetworkSpec& enc = *net("encoder");

    auto Need = [&](const TensorSpec& t) -> size_t {
        size_t b = 0;
        return MulOverflow(elements(t), type_bytes(t.runtime_dtype), &b) ? b
                                                                         : 0;
    };

    // Rollback on partial allocation failure.
    auto Rollback = [&]() {
        for (int s = 0; s < 2; ++s)
            for (int i = 0; i < 35; ++i)
                release(&state_[s][i]);
        Buffer* all[] = {&enc_x_, &enc_out_, &dec_in_, &dec_out_,
                         &join_enc_, &join_dec_, &join_out_};
        for (Buffer* b : all) release(b);
    };

    // Allocate dual-buffer state slots (35 × 2).
    for (const TensorSpec& t : enc.inputs) {
        if (t.name == "x") continue;
        size_t sz = Need(t);
        for (int s = 0; s < 2; ++s) {
            if (!alloc(&state_[s][state_in_[t.index]], sz, error)) {
                Rollback();
                return false;
            }
        }
    }

    // Look up the 7 main tensors by logical name.
    auto Find = [&](const NetworkSpec& net, const char* name,
                     bool is_out) -> const TensorSpec* {
        const auto& vec = is_out ? net.outputs : net.inputs;
        for (const auto& t : vec)
            if (t.name == name) return &t;
        return nullptr;
    };

    const TensorSpec *x = Find(enc, "x", false),
                     *eo = Find(enc, "encoder_out", true),
                     *ti = Find(*net("decoder"), "token_ids", false),
                     *do_ = Find(*net("decoder"), "decoder_out", true),
                     *je = Find(*net("joiner"), "enc_out", false),
                     *jd = Find(*net("joiner"), "dec_out", false),
                     *lo = Find(*net("joiner"), "logit", true);

    const TensorSpec* specs[] = {x, eo, ti, do_, je, jd, lo};
    for (auto* t : specs) {
        if (!t) {
            SetError(error, "required logical tensor missing");
            Rollback();
            return false;
        }
    }

    Buffer* bufs[] = {&enc_x_, &enc_out_, &dec_in_, &dec_out_,
                      &join_enc_, &join_dec_, &join_out_};
    for (int i = 0; i < 7; ++i) {
        if (!alloc(bufs[i], Need(*specs[i]), error)) {
            Rollback();
            return false;
        }
    }
    return true;
}

bool BmRuntime::launch(int k, std::vector<bm_tensor_t>* in,
                       std::vector<bm_tensor_t>* out, std::string* error) {
    if (!bmrt_launch_tensor_ex(runtime_[k], graph_[k].c_str(),
                               in->data(), in->size(),
                               out->data(), out->size(),
                               true, false) ||
        bm_thread_sync(handle_) != BM_SUCCESS) {
        SetError(error, "network launch / synchronisation failed");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Runtime API
// ---------------------------------------------------------------------------

bool BmRuntime::Reset(std::string* error) {
    if (!validated_) {
        SetError(error, "backend is not validated");
        return false;
    }
    for (int s = 0; s < 2; ++s)
        for (int i = 0; i < 35; ++i)
            if (bm_memset_device(handle_, 0, state_[s][i].mem) !=
                BM_SUCCESS) {
                SetError(error, "state reset failed");
                return false;
            }
    state_slot_ = 0;
    return true;
}

bool BmRuntime::Encoder(const std::vector<float>& features,
                        std::vector<float>* encoder_out,
                        std::string* error) {
    if (!validated_ || !encoder_out) {
        SetError(error, "encoder unavailable");
        return false;
    }

    const NetworkSpec& enc_net = *net("encoder");
    const TensorSpec* x_spec = nullptr;
    const TensorSpec* out_spec = nullptr;
    for (const auto& t : enc_net.inputs)
        if (t.name == "x") x_spec = &t;
    for (const auto& t : enc_net.outputs)
        if (t.name == "encoder_out") out_spec = &t;

    if (!x_spec || !out_spec ||
        features.size() != elements(*x_spec)) {
        SetError(error, "encoder input element count mismatch");
        return false;
    }

    // Host → Device.
    if (bm_memcpy_s2d(handle_, enc_x_.mem,
                      const_cast<float*>(features.data())) != BM_SUCCESS) {
        SetError(error, "encoder H2D failed");
        return false;
    }

    // Build input tensor list.
    std::vector<bm_tensor_t> in(info_[0]->input_num),
                             out(info_[0]->output_num);
    for (const auto& t : enc_net.inputs) {
        bm_shape_t s;
        bm_data_type_t d;
        if (!shape(t, &s) || !dtype(t.runtime_dtype, &d)) return false;
        bmrt_tensor_with_device(
            &in[t.index],
            t.name == "x"
                ? enc_x_.mem
                : state_[state_slot_][state_in_[t.index]].mem,
            d, s);
    }
    for (const auto& t : enc_net.outputs) {
        bm_shape_t s;
        bm_data_type_t d;
        if (!shape(t, &s) || !dtype(t.runtime_dtype, &d)) return false;
        bmrt_tensor_with_device(
            &out[t.index],
            t.name == "encoder_out"
                ? enc_out_.mem
                : state_[1 - state_slot_][state_out_[t.index]].mem,
            d, s);
    }

    // Launch.
    if (!launch(0, &in, &out, error)) return false;

    // Device → Host (encoder_out only; states stay on device).
    encoder_out->resize(elements(*out_spec));
    if (bm_memcpy_d2s(handle_, encoder_out->data(),
                      enc_out_.mem) != BM_SUCCESS) {
        SetError(error, "encoder_out D2H failed");
        return false;
    }

    // Swap state slots only on success.
    state_slot_ = 1 - state_slot_;
    return true;
}

bool BmRuntime::Decoder(const int64_t tokens[2],
                        std::vector<float>* decoder_out,
                        std::string* error) {
    if (!validated_ || !tokens || !decoder_out) {
        SetError(error, "decoder unavailable");
        return false;
    }

    const NetworkSpec& dec_net = *net("decoder");
    const TensorSpec* in_spec = nullptr;
    const TensorSpec* out_spec = nullptr;
    for (const auto& t : dec_net.inputs)
        if (t.name == "token_ids") in_spec = &t;
    for (const auto& t : dec_net.outputs)
        if (t.name == "decoder_out") out_spec = &t;

    if (!in_spec || !out_spec || elements(*in_spec) != 2) {
        SetError(error, "decoder contract invalid");
        return false;
    }

    // Validate and convert int64 → int32.
    int32_t ids[2];
    for (int i = 0; i < 2; ++i) {
        if (tokens[i] < 0 || tokens[i] >= manifest_.vocab_size ||
            tokens[i] > INT32_MAX) {
            SetError(error, "decoder token out of range");
            return false;
        }
        ids[i] = static_cast<int32_t>(tokens[i]);
    }

    if (bm_memcpy_s2d(handle_, dec_in_.mem, ids) != BM_SUCCESS) {
        SetError(error, "decoder H2D failed");
        return false;
    }

    bm_shape_t in_shape, out_shape;
    bm_data_type_t in_dt, out_dt;
    if (!shape(*in_spec, &in_shape) || !shape(*out_spec, &out_shape) ||
        !dtype(in_spec->runtime_dtype, &in_dt) ||
        !dtype(out_spec->runtime_dtype, &out_dt))
        return false;

    std::vector<bm_tensor_t> in(info_[1]->input_num),
                             out(info_[1]->output_num);
    bmrt_tensor_with_device(&in[in_spec->index], dec_in_.mem, in_dt,
                             in_shape);
    bmrt_tensor_with_device(&out[out_spec->index], dec_out_.mem, out_dt,
                             out_shape);

    if (!launch(1, &in, &out, error)) return false;

    decoder_out->resize(elements(*out_spec));
    if (bm_memcpy_d2s(handle_, decoder_out->data(),
                      dec_out_.mem) != BM_SUCCESS) {
        SetError(error, "decoder_out D2H failed");
        return false;
    }
    return true;
}

bool BmRuntime::Joiner(const std::vector<float>& enc_out,
                       const std::vector<float>& dec_out,
                       std::vector<float>* logits,
                       std::string* error) {
    if (!validated_ || !logits) {
        SetError(error, "joiner unavailable");
        return false;
    }

    const NetworkSpec& joi_net = *net("joiner");
    const TensorSpec *enc_spec = nullptr, *dec_spec = nullptr,
                     *out_spec = nullptr;
    for (const auto& t : joi_net.inputs) {
        if (t.name == "enc_out") enc_spec = &t;
        if (t.name == "dec_out") dec_spec = &t;
    }
    for (const auto& t : joi_net.outputs)
        if (t.name == "logit") out_spec = &t;

    if (!enc_spec || !dec_spec || !out_spec ||
        enc_out.size() != elements(*enc_spec) ||
        dec_out.size() != elements(*dec_spec)) {
        SetError(error, "joiner input mismatch");
        return false;
    }

    if (bm_memcpy_s2d(handle_, join_enc_.mem,
                      const_cast<float*>(enc_out.data())) != BM_SUCCESS ||
        bm_memcpy_s2d(handle_, join_dec_.mem,
                      const_cast<float*>(dec_out.data())) != BM_SUCCESS) {
        SetError(error, "joiner H2D failed");
        return false;
    }

    bm_shape_t s_enc, s_dec, s_out;
    bm_data_type_t d_enc, d_dec, d_out;
    if (!shape(*enc_spec, &s_enc) || !shape(*dec_spec, &s_dec) ||
        !shape(*out_spec, &s_out) ||
        !dtype(enc_spec->runtime_dtype, &d_enc) ||
        !dtype(dec_spec->runtime_dtype, &d_dec) ||
        !dtype(out_spec->runtime_dtype, &d_out))
        return false;

    std::vector<bm_tensor_t> in(info_[2]->input_num),
                             out(info_[2]->output_num);
    bmrt_tensor_with_device(&in[enc_spec->index], join_enc_.mem, d_enc,
                             s_enc);
    bmrt_tensor_with_device(&in[dec_spec->index], join_dec_.mem, d_dec,
                             s_dec);
    bmrt_tensor_with_device(&out[out_spec->index], join_out_.mem, d_out,
                             s_out);

    if (!launch(2, &in, &out, error)) return false;

    logits->resize(elements(*out_spec));
    if (bm_memcpy_d2s(handle_, logits->data(),
                      join_out_.mem) != BM_SUCCESS) {
        SetError(error, "logit D2H failed");
        return false;
    }
    return true;
}

#endif  // ZIPFORMER_WITH_BM
