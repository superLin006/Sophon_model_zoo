// Qwen3-ASR BM1684X 推理引擎（纯 bmrt）
// 结构复刻 Eureka-Audio eureka_audio.cpp，decode 修正 off-by-one（pos=token_length）

#include "asr_engine.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>

namespace asr {

static inline void bm_check(bm_status_t s, const char* msg) {
    if (s != BM_SUCCESS) {
        fprintf(stderr, "[ASR] BM error %d: %s\n", (int)s, msg);
        throw std::runtime_error(msg);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// EncoderBmodel
// ══════════════════════════════════════════════════════════════════════════════

EncoderBmodel::~EncoderBmodel() { release(); }

void EncoderBmodel::release() {
    if (p_bmrt_) { bmrt_destroy(p_bmrt_); p_bmrt_ = nullptr; }
    inited_ = false;
}

bool EncoderBmodel::init(bm_handle_t handle, const std::string& path, const char* net_name) {
    handle_ = handle;
    p_bmrt_ = bmrt_create(handle_);
    if (!p_bmrt_) { fprintf(stderr, "[Enc] bmrt_create failed\n"); return false; }
    if (!bmrt_load_bmodel(p_bmrt_, path.c_str())) {
        fprintf(stderr, "[Enc] load bmodel failed: %s\n", path.c_str());
        bmrt_destroy(p_bmrt_); p_bmrt_ = nullptr;
        return false;
    }
    net_ = bmrt_get_network_info(p_bmrt_, net_name);
    if (!net_) {
        fprintf(stderr, "[Enc] network '%s' not found\n", net_name);
        return false;
    }
    inited_ = true;
    printf("[Enc] loaded %s (net=%s)\n", path.c_str(), net_name);
    return true;
}

bool EncoderBmodel::run(const float* mel, std::vector<float>& out) {
    if (!inited_) return false;
    // 注意：不能用 net_->stages[0].input_mems 预分配 mem（combine 后该字段未正确填充，
    // bm_mem_get_device_size 返回垃圾值）。用 bmrt_tensor 自动分配 + s2d/d2s 显式拷贝。
    const bm_shape_t& in_sh  = net_->stages[0].input_shapes[0];
    const bm_shape_t& out_sh = net_->stages[0].output_shapes[0];
    int in_elems = 1, out_elems = 1;
    for (int i = 0; i < in_sh.num_dims; i++)  in_elems  *= in_sh.dims[i];
    for (int i = 0; i < out_sh.num_dims; i++) out_elems *= out_sh.dims[i];

    // sail 的 SYSIO 模式 = bmrt_launch_data（host 内存直传）。
    // 实测：LLM bmodel 加载后，bmrt_launch_tensor(_ex) 路径的 encoder 输出 NaN，
    // bmrt_launch_data 与 sail 同路径（板上已验证正常）。
    out.resize(out_elems);
    void* in_datas[1]  = {(void*)mel};
    bm_shape_t in_shapes[1]  = {in_sh};
    void* out_datas[1] = {out.data()};
    bm_shape_t out_shapes[1] = {out_sh};
    bool launch_ok = bmrt_launch_data(p_bmrt_, net_->name, in_datas, in_shapes, 1,
                                      out_datas, out_shapes, 1, true);
    if (!launch_ok) {
        fprintf(stderr, "[Enc] launch failed\n");
        return false;
    }
    // 输出 dtype：若 F16/BF16 需转换（bmrt_launch_data 写 host 内存按 net dtype）
    if (net_->output_dtypes[0] != BM_FLOAT32) {
        std::vector<uint16_t> tmp(out.begin(), out.end());
        if (net_->output_dtypes[0] == BM_BFLOAT16) {
            for (int i = 0; i < out_elems; i++) {
                uint32_t u = (uint32_t)tmp[i] << 16;
                float f; memcpy(&f, &u, 4);
                out[i] = f;
            }
        } else {  // BM_FLOAT16
            for (int i = 0; i < out_elems; i++) {
                uint16_t h = tmp[i];
                uint32_t s = (uint32_t)(h & 0x8000) << 16;
                uint32_t e = (uint32_t)((h >> 10) & 0x1f);
                uint32_t m = (uint32_t)(h & 0x3ff);
                uint32_t f;
                if (e == 0) {
                    if (m == 0) f = s;
                    else {
                        e = 1;
                        while (!(m & 0x400)) { m <<= 1; e--; }
                        m &= 0x3ff;
                        e += 127 - 15;
                        f = s | (e << 23) | (m << 13);
                    }
                } else if (e == 31) {
                    f = s | 0x7f800000 | (m << 13);
                } else {
                    e += 127 - 15;
                    f = s | (e << 23) | (m << 13);
                }
                float v; memcpy(&v, &f, 4);
                out[i] = v;
            }
        }
    }
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Qwen3Bmodel
// ══════════════════════════════════════════════════════════════════════════════

Qwen3Bmodel::~Qwen3Bmodel() { deinit(); }

void Qwen3Bmodel::deinit() {
    if (!inited_) return;
    for (auto& m : past_key_dev_)   bm_free_device(handle_, m);
    for (auto& m : past_value_dev_) bm_free_device(handle_, m);
    past_key_dev_.clear(); past_value_dev_.clear();
    if (dec_io_ready_) {
        bm_free_device(handle_, dec_hidden_);  bm_free_device(handle_, dec_pos_);
        bm_free_device(handle_, dec_mask_);    bm_free_device(handle_, dec_newk_);
        bm_free_device(handle_, dec_newv_);    bm_free_device(handle_, dec_emb_out_);
        dec_io_ready_ = false;
    }
    if (pre_io_ready_) {
        bm_free_device(handle_, pre_hidden_a_); bm_free_device(handle_, pre_hidden_b_);
        bm_free_device(handle_, pre_pos_);      bm_free_device(handle_, pre_mask_);
        bm_free_device(handle_, pre_kv_);       bm_free_device(handle_, pre_v_);
        pre_io_ready_ = false;
    }
    if (p_bmrt_) { bmrt_destroy(p_bmrt_); p_bmrt_ = nullptr; }
    inited_ = false;
}

void Qwen3Bmodel::init_by_names() {
    auto num_nets = bmrt_get_network_number(p_bmrt_);
    const char** names = nullptr;
    bmrt_get_network_names(p_bmrt_, &names);
    auto find = [&](const char* n) -> bool {
        for (int i = 0; i < (int)num_nets; i++)
            if (strcmp(n, names[i]) == 0) return true;
        return false;
    };
    net_embed_cache_  = bmrt_get_network_info(p_bmrt_, "embedding_cache");
    net_lm_           = bmrt_get_network_info(p_bmrt_, "lm_head");
    if (find("greedy_head"))
        net_greedy_head_ = bmrt_get_network_info(p_bmrt_, "greedy_head");
    for (int i = 0; ; i++) {
        auto bn = "block_" + std::to_string(i);
        auto cn = "block_cache_" + std::to_string(i);
        if (!find(bn.c_str()) || !find(cn.c_str())) break;
        net_blocks_.push_back(bmrt_get_network_info(p_bmrt_, bn.c_str()));
        net_blocks_cache_.push_back(bmrt_get_network_info(p_bmrt_, cn.c_str()));
    }
    NUM_LAYERS = (int)net_blocks_.size();
    free(names);

    MAX_INPUT_LEN = net_blocks_[0]->stages[0].input_shapes[0].dims[1];
    hidden_size_  = net_blocks_[0]->stages[0].input_shapes[0].dims[2];
    SEQLEN        = net_blocks_cache_[0]->stages[0].input_shapes[3].dims[1];
    int n_kv      = net_blocks_cache_[0]->stages[0].input_shapes[3].dims[2];
    int head_dim  = net_blocks_cache_[0]->stages[0].input_shapes[3].dims[3];
    kv_per_token_ = n_kv * head_dim;
    printf("[Qwen3] Layers=%d MAX_INPUT=%d SEQLEN=%d kv_per_token=%d hidden=%d\n",
           NUM_LAYERS, MAX_INPUT_LEN, SEQLEN, kv_per_token_, hidden_size_);
}

bool Qwen3Bmodel::init(bm_handle_t handle, const std::string& path) {
    handle_ = handle;
    p_bmrt_ = bmrt_create(handle_);
    if (!p_bmrt_) { fprintf(stderr, "[Qwen3] bmrt_create failed\n"); return false; }
    if (!bmrt_load_bmodel(p_bmrt_, path.c_str())) {
        fprintf(stderr, "[Qwen3] load bmodel failed: %s\n", path.c_str());
        return false;
    }
    printf("[Qwen3] loaded %s\n", path.c_str());
    init_by_names();

    // KV 为 F16（KV 量化 bmodel：block_cache 的 past_k/v 输入 dtype 是 FLOAT16）
    // 用 net 的 input dtype 判断字节数
    bm_data_type_t kv_dtype = net_blocks_cache_[0]->input_dtypes[3];
    int kv_bytes_per_elem = (kv_dtype == BM_FLOAT32) ? 4 : 2;
    kv_layer_bytes_ = (size_t)SEQLEN * kv_per_token_ * kv_bytes_per_elem;
    kv_token_bytes_ = (size_t)kv_per_token_ * kv_bytes_per_elem;
    past_key_dev_.resize(NUM_LAYERS);
    past_value_dev_.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        bm_malloc_device_byte(handle_, &past_key_dev_[i], kv_layer_bytes_);
        bm_malloc_device_byte(handle_, &past_value_dev_[i], kv_layer_bytes_);
        int z = 0;
        bm_memset_device_ext(handle_, &z, 1, past_key_dev_[i]);
        bm_memset_device_ext(handle_, &z, 1, past_value_dev_[i]);
    }

    auto bc0 = net_blocks_cache_[0];
    bm_malloc_device_byte(handle_, &dec_hidden_,  bm_mem_get_device_size(bc0->stages[0].input_mems[0]));
    bm_malloc_device_byte(handle_, &dec_pos_,     bm_mem_get_device_size(bc0->stages[0].input_mems[1]));
    bm_malloc_device_byte(handle_, &dec_mask_,    bm_mem_get_device_size(bc0->stages[0].input_mems[2]));
    bm_malloc_device_byte(handle_, &dec_newk_,    bm_mem_get_device_size(bc0->stages[0].output_mems[1]));
    bm_malloc_device_byte(handle_, &dec_newv_,    bm_mem_get_device_size(bc0->stages[0].output_mems[2]));
    bm_malloc_device_byte(handle_, &dec_emb_out_, bm_mem_get_device_size(net_embed_cache_->stages[0].output_mems[0]));
    dec_io_ready_ = true;

    // prefill IO buffer（hidden ping-pong 2 份 + pos/mask + KV 输出）
    auto b0 = net_blocks_[0];
    bm_malloc_device_byte(handle_, &pre_hidden_a_, bm_mem_get_device_size(b0->stages[0].input_mems[0]));
    bm_malloc_device_byte(handle_, &pre_hidden_b_, bm_mem_get_device_size(b0->stages[0].input_mems[0]));
    bm_malloc_device_byte(handle_, &pre_pos_,      bm_mem_get_device_size(b0->stages[0].input_mems[1]));
    bm_malloc_device_byte(handle_, &pre_mask_,     bm_mem_get_device_size(b0->stages[0].input_mems[2]));
    bm_malloc_device_byte(handle_, &pre_kv_,       bm_mem_get_device_size(b0->stages[0].output_mems[1]));
    bm_malloc_device_byte(handle_, &pre_v_,        bm_mem_get_device_size(b0->stages[0].output_mems[2]));
    pre_io_ready_ = true;

    inited_ = true;
    return true;
}

// 通用 launch：host 输入 → device 推理 → host 输出
static bool launch_host(void* rt, bm_handle_t h, const bm_net_info_t* net,
                        const std::vector<const void*>& in_hosts,
                        const std::vector<void*>& out_hosts) {
    int ni = net->input_num, no = net->output_num;
    std::vector<bm_tensor_t> ins(ni), outs(no);
    for (int i = 0; i < ni; i++) {
        bmrt_tensor(&ins[i], rt, net->input_dtypes[i], net->stages[0].input_shapes[i]);
        if (in_hosts[i])
            bm_memcpy_s2d(h, ins[i].device_mem, (void*)in_hosts[i]);
    }
    for (int i = 0; i < no; i++)
        bmrt_tensor(&outs[i], rt, net->output_dtypes[i], net->stages[0].output_shapes[i]);
    bool ok = bmrt_launch_tensor_ex(rt, net->name, ins.data(), ni, outs.data(), no, true, false);
    if (ok) bm_thread_sync(h);
    if (ok)
        for (int i = 0; i < no; i++)
            if (i < (int)out_hosts.size() && out_hosts[i])
                bm_memcpy_d2s(h, out_hosts[i], outs[i].device_mem);
    for (int i = 0; i < ni; i++) bm_free_device(h, ins[i].device_mem);
    for (int i = 0; i < no; i++) bm_free_device(h, outs[i].device_mem);
    return ok;
}

static int argmax_logits(const std::vector<float>& logits) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < (int)logits.size(); i++)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

// hidden(host) → lm_head → greedy_head（device argmax）→ token
static int lm_greedy(void* rt, bm_handle_t h, const bm_net_info_t* lm,
                     const bm_net_info_t* greedy, const float* hidden_host) {
    bm_tensor_t lm_in, lm_out;
    bmrt_tensor(&lm_in,  rt, lm->input_dtypes[0],  lm->stages[0].input_shapes[0]);
    bmrt_tensor(&lm_out, rt, lm->output_dtypes[0], lm->stages[0].output_shapes[0]);
    bm_memcpy_s2d(h, lm_in.device_mem, (void*)hidden_host);
    bmrt_launch_tensor_ex(rt, lm->name, &lm_in, 1, &lm_out, 1, true, false);
    bm_thread_sync(h);
    bm_tensor_t g_in, g_out;
    bmrt_tensor(&g_in,  rt, greedy->input_dtypes[0],  greedy->stages[0].input_shapes[0]);
    bmrt_tensor(&g_out, rt, greedy->output_dtypes[0], greedy->stages[0].output_shapes[0]);
    bm_memcpy_d2d_byte(h, g_in.device_mem, 0, lm_out.device_mem, 0,
                       bm_mem_get_device_size(g_in.device_mem));
    bmrt_launch_tensor_ex(rt, greedy->name, &g_in, 1, &g_out, 1, true, false);
    bm_thread_sync(h);
    int token = 0;
    bm_memcpy_d2s(h, &token, g_out.device_mem);
    bm_free_device(h, lm_in.device_mem);  bm_free_device(h, lm_out.device_mem);
    bm_free_device(h, g_in.device_mem);   bm_free_device(h, g_out.device_mem);
    return token;
}

int Qwen3Bmodel::forward_first(const std::vector<float>& embeds_f32, int tlen) {
    token_length = tlen;
    const int S = MAX_INPUT_LEN, H = hidden_size_;
    const size_t kv_tlen_bytes = kv_token_bytes_ * tlen;  // 有效 KV 段（F16 时自动减半）

    // 1. pos + mask 预填一次（每层复用同一 device buffer）
    std::vector<int>   position_id(S, 0);
    std::vector<float> attention_mask((size_t)S * S, mask_value_f32_);
    for (int i = 0; i < tlen; i++) position_id[i] = i;
    for (int i = 0; i < tlen; i++)
        for (int j = 0; j <= i; j++)
            attention_mask[(size_t)i * S + j] = 0.0f;
    bm_memcpy_s2d(handle_, pre_pos_,   position_id.data());
    bm_memcpy_s2d(handle_, pre_mask_,  attention_mask.data());

    // 2. embeds → pre_hidden_a
    bm_memcpy_s2d(handle_, pre_hidden_a_, (void*)embeds_f32.data());

    // 3. 28 层：hidden ping-pong（device 内接力，无 host 中转）+ KV d2d 拷贝（免 host 往返）
    for (int i = 0; i < NUM_LAYERS; i++) {
        bm_device_mem_t& in_h  = (i % 2 == 0) ? pre_hidden_a_ : pre_hidden_b_;
        bm_device_mem_t& out_h = (i % 2 == 0) ? pre_hidden_b_ : pre_hidden_a_;
        auto net = net_blocks_[i];
        bm_tensor_t ins[3], outs[3];
        bmrt_tensor_with_device(&ins[0],  in_h,    net->input_dtypes[0],  net->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&ins[1],  pre_pos_, net->input_dtypes[1],  net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&ins[2],  pre_mask_, net->input_dtypes[2], net->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(&outs[0], out_h,   net->output_dtypes[0], net->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(&outs[1], pre_kv_, net->output_dtypes[1], net->stages[0].output_shapes[1]);
        bmrt_tensor_with_device(&outs[2], pre_v_,  net->output_dtypes[2], net->stages[0].output_shapes[2]);
        if (!bmrt_launch_tensor_ex(p_bmrt_, net->name, ins, 3, outs, 3, true, false)) {
            fprintf(stderr, "[Qwen3] block_%d launch failed\n", i);
            return -1;
        }
        // KV 有效段 d2d 进常驻 buffer（只拷 tlen 段；同一 handle 串行，无需额外 sync）
        bm_memcpy_d2d_byte(handle_, past_key_dev_[i],   0, pre_kv_, 0, kv_tlen_bytes);
        bm_memcpy_d2d_byte(handle_, past_value_dev_[i], 0, pre_v_,  0, kv_tlen_bytes);
    }
    bm_thread_sync(handle_);

    // 4. 最后一层输出（NUM_LAYERS 偶数 → pre_hidden_a）取最后一个有效 token
    bm_device_mem_t& last_h = (NUM_LAYERS % 2 == 0) ? pre_hidden_a_ : pre_hidden_b_;
    std::vector<float> last_hidden(H);
    bm_memcpy_d2s_partial_offset(handle_, last_hidden.data(), last_h, H * sizeof(float),
                                 (size_t)(tlen - 1) * H * sizeof(float));
    int token;
    if (net_greedy_head_) {
        token = lm_greedy(p_bmrt_, handle_, net_lm_, net_greedy_head_, last_hidden.data());
    } else {
        int vocab = net_lm_->stages[0].output_shapes[0].dims[2];
        std::vector<float> logits(vocab);
        launch_host(p_bmrt_, handle_, net_lm_, { last_hidden.data() }, { logits.data() });
        token = argmax_logits(logits);
    }
    cur_token_ = token;
    return token;
}

int Qwen3Bmodel::forward_next() {
    if (token_length >= SEQLEN) {
        fprintf(stderr, "[Qwen3] token_length=%d reached SEQLEN=%d\n", token_length, SEQLEN);
        return -1;
    }
    int cur_token = cur_token_;
    const int H = hidden_size_;
    // 修正 off-by-one：本步处理位置 token_length（0-indexed），写回 KV slot token_length
    int pos = token_length;

    {
        auto ec = net_embed_cache_;
        bm_tensor_t ein, eout;
        bmrt_tensor(&ein, p_bmrt_, ec->input_dtypes[0], ec->stages[0].input_shapes[0]);
        int32_t tok = cur_token;
        bm_memcpy_s2d(handle_, ein.device_mem, &tok);
        bmrt_tensor_with_device(&eout, dec_emb_out_, ec->output_dtypes[0], ec->stages[0].output_shapes[0]);
        bmrt_launch_tensor_ex(p_bmrt_, ec->name, &ein, 1, &eout, 1, true, false);
        bm_thread_sync(handle_);
        bm_free_device(handle_, ein.device_mem);
    }
    bm_memcpy_d2d_byte(handle_, dec_hidden_, 0, dec_emb_out_, 0, bm_mem_get_device_size(dec_hidden_));

    int32_t position_id = pos;
    bm_memcpy_s2d(handle_, dec_pos_, &position_id);
    // mask（与板上验证正确的 python 版一致）：
    //   0..pos-1 是已写入的 KV 槽位（可见），pos..SEQ-1 无效槽位屏蔽，
    //   SEQ 位（本次新 key）保持 0 可见。
    // 注意不能从 pos+1 起屏蔽——那会放行 pos 位的全零 KV 槽位导致 attention 错乱。
    std::vector<float> attention_mask(SEQLEN + 1, 0.0f);
    for (int i = pos; i < SEQLEN; i++) attention_mask[i] = mask_value_f32_;
    bm_memcpy_s2d(handle_, dec_mask_, attention_mask.data());

    for (int i = 0; i < NUM_LAYERS; i++) {
        auto net = net_blocks_cache_[i];
        bm_tensor_t ins[5], outs[3];
        bmrt_tensor_with_device(&ins[0], dec_hidden_,    net->input_dtypes[0], net->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&ins[1], dec_pos_,       net->input_dtypes[1], net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&ins[2], dec_mask_,      net->input_dtypes[2], net->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(&ins[3], past_key_dev_[i],   net->input_dtypes[3], net->stages[0].input_shapes[3]);
        bmrt_tensor_with_device(&ins[4], past_value_dev_[i], net->input_dtypes[4], net->stages[0].input_shapes[4]);
        bmrt_tensor_with_device(&outs[0], dec_hidden_,   net->output_dtypes[0], net->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(&outs[1], dec_newk_,     net->output_dtypes[1], net->stages[0].output_shapes[1]);
        bmrt_tensor_with_device(&outs[2], dec_newv_,     net->output_dtypes[2], net->stages[0].output_shapes[2]);
        bool ok = bmrt_launch_tensor_ex(p_bmrt_, net->name, ins, 5, outs, 3, true, false);
        if (!ok) { fprintf(stderr, "[Qwen3] block_cache_%d failed\n", i); return -1; }
        size_t dst_off = (size_t)pos * kv_token_bytes_;
        bm_memcpy_d2d_byte(handle_, past_key_dev_[i],   dst_off, dec_newk_, 0, kv_token_bytes_);
        bm_memcpy_d2d_byte(handle_, past_value_dev_[i], dst_off, dec_newv_, 0, kv_token_bytes_);
    }

    bm_thread_sync(handle_);
    std::vector<float> dh(H);
    bm_memcpy_d2s(handle_, dh.data(), dec_hidden_);
    int token;
    if (net_greedy_head_) {
        token = lm_greedy(p_bmrt_, handle_, net_lm_, net_greedy_head_, dh.data());
    } else {
        int vocab = net_lm_->stages[0].output_shapes[0].dims[2];
        std::vector<float> logits(vocab);
        launch_host(p_bmrt_, handle_, net_lm_, { dh.data() }, { logits.data() });
        token = argmax_logits(logits);
    }
    token_length++;
    cur_token_ = token;
    return token;
}

void Qwen3Bmodel::clear_kv() {
    int z = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        bm_memset_device_ext(handle_, &z, 1, past_key_dev_[i]);
        bm_memset_device_ext(handle_, &z, 1, past_value_dev_[i]);
    }
    token_length = 0;
}

// ══════════════════════════════════════════════════════════════════════════════
// AsrPipeline
// ══════════════════════════════════════════════════════════════════════════════

bool AsrPipeline::init(const std::string& encoder_path, const std::string& qwen3_path,
                       const std::string& model_dir, int device_id) {
    // 两个独立 handle（实测共享 handle 时 encoder 推理输出 NaN）
    if (bm_dev_request(&bm_handle_, device_id) != BM_SUCCESS) {
        fprintf(stderr, "[ASR] bm_dev_request failed\n");
        return false;
    }
    if (bm_dev_request(&bm_handle_enc_, device_id) != BM_SUCCESS) {
        fprintf(stderr, "[ASR] bm_dev_request (enc) failed\n");
        return false;
    }
    if (!qwen3_.init(bm_handle_, qwen3_path)) return false;
    {
        bm_dev_stat_t st;
        if (bm_get_stat(bm_handle_, &st) == BM_SUCCESS)
            printf("[ASR] devmem after LLM: used=%d/%d MB\n", st.mem_used, st.mem_total);
    }
    if (!enc_.init(bm_handle_enc_, encoder_path)) return false;
    {
        bm_dev_stat_t st;
        if (bm_get_stat(bm_handle_enc_, &st) == BM_SUCCESS)
            printf("[ASR] devmem after enc: used=%d/%d MB\n", st.mem_used, st.mem_total);
    }

    // 加载 prefix/suffix embeds（.bin → float 向量）
    auto load_bin = [](const std::string& path, std::vector<float>& out, int dim) -> int {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) return -1;
        fseek(fp, 0, SEEK_END);
        long fsize = ftell(fp);
        rewind(fp);
        out.resize(fsize / sizeof(float));
        if (fread(out.data(), sizeof(float), out.size(), fp) != out.size()) {
            fclose(fp); return -1;
        }
        fclose(fp);
        return (int)(out.size() / dim);
    };
    plen_ = load_bin(model_dir + "/prefix_embeds.bin", prefix_embeds_, cfg_.hidden_size);
    slen_ = load_bin(model_dir + "/suffix_embeds.bin", suffix_embeds_, cfg_.hidden_size);
    if (plen_ <= 0 || slen_ <= 0) {
        fprintf(stderr, "[ASR] prefix/suffix embeds load failed (plen=%d slen=%d)\n", plen_, slen_);
        return false;
    }
    printf("[ASR] prefix %d tokens, suffix %d tokens\n", plen_, slen_);

    // mel 滤波器 + tokenizer
    if (!mel_.load_filters(model_dir + "/mel_filters.npz")) return false;
    if (!tok_.load(model_dir)) {   // load 内部拼 /tokenizer.json
        fprintf(stderr, "[ASR] tokenizer load failed\n");
        return false;
    }
    printf("[ASR] tokenizer loaded\n");
    loaded_ = true;
    return true;
}

int AsrPipeline::mel_frames_to_tokens(int t_real_mel) {
    // 与 processor._get_audio_token_length 一致：
    //   每完整 chunk(100 帧) → 13 token；末尾不完整 chunk 3 次 stride2 conv 折算
    if (t_real_mel <= 0) return 0;
    int chunk_len = 100;
    int n_chunks = (t_real_mel + chunk_len - 1) / chunk_len;
    int t = t_real_mel % chunk_len;
    int feat = (t > 0) ? ((t - 1) / 2 + 1) : 13;
    feat = (feat - 1) / 2 + 1;
    feat = (feat - 1) / 2 + 1;
    return feat + (n_chunks - 1) * 13;
}

void AsrPipeline::deinit() {
    qwen3_.deinit();
    enc_.release();
    if (bm_handle_enc_) { bm_dev_free(bm_handle_enc_); bm_handle_enc_ = nullptr; }
    if (bm_handle_) { bm_dev_free(bm_handle_); bm_handle_ = nullptr; }
    loaded_ = false;
}

bool AsrPipeline::build_inputs_embeds(const std::vector<float>& audio_embeds, int alen,
                                      std::vector<float>& embeds, int& tlen) const {
    const int S = cfg_.seq_length, H = cfg_.hidden_size;
    tlen = plen_ + alen + slen_;
    if (tlen > S) { fprintf(stderr, "[ASR] tlen %d > SEQ %d\n", tlen, S); return false; }
    embeds.assign((size_t)S * H, 0.0f);
    memcpy(embeds.data(), prefix_embeds_.data(), (size_t)plen_ * H * sizeof(float));
    memcpy(embeds.data() + (size_t)plen_ * H, audio_embeds.data(), (size_t)alen * H * sizeof(float));
    memcpy(embeds.data() + (size_t)(plen_ + alen) * H, suffix_embeds_.data(),
           (size_t)slen_ * H * sizeof(float));
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// 流式推理（对齐官方：音频 1s chunk，最近 5s（500 mel 帧）重编码窗口）
// ══════════════════════════════════════════════════════════════════════════════

bool AsrPipeline::init_stream(const std::string& encoder_w500_path) {
    if (stream_inited_) return true;
    if (!enc_w500_.init(bm_handle_enc_, encoder_w500_path, "qwen3_asr_encoder_w500")) {
        fprintf(stderr, "[ASR] stream encoder init failed\n");
        return false;
    }
    stream_inited_ = true;
    printf("[ASR] stream encoder (w500) loaded\n");
    return true;
}

void AsrPipeline::stream_push(const float* samples, int n) {
    stream_samples_.insert(stream_samples_.end(), samples, samples + n);
}

// 用 mel 缓存重编码最近 win 帧窗口（win ≤ 500，起点对齐 100），返回窗口输出 token 数
// 输出累积到 stream_audio_tokens_（替换最后 win/100*13 帧，保留更早固定帧）
int AsrPipeline::reencode_window(int win) {
    // 起点必须对齐 100：T 截断到 100 倍数（尾巴帧不算，finish 单独处理）
    int T = stream_mel_frames_ / 100 * 100;
    if (T < win) win = T;   // 总长不足窗口
    int start = T - win;   // T 与 win 都是 100 倍数 → 起点对齐
    std::vector<float> mel_t((size_t)128 * win);
    for (int t = 0; t < win; ++t)
        for (int m = 0; m < 128; ++m)
            mel_t[(size_t)m * win + t] = stream_mel_[(size_t)(start + t) * 128 + m];
    std::vector<float> win_out;
    if (!enc_w500_.run(mel_t.data(), win_out)) return -1;

    int win_tokens = win / 100 * 13;
    if (T <= win) {
        stream_audio_tokens_.assign(win_out.begin(), win_out.begin() + (size_t)win_tokens * 1024);
    } else {
        int keep = stream_audio_count_ - win_tokens;
        stream_audio_tokens_.resize((size_t)keep * 1024);
        stream_audio_tokens_.insert(stream_audio_tokens_.end(),
                                    win_out.begin(), win_out.begin() + (size_t)win_tokens * 1024);
    }
    stream_audio_count_ = (T <= win) ? win_tokens : stream_audio_count_;
    return win_tokens;
}

std::string AsrPipeline::stream_update(int max_new_tokens) {
    if (!stream_inited_ || stream_samples_.empty()) return "";

    // 1. mel 增量：只算新帧（FFT+filter+log10），然后全局重裁剪（max-8 + 缩放）
    int n_frames = 0;
    mel_.log_mel_frames(stream_samples_, stream_mel_frames_, stream_mel_log_, n_frames);
    if (n_frames > stream_mel_frames_) {
        stream_mel_frames_ = n_frames;
        stream_mel_.resize((size_t)n_frames * 128);
        float gmax = -1e30f;
        for (float v : stream_mel_log_) if (v > gmax) gmax = v;
        for (size_t i = 0; i < stream_mel_log_.size(); ++i) {
            float v = stream_mel_log_[i];
            if (v < gmax - 8.0f) v = gmax - 8.0f;
            stream_mel_[i] = (v + 4.0f) / 4.0f;
        }
    }

    // 2. 只处理完整 chunk（T 截断到 100 倍数，保证窗口起点对齐）；尾巴留到 finish
    int T_full = stream_mel_frames_ / 100 * 100;
    if (T_full == 0) return "";
    int win = std::min(T_full, 500);
    if (reencode_window(win) < 0) return "";

    // 3. 音频过短（<3s）不生成（模型在 audio tokens 太少时输出不可靠，官方同款延迟策略）
    if (stream_audio_count_ < 39) return "";

    // 4. prefill + 短生成 → 中间转写
    return generate_from_audio(max_new_tokens);
}

std::string AsrPipeline::stream_finish(int max_new_tokens) {
    if (!stream_inited_) return "";
    // 定稿：用离线 encoder 对全部音频全量重编码（精度与离线版一致；尾巴也自然处理）
    int T_real = 0;
    std::vector<float> mel = mel_.log_mel_spectrogram(stream_samples_, T_real);
    if (T_real <= 0) return "";
    // pad 到 3000（复制最后一帧）
    int copy = std::min(T_real, 3000);
    std::vector<float> mel_pad((size_t)3000 * 128);
    if (copy > 0) {
        for (int t = 0; t < copy; ++t)
            for (int m = 0; m < 128; ++m)
                mel_pad[(size_t)m * 3000 + t] = mel[(size_t)t * 128 + m];
        for (int t = copy; t < 3000; ++t)
            for (int m = 0; m < 128; ++m)
                mel_pad[(size_t)m * 3000 + t] = mel[(size_t)(copy - 1) * 128 + m];
    }
    std::vector<float> audio_full;
    if (!enc_.run(mel_pad.data(), audio_full)) return "";
    int alen = mel_frames_to_tokens(T_real);
    if (alen > 390) alen = 390;
    stream_audio_tokens_.assign(audio_full.begin(), audio_full.begin() + (size_t)alen * 1024);
    stream_audio_count_ = alen;
    // 完整生成定稿
    return generate_from_audio(max_new_tokens);
}

std::string AsrPipeline::generate_from_audio(int max_new_tokens) {
    const int H = cfg_.hidden_size;
    int alen = stream_audio_count_;
    if (alen == 0) return "";
    // 截取有效段（window encoder 输出可能含 65 帧上限；alen 是精确的）
    std::vector<float> audio(stream_audio_tokens_.begin(),
                             stream_audio_tokens_.begin() + (size_t)alen * H);

    std::vector<float> embeds;
    int tlen = 0;
    if (!build_inputs_embeds(audio, alen, embeds, tlen)) return "";
    qwen3_.clear_kv();
    int cur = qwen3_.forward_first(embeds, tlen);
    if (cur < 0) return "";

    std::vector<int> result;
    const int EOS1 = cfg_.eos_im_end, EOS2 = cfg_.eos_eot;
    while ((int)result.size() < max_new_tokens) {
        if (cur == EOS1 || cur == EOS2) break;
        result.push_back(cur);
        cur = qwen3_.forward_next();
        if (cur < 0) break;
    }
    return tok_.decode(result);
}

std::string AsrPipeline::transcribe(const std::string& wav_path, int max_new_tokens) {
    if (!loaded_) return "";
    auto t0 = std::chrono::steady_clock::now();
    auto t_mel = t0, t_enc = t0, t_pre = t0;

    // ── 1. mel：wav → [3000,128]（帧-major）→ 转置 [128,3000] ──
    std::vector<float> mel;
    int t_real = 0;
    if (!mel_.wav_to_mel(wav_path, mel, t_real)) return "";
    // 转置 [3000,128] → [128,3000]（encoder bmodel 输入布局）
    std::vector<float> mel_t(128 * 3000);
    for (int t = 0; t < 3000; ++t)
        for (int m = 0; m < 128; ++m)
            mel_t[(size_t)m * 3000 + t] = mel[(size_t)t * 128 + m];

    // ── 2. encoder → audio_embeds [390,1024]，截取真实帧 ──
    t_mel = std::chrono::steady_clock::now();
    std::vector<float> audio_embeds_full;
    if (!enc_.run(mel_t.data(), audio_embeds_full)) return "";
    t_enc = std::chrono::steady_clock::now();
    int alen = mel_frames_to_tokens(t_real);
    if (alen > 390) alen = 390;
    const int H = cfg_.hidden_size;
    std::vector<float> audio_embeds(audio_embeds_full.begin(),
                                    audio_embeds_full.begin() + (size_t)alen * H);

    // ── 3. 拼 embeds → prefill → decode ──
    std::vector<float> embeds;
    int tlen = 0;
    if (!build_inputs_embeds(audio_embeds, alen, embeds, tlen)) return "";
    qwen3_.clear_kv();
    int cur = qwen3_.forward_first(embeds, tlen);
    t_pre = std::chrono::steady_clock::now();
    if (cur < 0) return "";

    std::vector<int> result;
    const int EOS1 = cfg_.eos_im_end, EOS2 = cfg_.eos_eot;
    while ((int)result.size() < max_new_tokens) {
        if (cur == EOS1 || cur == EOS2) break;
        result.push_back(cur);
        cur = qwen3_.forward_next();
        if (cur < 0) break;
    }

    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    double d_mel = std::chrono::duration<double>(t_mel - t0).count();
    double d_enc = std::chrono::duration<double>(t_enc - t_mel).count();
    double d_pre = std::chrono::duration<double>(t_pre - t_enc).count();
    double d_dec = std::chrono::duration<double>(t1 - t_pre).count();
    fprintf(stderr, "[ASR] %d tokens, total %.2fs (mel %.2f + enc %.2f + prefill %.2f + decode %.2f, %.1f tok/s)\n",
            (int)result.size(), dt, d_mel, d_enc, d_pre, d_dec,
            result.empty() ? 0 : result.size() / (d_dec > 0 ? d_dec : 1));

    // ── 4. 解码 ──
    return tok_.decode(result);
}

}  // namespace asr
