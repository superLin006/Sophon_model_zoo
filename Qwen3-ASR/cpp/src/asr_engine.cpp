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
    p_bmrt_ = nullptr;   // 主 bmrt 由 Qwen3Bmodel 管理，不 destroy
    inited_ = false;
}

bool EncoderBmodel::init(void* p_bmrt, const char* net_name) {
    p_bmrt_ = p_bmrt;   // 主 bmrt（单文件已加载全部网络，不独立加载）
    net_ = bmrt_get_network_info(p_bmrt_, net_name);
    if (!net_) {
        fprintf(stderr, "[Enc] network '%s' not found\n", net_name);
        p_bmrt_ = nullptr;
        return false;
    }
    inited_ = true;
    printf("[Enc] loaded (net=%s)\n", net_name);
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
    // past_key_dev_/past_value_dev_ 是网络预分配 mem（addr mode 1），不能 free
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
    net_embed_        = bmrt_get_network_info(p_bmrt_, "embedding");
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
    // mask -inf 值按输入 dtype：fp16=0xF0E2，bf16=0xC61C（QwenEngine 规则，混用会致屏蔽失效）
    mask_bf16_ = (net_blocks_[0]->input_dtypes[2] == BM_FLOAT16) ? 0xF0E2 : 0xC61C;
    // addr mode 1：KV 常驻 buffer 直接用网络预分配的 input_mems[3]/[4]（QwenEngine 同款）
    past_key_dev_.resize(NUM_LAYERS);
    past_value_dev_.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        past_key_dev_[i]   = net_blocks_cache_[i]->stages[0].input_mems[3];
        past_value_dev_[i] = net_blocks_cache_[i]->stages[0].input_mems[4];
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

// fp32 → bf16（四舍五入，Eureka 同款）
static std::vector<uint16_t> fp32_to_bf16_vec(const float* data, int n) {
    std::vector<uint16_t> out(n);
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &data[i], 4);
        u += 0x7FFF + ((u >> 16) & 1);
        out[i] = (uint16_t)(u >> 16);
    }
    return out;
}
// bf16 → fp32
static void bf16_to_fp32_vec(const uint16_t* in, int n, float* out) {
    for (int i = 0; i < n; i++) {
        uint32_t u = (uint32_t)in[i] << 16;
        float f; memcpy(&f, &u, 4);
        out[i] = f;
    }
}

int Qwen3Bmodel::forward_first(const std::vector<int>& input_ids, int tlen,
                               const std::vector<float>& audio_embeds, int audio_start) {
    token_length = tlen;
    const int S = MAX_INPUT_LEN, H = hidden_size_;
    const size_t kv_tlen_bytes = kv_token_bytes_ * tlen;  // 有效 KV 段（bf16 自动减半）

    // 1. pos + mask 预填一次（mask -inf 值按 dtype：fp16=0xF0E2，bf16=0xC61C）
    std::vector<int>   position_id(S, 0);
    std::vector<uint16_t> attention_mask((size_t)S * S, mask_bf16_);
    for (int i = 0; i < tlen; i++) position_id[i] = i;
    for (int i = 0; i < tlen; i++)
        for (int j = 0; j <= i; j++)
            attention_mask[(size_t)i * S + j] = 0;
    bm_memcpy_s2d(handle_, pre_pos_,  position_id.data());
    bm_memcpy_s2d(handle_, pre_mask_, attention_mask.data());

    // 2. embedding 网络：input_ids → 文本 embeds（bf16）→ host 转 f32（addr mode 1：用 input_mems）
    std::vector<float> embeds((size_t)S * H, 0.0f);
    if (net_embed_) {
        auto& e_in  = net_embed_->stages[0].input_mems[0];
        auto& e_out = net_embed_->stages[0].output_mems[0];
        std::vector<int32_t> ids_pad(S, 0);
        for (int i = 0; i < tlen && i < (int)input_ids.size(); i++) ids_pad[i] = input_ids[i];
        bm_memcpy_s2d(handle_, e_in, ids_pad.data());
        bm_tensor_t ein, eout;
        bmrt_tensor_with_device(&ein,  e_in,  net_embed_->input_dtypes[0],  net_embed_->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&eout, e_out, net_embed_->output_dtypes[0], net_embed_->stages[0].output_shapes[0]);
        if (!bmrt_launch_tensor_ex(p_bmrt_, net_embed_->name, &ein, 1, &eout, 1, true, false)) {
            fprintf(stderr, "[Qwen3] embedding launch failed\n"); return -1;
        }
        bm_thread_sync(handle_);
        std::vector<uint16_t> e_bf16((size_t)tlen * H);
        bm_memcpy_d2s_partial_offset(handle_, e_bf16.data(), e_out,
                                     (size_t)tlen * H * 2, 0);
        bf16_to_fp32_vec(e_bf16.data(), (size_t)tlen * H, embeds.data());
    }

    // 3. audio embeds 替换占位位置
    if (!audio_embeds.empty()) {
        memcpy(embeds.data() + (size_t)audio_start * H, audio_embeds.data(),
               audio_embeds.size() * sizeof(float));
    }

    // 4. embeds（f32）→ bf16 → dev buffer（层间 d2d 中转）
    auto emb_bf16 = fp32_to_bf16_vec(embeds.data(), (size_t)S * H);
    bm_memcpy_s2d(handle_, pre_hidden_a_, emb_bf16.data());

    // 5. 28 层（addr mode 1：用网络预分配 input_mems；每层 launch 后必须 sync——
    //    bmrt_launch_tensor_ex 是异步的，CPU 不被阻塞）
    const size_t hidden_t_bytes = (size_t)tlen * H * 2;   // bf16 有效段
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto net = net_blocks_[i];
        auto& in0 = net->stages[0].input_mems[0];
        auto& in1 = net->stages[0].input_mems[1];
        auto& in2 = net->stages[0].input_mems[2];
        if (i == 0) {
            bm_memcpy_s2d(handle_, in0, emb_bf16.data());
        }
        if (i == 0) {   // pos/mask 各层共享同一 input_mem（已验证地址相同），只传一次
            bm_memcpy_s2d(handle_, in1, position_id.data());
            bm_memcpy_s2d(handle_, in2, attention_mask.data());
        }
        bm_tensor_t ins[3], outs[3];
        bmrt_tensor_with_device(&ins[0],  in0,  net->input_dtypes[0],  net->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&ins[1],  in1,  net->input_dtypes[1],  net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&ins[2],  in2,  net->input_dtypes[2],  net->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(&outs[0], net->stages[0].output_mems[0], net->output_dtypes[0], net->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(&outs[1], net->stages[0].output_mems[1], net->output_dtypes[1], net->stages[0].output_shapes[1]);
        bmrt_tensor_with_device(&outs[2], net->stages[0].output_mems[2], net->output_dtypes[2], net->stages[0].output_shapes[2]);
        if (!bmrt_launch_tensor_ex(p_bmrt_, net->name, ins, 3, outs, 3, true, false)) {
            fprintf(stderr, "[Qwen3] block_%d launch failed\n", i);
            return -1;
        }
        bm_thread_sync(handle_);
        // 层间 d2d：输出 → 下一层 in0（QwenEngine 同款；d2d 异步，sync 后再 launch）
        if (i + 1 < NUM_LAYERS) {
            auto& next_in0 = net_blocks_[i + 1]->stages[0].input_mems[0];
            bm_memcpy_d2d_byte(handle_, next_in0, 0, net->stages[0].output_mems[0], 0, hidden_t_bytes);
            bm_thread_sync(handle_);
        }
        // KV 有效段 d2d 进常驻 buffer
        bm_memcpy_d2d_byte(handle_, past_key_dev_[i],   0, net->stages[0].output_mems[1], 0, kv_tlen_bytes);
        bm_memcpy_d2d_byte(handle_, past_value_dev_[i], 0, net->stages[0].output_mems[2], 0, kv_tlen_bytes);
    }
    bm_thread_sync(handle_);

    // 6. lm_head：最后一层输出最后一个有效 token（d2d 到 lm_head 输入）→ token_id
    const bm_net_info_t* last_net = net_blocks_[NUM_LAYERS - 1];
    auto& lm_in  = net_lm_->stages[0].input_mems[0];
    auto& lm_out = net_lm_->stages[0].output_mems[0];
    bm_memcpy_d2d_byte(handle_, lm_in, 0, last_net->stages[0].output_mems[0],
                       (size_t)(tlen - 1) * H * 2, (size_t)H * 2);
    bm_thread_sync(handle_);
    bm_tensor_t lt_in, lt_out;
    bmrt_tensor_with_device(&lt_in,  lm_in,  net_lm_->input_dtypes[0],  net_lm_->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(&lt_out, lm_out, net_lm_->output_dtypes[0], net_lm_->stages[0].output_shapes[0]);
    if (!bmrt_launch_tensor_ex(p_bmrt_, net_lm_->name, &lt_in, 1, &lt_out, 1, true, false)) {
        fprintf(stderr, "[Qwen3] lm_head launch failed\n");
        return -1;
    }
    bm_thread_sync(handle_);
    // 标准 bmodel：lm_head 输出直接是 token_id（int32，内置 argmax）
    int token = 0;
    bm_memcpy_d2s(handle_, &token, lm_out);
    cur_token_ = token;
    return token;
}

// hidden（f32 host）→ bf16 → lm_head → argmax
int Qwen3Bmodel::lm_head_argmax(const float* hidden_f32) {
    auto h_bf16 = fp32_to_bf16_vec(hidden_f32, hidden_size_);
    int vocab = net_lm_->stages[0].output_shapes[0].dims[2];
    std::vector<float> logits(vocab);
    launch_host(p_bmrt_, handle_, net_lm_, { h_bf16.data() }, { logits.data() });
    return argmax_logits(logits);
}

int Qwen3Bmodel::forward_next() {
    if (token_length >= SEQLEN) {
        fprintf(stderr, "[Qwen3] token_length=%d reached SEQLEN=%d\n", token_length, SEQLEN);
        return -1;
    }
    int cur_token = cur_token_;
    const int H = hidden_size_;
    int pos = token_length;   // 修正 off-by-one：本步处理位置 token_length

    // decode mask（按 dtype 选 -inf 值）：0..pos 与 SEQ 位（新 key）可见
    std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
    for (int i = pos + 1; i < SEQLEN; i++) attention_mask[i] = mask_bf16_;   // 无效槽位屏蔽
    int32_t position_id = pos;

    // embedding_cache：token ids → hidden（网络 mem 流转）
    auto ec = net_embed_cache_;
    auto& e_in  = ec->stages[0].input_mems[0];
    auto& e_out = ec->stages[0].output_mems[0];
    bm_memcpy_s2d(handle_, e_in, (void*)&cur_token);
    bm_tensor_t ein, eout;
    bmrt_tensor_with_device(&ein,  e_in,  ec->input_dtypes[0],  ec->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(&eout, e_out, ec->output_dtypes[0], ec->stages[0].output_shapes[0]);
    if (!bmrt_launch_tensor_ex(p_bmrt_, ec->name, &ein, 1, &eout, 1, true, false)) return -1;
    bm_thread_sync(handle_);

    // 28 层 block_cache（addr mode 1：全用网络 input_mems）
    const size_t kv_off = (size_t)pos * kv_token_bytes_;   // 新 KV 写回偏移
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto net = net_blocks_cache_[i];
        auto& in1 = net->stages[0].input_mems[1];
        auto& in2 = net->stages[0].input_mems[2];
        auto& in3 = net->stages[0].input_mems[3];
        auto& in4 = net->stages[0].input_mems[4];
        auto& out0 = net->stages[0].output_mems[0];
        if (i == 0) {
            bm_memcpy_s2d(handle_, in1, (void*)&position_id);
            bm_memcpy_s2d(handle_, in2, (void*)attention_mask.data());
        }
        bm_tensor_t ins[5], outs[3];
        // in0 = 上一层的输出（e_out 或 out0，网络 mem）
        bmrt_tensor_with_device(&ins[0], (i == 0) ? e_out : net_blocks_cache_[i-1]->stages[0].output_mems[0],
                                net->input_dtypes[0], net->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&ins[1], net_blocks_cache_[0]->stages[0].input_mems[1],
                                net->input_dtypes[1], net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&ins[2], net_blocks_cache_[0]->stages[0].input_mems[2],
                                net->input_dtypes[2], net->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(&ins[3], in3, net->input_dtypes[3], net->stages[0].input_shapes[3]);
        bmrt_tensor_with_device(&ins[4], in4, net->input_dtypes[4], net->stages[0].input_shapes[4]);
        bmrt_tensor_with_device(&outs[0], out0, net->output_dtypes[0], net->stages[0].output_shapes[0]);
        // 新 KV 直接写进累积 buffer 的偏移（bm_mem_from_device 地址偏移）
        bm_device_mem_t k_mem = bm_mem_from_device(in3.u.device.device_addr + kv_off, kv_token_bytes_);
        bm_device_mem_t v_mem = bm_mem_from_device(in4.u.device.device_addr + kv_off, kv_token_bytes_);
        bmrt_tensor_with_device(&outs[1], k_mem, net->output_dtypes[1], net->stages[0].output_shapes[1]);
        bmrt_tensor_with_device(&outs[2], v_mem, net->output_dtypes[2], net->stages[0].output_shapes[2]);
        if (!bmrt_launch_tensor_ex(p_bmrt_, net->name, ins, 5, outs, 3, true, false)) {
            fprintf(stderr, "[Qwen3] block_cache_%d failed\n", i);
            return -1;
        }
        // launch 异步：下一层输入依赖本层输出，必须 sync 后再 launch
        bm_thread_sync(handle_);
    }

    // lm_head：最后一层输出 → logits → argmax
    auto& lm_in  = net_lm_->stages[0].input_mems[0];
    auto& lm_out = net_lm_->stages[0].output_mems[0];
    auto& last_out = net_blocks_cache_[NUM_LAYERS-1]->stages[0].output_mems[0];
    bm_memcpy_d2d_byte(handle_, lm_in, 0, last_out, 0, (size_t)H * 2);
    bm_thread_sync(handle_);   // d2d 异步，必须 sync 后再 launch lm_head
    bm_tensor_t lt_in, lt_out;
    bmrt_tensor_with_device(&lt_in,  lm_in,  net_lm_->input_dtypes[0],  net_lm_->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(&lt_out, lm_out, net_lm_->output_dtypes[0], net_lm_->stages[0].output_shapes[0]);
    if (!bmrt_launch_tensor_ex(p_bmrt_, net_lm_->name, &lt_in, 1, &lt_out, 1, true, false)) return -1;
    bm_thread_sync(handle_);
    // 标准 bmodel：lm_head 输出直接是 token_id（int32）
    int token = 0;
    bm_memcpy_d2s(handle_, &token, lm_out);
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

bool AsrPipeline::init(const std::string& bmodel_path, const std::string& model_dir,
                       int device_id) {
    if (bm_dev_request(&bm_handle_, device_id) != BM_SUCCESS) {
        fprintf(stderr, "[ASR] bm_dev_request failed\n");
        return false;
    }
    if (!qwen3_.init(bm_handle_, bmodel_path)) return false;
    {
        bm_dev_stat_t st;
        if (bm_get_stat(bm_handle_, &st) == BM_SUCCESS)
            printf("[ASR] devmem after LLM: used=%d/%d MB\n", st.mem_used, st.mem_total);
    }
    // encoder 复用 LLM 的 bmrt（单文件已加载，单份 device 内存）
    if (!enc_.init(qwen3_.bmrt())) return false;
    {
        bm_dev_stat_t st;
        if (bm_get_stat(bm_handle_, &st) == BM_SUCCESS)
            printf("[ASR] devmem after enc: used=%d/%d MB\n", st.mem_used, st.mem_total);
    }

    // 加载 prefix/suffix token ids（标准 Qwen3 bmodel：input_ids + embedding 网络）
    auto load_ids = [](const std::string& path, std::vector<int>& out) -> bool {
        FILE* fp = fopen(path.c_str(), "r");
        if (!fp) return false;
        int id;
        while (fscanf(fp, "%d", &id) == 1) out.push_back(id);
        fclose(fp);
        return !out.empty();
    };
    if (!load_ids(model_dir + "/prefix_ids.txt", prefix_ids_) ||
        !load_ids(model_dir + "/suffix_ids.txt", suffix_ids_)) {
        fprintf(stderr, "[ASR] prefix/suffix ids load failed\n");
        return false;
    }
    plen_ = (int)prefix_ids_.size();
    slen_ = (int)suffix_ids_.size();
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
    //   完整 chunk(100 帧) → 13 token；末尾不完整 chunk 才做 3 次 stride2 conv 折算
    //   （t==0 时若再折算会把 13 → 4，整数秒音频 alen 算错）
    if (t_real_mel <= 0) return 0;
    int chunk_len = 100;
    int n_chunks = (t_real_mel + chunk_len - 1) / chunk_len;
    int t = t_real_mel % chunk_len;
    int feat;
    if (t > 0) {
        feat = (t - 1) / 2 + 1;
        feat = (feat - 1) / 2 + 1;
        feat = (feat - 1) / 2 + 1;
    } else {
        feat = 13;
    }
    return feat + (n_chunks - 1) * 13;
}

void AsrPipeline::deinit() {
    qwen3_.deinit();
    enc_.release();
    if (bm_handle_) { bm_dev_free(bm_handle_); bm_handle_ = nullptr; }
    loaded_ = false;
}

int AsrPipeline::build_input_ids(int alen, std::vector<int>& ids_out, int& tlen) const {
    const int S = qwen3_.MAX_INPUT_LEN;   // bmodel 实际 seq（cfg_.seq_length=2048 是旧配置残留，勿用）
    tlen = plen_ + alen + slen_;
    if (tlen > S) { fprintf(stderr, "[ASR] tlen %d > SEQ %d (音频约 %.1fs 超上限)\n", tlen, S, alen / 13.0); return -1; }
    ids_out.assign(S, 0);
    int p = 0;
    for (int id : prefix_ids_) ids_out[p++] = id;
    for (int i = 0; i < alen; i++) ids_out[p++] = cfg_.audio_token_id;   // <|audio_pad|>
    for (int id : suffix_ids_) ids_out[p++] = id;
    return plen_;   // audio 段起始位置
}

// ══════════════════════════════════════════════════════════════════════════════
// 流式推理（无 VAD：音频持续灌入 + 全量重编码 + 全量 prefill，对齐官方 Python 语义）
// ══════════════════════════════════════════════════════════════════════════════

bool AsrPipeline::init_stream() {
    stream_inited_ = true;   // 全量 encoder 就是主 encoder（enc_），无额外初始化
    return true;
}

void AsrPipeline::stream_push(const float* samples, int n) {
    stream_samples_.insert(stream_samples_.end(), samples, samples + n);
}

// 在 center 两侧 ±search 帧内找**最近**的静音窗口中心（先向前再向后，同距离优先向前）：
// mel_log（log10 值）10 帧窗口帧均值 < -7 判静音（语音帧 log10 ≈ -3~0，静音帧 ≈ -10）。
// 最近优先保证段长均匀（避免跳到远处最安静处）；找不到返回 center（原样截断）。
int AsrPipeline::find_silence_center(const std::vector<float>& mel_log, int T,
                                     int center, int search) {
    auto is_silence = [&](int c) -> bool {
        if (c < 10 || c >= T - 10) return false;
        float sum = 0; int n = 0;
        for (int t = c - 5; t < c + 5; ++t)
            for (int m = 0; m < 128; m += 8) {   // 采样 16 个 bin 加速
                sum += mel_log[(size_t)t * 128 + m]; n++;
            }
        return (sum / n) < -7.0f;
    };
    for (int d = 0; d <= search; ++d) {
        if (center - d >= 10 && is_silence(center - d)) return center - d;
        if (center + d < T - 10 && is_silence(center + d)) return center + d;
    }
    return center;
}

// mel（[T,128] t-major，T 为真实帧数）→ pad 到 3000（复制尾帧）→ 全量 encoder
// → 输出 [390,1024] → 截取真实帧对应 tokens（mel_frames_to_tokens）
bool AsrPipeline::encode_full_mel(const std::vector<float>& mel, int T,
                                  std::vector<float>& audio_out, int& alen) {
    const int H = cfg_.hidden_size;
    if (T <= 0) return false;
    int copy = std::min(T, 3000);
    std::vector<float> mel_t((size_t)128 * 3000);
    for (int t = 0; t < copy; ++t)
        for (int m = 0; m < 128; ++m)
            mel_t[(size_t)m * 3000 + t] = mel[(size_t)t * 128 + m];
    for (int t = copy; t < 3000; ++t)
        for (int m = 0; m < 128; ++m)
            mel_t[(size_t)m * 3000 + t] = mel[(size_t)(copy - 1) * 128 + m];
    if (!enc_.run(mel_t.data(), audio_out)) return false;
    alen = mel_frames_to_tokens(T);
    if (alen > 390) alen = 390;
    audio_out.resize((size_t)alen * H);   // 只保留真实帧（forward_first 按 size 替换，全量会越界写）
    return true;
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

    // 2. 全量 mel 重编码（含尾巴；encoder 固定 3000 帧输入 0.08s，1s 块下实时）
    if (stream_mel_frames_ <= 0) return "";
    std::vector<float> audio_full;
    int alen = 0;
    if (!encode_full_mel(stream_mel_, stream_mel_frames_, audio_full, alen)) return "";
    stream_audio_tokens_.assign(audio_full.begin(), audio_full.begin() + (size_t)alen * 1024);
    stream_audio_count_ = alen;

    // 3. 分段定稿：audio tokens 将超 KV 上限（留 128 槽给生成）→ 定稿当前段并重置，
    //    支持无限持续流式（每段 ~28s，历史文本保留在 stream_partial_text_）
    const int MAX_AUDIO = qwen3_.MAX_INPUT_LEN - 15 - 128;
    if (alen > MAX_AUDIO) {
        int T_seg = stream_mel_frames_;
        while (T_seg > 0 && mel_frames_to_tokens(T_seg) > MAX_AUDIO) T_seg -= 100;
        // 分段点对齐最近静音（双向 ±1.5s：避免截断在词中间 → 段边界自然衔接）；
        // 向后对齐若挤占生成空间（tokens > SEQ-15-64）则回退 T_seg（原样截断）
        int cand = find_silence_center(stream_mel_log_, (int)stream_mel_log_.size() / 128, T_seg);
        if (cand > T_seg && mel_frames_to_tokens(cand) > qwen3_.MAX_INPUT_LEN - 15 - 64) cand = T_seg;
        T_seg = cand;
        if (T_seg > 0) {
            std::vector<float> seg_full;
            int seg_alen = 0;
            if (encode_full_mel(stream_mel_, T_seg, seg_full, seg_alen)) {
                stream_audio_tokens_.assign(seg_full.begin(), seg_full.begin() + (size_t)seg_alen * 1024);
                stream_audio_count_ = seg_alen;
                std::string seg = generate_from_audio(256);   // 定稿段：完整生成
                if (!seg.empty()) { stream_partial_text_ += seg; stream_partial_text_ += "\n"; }
                fprintf(stderr, "[ASR] 分段定稿 @%.1fs, 历史 %zu 字符\n",
                        T_seg / 100.0, stream_partial_text_.size());
            }
        }
        // 重置：samples 截断到剩余部分（新段从 T_seg 之后的帧开始），
        // mel 缓存清空（samples 起点已变，重新从 0 算）
        int cut_samples = T_seg * 160;
        if (cut_samples > 0 && cut_samples < (int)stream_samples_.size())
            stream_samples_.erase(stream_samples_.begin(), stream_samples_.begin() + cut_samples);
        stream_mel_log_.clear();
        stream_mel_.clear();
        stream_mel_frames_ = 0;
        stream_audio_tokens_.clear();
        stream_audio_count_ = 0;
        return stream_partial_text_;
    }

    // 4. 音频过短（<1s=13 tokens）不生成：官方 Python 实测 1s 可转写、0.5s 破碎，
    //    故 1s 为最低可靠阈值；更短的语音由 finish 兜底（VAD 结束即定稿，不吞语音）
    if (stream_audio_count_ < 13) return stream_partial_text_;

    // 5. prefill + 短生成 → 历史 + 当前段中间转写
    return stream_partial_text_ + generate_from_audio(max_new_tokens);
}

std::string AsrPipeline::stream_finish(int max_new_tokens) {
    if (!stream_inited_) return stream_partial_text_;
    // 定稿最后一段：与 update 同一路径（当前段 mel 全量重编码），分段后历史已在 stream_partial_text_
    if (stream_mel_frames_ <= 0) return stream_partial_text_;
    std::vector<float> audio_full;
    int alen = 0;
    if (!encode_full_mel(stream_mel_, stream_mel_frames_, audio_full, alen)) return stream_partial_text_;
    stream_audio_tokens_.assign(audio_full.begin(), audio_full.begin() + (size_t)alen * 1024);
    stream_audio_count_ = alen;
    return stream_partial_text_ + generate_from_audio(max_new_tokens);
}

std::string AsrPipeline::generate_from_audio(int max_new_tokens) {
    const int H = cfg_.hidden_size;
    int alen = stream_audio_count_;
    if (alen == 0) return "";
    // 截取有效段（window encoder 输出可能含 65 帧上限；alen 是精确的）
    std::vector<float> audio(stream_audio_tokens_.begin(),
                             stream_audio_tokens_.begin() + (size_t)alen * H);

    std::vector<int> ids;
    int tlen = 0;
    int audio_start = build_input_ids(alen, ids, tlen);
    if (audio_start < 0) {
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "[ASR] 流式超 KV 上限，后续块停止更新（音频 %.1fs > %.1fs）\n", alen / 13.0, (qwen3_.MAX_INPUT_LEN - 15) / 13.0); }
        return "";
    }
    qwen3_.clear_kv();
    int cur = qwen3_.forward_first(ids, tlen, audio, audio_start);
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

// 纯文本生成（调试用：绕过 encoder/audio，直接验证 LLM 链路）
std::string AsrPipeline::text_generate(const std::vector<int>& input_ids, int max_new_tokens) {
    if (!loaded_) return "";
    qwen3_.clear_kv();
    int cur = qwen3_.forward_first(input_ids, (int)input_ids.size(), {}, -1);
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

    // ── 1. mel：wav → [T,128]（帧-major，T 为真实帧数）──
    std::vector<float> mel;
    int t_real = 0;
    if (!mel_.wav_to_mel(wav_path, mel, t_real)) return "";

    // ── 2. 全量 encoder：pad 到 3000 → [390,1024] → 截取真实帧对应 tokens ──
    t_mel = std::chrono::steady_clock::now();
    std::vector<float> audio_embeds;
    int alen = 0;
    if (!encode_full_mel(mel, t_real, audio_embeds, alen)) return "";
    t_enc = std::chrono::steady_clock::now();

    // ── 3. 拼 embeds → prefill → decode ──
    std::vector<int> ids;
    int tlen = 0;
    int audio_start = build_input_ids(alen, ids, tlen);
    if (audio_start < 0) {
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "[ASR] 流式超 KV 上限，后续块停止更新（音频 %.1fs > %.1fs）\n", alen / 13.0, (qwen3_.MAX_INPUT_LEN - 15) / 13.0); }
        return "";
    }
    qwen3_.clear_kv();
    int cur = qwen3_.forward_first(ids, tlen, audio_embeds, audio_start);
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
