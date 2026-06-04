#include "eureka_audio.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>

namespace eureka {

// ── 工具 ──────────────────────────────────────────────────────────────────────

static inline void bm_check(bm_status_t s, const char* msg) {
    if (s != BM_SUCCESS) {
        fprintf(stderr, "[eureka] BM error %d: %s\n", (int)s, msg);
        throw std::runtime_error(msg);
    }
}

// 打印 device memory 使用量（MB），用于诊断 npu heap 是否溢出
static void log_devmem(bm_handle_t h, const char* tag) {
    bm_dev_stat_t stat;
    if (bm_get_stat(h, &stat) == BM_SUCCESS) {
        printf("[devmem] %-20s used=%d/%d MB\n", tag, stat.mem_used, stat.mem_total);
    }
}

// ── float32 <-> bfloat16 ─────────────────────────────────────────────────────

float EurekaAudioPipeline::bf16_to_fp32(uint16_t bf16) {
    uint32_t tmp = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &tmp, 4);
    return f;
}

std::vector<uint16_t> EurekaAudioPipeline::fp32_to_bf16_vec(const float* data, int n) {
    std::vector<uint16_t> out(n);
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &data[i], 4);
        // 四舍五入到 BF16
        u += 0x7FFF + ((u >> 16) & 1);
        out[i] = (uint16_t)(u >> 16);
    }
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// WhisperEncoderBmodel
// ══════════════════════════════════════════════════════════════════════════════

WhisperEncoderBmodel::~WhisperEncoderBmodel() { release(); }

void WhisperEncoderBmodel::release() {
    if (p_bmrt_) { bmrt_destroy(p_bmrt_); p_bmrt_ = nullptr; }
    inited_ = false;
}

bool WhisperEncoderBmodel::init(bm_handle_t handle, const std::string& path) {
    handle_ = handle;
    p_bmrt_ = bmrt_create(handle_);
    if (!p_bmrt_) { fprintf(stderr, "[WhisperEnc] bmrt_create failed\n"); return false; }

    if (!bmrt_load_bmodel(p_bmrt_, path.c_str())) {
        fprintf(stderr, "[WhisperEnc] load bmodel failed: %s\n", path.c_str());
        bmrt_destroy(p_bmrt_); p_bmrt_ = nullptr;
        return false;
    }
    net_enc_ = bmrt_get_network_info(p_bmrt_, "whisper_encoder");
    if (!net_enc_) {
        fprintf(stderr, "[WhisperEnc] network 'whisper_encoder' not found\n");
        return false;
    }
    net_adaptor_ = bmrt_get_network_info(p_bmrt_, "audio_adaptor");
    if (!net_adaptor_) {
        fprintf(stderr, "[WhisperEnc] network 'audio_adaptor' not found\n");
        return false;
    }
    inited_ = true;
    printf("[WhisperEnc] loaded %s\n", path.c_str());
    return true;
}

bool WhisperEncoderBmodel::run(const float* mel, int n_chunks, std::vector<float>& out) {
    if (!inited_) return false;

    // whisper_encoder: [1,128,3000] → [1,1500,1280]
    // audio_adaptor:   [1500,1280]  → [375,2048]
    auto& enc_in   = net_enc_->stages[0].input_mems[0];
    auto& enc_out  = net_enc_->stages[0].output_mems[0];
    auto& adp_in   = net_adaptor_->stages[0].input_mems[0];
    auto& adp_out  = net_adaptor_->stages[0].output_mems[0];

    int mel_elems           = 128 * 3000;
    int tokens_per_chunk    = 375;
    int embed_dim           = 2048;
    int out_elems_per_chunk = tokens_per_chunk * embed_dim;

    out.resize((size_t)n_chunks * out_elems_per_chunk);

    for (int c = 0; c < n_chunks; c++) {
        // Step 1: whisper_encoder
        bm_memcpy_s2d(handle_, enc_in, (void*)(mel + c * mel_elems));
        bm_tensor_t in_t, out_t;
        bmrt_tensor_with_device(&in_t,  enc_in,  net_enc_->input_dtypes[0],
                                net_enc_->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&out_t, enc_out, net_enc_->output_dtypes[0],
                                net_enc_->stages[0].output_shapes[0]);
        if (!bmrt_launch_tensor_ex(p_bmrt_, net_enc_->name,
                                   &in_t, 1, &out_t, 1, true, false)) {
            fprintf(stderr, "[WhisperEnc] encoder launch failed chunk %d\n", c);
            return false;
        }
        bm_thread_sync(handle_);

        // Step 2: audio_adaptor — input shares device mem with encoder output
        bm_memcpy_d2d_byte(handle_, adp_in, 0, enc_out, 0,
                           bm_mem_get_device_size(enc_out));
        bm_tensor_t adp_in_t, adp_out_t;
        bmrt_tensor_with_device(&adp_in_t,  adp_in,  net_adaptor_->input_dtypes[0],
                                net_adaptor_->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&adp_out_t, adp_out, net_adaptor_->output_dtypes[0],
                                net_adaptor_->stages[0].output_shapes[0]);
        if (!bmrt_launch_tensor_ex(p_bmrt_, net_adaptor_->name,
                                   &adp_in_t, 1, &adp_out_t, 1, true, false)) {
            fprintf(stderr, "[WhisperEnc] adaptor launch failed chunk %d\n", c);
            return false;
        }
        bm_thread_sync(handle_);

        // Step 3: download audio_adaptor output → float32
        size_t adp_out_bytes = bm_mem_get_device_size(adp_out);
        if (net_adaptor_->output_dtypes[0] == BM_FLOAT32) {
            bm_memcpy_d2s(handle_, out.data() + c * out_elems_per_chunk, adp_out);
        } else {
            // BF16 → float32
            std::vector<uint16_t> tmp(out_elems_per_chunk);
            bm_memcpy_d2s(handle_, tmp.data(), adp_out);
            for (int i = 0; i < out_elems_per_chunk; i++) {
                uint32_t u = (uint32_t)tmp[i] << 16;
                float f; memcpy(&f, &u, 4);
                out[c * out_elems_per_chunk + i] = f;
            }
        }
    }
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Qwen3EmbedsBmodel
// ══════════════════════════════════════════════════════════════════════════════

Qwen3EmbedsBmodel::~Qwen3EmbedsBmodel() { deinit(); }

void Qwen3EmbedsBmodel::deinit() {
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
    if (p_bmrt_) { bmrt_destroy(p_bmrt_); p_bmrt_ = nullptr; }
    inited_ = false;
}

void Qwen3EmbedsBmodel::init_by_names() {
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
    int num_blocks = 0;
    for (int i = 0; ; i++) {
        auto bn = "block_" + std::to_string(i);
        auto cn = "block_cache_" + std::to_string(i);
        if (!find(bn.c_str()) || !find(cn.c_str())) break;
        net_blocks_.push_back(bmrt_get_network_info(p_bmrt_, bn.c_str()));
        net_blocks_cache_.push_back(bmrt_get_network_info(p_bmrt_, cn.c_str()));
        num_blocks++;
    }
    NUM_LAYERS = num_blocks;
    free(names);

    mask_value_f32_ = -1e9f;
    MAX_INPUT_LEN = net_blocks_[0]->stages[0].input_shapes[0].dims[1];   // 512
    hidden_size_  = net_blocks_[0]->stages[0].input_shapes[0].dims[2];   // 2048
    // block_cache past_k shape [1, SEQ, N_KV_HEADS, HEAD_DIM]（新 layout）→ SEQ=dims[1]
    SEQLEN        = net_blocks_cache_[0]->stages[0].input_shapes[3].dims[1];   // 512
    int n_kv      = net_blocks_cache_[0]->stages[0].input_shapes[3].dims[2];   // 8
    int head_dim  = net_blocks_cache_[0]->stages[0].input_shapes[3].dims[3];   // 128
    kv_per_token_ = n_kv * head_dim;   // 1024
    printf("[Qwen3] Layers=%d MAX_INPUT=%d SEQLEN=%d kv_per_token=%d hidden=%d\n",
           NUM_LAYERS, MAX_INPUT_LEN, SEQLEN, kv_per_token_, hidden_size_);
}

bool Qwen3EmbedsBmodel::init(bm_handle_t handle, const std::string& path) {
    handle_ = handle;
    p_bmrt_ = bmrt_create(handle_);
    if (!p_bmrt_) { fprintf(stderr, "[Qwen3] bmrt_create failed\n"); return false; }
    if (!bmrt_load_bmodel(p_bmrt_, path.c_str())) {
        fprintf(stderr, "[Qwen3] load bmodel failed: %s\n", path.c_str());
        return false;
    }
    printf("[Qwen3] loaded %s\n", path.c_str());
    init_by_names();

    visited_tokens_.resize(SEQLEN, 0);
    // KV cache 常驻 device（独立分配）：每层 [1,N_KV,SEQLEN,HEAD_DIM] f32
    kv_layer_bytes_ = (size_t)SEQLEN * kv_per_token_ * sizeof(float);  // 2MB/层
    kv_token_bytes_ = (size_t)kv_per_token_ * sizeof(float);          // 一层一个token的KV(跨head需分散)
    past_key_dev_.resize(NUM_LAYERS);
    past_value_dev_.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        bm_malloc_device_byte(handle_, &past_key_dev_[i], kv_layer_bytes_);
        bm_malloc_device_byte(handle_, &past_value_dev_[i], kv_layer_bytes_);
        int z = 0;
        bm_memset_device_ext(handle_, &z, 1, past_key_dev_[i]);
        bm_memset_device_ext(handle_, &z, 1, past_value_dev_[i]);
    }

    // decode 复用 IO device buffer（block_cache 各层 shape 相同，共用一套，避免每步 malloc）
    auto bc0 = net_blocks_cache_[0];
    bm_malloc_device_byte(handle_, &dec_hidden_,  bm_mem_get_device_size(bc0->stages[0].input_mems[0]));
    bm_malloc_device_byte(handle_, &dec_pos_,     bm_mem_get_device_size(bc0->stages[0].input_mems[1]));
    bm_malloc_device_byte(handle_, &dec_mask_,    bm_mem_get_device_size(bc0->stages[0].input_mems[2]));
    bm_malloc_device_byte(handle_, &dec_newk_,    bm_mem_get_device_size(bc0->stages[0].output_mems[1]));
    bm_malloc_device_byte(handle_, &dec_newv_,    bm_mem_get_device_size(bc0->stages[0].output_mems[2]));
    bm_malloc_device_byte(handle_, &dec_emb_out_, bm_mem_get_device_size(net_embed_cache_->stages[0].output_mems[0]));
    dec_io_ready_ = true;

    inited_ = true;
    return true;
}

// 通用 launch：whisper 范式，bmrt_tensor 独立 IO + host↔device 完整往返
// inputs/outputs 为 host 数据指针（按 net 的 input/output 顺序），dtypes 用 net 自带
// 返回 false 表示失败。outputs 里 nullptr 表示该输出不下载。
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

// 取 logits（[1,1,VOCAB] f32）→ host argmax → token
static int argmax_logits(const std::vector<float>& logits) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < (int)logits.size(); i++)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

// hidden(host [1,1,H]) → lm_head → greedy_head（device 上 argmax）→ token（只下载 1 int）
// 省去每步下载 600KB logits + host argmax
static int lm_greedy(void* rt, bm_handle_t h, const bm_net_info_t* lm,
                     const bm_net_info_t* greedy, const float* hidden_host) {
    // lm_head: in hidden[1,1,H] → out logits[1,1,VOCAB]
    bm_tensor_t lm_in, lm_out;
    bmrt_tensor(&lm_in,  rt, lm->input_dtypes[0],  lm->stages[0].input_shapes[0]);
    bmrt_tensor(&lm_out, rt, lm->output_dtypes[0], lm->stages[0].output_shapes[0]);
    bm_memcpy_s2d(h, lm_in.device_mem, (void*)hidden_host);
    bmrt_launch_tensor_ex(rt, lm->name, &lm_in, 1, &lm_out, 1, true, false);
    bm_thread_sync(h);
    // greedy_head: in logits[1,VOCAB] → out token[1,1]，logits 留 device 直接 d2d 进 greedy 输入
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

int Qwen3EmbedsBmodel::forward_first(const std::vector<float>& embeds_f32, int tlen) {
    token_length = tlen;
    const int S = MAX_INPUT_LEN, H = hidden_size_;

    std::vector<int>   position_id(S, 0);
    std::vector<float> attention_mask((size_t)S * S, mask_value_f32_);
    for (int i = 0; i < tlen; i++) position_id[i] = i;
    for (int i = 0; i < tlen; i++)
        for (int j = 0; j <= i; j++)
            attention_mask[(size_t)i * S + j] = 0.0f;

    // hidden 在 host 上逐层传递（[S, H] f32）
    std::vector<float> hidden = embeds_f32;          // [S*H]
    std::vector<float> hidden_out((size_t)S * H);
    std::vector<float> pk((size_t)S * kv_per_token_); // block 输出 past_k [1,N_KV,S,HEAD] = [S,kv_per_token]
    std::vector<float> pv((size_t)S * kv_per_token_);

    for (int i = 0; i < NUM_LAYERS; i++) {
        // block_i: in[0]=hidden[1,S,H] in[1]=pos[1,S] in[2]=mask[1,1,S,S]
        //          out[0]=hidden[1,S,H] out[1]=past_k[1,N_KV,S,HEAD] out[2]=past_v
        bool ok = launch_host(p_bmrt_, handle_, net_blocks_[i],
            { hidden.data(), position_id.data(), attention_mask.data() },
            { hidden_out.data(), pk.data(), pv.data() });
        if (!ok) { fprintf(stderr, "[Qwen3] block_%d launch failed\n", i); return -1; }
        hidden.swap(hidden_out);
        // prefill KV 整块上传到 device 常驻 buffer（layout [1,8,512,128]，pad 部分为0）
        // 一次性上传，decode 阶段 KV 留在 device 不再往返
        bm_memcpy_s2d(handle_, past_key_dev_[i],   pk.data());
        bm_memcpy_s2d(handle_, past_value_dev_[i], pv.data());
    }

    // lm_head：输入最后一个有效 token 的 hidden [1,1,H]
    std::vector<float> last_hidden(H);
    std::copy(hidden.begin() + (size_t)(tlen - 1) * H,
              hidden.begin() + (size_t)tlen * H, last_hidden.begin());
    int token;
    if (net_greedy_head_) {
        token = lm_greedy(p_bmrt_, handle_, net_lm_, net_greedy_head_, last_hidden.data());
    } else {
        int vocab = net_lm_->stages[0].output_shapes[0].dims[2];
        std::vector<float> logits(vocab);
        launch_host(p_bmrt_, handle_, net_lm_, { last_hidden.data() }, { logits.data() });
        token = argmax_logits(logits);
    }
    // 对齐 Python：prefill 后 token_length 保持 = tlen（不 ++），当前 token 用成员变量传递
    cur_token_ = token;
    return token;
}

int Qwen3EmbedsBmodel::forward_next() {
    if (token_length >= SEQLEN) {
        fprintf(stderr, "[Qwen3] token_length=%d reached SEQLEN=%d\n", token_length, SEQLEN);
        return -1;
    }
    // Python 语义：cur = 上一步生成的 token（成员变量传递）
    int cur_token = cur_token_;
    const int H = hidden_size_;
    int pos = token_length - 1;

    // embedding_cache: in input_ids[1,1] → out hidden[1,1,H]，输出留 device（dec_emb_out_）
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
    // hidden 流转用 dec_hidden_，初值 = embedding 输出
    bm_memcpy_d2d_byte(handle_, dec_hidden_, 0, dec_emb_out_, 0, bm_mem_get_device_size(dec_hidden_));

    // pos / mask 上传一次（各层共用复用 buffer）
    int32_t position_id = pos;
    bm_memcpy_s2d(handle_, dec_pos_, &position_id);
    std::vector<float> attention_mask(SEQLEN + 1, 0.0f);
    for (int i = pos; i < SEQLEN; i++) attention_mask[i] = mask_value_f32_;
    bm_memcpy_s2d(handle_, dec_mask_, attention_mask.data());

    for (int i = 0; i < NUM_LAYERS; i++) {
        auto net = net_blocks_cache_[i];
        bm_tensor_t ins[5], outs[3];
        // in0 hidden(复用 dec_hidden_) in1 pos in2 mask in3/4 past_k/v(device常驻)
        bmrt_tensor_with_device(&ins[0], dec_hidden_, net->input_dtypes[0], net->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&ins[1], dec_pos_,    net->input_dtypes[1], net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&ins[2], dec_mask_,   net->input_dtypes[2], net->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(&ins[3], past_key_dev_[i],   net->input_dtypes[3], net->stages[0].input_shapes[3]);
        bmrt_tensor_with_device(&ins[4], past_value_dev_[i], net->input_dtypes[4], net->stages[0].input_shapes[4]);
        // out0 hidden 写回 dec_hidden_（下一层输入），out1/2 新 KV 写 dec_newk_/dec_newv_
        bmrt_tensor_with_device(&outs[0], dec_hidden_, net->output_dtypes[0], net->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(&outs[1], dec_newk_,   net->output_dtypes[1], net->stages[0].output_shapes[1]);
        bmrt_tensor_with_device(&outs[2], dec_newv_,   net->output_dtypes[2], net->stages[0].output_shapes[2]);
        // 去掉逐层 bm_thread_sync：同一 handle 上 launch/d2d 按提交顺序串行执行，
        // 依赖自动保证。只在循环末尾 d2s 时统一同步，省 28 次/步同步开销。
        bool ok = bmrt_launch_tensor_ex(p_bmrt_, net->name, ins, 5, outs, 3, true, false);
        if (!ok) { fprintf(stderr, "[Qwen3] block_cache_%d failed\n", i); return -1; }
        // KV layout [1,SEQ,N_KV,HEAD]：单 token KV 连续。新 KV → KV buffer 第 pos 个 token，一次 d2d
        size_t dst_off = (size_t)pos * kv_token_bytes_;
        bm_memcpy_d2d_byte(handle_, past_key_dev_[i],   dst_off, dec_newk_, 0, kv_token_bytes_);
        bm_memcpy_d2d_byte(handle_, past_value_dev_[i], dst_off, dec_newv_, 0, kv_token_bytes_);
    }

    // lm_head + greedy 用 dec_hidden_（device）作输入
    bm_thread_sync(handle_);   // 统一同步所有 block_cache launch + KV d2d
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
    token_length++;     // 先推进：本步处理了 position token_length-1，写了对应 KV
    cur_token_ = token;
    return token;
}

void Qwen3EmbedsBmodel::clear_kv() {
    int z = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        bm_memset_device_ext(handle_, &z, 1, past_key_dev_[i]);
        bm_memset_device_ext(handle_, &z, 1, past_value_dev_[i]);
    }
    token_length = 0;
}

// ══════════════════════════════════════════════════════════════════════════════
// EurekaAudioPipeline
// ══════════════════════════════════════════════════════════════════════════════

bool EurekaAudioPipeline::init(const std::string& whisper_path,
                                const std::string& qwen3_path,
                                int device_id) {
    // device mem 实测 9GB（bm_get_stat mem_total=9070MB），两个 bmodel
    // (whisper 1.4G + qwen3 2.7G = 4.1G) 同时常驻完全够用，直接一起加载。
    whisper_path_ = whisper_path;
    qwen3_path_   = qwen3_path;
    if (bm_dev_request(&bm_handle_, device_id) != BM_SUCCESS) {
        fprintf(stderr, "[Eureka] bm_dev_request failed\n");
        return false;
    }
    if (!whisper_enc_.init(bm_handle_, whisper_path)) return false;
    if (!qwen3_.init(bm_handle_, qwen3_path)) return false;
    log_devmem(bm_handle_, "after load both");
    printf("[Eureka] pipeline ready device=%d\n", device_id);
    return true;
}

void EurekaAudioPipeline::deinit() {
    qwen3_.deinit();
    whisper_enc_.release();
    if (bm_handle_) { bm_dev_free(bm_handle_); bm_handle_ = nullptr; }
}

std::vector<float> EurekaAudioPipeline::build_inputs_embeds(
    const float* prefix, int plen,
    const float* audio,  int alen,
    const float* suffix, int slen)
{
    int max_input = qwen3_.MAX_INPUT_LEN;
    int D = config.hidden_size;
    // 防越界：prefix/suffix 必须保留，超限时截断 audio（release 下 assert 失效会静默越界写）
    if (plen + alen + slen > max_input) {
        int new_alen = max_input - plen - slen;
        if (new_alen < 0) new_alen = 0;
        fprintf(stderr, "[Eureka] inputs %d > MAX_INPUT %d，audio token %d→%d 截断\n",
                plen + alen + slen, max_input, alen, new_alen);
        alen = new_alen;
    }

    // 分配 [MAX_INPUT_LEN, 2048] F32，初始化为 0
    std::vector<float> buf((size_t)max_input * D, 0.0f);

    int off = 0;
    memcpy(buf.data() + (size_t)off * D, prefix, (size_t)plen * D * sizeof(float));
    off += plen;
    memcpy(buf.data() + (size_t)off * D, audio,  (size_t)alen * D * sizeof(float));
    off += alen;
    if (suffix && slen > 0)
        memcpy(buf.data() + (size_t)off * D, suffix, (size_t)slen * D * sizeof(float));
    return buf;
}

std::vector<int> EurekaAudioPipeline::generate(
    const float* mel,    int n_chunks,
    const float* prefix, int plen,
    const float* suffix, int slen,
    int max_new_tokens,  int eos_token_id,
    int real_audio_tokens)
{
    using clk = std::chrono::steady_clock;
    auto sec = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    // 1. whisper encoder：mel → audio_embeds（bmodel 已在 init 加载）
    auto t_w0 = clk::now();
    std::vector<float> audio_embeds;
    if (!whisper_enc_.run(mel, n_chunks, audio_embeds)) {
        fprintf(stderr, "[Eureka] whisper encoder failed\n");
        return {};
    }
    auto t_w1 = clk::now();
    last_whisper_s = sec(t_w0, t_w1);
    int alen = n_chunks * config.tokens_per_chunk;  // 375 * n_chunks
    // 只保留实际音频对应的 token，丢弃 pad 静音（否则静音淹没语义，输出乱码）
    if (real_audio_tokens > 0 && real_audio_tokens < alen) {
        alen = real_audio_tokens;
        audio_embeds.resize((size_t)alen * config.hidden_size);
    }

    // 2. 拼接 inputs_embeds [prefix | audio | suffix]
    int total_len = plen + alen + slen;
    last_prefill_len = total_len;
    auto embeds_f32 = build_inputs_embeds(prefix, plen,
                                          audio_embeds.data(), alen,
                                          suffix, slen);

    // 阶段 3：prefill
    auto t_p0 = clk::now();
    qwen3_.clear_kv();
    int token = qwen3_.forward_first(embeds_f32, total_len);
    auto t_p1 = clk::now();
    last_prefill_s = sec(t_p0, t_p1);

    // 4. decode loop（KV cache 上限 = SEQLEN，到顶提前停止）
    auto t_d0 = clk::now();
    std::vector<int> result;
    while ((int)result.size() < max_new_tokens) {
        if (token == eos_token_id) break;
        result.push_back(token);
        if (qwen3_.token_length >= qwen3_.SEQLEN) break;
        token = qwen3_.forward_next();
    }
    auto t_d1 = clk::now();
    last_decode_s = sec(t_d0, t_d1);
    last_decode_tokens = (int)result.size();
    return result;
}

}  // namespace eureka
