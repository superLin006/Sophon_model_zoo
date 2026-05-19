// GPT engine for ChatTTS on BM1684X — pure bmruntime, no sail dependency.
// KV cache managed manually on host, uploaded each decode step.
//
// Graph shapes (static, from bmodel inspection):
//   embedding_text      in:[1,1024]int32          out:[1,1024,768]f16
//   embedding_code_cache in:[1,1,4]int32           out:[1,1,768]f16
//   block_i             in:[1,1024,768]f16,[1,1024]int32,[1,1,1024,1024]f16
//                       out:[1,1024,768]f16,[1,1024,12,64]f16,[1,1024,12,64]f16
//   block_cache_i       in:[1,1,768]f16,[1,1]int32,[1,1,1,1025]f16,
//                          [1,1024,12,64]f16,[1,1024,12,64]f16
//                       out:[1,1,768]f16,[1,1,12,64]f16,[1,1,12,64]f16
//   lm_head_code        in:[1,768]f16              out:[1,626,4]f32

#include "gpt_engine.h"
#include <bmlib_runtime.h>
#include <bmruntime_interface.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <string>
#include <vector>

// ── bmruntime helpers ────────────────────────────────────────────────────────

static void bm_launch(bm_handle_t hdl, void* rt, const bm_net_info_t* net,
                      std::vector<bm_tensor_t>& ins,
                      std::vector<bm_tensor_t>& outs) {
    bool ok = bmrt_launch_tensor_ex(rt, net->name,
                                    ins.data(),  ins.size(),
                                    outs.data(), outs.size(),
                                    true, false);
    if (!ok) throw std::runtime_error(std::string("bmrt_launch failed: ") + net->name);
    bm_thread_sync(hdl);
}

// Alloc device mem + fill from host, build bm_tensor_t
static bm_tensor_t make_in(void* rt, bm_handle_t hdl,
                            const bm_net_info_t* net, int idx,
                            const void* host_data) {
    bm_tensor_t t;
    bmrt_tensor(&t, rt, net->input_dtypes[idx], net->stages[0].input_shapes[idx]);
    bm_memcpy_s2d(hdl, t.device_mem, const_cast<void*>(host_data));
    return t;
}

// Alloc device mem for output
static bm_tensor_t make_out(void* rt, bm_handle_t hdl,
                             const bm_net_info_t* net, int idx) {
    bm_tensor_t t;
    bmrt_tensor(&t, rt, net->output_dtypes[idx], net->stages[0].output_shapes[idx]);
    return t;
}

static void free_tensors(bm_handle_t hdl, std::vector<bm_tensor_t>& ts) {
    for (auto& t : ts) bm_free_device(hdl, t.device_mem);
    ts.clear();
}

// ── sampling helpers ─────────────────────────────────────────────────────────

static float f16_to_f32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    float val;
    if (exponent == 0)       val = std::ldexp((float)mantissa, -24);
    else if (exponent == 31) val = mantissa ? std::numeric_limits<float>::quiet_NaN()
                                            : std::numeric_limits<float>::infinity();
    else                     val = std::ldexp((float)(mantissa | 0x400), (int)exponent - 25);
    return sign ? -val : val;
}

static void apply_repetition_penalty(std::vector<float>& logits,
                                     const std::vector<int>& prev,
                                     float penalty) {
    if (penalty == 1.0f || prev.empty()) return;
    for (int tok : prev) {
        if (tok < 0 || tok >= (int)logits.size()) continue;
        logits[tok] = logits[tok] > 0 ? logits[tok] / penalty
                                      : logits[tok] * penalty;
    }
}

static int sample_top_k_top_p(const std::vector<float>& logits,
                               float temperature, float top_p, int top_k,
                               std::mt19937& rng) {
    int n = (int)logits.size();
    std::vector<std::pair<float,int>> scored(n);
    for (int i = 0; i < n; ++i) scored[i] = {logits[i] / temperature, i};
    if (top_k > 0 && top_k < n) {
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        scored.resize(top_k);
    } else {
        std::sort(scored.begin(), scored.end(),
                  [](auto& a, auto& b){ return a.first > b.first; });
    }
    float mx = scored[0].first, sum = 0.f;
    for (auto& p : scored) { p.first = std::exp(p.first - mx); sum += p.first; }
    for (auto& p : scored) p.first /= sum;
    float cumsum = 0.f;
    int keep = (int)scored.size();
    for (int i = 0; i < (int)scored.size(); ++i) {
        cumsum += scored[i].first;
        if (cumsum >= top_p) { keep = i + 1; break; }
    }
    scored.resize(keep);
    sum = 0.f;
    for (auto& p : scored) sum += p.first;
    std::vector<float> probs(keep);
    for (int i = 0; i < keep; ++i) probs[i] = scored[i].first / sum;
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return scored[dist(rng)].second;
}

// ── GPTEngine::Impl ──────────────────────────────────────────────────────────

struct GPTEngine::Impl {
    GPTConfig cfg;

    bm_handle_t hdl      = nullptr;
    void*       rt       = nullptr;
    bool        owns_hdl = true;

    // Network info pointers — valid for lifetime of rt
    const bm_net_info_t* net_embed_text  = nullptr;
    const bm_net_info_t* net_embed_code  = nullptr;
    const bm_net_info_t* net_lm_code     = nullptr;
    std::vector<const bm_net_info_t*> net_blocks;        // block_0 .. block_19
    std::vector<const bm_net_info_t*> net_blocks_cache;  // block_cache_0 .. block_cache_19

    // Dimensions read from bmodel
    int SEQLEN      = 1024;
    int HIDDEN_SIZE = 768;
    int ATTEN_HEAD  = 12;
    int ATTEN_DIM   = 64;
    int NUM_LAYERS  = 20;
    int NUM_VQ      = 4;

    // KV cache: device memory [NUM_LAYERS], each [1,SEQLEN,ATTEN_HEAD,ATTEN_DIM] f16
    // Stays on device the entire session — no host↔device round-trips during decode.
    std::vector<bm_device_mem_t> dev_k;  // [layer]
    std::vector<bm_device_mem_t> dev_v;

    // Pre-allocated decode buffers — reused every step to avoid per-step malloc/free
    bm_device_mem_t dec_pid_dm  = {};  // [1,1] int32 = 4 bytes
    bm_device_mem_t dec_mask_dm = {};  // [1,1,1,SEQLEN+1] f16
    bm_device_mem_t dec_hid_dm  = {};  // [1,1,HIDDEN_SIZE] f16 (hidden state through blocks)
    bm_device_mem_t dec_nk_dm   = {};  // [1,1,ATTEN_HEAD,ATTEN_DIM] f16 (new_k scratch)
    bm_device_mem_t dec_nv_dm   = {};  // [1,1,ATTEN_HEAD,ATTEN_DIM] f16 (new_v scratch)
    bool dec_bufs_ready = false;

    // Pre-allocated prefill buffers — reused every prefill() call
    bm_device_mem_t pf_em_in_dm  = {};  // [1, SEQLEN] int32 (embedding_text input)
    bm_device_mem_t pf_em_out_dm = {};  // [1, SEQLEN, HIDDEN_SIZE] f16 (ping)
    bm_device_mem_t pf_hid_dm    = {};  // [1, SEQLEN, HIDDEN_SIZE] f16 (pong)
    bm_device_mem_t pf_pid_dm    = {};  // [1, SEQLEN] int32 (block position ids)
    bm_device_mem_t pf_mask_dm   = {};  // [1, 1, SEQLEN, SEQLEN] f16 (block attn mask)
    bm_device_mem_t pf_lm_in_dm  = {};  // [1, HIDDEN_SIZE] f16 (lm_head input)
    bm_device_mem_t pf_lm_out_dm = {};  // [1, NUM_AUDIO_TOKENS, NUM_VQ] f32
    // dec path: embed_code output + lm_head buffers (tiny but still fragment)
    bm_device_mem_t dec_ec_out_dm = {};  // [1, 1, HIDDEN_SIZE] f16
    bm_device_mem_t dec_lm_in_dm  = {};  // [1, HIDDEN_SIZE] f16
    bm_device_mem_t dec_lm_out_dm = {};  // [1, NUM_AUDIO_TOKENS, NUM_VQ] f32
    bool pf_bufs_ready = false;

    int  decode_step  = 0;
    int  text_tok_len = 0;
    std::mt19937 rng{42};

    void init(const std::string& bmodel_path, int tpu_id, const GPTConfig& c);

    std::pair<std::vector<float>, std::vector<uint16_t>>
    prefill(const std::vector<int>& tokens, int spk_idx,
            const std::vector<uint16_t>& spk_emb_f16);

    std::pair<std::vector<float>, std::vector<uint16_t>>
    decode(const std::vector<int>& vq_codes);

    ~Impl() {
        for (auto& m : dev_k) bm_free_device(hdl, m);
        for (auto& m : dev_v) bm_free_device(hdl, m);
        if (dec_bufs_ready) {
            bm_free_device(hdl, dec_pid_dm);
            bm_free_device(hdl, dec_mask_dm);
            bm_free_device(hdl, dec_hid_dm);
            bm_free_device(hdl, dec_nk_dm);
            bm_free_device(hdl, dec_nv_dm);
        }
        if (pf_bufs_ready) {
            bm_free_device(hdl, pf_em_in_dm);
            bm_free_device(hdl, pf_em_out_dm);
            bm_free_device(hdl, pf_hid_dm);
            bm_free_device(hdl, pf_pid_dm);
            bm_free_device(hdl, pf_mask_dm);
            bm_free_device(hdl, pf_lm_in_dm);
            bm_free_device(hdl, pf_lm_out_dm);
            bm_free_device(hdl, dec_ec_out_dm);
            bm_free_device(hdl, dec_lm_in_dm);
            bm_free_device(hdl, dec_lm_out_dm);
        }
        // Always destroy bmrt (created by this engine in both owned and shared handle cases)
        if (rt) bmrt_destroy(rt);
        if (owns_hdl && hdl) bm_dev_free(hdl);
    }
};

// ── init ─────────────────────────────────────────────────────────────────────

void GPTEngine::Impl::init(const std::string& bmodel_path, int tpu_id,
                            const GPTConfig& c) {
    cfg        = c;
    NUM_LAYERS = c.num_layers;
    NUM_VQ     = c.num_vq;

    if (owns_hdl) {
        if (bm_dev_request(&hdl, tpu_id) != BM_SUCCESS)
            throw std::runtime_error("bm_dev_request failed");
        fprintf(stderr, "[GPT] bm_dev_request ok\n");
        rt = bmrt_create(hdl);
        if (!rt) throw std::runtime_error("bmrt_create failed");
    } else {
        fprintf(stderr, "[GPT] using shared handle\n");
        // rt already set by shared-handle constructor
    }

    if (!bmrt_load_bmodel(rt, bmodel_path.c_str()))
        throw std::runtime_error("bmrt_load_bmodel failed: " + bmodel_path);
    fprintf(stderr, "[GPT] bmodel loaded\n");

    // Fetch all network info pointers by name
    auto get_net = [&](const std::string& name) -> const bm_net_info_t* {
        auto* p = bmrt_get_network_info(rt, name.c_str());
        if (!p) throw std::runtime_error("network not found: " + name);
        return p;
    };

    net_embed_text = get_net("embedding_text");
    net_embed_code = get_net("embedding_code_cache");
    net_lm_code    = get_net("lm_head_code");

    net_blocks.resize(NUM_LAYERS);
    net_blocks_cache.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; ++i) {
        net_blocks[i]       = get_net("block_"       + std::to_string(i));
        net_blocks_cache[i] = get_net("block_cache_" + std::to_string(i));
    }
    fprintf(stderr, "[GPT] all networks found\n");

    // Read dimensions from bmodel shapes
    // block_0 in[0]: [1, SEQLEN, HIDDEN_SIZE]
    {
        auto& sh = net_blocks[0]->stages[0].input_shapes[0];
        SEQLEN      = sh.dims[1];
        HIDDEN_SIZE = sh.dims[2];
    }
    // block_cache_0 in[3]: [1, SEQLEN, ATTEN_HEAD, ATTEN_DIM]
    {
        auto& sh = net_blocks_cache[0]->stages[0].input_shapes[3];
        ATTEN_HEAD = sh.dims[2];
        ATTEN_DIM  = sh.dims[3];
    }
    fprintf(stderr, "[GPT] SEQLEN=%d HIDDEN=%d HEADS=%d DIM=%d\n",
            SEQLEN, HIDDEN_SIZE, ATTEN_HEAD, ATTEN_DIM);

    // Allocate KV cache on device — stays resident, no host↔device round-trips during decode
    // Shape matches block_cache input: [1, SEQLEN, ATTEN_HEAD, ATTEN_DIM] f16
    bm_shape_t kv_shape = {{4}, {1, SEQLEN, ATTEN_HEAD, ATTEN_DIM}};
    size_t kv_bytes = (size_t)SEQLEN * ATTEN_HEAD * ATTEN_DIM * sizeof(uint16_t);
    dev_k.resize(NUM_LAYERS);
    dev_v.resize(NUM_LAYERS);
    std::vector<uint16_t> zeros(SEQLEN * ATTEN_HEAD * ATTEN_DIM, 0);
    for (int i = 0; i < NUM_LAYERS; ++i) {
        bm_malloc_device_byte(hdl, &dev_k[i], kv_bytes);
        bm_malloc_device_byte(hdl, &dev_v[i], kv_bytes);
        bm_memcpy_s2d(hdl, dev_k[i], zeros.data());
        bm_memcpy_s2d(hdl, dev_v[i], zeros.data());
    }

    // Pre-allocate decode reusable buffers (pid, mask, hidden)
    // Sizes taken from block_cache_0 shapes
    {
        const bm_net_info_t* bc0 = net_blocks_cache[0];
        // pid: input[1] shape [1,1] int32
        bm_malloc_device_byte(hdl, &dec_pid_dm, sizeof(int32_t));
        // mask: input[2] shape [1,1,1,SEQLEN+1] f16
        bm_malloc_device_byte(hdl, &dec_mask_dm, (size_t)(SEQLEN + 1) * sizeof(uint16_t));
        // hidden ping-pong partner: output[0] shape [1,1,HIDDEN_SIZE] f16
        bm_malloc_device_byte(hdl, &dec_hid_dm, (size_t)HIDDEN_SIZE * sizeof(uint16_t));
        // new_k/v scratch: output[1/2] shape [1,1,ATTEN_HEAD,ATTEN_DIM] f16
        size_t nkv_bytes = (size_t)ATTEN_HEAD * ATTEN_DIM * sizeof(uint16_t);
        bm_malloc_device_byte(hdl, &dec_nk_dm, nkv_bytes);
        bm_malloc_device_byte(hdl, &dec_nv_dm, nkv_bytes);
        dec_bufs_ready = true;
        (void)bc0;
    }

    // Pre-allocate prefill persistent buffers
    {
        int NA = cfg.num_audio_tokens, NV = cfg.num_vq;
        bm_malloc_device_byte(hdl, &pf_em_in_dm,  (size_t)SEQLEN * sizeof(int32_t));
        bm_malloc_device_byte(hdl, &pf_em_out_dm, (size_t)SEQLEN * HIDDEN_SIZE * sizeof(uint16_t));
        bm_malloc_device_byte(hdl, &pf_hid_dm,    (size_t)SEQLEN * HIDDEN_SIZE * sizeof(uint16_t));
        bm_malloc_device_byte(hdl, &pf_pid_dm,    (size_t)SEQLEN * sizeof(int32_t));
        bm_malloc_device_byte(hdl, &pf_mask_dm,   (size_t)SEQLEN * SEQLEN * sizeof(uint16_t));
        bm_malloc_device_byte(hdl, &pf_lm_in_dm,  (size_t)HIDDEN_SIZE * sizeof(uint16_t));
        bm_malloc_device_byte(hdl, &pf_lm_out_dm, (size_t)NA * NV * sizeof(float));
        bm_malloc_device_byte(hdl, &dec_ec_out_dm, (size_t)HIDDEN_SIZE * sizeof(uint16_t));
        bm_malloc_device_byte(hdl, &dec_lm_in_dm,  (size_t)HIDDEN_SIZE * sizeof(uint16_t));
        bm_malloc_device_byte(hdl, &dec_lm_out_dm, (size_t)NA * NV * sizeof(float));
        pf_bufs_ready = true;
    }

    fprintf(stderr, "[GPT] init done\n");
}

// ── prefill ───────────────────────────────────────────────────────────────────

std::pair<std::vector<float>, std::vector<uint16_t>>
GPTEngine::Impl::prefill(const std::vector<int>& tokens, int spk_idx,
                          const std::vector<uint16_t>& spk_emb_f16) {
    int tok_len = (int)tokens.size();
    fprintf(stderr, "[GPT] prefill: %d tokens\n", tok_len);

    // ── embedding_text ────────────────────────────────────────────────────────
    // in[0]: [1, SEQLEN] int32 — pad with 0
    std::vector<int32_t> ids(SEQLEN, 0);
    for (int i = 0; i < tok_len; ++i) ids[i] = tokens[i];
    bm_memcpy_s2d(hdl, pf_em_in_dm, ids.data());

    std::vector<bm_tensor_t> em_in(1), em_out(1);
    bmrt_tensor_with_device(&em_in[0],  pf_em_in_dm,  net_embed_text->input_dtypes[0],
                             net_embed_text->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(&em_out[0], pf_em_out_dm, net_embed_text->output_dtypes[0],
                             net_embed_text->stages[0].output_shapes[0]);
    bm_launch(hdl, rt, net_embed_text, em_in, em_out);
    // no free — both are persistent
    fprintf(stderr, "[GPT] embedding_text done\n");

    // em_out[0] holds hidden: [1, SEQLEN, HIDDEN_SIZE] f16 on device
    // We keep this device buffer and pass it through the block chain.
    // After all blocks, we download the result.

    // ── inject speaker embedding (if provided) ────────────────────────────────
    // Approach: download hidden, overwrite position spk_idx, re-upload
    if (spk_idx >= 0 && !spk_emb_f16.empty()) {
        size_t hidden_elems = (size_t)SEQLEN * HIDDEN_SIZE;
        std::vector<uint16_t> hidden_host(hidden_elems);
        bm_memcpy_d2s(hdl, hidden_host.data(), pf_em_out_dm);
        size_t off = (size_t)spk_idx * HIDDEN_SIZE;
        std::memcpy(hidden_host.data() + off, spk_emb_f16.data(),
                    HIDDEN_SIZE * sizeof(uint16_t));
        bm_memcpy_s2d(hdl, pf_em_out_dm, hidden_host.data());
        fprintf(stderr, "[GPT] speaker embedding injected at pos %d\n", spk_idx);
    }

    // ── build position_id and attention_mask ──────────────────────────────────
    std::vector<int32_t> pid(SEQLEN, 0);
    for (int i = 0; i < tok_len; ++i) pid[i] = i;
    bm_memcpy_s2d(hdl, pf_pid_dm, pid.data());

    const uint16_t NEG_INF_F16 = 0xF9C0; // f16(-10000)
    size_t mask_elems = (size_t)SEQLEN * SEQLEN;
    std::vector<uint16_t> mask(mask_elems, NEG_INF_F16);
    for (int r = 0; r < tok_len; ++r)
        for (int c = 0; c <= r; ++c)
            mask[(size_t)r * SEQLEN + c] = 0;
    bm_memcpy_s2d(hdl, pf_mask_dm, mask.data());

    // ── 20 transformer blocks ─────────────────────────────────────────────────
    // Ping-pong between pf_em_out_dm and pf_hid_dm so we never alloc/free per block.
    // K/V outputs go directly into persistent dev_k[i]/dev_v[i].
    size_t pad_elems = (tok_len < SEQLEN) ? (size_t)(SEQLEN - tok_len) * HIDDEN_SIZE : 0;

    // ping = pf_em_out_dm (embedding output), pong = pf_hid_dm
    bm_device_mem_t pf_ping = pf_em_out_dm;
    bm_device_mem_t pf_pong = pf_hid_dm;

    for (int i = 0; i < NUM_LAYERS; ++i) {
        const bm_net_info_t* net = net_blocks[i];

        bm_device_mem_t cur_in  = (i % 2 == 0) ? pf_ping : pf_pong;
        bm_device_mem_t cur_out = (i % 2 == 0) ? pf_pong : pf_ping;

        std::vector<bm_tensor_t> blk_in(3), blk_out(3);
        bmrt_tensor_with_device(&blk_in[0], cur_in,    net->input_dtypes[0],  net->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&blk_in[1], pf_pid_dm, net->input_dtypes[1],  net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&blk_in[2], pf_mask_dm,net->input_dtypes[2],  net->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(&blk_out[0],cur_out,   net->output_dtypes[0], net->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(&blk_out[1], dev_k[i], net->output_dtypes[1], net->stages[0].output_shapes[1]);
        bmrt_tensor_with_device(&blk_out[2], dev_v[i], net->output_dtypes[2], net->stages[0].output_shapes[2]);

        bm_launch(hdl, rt, net, blk_in, blk_out);
        // all tensors point at persistent buffers — no alloc/free

        // Zero out padded rows in hidden output to prevent NaN propagation through layer norm.
        if (i < NUM_LAYERS - 1 && pad_elems > 0) {
            size_t total = (size_t)SEQLEN * HIDDEN_SIZE;
            std::vector<uint16_t> h(total);
            bm_memcpy_d2s(hdl, h.data(), cur_out);
            std::fill(h.data() + (size_t)tok_len * HIDDEN_SIZE, h.data() + total, uint16_t(0));
            bm_memcpy_s2d(hdl, cur_out, h.data());
        }
    }
    // Final hidden is in the buffer that received the last block output.
    // Last block i=NUM_LAYERS-1: cur_out = ((NUM_LAYERS-1)%2==0) ? pf_pong : pf_ping
    bm_device_mem_t pf_final_hid = ((NUM_LAYERS - 1) % 2 == 0) ? pf_pong : pf_ping;
    fprintf(stderr, "[GPT] prefill blocks done\n");

    // ── lm_head_code ─────────────────────────────────────────────────────────
    // Download final hidden [SEQLEN,HIDDEN_SIZE], extract last real token row
    size_t hidden_total = (size_t)SEQLEN * HIDDEN_SIZE;
    std::vector<uint16_t> hidden_host(hidden_total);
    bm_memcpy_d2s(hdl, hidden_host.data(), pf_final_hid);

    int last_off = (tok_len - 1) * HIDDEN_SIZE;
    std::vector<uint16_t> last_hidden(HIDDEN_SIZE);
    std::memcpy(last_hidden.data(), hidden_host.data() + last_off,
                HIDDEN_SIZE * sizeof(uint16_t));

    // lm_head_code: in [1,HIDDEN_SIZE] f16, out [1,NUM_AUDIO_TOKENS,NUM_VQ] f32
    bm_memcpy_s2d(hdl, pf_lm_in_dm, last_hidden.data());
    std::vector<bm_tensor_t> lm_in(1), lm_out(1);
    bmrt_tensor_with_device(&lm_in[0],  pf_lm_in_dm,  net_lm_code->input_dtypes[0],
                             net_lm_code->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(&lm_out[0], pf_lm_out_dm, net_lm_code->output_dtypes[0],
                             net_lm_code->stages[0].output_shapes[0]);
    bm_launch(hdl, rt, net_lm_code, lm_in, lm_out);
    // no free — persistent buffers

    int logit_n = cfg.num_audio_tokens * cfg.num_vq;
    std::vector<float> logits(logit_n);
    bm_memcpy_d2s(hdl, logits.data(), pf_lm_out_dm);

    decode_step  = 0;
    text_tok_len = tok_len;
    return {logits, last_hidden};
}

// ── decode ────────────────────────────────────────────────────────────────────

std::pair<std::vector<float>, std::vector<uint16_t>>
GPTEngine::Impl::decode(const std::vector<int>& vq_codes) {
    int step = decode_step;
    decode_step++;

    // ── embedding_code_cache ──────────────────────────────────────────────────
    // in[0]: [1,1,NUM_VQ] int32
    std::vector<int32_t> ids(NUM_VQ);
    for (int i = 0; i < NUM_VQ; ++i) ids[i] = vq_codes[i];

    // Reuse pf_em_in_dm (same dtype int32, big enough: SEQLEN ints vs NUM_VQ ints)
    bm_memcpy_s2d(hdl, pf_em_in_dm, ids.data());
    std::vector<bm_tensor_t> em_in(1), em_out(1);
    bmrt_tensor_with_device(&em_in[0],  pf_em_in_dm,  net_embed_code->input_dtypes[0],
                             net_embed_code->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(&em_out[0], dec_ec_out_dm, net_embed_code->output_dtypes[0],
                             net_embed_code->stages[0].output_shapes[0]);
    bm_launch(hdl, rt, net_embed_code, em_in, em_out);
    // no free — persistent buffers

    // ── position_id and attention_mask ────────────────────────────────────────
    // Position in the sequence = text_tok_len + step (decode tokens follow text tokens)
    int seq_pos = text_tok_len + step;
    int32_t pos_val = (int32_t)seq_pos;

    // Upload pid and mask once — reused by all 20 blocks this step
    bm_memcpy_s2d(hdl, dec_pid_dm, &pos_val);

    // attention_mask: [1,1,1,SEQLEN+1] f16
    // Positions 0..seq_pos-1 visible; seq_pos..SEQLEN-1 masked; slot SEQLEN unmasked (0).
    const uint16_t NEG_INF_F16 = 0xF0E2; // f16(-10000.0)
    {
        std::vector<uint16_t> mask(SEQLEN + 1, 0);
        for (int i = seq_pos; i < SEQLEN; ++i) mask[i] = NEG_INF_F16;
        bm_memcpy_s2d(hdl, dec_mask_dm, mask.data());
    }

    // ── 20 block_cache passes ─────────────────────────────────────────────────
    // in[0]: hidden [1,1,HIDDEN_SIZE] f16
    // in[1]: pid    [1,1] int32          ← shared dec_pid_dm, no alloc/free
    // in[2]: mask   [1,1,1,SEQLEN+1] f16 ← shared dec_mask_dm, no alloc/free
    // in[3]: past_k [1,SEQLEN,ATTEN_HEAD,ATTEN_DIM] f16  ← persistent dev_k[i]
    // in[4]: past_v same                                  ← persistent dev_v[i]
    // out[0]: hidden [1,1,HIDDEN_SIZE] f16  ← pre-allocated dec_hid_dm
    // out[1]: new_k  [1,1,ATTEN_HEAD,ATTEN_DIM] f16
    // out[2]: new_v  same
    //
    // Block 0 input hidden comes from embedding_code_cache output (em_out[0]).
    // Blocks 1..N: input hidden = dec_hid_dm (output of previous block).
    // Since input and output are different device buffers (em_out vs dec_hid_dm for block 0,
    // then dec_hid_dm→dec_hid_dm for blocks 1..N which is the same buffer),
    // we need a second persistent buffer for blocks 1..N to avoid in-place aliasing.
    // Use em_out[0].device_mem as the "A" buffer and dec_hid_dm as "B", ping-ponging.

    bm_device_mem_t ping = dec_ec_out_dm;  // embedding_code output — persistent
    bm_device_mem_t pong = dec_hid_dm;    // persistent pre-allocated

    int step_elems = ATTEN_HEAD * ATTEN_DIM;
    std::vector<uint16_t> new_k(step_elems), new_v(step_elems);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        const bm_net_info_t* net = net_blocks_cache[i];

        bm_device_mem_t cur_in  = (i % 2 == 0) ? ping : pong;
        bm_device_mem_t cur_out = (i % 2 == 0) ? pong : ping;

        bm_tensor_t t_in, t_pid, t_mask, t_pk, t_pv, t_out, t_nk, t_nv;
        bmrt_tensor_with_device(&t_in,  cur_in,      net->input_dtypes[0],  net->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(&t_pid, dec_pid_dm,  net->input_dtypes[1],  net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&t_mask,dec_mask_dm, net->input_dtypes[2],  net->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(&t_pk,  dev_k[i],    net->input_dtypes[3],  net->stages[0].input_shapes[3]);
        bmrt_tensor_with_device(&t_pv,  dev_v[i],    net->input_dtypes[4],  net->stages[0].input_shapes[4]);
        bmrt_tensor_with_device(&t_out, cur_out,    net->output_dtypes[0], net->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(&t_nk,  dec_nk_dm,  net->output_dtypes[1], net->stages[0].output_shapes[1]);
        bmrt_tensor_with_device(&t_nv,  dec_nv_dm,  net->output_dtypes[2], net->stages[0].output_shapes[2]);

        std::vector<bm_tensor_t> blk_in  = {t_in, t_pid, t_mask, t_pk, t_pv};
        std::vector<bm_tensor_t> blk_out = {t_out, t_nk, t_nv};
        bm_launch(hdl, rt, net, blk_in, blk_out);
        // all tensors point at persistent buffers — no free

        bm_memcpy_d2s(hdl, new_k.data(), dec_nk_dm);
        bm_memcpy_d2s(hdl, new_v.data(), dec_nv_dm);

        bm_device_mem_t dk_slice = dev_k[i];
        dk_slice.u.device.device_addr += (size_t)seq_pos * step_elems * sizeof(uint16_t);
        dk_slice.size = step_elems * sizeof(uint16_t);
        bm_memcpy_s2d(hdl, dk_slice, new_k.data());

        bm_device_mem_t dv_slice = dev_v[i];
        dv_slice.u.device.device_addr += (size_t)seq_pos * step_elems * sizeof(uint16_t);
        dv_slice.size = step_elems * sizeof(uint16_t);
        bm_memcpy_s2d(hdl, dv_slice, new_v.data());
    }

    // Final hidden is in whichever buffer received the last output.
    // Last iteration i=NUM_LAYERS-1: cur_out = ((NUM_LAYERS-1)%2==0) ? pong : ping
    // For NUM_LAYERS=20: i=19 is odd → cur_out = ping = em_out[0].device_mem
    bm_device_mem_t final_hid = ((NUM_LAYERS - 1) % 2 == 0) ? pong : ping;

    std::vector<uint16_t> hidden_host(HIDDEN_SIZE);
    bm_memcpy_d2s(hdl, hidden_host.data(), final_hid);
    // both ping (dec_ec_out_dm) and pong (dec_hid_dm) are persistent — no free

    // ── lm_head_code ─────────────────────────────────────────────────────────
    bm_memcpy_s2d(hdl, dec_lm_in_dm, hidden_host.data());
    std::vector<bm_tensor_t> lm_in(1), lm_out(1);
    bmrt_tensor_with_device(&lm_in[0],  dec_lm_in_dm,  net_lm_code->input_dtypes[0],
                             net_lm_code->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(&lm_out[0], dec_lm_out_dm, net_lm_code->output_dtypes[0],
                             net_lm_code->stages[0].output_shapes[0]);
    bm_launch(hdl, rt, net_lm_code, lm_in, lm_out);
    // no free — persistent buffers

    int logit_n = cfg.num_audio_tokens * cfg.num_vq;
    std::vector<float> logits(logit_n);
    bm_memcpy_d2s(hdl, logits.data(), dec_lm_out_dm);

    return {logits, hidden_host};
}

// ── Shared sampling state helper (used by both batch and step-by-step API) ───

struct SamplingState {
    int NT, NV, eos;
    float top_p;
    int   top_k;
    float repetition_penalty;
    int   min_new_token;
    std::vector<float> temperature;
    std::vector<std::vector<int>> generated;
    std::vector<int> curr_codes;
    std::mt19937* rng;

    bool sample(const std::vector<float>& raw, int step) {
        bool any_eos = false;
        for (int v = 0; v < NV; ++v) {
            std::vector<float> lv(NT);
            for (int k = 0; k < NT; ++k) lv[k] = raw[(size_t)k * NV + v];
            apply_repetition_penalty(lv, generated[v], repetition_penalty);
            if (step < min_new_token) lv[eos] = -1e9f;
            int tok = sample_top_k_top_p(lv, temperature[v], top_p, top_k, *rng);
            curr_codes[v] = tok;
            if (tok == eos) any_eos = true;
            else generated[v].push_back(tok);
        }
        return any_eos;
    }
};

// ── Public API ────────────────────────────────────────────────────────────────

GPTEngine::GPTEngine(const std::string& bmodel_path, int tpu_id, const GPTConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
    impl_->owns_hdl = true;
    impl_->init(bmodel_path, tpu_id, cfg);
}

GPTEngine::GPTEngine(const std::string& bmodel_path, void* bm_handle, void* /*ignored*/,
                     const GPTConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
    impl_->owns_hdl = false;
    impl_->hdl      = static_cast<bm_handle_t>(bm_handle);
    // Create own bmrt using the shared handle (shared allocator, separate network namespace)
    impl_->rt = bmrt_create(impl_->hdl);
    if (!impl_->rt) throw std::runtime_error("GPTEngine: bmrt_create failed");
    impl_->init(bmodel_path, -1, cfg);  // tpu_id ignored when owns_hdl=false
}

GPTEngine::~GPTEngine() = default;

GPTResult GPTEngine::generate(const std::vector<int>&      input_ids,
                               const std::vector<uint16_t>& spk_emb,
                               int                          spk_emb_idx,
                               const std::vector<float>&    temperature,
                               float                        top_p,
                               int                          top_k,
                               float                        repetition_penalty,
                               int                          max_new_token,
                               int                          min_new_token) {
    auto& im = *impl_;
    im.rng.seed(42);

    GPTResult result;
    SamplingState ss;
    ss.NT = im.cfg.num_audio_tokens; ss.NV = im.cfg.num_vq; ss.eos = ss.NT - 1;
    ss.top_p = top_p; ss.top_k = top_k; ss.repetition_penalty = repetition_penalty;
    ss.min_new_token = min_new_token; ss.temperature = temperature;
    ss.generated.resize(ss.NV); ss.curr_codes.resize(ss.NV); ss.rng = &im.rng;

    // Prefill
    auto [logits0, hidden0] = im.prefill(input_ids, spk_emb_idx, spk_emb);
    bool done = ss.sample(logits0, 0);
    result.hiddens.push_back(hidden0);
    result.codes.push_back(ss.curr_codes);
    fprintf(stderr, "[GPT] step=0 codes=[%d,%d,%d,%d] eos=%d\n",
            ss.curr_codes[0], ss.curr_codes[1], ss.curr_codes[2], ss.curr_codes[3], done?1:0);

    // Decode loop
    for (int step = 1; step < max_new_token && !done; ++step) {
        if (im.decode_step >= im.SEQLEN) break;
        auto [logits_n, hidden_n] = im.decode(ss.curr_codes);
        done = ss.sample(logits_n, step);
        result.hiddens.push_back(hidden_n);
        result.codes.push_back(ss.curr_codes);
        if (step <= 5 || step % 20 == 0)
            fprintf(stderr, "[GPT] step=%d codes=[%d,%d,%d,%d] eos=%d\n",
                    step, ss.curr_codes[0], ss.curr_codes[1], ss.curr_codes[2], ss.curr_codes[3], done?1:0);
    }
    fprintf(stderr, "[GPT] done: %d steps\n", (int)result.codes.size());
    return result;
}

// ── Step-by-step API ─────────────────────────────────────────────────────────

GPTStepResult GPTEngine::prefill_step(const std::vector<int>&      input_ids,
                                       const std::vector<uint16_t>& spk_emb,
                                       int                          spk_emb_idx) {
    impl_->rng.seed(42);
    auto [logits, hidden] = impl_->prefill(input_ids, spk_emb_idx, spk_emb);
    return {std::move(logits), std::move(hidden)};
}

GPTStepResult GPTEngine::decode_step(const std::vector<int>& vq_codes) {
    auto [logits, hidden] = impl_->decode(vq_codes);
    return {std::move(logits), std::move(hidden)};
}

int GPTEngine::current_decode_step() const {
    return impl_->decode_step;
}

int GPTEngine::seqlen() const {
    return impl_->SEQLEN;
}
