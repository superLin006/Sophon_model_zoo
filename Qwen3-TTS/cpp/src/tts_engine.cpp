#include "tts_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

// bf16/f16 <-> f32（bmrt 网络输入输出可能是量化 dtype）
// 注意：bf16 转换必须 round-to-nearest-even（截断会引入系统性负偏，长序列累积放大）
static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t round = 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)((u + round) >> 16);
}
static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// IEEE 半精度（round-to-nearest-even）
static uint16_t f32_to_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = (int32_t)((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = u & 0x7fffff;
    if (((u >> 23) & 0xff) == 0xff)  // inf/nan
        return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);  // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  // underflow -> 0
        mant |= 0x800000;
        int shift = 14 - exp;
        uint32_t half = mant >> shift;
        if ((mant & ((1u << shift) - 1)) > (1u << (shift - 1)) ||
            ((mant & ((1u << shift) - 1)) == (1u << (shift - 1)) && (half & 1)))
            half++;
        return (uint16_t)(sign | half);
    }
    uint16_t h = (uint16_t)(sign | (exp << 10) | (mant >> 13));
    if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1))) h++;
    return h;
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {  // subnormal
            int e = -1;
            uint32_t m = mant;
            while (!(m & 0x400)) { m <<= 1; e--; }
            u = sign | ((uint32_t)(127 + e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000 | (mant << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

enum class NetDtype { F32, BF16, F16 };
static NetDtype dtype_kind(bm_data_type_t t) {
    if (t == BM_BFLOAT16) return NetDtype::BF16;
    if (t == BM_FLOAT16) return NetDtype::F16;
    return NetDtype::F32;
}
// f32 host 数据 → 网络期望字节（F32 原样 / BF16 / F16）
static void f32_to_net(const float* src, uint16_t* dst, size_t n, NetDtype d) {
    if (d == NetDtype::F32) {
        std::memcpy(dst, src, n * sizeof(float));
        return;
    }
    for (size_t i = 0; i < n; i++)
        dst[i] = d == NetDtype::BF16 ? f32_to_bf16(src[i]) : f32_to_f16(src[i]);
}
// 网络字节 → f32 host
static void net_to_f32(const uint16_t* src, float* dst, size_t n, NetDtype d) {
    if (d == NetDtype::F32) {
        std::memcpy(dst, src, n * sizeof(float));
        return;
    }
    for (size_t i = 0; i < n; i++)
        dst[i] = d == NetDtype::BF16 ? bf16_to_f32(src[i]) : f16_to_f32(src[i]);
}

namespace qwen3tts {

TtsEngine::~TtsEngine() { deinit(); }

bool TtsEngine::init(const std::string& talker_bmodel, const std::string& cp_bmodel,
                     const std::string& codec_bmodel, const std::string& model_dir, int device,
                     const std::string& cp_cache_bmodel) {
    if (bm_dev_request(&handle_, device) != BM_SUCCESS) {
        fprintf(stderr, "[TTS] bm_dev_request failed\n"); return false;
    }
    // 先加载小 bmodel（cp/codec），最后加载大的 talker——验证多 bmodel 加载顺序是否影响 CP 精度
    cp_rt_ = bmrt_create(handle_);
    if (!cp_rt_ || !bmrt_load_bmodel(cp_rt_, cp_bmodel.c_str())) {
        fprintf(stderr, "[TTS] load cp failed\n"); deinit(); return false;
    }
    cp_cache_rt_ = cp_rt_;  // 默认 cache 网络在 cp bmodel 内
    if (!cp_cache_bmodel.empty()) {
        cp_cache_rt_ = bmrt_create(handle_);
        if (!cp_cache_rt_ || !bmrt_load_bmodel(cp_cache_rt_, cp_cache_bmodel.c_str())) {
            fprintf(stderr, "[TTS] load cp_cache failed\n"); deinit(); return false;
        }
        fprintf(stderr, "[TTS] cp cache 使用独立 bmodel: %s\n", cp_cache_bmodel.c_str());
    }
    codec_rt_ = bmrt_create(handle_);
    if (!codec_rt_ || !bmrt_load_bmodel(codec_rt_, codec_bmodel.c_str())) {
        fprintf(stderr, "[TTS] load codec failed\n"); deinit(); return false;
    }
    talker_rt_ = bmrt_create(handle_);
    if (!talker_rt_ || !bmrt_load_bmodel(talker_rt_, talker_bmodel.c_str())) {
        fprintf(stderr, "[TTS] load talker failed\n"); deinit(); return false;
    }

    net_embed_text_ = bmrt_get_network_info(talker_rt_, "embedding_text");
    net_embed_code_ = bmrt_get_network_info(talker_rt_, "embedding_code");
    // codec_head 优先从 cp bmodel 取（可选 F32 版 combine 在 cp_allf32 里），否则 talker
    net_codec_head_ = bmrt_get_network_info(cp_rt_, "codec_head");
    if (!net_codec_head_) net_codec_head_ = bmrt_get_network_info(talker_rt_, "codec_head");
    for (int i = 0; i < TtsConfig::NUM_LAYERS; i++) {
        auto* b = bmrt_get_network_info(talker_rt_, ("talker_block_" + std::to_string(i)).c_str());
        auto* c = bmrt_get_network_info(talker_rt_, ("talker_block_cache_" + std::to_string(i)).c_str());
        if (!b || !c) {
            fprintf(stderr, "[TTS] talker bmodel 缺网络 talker_block_%d / cache\n", i);
            deinit(); return false;
        }
        talker_blocks_.push_back(b);
        talker_blocks_cache_.push_back(c);
    }
    for (int i = 0; i < TtsConfig::CP_LAYERS; i++) {
        auto* b = bmrt_get_network_info(cp_rt_, ("cp_block_" + std::to_string(i)).c_str());
        auto* c = bmrt_get_network_info(cp_cache_rt_, ("cp_block_cache_" + std::to_string(i)).c_str());
        if (!b || !c) {
            fprintf(stderr, "[TTS] cp bmodel 缺网络 cp_block_%d / cache\n", i);
            deinit(); return false;
        }
        cp_blocks_.push_back(b);
        cp_blocks_cache_.push_back(c);
    }
    for (int g = 0; g < 15; g++) {
        auto* h = bmrt_get_network_info(cp_rt_, ("cp_lm_head_" + std::to_string(g)).c_str());
        if (!h) {
            fprintf(stderr, "[TTS] cp bmodel 缺网络 cp_lm_head_%d（导出 15 个，勿漏 embedding_14）\n", g);
            deinit(); return false;
        }
        cp_lm_head_.push_back(h);
    }
    for (int e = 0; e < 15; e++) {
        auto* em = bmrt_get_network_info(cp_rt_, ("cp_embedding_" + std::to_string(e)).c_str());
        if (!em) {
            fprintf(stderr, "[TTS] cp bmodel 缺网络 cp_embedding_%d\n", e);
            deinit(); return false;
        }
        cp_embedding_.push_back(em);
    }
    cp_lm_head_all_ = bmrt_get_network_info(cp_rt_, "cp_lm_head_all");       // 可选（可为 null 回退单网络）
    cp_embedding_all_ = bmrt_get_network_info(cp_rt_, "cp_embedding_all");
    net_codec_ = bmrt_get_network_info(codec_rt_, "codec_decoder");
    if (!net_codec_) {
        fprintf(stderr, "[TTS] codec bmodel 缺网络 codec_decoder\n"); deinit(); return false;
    }

    // KV cache 设备常驻（talker [8,SEQLEN,128] f32，CP [8,15,128] f32）
    const size_t tk_n = (size_t)TtsConfig::KV_HEADS * TtsConfig::SEQLEN * TtsConfig::HEAD_DIM;
    const size_t ck_n = (size_t)TtsConfig::KV_HEADS * 16 * TtsConfig::HEAD_DIM;
    std::vector<float> zeros(std::max(tk_n, ck_n), 0.0f);
    talker_k_dev_.resize(TtsConfig::NUM_LAYERS);
    talker_v_dev_.resize(TtsConfig::NUM_LAYERS);
    for (int i = 0; i < TtsConfig::NUM_LAYERS; i++) {
        if (bm_malloc_device_byte(handle_, &talker_k_dev_[i], tk_n * sizeof(float)) != BM_SUCCESS ||
            bm_malloc_device_byte(handle_, &talker_v_dev_[i], tk_n * sizeof(float)) != BM_SUCCESS) {
            fprintf(stderr, "[TTS] alloc talker KV dev failed\n"); deinit(); return false;
        }
        bm_memcpy_s2d(handle_, talker_k_dev_[i], zeros.data());
        bm_memcpy_s2d(handle_, talker_v_dev_[i], zeros.data());
    }
    cp_k_dev_.resize(TtsConfig::CP_LAYERS);
    cp_v_dev_.resize(TtsConfig::CP_LAYERS);
    for (int i = 0; i < TtsConfig::CP_LAYERS; i++) {
        if (bm_malloc_device_byte(handle_, &cp_k_dev_[i], ck_n * sizeof(float)) != BM_SUCCESS ||
            bm_malloc_device_byte(handle_, &cp_v_dev_[i], ck_n * sizeof(float)) != BM_SUCCESS) {
            fprintf(stderr, "[TTS] alloc cp KV dev failed\n"); deinit(); return false;
        }
        bm_memcpy_s2d(handle_, cp_k_dev_[i], zeros.data());
        bm_memcpy_s2d(handle_, cp_v_dev_[i], zeros.data());
    }
    if (!tokenizer_.load(model_dir)) { deinit(); return false; }
    inited_ = true;
    return true;
}

void TtsEngine::deinit() {
    // 无条件执行（幂等）：init 失败路径也调用，确保不泄漏已分配资源
    if (handle_) {
        for (auto& m : talker_k_dev_) if (m.u.device.device_addr) bm_free_device(handle_, m);
        for (auto& m : talker_v_dev_) if (m.u.device.device_addr) bm_free_device(handle_, m);
        for (auto& m : cp_k_dev_) if (m.u.device.device_addr) bm_free_device(handle_, m);
        for (auto& m : cp_v_dev_) if (m.u.device.device_addr) bm_free_device(handle_, m);
    }
    talker_k_dev_.clear(); talker_v_dev_.clear();
    cp_k_dev_.clear(); cp_v_dev_.clear();
    if (talker_rt_) bmrt_destroy(talker_rt_);
    if (cp_cache_rt_ && cp_cache_rt_ != cp_rt_) bmrt_destroy(cp_cache_rt_);
    if (cp_rt_) bmrt_destroy(cp_rt_);
    if (codec_rt_) bmrt_destroy(codec_rt_);
    if (handle_) bm_dev_free(handle_);
    talker_rt_ = cp_rt_ = cp_cache_rt_ = codec_rt_ = nullptr;
    handle_ = nullptr;
    inited_ = false;
}

bool TtsEngine::test_cp_only() {
    // 跳过 talker：从文件读固定 last_hidden（board 输入），跑完整 CP 链路
    FILE* f = fopen("/tmp/test_last_hidden.bin", "rb");
    if (!f) { fprintf(stderr, "[TEST] /tmp/test_last_hidden.bin not found\n"); return false; }
    std::vector<float> ph(TtsConfig::HIDDEN);
    if (fread(ph.data(), sizeof(float), TtsConfig::HIDDEN, f) != TtsConfig::HIDDEN) {
        fclose(f); return false;
    }
    fclose(f);
    std::vector<int> codes16;
    bool ok = code_predictor_generate(1995, ph, codes16);
    if (ok && codes16.size() == 16)
        fprintf(stderr, "[TEST] cp_only codes16[0..15]=%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                codes16[0], codes16[1], codes16[2], codes16[3], codes16[4], codes16[5], codes16[6], codes16[7],
                codes16[8], codes16[9], codes16[10], codes16[11], codes16[12], codes16[13], codes16[14], codes16[15]);
    else
        fprintf(stderr, "[TEST] cp_only failed, codes16 size=%zu\n", codes16.size());
    return ok;
}

int TtsEngine::argmax(const std::vector<float>& v) const {
    int best = 0;
    for (int i = 1; i < (int)v.size(); i++) if (v[i] > v[best]) best = i;
    return best;
}

int TtsEngine::speaker_id(const std::string& s) const {
    if (s == "Vivian" || s == "vivian") return 3065;
    if (s == "Serena" || s == "serena") return 3066;
    if (s == "Uncle_Fu" || s == "uncle_fu") return 3010;
    if (s == "Ryan" || s == "ryan") return 3061;
    if (s == "Aiden" || s == "aiden") return 2861;
    if (s == "Ono_Anna" || s == "ono_anna") return 2873;
    if (s == "Sohee" || s == "sohee") return 2864;
    if (s == "Eric" || s == "eric") return 2875;
    if (s == "Dylan" || s == "dylan") return 2878;
    return 3065;
}

int TtsEngine::language_id(const std::string& l) const {
    if (l == "Chinese" || l == "chinese" || l == "zh") return 2055;
    if (l == "English" || l == "english" || l == "en") return 2050;
    return 2055;
}

bool TtsEngine::run_net(void* rt, const bm_net_info_t* net,
                        const std::vector<const void*>& in_hosts,
                        const std::vector<void*>& out_hosts) {
    int ni = net->input_num, no = net->output_num;
    static int rn_calls = 0;
    bool dbg = getenv("TIME_CP") != nullptr;
    rn_calls++;
    std::vector<bm_tensor_t> ins(ni), outs(no);
    if (dbg) fprintf(stderr, "[RN%d] %s in=%d out=%d\n", rn_calls, net->name, ni, no);
    for (int i = 0; i < ni; i++) {
        bmrt_tensor(&ins[i], rt, net->input_dtypes[i], net->stages[0].input_shapes[i]);
        if (dbg) fprintf(stderr, "[RN%d]  in[%d] s2d %s\n", rn_calls, i, net->input_names[i]);
        if (i < (int)in_hosts.size() && in_hosts[i])
            bm_memcpy_s2d(handle_, ins[i].device_mem, (void*)in_hosts[i]);
    }
    for (int i = 0; i < no; i++)
        bmrt_tensor(&outs[i], rt, net->output_dtypes[i], net->stages[0].output_shapes[i]);
    if (dbg) fprintf(stderr, "[RN%d] launch %s\n", rn_calls, net->name);
    bool ok = bmrt_launch_tensor_ex(rt, net->name, ins.data(), ni, outs.data(), no, true, false);
    if (dbg) fprintf(stderr, "[RN%d] launch ret=%d\n", rn_calls, ok);
    if (ok) bm_thread_sync(handle_);
    if (ok)
        for (int i = 0; i < no; i++)
            if (i < (int)out_hosts.size() && out_hosts[i])
                bm_memcpy_d2s(handle_, out_hosts[i], outs[i].device_mem);
    for (int i = 0; i < ni; i++) bm_free_device(handle_, ins[i].device_mem);
    for (int i = 0; i < no; i++) bm_free_device(handle_, outs[i].device_mem);
    return ok;
}

std::vector<float> TtsEngine::codec_embed(const std::vector<int>& ids) {
    std::vector<float> result;
    if (ids.empty()) return result;
    // pad 长度自适应网络输入 shape（bmodel 按 SEQLEN 编译，兼容不同版本）
    const auto& sh = net_embed_code_->stages[0].input_shapes[0];
    int pad_len = sh.num_dims > 1 ? sh.dims[1] : 1;
    std::vector<int32_t> ids_pad(pad_len, 0);
    for (size_t i = 0; i < ids.size() && i < (size_t)pad_len; i++) ids_pad[i] = ids[i];
    std::vector<float> out((size_t)pad_len * TtsConfig::HIDDEN);
    if (!run_net(talker_rt_, net_embed_code_, {ids_pad.data()}, {out.data()})) return result;
    result.resize(ids.size() * TtsConfig::HIDDEN);
    std::memcpy(result.data(), out.data(), result.size() * sizeof(float));
    return result;
}

std::vector<float> TtsEngine::text_embed(const std::vector<int>& ids) {
    std::vector<float> result;
    if (ids.empty()) return result;
    const auto& sh = net_embed_text_->stages[0].input_shapes[0];
    int pad_len = sh.num_dims > 1 ? sh.dims[1] : 1;
    std::vector<int32_t> ids_pad(pad_len, 0);
    for (size_t i = 0; i < ids.size() && i < (size_t)pad_len; i++) ids_pad[i] = ids[i];
    std::vector<float> out((size_t)pad_len * TtsConfig::HIDDEN);
    if (!run_net(talker_rt_, net_embed_text_, {ids_pad.data()}, {out.data()})) return result;
    result.resize(ids.size() * TtsConfig::HIDDEN);
    std::memcpy(result.data(), out.data(), result.size() * sizeof(float));
    return result;
}

static void add_vec(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

bool TtsEngine::build_prefill_embeds(const std::vector<int>& input_ids,
                                     std::vector<float>& embeds, int& prefill_len) {
    const int H = TtsConfig::HIDDEN;
    int L = (int)input_ids.size();
    // H1 防护：prefill 长度超过 SEQLEN 时 talker_prefill 的固定 shape 缓冲会越界写，直接拒绝
    if (L + 3 > TtsConfig::SEQLEN) {   // prefill_len = L + 3
        fprintf(stderr, "[TTS] text too long: prefill %d > SEQLEN %d\n", L + 3, TtsConfig::SEQLEN);
        return false;
    }
    std::vector<float> text = text_embed(input_ids);
    if ((int)text.size() != L * H) return false;

    auto tts_bos = text_embed({TtsConfig::TTS_BOS});
    auto tts_eos = text_embed({TtsConfig::TTS_EOS});
    if (tts_pad_emb_.size() != (size_t)H) tts_pad_emb_ = text_embed({TtsConfig::TTS_PAD});
    const std::vector<float>& tts_pad = tts_pad_emb_;

    int lang = language_id(current_lang_);
    int spk = speaker_id(current_speaker_);
    std::vector<float> codec = codec_embed({TtsConfig::CODEC_THINK, TtsConfig::CODEC_THINK_BOS, lang,
                                            TtsConfig::CODEC_THINK_EOS, spk, TtsConfig::CODEC_PAD, TtsConfig::CODEC_BOS});
    if ((int)codec.size() != 7 * H) return false;
    auto ce = [&](int idx) { return &codec[(size_t)idx * H]; };
    auto te = [&](int idx) { return &text[(size_t)idx * H]; };

    int n_body = L - 3 - 5;
    prefill_len = 3 + 6 + (n_body + 1) + 1;
    embeds.assign((size_t)prefill_len * H, 0.0f);
    float* e = embeds.data();
    for (int i = 0; i < 3; i++) std::memcpy(e + (size_t)i * H, te(i), H * sizeof(float));
    e += 3 * H;
    for (int i = 0; i < 6; i++) {
        const float* base = (i == 5) ? tts_bos.data() : tts_pad.data();
        std::memcpy(e, base, H * sizeof(float));
        add_vec(e, ce(i), H);
        e += H;
    }
    std::vector<float> pad_emb = codec_embed({TtsConfig::CODEC_PAD});
    for (int i = 0; i < n_body; i++) {
        std::memcpy(e, te(3 + i), H * sizeof(float));
        add_vec(e, pad_emb.data(), H);
        e += H;
    }
    std::memcpy(e, tts_eos.data(), H * sizeof(float));
    add_vec(e, pad_emb.data(), H);
    e += H;
    std::vector<float> bos_emb = codec_embed({TtsConfig::CODEC_BOS});
    std::memcpy(e, tts_pad.data(), H * sizeof(float));
    add_vec(e, bos_emb.data(), H);
    return true;
}

bool TtsEngine::talker_prefill(const std::vector<float>& embeds, int prefill_len,
                               std::vector<float>& first_logits) {
    const int H = TtsConfig::HIDDEN, S = TtsConfig::SEQLEN;
    std::vector<float> hid_in((size_t)S * H, 0.0f);
    std::memcpy(hid_in.data(), embeds.data(), (size_t)prefill_len * H * sizeof(float));
    std::vector<int32_t> pos(3 * S);
    for (int d = 0; d < 3; d++) for (int i = 0; i < S; i++) pos[d * S + i] = i;
    std::vector<float> mask((size_t)S * S, -1e30f);
    for (int i = 0; i < prefill_len; i++) for (int j = 0; j <= i; j++) mask[(size_t)i * S + j] = 0.0f;

    std::vector<float> hid_out((size_t)S * H);
    const size_t kv_n = (size_t)TtsConfig::KV_HEADS * S * TtsConfig::HEAD_DIM;
    std::vector<float> k(kv_n), v(kv_n);
    for (int i = 0; i < TtsConfig::NUM_LAYERS; i++) {
        if (!run_net(talker_rt_, talker_blocks_[i], {hid_in.data(), pos.data(), mask.data()},
                     {hid_out.data(), k.data(), v.data()})) return false;
        // ChatTTS 已知坑：padding 位置全 -inf 产生 NaN，污染后续层。每层后清零 padding hidden。
        std::fill(hid_out.begin() + (size_t)prefill_len * H, hid_out.end(), 0.0f);
        // 同时清零 padding KV（避免 decode 阶段 Q·K 点积读到 NaN）
        for (int h = 0; h < TtsConfig::KV_HEADS; h++)
            for (int sp = prefill_len; sp < S; sp++) {
                std::fill(k.begin() + ((size_t)h * S + sp) * TtsConfig::HEAD_DIM,
                          k.begin() + ((size_t)h * S + sp + 1) * TtsConfig::HEAD_DIM, 0.0f);
                std::fill(v.begin() + ((size_t)h * S + sp) * TtsConfig::HEAD_DIM,
                          v.begin() + ((size_t)h * S + sp + 1) * TtsConfig::HEAD_DIM, 0.0f);
            }
        // 写入设备常驻 KV（[8,S,128] 布局与网络输出一致，一次 s2d）
        bm_memcpy_s2d(handle_, talker_k_dev_[i], k.data());
        bm_memcpy_s2d(handle_, talker_v_dev_[i], v.data());
        std::swap(hid_in, hid_out);
    }
    prefill_last_hidden_.assign(H, 0.0f);
    std::memcpy(prefill_last_hidden_.data(), hid_in.data() + (size_t)(prefill_len - 1) * H, H * sizeof(float));
    // debug：dump 固定 last_hidden 供 --cp_only 复现
    if (getenv("TTS_DUMP_LAST_HIDDEN")) {
        FILE* f = fopen("/tmp/test_last_hidden.bin", "wb");
        if (f) { fwrite(prefill_last_hidden_.data(), sizeof(float), H, f); fclose(f); }
    }
    fprintf(stderr, "[TTS] last_hidden[0..3]=%.3f %.3f %.3f %.3f\n",
            prefill_last_hidden_[0], prefill_last_hidden_[1], prefill_last_hidden_[2], prefill_last_hidden_[3]);

    std::vector<float> logits(TtsConfig::VOCAB);
    if (!run_net(talker_rt_, net_codec_head_, {prefill_last_hidden_.data()}, {logits.data()})) return false;
    first_logits = logits;
    return true;
}

bool TtsEngine::talker_decode_step(int code0, const std::vector<int>& codes16,
                                   int pos, std::vector<float>& next_logits) {
    static double acc = 0; static int cnt = 0;
    auto t0 = std::chrono::steady_clock::now();
    const int H = TtsConfig::HIDDEN, S = TtsConfig::SEQLEN;
    // 组合输入 embedding：code0 + code1..code15（来自 CP 缓存的 last_cp_embs_，免重复 launch）+ TTS_PAD
    std::vector<float> emb(H, 0.0f);
    if (last_cp_embs_.size() == 16) {
        for (const auto& e : last_cp_embs_) add_vec(emb.data(), e.data(), H);
    } else {
        // 兜底：无缓存时重新计算（不应发生）
        std::vector<float> c0 = codec_embed({code0});
        std::memcpy(emb.data(), c0.data(), H * sizeof(float));
        for (int i = 1; i < 16; i++) {
            std::vector<int32_t> id = {codes16[i]};
            std::vector<float> out(H);
            if (!run_net(cp_rt_, cp_embedding_[i - 1], {id.data()}, {out.data()})) return false;
            add_vec(emb.data(), out.data(), H);
        }
    }
    if (tts_pad_emb_.size() != (size_t)H) tts_pad_emb_ = text_embed({TtsConfig::TTS_PAD});
    add_vec(emb.data(), tts_pad_emb_.data(), H);

    if (pos >= S) {  // KV cache 长度上限（SEQLEN=256），超出即终止
        fprintf(stderr, "[TTS] talker pos %d >= SEQLEN %d, force stop\n", pos, S);
        return false;
    }
    std::vector<float> hid = emb;
    std::vector<int32_t> pos3 = {pos, pos, pos};
    std::vector<float> mask(S + 1, -1e30f);
    for (int j = 0; j < pos; j++) mask[j] = 0.0f;   // 0..pos-1 真实历史
    mask[S] = 0.0f;                                  // 新 token 自身（concat 索引 S）

    std::vector<float> hid_out(H);
    const size_t nk_n = (size_t)TtsConfig::KV_HEADS * TtsConfig::HEAD_DIM;
    std::vector<float> nk(nk_n), nv(nk_n);
    // ---- 批处理：28 层连续 launch（hidden 走设备内存直连），一次 sync，批量 d2s ----
    const int L = TtsConfig::NUM_LAYERS;
    std::vector<std::vector<bm_tensor_t>> ins_t(L), outs_t(L);
    for (int i = 0; i < L; i++) {
        const bm_net_info_t* net = talker_blocks_cache_[i];
        int ni = net->input_num, no = net->output_num;
        ins_t[i].resize(ni);
        outs_t[i].resize(no);
        for (int j = 0; j < ni; j++) {
            const char* nm = net->input_names[j];
            if (!strcmp(nm, "input_states")) {
                if (i == 0) {
                    bmrt_tensor(&ins_t[i][j], talker_rt_, net->input_dtypes[j],
                                net->stages[0].input_shapes[j]);
                    bm_memcpy_s2d(handle_, ins_t[i][j].device_mem, hid.data());
                } else {
                    // hidden 直连上一层输出（设备内存，零拷贝）
                    bmrt_tensor_with_device(&ins_t[i][j], outs_t[i - 1][0].device_mem,
                                            net->input_dtypes[j], net->stages[0].input_shapes[j]);
                }
            } else if (!strcmp(nm, "position_ids") || !strcmp(nm, "attention_mask")) {
                // 每层共享同一份 pos/mask（只上传一次）
                if (i == 0) {
                    bmrt_tensor(&ins_t[i][j], talker_rt_, net->input_dtypes[j],
                                net->stages[0].input_shapes[j]);
                    bm_memcpy_s2d(handle_, ins_t[i][j].device_mem,
                                  !strcmp(nm, "position_ids") ? (void*)pos3.data() : (void*)mask.data());
                } else {
                    bmrt_tensor_with_device(&ins_t[i][j], ins_t[0][j].device_mem,
                                            net->input_dtypes[j], net->stages[0].input_shapes[j]);
                }
            } else {  // history_k / history_v：零拷贝直连设备常驻 KV
                bm_device_mem_t kv = !strcmp(nm, "history_k") ? talker_k_dev_[i]
                                                              : talker_v_dev_[i];
                bmrt_tensor_with_device(&ins_t[i][j], kv, net->input_dtypes[j],
                                        net->stages[0].input_shapes[j]);
            }
        }
        for (int j = 0; j < no; j++)
            bmrt_tensor(&outs_t[i][j], talker_rt_, net->output_dtypes[j],
                        net->stages[0].output_shapes[j]);
    }
    for (int i = 0; i < L; i++) {
        const bm_net_info_t* net = talker_blocks_cache_[i];
        if (!bmrt_launch_tensor_ex(talker_rt_, net->name, ins_t[i].data(), ins_t[i].size(),
                                   outs_t[i].data(), outs_t[i].size(), true, false))
            return false;
    }
    bm_thread_sync(handle_);
    // 新 KV 直接 d2d 写设备常驻槽 pos（head-major 逐 head 512B 小拷贝）；仅 hidden d2s
    const size_t row = (size_t)TtsConfig::HEAD_DIM * sizeof(float);   // 512B
    for (int i = 0; i < L; i++) {
        for (int h = 0; h < TtsConfig::KV_HEADS; h++) {
            size_t dst_off = ((size_t)h * TtsConfig::SEQLEN + pos) * row;
            bm_memcpy_d2d_byte(handle_, talker_k_dev_[i], dst_off, outs_t[i][1].device_mem,
                               (size_t)h * row, row);
            bm_memcpy_d2d_byte(handle_, talker_v_dev_[i], dst_off, outs_t[i][2].device_mem,
                               (size_t)h * row, row);
        }
    }
    bm_memcpy_d2s(handle_, hid_out.data(), outs_t[L - 1][0].device_mem);
    bm_thread_sync(handle_);
    // 释放临时 tensor
    for (int i = 0; i < L; i++) {
        const bm_net_info_t* net = talker_blocks_cache_[i];
        for (int j = 0; j < net->input_num; j++) {
            const char* nm = net->input_names[j];
            bool shared = !strcmp(nm, "position_ids") || !strcmp(nm, "attention_mask");
            bool chained = !strcmp(nm, "input_states") && i > 0;
            bool kvdev = !strcmp(nm, "history_k") || !strcmp(nm, "history_v");
            if (!shared && !chained && !kvdev) bm_free_device(handle_, ins_t[i][j].device_mem);
        }
        for (int j = 0; j < net->output_num; j++)
            bm_free_device(handle_, outs_t[i][j].device_mem);
    }
    if (ins_t[0].size() > 2) {
        bm_free_device(handle_, ins_t[0][1].device_mem);  // position_ids
        bm_free_device(handle_, ins_t[0][2].device_mem);  // attention_mask
    }
    hid = hid_out;
    prefill_last_hidden_ = hid;  // 新的 past_hidden（下一步 code_predictor 用）
    // debug：dump 每帧 last_hidden（差分定位长句误差累积）
    if (getenv("TTS_DUMP_FRAME_HIDDEN")) {
        static FILE* fd = fopen("/tmp/frame_hidden.bin", "wb");
        if (fd) { fwrite(hid.data(), sizeof(float), hid.size(), fd); }
    }
    std::vector<float> logits(TtsConfig::VOCAB);
    if (!run_net(talker_rt_, net_codec_head_, {hid.data()}, {logits.data()})) return false;
    next_logits = logits;
    acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cnt++;
    if (getenv("TIME_CP") && cnt % 32 == 0)
        fprintf(stderr, "[TIME_TALKER] avg=%.1fms/step\n", acc / cnt * 1000);
    return true;
}

bool TtsEngine::code_predictor_generate(int code0, const std::vector<float>& past_hidden,
                                        std::vector<int>& codes16) {
    const int H = TtsConfig::HIDDEN;
    const bool tcp = getenv("TIME_CP") != nullptr;
    auto tc0 = std::chrono::steady_clock::now();
    double t_prefill = 0, t_emb = 0, t_blocks = 0, t_lm = 0;
    codes16.clear();
    codes16.push_back(code0);
    std::vector<float> c0 = codec_embed({code0});
    fprintf(stderr, "[DBG] cp: codec_embed done\n");
    last_cp_embs_.clear();  // 缓存 code0 + 15 个 code 的 embedding（供 talker_decode_step 直接求和）
    last_cp_embs_.push_back(c0);
    std::vector<float> pf(2 * H, 0.0f);
    std::memcpy(pf.data(), past_hidden.data(), H * sizeof(float));
    std::memcpy(pf.data() + H, c0.data(), H * sizeof(float));
    std::vector<int32_t> pos2 = {0, 1};
    std::vector<float> mask2 = {0.0f, -1e30f, 0.0f, 0.0f};

    std::vector<float> hid_out(2 * H);
    const size_t ckv2 = (size_t)TtsConfig::KV_HEADS * 2 * TtsConfig::HEAD_DIM;
    const size_t ck15n = (size_t)TtsConfig::KV_HEADS * 16 * TtsConfig::HEAD_DIM;  // 16 槽真实 shape
    std::vector<float> k(ckv2), v(ckv2);
    fprintf(stderr, "[DBG] cp: prefill start\n");
    for (int i = 0; i < TtsConfig::CP_LAYERS; i++) {
        if (!run_net(cp_rt_, cp_blocks_[i], {pf.data(), pos2.data(), mask2.data()},
                     {hid_out.data(), k.data(), v.data()})) return false;
        // 写设备常驻 KV（[8,16,128]）：清零整层后逐 head d2d 前 2 槽（head-major）
        {
            std::vector<float> z16(ck15n, 0.0f);
            bm_memcpy_s2d(handle_, cp_k_dev_[i], z16.data());
            bm_memcpy_s2d(handle_, cp_v_dev_[i], z16.data());
            std::vector<float> k2((size_t)TtsConfig::KV_HEADS * 16 * TtsConfig::HEAD_DIM, 0.0f);
            std::vector<float> v2((size_t)TtsConfig::KV_HEADS * 16 * TtsConfig::HEAD_DIM, 0.0f);
            for (int h = 0; h < TtsConfig::KV_HEADS; h++) {
                std::memcpy(k2.data() + h * 16 * TtsConfig::HEAD_DIM,
                            k.data() + h * 2 * TtsConfig::HEAD_DIM,
                            2 * TtsConfig::HEAD_DIM * sizeof(float));
                std::memcpy(v2.data() + h * 16 * TtsConfig::HEAD_DIM,
                            v.data() + h * 2 * TtsConfig::HEAD_DIM,
                            2 * TtsConfig::HEAD_DIM * sizeof(float));
            }
            bm_memcpy_s2d(handle_, cp_k_dev_[i], k2.data());
            bm_memcpy_s2d(handle_, cp_v_dev_[i], v2.data());
        }
        std::swap(pf, hid_out);
    }
    if (tcp) t_prefill += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
    auto tc1 = std::chrono::steady_clock::now();
    std::vector<float> last(pf.begin() + H, pf.end());
    std::vector<float> lm(TtsConfig::CP_VOCAB);
    if (cp_lm_head_all_) {
        std::vector<int32_t> idx0 = {0};
        if (!run_net(cp_rt_, cp_lm_head_all_, {last.data(), idx0.data()}, {lm.data()})) return false;
    } else {
        if (!run_net(cp_rt_, cp_lm_head_[0], {last.data()}, {lm.data()})) return false;
    }
    if (tcp) t_lm += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc1).count();
    codes16.push_back(argmax(lm));

    std::vector<float> cur = last;
    for (int g = 1; g < 15; g++) {
        std::vector<int32_t> id = {codes16[g]};
        std::vector<float> e_out(H);
        if (tcp) t_emb += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc1).count();
        auto tc2 = std::chrono::steady_clock::now();
        if (cp_embedding_all_) {
            std::vector<int32_t> eidx = {g - 1};
            if (!run_net(cp_rt_, cp_embedding_all_, {id.data(), eidx.data()}, {e_out.data()})) return false;
        } else {
            if (!run_net(cp_rt_, cp_embedding_[g - 1], {id.data()}, {e_out.data()})) return false;
        }
        last_cp_embs_.push_back(e_out);  // 缓存 embedding（talker decode 直接求和）
        int pp = g + 1;  // 新 KV 写入槽（16 槽真实 shape）
        std::vector<int32_t> p1 = {pp};               // 新 token rotary 位置 = g+1（动态，与 15 槽版一致）
        std::vector<float> dm(17, -1e30f);
        for (int j = 0; j <= g; j++) dm[j] = 0.0f;   // 0..g 真实历史
        dm[16] = 0.0f;                                // 新 token 自身（concat 索引 16）
        std::vector<float> h_out(H);
        const size_t nk_n = (size_t)TtsConfig::KV_HEADS * TtsConfig::HEAD_DIM;
        std::vector<float> nk(nk_n), nv(nk_n);
        // ---- CP 5 层批处理：连续 launch + 一次 sync + 批量 d2s ----
        // 支持 bf16/f16 量化网络：输入 f32→量化，输出量化→f32（按网络 dtype 自动适配）
        const int L = TtsConfig::CP_LAYERS;
        std::vector<std::vector<bm_tensor_t>> ins_t(L), outs_t(L);
        std::vector<NetDtype> in_dt(L), out_dt(L);
        std::vector<std::vector<uint16_t>> q_hid(L), q_hk(L), q_hv(L);
        for (int l = 0; l < L; l++) {
            const bm_net_info_t* net = cp_blocks_cache_[l];
            int ni = net->input_num, no = net->output_num;
            ins_t[l].resize(ni);
            outs_t[l].resize(no);
            in_dt[l] = dtype_kind(net->input_dtypes[0]);
            out_dt[l] = dtype_kind(net->output_dtypes[0]);
            for (int j = 0; j < ni; j++) {
                const char* nm = net->input_names[j];
                NetDtype d = dtype_kind(net->input_dtypes[j]);
                if (!strcmp(nm, "input_states")) {
                    if (l == 0) {
                        bmrt_tensor(&ins_t[l][j], cp_cache_rt_, net->input_dtypes[j],
                                    net->stages[0].input_shapes[j]);
                        if (d != NetDtype::F32) {
                            q_hid[l].resize(H);
                            f32_to_net(e_out.data(), q_hid[l].data(), H, d);
                            bm_memcpy_s2d(handle_, ins_t[l][j].device_mem, q_hid[l].data());
                        } else {
                            bm_memcpy_s2d(handle_, ins_t[l][j].device_mem, e_out.data());
                        }
                    } else {
                        bmrt_tensor_with_device(&ins_t[l][j], outs_t[l - 1][0].device_mem,
                                                net->input_dtypes[j], net->stages[0].input_shapes[j]);
                    }
                } else if (!strcmp(nm, "position_ids") || !strcmp(nm, "attention_mask")) {
                    if (l == 0) {
                        bmrt_tensor(&ins_t[l][j], cp_cache_rt_, net->input_dtypes[j],
                                    net->stages[0].input_shapes[j]);
                        bm_memcpy_s2d(handle_, ins_t[l][j].device_mem,
                                      !strcmp(nm, "position_ids") ? (void*)p1.data() : (void*)dm.data());
                    } else {
                        bmrt_tensor_with_device(&ins_t[l][j], ins_t[0][j].device_mem,
                                                net->input_dtypes[j], net->stages[0].input_shapes[j]);
                    }
                } else {  // history_k / history_v：零拷贝直连设备常驻 KV（F32 网络）
                    bm_device_mem_t kv = !strcmp(nm, "history_k") ? cp_k_dev_[l]
                                                                  : cp_v_dev_[l];
                    bmrt_tensor_with_device(&ins_t[l][j], kv, net->input_dtypes[j],
                                            net->stages[0].input_shapes[j]);
                }
            }
            for (int j = 0; j < no; j++)
                bmrt_tensor(&outs_t[l][j], cp_cache_rt_, net->output_dtypes[j],
                            net->stages[0].output_shapes[j]);
        }
        for (int l = 0; l < L; l++) {
            const bm_net_info_t* net = cp_blocks_cache_[l];
            if (!bmrt_launch_tensor_ex(cp_cache_rt_, net->name, ins_t[l].data(), ins_t[l].size(),
                                       outs_t[l].data(), outs_t[l].size(), true, false))
                return false;
        }
        bm_thread_sync(handle_);
        if (tcp) t_blocks += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc2).count();
        auto tc3 = std::chrono::steady_clock::now();
        // 新 KV 直接 d2d 写设备常驻槽 pp（head-major 逐 head 512B）；仅 hidden d2s
        {
            const size_t row = (size_t)TtsConfig::HEAD_DIM * sizeof(float);
            for (int l = 0; l < L; l++) {
                for (int h = 0; h < TtsConfig::KV_HEADS; h++) {
                    size_t dst_off = ((size_t)h * 16 + pp) * row;
                    bm_memcpy_d2d_byte(handle_, cp_k_dev_[l], dst_off, outs_t[l][1].device_mem,
                                       (size_t)h * row, row);
                    bm_memcpy_d2d_byte(handle_, cp_v_dev_[l], dst_off, outs_t[l][2].device_mem,
                                       (size_t)h * row, row);
                }
            }
        }
        if (out_dt[L - 1] != NetDtype::F32) {
            std::vector<uint16_t> q_h(H);
            bm_memcpy_d2s(handle_, q_h.data(), outs_t[L - 1][0].device_mem);
            net_to_f32(q_h.data(), h_out.data(), H, out_dt[L - 1]);
        } else {
            bm_memcpy_d2s(handle_, h_out.data(), outs_t[L - 1][0].device_mem);
        }
        bm_thread_sync(handle_);
        // 释放临时 tensor
        for (int l = 0; l < L; l++) {
            const bm_net_info_t* net = cp_blocks_cache_[l];
            for (int j = 0; j < net->input_num; j++) {
                const char* nm = net->input_names[j];
                bool shared = !strcmp(nm, "position_ids") || !strcmp(nm, "attention_mask");
                bool chained = !strcmp(nm, "input_states") && l > 0;
                bool kvdev = !strcmp(nm, "history_k") || !strcmp(nm, "history_v");
                if (!shared && !chained && !kvdev) bm_free_device(handle_, ins_t[l][j].device_mem);
            }
            for (int j = 0; j < net->output_num; j++)
                bm_free_device(handle_, outs_t[l][j].device_mem);
        }
        for (int j = 1; j < (int)ins_t[0].size(); j++) {
            const char* nm = cp_blocks_cache_[0]->input_names[j];
            if (!strcmp(nm, "position_ids") || !strcmp(nm, "attention_mask"))
                bm_free_device(handle_, ins_t[0][j].device_mem);
        }
        e_out = h_out;
        std::vector<float> lm2(TtsConfig::CP_VOCAB);
        if (cp_lm_head_all_) {
            std::vector<int32_t> gidx = {g};
            if (!run_net(cp_rt_, cp_lm_head_all_, {e_out.data(), gidx.data()}, {lm2.data()})) return false;
        } else {
            if (!run_net(cp_rt_, cp_lm_head_[g], {e_out.data()}, {lm2.data()})) return false;
        }
        if (tcp) t_lm += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc3).count();
        codes16.push_back(argmax(lm2));
        cur = e_out;
    }
    // 补 code15 的 embedding（embedding_14），凑齐 16 个供 talker_decode_step 直接求和
    if (tcp) {
        auto tc4 = std::chrono::steady_clock::now();
        double tot = std::chrono::duration<double>(tc4 - tc0).count();
        fprintf(stderr, "[TIME_CP] total=%.1fms prefill=%.1f emb=%.1f blocks=%.1f lm=%.1f other=%.1f\n",
                tot * 1000, t_prefill * 1000, t_emb * 1000, t_blocks * 1000, t_lm * 1000,
                (tot - t_prefill - t_emb - t_blocks - t_lm) * 1000);
    }
    {
        std::vector<int32_t> id15 = {codes16[15]};
        std::vector<float> e15(H);
        if (cp_embedding_all_) {
            std::vector<int32_t> eidx = {14};
            if (!run_net(cp_rt_, cp_embedding_all_, {id15.data(), eidx.data()}, {e15.data()})) return false;
        } else {
            if (!run_net(cp_rt_, cp_embedding_[14], {id15.data()}, {e15.data()})) return false;
        }
        last_cp_embs_.push_back(e15);
    }
    return true;
}

bool TtsEngine::codec_decode(const std::vector<int>& codes, std::vector<float>& pcm, int& sr) {
    int T = (int)codes.size() / 16;
    if (T <= 0 || T > 325) return false;
    std::vector<int32_t> in_t(16 * 325, 0);
    for (int i = 0; i < T; i++)
        for (int q = 0; q < 16; q++) in_t[q * 325 + i] = codes[i * 16 + q];
    std::vector<float> wav(624000);
    if (!run_net(codec_rt_, net_codec_, {in_t.data()}, {wav.data()})) return false;
    pcm.assign(wav.begin(), wav.begin() + T * 1920);
    sr = 24000;
    return true;
}

int TtsEngine::sample_logits(const std::vector<float>& logits, int vocab, bool do_sample,
                             int top_k, float top_p, float temperature) {
    if (!do_sample) return argmax(logits);
    // 防护：logits 含 NaN/Inf（talker 量化偶发）时离散采样会死循环，回退 argmax
    for (int i = 0; i < vocab; i++)
        if (!std::isfinite(logits[i])) return argmax(logits);
    std::mt19937& rng = rng_;   // generate() 内已按 seed 播种，此处直接使用
    std::vector<std::pair<float, int>> p;
    p.reserve(vocab);
    float mx = *std::max_element(logits.begin(), logits.end());
    for (int i = 0; i < vocab; i++) p.push_back({logits[i] - mx, i});
    int k = std::min(top_k, vocab);
    std::partial_sort(p.begin(), p.begin() + k, p.end(), [](auto& a, auto& b) { return a.first > b.first; });
    p.resize(k);
    std::vector<float> w(k);
    float sum = 0;
    for (int i = 0; i < k; i++) { w[i] = std::exp(p[i].first / temperature); sum += w[i]; }
    // 防护：全零权重（exp 下溢）时 discrete_distribution 未定义行为，回退 argmax
    if (!(sum > 0.0f) || !std::isfinite(sum)) return argmax(logits);
    for (auto& x : w) x /= sum;
    float c = 0; int n = 1;
    for (int i = 0; i < k; i++) { c += w[i]; n = i + 1; if (c >= top_p) break; }
    std::discrete_distribution<int> dist(w.begin(), w.begin() + n);
    return p[dist(rng)].second;
}

bool TtsEngine::generate(const std::string& text, const std::string& speaker,
                         const std::string& language, std::vector<float>& pcm,
                         int& sample_rate, int max_new_tokens, bool do_sample, int seed) {
    if (!inited_) return false;
    current_speaker_ = speaker;
    current_lang_ = language;
    seed_ = seed;

    std::string full = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> input_ids = tokenizer_.encode(full);
    fprintf(stderr, "[TTS] input_ids=%zu\n", input_ids.size());
    if (input_ids.empty()) return false;

    std::vector<float> embeds;
    int plen = 0;
    if (!build_prefill_embeds(input_ids, embeds, plen)) return false;
    if (plen > TtsConfig::SEQLEN) {   // 双保险：text_embed 缓冲与 mask 均按 SEQLEN 分配
        fprintf(stderr, "[TTS] prefill_len %d > SEQLEN %d, reject\n", plen, TtsConfig::SEQLEN);
        return false;
    }
    fprintf(stderr, "[TTS] prefill_len=%d\n", plen);
    fprintf(stderr, "[TTS] emb[0..2]=%.3f %.3f %.3f  emb[last]=%.3f %.3f %.3f\n",
            embeds[0], embeds[1], embeds[2],
            embeds[(size_t)(plen-1)*1024], embeds[(size_t)(plen-1)*1024+1], embeds[(size_t)(plen-1)*1024+2]);
    std::vector<float> logits;
    if (!talker_prefill(embeds, plen, logits)) return false;
    int code0 = sample_logits(logits, TtsConfig::VOCAB, do_sample, 50, 1.0f, 0.9f);
    fprintf(stderr, "[TTS] logits[0..7]: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
            logits[0], logits[1], logits[2], logits[3], logits[4], logits[5], logits[6], logits[7]);
    fprintf(stderr, "[TTS] logits max idx=%d val=%.2f\n", argmax(logits), logits[argmax(logits)]);
    fprintf(stderr, "[TTS] prefill ok code0=%d\n", code0);

    rng_.seed(static_cast<uint32_t>(seed));   // M2：每条样本重新播种，seed 可逐条复现
    std::vector<int> all_codes;
    int pos = plen;
    for (int step = 0; step < max_new_tokens; step++) {
        if (pos >= TtsConfig::SEQLEN) {   // M1：KV 槽位耗尽，截断收尾（用已有 codes 继续 codec）
            fprintf(stderr, "[TTS] reach SEQLEN %d at pos %d, truncate\n", TtsConfig::SEQLEN, pos);
            break;
        }
        if (code0 == TtsConfig::CODEC_EOS) break;   // H3：EOS 帧不进入 codec（避免杂音帧）
        fprintf(stderr, "[TTS] step %d code0=%d\n", step, code0);
        std::vector<int> codes16;
        if (!code_predictor_generate(code0, prefill_last_hidden_, codes16)) return false;
        if (step < 2)
            fprintf(stderr, "[TTS] codes16[0..15]=%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                    codes16[0], codes16[1], codes16[2], codes16[3], codes16[4], codes16[5], codes16[6], codes16[7],
                    codes16[8], codes16[9], codes16[10], codes16[11], codes16[12], codes16[13], codes16[14], codes16[15]);
        for (int q = 0; q < 16; q++) all_codes.push_back(codes16[q]);
        std::vector<float> next_logits;
        if (!talker_decode_step(code0, codes16, pos, next_logits)) return false;
        code0 = sample_logits(next_logits, TtsConfig::VOCAB, do_sample, 50, 1.0f, 0.9f);
        pos++;
    }

    fprintf(stderr, "[TTS] decode done codes=%zu\n", all_codes.size());
    if (!codec_decode(all_codes, pcm, sample_rate)) return false;
    fprintf(stderr, "[TTS] codec ok\n");
    return true;
}

}  // namespace qwen3tts
