#include "moonshine_inference.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "utils/moonshine_features.h"
#include "utils/wav_reader.h"

using namespace moonshine;

namespace {
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bool run_net(void* rt, const bm_net_info_t* info,
             std::vector<bm_tensor_t>& ins, std::vector<bm_tensor_t>& outs,
             bm_handle_t handle) {
    if (!bmrt_launch_tensor_ex(rt, info->name, ins.data(), ins.size(),
                               outs.data(), outs.size(), true, false)) {
        fprintf(stderr, "[ERROR] bmrt_launch_tensor_ex failed for %s\n", info->name);
        return false;
    }
    bm_thread_sync(handle);
    return true;
}

size_t tensor_elems(const bm_shape_t& s) {
    size_t n = 1;
    for (int i = 0; i < s.num_dims; i++) n *= (size_t)s.dims[i];
    return n;
}

void save_bin(const std::string& path, const void* d, size_t bytes) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[ERROR] cannot write %s\n", path.c_str()); return; }
    if (bytes) fwrite(d, 1, bytes, f);
    fclose(f);
}
}  // namespace

MoonshineInference::MoonshineInference() = default;
MoonshineInference::~MoonshineInference() { release(); }

void MoonshineInference::release() {
    if (dec_rt_) { bmrt_destroy(dec_rt_); dec_rt_ = nullptr; }
    if (enc_rt_) { bmrt_destroy(enc_rt_); enc_rt_ = nullptr; }
    if (bm_handle_) { bm_dev_free(bm_handle_); bm_handle_ = nullptr; }
    enc_info_ = dec_info_ = nullptr;
    initialized_ = false;
}

// ---------- tokens.txt: "<token>\t<id>", 32768 行 ----------
bool MoonshineInference::load_vocab(const std::string& path) {
    vocab_.clear();
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[ERROR] cannot open %s\n", path.c_str());
        return false;
    }
    std::string line;
    int max_id = -1;
    std::vector<std::pair<int, std::string>> entries;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        int id = atoi(line.c_str() + tab + 1);
        entries.emplace_back(id, line.substr(0, tab));
        max_id = std::max(max_id, id);
    }
    vocab_.resize((size_t)max_id + 1);
    for (auto& e : entries) vocab_[e.first] = e.second;
    printf("[Vocab] %zu tokens loaded\n", vocab_.size());
    return vocab_.size() == (size_t)VOCAB;
}

int MoonshineInference::init(const char* model_dir, const char* precision) {
    if (initialized_) release();
    std::string dir = model_dir;
    std::string enc_path = dir + "/moonshine_encoder_" + precision + ".bmodel";
    std::string dec_path = dir + "/moonshine_decoder_" + precision + ".bmodel";

    if (!load_vocab(dir + "/tokens.txt")) return -1;

    if (bm_dev_request(&bm_handle_, 0) != BM_SUCCESS) {
        fprintf(stderr, "[ERROR] bm_dev_request failed\n");
        return -1;
    }
    enc_rt_ = bmrt_create(bm_handle_);
    if (!enc_rt_ || !bmrt_load_bmodel(enc_rt_, enc_path.c_str())) {
        fprintf(stderr, "[ERROR] load encoder bmodel %s\n", enc_path.c_str());
        release(); return -1;
    }
    const char** names = nullptr;
    bmrt_get_network_names(enc_rt_, &names);
    enc_info_ = bmrt_get_network_info(enc_rt_, names[0]);
    if (!enc_info_ || enc_info_->input_num != 1 || enc_info_->output_num != 1) {
        fprintf(stderr, "[ERROR] unexpected encoder io num\n");
        release(); return -1;
    }

    dec_rt_ = bmrt_create(bm_handle_);
    if (!dec_rt_ || !bmrt_load_bmodel(dec_rt_, dec_path.c_str())) {
        fprintf(stderr, "[ERROR] load decoder bmodel %s\n", dec_path.c_str());
        release(); return -1;
    }
    bmrt_get_network_names(dec_rt_, &names);
    dec_info_ = bmrt_get_network_info(dec_rt_, names[0]);
    if (!dec_info_ || dec_info_->input_num != 23 || dec_info_->output_num != 21) {
        fprintf(stderr, "[ERROR] unexpected decoder io num\n");
        release(); return -1;
    }
    printf("[Init] encoder %s, decoder %s (prec=%s)\n",
           enc_info_->input_names[0], dec_info_->input_names[0], precision);
    initialized_ = true;
    return 0;
}

bool MoonshineInference::run_encoder(const std::vector<float>& x_frames,
                                     std::vector<float>& enc_out) {
    std::vector<bm_tensor_t> ins(1), outs(1);
    bmrt_tensor(&ins[0], enc_rt_, enc_info_->input_dtypes[0],
                enc_info_->stages[0].input_shapes[0]);
    bmrt_tensor(&outs[0], enc_rt_, enc_info_->output_dtypes[0],
                enc_info_->stages[0].output_shapes[0]);
    bm_memcpy_s2d(bm_handle_, ins[0].device_mem, (void*)x_frames.data());
    bool ok = run_net(enc_rt_, enc_info_, ins, outs, bm_handle_);
    if (ok) {
        enc_out.resize(tensor_elems(outs[0].shape));
        bm_memcpy_d2s(bm_handle_, enc_out.data(), outs[0].device_mem);
        if (!debug_dir_.empty())
            save_bin(debug_dir_ + "/encoder_out.bin", enc_out.data(), enc_out.size() * 4);
    }
    bm_free_device(bm_handle_, ins[0].device_mem);
    bm_free_device(bm_handle_, outs[0].device_mem);
    return ok;
}

bool MoonshineInference::run_decoder(const std::vector<float>& enc_out,
                                     std::vector<int>& ids) {
    const bm_shape_t& s0 = dec_info_->stages[0].input_shapes[3];  // past_k_0
    int kv_t = s0.dims[1];    // 128
    int hid_d = s0.dims[2];   // 512
    int vocab = dec_info_->stages[0].output_shapes[0].dims[2];  // 32768
    int max_dec_len = kv_t;   // 128

    std::vector<bm_tensor_t> dins(dec_info_->input_num), douts(dec_info_->output_num);
    for (int i = 0; i < dec_info_->input_num; i++)
        bmrt_tensor(&dins[i], dec_rt_, dec_info_->input_dtypes[i],
                    dec_info_->stages[0].input_shapes[i]);
    for (int i = 0; i < dec_info_->output_num; i++)
        bmrt_tensor(&douts[i], dec_rt_, dec_info_->output_dtypes[i],
                    dec_info_->stages[0].output_shapes[i]);

    // KV cache 初始全 0; 第 t 步 new_k_i/new_v_i 写入槽位 t
    std::vector<int32_t> cache_k((size_t)N_LAYER * kv_t * hid_d, 0);
    std::vector<int32_t> cache_v((size_t)N_LAYER * kv_t * hid_d, 0);
    std::vector<float> logits(vocab), new_k(hid_d), new_v(hid_d);

    ids.clear();
    bool ok = true;
    int tok = TOK_SOS;
    FILE* tf = nullptr;
    if (!debug_dir_.empty()) {
        tf = fopen((debug_dir_ + "/tokens.txt").c_str(), "w");
    }
    for (int step = 0; step < max_dec_len; step++) {
        int32_t tok32 = tok, cl32 = step;
        bm_memcpy_s2d(bm_handle_, dins[0].device_mem, &tok32);
        bm_memcpy_s2d(bm_handle_, dins[1].device_mem, (void*)enc_out.data());
        bm_memcpy_s2d(bm_handle_, dins[2].device_mem, &cl32);
        for (int l = 0; l < N_LAYER; l++) {
            bm_memcpy_s2d(bm_handle_, dins[3 + l].device_mem,
                          &cache_k[(size_t)l * kv_t * hid_d]);
            bm_memcpy_s2d(bm_handle_, dins[13 + l].device_mem,
                          &cache_v[(size_t)l * kv_t * hid_d]);
        }
        if (!run_net(dec_rt_, dec_info_, dins, douts, bm_handle_)) { ok = false; break; }
        bm_memcpy_d2s(bm_handle_, logits.data(), douts[0].device_mem);
        for (int l = 0; l < N_LAYER; l++) {
            bm_memcpy_d2s(bm_handle_, new_k.data(), douts[1 + l].device_mem);
            bm_memcpy_d2s(bm_handle_, new_v.data(), douts[11 + l].device_mem);
            memcpy(&cache_k[(size_t)l * kv_t * hid_d + (size_t)step * hid_d],
                   new_k.data(), (size_t)hid_d * 4);
            memcpy(&cache_v[(size_t)l * kv_t * hid_d + (size_t)step * hid_d],
                   new_v.data(), (size_t)hid_d * 4);
        }
        int nid = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        ids.push_back(nid);
        tok = nid;
        if (!debug_dir_.empty()) {
            char p[256];
            snprintf(p, sizeof p, "%s/logits_%02d.bin", debug_dir_.c_str(), step);
            save_bin(p, logits.data(), (size_t)vocab * 4);
            if (tf) fprintf(tf, "%d\n", nid);
        }
        if (nid == TOK_EOS) break;
    }
    if (tf) fclose(tf);
    for (auto& t : dins) bm_free_device(bm_handle_, t.device_mem);
    for (auto& t : douts) bm_free_device(bm_handle_, t.device_mem);
    return ok;
}

// ---------- token 解码: 按 id 查表直接拼接(无 BPE merge) ----------
// ▁(U+2581) 开头的 token 前面加空格; 跳过特殊 token
// (<unk>/<s></s> = id 0/1/2, <0x00>..<0xFF> 字节 token = id 3..258)
std::string MoonshineInference::decode_tokens(const std::vector<int>& ids) {
    std::string text;
    for (int id : ids) {
        if (id < 0 || id >= (int)vocab_.size() || vocab_[id].empty()) continue;
        if (id <= 2) continue;              // <unk> <s> </s>
        if (id >= 3 && id <= 258) continue; // <0x00>..<0xFF> 字节 token
        const std::string& t = vocab_[id];
        // "▁" = UTF-8 E2 96 81
        if (t.size() >= 3 && (unsigned char)t[0] == 0xE2 &&
            (unsigned char)t[1] == 0x96 && (unsigned char)t[2] == 0x81) {
            if (!text.empty()) text += ' ';
            text += t.substr(3);
        } else {
            text += t;
        }
    }
    // 去除尾部多余空格(若末位 token 是独立 "▁")
    while (!text.empty() && text.back() == ' ') text.pop_back();
    return text;
}

std::string MoonshineInference::run(const char* wav_path, MoonshineStats* stats) {
    MoonshineStats st;
    if (!initialized_) {
        fprintf(stderr, "[ERROR] not initialized\n");
        return "";
    }

    // ---- 读 WAV(含 8k -> 16k 线性插值重采样) ----
    WavData wav;
    std::string err;
    if (!ReadWavResample16k(wav_path, &wav, &err)) {
        fprintf(stderr, "[ERROR] %s\n", err.c_str());
        return "";
    }
    st.audio_ms = wav.samples.size() * 1000.0 / 16000.0;
    printf("[Audio] %s: %.3fs @16k (%zu samples)\n", wav_path,
           wav.samples.size() / 16000.0, wav.samples.size());

    // ---- 特征提取(补零 + 分帧 + CMVN + asinh) ----
    auto t0 = Clock::now();
    std::vector<float> x_frames;
    moonshine::compute_x_frames(wav.samples, x_frames);
    st.feat_ms = ms_since(t0);
    if (!debug_dir_.empty())
        save_bin(debug_dir_ + "/x_frames.bin", x_frames.data(), x_frames.size() * 4);

    // ---- encoder ----
    std::vector<float> enc_out;
    if (!run_encoder(x_frames, enc_out)) return "";

    // ---- decoder 逐步贪心 ----
    auto t1 = Clock::now();
    std::vector<int> ids;
    if (!run_decoder(enc_out, ids)) return "";
    st.infer_ms = ms_since(t1);
    st.steps = (int)ids.size();

    std::string text = decode_tokens(ids);
    st.rtf = (st.feat_ms + st.infer_ms) / st.audio_ms;
    if (stats) *stats = st;
    return text;
}
