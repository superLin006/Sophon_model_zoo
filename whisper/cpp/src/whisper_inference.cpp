#include "whisper_inference.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <algorithm>

#define LOG(msg)   std::cout << "[INFO] " << msg << std::endl
#define LOGE(msg)  std::cerr << "[ERROR] " << msg << std::endl
#define LOGD(msg)  if (debug_mode_) { std::cout << "[DEBUG] " << msg << std::endl; }

// ============================================================
// BMRuntime 辅助：把 host float 数组推理一次
// 封装 bmrt_launch_tensor_ex 的样板代码
// ============================================================

// 用 bmrt 做单次推理：inputs/outputs 已在 device 上分配
static bool bm_infer(void* rt, const bm_net_info_t* info,
                     std::vector<bm_tensor_t>& in_tensors,
                     std::vector<bm_tensor_t>& out_tensors) {
    bool ok = bmrt_launch_tensor_ex(rt, info->name,
                                    in_tensors.data(),  in_tensors.size(),
                                    out_tensors.data(), out_tensors.size(),
                                    true,   // need_perf_opt
                                    false); // sync
    if (!ok) return false;
    // 等待推理完成（bmrt_get_bm_handle 返回 handle，sp4 API 只有一个参数）
    bm_handle_t handle = (bm_handle_t)bmrt_get_bm_handle(rt);
    bm_thread_sync(handle);
    return true;
}

// ============================================================
// Constructor / Destructor
// ============================================================

WhisperInference::WhisperInference() {}

WhisperInference::~WhisperInference() { release(); }

// ============================================================
// init
// ============================================================

int WhisperInference::init(const char* model_dir, const char* precision,
                           const char* model_name) {
    if (initialized_) return 0;

    const char* dbg = std::getenv("WHISPER_DEBUG");
    debug_mode_ = (dbg && std::string(dbg) == "1");

    LOG("Initializing from: " << model_dir << "  precision: " << precision
        << "  model: " << model_name);
    std::string base(model_dir);
    std::string prec(precision);
    std::string mname(model_name);

    // BM handle（device 0）
    if (bm_dev_request(&bm_handle_, 0) != BM_SUCCESS) {
        LOGE("bm_dev_request failed"); return -1;
    }

    // 加载 bmodel（文件名按 model_name 前缀）
    if (!load_encoder(base + "/whisper_" + mname + "_encoder_" + prec + ".bmodel")) return -1;
    if (!load_decoder(base + "/whisper_" + mname + "_decoder_" + prec + ".bmodel")) return -1;

    // ---- 从 bmodel net_info 运行时读取维度（base / turbo 通用）----
    // encoder: input mel [1, n_mels, 3000]，output audio_features [1, n_audio_ctx, n_state]
    {
        const auto& mel_shape = encoder_info_->stages[0].input_shapes[0];
        const auto& af_shape  = encoder_info_->stages[0].output_shapes[0];
        n_mels_      = mel_shape.dims[1];
        n_audio_ctx_ = af_shape.dims[1];
        n_state_     = af_shape.dims[2];
    }
    // decoder: 输入 4 个头 + 4*n_layer 个 KV → n_layer = (input_num - 4) / 4
    //          past_self_k_0 [1, padding_size, n_state]，logits [1,1,vocab_num]
    {
        n_layer_      = (decoder_info_->input_num - 4) / 4;
        padding_size_ = decoder_info_->stages[0].input_shapes[4].dims[1];  // past_self_k_0
        // logits 是第 0 个输出
        const auto& logits_shape = decoder_info_->stages[0].output_shapes[0];
        vocab_num_    = logits_shape.dims[logits_shape.num_dims - 1];
    }
    // 语言块后的特殊 token：turbo(100 langs) 比 base(99) 偏移 1，按 vocab 推导。
    //   num_languages = vocab_num - 51766；transcribe = EN + num_languages + 1
    {
        int num_languages = vocab_num_ - 51766;
        tok_transcribe_    = WHISPER_LANG_EN + num_languages + 1;
        tok_no_timestamps_ = tok_transcribe_ + 4;
    }
    LOG("Dims: n_mels=" << n_mels_ << " n_audio_ctx=" << n_audio_ctx_
        << " n_state=" << n_state_ << " n_layer=" << n_layer_
        << " padding=" << padding_size_ << " vocab=" << vocab_num_
        << " | transcribe=" << tok_transcribe_ << " notimestamps=" << tok_no_timestamps_);

    // mel filters [n_mels x 201]，文件名按 n_mels（mel_80_filters.txt / mel_128_filters.txt）
    {
        std::string mel_path = base + "/mel_" + std::to_string(n_mels_) + "_filters.txt";
        mel_filters_.resize((size_t)n_mels_ * MELS_FILTERS_SIZE);
        if (read_mel_filters(mel_path.c_str(), mel_filters_.data(),
                             n_mels_ * MELS_FILTERS_SIZE) != 0) {
            LOGE("Failed to load mel filters: " << mel_path); return -1;
        }
    }

    // positional embedding [padding_size, n_state]，npy 文件（跳过 128 字节 header）
    {
        std::string pe_path = base + "/positional_embedding.npy";
        FILE* fp = fopen(pe_path.c_str(), "rb");
        if (!fp) { LOGE("positional_embedding.npy not found: " << pe_path); return -1; }
        fseek(fp, 128, SEEK_SET);
        size_t sz = (size_t)padding_size_ * n_state_;
        positional_embedding_.resize(sz);
        if (fread(positional_embedding_.data(), sizeof(float), sz, fp) != sz) {
            LOGE("positional_embedding read error"); fclose(fp); return -1;
        }
        fclose(fp);
        LOG("Positional embedding loaded: [" << padding_size_ << ", " << n_state_ << "]");
    }

    // vocab
    vocab_.resize(vocab_num_);
    if (read_vocab((base + "/vocab.txt").c_str(), vocab_.data(), vocab_num_) != 0) {
        LOGE("Failed to load vocab"); return -1;
    }

    // KV cache 常驻 device，无需 host buffer（在 run_decoder 内绑定到 decoder KV input tensor）

    initialized_ = true;
    LOG("Init done");
    return 0;
}

bool WhisperInference::load_encoder(const std::string& path) {
    encoder_rt_ = bmrt_create(bm_handle_);
    if (!bmrt_load_bmodel(encoder_rt_, path.c_str())) {
        LOGE("Load encoder bmodel failed: " << path); return false;
    }
    // sp4 API: bmrt_get_network_names(rt, &names)
    const char** names = nullptr;
    bmrt_get_network_names(encoder_rt_, &names);
    encoder_info_ = bmrt_get_network_info(encoder_rt_, names[0]);
    LOG("Encoder loaded: " << path);
    return true;
}

bool WhisperInference::load_decoder(const std::string& path) {
    decoder_rt_ = bmrt_create(bm_handle_);
    if (!bmrt_load_bmodel(decoder_rt_, path.c_str())) {
        LOGE("Load decoder bmodel failed: " << path); return false;
    }
    const char** names = nullptr;
    bmrt_get_network_names(decoder_rt_, &names);
    decoder_info_ = bmrt_get_network_info(decoder_rt_, names[0]);
    LOG("Decoder loaded: " << path);
    return true;
}

// ============================================================
// run
// ============================================================

std::string WhisperInference::run(const char* audio_file, const char* language,
                                  TokenCallback callback) {
    if (!initialized_) { LOGE("Not initialized"); return ""; }

    LOG("Running on: " << audio_file);
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. 加载音频
    audio_buffer_t audio{};
    if (load_audio(audio_file, &audio) != 0) { LOGE("load_audio failed"); return ""; }

    // 2. Mel spectrogram
    std::vector<float> mel;
    audio_preprocess(&audio, mel_filters_.data(), n_mels_, mel);
    free_audio(&audio);
    // mel: [n_mels_, 3000]，需要扩展 batch 维 → [1, n_mels_, 3000]（内存连续，直接用）

    // 3. Encoder
    std::vector<float> audio_features;
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!run_encoder(mel, audio_features)) { LOGE("encoder failed"); return ""; }
    auto t2 = std::chrono::high_resolution_clock::now();

    // 4. Decoder
    int lang_token = WHISPER_LANG_EN;
    if (language && std::string(language) == "zh") lang_token = WHISPER_LANG_ZH;

    std::vector<int> tokens;
    if (!run_decoder(audio_features, lang_token, tokens, callback)) {
        LOGE("decoder failed"); return "";
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    std::string text = decode_tokens(tokens);

    auto enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto dec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto tot_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();

    LOG("Encoder: " << enc_ms << "ms  Decoder: " << dec_ms << "ms  Total: " << tot_ms << "ms"
        << "  Tokens: " << tokens.size());
    LOG("Result: " << text);
    return text;
}

// ============================================================
// Encoder inference
// ============================================================

bool WhisperInference::run_encoder(const std::vector<float>& mel,
                                   std::vector<float>& audio_features) {
    // 输入: mel [1, n_mels, 3000]
    // 输出: audio_features [1, n_audio_ctx, n_state]
    int in_count  = encoder_info_->input_num;
    int out_count = encoder_info_->output_num;

    std::vector<bm_tensor_t> ins(in_count), outs(out_count);

    // 分配输入 device mem 并上传
    bmrt_tensor(&ins[0], encoder_rt_, BM_FLOAT32,
                encoder_info_->stages[0].input_shapes[0]);
    bm_memcpy_s2d(bm_handle_, ins[0].device_mem,
                  (void*)mel.data());

    // 分配输出 device mem
    bmrt_tensor(&outs[0], encoder_rt_, BM_FLOAT32,
                encoder_info_->stages[0].output_shapes[0]);

    if (!bm_infer(encoder_rt_, encoder_info_, ins, outs)) {
        LOGE("encoder inference failed"); return false;
    }

    // 下载结果
    size_t out_sz = (size_t)n_audio_ctx_ * n_state_;
    audio_features.resize(out_sz);
    bm_memcpy_d2s(bm_handle_, audio_features.data(), outs[0].device_mem);

    // 释放 device mem
    bm_free_device(bm_handle_, ins[0].device_mem);
    bm_free_device(bm_handle_, outs[0].device_mem);

    LOGD("Encoder done, audio_features[0..4]: "
         << audio_features[0] << " " << audio_features[1] << " "
         << audio_features[2] << " " << audio_features[3]);
    return true;
}

// ============================================================
// Decoder inference（与 MTK 逻辑完全一致）
// ============================================================

void WhisperInference::reset_kv_cache() {
    // KV cache 现常驻 device，清零在 run_decoder 分配 in_ts 后做（bm_memset device buffer）。
    // 这里只重置状态标志。
    cache_len_               = 0;
    cross_cache_initialized_ = false;
}

void WhisperInference::get_position_embedding(int pos, float* out) {
    // positional_embedding_: [padding_size_, n_state_]
    if (pos >= 0 && pos < padding_size_) {
        std::memcpy(out,
                    positional_embedding_.data() + (size_t)pos * n_state_,
                    n_state_ * sizeof(float));
    } else {
        std::fill(out, out + n_state_, 0.f);
    }
}

void WhisperInference::create_self_attn_mask(int cache_len, float* out) {
    // [1, 1, 1, padding_size_+1]：有效位=0，无效位=-1e9
    // 与 MTK create_self_attn_mask 完全一致
    int len = padding_size_ + 1;
    std::fill(out, out + len, -1e9f);
    for (int i = 0; i < cache_len; i++) out[i] = 0.f;
    out[padding_size_] = 0.f;  // 当前 token 固定在末尾位
}

bool WhisperInference::run_decoder(const std::vector<float>& audio_features,
                                   int language_token,
                                   std::vector<int>& tokens,
                                   TokenCallback callback) {
    reset_kv_cache();
    tokens.clear();

    // 初始 SOT 序列：[SOT, lang, TRANSCRIBE, NO_TIMESTAMPS]
    // transcribe/notimestamps 按 vocab 推导，base/turbo 不同（见 init）。
    std::vector<int> init_tokens = {
        WHISPER_SOT, language_token, tok_transcribe_, tok_no_timestamps_
    };
    LOG("Initial tokens: [" << init_tokens[0] << ", " << init_tokens[1]
        << ", " << init_tokens[2] << ", " << init_tokens[3] << "]");

    // Decoder 输入布局（共 4 + 4*n_layer 个）：
    //   [0]=token [1]=audio_features [2]=pos_emb [3]=mask
    //   past_self_k[n_layer], past_self_v[n_layer], cross_k[n_layer], cross_v[n_layer]
    // 输出布局（共 1 + 4*n_layer 个）：
    //   [0]=logits, new_self_k[n_layer], new_self_v[n_layer], new_cross_k[n_layer], new_cross_v[n_layer]
    const int L = n_layer_;
    // input KV 段起点
    const int IN_PSK = 4;            // past_self_k
    const int IN_PSV = 4 + L;        // past_self_v
    const int IN_CK  = 4 + 2 * L;    // cross_k
    const int IN_CV  = 4 + 3 * L;    // cross_v
    // output KV 段起点
    const int OUT_NSK = 1;           // new_self_k
    const int OUT_NSV = 1 + L;       // new_self_v
    const int OUT_NCK = 1 + 2 * L;   // new_cross_k
    const int OUT_NCV = 1 + 3 * L;   // new_cross_v

    int in_num  = decoder_info_->input_num;   // 4 + 4*L
    int out_num = decoder_info_->output_num;  // 1 + 4*L

    // 工作 buffer（KV 全部常驻 device，host 侧只保留 pos/mask/logits 小 buffer）
    std::vector<float> pos_emb(n_state_);
    std::vector<float> self_attn_mask(padding_size_ + 1);
    std::vector<float> logits(vocab_num_);

    // 分配 device tensors（一次性，循环复用）
    std::vector<bm_tensor_t> in_ts(in_num), out_ts(out_num);
    for (int i = 0; i < in_num;  i++)
        bmrt_tensor(&in_ts[i],  decoder_rt_, BM_FLOAT32,
                    decoder_info_->stages[0].input_shapes[i]);
    for (int i = 0; i < out_num; i++)
        bmrt_tensor(&out_ts[i], decoder_rt_, BM_FLOAT32,
                    decoder_info_->stages[0].output_shapes[i]);
    // token 输入：ONNX 是 int64，BM1684 编译后降为 INT32，用 BM_INT32 传
    bm_free_device(bm_handle_, in_ts[0].device_mem);
    bmrt_tensor(&in_ts[0], decoder_rt_, BM_INT32,
                decoder_info_->stages[0].input_shapes[0]);

    // KV cache 常驻 device：past_self + cross 段全程留 device（in_ts[4 .. 4+4L-1]）。
    // 首步前清零这些 input buffer（原靠每步 host 上传清零的 host buffer 保证，现直接清 device）。
    {
        int z = 0;
        for (int i = IN_PSK; i < in_num; i++)
            bm_memset_device_ext(bm_handle_, &z, 1, in_ts[i].device_mem);
    }

    // audio_features [1, n_audio_ctx, n_state] 编码器算出后全程不变，只需上传一次
    // （避免每步重复上传：turbo 7.7MB/步 × 几十步 ≈ 上百 MB 的无谓 s2d 拷贝）。
    bm_memcpy_s2d(bm_handle_, in_ts[1].device_mem, (void*)audio_features.data());

    auto run_one_step = [&](int token_id) -> bool {
        // --- 准备输入 ---
        // 0. token [1,1]，BM1684 bmodel 编译后为 INT32
        int32_t tok = (int32_t)token_id;
        bm_memcpy_s2d(bm_handle_, in_ts[0].device_mem, &tok);

        // 1. audio_features 已在循环外上传一次（不变），此处不再重复

        // 2. pos_emb [1, 1, n_state]
        get_position_embedding(cache_len_, pos_emb.data());
        bm_memcpy_s2d(bm_handle_, in_ts[2].device_mem, pos_emb.data());

        // 3. self_attn_mask [1, 1, 1, padding_size+1]
        create_self_attn_mask(cache_len_, self_attn_mask.data());
        bm_memcpy_s2d(bm_handle_, in_ts[3].device_mem, self_attn_mask.data());

        // 4.. past_self_k/v + cross_k/v：全部常驻 device，不再每步上传。
        //   - past_self KV：上一步输出 new_self 已 d2d 写入这些 input buffer 的 cache_len 位置
        //   - cross KV：首步推理后一次性 d2d 写入，之后内容固定不变
        //   首步（cache_len_==0）时这些 input buffer 已被首步前的 device memset 清零

        // --- 推理 ---
        if (!bm_infer(decoder_rt_, decoder_info_, in_ts, out_ts)) return false;

        // --- 处理输出 ---
        // 0. logits [1, 1, vocab_num]：下载到 host 做 greedy argmax（小数据，无法避免）
        bm_memcpy_d2s(bm_handle_, logits.data(), out_ts[0].device_mem);

        // self KV：output new_self d2d 写入 input past_self 的 cache_len_ 偏移位置。
        //   device 内传输，不经过 host。past_self input 与 new_self output 是不同
        //   device tensor，读写无冲突（同 Eureka 模式）。
        // 防御：past_self buffer 每层容量 padding_size_ 个 token，cache_len_ 必须 < padding_size_
        if (cache_len_ >= padding_size_) {
            LOGE("cache_len " << cache_len_ << " >= padding_size " << padding_size_
                 << "，KV cache 已满"); return false;
        }
        const size_t self_step_bytes = (size_t)n_state_ * sizeof(float);  // 单层单 token KV
        const size_t self_off        = (size_t)cache_len_ * n_state_ * sizeof(float);
        for (int l = 0; l < L; l++) {
            // new_self_k_l (out[OUT_NSK+l]) → past_self_k_l (in[IN_PSK+l]) @ cache_len_
            bm_memcpy_d2d_byte(bm_handle_, in_ts[IN_PSK + l].device_mem, self_off,
                               out_ts[OUT_NSK + l].device_mem, 0, self_step_bytes);
            // new_self_v_l (out[OUT_NSV+l]) → past_self_v_l (in[IN_PSV+l]) @ cache_len_
            bm_memcpy_d2d_byte(bm_handle_, in_ts[IN_PSV + l].device_mem, self_off,
                               out_ts[OUT_NSV + l].device_mem, 0, self_step_bytes);
        }

        // cross KV：只首步 d2d。output new_cross → input cross 整块复制，之后固定不再触碰。
        if (!cross_cache_initialized_) {
            const size_t cross_bytes = (size_t)n_audio_ctx_ * n_state_ * sizeof(float);
            for (int l = 0; l < L; l++) {
                bm_memcpy_d2d_byte(bm_handle_, in_ts[IN_CK + l].device_mem, 0,
                                   out_ts[OUT_NCK + l].device_mem, 0, cross_bytes);
                bm_memcpy_d2d_byte(bm_handle_, in_ts[IN_CV + l].device_mem, 0,
                                   out_ts[OUT_NCV + l].device_mem, 0, cross_bytes);
            }
            cross_cache_initialized_ = true;
        }
        cache_len_++;
        return true;
    };

    // Phase 1：处理 SOT 序列
    LOG("Phase 1: processing " << init_tokens.size() << " initial tokens...");
    for (int tok : init_tokens) {
        if (!run_one_step(tok)) { LOGE("decoder step failed"); goto cleanup; }
    }

    // Phase 2：自回归生成
    LOG("Phase 2: generating tokens (max " << (padding_size_ - cache_len_) << ")...");
    {
        int max_iter = padding_size_ - cache_len_;
        for (int iter = 0; iter < max_iter; iter++) {
            // Greedy：取前 WHISPER_EOT+1 个 token 中 logit 最大的
            int next_tok = 0;
            float max_val = -1e30f;
            for (int i = 0; i <= WHISPER_EOT; i++) {
                if (logits[i] > max_val) { max_val = logits[i]; next_tok = i; }
            }

            if (next_tok == WHISPER_EOT) {
                LOG("EOT at iter " << iter);
                break;
            }

            tokens.push_back(next_tok);

            if (callback) {
                std::string piece = decode_single_token(next_tok);
                if (!piece.empty()) callback(piece);
            }

            if (!run_one_step(next_tok)) { LOGE("decoder step failed"); goto cleanup; }

            if (cache_len_ >= padding_size_) {
                LOG("Reached max cache length");
                break;
            }
        }
    }

cleanup:
    // 释放 device mem
    for (auto& t : in_ts)  bm_free_device(bm_handle_, t.device_mem);
    for (auto& t : out_ts) bm_free_device(bm_handle_, t.device_mem);

    LOG("Decoder done, " << tokens.size() << " tokens");
    return true;
}

// ============================================================
// 文本解码
// ============================================================

std::string WhisperInference::decode_single_token(int id) {
    if (id < 0 || id >= vocab_num_) return "";
    const std::string& tok = vocab_[id].token;
    if (tok.size() >= 2 && tok[0] == '<' && tok[1] == '|') return "";
    std::string s = base64_decode(tok);
    replace_substr(s, "\xc4\xa0", " ");  // BPE space Ġ → ASCII space
    return s;
}

std::string WhisperInference::decode_tokens(const std::vector<int>& tokens) {
    std::string text;
    for (int id : tokens) {
        if (id < 0 || id >= vocab_num_) continue;
        const std::string& tok = vocab_[id].token;
        if (tok.size() >= 2 && tok[0] == '<' && tok[1] == '|') continue;
        std::string s = base64_decode(tok);
        text += s;
    }
    replace_substr(text, "\xc4\xa0", " ");
    replace_substr(text, "\n", "");
    return text;
}

// ============================================================
// release
// ============================================================

void WhisperInference::release() {
    if (encoder_rt_) { bmrt_destroy(encoder_rt_); encoder_rt_ = nullptr; }
    if (decoder_rt_) { bmrt_destroy(decoder_rt_); decoder_rt_ = nullptr; }
    if (bm_handle_)  { bm_dev_free(bm_handle_);   bm_handle_  = nullptr; }
    initialized_ = false;
    LOG("Released");
}
