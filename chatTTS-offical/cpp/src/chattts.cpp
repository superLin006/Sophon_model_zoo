#include "chattts.h"
#include "normalizer.h"
#include "tokenizer.h"
#include "gpt_engine.h"
#include "decoder_engine.h"
#include "vocos_engine.h"
#include "istft.h"

#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <algorithm>

// ── Half-float helper ────────────────────────────────────────────────────────

static uint16_t f32_to_f16(float v) {
    uint32_t bits; std::memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp >= 16)  return (uint16_t)((sign << 15) | 0x7C00);
    if (exp < -24)  return (uint16_t)(sign << 15);
    if (exp < -14) { mant = (mant | 0x800000) >> (-14 - exp); return (uint16_t)((sign << 15) | (mant >> 13)); }
    return (uint16_t)((sign << 15) | ((uint32_t)(exp + 15) << 10) | (mant >> 13));
}

// ── ChatTTS::Impl ────────────────────────────────────────────────────────────

struct ChatTTS::Impl {
    ChatTTSConfig cfg;

    std::unique_ptr<Normalizer>     normalizer;
    std::unique_ptr<BertTokenizer>  tokenizer;
    std::unique_ptr<GPTEngine>      gpt;
    std::unique_ptr<DecoderEngine>  decoder;
    std::unique_ptr<VocosEngine>    vocos;
    std::unique_ptr<ISTFT>          istft;

    std::vector<uint16_t> spk_emb;   // float16 [hidden_size=768]
    int spk_emb_token_id = -1;       // id of "[spk_emb]" in vocab
    int stts_token_id    = -1;       // id of "[Stts]"
    int ptts_token_id    = -1;       // id of "[Ptts]"

    // Build the decorated prompt string, same as Python speaker.decorate_code_prompts
    std::string decorate(const std::string& text, int speed) const {
        std::string speed_tag = "[speed_" + std::to_string(speed) + "]";
        // "[Stts][spk_emb]{speed_tag}{text}[Ptts]"
        return "[Stts][spk_emb]" + speed_tag + text + "[Ptts]";
    }
};

// ── Constructor ──────────────────────────────────────────────────────────────

ChatTTS::ChatTTS(const ChatTTSConfig& cfg) : impl_(std::make_unique<Impl>()) {
    impl_->cfg = cfg;

    impl_->normalizer = std::make_unique<Normalizer>(cfg.homophones_map_path);
    fprintf(stderr, "[ChatTTS] normalizer OK\n");
    impl_->tokenizer  = std::make_unique<BertTokenizer>(cfg.vocab_path);
    fprintf(stderr, "[ChatTTS] tokenizer OK\n");

    GPTConfig gpt_cfg;  // defaults match ChatTTS architecture
    impl_->gpt     = std::make_unique<GPTEngine>(cfg.gpt_model_path, cfg.tpu_id, gpt_cfg);
    fprintf(stderr, "[ChatTTS] GPT OK\n");
    impl_->decoder = std::make_unique<DecoderEngine>(cfg.decoder_model_path, cfg.tpu_id);
    fprintf(stderr, "[ChatTTS] decoder OK\n");
    impl_->vocos   = std::make_unique<VocosEngine>(cfg.vocos_model_path, cfg.tpu_id);
    fprintf(stderr, "[ChatTTS] vocos OK\n");
    impl_->istft   = std::make_unique<ISTFT>(1024, 256, 1024);
    fprintf(stderr, "[ChatTTS] ISTFT OK\n");

    impl_->spk_emb_token_id = impl_->tokenizer->token_to_id("[spk_emb]");
    fprintf(stderr, "[ChatTTS] ctor done, spk_emb_token_id=%d\n", impl_->spk_emb_token_id);
}

ChatTTS::~ChatTTS() = default;

// ── Speaker loading ──────────────────────────────────────────────────────────

bool ChatTTS::load_speaker(const std::string& spk_emb_path) {
    std::ifstream f(spk_emb_path, std::ios::binary);
    if (!f.is_open()) return false;
    // File format: raw float32 array of hidden_size (768) values
    std::vector<float> buf(768);
    f.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
    if (!f) return false;
    set_speaker(buf);
    return true;
}

void ChatTTS::set_speaker(const std::vector<float>& spk_emb_f32) {
    // L2-normalize the speaker embedding before storing (matches Python: F.normalize(spk, p=2, dim=0))
    double norm2 = 0.0;
    for (float v : spk_emb_f32) norm2 += (double)v * v;
    float scale = (norm2 > 1e-24) ? (float)(1.0 / std::sqrt(norm2)) : 1.0f;
    impl_->spk_emb.resize(spk_emb_f32.size());
    for (size_t i = 0; i < spk_emb_f32.size(); ++i)
        impl_->spk_emb[i] = f32_to_f16(spk_emb_f32[i] * scale);
}

// ── Infer ────────────────────────────────────────────────────────────────────

std::vector<float> ChatTTS::infer(const std::string& text,
                                   const InferParams& params,
                                   bool do_normalize) {
    // 1. Text normalization
    std::string norm_text = do_normalize ? impl_->normalizer->normalize(text) : text;

    // 2. Decorate with speaker and speed prompts
    std::string decorated = impl_->decorate(norm_text, params.speed);

    // 3. Tokenize
    std::vector<int> input_ids = impl_->tokenizer->encode(decorated);
    if (input_ids.empty())
        throw std::runtime_error("ChatTTS::infer: tokenization produced empty ids");

    // Find position of [spk_emb] token for speaker injection
    int spk_idx = -1;
    for (int i = 0; i < (int)input_ids.size(); ++i) {
        if (input_ids[i] == impl_->spk_emb_token_id) { spk_idx = i; break; }
    }

    fprintf(stderr, "[TTS] text tokens: %d, spk_idx=%d\n", (int)input_ids.size(), spk_idx);

    // 4. GPT generate
    std::vector<float> temps(4, params.temperature);
    GPTResult gpt_out = impl_->gpt->generate(
        input_ids,
        impl_->spk_emb,
        spk_idx,
        temps,
        params.top_p,
        params.top_k,
        params.repetition_penalty,
        params.max_new_token,
        params.min_new_token
    );

    if (gpt_out.hiddens.empty())
        return {};

    int T           = (int)gpt_out.hiddens.size();
    int hidden_size = 768;

    // 5. Flatten hiddens: [T, hidden_size] f16 → pass to decoder
    std::vector<uint16_t> hiddens_flat(T * hidden_size);
    for (int t = 0; t < T; ++t)
        std::copy(gpt_out.hiddens[t].begin(), gpt_out.hiddens[t].end(),
                  hiddens_flat.begin() + t * hidden_size);

    // 6. Decoder: hiddens → mel [100, T*2]
    std::vector<float> mel = impl_->decoder->infer(hiddens_flat, hidden_size, T);
    if (mel.empty()) return {};

    int mel_T = impl_->decoder->input_T() * 2;  // decoder output time steps

    // 7. Vocos: mel → mag/x/y
    VocosOutput voc = impl_->vocos->infer(mel, 100, mel_T);
    if (voc.mag.empty()) return {};

    // 8. ISTFT: complex spectrogram → PCM
    std::vector<float> audio = impl_->istft->forward(voc.mag, voc.x, voc.y, voc.T);

    // 9. Clip silence from tail (|x| > 1e-5)
    int keep = 0;
    for (int i = (int)audio.size() - 1; i >= 0; --i) {
        if (std::abs(audio[i]) > 1e-5f) { keep = i + 1; break; }
    }
    audio.resize(keep);

    return audio;
}

// ── WAV write ────────────────────────────────────────────────────────────────

bool ChatTTS::save_wav(const std::string& path,
                        const std::vector<float>& pcm,
                        int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    int n_samples  = (int)pcm.size();
    int n_channels = 1;
    int bits       = 16;
    int byte_rate  = sample_rate * n_channels * bits / 8;
    int block_align= n_channels * bits / 8;
    int data_size  = n_samples * block_align;

    auto write32 = [&](uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
    auto write16 = [&](uint16_t v) { f.write(reinterpret_cast<char*>(&v), 2); };

    f.write("RIFF", 4);
    write32(36 + data_size);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    write32(16);
    write16(1);                           // PCM
    write16((uint16_t)n_channels);
    write32((uint32_t)sample_rate);
    write32((uint32_t)byte_rate);
    write16((uint16_t)block_align);
    write16((uint16_t)bits);
    f.write("data", 4);
    write32((uint32_t)data_size);

    // Convert float32 → int16
    for (float s : pcm) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        auto  i16     = static_cast<int16_t>(std::lround(clamped * 32767.0f));
        f.write(reinterpret_cast<char*>(&i16), 2);
    }
    return true;
}
