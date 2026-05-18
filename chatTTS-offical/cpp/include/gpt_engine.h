#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// bmruntime handle forward declared via opaque pointer (actual type in bmlib_runtime.h)

struct GPTConfig {
    int  num_layers       = 20;
    int  hidden_size      = 768;
    int  num_vq           = 4;
    int  num_audio_tokens = 626;   // EOS = 625
    int  num_text_tokens  = 21178;
    int  seq_len          = 1024;  // max sequence length bmodel compiled with
    int  atten_head       = 12;
    int  atten_dim        = 64;
};

struct GPTResult {
    // hiddens[i]: float16 hidden vector [hidden_size] for step i
    std::vector<std::vector<uint16_t>> hiddens;
    // codes[i]: 4 vq codes generated at step i
    std::vector<std::vector<int>>      codes;
};

class GPTEngine {
public:
    GPTEngine(const std::string& bmodel_path, int tpu_id, const GPTConfig& cfg);
    ~GPTEngine();

    // Generate audio codes from input_ids (text prompt token sequence)
    // spk_emb: float16 speaker embedding [hidden_size], or empty to skip
    // temperature: per-vq temperature (length == num_vq)
    // Returns token sequences + hidden states for decoder
    GPTResult generate(const std::vector<int>&       input_ids,
                       const std::vector<uint16_t>&  spk_emb,
                       int                           spk_emb_idx,
                       const std::vector<float>&     temperature,
                       float                         top_p,
                       int                           top_k,
                       float                         repetition_penalty,
                       int                           max_new_token = 2048,
                       int                           min_new_token = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
