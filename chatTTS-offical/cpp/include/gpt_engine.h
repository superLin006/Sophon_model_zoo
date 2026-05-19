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

// Output of a single prefill or decode step
struct GPTStepResult {
    std::vector<float>    logits;   // [num_audio_tokens * num_vq] f32
    std::vector<uint16_t> hidden;   // [hidden_size] f16
};

class GPTEngine {
public:
    GPTEngine(const std::string& bmodel_path, int tpu_id, const GPTConfig& cfg);
    ~GPTEngine();

    // ── Batch API (non-streaming) ────────────────────────────────────────────
    GPTResult generate(const std::vector<int>&       input_ids,
                       const std::vector<uint16_t>&  spk_emb,
                       int                           spk_emb_idx,
                       const std::vector<float>&     temperature,
                       float                         top_p,
                       int                           top_k,
                       float                         repetition_penalty,
                       int                           max_new_token = 2048,
                       int                           min_new_token = 0);

    // ── Step-by-step API (streaming) ─────────────────────────────────────────
    // Call prefill_step() once, then decode_step() in a loop.

    // Run prefill: embed + 20 blocks. Returns logits + last-token hidden.
    // Resets internal decode state (decode_step counter, text_tok_len).
    GPTStepResult prefill_step(const std::vector<int>&      input_ids,
                               const std::vector<uint16_t>& spk_emb,
                               int                          spk_emb_idx);

    // Run one decode step for the given vq_codes.
    // Returns logits + hidden for this step.
    GPTStepResult decode_step(const std::vector<int>& vq_codes);

    // Number of decode steps executed since last prefill_step()
    int current_decode_step() const;

    // Max sequence length the bmodel was compiled with
    int seqlen() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
