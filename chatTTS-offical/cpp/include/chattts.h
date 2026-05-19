#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

struct ChatTTSConfig {
    // Model paths
    std::string gpt_model_path;
    std::string decoder_model_path;
    std::string vocos_model_path;
    // Asset paths
    std::string vocab_path;           // vocab.txt
    std::string homophones_map_path;  // homophones_map.json
    // Runtime
    int tpu_id    = 0;
    int sample_rate = 24000;
};

struct InferParams {
    float temperature        = 0.0001f;
    float top_p              = 0.7f;
    int   top_k              = 20;
    float repetition_penalty = 1.05f;
    int   max_new_token      = 2048;
    int   min_new_token      = 0;
    int   speed              = 5;  // [speed_N] tag, 1-9
};

struct StreamParams {
    // Number of GPT decode steps to accumulate before calling decoder+vocos+iSTFT.
    // Larger = lower overhead, higher first-chunk latency. Python default: 24.
    int stream_batch       = 24;
    // Skip the first N chunks (decoder output can be noisy at the very start).
    int pass_first_n_batches = 2;
};

class ChatTTS {
public:
    explicit ChatTTS(const ChatTTSConfig& cfg);
    ~ChatTTS();

    // Load speaker embedding from binary file (768 float16 values)
    // Returns false if file not found
    bool load_speaker(const std::string& spk_emb_path);

    // Set speaker embedding directly from float32 array (will be converted to f16)
    void set_speaker(const std::vector<float>& spk_emb_f32);

    // Run full TTS pipeline: text → PCM float32 (non-streaming)
    std::vector<float> infer(const std::string& text,
                             const InferParams& params = InferParams(),
                             bool do_normalize = true);

    // Streaming TTS: callback is invoked with each PCM chunk as it becomes ready.
    // chunk_callback(pcm_chunk) is called synchronously from the inference thread.
    // Returns total number of PCM samples generated.
    // do_normalize: apply homophones replacement
    int infer_stream(const std::string& text,
                     const InferParams&  params,
                     const StreamParams& sparams,
                     std::function<void(const std::vector<float>&)> chunk_callback,
                     bool do_normalize = true);

    // Write PCM to WAV file (16-bit, mono)
    static bool save_wav(const std::string& path,
                         const std::vector<float>& pcm,
                         int sample_rate = 24000);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
