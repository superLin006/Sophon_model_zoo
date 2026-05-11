/* Tokenizer for SenseVoice
 *
 * Handles token ID to text conversion and CTC decoding.
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "sensevoice_config.h"

namespace sensevoice {

// Recognition result
struct RecognitionResult {
    std::string text;                      // Final transcription text
    std::vector<std::string> tokens;       // Individual tokens
    std::vector<float> timestamps;         // Timestamp for each token (seconds)

    // Metadata (extracted from first 4 frames)
    std::string language;
    std::string emotion;
    std::string event;
};

// CTC decoder result (internal)
struct CTCDecoderResult {
    std::vector<int64_t> token_ids;
    std::vector<int32_t> frame_indices;
};

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Load tokens from file (format: "token id" per line)
    bool Load(const std::string& tokens_file);

    // Get token string by ID
    std::string IdToToken(int64_t id) const;

    // Get token ID by string
    int64_t TokenToId(const std::string& token) const;

    // Get vocabulary size
    int32_t VocabSize() const { return static_cast<int32_t>(id_to_token_.size()); }

    // Check if token is blank
    bool IsBlank(int64_t id) const { return id == blank_id_; }

    // CTC greedy search decoding
    // Input: logits [num_frames, vocab_size]
    // Output: decoded token IDs and frame indices
    CTCDecoderResult CTCGreedySearch(const float* logits,
                                     int32_t num_frames,
                                     int32_t vocab_size) const;

    // Convert CTC result to recognition result
    RecognitionResult ConvertResult(const CTCDecoderResult& ctc_result,
                                    int32_t frame_shift_ms = 10,
                                    int32_t lfr_window_shift = 6) const;

    // Full decode pipeline: logits -> RecognitionResult
    RecognitionResult Decode(const float* logits,
                             int32_t num_frames,
                             int32_t vocab_size,
                             int32_t frame_shift_ms = 10,
                             int32_t lfr_window_shift = 6) const;

private:
    std::unordered_map<int64_t, std::string> id_to_token_;
    std::unordered_map<std::string, int64_t> token_to_id_;
    int64_t blank_id_ = 0;

    // Special token IDs for SenseVoice metadata
    static constexpr int32_t kNumMetadataFrames = 4;  // language, emotion, event, text_norm
};

}  // namespace sensevoice
