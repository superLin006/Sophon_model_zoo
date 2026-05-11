/* Tokenizer Implementation
 *
 * Handles CTC decoding and token conversion.
 */

#include "tokenizer.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace sensevoice {

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

bool Tokenizer::Load(const std::string& tokens_file) {
    std::ifstream file(tokens_file);
    if (!file.is_open()) {
        return false;
    }

    id_to_token_.clear();
    token_to_id_.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Find the last space to split token and id
        size_t last_space = line.rfind(' ');
        if (last_space == std::string::npos) {
            // Try tab separator
            last_space = line.rfind('\t');
        }

        if (last_space == std::string::npos) {
            continue;  // Invalid line
        }

        std::string token = line.substr(0, last_space);
        std::string id_str = line.substr(last_space + 1);

        // Trim whitespace
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
            token.pop_back();
        }
        while (!id_str.empty() && (id_str.front() == ' ' || id_str.front() == '\t')) {
            id_str.erase(0, 1);
        }

        try {
            int64_t id = std::stoll(id_str);
            id_to_token_[id] = token;
            token_to_id_[token] = id;
        } catch (...) {
            continue;  // Skip invalid lines
        }
    }

    // Set blank_id (usually 0, but verify)
    if (token_to_id_.count("<blank>")) {
        blank_id_ = token_to_id_["<blank>"];
    } else if (token_to_id_.count("<blk>")) {
        blank_id_ = token_to_id_["<blk>"];
    } else {
        blank_id_ = 0;  // Default
    }

    return !id_to_token_.empty();
}

std::string Tokenizer::IdToToken(int64_t id) const {
    auto it = id_to_token_.find(id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    return "<unk>";
}

int64_t Tokenizer::TokenToId(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return -1;
}

CTCDecoderResult Tokenizer::CTCGreedySearch(const float* logits,
                                            int32_t num_frames,
                                            int32_t vocab_size) const {
    CTCDecoderResult result;

    int64_t prev_id = -1;

    for (int32_t t = 0; t < num_frames; ++t) {
        // Find argmax
        const float* frame_logits = logits + t * vocab_size;
        int64_t max_id = 0;
        float max_val = frame_logits[0];

        for (int32_t v = 1; v < vocab_size; ++v) {
            if (frame_logits[v] > max_val) {
                max_val = frame_logits[v];
                max_id = v;
            }
        }

        // Skip blank and consecutive duplicates
        if (max_id != blank_id_ && max_id != prev_id) {
            result.token_ids.push_back(max_id);
            result.frame_indices.push_back(t);
        }

        prev_id = max_id;
    }

    return result;
}

RecognitionResult Tokenizer::ConvertResult(const CTCDecoderResult& ctc_result,
                                           int32_t frame_shift_ms,
                                           int32_t lfr_window_shift) const {
    RecognitionResult result;

    if (ctc_result.token_ids.empty()) {
        return result;
    }

    // Extract metadata from first 4 tokens (if available)
    int32_t start_idx = 0;
    if (ctc_result.token_ids.size() >= 4) {
        // First 4 frames contain: language, emotion, event, text_norm
        result.language = IdToToken(ctc_result.token_ids[0]);
        result.emotion = IdToToken(ctc_result.token_ids[1]);
        result.event = IdToToken(ctc_result.token_ids[2]);
        // token_ids[3] is text_norm, we skip it
        start_idx = kNumMetadataFrames;
    }

    // Convert remaining tokens to text
    std::string text;
    float frame_shift_s = static_cast<float>(frame_shift_ms) / 1000.0f * lfr_window_shift;

    for (size_t i = start_idx; i < ctc_result.token_ids.size(); ++i) {
        std::string token = IdToToken(ctc_result.token_ids[i]);

        // Handle special tokens
        if (token.empty() || token[0] == '<') {
            continue;  // Skip special tokens like <unk>, <sos>, etc.
        }

        // SenseVoice uses SentencePiece-like encoding
        // Replace special unicode character for space
        std::string processed_token;
        for (size_t j = 0; j < token.size(); ++j) {
            // Check for SentencePiece space marker (U+2581)
            if (j + 2 < token.size() &&
                static_cast<unsigned char>(token[j]) == 0xE2 &&
                static_cast<unsigned char>(token[j+1]) == 0x96 &&
                static_cast<unsigned char>(token[j+2]) == 0x81) {
                processed_token += ' ';
                j += 2;
            } else {
                processed_token += token[j];
            }
        }

        text += processed_token;
        result.tokens.push_back(processed_token);

        // Calculate timestamp
        float timestamp = frame_shift_s * (ctc_result.frame_indices[i] - start_idx);
        result.timestamps.push_back(std::max(0.0f, timestamp));
    }

    // Trim leading/trailing whitespace
    size_t start = text.find_first_not_of(" \t\n\r");
    size_t end = text.find_last_not_of(" \t\n\r");
    if (start != std::string::npos && end != std::string::npos) {
        result.text = text.substr(start, end - start + 1);
    } else {
        result.text = text;
    }

    return result;
}

RecognitionResult Tokenizer::Decode(const float* logits,
                                    int32_t num_frames,
                                    int32_t vocab_size,
                                    int32_t frame_shift_ms,
                                    int32_t lfr_window_shift) const {
    CTCDecoderResult ctc_result = CTCGreedySearch(logits, num_frames, vocab_size);
    return ConvertResult(ctc_result, frame_shift_ms, lfr_window_shift);
}

}  // namespace sensevoice
