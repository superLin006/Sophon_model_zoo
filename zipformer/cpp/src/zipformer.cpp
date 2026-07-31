#include "zipformer.h"
#include <algorithm>
#include <limits>

static const TensorSpec* find_tensor(const NetworkSpec& net,
                                     const char* name, bool is_output) {
    const auto& vec = is_output ? net.outputs : net.inputs;
    for (const auto& t : vec)
        if (t.name == name) return &t;
    return nullptr;
}

bool Zipformer::Init(const std::string& manifest_path, Runtime* runtime,
                     std::string* error) {
    if (!runtime) return false;
    if (!LoadManifest(manifest_path, &manifest_, error)) return false;
    if (!runtime->Validate(manifest_, error)) return false;

    // Locate encoder output shape for the greedy loop.
    const NetworkSpec* enc = nullptr;
    for (const auto& n : manifest_.networks)
        if (n.name == "encoder") enc = &n;
    const TensorSpec* eo = enc ? find_tensor(*enc, "encoder_out", true) : nullptr;
    if (!eo || eo->shape.size() != 3 ||
        eo->shape[0] != 1 || eo->shape[1] <= 0 || eo->shape[2] <= 0) {
        if (error) *error = "encoder_out must have rank 3 and batch 1";
        return false;
    }

    // Verify Joiner enc_out width matches Encoder output width.
    const NetworkSpec* joiner = nullptr;
    for (const auto& n : manifest_.networks)
        if (n.name == "joiner") joiner = &n;
    const TensorSpec* je = joiner ? find_tensor(*joiner, "enc_out", false) : nullptr;
    if (!je || je->shape.size() != 2 ||
        je->shape[0] != 1 || je->shape[1] != eo->shape[2]) {
        if (error) *error = "joiner enc_out width mismatch";
        return false;
    }

    if (!runtime->Reset(error)) return false;
    runtime_ = runtime;
    return true;
}

bool Zipformer::Greedy(const std::vector<std::vector<float>>& chunks,
                       std::vector<int>* token_ids, std::string* error) {
    if (!runtime_ || !token_ids) {
        if (error) *error = "runtime or output is null";
        return false;
    }
    token_ids->clear();
    if (!runtime_->Reset(error)) return false;

    // Look up encoder output width from the parsed manifest.
    const NetworkSpec* enc = nullptr;
    for (const auto& n : manifest_.networks)
        if (n.name == "encoder") enc = &n;
    const TensorSpec* eo = find_tensor(*enc, "encoder_out", true);
    size_t width = static_cast<size_t>(eo->shape[2]);

    // Initial decoder context: [blank, blank].
    int64_t context[2] = {manifest_.blank, manifest_.blank};
    std::vector<float> enc_out, dec_out, logits;

    // Run Decoder once with the initial blank context.
    if (!runtime_->Decoder(context, &dec_out, error)) return false;

    for (const auto& chunk : chunks) {
        // Encoder streaming chunk → [1, 24, width].
        if (!runtime_->Encoder(chunk, &enc_out, error)) return false;
        if (width == 0 || enc_out.size() % width != 0) {
            if (error) *error = "bad encoder output shape";
            return false;
        }

        size_t num_frames = enc_out.size() / width;
        for (size_t t = 0; t < num_frames; ++t) {
            // Join one encoder frame with current decoder state.
            std::vector<float> frame(enc_out.begin() + t * width,
                                     enc_out.begin() + (t + 1) * width);
            if (!runtime_->Joiner(frame, dec_out, &logits, error) ||
                logits.empty())
                return false;

            // Greedy argmax.
            size_t best = static_cast<size_t>(
                std::max_element(logits.begin(), logits.end()) -
                logits.begin());

            // Skip blank, unk, and out-of-range tokens.
            if (best == static_cast<size_t>(manifest_.blank) ||
                best == static_cast<size_t>(manifest_.unk) ||
                (manifest_.vocab_size > 0 &&
                 best >= static_cast<size_t>(manifest_.vocab_size)))
                continue;

            // Accept token, shift context, re-run Decoder.
            token_ids->push_back(static_cast<int>(best));
            context[0] = context[1];
            context[1] = static_cast<int64_t>(best);
            if (!runtime_->Decoder(context, &dec_out, error)) return false;
        }
    }
    return true;
}
