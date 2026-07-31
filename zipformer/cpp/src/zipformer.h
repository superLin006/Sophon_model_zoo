#pragma once
#include "manifest.h"
#include <cstdint>
#include <string>
#include <vector>

// Abstract runtime backend.  FakeRuntime is used in host tests;
// BmRuntime is the real BM1684X backend.
class Runtime {
public:
    virtual ~Runtime() = default;
    virtual bool Validate(const Manifest&, std::string* error) = 0;
    virtual bool Reset(std::string* error) = 0;
    virtual bool Encoder(const std::vector<float>& features,
                         std::vector<float>* encoder_out,
                         std::string* error) = 0;
    virtual bool Decoder(const int64_t token_ids[2],
                         std::vector<float>* decoder_out,
                         std::string* error) = 0;
    virtual bool Joiner(const std::vector<float>& enc_out,
                        const std::vector<float>& dec_out,
                        std::vector<float>* logits,
                        std::string* error) = 0;
};

// Test-only runtime: tracks call counts, returns synthetic logits.
struct FakeRuntime : public Runtime {
    int decoder_calls = 0;

    bool Validate(const Manifest&, std::string*) override { return true; }
    bool Reset(std::string*) override { return true; }
    bool Encoder(const std::vector<float>&, std::vector<float>* out,
                 std::string*) override {
        out->assign(24 * 256, 0.0f);
        return true;
    }
    bool Decoder(const int64_t[2], std::vector<float>* out,
                 std::string*) override {
        ++decoder_calls;
        out->assign(512, 0.0f);
        return true;
    }
    bool Joiner(const std::vector<float>&, const std::vector<float>&,
                std::vector<float>* out, std::string*) override {
        out->assign(6254, 0.0f);
        return true;
    }
};

// Streaming Transducer greedy decoder.
class Zipformer {
public:
    bool Init(const std::string& manifest_path, Runtime* runtime,
              std::string* error = nullptr);
    bool Greedy(const std::vector<std::vector<float>>& chunks,
                std::vector<int>* token_ids, std::string* error = nullptr);
    const Manifest& manifest() const { return manifest_; }

private:
    Manifest manifest_;
    Runtime* runtime_ = nullptr;
};
