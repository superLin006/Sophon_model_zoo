#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// Wraps decoder_1-768-1024_bm1684x.bmodel
// Input:  hidden states [1, 768, T], float32 (converted from float16 inside)
// Output: mel spectrogram [1, 100, T*2], float32
class DecoderEngine {
public:
    DecoderEngine(const std::string& bmodel_path, int tpu_id);
    // Shared-handle ctor: loads bmodel into an existing bmrt (does not own handle/bmrt)
    DecoderEngine(const std::string& bmodel_path, void* bm_handle, void* bmrt);
    ~DecoderEngine();

    // hiddens_f16: flattened [hidden_size * T] float16 values (row-major: T vectors of hidden_size)
    // hidden_size: 768, T: number of generated steps
    // Returns mel [100 * T*2] float32, or empty on error
    std::vector<float> infer(const std::vector<uint16_t>& hiddens_f16,
                             int hidden_size,
                             int T);

    int input_T() const;  // max T the bmodel accepts

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
