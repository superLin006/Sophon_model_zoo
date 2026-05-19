#pragma once
#include <string>
#include <vector>
#include <memory>

// Wraps vocos_1-100-2048_bm1684x.bmodel
// Input:  mel spectrogram [1, 100, T], float32
// Output: mag [1, 513, T] + x [1, 513, T] + y [1, 513, T], float32
struct VocosOutput {
    std::vector<float> mag;  // [513 * T]
    std::vector<float> x;    // [513 * T]
    std::vector<float> y;    // [513 * T]
    int T = 0;               // actual time frames
};

class VocosEngine {
public:
    VocosEngine(const std::string& bmodel_path, int tpu_id);
    // Shared-handle ctor: loads bmodel into an existing bmrt (does not own handle/bmrt)
    VocosEngine(const std::string& bmodel_path, void* bm_handle, void* bmrt);
    ~VocosEngine();

    VocosOutput infer(const std::vector<float>& mel, int n_mels, int T);

    int input_T() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
