#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct TensorSpec {
    std::string name;
    std::string dtype;
    std::string runtime_dtype;
    int index = -1;
    std::vector<int64_t> shape;
    size_t bytes = 0;
};

struct NetworkSpec {
    std::string name;
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
};

struct Manifest {
    std::vector<NetworkSpec> networks;
    int segment = 103;
    int offset = 96;
    int sample_rate = 16000;
    int n_mels = 80;
    float tail_seconds = 1.03f;
    int blank = 0;
    int unk = 2;
    int context = 2;
    int vocab_size = 0;
};

// Parse a tensor_manifest.json file.  Returns false and sets *error on failure.
bool LoadManifest(const std::string& path, Manifest* manifest,
                  std::string* error = nullptr);
