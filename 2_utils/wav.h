#pragma once
#include <string>
#include <vector>

struct Wav {
    int sample_rate = 0;
    int num_channels = 0;
    std::vector<float> samples;
};

// Reads 16kHz PCM16 WAV (mono or multi-channel).
// Multi-channel is averaged to mono. Non-16kHz or non-PCM16 is rejected.
bool ReadWav16(const std::string& path, Wav* wav, std::string* error = nullptr);
