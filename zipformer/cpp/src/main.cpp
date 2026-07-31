#include "manifest.h"
#include "wav.h"
#include "audio_frontend.h"
#include "tokenizer.h"
#include "zipformer.h"
#include "bmruntime_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: zipformer_cli manifest.json input.wav "
                     "encoder.bmodel decoder.bmodel joiner.bmodel tokens.txt\n";
        return 2;
    }

    std::string error;

    // 1. Load manifest.
    Manifest manifest;
    if (!LoadManifest(argv[1], &manifest, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    // 2. Read WAV.
    Wav wav;
    if (!ReadWav16(argv[2], &wav, &error)) {
        std::cerr << "WAV error: " << error << "\n";
        return 1;
    }
    if (wav.sample_rate != manifest.sample_rate) {
        std::cerr << "WAV sample rate " << wav.sample_rate
                  << " != " << manifest.sample_rate << "\n";
        return 1;
    }

    // 3. Compute fbank and split into streaming chunks.
    zipformer::FbankConfig fbank_cfg;
    fbank_cfg.sample_rate = manifest.sample_rate;
    fbank_cfg.mel_bins = manifest.n_mels;
    int num_frames = 0;
    std::vector<float> features =
        zipformer::ComputeStrictFbank(wav.samples, &num_frames, fbank_cfg);
    std::vector<std::vector<float>> chunks =
        zipformer::MakeFeatureChunks(features, num_frames, manifest.n_mels,
                                     manifest.segment, manifest.offset);
    std::cerr << "features shape=[" << num_frames << "," << manifest.n_mels
              << "] chunks=" << chunks.size() << "\n";

    // 4. Load bmodels.
    BmRuntime runtime;
    if (!runtime.Load(argv[3], argv[4], argv[5], &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    // 5. Run streaming greedy decode.
    Zipformer zipformer;
    std::vector<int> token_ids;
    auto started = std::chrono::steady_clock::now();
    if (!zipformer.Init(argv[1], &runtime, &error) ||
        !zipformer.Greedy(chunks, &token_ids, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started)
            .count();
    double audio_duration = wav.samples.empty()
                                ? 0.0
                                : static_cast<double>(wav.samples.size()) /
                                      wav.sample_rate;

    // 6. Decode tokens to text.
    Tokenizer tokenizer;
    if (!tokenizer.Load(argv[6], &error) ||
        !tokenizer.Covers(manifest.vocab_size, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    std::cout << "tokens:";
    for (int id : token_ids) std::cout << " " << id;
    std::cout << "\ntext: "
              << tokenizer.Decode(token_ids, manifest.blank, manifest.unk)
              << "\naudio_duration_s=" << audio_duration
              << "\nmodel_latency_ms=" << elapsed * 1000.0
              << " rtf=" << (audio_duration > 0 ? elapsed / audio_duration : 0)
              << "\n";
    return 0;
}
