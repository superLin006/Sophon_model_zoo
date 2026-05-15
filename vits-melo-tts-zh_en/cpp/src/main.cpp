#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "tts_inference.h"
#include "wav_writer.h"

static std::vector<int64_t> read_int64_bin(const char* path, int expected_len) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Error] Cannot open: " << path << "\n";
        return {};
    }
    f.seekg(0, std::ios::end);
    size_t file_size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);

    int n = (int)(file_size / sizeof(int64_t));
    if (n != expected_len) {
        std::cerr << "[Warn] " << path << ": expected " << expected_len
                  << " int64s, got " << n << "\n";
    }
    std::vector<int64_t> data(n);
    f.read((char*)data.data(), file_size);
    return data;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <tokens.bin> <tones.bin> <seq_len> <model_dir> <output.wav> [F32|F16]\n";
        std::cerr << "  tokens.bin : raw int64 binary (with blank, no padding)\n";
        std::cerr << "  tones.bin  : raw int64 binary (same length)\n";
        std::cerr << "  seq_len    : actual sequence length\n";
        std::cerr << "  model_dir  : dir with model_part1.onnx + bmodel\n";
        std::cerr << "  output.wav : output 16-bit PCM WAV @ 44100Hz\n";
        std::cerr << "  precision  : F32 (default) or F16\n";
        return 1;
    }

    const char* tokens_file = argv[1];
    const char* tones_file  = argv[2];
    int         seq_len     = std::atoi(argv[3]);
    const char* model_dir   = argv[4];
    const char* output_wav  = argv[5];
    const char* precision   = (argc >= 7) ? argv[6] : "F32";

    std::cout << "================================================================\n";
    std::cout << " VITS-MeloTTS BM1684X Inference\n";
    std::cout << "================================================================\n";
    std::cout << " tokens_file : " << tokens_file << "\n";
    std::cout << " tones_file  : " << tones_file  << "\n";
    std::cout << " seq_len     : " << seq_len     << "\n";
    std::cout << " model_dir   : " << model_dir   << "\n";
    std::cout << " output_wav  : " << output_wav  << "\n";
    std::cout << " precision   : " << precision   << "\n";
    std::cout << "----------------------------------------------------------------\n";

    if (seq_len <= 0 || seq_len > vits_tts::L_MAX) {
        std::cerr << "[Error] seq_len=" << seq_len << " out of range [1, " << vits_tts::L_MAX << "]\n";
        return 1;
    }

    // Read token/tone binary files
    std::vector<int64_t> tokens = read_int64_bin(tokens_file, seq_len);
    std::vector<int64_t> tones  = read_int64_bin(tones_file,  seq_len);
    if (tokens.empty() || tones.empty()) return 1;
    if ((int)tokens.size() < seq_len || (int)tones.size() < seq_len) {
        std::cerr << "[Error] File too short for seq_len=" << seq_len << "\n";
        return 1;
    }

    // Init inference
    vits_tts::TTSInference tts;
    if (tts.init(model_dir, precision) != 0) {
        std::cerr << "[Error] TTS init failed\n";
        return 1;
    }

    // Run inference
    vits_tts::TTSResult result = tts.run(tokens.data(), tones.data(), seq_len);
    tts.release();

    if (result.n_samples == 0) {
        std::cerr << "[Error] Inference returned 0 samples\n";
        return 1;
    }

    // Write WAV
    write_wav(output_wav, result.audio.data(), result.n_samples, vits_tts::SAMPLE_RATE);

    std::cout << "\n=== DONE ===\n";
    std::cout << "Output WAV : " << output_wav << "\n";
    std::cout << "Samples    : " << result.n_samples << " ("
              << (double)result.n_samples / vits_tts::SAMPLE_RATE << "s @ "
              << vits_tts::SAMPLE_RATE << "Hz)\n";
    std::cout << "PartA(enc+dp): " << result.part_a_ms << " ms\n";
    std::cout << "PartB(MAS)   : " << result.part_b_ms << " ms\n";
    std::cout << "PartC(flow+dec):" << result.part_c_ms << " ms\n";
    std::cout << "Total        : " << result.total_ms << " ms\n";
    std::cout << "RTF          : " << result.rtf << "\n";

    return 0;
}
