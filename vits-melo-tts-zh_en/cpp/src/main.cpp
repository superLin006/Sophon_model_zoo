#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tts_inference.h"
#include "wav_writer.h"

static std::vector<int64_t> read_int64_bin(const char* path, int expected_len) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Error] Cannot open: " << path << "\n";
        return {};
    }
    f.seekg(0, std::ios::end);
    std::streamoff file_size = f.tellg();
    if (file_size < 0 || file_size % static_cast<std::streamoff>(sizeof(int64_t)) != 0) {
        std::cerr << "[Error] Invalid int64 binary size: " << path << "\n";
        return {};
    }
    f.seekg(0, std::ios::beg);

    size_t n = static_cast<size_t>(file_size / sizeof(int64_t));
    if (n > static_cast<size_t>(INT_MAX)) {
        std::cerr << "[Error] Input is too large: " << path << "\n";
        return {};
    }
    if (static_cast<int>(n) != expected_len) {
        std::cerr << "[Warn] " << path << ": expected " << expected_len
                  << " int64s, got " << n << "\n";
    }
    std::vector<int64_t> data(n);
    if (!f.read(reinterpret_cast<char*>(data.data()), file_size)) {
        std::cerr << "[Error] Failed to read: " << path << "\n";
        return {};
    }
    return data;
}

namespace {

struct ManifestEntry {
    std::string tokens_path;
    std::string tones_path;
    int seq_len = 0;
    std::string label;
};

std::string manifest_parent(const std::string& path) {
    const std::string::size_type slash = path.find_last_of("/\\");
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string resolve_manifest_path(const std::string& parent, const std::string& path) {
    if (path.empty() || path[0] == '/' || (path.size() > 1 && path[1] == ':')) {
        return path;
    }
    return parent + "/" + path;
}

bool load_manifest(const std::string& path, std::vector<ManifestEntry>* entries) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[Error] Cannot open stream manifest: " << path << "\n";
        return false;
    }

    const std::string parent = manifest_parent(path);
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::istringstream input(line);
        std::string tokens_path;
        std::string tones_path;
        int seq_len = 0;
        if (!(input >> tokens_path)) {
            continue;
        }
        if (tokens_path[0] == '#') {
            continue;
        }
        if (!(input >> tones_path >> seq_len) || seq_len <= 0 ||
            seq_len > vits_tts::L_MAX) {
            std::cerr << "[Error] Invalid manifest line " << line_number
                      << ": expected <tokens.bin> <tones.bin> <seq_len> [label]\n";
            return false;
        }
        std::string label;
        std::getline(input, label);
        const std::string::size_type first = label.find_first_not_of(" \t");
        if (first != std::string::npos) {
            label.erase(0, first);
        } else {
            label.clear();
        }
        entries->push_back({resolve_manifest_path(parent, tokens_path),
                            resolve_manifest_path(parent, tones_path), seq_len,
                            label});
    }
    if (entries->empty()) {
        std::cerr << "[Error] Stream manifest contains no segments\n";
        return false;
    }
    return true;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <tokens.bin> <tones.bin> <seq_len> <model_dir> <output.wav> [F32|F16]\n";
    std::cerr << "       " << prog
              << " --stream-manifest <manifest> --model-dir <dir>"
                 " (--wav-out <file> | --pcm-s16le-out <file|->)"
                 " [--precision F16|F32] [--window-stream]\n";
    std::cerr << "  --window-stream: run C2 by mel windows and emit in-utterance chunks\n";
    std::cerr << "  --allow-truncation: stream mode diagnostic compatibility switch\n";
    std::cerr << "  legacy: raw int64 token/tone input, one complete WAV\n";
    std::cerr << "  stream manifest: one segment per line:"
                 " <tokens.bin> <tones.bin> <seq_len> [label]\n";
}

int run_stream_cli(int argc, char* argv[]) {
    std::string manifest_path;
    std::string model_dir = "models";
    std::string wav_path;
    std::string pcm_path;
    std::string precision = "F16";
    bool strict_mel_limit = true;
    bool windowed = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--stream-manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--wav-out" && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (arg == "--pcm-s16le-out" && i + 1 < argc) {
            pcm_path = argv[++i];
        } else if (arg == "--precision" && i + 1 < argc) {
            precision = argv[++i];
        } else if (arg == "--strict-mel-limit") {
            strict_mel_limit = true;
        } else if (arg == "--window-stream") {
            windowed = true;
            strict_mel_limit = true;
        } else if (arg == "--allow-truncation") {
            strict_mel_limit = false;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (manifest_path.empty() || (wav_path.empty() == pcm_path.empty())) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<ManifestEntry> manifest;
    if (!load_manifest(manifest_path, &manifest)) {
        return 1;
    }

    std::vector<vits_tts::TTSInputSegment> segments;
    segments.reserve(manifest.size());
    for (const auto& entry : manifest) {
        std::vector<int64_t> tokens = read_int64_bin(entry.tokens_path.c_str(), entry.seq_len);
        std::vector<int64_t> tones = read_int64_bin(entry.tones_path.c_str(), entry.seq_len);
        if (tokens.size() != static_cast<size_t>(entry.seq_len) ||
            tones.size() != static_cast<size_t>(entry.seq_len)) {
            std::cerr << "[Error] Segment input size mismatch: " << entry.tokens_path
                      << " / " << entry.tones_path << "\n";
            return 1;
        }
        segments.push_back({std::move(tokens), std::move(tones), entry.label});
    }

    vits_tts::TTSInference tts;
    if (tts.init(model_dir.c_str(), precision.c_str()) != 0) {
        std::cerr << "[Error] TTS init failed\n";
        return 1;
    }

    StreamingWavWriter wav_writer;
    FILE* pcm_file = nullptr;
    if (!wav_path.empty()) {
        if (!wav_writer.open(wav_path.c_str(), vits_tts::SAMPLE_RATE)) {
            std::cerr << "[Error] Cannot open WAV output: " << wav_path << "\n";
            tts.release();
            return 1;
        }
    } else {
        pcm_file = pcm_path == "-" ? stdout : std::fopen(pcm_path.c_str(), "wb");
        if (!pcm_file) {
            std::cerr << "[Error] Cannot open PCM output: " << pcm_path << "\n";
            tts.release();
            return 1;
        }
    }

    if (windowed) {
        int64_t total_samples = 0;
        int completed_chunks = 0;
        std::string error;
        bool output_ok = true;
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& segment = segments[i];
            vits_tts::TTSWindowStreamOptions window_options;
            window_options.strict_mel_limit = strict_mel_limit;
            window_options.on_audio = [&](const float* samples, int n_samples,
                                           int chunk_index, int chunk_count,
                                           int64_t sample_offset) {
                const int64_t global_offset = total_samples + sample_offset;
                bool ok = wav_writer.is_open()
                              ? wav_writer.append(samples, n_samples)
                              : write_pcm_s16le(pcm_file, samples, n_samples);
                if (ok) {
                    std::cerr << "[Window] segment " << (i + 1) << "/"
                              << segments.size() << " chunk " << (chunk_index + 1)
                              << "/" << chunk_count << " samples=" << n_samples
                              << " offset=" << global_offset << "\n";
                }
                return ok;
            };
            vits_tts::TTSWindowStreamResult result = tts.run_windowed(
                segment.tokens.data(), segment.tones.data(),
                static_cast<int>(segment.tokens.size()), window_options);
            total_samples += result.n_samples;
            completed_chunks += result.completed_chunks;
            if (!result.success) {
                output_ok = false;
                error = result.error;
                break;
            }
        }
        if (wav_writer.is_open()) {
            output_ok = wav_writer.finalize() && output_ok;
        } else if (pcm_file != stdout) {
            output_ok = std::fclose(pcm_file) == 0 && output_ok;
        } else {
            output_ok = std::fflush(pcm_file) == 0 && output_ok;
        }
        tts.release();
        if (!output_ok) {
            std::cerr << "[Error] Window stream failed: " << error << "\n";
            return 1;
        }
        std::cerr << "[Window] complete chunks=" << completed_chunks
                  << " samples=" << total_samples << "\n";
        return 0;
    }

    vits_tts::TTSStreamOptions options;
    options.strict_mel_limit = strict_mel_limit;
    options.on_audio = [&](const float* samples, int n_samples, int segment_index,
                           int segment_count, int64_t sample_offset) {
        bool ok = wav_writer.is_open()
                      ? wav_writer.append(samples, n_samples)
                      : write_pcm_s16le(pcm_file, samples, n_samples);
        if (ok) {
            std::cerr << "[Stream] segment " << (segment_index + 1) << "/"
                      << segment_count << " samples=" << n_samples
                      << " offset=" << sample_offset << "\n";
        }
        return ok;
    };

    vits_tts::TTSStreamResult result = tts.run_segments(segments, options);
    bool output_ok = result.success;
    if (wav_writer.is_open()) {
        output_ok = wav_writer.finalize() && output_ok;
    } else if (pcm_file != stdout) {
        output_ok = std::fclose(pcm_file) == 0 && output_ok;
    } else {
        output_ok = std::fflush(pcm_file) == 0 && output_ok;
    }
    tts.release();

    if (!output_ok) {
        std::cerr << "[Error] Stream failed after " << result.completed_segments
                  << "/" << segments.size() << " segments: " << result.error << "\n";
        return 1;
    }
    std::cerr << "[Stream] complete segments=" << result.completed_segments
              << " samples=" << result.n_samples
              << " first_chunk_ms=" << result.first_chunk_ms
              << " total_ms=" << result.total_ms << "\n";
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--stream-manifest") == 0) {
        return run_stream_cli(argc, argv);
    }
    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    const char* tokens_file = argv[1];
    const char* tones_file  = argv[2];
    int         seq_len     = std::atoi(argv[3]);
    const char* model_dir   = argv[4];
    const char* output_wav  = argv[5];
    const char* precision   = (argc >= 7) ? argv[6] : "F32";

    if (seq_len <= 0 || seq_len > vits_tts::L_MAX) {
        std::cerr << "[Error] seq_len=" << seq_len << " out of range [1, "
                  << vits_tts::L_MAX << "]\n";
        return 1;
    }

    std::vector<int64_t> tokens = read_int64_bin(tokens_file, seq_len);
    std::vector<int64_t> tones  = read_int64_bin(tones_file, seq_len);
    if (tokens.empty() || tones.empty() ||
        tokens.size() < static_cast<size_t>(seq_len) ||
        tones.size() < static_cast<size_t>(seq_len)) {
        return 1;
    }

    vits_tts::TTSInference tts;
    if (tts.init(model_dir, precision) != 0) {
        std::cerr << "[Error] TTS init failed\n";
        return 1;
    }

    vits_tts::TTSResult result = tts.run(tokens.data(), tones.data(), seq_len);
    tts.release();

    if (result.n_samples == 0) {
        std::cerr << "[Error] Inference returned 0 samples\n";
        return 1;
    }
    if (!write_wav(output_wav, result.audio.data(), result.n_samples,
                   vits_tts::SAMPLE_RATE)) {
        std::cerr << "[Error] Failed to write WAV: " << output_wav << "\n";
        return 1;
    }

    std::cout << "\n=== DONE ===\n";
    std::cout << "Output WAV : " << output_wav << "\n";
    std::cout << "Samples    : " << result.n_samples << " ("
              << (double)result.n_samples / vits_tts::SAMPLE_RATE << "s @ "
              << vits_tts::SAMPLE_RATE << "Hz)\n";
    std::cout << "PartA(enc+dp)  : " << result.part_a_ms << " ms\n";
    std::cout << "PartB(MAS)     : " << result.part_b_ms << " ms\n";
    std::cout << "PartC(c1+c2)   : " << result.part_c_ms << " ms\n";
    std::cout << "Total        : " << result.total_ms << " ms\n";
    std::cout << "RTF          : " << result.rtf << "\n";
    return 0;
}
