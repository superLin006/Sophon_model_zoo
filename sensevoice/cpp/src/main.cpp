#include <iostream>
#include <cstdlib>

#include "sensevoice_inference.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <audio.wav> [F32|F16]\n";
        std::cerr << "  model_dir : directory with bmodel + tokens.txt\n";
        std::cerr << "  audio.wav : 16kHz mono WAV\n";
        std::cerr << "  precision : F32 (default) or F16\n";
        return 1;
    }

    const char* model_dir = argv[1];
    const char* audio_file = argv[2];
    const char* precision  = (argc >= 4) ? argv[3] : "F32";

    sensevoice::SenseVoiceInference sv;
    if (sv.init(model_dir, precision) != 0) {
        std::cerr << "[Error] init failed\n";
        return 1;
    }

    sensevoice::RecognitionResult result = sv.run(audio_file);

    std::cout << "\n--- SenseVoice Result ---\n";
    std::cout << "Text     : " << result.text << "\n";
    std::cout << "Language : " << result.language << "\n";
    std::cout << "Emotion  : " << result.emotion << "\n";
    std::cout << "Event    : " << result.event << "\n";

    sv.release();
    return 0;
}
