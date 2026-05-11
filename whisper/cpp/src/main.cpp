#include <iostream>
#include <string>
#include "whisper_inference.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_dir> <audio.wav> [language=zh|en]" << std::endl;
        std::cerr << "  model_dir: directory containing bmodel files, mel filters, vocab" << std::endl;
        return -1;
    }

    const char* model_dir = argv[1];
    const char* audio_file = argv[2];
    const char* language = (argc >= 4) ? argv[3] : "zh";

    WhisperInference whisper;

    if (whisper.init(model_dir) != 0) {
        std::cerr << "[ERROR] Init failed" << std::endl;
        return -1;
    }

    // 流式回调：每识别出一个词片段立即打印
    auto callback = [](const std::string& piece) {
        std::cout << piece << std::flush;
    };

    std::cout << "\n[Recognition Result]" << std::endl;
    std::string result = whisper.run(audio_file, language, callback);
    std::cout << std::endl;

    return 0;
}
