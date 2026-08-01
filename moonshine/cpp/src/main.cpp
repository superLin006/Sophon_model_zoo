#include <cstdio>
#include <cstring>

#include "moonshine_inference.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <model_dir> <wav_path> [F32|F16] [debug_dir]\n"
                "  model_dir : 含 moonshine_encoder_<prec>.bmodel / "
                "moonshine_decoder_<prec>.bmodel / tokens.txt\n"
                "  debug_dir : 可选, 保存 x_frames/encoder_out/logits/tokens 供对比\n",
                argv[0]);
        return -1;
    }
    const char* model_dir = argv[1];
    const char* wav_path = argv[2];
    const char* precision = argc > 3 ? argv[3] : "F32";

    MoonshineInference ms;
    if (argc > 4) ms.set_debug_dir(argv[4]);
    if (ms.init(model_dir, precision) != 0) {
        fprintf(stderr, "[ERROR] init failed\n");
        return -1;
    }

    MoonshineStats stats;
    std::string text = ms.run(wav_path, &stats);
    if (text.empty()) {
        fprintf(stderr, "[ERROR] inference failed\n");
        return -1;
    }
    printf("\n[Text] %s\n", text.c_str());
    printf("[Timing] audio=%.1fms  feat=%.2fms  infer=%.2fms  total=%.2fms  "
           "RTF=%.4f  steps=%d\n",
           stats.audio_ms, stats.feat_ms, stats.infer_ms,
           stats.feat_ms + stats.infer_ms, stats.rtf, stats.steps);
    return 0;
}
