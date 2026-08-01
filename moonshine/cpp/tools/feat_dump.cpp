// 宿主机特征提取验证工具(原生编译, 不依赖 bmruntime)
// 用法: feat_dump <wav_path> <out.bin>
// 输出: [2000,80] float32 x_frames 原始二进制, 与 python 生成的参考对比
#include <cstdio>
#include <string>
#include <vector>

#include "../src/utils/moonshine_features.h"
#include "../src/utils/wav_reader.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: feat_dump <wav> <out.bin>\n");
        return 1;
    }
    WavData wav;
    std::string err;
    if (!ReadWavResample16k(argv[1], &wav, &err)) {
        fprintf(stderr, "[ERROR] %s\n", err.c_str());
        return 1;
    }
    std::vector<float> xf;
    moonshine::compute_x_frames(wav.samples, xf);

    FILE* f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "[ERROR] cannot write %s\n", argv[2]); return 1; }
    fwrite(xf.data(), 4, xf.size(), f);
    fclose(f);

    printf("wrote %s: %zu floats = [1,%d,%d] f32 (audio %.3fs, sr=%d)\n",
           argv[2], xf.size(), moonshine::N_FRAMES, moonshine::FRAME_LEN,
           wav.samples.size() / 16000.0, wav.sample_rate);
    return 0;
}
