#pragma once
#include <string>
#include <vector>

struct WavData {
    int sample_rate = 0;      // 原始采样率
    int num_channels = 0;     // 原始声道数
    std::vector<float> samples;  // mono float32 [-1,1]
};

// 读取 PCM16(PCM) 或 IEEE float32 格式的 WAV,多声道平均为 mono。
// 若采样率 != 16000,线性插值重采样到 16kHz(torchaudio resample 的近似,
// 8k.wav 为参考测试,0.wav 原生 16k 不受影响)。
// 返回 false 时 error 给出原因。
bool ReadWavResample16k(const std::string& path, WavData* wav,
                        std::string* error = nullptr);
