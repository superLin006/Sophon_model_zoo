#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>

inline void write_wav(const char* path, const float* data, int n_samples, int sample_rate = 44100) {
    int16_t* pcm = new int16_t[n_samples];
    for (int i = 0; i < n_samples; i++) {
        float v = data[i];
        if (v > 1.0f)  v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = (int16_t)(v * 32767.0f);
    }
    FILE* f = fopen(path, "wb");
    if (!f) {
        delete[] pcm;
        return;
    }
    // WAV header
    int data_bytes = n_samples * 2;
    int file_size  = 36 + data_bytes;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    int32_t chunk_size = 16;      fwrite(&chunk_size,  4, 1, f);
    int16_t audio_fmt  = 1;       fwrite(&audio_fmt,   2, 1, f);
    int16_t channels   = 1;       fwrite(&channels,    2, 1, f);
    fwrite(&sample_rate,           4, 1, f);
    int32_t byte_rate  = sample_rate * 2; fwrite(&byte_rate,   4, 1, f);
    int16_t block_align = 2;      fwrite(&block_align, 2, 1, f);
    int16_t bits       = 16;      fwrite(&bits,        2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_bytes, 4, 1, f);
    fwrite(pcm, 2, n_samples, f);
    fclose(f);
    delete[] pcm;
}
