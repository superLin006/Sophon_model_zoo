#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <vector>
#include <string>
#include <cstdint>

// whisper-base 参数（与 MTK large-v3-turbo 的区别：N_MELS=80, VOCAB_NUM=51865）
#define N_MELS           80
#define N_FFT           400
#define HOP_LENGTH      160
#define MELS_FILTERS_SIZE 201
#define MAX_AUDIO_LENGTH 480000   // 30s at 16kHz
#define MAX_TOKENS       448
#define VOCAB_NUM        51865

typedef struct {
    float* data;
    int    num_frames;
    int    sample_rate;
} audio_buffer_t;

typedef struct {
    std::string token;
    int         index;
} VocabEntry;

int  load_audio(const char* filename, audio_buffer_t* audio);
void free_audio(audio_buffer_t* audio);

void audio_preprocess(audio_buffer_t* audio, float* mel_filters,
                      std::vector<float>& mel_spec);

int  read_mel_filters(const char* filename, float* data, int max_lines);
int  read_vocab(const char* filename, VocabEntry* vocab);

void        replace_substr(std::string& str, const std::string& from,
                           const std::string& to);
std::string base64_decode(const std::string& encoded_string);
int         argmax(float* array);

#endif // AUDIO_UTILS_H
