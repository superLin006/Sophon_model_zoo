#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <vector>
#include <string>
#include <cstdint>

// 与模型无关的固定常量。n_mels / vocab 等随模型变化的维度由调用方运行时传入。
#define N_FFT           400
#define HOP_LENGTH      160
#define MELS_FILTERS_SIZE 201
#define MAX_AUDIO_LENGTH 480000   // 30s at 16kHz

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

void audio_preprocess(audio_buffer_t* audio, float* mel_filters, int n_mels,
                      std::vector<float>& mel_spec);

int  read_mel_filters(const char* filename, float* data, int max_lines);
int  read_vocab(const char* filename, VocabEntry* vocab, int vocab_num);

void        replace_substr(std::string& str, const std::string& from,
                           const std::string& to);
std::string base64_decode(const std::string& encoded_string);

#endif // AUDIO_UTILS_H
