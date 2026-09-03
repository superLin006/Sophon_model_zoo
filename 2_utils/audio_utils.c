#include <limits.h>
#include <math.h>
#include <sndfile.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "audio_utils.h"

int read_audio(const char *path, audio_buffer_t *audio)
{
    if (!path || !audio)
    {
        return -1;
    }

    SNDFILE *infile;
    SF_INFO sfinfo = {0};
    infile = sf_open(path, SFM_READ, &sfinfo);
    if (!infile)
    {
        fprintf(stderr, "Error: failed to open file '%s': %s\n", path, sf_strerror(NULL));
        return -1;
    }

    if (sfinfo.frames < 0 || sfinfo.frames > INT_MAX ||
        sfinfo.channels <= 0 || sfinfo.channels > INT_MAX ||
        (size_t)sfinfo.frames > SIZE_MAX / (size_t)sfinfo.channels / sizeof(float))
    {
        fprintf(stderr, "Error: audio dimensions are not supported.\n");
        sf_close(infile);
        return -1;
    }

    audio->num_frames = (int)sfinfo.frames;
    audio->num_channels = sfinfo.channels;
    audio->sample_rate = sfinfo.samplerate;
    size_t sample_count = (size_t)audio->num_frames * (size_t)audio->num_channels;
    audio->data = (float *)malloc(sample_count * sizeof(float));
    if (!audio->data)
    {
        fprintf(stderr, "Error: failed to allocate memory.\n");
        sf_close(infile);
        return -1;
    }

    sf_count_t num_read_frames = sf_readf_float(infile, audio->data, audio->num_frames);
    if (num_read_frames != audio->num_frames)
    {
        fprintf(stderr, "Error: failed to read all frames. Expected %d, got %ld.\n",
                audio->num_frames, (long)num_read_frames);
        free(audio->data);
        audio->data = NULL;
        sf_close(infile);
        return -1;
    }

    sf_close(infile);
    return 0;
}

int save_audio(const char *path, float *data, int num_frames, int sample_rate, int num_channels)
{
    if (!path || !data || num_frames <= 0 || sample_rate <= 0 || num_channels <= 0)
    {
        return -1;
    }

    SNDFILE *outfile;
    SF_INFO sfinfo = {0};
    sfinfo.frames = num_frames;
    sfinfo.samplerate = sample_rate;
    sfinfo.channels = num_channels;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    outfile = sf_open(path, SFM_WRITE, &sfinfo);
    if (!outfile)
    {
        fprintf(stderr, "Error: failed to open file '%s' for writing: %s\n", path, sf_strerror(NULL));
        return -1;
    }

    sf_count_t num_written_frames = sf_writef_float(outfile, data, num_frames);
    if (num_written_frames != num_frames)
    {
        fprintf(stderr, "Error: failed to write all frames. Expected %d, wrote %ld.\n",
                num_frames, (long)num_written_frames);
        sf_close(outfile);
        return -1;
    }

    sf_close(outfile);
    return 0;
}

int resample_audio(audio_buffer_t *audio, int original_sample_rate, int desired_sample_rate)
{
    if (!audio || !audio->data || audio->num_frames <= 0 || audio->num_channels <= 0 ||
        original_sample_rate <= 0 || desired_sample_rate <= 0)
    {
        return -1;
    }

    int original_length = audio->num_frames;
    int channels = audio->num_channels;
    long long out_length_ll = llround((double)original_length * desired_sample_rate /
                                      original_sample_rate);
    if (out_length_ll <= 0 || out_length_ll > INT_MAX ||
        (size_t)out_length_ll > SIZE_MAX / (size_t)channels / sizeof(float))
    {
        return -1;
    }

    int out_length = (int)out_length_ll;
    printf("resample_audio: %d HZ -> %d HZ\n", original_sample_rate, desired_sample_rate);

    size_t out_sample_count = (size_t)out_length * (size_t)channels;
    float *resampled_data = (float *)malloc(out_sample_count * sizeof(float));
    if (!resampled_data)
    {
        return -1;
    }

    for (int i = 0; i < out_length; ++i)
    {
        double src_index = i * (double)original_sample_rate / desired_sample_rate;
        int left_index = (int)floor(src_index);
        if (left_index >= original_length)
        {
            left_index = original_length - 1;
        }
        int right_index = (left_index + 1 < original_length) ? left_index + 1 : left_index;
        double fraction = src_index - left_index;
        for (int channel = 0; channel < channels; ++channel)
        {
            size_t left = (size_t)left_index * (size_t)channels + (size_t)channel;
            size_t right = (size_t)right_index * (size_t)channels + (size_t)channel;
            size_t output = (size_t)i * (size_t)channels + (size_t)channel;
            resampled_data[output] = (float)((1.0 - fraction) * audio->data[left] +
                                             fraction * audio->data[right]);
        }
    }

    audio->num_frames = out_length;
    audio->sample_rate = desired_sample_rate;
    free(audio->data);
    audio->data = resampled_data;
    return 0;
}

int convert_channels(audio_buffer_t *audio)
{
    if (!audio || !audio->data || audio->num_frames <= 0 || audio->num_channels <= 0)
    {
        return -1;
    }
    if (audio->num_channels == 1)
    {
        return 0;
    }

    int original_num_channels = audio->num_channels;
    printf("convert_channels: %d -> 1\n", original_num_channels);

    float *converted_data = (float *)malloc((size_t)audio->num_frames * sizeof(float));
    if (!converted_data)
    {
        return -1;
    }

    for (int frame = 0; frame < audio->num_frames; ++frame)
    {
        double sum = 0.0;
        for (int channel = 0; channel < original_num_channels; ++channel)
        {
            sum += audio->data[(size_t)frame * (size_t)original_num_channels + (size_t)channel];
        }
        converted_data[frame] = (float)(sum / original_num_channels);
    }

    audio->num_channels = 1;
    free(audio->data);
    audio->data = converted_data;
    return 0;
}
