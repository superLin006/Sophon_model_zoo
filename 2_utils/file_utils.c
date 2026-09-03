#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "file_utils.h"

static int file_size(FILE *file, long *size)
{
    if (fseek(file, 0, SEEK_END) != 0)
    {
        return -1;
    }
    long value = ftell(file);
    if (value < 0 || value > INT_MAX || fseek(file, 0, SEEK_SET) != 0)
    {
        return -1;
    }
    *size = value;
    return 0;
}

unsigned char *load_model(const char *filename, int *model_size)
{
    if (!filename || !model_size)
    {
        return NULL;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s fail!\n", filename);
        return NULL;
    }

    long length;
    if (file_size(fp, &length) != 0 || length == 0)
    {
        fprintf(stderr, "invalid model size: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    unsigned char *model = (unsigned char *)malloc((size_t)length);
    if (!model)
    {
        fclose(fp);
        return NULL;
    }
    if (fread(model, 1, (size_t)length, fp) != (size_t)length)
    {
        fprintf(stderr, "fread %s fail!\n", filename);
        free(model);
        fclose(fp);
        return NULL;
    }

    *model_size = (int)length;
    fclose(fp);
    return model;
}

int read_data_from_file(const char *path, char **out_data)
{
    if (!path || !out_data)
    {
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s fail!\n", path);
        return -1;
    }

    long length;
    if (file_size(fp, &length) != 0)
    {
        fclose(fp);
        return -1;
    }

    char *data = (char *)malloc((size_t)length + 1);
    if (!data)
    {
        fclose(fp);
        return -1;
    }
    if (fread(data, 1, (size_t)length, fp) != (size_t)length)
    {
        fprintf(stderr, "fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    data[length] = '\0';
    *out_data = data;
    fclose(fp);
    return (int)length;
}

int write_data_to_file(const char *path, const char *data, unsigned int size)
{
    if (!path || (!data && size > 0))
    {
        return -1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp)
    {
        fprintf(stderr, "open error: %s\n", path);
        return -1;
    }

    size_t written = fwrite(data, 1, size, fp);
    int result = (written == size && fflush(fp) == 0) ? 0 : -1;
    fclose(fp);
    return result;
}

int count_lines(FILE *file)
{
    if (!file)
    {
        return -1;
    }

    rewind(file);
    int lines = 0;
    int last = '\n';
    int has_data = 0;
    int ch;
    while ((ch = fgetc(file)) != EOF)
    {
        has_data = 1;
        last = ch;
        if (ch == '\n')
        {
            if (lines == INT_MAX)
            {
                rewind(file);
                return -1;
            }
            ++lines;
        }
    }
    if (has_data && last != '\n')
    {
        if (lines == INT_MAX)
        {
            rewind(file);
            return -1;
        }
        ++lines;
    }
    rewind(file);
    return lines;
}

char **read_lines_from_file(const char *filename, int *line_count)
{
    if (!filename || !line_count)
    {
        return NULL;
    }

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Failed to open the file: %s\n", filename);
        return NULL;
    }

    int capacity = 16;
    int count = 0;
    char **lines = (char **)calloc((size_t)capacity, sizeof(char *));
    if (!lines)
    {
        fclose(file);
        return NULL;
    }

    char *buffer = NULL;
    size_t buffer_size = 0;
    ssize_t length;
    while ((length = getline(&buffer, &buffer_size, file)) != -1)
    {
        if (length > 0 && buffer[length - 1] == '\n')
        {
            buffer[--length] = '\0';
        }
        if (length > 0 && buffer[length - 1] == '\r')
        {
            buffer[--length] = '\0';
        }

        if (count == capacity)
        {
            if (capacity > INT_MAX / 2)
            {
                goto fail;
            }
            int new_capacity = capacity * 2;
            char **grown = (char **)realloc(lines, (size_t)new_capacity * sizeof(char *));
            if (!grown)
            {
                goto fail;
            }
            memset(grown + capacity, 0, (size_t)(new_capacity - capacity) * sizeof(char *));
            lines = grown;
            capacity = new_capacity;
        }
        lines[count] = (char *)malloc((size_t)length + 1);
        if (!lines[count])
        {
            goto fail;
        }
        memcpy(lines[count], buffer, (size_t)length + 1);
        ++count;
    }
    if (ferror(file))
    {
        goto fail;
    }

    free(buffer);
    fclose(file);
    *line_count = count;
    return lines;

fail:
    free(buffer);
    fclose(file);
    free_lines(lines, count);
    return NULL;
}

void free_lines(char **lines, int line_count)
{
    if (!lines)
    {
        return;
    }
    for (int i = 0; i < line_count; ++i)
    {
        free(lines[i]);
    }
    free(lines);
}
