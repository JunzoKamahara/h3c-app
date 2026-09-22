#include "h3_ffmpeg.h"

#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static const char *ffmpeg_program(void) {
    const char *override = getenv("H3_FFMPEG");
    return override && *override ? override : "ffmpeg";
}

static const char *ffprobe_program(void) {
    const char *override = getenv("H3_FFPROBE");
    return override && *override ? override : "ffprobe";
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

int h3_ffprobe_visual_size(const char *path, int *width, int *height,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (width) *width = 0;
    if (height) *height = 0;
    if (!path || !*path || !width || !height) {
        fail(error, error_size, "invalid FFprobe visual-size arguments");
        return 0;
    }
    int stream[2];
    if (pipe(stream) != 0) {
        fail(error, error_size, "cannot create FFprobe pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
        (char *)path, NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffprobe_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        fail(error, error_size, "cannot start FFprobe: %s", strerror(code));
        return 0;
    }
    char output[128];
    size_t received = 0;
    int overflow = 0;
    while (1) {
        char byte;
        ssize_t amount = read(stream[0], &byte, 1);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        if (received + 1 < sizeof(output)) output[received++] = byte;
        else overflow = 1;
    }
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        fail(error, error_size, "cannot wait for FFprobe: %s", strerror(errno));
        return 0;
    }
    output[received] = '\0';
    int parsed_width = 0, parsed_height = 0, consumed = 0;
    if (overflow || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        sscanf(output, "%dx%d%n", &parsed_width, &parsed_height, &consumed) != 2) {
        fail(error, error_size, "FFprobe could not inspect visual stream %s",
             path);
        return 0;
    }
    for (char *cursor = output + consumed; *cursor; cursor++) {
        if (*cursor != ' ' && *cursor != '\t' && *cursor != '\r' &&
            *cursor != '\n') {
            fail(error, error_size, "FFprobe returned an invalid visual size");
            return 0;
        }
    }
    if (parsed_width < 1 || parsed_height < 1) {
        fail(error, error_size, "visual stream has invalid dimensions %dx%d",
             parsed_width, parsed_height);
        return 0;
    }
    *width = parsed_width;
    *height = parsed_height;
    return 1;
}

int h3_ffmpeg_read_image_f32(const char *path, int width, int height,
                             h3_image_fit fit, float **pixels,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (!path || !*path || !pixels || width < 1 || height < 1 ||
        (fit != H3_IMAGE_FIT_STRETCH && fit != H3_IMAGE_FIT_COVER)) {
        fail(error, error_size, "invalid FFmpeg image input arguments");
        return 0;
    }
    if ((size_t)width > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "decoded image size overflows");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3 ||
        area * 3 > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "decoded image size overflows");
        return 0;
    }
    size_t bytes = area * 3;
    uint8_t *rgb = malloc(bytes);
    float *channel_major = malloc(bytes * sizeof(*channel_major));
    if (!rgb || !channel_major) {
        free(rgb);
        free(channel_major);
        fail(error, error_size, "out of memory decoding input image");
        return 0;
    }
    char filter[256];
    if (fit == H3_IMAGE_FIT_STRETCH) {
        snprintf(filter, sizeof(filter), "scale=%d:%d:flags=lanczos",
                 width, height);
    } else {
        snprintf(filter, sizeof(filter),
                 "scale=%d:%d:force_original_aspect_ratio=increase:flags=lanczos,"
                 "crop=%d:%d", width, height, width, height);
    }
    int stream[2];
    if (pipe(stream) != 0) {
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot create FFmpeg image pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-frames:v", "1", "-vf", filter,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t received = 0;
    while (received < bytes) {
        ssize_t amount = read(stream[0], rgb + received, bytes - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        received != bytes || trailing != 0) {
        free(rgb);
        free(channel_major);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            fail(error, error_size, "FFmpeg could not decode image %s (status %d)",
                 path, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        else
            fail(error, error_size,
                 "FFmpeg decoded %zu bytes for %dx%d image, expected %zu",
                 received + (trailing > 0 ? 1u : 0u), width, height, bytes);
        return 0;
    }
    const float scale = 1.0f / 255.0f;
    for (size_t pixel = 0; pixel < area; pixel++) {
        channel_major[pixel] = (float)rgb[3 * pixel] * scale;
        channel_major[area + pixel] = (float)rgb[3 * pixel + 1] * scale;
        channel_major[2 * area + pixel] = (float)rgb[3 * pixel + 2] * scale;
    }
    free(rgb);
    *pixels = channel_major;
    return 1;
}

int h3_ffmpeg_read_video_f32(const char *path, int width, int height,
                             int max_frames, float **pixels, int *frames,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (frames) *frames = 0;
    if (!path || !*path || !pixels || !frames || width < 1 || height < 1 ||
        max_frames < 5 || (size_t)width > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "invalid FFmpeg video input arguments");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3) {
        fail(error, error_size, "decoded video frame size overflows");
        return 0;
    }
    size_t frame_bytes = area * 3;
    if ((size_t)max_frames > SIZE_MAX / frame_bytes) {
        fail(error, error_size, "decoded video size overflows");
        return 0;
    }
    size_t capacity = (size_t)max_frames * frame_bytes;
    uint8_t *rgb = malloc(capacity);
    if (!rgb) {
        fail(error, error_size, "out of memory decoding input video");
        return 0;
    }
    char filter[256], frame_limit[32];
    snprintf(filter, sizeof(filter),
             "fps=24,scale=%d:%d:flags=lanczos,setsar=1", width, height);
    snprintf(frame_limit, sizeof(frame_limit), "%d", max_frames);
    int stream[2];
    if (pipe(stream) != 0) {
        free(rgb);
        fail(error, error_size, "cannot create FFmpeg video pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-map", "0:v:0", "-an", "-vf", filter,
        "-frames:v", frame_limit, "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(rgb);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t received = 0;
    while (received < capacity) {
        ssize_t amount = read(stream[0], rgb + received, capacity - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(rgb);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || trailing != 0 ||
        received % frame_bytes) {
        free(rgb);
        fail(error, error_size, "FFmpeg could not decode bounded video %s",
             path);
        return 0;
    }
    int frame_count = (int)(received / frame_bytes);
    if (frame_count < 5) {
        free(rgb);
        fail(error, error_size,
             "reference videos require at least 5 decoded frames");
        return 0;
    }
    while (frame_count % 17 != 5) frame_count--;
    if ((size_t)frame_count > SIZE_MAX / 3 / area ||
        (size_t)frame_count * 3 * area > SIZE_MAX / sizeof(float)) {
        free(rgb);
        fail(error, error_size, "converted video size overflows");
        return 0;
    }
    size_t values = (size_t)frame_count * 3 * area;
    float *channel_major = malloc(values * sizeof(*channel_major));
    if (!channel_major) {
        free(rgb);
        fail(error, error_size, "out of memory converting input video");
        return 0;
    }
    const float scale = 1.0f / 255.0f;
    for (int time = 0; time < frame_count; time++)
        for (size_t pixel = 0; pixel < area; pixel++)
            for (size_t channel = 0; channel < 3; channel++) {
                size_t source = ((size_t)time * area + pixel) * 3 + channel;
                size_t destination = (channel * (size_t)frame_count +
                                      (size_t)time) * area + pixel;
                channel_major[destination] = (float)rgb[source] * scale;
            }
    free(rgb);
    *pixels = channel_major;
    *frames = frame_count;
    return 1;
}

int h3_ffmpeg_read_audio_f32(const char *path, int max_samples,
                             int truncate_at_limit,
                             float **pcm, int *samples,
                             char *error, size_t error_size) {
    enum { AUDIO_RATE = 32000, AUDIO_CHANNELS = 2, MIN_SAMPLES = 64000 };
    if (error && error_size) error[0] = '\0';
    if (pcm) *pcm = NULL;
    if (samples) *samples = 0;
    if (!path || !*path || !pcm || !samples || max_samples < MIN_SAMPLES ||
        max_samples > AUDIO_RATE * 15 ||
        (truncate_at_limit != 0 && truncate_at_limit != 1)) {
        fail(error, error_size, "invalid FFmpeg audio input arguments");
        return 0;
    }
    size_t elements = (size_t)max_samples * AUDIO_CHANNELS;
    if (elements > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "decoded audio size overflows");
        return 0;
    }
    float *interleaved = malloc(elements * sizeof(*interleaved));
    if (!interleaved) {
        fail(error, error_size, "out of memory decoding reference audio");
        return 0;
    }
    char duration[64];
    double seconds = (double)max_samples / (double)AUDIO_RATE;
    if (!truncate_at_limit) seconds += 1.0 / (double)AUDIO_RATE;
    snprintf(duration, sizeof(duration), "%.9f", seconds);
    int stream[2];
    if (pipe(stream) != 0) {
        free(interleaved);
        fail(error, error_size, "cannot create FFmpeg audio pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "32000",
        "-t", duration, "-f", "f32le", "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(interleaved);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t capacity = elements * sizeof(*interleaved);
    size_t received = 0;
    while (received < capacity) {
        ssize_t amount = read(stream[0], (uint8_t *)interleaved + received,
                              capacity - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(interleaved);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    size_t frame_bytes = AUDIO_CHANNELS * sizeof(float);
    if (!truncate_at_limit && trailing > 0) {
        free(interleaved);
        fail(error, error_size,
             "reference audio exceeds the 15 second total limit");
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        received % frame_bytes) {
        free(interleaved);
        fail(error, error_size,
             "FFmpeg could not decode a stereo soundtrack from %s", path);
        return 0;
    }
    int sample_count = (int)(received / frame_bytes);
    if (sample_count < MIN_SAMPLES) {
        free(interleaved);
        fail(error, error_size,
             "reference audio requires at least 2 seconds at 32 kHz");
        return 0;
    }
    size_t output_elements = (size_t)sample_count * AUDIO_CHANNELS;
    float *channel_major = malloc(output_elements * sizeof(*channel_major));
    if (!channel_major) {
        free(interleaved);
        fail(error, error_size, "out of memory converting reference audio");
        return 0;
    }
    for (int sample = 0; sample < sample_count; sample++)
        for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
            channel_major[(size_t)channel * (size_t)sample_count +
                          (size_t)sample] =
                interleaved[(size_t)sample * AUDIO_CHANNELS +
                            (size_t)channel];
    free(interleaved);
    *pcm = channel_major;
    *samples = sample_count;
    return 1;
}
