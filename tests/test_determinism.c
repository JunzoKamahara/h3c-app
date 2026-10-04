/* Repeat-run determinism check for the Metal kernels that reduce into a
 * threadgroup array and then reuse it (h3_gqa_causal_bf16,
 * h3_vae_encoder_group_norm_silu_f32). A missing barrier between the
 * broadcast read of reductions[0] and the next write made the same inputs
 * intermittently give different outputs, which showed up as same-seed
 * videos differing. Every run must match the first one bit for bit.
 *
 * Usage: h3_determinism_tests [shader.metal] [iterations] */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    /* Qwen text encoder attention shape; 1024 tokens is in the range of a
     * Ref2VA prompt with one reference image. With the barrier removed this
     * differed in 15-49 of 50 runs. */
    GQA_SEQUENCE = 1024,
    GQA_QUERY_HEADS = 64,
    GQA_KV_HEADS = 8,
    GQA_HEAD_DIM = 128,
    /* GroupNorm with 32 groups, as in h3_video_encoder.c. Many small planes
     * (16384 threadgroups, 32 elements per thread) keep the window between
     * the mean read and the variance write short, which is what exposes a
     * missing barrier: with the barrier removed this shape differed in
     * 10-13 of 50 runs, while a few large planes (3x64x64) differed in
     * about 1 of 150. */
    NORM_DEPTH = 512,
    NORM_HEIGHT = 16,
    NORM_WIDTH = 16,
    NORM_CHANNELS = 128,
    NORM_GROUPS = 32,
};

static h3_gpu *gpu;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_determinism.c: %s\n", message);
    exit(1);
}

static void gpu_ok(int ok, const char *operation) {
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_determinism.c: %s: %s\n", operation,
                gpu ? h3_gpu_error(gpu) : "no GPU");
        exit(1);
    }
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static float next_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)(rng_state >> 40) / (float)(1u << 24) * 2.0f - 1.0f;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static void *checked_malloc(size_t bytes) {
    void *memory = malloc(bytes);
    if (!memory) die("out of memory");
    return memory;
}

static uint16_t *random_bf16(size_t count, float scale) {
    uint16_t *values = checked_malloc(count * sizeof(*values));
    for (size_t index = 0; index < count; index++)
        values[index] = to_bf16(next_uniform() * scale);
    return values;
}

static float *random_f32(size_t count, float offset, float scale) {
    float *values = checked_malloc(count * sizeof(*values));
    for (size_t index = 0; index < count; index++)
        values[index] = offset + next_uniform() * scale;
    return values;
}

/* Returns the number of runs that differed from the first. */
static int check_gqa(int iterations) {
    size_t query_count = (size_t)GQA_SEQUENCE * GQA_QUERY_HEADS * GQA_HEAD_DIM;
    size_t kv_count = (size_t)GQA_SEQUENCE * GQA_KV_HEADS * GQA_HEAD_DIM;
    uint16_t *query_values = random_bf16(query_count, 2.0f);
    uint16_t *key_values = random_bf16(kv_count, 2.0f);
    uint16_t *value_values = random_bf16(kv_count, 1.0f);
    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(gpu, query_values, query_count);
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(gpu, key_values, kv_count);
    h3_gpu_tensor *value = h3_gpu_tensor_from_bf16(gpu, value_values, kv_count);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, query_count);
    if (!query || !key || !value || !output) die("GQA tensor allocation failed");
    uint16_t *first = checked_malloc(query_count * sizeof(*first));
    uint16_t *got = checked_malloc(query_count * sizeof(*got));
    int differing = 0;
    for (int run = 0; run < iterations; run++) {
        gpu_ok(h3_gpu_begin(gpu), "begin");
        gpu_ok(h3_gpu_gqa_causal_bf16(gpu, output, query, key, value,
                                      GQA_SEQUENCE, GQA_QUERY_HEADS,
                                      GQA_KV_HEADS, GQA_HEAD_DIM,
                                      1.0f / sqrtf((float)GQA_HEAD_DIM)),
               "causal GQA");
        gpu_ok(h3_gpu_submit(gpu), "submit");
        gpu_ok(h3_gpu_tensor_read_bf16(output, run ? got : first, query_count),
               "read GQA output");
        if (run && memcmp(first, got, query_count * sizeof(*got))) {
            size_t mismatches = 0;
            for (size_t index = 0; index < query_count; index++)
                mismatches += first[index] != got[index];
            fprintf(stderr, "causal GQA run %d differs from run 0 in %zu of "
                    "%zu values\n", run, mismatches, query_count);
            differing++;
        }
    }
    printf("determinism causal GQA       %d runs, %d differing\n", iterations,
           differing);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(output);
    free(query_values);
    free(key_values);
    free(value_values);
    free(first);
    free(got);
    return differing;
}

static int check_group_norm(int iterations) {
    size_t count = (size_t)NORM_DEPTH * NORM_HEIGHT * NORM_WIDTH * NORM_CHANNELS;
    float *input_values = random_f32(count, 0.3f, 2.0f);
    float *weight_values = random_f32(NORM_CHANNELS, 1.0f, 0.5f);
    float *bias_values = random_f32(NORM_CHANNELS, 0.0f, 0.5f);
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, input_values, count);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_f32(gpu, weight_values,
                                                   NORM_CHANNELS);
    h3_gpu_tensor *bias = h3_gpu_tensor_from_f32(gpu, bias_values, NORM_CHANNELS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, count);
    if (!input || !weight || !bias || !output)
        die("GroupNorm tensor allocation failed");
    float *first = checked_malloc(count * sizeof(*first));
    float *got = checked_malloc(count * sizeof(*got));
    int differing = 0;
    for (int run = 0; run < iterations; run++) {
        gpu_ok(h3_gpu_begin(gpu), "begin");
        gpu_ok(h3_gpu_vae_encoder_group_norm_silu_f32(
                   gpu, output, input, weight, bias, 1, NORM_DEPTH,
                   NORM_HEIGHT, NORM_WIDTH, NORM_CHANNELS, NORM_GROUPS, 1e-6f),
               "GroupNorm SiLU");
        gpu_ok(h3_gpu_submit(gpu), "submit");
        gpu_ok(h3_gpu_tensor_read_f32(output, run ? got : first, count),
               "read GroupNorm output");
        if (run && memcmp(first, got, count * sizeof(*got))) {
            size_t mismatches = 0;
            for (size_t index = 0; index < count; index++)
                mismatches += memcmp(&first[index], &got[index],
                                     sizeof(*got)) != 0;
            fprintf(stderr, "GroupNorm SiLU run %d differs from run 0 in %zu "
                    "of %zu values\n", run, mismatches, count);
            differing++;
        }
    }
    printf("determinism GroupNorm SiLU   %d runs, %d differing\n", iterations,
           differing);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(bias);
    h3_gpu_tensor_free(output);
    free(input_values);
    free(weight_values);
    free(bias_values);
    free(first);
    free(got);
    return differing;
}

int main(int argc, char **argv) {
    const char *shaders = argc > 1 ? argv[1] : "h3_shaders.metal";
    int iterations = argc > 2 ? atoi(argv[2]) : 50;
    if (iterations < 2) die("iterations must be at least 2");
    char error[512];
    gpu = h3_gpu_create(shaders, error, sizeof(error));
    if (!gpu) die(error);
    int differing = check_gqa(iterations) + check_group_norm(iterations);
    h3_gpu_free(gpu);
    if (differing) die("repeated runs on identical inputs gave different output");
    return 0;
}
