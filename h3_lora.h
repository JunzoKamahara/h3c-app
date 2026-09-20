#ifndef H3_LORA_H
#define H3_LORA_H

/* Shared LoRA-fusion primitives: fusing a diffusers/peft-format adapter
 * (separate to_q/to_k/to_v/to_out/ff.net lora_A/lora_B pairs) into base
 * MiniMax-H3 DiT BF16 weights. Used by:
 *   - h3_build_lora_cache.c: the offline tool that pre-fuses+quantizes a
 *     whole H3_ATTENTION_CACHE-format file once.
 *   - h3_dit.c: H3_LORA_PATH, which fuses the same way at model-load time
 *     for the resident BF16/int8 path, and (materializing a cache file
 *     through the same routines this header exposes) for H3_ATTENTION_CACHE
 *     streaming too.
 *
 * The DiT's dimensions are fixed for this model, so every LoRA-aware file
 * shares them here instead of each redefining its own copy. */
enum {
    H3_LORA_HIDDEN = 5376,
    H3_LORA_HEADS = 56,
    H3_LORA_HEAD_DIM = 128,
    H3_LORA_INNER = H3_LORA_HEADS * H3_LORA_HEAD_DIM,
    H3_LORA_FFN = 14336,
    H3_LORA_DIT_BLOCKS = 50,
};

#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

uint16_t h3_lora_f32_to_bf16(float value);
float h3_lora_bf16_to_f32(uint16_t value);

/* Reads a 2D BF16 tensor (base checkpoint or LoRA file - both go through
 * the same h3_st_header/h3_st_tensor interface) with an exactly-known
 * shape as a freshly allocated row-major F32 array. */
float *h3_lora_read_tensor_f32(const h3_st_header *header, const char *name,
                               uint64_t rows, uint64_t columns, char *error,
                               size_t error_size);

/* One projection's LoRA source: delta[out_dim, in_dim] = scale *
 * B[out_dim, rank] @ A[rank, in_dim], added into base[rows_total, in_dim]
 * starting at row_offset (so three sources back to back build the
 * concatenated QKV delta in place - rank is read from lora_A's own shape,
 * not assumed). */
typedef struct {
    const char *a_name;
    const char *b_name;
    uint64_t out_dim;
    uint64_t row_offset;
} h3_lora_source;

int h3_lora_fuse_projection(const h3_st_header *lora, const char *a_name,
                            const char *b_name, uint64_t out_dim,
                            uint64_t in_dim, float scale, float *base,
                            uint64_t row_offset, char *error,
                            size_t error_size);

/* Loads base_name [rows, columns] from store, fuses zero or more sources
 * into it, and returns the result as a freshly allocated row-major F32
 * array (caller frees it). */
float *h3_lora_fuse_weight_f32(h3_weight_store *store, const h3_st_header *lora,
                               const char *base_name, uint64_t rows,
                               uint64_t columns, const h3_lora_source *sources,
                               int source_count, float scale, char *error,
                               size_t error_size);

/* Fuses base_name the same way, quantizes the F32 result on gpu exactly
 * like the resident int8 path (h3_gpu_quantize_weight_int8), and appends
 * the int8 payload + per-row F32 scales to `out` - one record in the
 * H3AC cache layout h3_build_attention_cache.c/h3_dit.c share. */
int h3_lora_fuse_quantize_and_write(h3_gpu *gpu, h3_weight_store *store,
                                    const h3_st_header *lora,
                                    const char *base_name, uint64_t rows,
                                    uint64_t columns,
                                    const h3_lora_source *sources,
                                    int source_count, float scale, FILE *out,
                                    char *error, size_t error_size);

/* Fuses base_name and writes it as raw BF16 (no quantization) to `out`. */
int h3_lora_fuse_and_write_bf16(h3_weight_store *store,
                                const h3_st_header *lora,
                                const char *base_name, uint64_t rows,
                                uint64_t columns,
                                const h3_lora_source *sources,
                                int source_count, float scale, FILE *out,
                                char *error, size_t error_size);

/* Copies one tensor through unchanged (no LoRA delta), raw BF16. */
int h3_lora_copy_raw_bf16(h3_weight_store *store, const char *name, FILE *out,
                          char *error, size_t error_size);

/* One block's four LoRA-able projections (qkv/out/fc1/fc2), with their
 * checkpoint-side base name and matching LoRA-file sources already filled
 * in. name_buffers must hold at least H3_LORA_NAME_BUFFERS buffers of
 * >=160 bytes - the returned sources' string pointers (and each
 * projection's checkpoint_name) point into it, so it must outlive any use
 * of the result. */
enum { H3_LORA_NAME_BUFFERS = 16 };

typedef struct {
    const char *checkpoint_name;
    uint64_t rows;
    uint64_t columns;
    h3_lora_source sources[3];
    int source_count;
} h3_lora_projection;

enum { H3_LORA_QKV = 0, H3_LORA_OUT = 1, H3_LORA_FC1 = 2, H3_LORA_FC2 = 3 };

void h3_lora_block_projections(const char *checkpoint_prefix,
                               const char *lora_prefix,
                               char name_buffers[H3_LORA_NAME_BUFFERS][160],
                               h3_lora_projection out[4]);

/* Auto-detects the LoRA scale from the adapter's own "alpha" metadata and
 * its block-0 to_q rank (delta = alpha/rank * B @ A is the diffusers/peft
 * convention when alpha != rank; lightx2v's Minimax-h3-Turbo ships both
 * kinds - some releases have alpha == rank, needing no scaling, others
 * alpha = 8 with rank 128). Returns 1 and sets *out_scale if the adapter
 * carries "alpha" metadata and its rank could be read; returns 0 (leaving
 * *out_scale untouched) if there is no "alpha" key, so the caller can fall
 * back to its own default (1.0) for adapters that do not publish it. */
int h3_lora_detect_scale(const h3_st_header *lora, float *out_scale,
                         char *error, size_t error_size);

#endif
