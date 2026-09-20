/* Fuses a diffusers/peft-style LoRA adapter (separate to_q/to_k/to_v/
 * to_out/ff.net LoRA pairs, e.g. lightx2v/Minimax-h3-Turbo) into the base
 * BF16 DiT weights, then quantizes the result into exactly the same H3AC
 * v2 cache format h3_build_attention_cache.c produces.
 *
 * This is the whole trick for getting "runtime LoRA composition" out of
 * h3.c without touching the hot path at all: the fusion happens once,
 * offline, on the CPU (the low-rank deltas are tiny - rank 128 in the
 * lightx2v release - so a BLAS sgemm per projection is instant compared to
 * reading the ~62GB base model). The output file is byte-for-byte the same
 * layout an ordinary (non-LoRA) attention cache would have, so
 * H3_ATTENTION_CACHE in h3_dit.c needs zero changes to stream it - it has
 * no way to tell the difference, and that is the point.
 *
 * h3.c's own checkpoint stores QKV already fused into one
 * "attn.qkv_proj.weight" matrix ([INNER*3, HIDDEN]), while the LoRA has
 * three separate low-rank adapters (to_q/to_k/to_v, [INNER, HIDDEN] each).
 * We compute each delta separately and concatenate them along the output
 * (row) dimension in Q,K,V order to match. attn.out_proj / mlp.fc1 /
 * mlp.fc2 map onto to_out.0 / ff.net.0.proj / ff.net.2 one-to-one with no
 * concatenation needed. LoRA rank is read from each lora_A tensor's own
 * shape rather than assumed, so adapters with a different rank per
 * projection (or a different rank than the 128 lightx2v ships) still work.
 * All of this (and the underlying BLAS fusion) lives in h3_lora.c/h3_lora.h,
 * shared with h3_dit.c's H3_LORA_PATH (see its top comment there).
 *
 * The lightx2v Minimax-h3-Turbo release ships alpha == rank (both 128), so
 * the default lora_scale of 1.0 applies no extra scaling. Pass a different
 * scale as the 4th argument for adapters where that does not hold
 * (delta = scale * B @ A).
 *
 * Also writes a second, much smaller file for the two BF16-resident
 * text token_refiner blocks (the LoRA's "token_refiner.refiner_blocks.0/1"
 * - a separate component H3_ATTENTION_CACHE never touches, see the
 * write_token_refiner_cache comment below), named by inserting "_refiner"
 * before the main cache path's extension. h3_dit.c's H3_TOKEN_REFINER_LORA
 * env var points at this file - though H3_LORA_PATH alone (no separate
 * build step) now also covers the refiner, since it stays BF16-resident
 * either way and gets fused live at load time.
 *
 * Usage: build_lora_cache <FL2VA/transformer dir> <lora .safetensors> \
 *                          <output cache file> [lora_scale]
 */
#include "h3_gpu.h"
#include "h3_lora.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = H3_LORA_HIDDEN,
    INNER = H3_LORA_INNER,
    FFN = H3_LORA_FFN,
    HEAD_DIM = H3_LORA_HEAD_DIM,
    DIT_BLOCKS = H3_LORA_DIT_BLOCKS,
};

#define CACHE_MAGIC "H3AC"
#define CACHE_VERSION 2u

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t block_count;
    uint32_t hidden;
    uint32_t inner;
    uint32_t ffn;
    uint32_t reserved[10];
} cache_header;

/* --- token_refiner: 2 small BF16-resident blocks, not part of the H3AC
 * int8 cache above. h3_dit.c loads them straight from the checkpoint on
 * every run (see load_block() there), so there is no existing streaming
 * slot to drop a fused int8 payload into - instead this writes a small
 * standalone BF16 file in the exact tensor order load_block() reads
 * (norm1, norm2, qkv, q_norm, k_norm, out, fc1, fc2) per block, which
 * H3_TOKEN_REFINER_LORA in h3_dit.c reads back by fixed offset. norm1/
 * norm2/q_norm/k_norm carry no LoRA delta - they are copied through
 * unchanged, byte for byte, so the override file is a complete drop-in
 * replacement for both refiner blocks rather than a delta-only patch. */
#define TOKEN_REFINER_MAGIC "H3RF"
#define TOKEN_REFINER_VERSION 1u
enum { TOKEN_REFINER_BLOCKS = 2 };

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t block_count;
    uint32_t hidden;
    uint32_t inner;
    uint32_t ffn;
    uint32_t head_dim;
    uint32_t reserved[9];
} token_refiner_header;

static int write_token_refiner_cache(h3_weight_store *store,
                                     const h3_st_header *lora,
                                     float lora_scale, const char *path,
                                     char *error, size_t error_size) {
    FILE *out = fopen(path, "wb");
    if (!out) {
        snprintf(error, error_size, "cannot open %s: %s", path,
                 strerror(errno));
        return 0;
    }
    token_refiner_header header = {0};
    memcpy(header.magic, TOKEN_REFINER_MAGIC, 4);
    header.version = TOKEN_REFINER_VERSION;
    header.block_count = TOKEN_REFINER_BLOCKS;
    header.hidden = HIDDEN;
    header.inner = INNER;
    header.ffn = FFN;
    header.head_dim = HEAD_DIM;
    if (fwrite(&header, sizeof(header), 1, out) != 1) {
        snprintf(error, error_size, "cannot write token-refiner header: %s",
                 strerror(errno));
        fclose(out);
        return 0;
    }

    for (uint32_t block = 0; block < TOKEN_REFINER_BLOCKS; block++) {
        char checkpoint_prefix[64], lora_prefix[64];
        snprintf(checkpoint_prefix, sizeof(checkpoint_prefix),
                 "token_refiner.blocks.%u.", block);
        snprintf(lora_prefix, sizeof(lora_prefix),
                 "token_refiner.refiner_blocks.%u.", block);
        char name_buffers[H3_LORA_NAME_BUFFERS][160];
        h3_lora_projection projections[4];
        h3_lora_block_projections(checkpoint_prefix, lora_prefix,
                                  name_buffers, projections);
        error[0] = '\0';

        char base_name[160];
        snprintf(base_name, sizeof(base_name), "%snorm1.weight",
                 checkpoint_prefix);
        if (!h3_lora_copy_raw_bf16(store, base_name, out, error,
                                   error_size)) {
            fclose(out);
            return 0;
        }
        snprintf(base_name, sizeof(base_name), "%snorm2.weight",
                 checkpoint_prefix);
        if (!h3_lora_copy_raw_bf16(store, base_name, out, error,
                                   error_size)) {
            fclose(out);
            return 0;
        }

        const h3_lora_projection *qkv = &projections[H3_LORA_QKV];
        if (!h3_lora_fuse_and_write_bf16(store, lora, qkv->checkpoint_name,
                                         qkv->rows, qkv->columns,
                                         qkv->sources, qkv->source_count,
                                         lora_scale, out, error,
                                         error_size)) {
            fclose(out);
            return 0;
        }

        snprintf(base_name, sizeof(base_name), "%sattn.q_norm.weight",
                 checkpoint_prefix);
        if (!h3_lora_copy_raw_bf16(store, base_name, out, error,
                                   error_size)) {
            fclose(out);
            return 0;
        }
        snprintf(base_name, sizeof(base_name), "%sattn.k_norm.weight",
                 checkpoint_prefix);
        if (!h3_lora_copy_raw_bf16(store, base_name, out, error,
                                   error_size)) {
            fclose(out);
            return 0;
        }

        const h3_lora_projection *out_proj = &projections[H3_LORA_OUT];
        if (!h3_lora_fuse_and_write_bf16(
                store, lora, out_proj->checkpoint_name, out_proj->rows,
                out_proj->columns, out_proj->sources, out_proj->source_count,
                lora_scale, out, error, error_size)) {
            fclose(out);
            return 0;
        }

        const h3_lora_projection *fc1 = &projections[H3_LORA_FC1];
        if (!h3_lora_fuse_and_write_bf16(store, lora, fc1->checkpoint_name,
                                         fc1->rows, fc1->columns,
                                         fc1->sources, fc1->source_count,
                                         lora_scale, out, error,
                                         error_size)) {
            fclose(out);
            return 0;
        }

        const h3_lora_projection *fc2 = &projections[H3_LORA_FC2];
        if (!h3_lora_fuse_and_write_bf16(store, lora, fc2->checkpoint_name,
                                         fc2->rows, fc2->columns,
                                         fc2->sources, fc2->source_count,
                                         lora_scale, out, error,
                                         error_size)) {
            fclose(out);
            return 0;
        }

        fprintf(stderr, "h3: token-refiner lora block %u/%u\n", block + 1,
                TOKEN_REFINER_BLOCKS);
    }

    fclose(out);
    return 1;
}

/* Inserts "_refiner" before the cache path's final extension (or appends
 * it if there is none), so build_lora_cache's two output files sit next
 * to each other with obviously related names. Caller frees the result. */
static char *derive_refiner_path(const char *cache_path) {
    const char *dot = strrchr(cache_path, '.');
    const char *slash = strrchr(cache_path, '/');
    if (dot && slash && dot < slash) dot = NULL; /* dot was in a directory */
    size_t stem_len = dot ? (size_t)(dot - cache_path) : strlen(cache_path);
    const char *suffix = "_refiner";
    const char *extension = dot ? dot : "";
    size_t total = stem_len + strlen(suffix) + strlen(extension) + 1;
    char *result = malloc(total);
    if (!result) return NULL;
    memcpy(result, cache_path, stem_len);
    memcpy(result + stem_len, suffix, strlen(suffix));
    memcpy(result + stem_len + strlen(suffix), extension,
           strlen(extension) + 1);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr,
                "usage: %s <FL2VA/transformer dir> <lora .safetensors> "
                "<output cache file> [lora_scale]\n", argv[0]);
        return 1;
    }
    int scale_given = argc == 5;
    float lora_scale = scale_given ? (float)atof(argv[4]) : 1.0f;
    char error[512] = {0};

    h3_weight_store *store = h3_weight_store_open(argv[1], error,
                                                   sizeof(error));
    if (!store) {
        fprintf(stderr, "h3: %s\n", error);
        return 1;
    }
    h3_st_header lora;
    if (!h3_st_read_header(argv[2], &lora, error, sizeof(error))) {
        fprintf(stderr, "h3: %s\n", error);
        h3_weight_store_free(store);
        return 1;
    }
    if (!scale_given) {
        /* No explicit scale argument - try the adapter's own "alpha"
         * metadata (see h3_lora_detect_scale's comment in h3_lora.h).
         * Using 1.0 unconditionally silently over-applies adapters whose
         * alpha != rank by a large factor, producing badly corrupted
         * output with no error. */
        float detected_scale;
        if (h3_lora_detect_scale(&lora, &detected_scale, error,
                                 sizeof(error))) {
            lora_scale = detected_scale;
            fprintf(stderr, "h3: detected LoRA alpha metadata -> scale=%.6f\n",
                    (double)detected_scale);
        } else if (error[0]) {
            fprintf(stderr, "h3: %s\n", error);
            h3_st_free_header(&lora);
            h3_weight_store_free(store);
            return 1;
        }
    }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "h3: %s\n", error);
        h3_st_free_header(&lora);
        h3_weight_store_free(store);
        return 1;
    }
    if (!h3_gpu_has_int8_mlp(gpu)) {
        fprintf(stderr,
                "h3: this GPU lacks the int8 path the cache is built for\n");
        h3_gpu_free(gpu);
        h3_st_free_header(&lora);
        h3_weight_store_free(store);
        return 1;
    }

    FILE *out = fopen(argv[3], "wb");
    if (!out) {
        fprintf(stderr, "h3: cannot open %s: %s\n", argv[3],
                strerror(errno));
        h3_gpu_free(gpu);
        h3_st_free_header(&lora);
        h3_weight_store_free(store);
        return 1;
    }

    cache_header header = {0};
    memcpy(header.magic, CACHE_MAGIC, 4);
    header.version = CACHE_VERSION;
    header.block_count = DIT_BLOCKS;
    header.hidden = HIDDEN;
    header.inner = INNER;
    header.ffn = FFN;
    if (fwrite(&header, sizeof(header), 1, out) != 1) {
        fprintf(stderr, "h3: cannot write cache header: %s\n",
                strerror(errno));
        fclose(out);
        h3_gpu_free(gpu);
        h3_st_free_header(&lora);
        h3_weight_store_free(store);
        return 1;
    }

    fprintf(stderr, "h3: fusing LoRA (scale=%.4f) into %u blocks\n",
            (double)lora_scale, (unsigned)DIT_BLOCKS);

    for (uint32_t block = 0; block < DIT_BLOCKS; block++) {
        char checkpoint_prefix[64], lora_prefix[64];
        snprintf(checkpoint_prefix, sizeof(checkpoint_prefix), "blocks.%u.",
                 block);
        snprintf(lora_prefix, sizeof(lora_prefix), "transformer_blocks.%u.",
                 block);
        char name_buffers[H3_LORA_NAME_BUFFERS][160];
        h3_lora_projection projections[4];
        h3_lora_block_projections(checkpoint_prefix, lora_prefix,
                                  name_buffers, projections);

        for (int p = 0; p < 4; p++) {
            const h3_lora_projection *proj = &projections[p];
            error[0] = '\0';
            if (!h3_lora_fuse_quantize_and_write(
                    gpu, store, &lora, proj->checkpoint_name, proj->rows,
                    proj->columns, proj->sources, proj->source_count,
                    lora_scale, out, error, sizeof(error))) {
                fprintf(stderr, "h3: %s\n", error);
                fclose(out);
                h3_gpu_free(gpu);
                h3_st_free_header(&lora);
                h3_weight_store_free(store);
                return 1;
            }
        }

        fprintf(stderr, "h3: lora cache block %2u/%u\n", block + 1,
                DIT_BLOCKS);
    }

    fclose(out);
    fprintf(stderr, "h3: wrote LoRA-fused attention cache to %s\n", argv[3]);

    /* token_refiner is a separate, much smaller component that
     * H3_ATTENTION_CACHE never touches (see this file's top comment) -
     * write it to its own file alongside the main cache. */
    char *refiner_path = derive_refiner_path(argv[3]);
    int refiner_ok = refiner_path != NULL;
    if (refiner_ok) {
        error[0] = '\0';
        refiner_ok = write_token_refiner_cache(store, &lora, lora_scale,
                                               refiner_path, error,
                                               sizeof(error));
    }
    if (!refiner_ok) {
        fprintf(stderr, "h3: %s\n",
                refiner_path ? error : "out of memory building refiner path");
    } else {
        fprintf(stderr, "h3: wrote LoRA-fused token-refiner override to %s\n",
                refiner_path);
    }
    free(refiner_path);

    h3_gpu_free(gpu);
    h3_st_free_header(&lora);
    h3_weight_store_free(store);
    return refiner_ok ? 0 : 1;
}
