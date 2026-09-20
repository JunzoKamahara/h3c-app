#include "h3_lora.h"

#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

/* Local aliases so the fusion math below reads the same as the two callers
 * that used to each define these constants themselves. */
enum {
    HIDDEN = H3_LORA_HIDDEN,
    INNER = H3_LORA_INNER,
    FFN = H3_LORA_FFN,
};

uint16_t h3_lora_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

float h3_lora_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

float *h3_lora_read_tensor_f32(const h3_st_header *header, const char *name,
                               uint64_t rows, uint64_t columns, char *error,
                               size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        snprintf(error, error_size, "tensor %s not found", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns) {
        snprintf(error, error_size,
                 "tensor %s has unexpected dtype/shape (want BF16 [%llu,%llu])",
                 name, (unsigned long long)rows, (unsigned long long)columns);
        return NULL;
    }
    size_t elements = (size_t)rows * columns;
    uint16_t *raw = malloc(elements * sizeof(uint16_t));
    float *values = malloc(elements * sizeof(float));
    if (!raw || !values) {
        snprintf(error, error_size, "out of memory reading %s", name);
        free(raw);
        free(values);
        return NULL;
    }
    int ok = h3_st_read_data(header, tensor, raw, elements * sizeof(uint16_t),
                             error, error_size);
    if (ok) {
        for (size_t i = 0; i < elements; i++) values[i] = h3_lora_bf16_to_f32(raw[i]);
    }
    free(raw);
    if (!ok) {
        free(values);
        return NULL;
    }
    return values;
}

static int load_lora_pair(const h3_st_header *lora, const char *a_name,
                          const char *b_name, uint64_t out_dim,
                          uint64_t in_dim, float **out_a, float **out_b,
                          uint64_t *out_rank, char *error, size_t error_size) {
    const h3_st_tensor *a_tensor = h3_st_find(lora, a_name);
    if (!a_tensor) {
        snprintf(error, error_size, "lora tensor %s not found", a_name);
        return 0;
    }
    if (a_tensor->dtype != H3_DTYPE_BF16 || a_tensor->ndim != 2 ||
        a_tensor->shape[1] != in_dim) {
        snprintf(error, error_size,
                 "lora_A %s has unexpected dtype/shape (want BF16 [rank,%llu])",
                 a_name, (unsigned long long)in_dim);
        return 0;
    }
    uint64_t rank = a_tensor->shape[0];

    float *a = h3_lora_read_tensor_f32(lora, a_name, rank, in_dim, error,
                                       error_size);
    if (!a) return 0;
    float *b = h3_lora_read_tensor_f32(lora, b_name, out_dim, rank, error,
                                       error_size);
    if (!b) {
        free(a);
        return 0;
    }
    *out_a = a;
    *out_b = b;
    *out_rank = rank;
    return 1;
}

int h3_lora_fuse_projection(const h3_st_header *lora, const char *a_name,
                            const char *b_name, uint64_t out_dim,
                            uint64_t in_dim, float scale, float *base,
                            uint64_t row_offset, char *error,
                            size_t error_size) {
    float *a, *b;
    uint64_t rank;
    if (!load_lora_pair(lora, a_name, b_name, out_dim, in_dim, &a, &b, &rank,
                        error, error_size))
        return 0;

    float *delta = malloc((size_t)out_dim * in_dim * sizeof(float));
    if (!delta) {
        snprintf(error, error_size, "out of memory computing delta for %s",
                 b_name);
        free(a);
        free(b);
        return 0;
    }
    /* The LP64 cblas_sgemm is deprecated in favor of an ILP64 interface
     * behind ACCELERATE_NEW_LAPACK, but these matrices (rank <= a few
     * hundred) never approach the 32-bit index range that distinction is
     * about - silence the notice rather than take on ILP64's wider types
     * project-wide for this one module. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               (int)out_dim, (int)in_dim, (int)rank, 1.0f, b, (int)rank, a,
               (int)in_dim, 0.0f, delta, (int)in_dim);
#pragma clang diagnostic pop
    for (uint64_t r = 0; r < out_dim; r++) {
        float *dest = base + (row_offset + r) * in_dim;
        const float *source = delta + r * in_dim;
        for (uint64_t c = 0; c < in_dim; c++) dest[c] += scale * source[c];
    }
    free(delta);
    free(b);
    free(a);
    return 1;
}

float *h3_lora_fuse_weight_f32(h3_weight_store *store, const h3_st_header *lora,
                               const char *base_name, uint64_t rows,
                               uint64_t columns, const h3_lora_source *sources,
                               int source_count, float scale, char *error,
                               size_t error_size) {
    const h3_st_header *base_header = NULL;
    const h3_st_tensor *base_tensor = h3_weight_find(store, base_name,
                                                      &base_header);
    if (!base_tensor) {
        snprintf(error, error_size, "base tensor %s not found", base_name);
        return NULL;
    }
    float *fused = h3_lora_read_tensor_f32(base_header, base_name, rows,
                                           columns, error, error_size);
    if (!fused) return NULL;
    for (int i = 0; i < source_count; i++) {
        const h3_lora_source *source = &sources[i];
        if (!h3_lora_fuse_projection(lora, source->a_name, source->b_name,
                                     source->out_dim, columns, scale, fused,
                                     source->row_offset, error, error_size)) {
            free(fused);
            return NULL;
        }
    }
    return fused;
}

int h3_lora_fuse_quantize_and_write(h3_gpu *gpu, h3_weight_store *store,
                                    const h3_st_header *lora,
                                    const char *base_name, uint64_t rows,
                                    uint64_t columns,
                                    const h3_lora_source *sources,
                                    int source_count, float scale, FILE *out,
                                    char *error, size_t error_size) {
    float *fused = h3_lora_fuse_weight_f32(store, lora, base_name, rows,
                                           columns, sources, source_count,
                                           scale, error, error_size);
    if (!fused) return 0;

    size_t elements = (size_t)rows * columns;
    uint16_t *fused_bf16 = malloc(elements * sizeof(uint16_t));
    if (!fused_bf16) {
        snprintf(error, error_size, "out of memory for fused %s", base_name);
        free(fused);
        return 0;
    }
    for (size_t i = 0; i < elements; i++)
        fused_bf16[i] = h3_lora_f32_to_bf16(fused[i]);
    free(fused);

    h3_gpu_tensor *bf16 = h3_gpu_tensor_from_bf16(gpu, fused_bf16, elements);
    free(fused_bf16);
    if (!bf16) {
        snprintf(error, error_size, "cannot upload fused %s: %s", base_name,
                 h3_gpu_error(gpu));
        return 0;
    }

    h3_gpu_tensor *i8 = h3_gpu_tensor_new_i8(gpu, elements);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, rows);
    int ok = i8 && scales &&
             h3_gpu_begin(gpu) &&
             h3_gpu_quantize_weight_int8(gpu, i8, scales, bf16, (uint32_t)rows,
                                         (uint32_t)columns) &&
             h3_gpu_submit(gpu);
    h3_gpu_tensor_free(bf16);
    if (!ok) {
        if (!error[0])
            snprintf(error, error_size, "cannot quantize %s: %s", base_name,
                     h3_gpu_error(gpu));
        h3_gpu_tensor_free(i8);
        h3_gpu_tensor_free(scales);
        return 0;
    }

    int8_t *i8_host = malloc(elements);
    float *scale_host = malloc((size_t)rows * sizeof(float));
    int result = i8_host && scale_host &&
        h3_gpu_tensor_read_i8(i8, i8_host, elements) &&
        h3_gpu_tensor_read_f32(scales, scale_host, rows) &&
        fwrite(i8_host, 1, elements, out) == elements &&
        fwrite(scale_host, sizeof(float), rows, out) == rows;
    free(i8_host);
    free(scale_host);
    h3_gpu_tensor_free(i8);
    h3_gpu_tensor_free(scales);
    if (!result && !error[0])
        snprintf(error, error_size, "cannot write cache payload for %s: %s",
                 base_name, strerror(errno));
    return result;
}

int h3_lora_fuse_and_write_bf16(h3_weight_store *store,
                                const h3_st_header *lora,
                                const char *base_name, uint64_t rows,
                                uint64_t columns,
                                const h3_lora_source *sources,
                                int source_count, float scale, FILE *out,
                                char *error, size_t error_size) {
    float *fused = h3_lora_fuse_weight_f32(store, lora, base_name, rows,
                                           columns, sources, source_count,
                                           scale, error, error_size);
    if (!fused) return 0;
    size_t elements = (size_t)rows * columns;
    uint16_t *fused_bf16 = malloc(elements * sizeof(uint16_t));
    if (!fused_bf16) {
        snprintf(error, error_size, "out of memory for fused %s", base_name);
        free(fused);
        return 0;
    }
    for (size_t i = 0; i < elements; i++)
        fused_bf16[i] = h3_lora_f32_to_bf16(fused[i]);
    free(fused);
    int ok = fwrite(fused_bf16, sizeof(uint16_t), elements, out) == elements;
    free(fused_bf16);
    if (!ok && !error[0])
        snprintf(error, error_size, "cannot write fused %s: %s", base_name,
                 strerror(errno));
    return ok;
}

int h3_lora_copy_raw_bf16(h3_weight_store *store, const char *name, FILE *out,
                          char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        snprintf(error, error_size, "base tensor %s not found", name);
        return 0;
    }
    if (tensor->dtype != H3_DTYPE_BF16) {
        snprintf(error, error_size, "tensor %s is not BF16", name);
        return 0;
    }
    uint64_t elements = h3_st_tensor_elements(tensor);
    uint16_t *raw = malloc(elements * sizeof(uint16_t));
    if (!raw) {
        snprintf(error, error_size, "out of memory reading %s", name);
        return 0;
    }
    int ok = h3_st_read_data(header, tensor, raw, elements * sizeof(uint16_t),
                             error, error_size) &&
             fwrite(raw, sizeof(uint16_t), elements, out) == elements;
    free(raw);
    if (!ok && !error[0])
        snprintf(error, error_size, "cannot write %s: %s", name,
                 strerror(errno));
    return ok;
}

int h3_lora_detect_scale(const h3_st_header *lora, float *out_scale,
                         char *error, size_t error_size) {
    const char *alpha_text = h3_st_metadata(lora, "alpha");
    if (!alpha_text) return 0;
    char *end = NULL;
    float alpha = strtof(alpha_text, &end);
    if (end == alpha_text || *end || !(alpha >= 0.0f)) {
        snprintf(error, error_size,
                 "lora metadata \"alpha\" is not a number: %s", alpha_text);
        return 0;
    }
    const h3_st_tensor *sample = h3_st_find(
        lora, "transformer_blocks.0.attn.to_q.lora_A.default.weight");
    if (!sample || sample->ndim != 2 || sample->shape[0] == 0) {
        snprintf(error, error_size,
                 "cannot read lora rank to resolve alpha=%s into a scale",
                 alpha_text);
        return 0;
    }
    *out_scale = alpha / (float)sample->shape[0];
    return 1;
}

void h3_lora_block_projections(const char *checkpoint_prefix,
                               const char *lora_prefix,
                               char name_buffers[H3_LORA_NAME_BUFFERS][160],
                               h3_lora_projection out[4]) {
    snprintf(name_buffers[0], 160, "%sattn.qkv_proj.weight", checkpoint_prefix);
    snprintf(name_buffers[1], 160, "%sattn.to_q.lora_A.default.weight", lora_prefix);
    snprintf(name_buffers[2], 160, "%sattn.to_q.lora_B.default.weight", lora_prefix);
    snprintf(name_buffers[3], 160, "%sattn.to_k.lora_A.default.weight", lora_prefix);
    snprintf(name_buffers[4], 160, "%sattn.to_k.lora_B.default.weight", lora_prefix);
    snprintf(name_buffers[5], 160, "%sattn.to_v.lora_A.default.weight", lora_prefix);
    snprintf(name_buffers[6], 160, "%sattn.to_v.lora_B.default.weight", lora_prefix);
    out[H3_LORA_QKV] = (h3_lora_projection){
        .checkpoint_name = name_buffers[0],
        .rows = (uint64_t)INNER * 3, .columns = HIDDEN,
        .sources = {
            { name_buffers[1], name_buffers[2], INNER, 0 },
            { name_buffers[3], name_buffers[4], INNER, INNER },
            { name_buffers[5], name_buffers[6], INNER, (uint64_t)INNER * 2 },
        },
        .source_count = 3,
    };

    snprintf(name_buffers[7], 160, "%sattn.out_proj.weight", checkpoint_prefix);
    snprintf(name_buffers[8], 160, "%sattn.to_out.0.lora_A.default.weight", lora_prefix);
    snprintf(name_buffers[9], 160, "%sattn.to_out.0.lora_B.default.weight", lora_prefix);
    out[H3_LORA_OUT] = (h3_lora_projection){
        .checkpoint_name = name_buffers[7],
        .rows = HIDDEN, .columns = INNER,
        .sources = { { name_buffers[8], name_buffers[9], HIDDEN, 0 } },
        .source_count = 1,
    };

    snprintf(name_buffers[10], 160, "%smlp.fc1.weight", checkpoint_prefix);
    snprintf(name_buffers[11], 160, "%sff.net.0.proj.lora_A.default.weight", lora_prefix);
    snprintf(name_buffers[12], 160, "%sff.net.0.proj.lora_B.default.weight", lora_prefix);
    out[H3_LORA_FC1] = (h3_lora_projection){
        .checkpoint_name = name_buffers[10],
        .rows = (uint64_t)FFN * 2, .columns = HIDDEN,
        .sources = { { name_buffers[11], name_buffers[12], (uint64_t)FFN * 2, 0 } },
        .source_count = 1,
    };

    snprintf(name_buffers[13], 160, "%smlp.fc2.weight", checkpoint_prefix);
    snprintf(name_buffers[14], 160, "%sff.net.2.lora_A.default.weight", lora_prefix);
    snprintf(name_buffers[15], 160, "%sff.net.2.lora_B.default.weight", lora_prefix);
    out[H3_LORA_FC2] = (h3_lora_projection){
        .checkpoint_name = name_buffers[13],
        .rows = HIDDEN, .columns = FFN,
        .sources = { { name_buffers[14], name_buffers[15], HIDDEN, 0 } },
        .source_count = 1,
    };
}
