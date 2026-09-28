// Experimental, opt-in attention backend bridging to liuliu/ccv's Metal
// int8 NAX attention kernel (NAInt8AttentionKernel, via the public
// ccv_nnc_mfa_encode_attention C API). See SPEEDUP_ROADMAP.md item 5 and
// tools/ccv_eval/README.md for the numbers this was validated against and
// how to reproduce them.
//
// Only built and linked when the Makefile has CCV_DIR set (see Makefile) -
// every other binary that links libh3.a is completely unaffected, since
// h3_gpu_ccv_dense_attention_bf16 (h3_gpu.h/h3_gpu.m) already fails cleanly
// on its own when nothing registers a real implementation. Linking this
// file causes that registration to happen automatically, via the
// constructor at the bottom of this file - no other code needs to change
// to enable or disable this backend, only the build (see Makefile,
// CCV_DIR).
//
// Bridging note: metal-cpp's MTL::Device/MTL::Buffer/MTL::CommandBuffer are
// thin wrappers holding exactly the same pointer value as the underlying
// Objective-C id<MTLDevice>/id<MTLBuffer>/id<MTLCommandBuffer> object - this
// is metal-cpp's documented interop mechanism, not a private assumption
// made here. h3_gpu_raw_device()/h3_gpu_raw_buffer()/
// h3_gpu_raw_command_buffer() (h3_gpu.m) hand out that same pointer value
// as void*, unretained (the underlying h3_gpu/h3_gpu_tensor keeps it
// alive); casting it straight to the matching mtl_*_t* here is exactly
// that interop, not a new bridge.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "h3_gpu.h"
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/ccv_nnc_mfa_attention.hpp"

namespace {

struct ccv_attention_state {
    ccv_nnc_mfa_context_t *context = nullptr;
    void *device_key = nullptr;
    id<MTLBuffer> q_f16 = nil;
    id<MTLBuffer> k_f16 = nil;
    id<MTLBuffer> v_f16 = nil;
    id<MTLBuffer> out_f16 = nil;
    size_t capacity_elements = 0;
};

/* Single-generation, single-threaded assumption matches how h3_dit.c drives
 * one h3_gpu at a time - not safe to share across concurrent generations. */
ccv_attention_state g_state;

bool ensure_scratch(id<MTLDevice> device, size_t elements) {
    if (elements <= g_state.capacity_elements && g_state.q_f16) return true;
    size_t bytes = elements * sizeof(uint16_t);
    MTLResourceOptions options = MTLResourceStorageModePrivate;
    g_state.q_f16 = [device newBufferWithLength:bytes options:options];
    g_state.k_f16 = [device newBufferWithLength:bytes options:options];
    g_state.v_f16 = [device newBufferWithLength:bytes options:options];
    g_state.out_f16 = [device newBufferWithLength:bytes options:options];
    if (!g_state.q_f16 || !g_state.k_f16 || !g_state.v_f16 || !g_state.out_f16) {
        g_state.capacity_elements = 0;
        return false;
    }
    g_state.capacity_elements = elements;
    return true;
}

int ccv_dense_attention_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *query,
                             const h3_gpu_tensor *key,
                             const h3_gpu_tensor *value, uint32_t rows,
                             uint32_t heads, uint32_t head_dim, float scale) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)h3_gpu_raw_device(gpu);
    if (!device) {
        h3_gpu_report_error(gpu, "ccv backend: no Metal device");
        return 0;
    }
    /* Read this before anything else touches it - same requirement as the
     * debug capture hook in h3_dit.c, and for the same reason (this
     * engine's default int8 QKV path leaves query/key/value head-major for
     * the immediately-following attention call to consume). */
    int head_major = h3_gpu_head_major_sdpa_inputs(gpu);

    void *raw_command_buffer = h3_gpu_raw_command_buffer(gpu);
    if (!raw_command_buffer) return 0; // error already set

    if (g_state.device_key != (__bridge void *)device) {
        if (g_state.context) ccv_nnc_deinit_mfa_context(g_state.context);
        g_state.context = ccv_nnc_init_mfa_context(
            (mtl_device_t *)(__bridge void *)device);
        g_state.device_key = (__bridge void *)device;
        g_state.q_f16 = g_state.k_f16 = g_state.v_f16 = g_state.out_f16 = nil;
        g_state.capacity_elements = 0;
    }
    if (!g_state.context ||
        !ccv_nnc_mfa_context_supported(g_state.context) ||
        !ccv_nnc_mfa_has_neural_accelerators(g_state.context)) {
        h3_gpu_report_error(gpu, "ccv backend: context unsupported or this "
                            "device lacks neural matrix accelerators");
        return 0;
    }

    size_t count = (size_t)rows * heads * head_dim;
    if (!ensure_scratch(device, count)) {
        h3_gpu_report_error(gpu, "ccv backend: scratch buffer allocation "
                            "failed");
        return 0;
    }

    if (!h3_gpu_ccv_cast_bf16_to_f16(gpu, (__bridge void *)g_state.q_f16,
                                    query, rows, heads, head_dim,
                                    head_major) ||
        !h3_gpu_ccv_cast_bf16_to_f16(gpu, (__bridge void *)g_state.k_f16,
                                    key, rows, heads, head_dim,
                                    head_major) ||
        !h3_gpu_ccv_cast_bf16_to_f16(gpu, (__bridge void *)g_state.v_f16,
                                    value, rows, heads, head_dim,
                                    head_major))
        return 0; // h3_gpu_ccv_cast_bf16_to_f16 already set the error

    ccv_nnc_mfa_attention_params_t params = {};
    params.data_type = MTL::DataTypeHalf;
    params.R = rows; params.C = rows; params.Hq = heads; params.Hk = heads;
    params.D = head_dim; params.output_rows = rows;
    /* K_trans=1 matches the exact configuration validated against this
     * engine's real production SDPA output in tools/ccv_eval/ (52-62 dB
     * PSNR across three real blocks) - do not change without re-running
     * that validation, since this parameter's effect on the quantized NAX
     * kernel path specifically was not independently confirmed (flipping
     * it made no measurable difference in that same validation, which is
     * itself worth re-checking before relying on it either way). */
    params.K_trans = 1;
    params.alpha = scale;
    params.use_neural_accelerators = 1;
    params.use_quantized_attention = 1;

    auto command_buffer = (mtl_command_buffer_t *)raw_command_buffer;
    auto batch = ccv_nnc_start_command_batch_from_command_buffer(
        command_buffer, /*commit_on_finish=*/0);
    /* ccv_nnc_mfa_encode_attention reads a fixed 10-slot tensor/offset
     * array internally (matching every call site in ccv's own bin/mfa
     * tools) - only the first 4 slots (Q, K, V, O) are meaningful for this
     * plain dense/no-mask/no-bias configuration, but passing a
     * shorter array is undefined behavior (a real SIGBUS was hit and
     * root-caused here before this comment was added: a 4-element array
     * reads past its end as soon as ccv's code touches slot 4+). */
    mtl_buffer_t *tensors[10] = {
        (mtl_buffer_t *)(__bridge void *)g_state.q_f16,
        (mtl_buffer_t *)(__bridge void *)g_state.k_f16,
        (mtl_buffer_t *)(__bridge void *)g_state.v_f16,
        (mtl_buffer_t *)(__bridge void *)g_state.out_f16,
    };
    size_t offsets[10] = {};
    ccv_nnc_mfa_encode_attention(g_state.context, params, batch, tensors,
                                offsets);
    ccv_nnc_finish_command_batch(batch);

    if (!h3_gpu_ccv_cast_f16_to_bf16(gpu, output,
                                    (__bridge void *)g_state.out_f16,
                                    (uint32_t)count))
        return 0;

    h3_gpu_set_head_major_sdpa_inputs(gpu, 0);
    h3_gpu_note_ccv_attention_dispatch(gpu);
    return 1;
}

} // namespace

/* Not a constructor: a static archive only links in a member that resolves
 * some other object's undefined symbol, so a self-registering constructor
 * with nothing else in this translation unit referenced would be invisible
 * to the linker and silently dropped. Instead, this file is kept OUT of
 * libh3.a entirely and linked as a loose object file whenever CCV_DIR is
 * set (see Makefile, Package.swift) - a loose .o on the link line is
 * always included, so this runs unconditionally in that build. */
__attribute__((constructor))
static void h3_gpu_ccv_attention_register(void) {
    h3_gpu_register_ccv_dense_attention(&ccv_dense_attention_bf16);
}
