// Replays one captured attention call (H3_DUMP_ATTENTION_QKV:
// prefix.{q,k,v,out}.bin, FP16 row-major [T, H, 128]) through ccv's int8
// kernel twice on identical inputs - the FP16 row-major bridge setup
// (H3_ATTENTION_BACKEND=ccv_dense) and the BF16 head-major batch=H setup
// (H3_CCV_DIRECT=1) - and diffs both against the captured production
// output: overall, per head, and over the trailing rows.
//
// Build (from ccv/bin/mfa, same flags as h3_real_attention_compare.cpp):
//   clang h3_direct_replay.cpp -o h3_direct_replay.o -c -std=c++17 -O3 \
//     -I"../.." -I"../../lib" -fblocks -D HAVE_CBLAS -D HAVE_PTHREAD \
//     -D HAVE_ACCELERATE_FRAMEWORK -D USE_DISPATCH -D HAVE_MPS
//   clang -o h3_direct_replay h3_direct_replay.o ../../lib/libccv.a -lm \
//     -lblas -lpthread -framework Accelerate -framework Foundation \
//     -framework CoreML -framework IOSurface -framework Metal -lc++ \
//     -framework QuartzCore
// Usage: ./h3_direct_replay T H prefix [tail_rows]
// Needs ccv with tools/ccv_eval/ccv-na-int8-bf16-lse-store.patch.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nnc/mfa/ccv_nnc_mfa.hpp"

static float f16_to_f32(uint16_t h) { __fp16 v; memcpy(&v, &h, 2); return (float)v; }
static float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

struct Stats { double err = 0, ref = 0, maxabs = 0; };
static void add(Stats &s, float ref, float x) {
  double d = (double)x - ref; s.err += d * d; s.ref += (double)ref * ref;
  if (fabs(d) > s.maxabs) s.maxabs = fabs(d);
}
static double rel(const Stats &s) { return s.ref > 0 ? sqrt(s.err / s.ref) : 0; }

static bool run(MTL::CommandQueue *queue, ccv_nnc_mfa_context_t *context,
                ccv_nnc_mfa_attention_params_t params, MTL::Buffer **bufs) {
  auto cb = queue->commandBuffer();
  auto batch = ccv_nnc_start_command_batch_from_command_buffer(cb, 0);
  MTL::Buffer *tensors[10] = { bufs[0], bufs[1], bufs[2], bufs[3] };
  size_t offsets[10] = {};
  ccv_nnc_mfa_encode_attention(context, params, batch, tensors, offsets);
  ccv_nnc_finish_command_batch(batch);
  cb->commit(); cb->waitUntilCompleted();
  if (cb->status() == MTL::CommandBufferStatusError) {
    fprintf(stderr, "%s\n", cb->error()->localizedDescription()->utf8String());
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s T H prefix [tail_rows]\n", argv[0]); return 2; }
  const uint32_t T = atoi(argv[1]), H = atoi(argv[2]), D = 128;
  const std::string prefix = argv[3];
  const uint32_t tail = argc > 4 ? atoi(argv[4]) : 128;
  const float scale = 1.0f / sqrtf((float)D);
  const size_t count = (size_t)T * H * D, bytes = count * 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (!ccv_nnc_mfa_context_supported(context) || !ccv_nnc_mfa_has_neural_accelerators(context)) {
    fprintf(stderr, "needs neural matrix accelerators\n"); return 2;
  }
  auto queue = NS::TransferPtr(device->newCommandQueue());

  std::vector<uint16_t> host[4]; // q k v out, FP16 row-major
  const char *suffix[4] = { "q", "k", "v", "out" };
  for (int i = 0; i < 4; ++i) {
    host[i].resize(count);
    std::ifstream file(prefix + "." + suffix[i] + ".bin", std::ios::binary);
    file.read((char *)host[i].data(), bytes);
    if ((size_t)file.gcount() != bytes) { fprintf(stderr, "size mismatch: %s\n", suffix[i]); return 2; }
  }

  // FP16 row-major, as the bridge feeds ccv.
  MTL::Buffer *fp16[4];
  for (int i = 0; i < 4; ++i) {
    fp16[i] = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (i < 3) memcpy(fp16[i]->contents(), host[i].data(), bytes);
  }
  // BF16 head-major, as the int8 QKV path leaves them. The captures are
  // BF16 values widened to FP16, so narrowing back must be exact.
  MTL::Buffer *bf16[4];
  size_t inexact = 0;
  for (int i = 0; i < 4; ++i) {
    bf16[i] = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (i == 3) continue;
    uint16_t *dst = (uint16_t *)bf16[i]->contents();
    for (uint32_t r = 0; r < T; ++r)
      for (uint32_t h = 0; h < H; ++h)
        for (uint32_t d = 0; d < D; ++d) {
          float f = f16_to_f32(host[i][((size_t)r * H + h) * D + d]);
          uint32_t u; memcpy(&u, &f, 4);
          if (u & 0xFFFF) inexact++;
          dst[((size_t)h * T + r) * D + d] = (uint16_t)(u >> 16);
        }
  }

  ccv_nnc_mfa_attention_params_t p = {};
  p.R = T; p.C = T; p.D = D; p.output_rows = T; p.K_trans = 1; p.alpha = scale;
  p.use_neural_accelerators = 1; p.use_quantized_attention = 1;
  ccv_nnc_mfa_attention_params_t bridge = p;
  bridge.data_type = MTL::DataTypeHalf; bridge.Hq = H; bridge.Hk = H;
  ccv_nnc_mfa_attention_params_t direct = p;
  direct.data_type = MTL::DataTypeBFloat; direct.Hq = 1; direct.Hk = 1;
  direct.batched = 1; direct.batch_dims_q[0] = H;
  if (!run(queue.get(), context, bridge, fp16) || !run(queue.get(), context, direct, bf16)) return 1;

  const uint16_t *ob = (const uint16_t *)fp16[3]->contents();
  const uint16_t *od = (const uint16_t *)bf16[3]->contents();
  Stats all_b, all_d, diff, tail_b, tail_d;
  std::vector<Stats> head_b(H), head_d(H);
  for (uint32_t r = 0; r < T; ++r)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t d = 0; d < D; ++d) {
        size_t row_major = ((size_t)r * H + h) * D + d;
        float ref = f16_to_f32(host[3][row_major]);
        float b = f16_to_f32(ob[row_major]);
        float x = bf16_to_f32(od[((size_t)h * T + r) * D + d]);
        add(all_b, ref, b); add(all_d, ref, x); add(diff, b, x);
        add(head_b[h], ref, b); add(head_d[h], ref, x);
        if (r >= T - tail) { add(tail_b, ref, b); add(tail_d, ref, x); }
      }
  double worst_b = 0, worst_d = 0; uint32_t wb = 0, wd = 0;
  for (uint32_t h = 0; h < H; ++h) {
    if (rel(head_b[h]) > worst_b) { worst_b = rel(head_b[h]); wb = h; }
    if (rel(head_d[h]) > worst_d) { worst_d = rel(head_d[h]); wd = h; }
  }
  printf("T=%u H=%u inexact_bf16_inputs=%zu\n", T, H, inexact);
  printf("bridge vs prod: rel_l2=%.6f max_abs=%.4f | worst head %u rel_l2=%.6f | last %u rows rel_l2=%.6f\n",
         rel(all_b), all_b.maxabs, wb, worst_b, tail, rel(tail_b));
  printf("direct vs prod: rel_l2=%.6f max_abs=%.4f | worst head %u rel_l2=%.6f | last %u rows rel_l2=%.6f\n",
         rel(all_d), all_d.maxabs, wd, worst_d, tail, rel(tail_d));
  printf("direct vs bridge: rel_l2=%.6f max_abs=%.4f\n", rel(diff), diff.maxabs);
  ccv_nnc_deinit_mfa_context(context);
  return 0;
}
