// Build: cd bin/mfa && clang h3_real_attention_compare.cpp -o h3_real_attention_compare \
//   -std=c++17 -O3 -I"../.." -I"../../lib" -fblocks -D HAVE_CBLAS -D HAVE_PTHREAD \
//   -D HAVE_ACCELERATE_FRAMEWORK -D USE_DISPATCH -D HAVE_MPS -I/usr/local/include && \
//   ../../lib/libccv.a -L/usr/local/lib -lm -lblas -lpthread -framework Accelerate \
//   -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//   -framework Foundation -framework CoreVideo -framework CoreML -framework IOSurface \
//   -framework Metal -lc++ -framework QuartzCore -o h3_real_attention_compare
//
// Usage: ./h3_real_attention_compare [T] [H] [raw-prefix] [begin] [scale]
// Loads real captured FP16 Q/K/V (prefix.{q,k,v}.bin), runs both ccv's plain
// dense int8 attention and its Sol sparse-routed attention on them (same
// setup na_int8_sol_attention_bench uses), and writes each result to
// prefix.ccv_dense.bin / prefix.ccv_sparse.bin for a direct diff against
// this app's own real captured production output (prefix.out.bin), not
// against any reference ccv computes internally.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include "nnc/mfa/ccv_nnc_mfa.hpp"

int main(int argc, char** argv)
{
  const uint32_t T = argc > 1 ? atoi(argv[1]) : 3211;
  const uint32_t H = argc > 2 ? atoi(argv[2]) : 56;
  const char* raw = argc > 3 ? argv[3] : nullptr;
  const uint32_t begin = argc > 4 ? atoi(argv[4]) : std::min(470u, T);
  const float scale = argc > 5 ? atof(argv[5]) : 0.08838834765f;
  if (!raw) { fprintf(stderr, "raw-prefix required.\n"); return 2; }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) return 2;
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (!ccv_nnc_mfa_context_supported(context) || !ccv_nnc_mfa_has_neural_accelerators(context)) {
    fprintf(stderr, "This tool requires neural matrix accelerators.\n"); return 2;
  }
  auto queue = NS::TransferPtr(device->newCommandQueue());
  const size_t count = size_t(T) * H * 128, bytes = count * 2;
  std::array<NS::SharedPtr<MTL::Buffer>, 5> buffers; // q k v dense_out sparse_out
  const char* suffix[] = { "q", "k", "v" };
  for (int i = 0; i < 5; ++i) {
    buffers[i] = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!buffers[i]) { fprintf(stderr, "Tensor allocation failed.\n"); return 2; }
    if (i < 3) {
      std::ifstream file(std::string(raw) + "." + suffix[i] + ".bin", std::ios::binary);
      file.read(static_cast<char*>(buffers[i]->contents()), bytes);
      if (size_t(file.gcount()) != bytes) {
        fprintf(stderr, "Capture size mismatch for %s (wanted %zu bytes).\n", suffix[i], bytes);
        return 2;
      }
    }
  }
  ccv_nnc_mfa_attention_params_t native = {};
  native.data_type = MTL::DataTypeHalf;
  native.R = T; native.C = T; native.Hq = H; native.Hk = H; native.D = 128;
  native.output_rows = T; native.K_trans = getenv("H3_KTRANS") ? atoi(getenv("H3_KTRANS")) : 1;
  native.alpha = scale;
  native.use_neural_accelerators = 1; native.use_quantized_attention = 1;
  ccv_nnc_mfa_prepare_attention(context, native);
  ccv_nnc_mfa_sol_attention_params_t sol = {};
  sol.N = 1; sol.T = T; sol.H = H; sol.block_size = 64; sol.query_block_size = 64;
  sol.approximation_start = begin; sol.approximation_end = T;
  sol.scale = scale; sol.tau = 0.5f; sol.use_neural_accelerators = 1; sol.local_block_radius = 1;

  {
    auto cb = queue->commandBuffer();
    auto batch = ccv_nnc_start_command_batch_from_command_buffer(cb, 0);
    MTL::Buffer* tensors[10] = { buffers[0].get(), buffers[1].get(), buffers[2].get(), buffers[3].get() };
    size_t offsets[10] = {};
    ccv_nnc_mfa_encode_attention(context, native, batch, tensors, offsets);
    ccv_nnc_finish_command_batch(batch);
    cb->commit(); cb->waitUntilCompleted();
    if (cb->status() == MTL::CommandBufferStatusError) {
      fprintf(stderr, "dense: %s\n", cb->error()->localizedDescription()->utf8String()); return 1;
    }
  }
  {
    auto cb = queue->commandBuffer();
    auto batch = ccv_nnc_start_command_batch_from_command_buffer(cb, 0);
    MTL::Buffer* tensors[10] = { buffers[0].get(), buffers[1].get(), buffers[2].get(), buffers[4].get() };
    size_t offsets[10] = {};
    ccv_nnc_mfa_encode_sol_attention(context, sol, batch, tensors, offsets);
    ccv_nnc_finish_command_batch(batch);
    cb->commit(); cb->waitUntilCompleted();
    if (cb->status() == MTL::CommandBufferStatusError) {
      fprintf(stderr, "sparse: %s\n", cb->error()->localizedDescription()->utf8String()); return 1;
    }
  }
  const char* out_suffix[] = { "ccv_dense", "ccv_sparse" };
  for (int i = 0; i < 2; ++i) {
    std::ofstream file(std::string(raw) + "." + out_suffix[i] + ".bin", std::ios::binary);
    file.write(static_cast<const char*>(buffers[3 + i]->contents()), bytes);
  }
  printf("wrote %s.{ccv_dense,ccv_sparse}.bin (T=%u H=%u scale=%g begin=%u)\n", raw, T, H, scale, begin);
  ccv_nnc_deinit_mfa_context(context);
  return 0;
}
