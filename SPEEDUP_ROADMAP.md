# h3c-app / h3.c speedup roadmap

Backlog of T2V/Ref2V speedup work, agreed 2026-09-27, with the analysis and
measurements behind each item. This is a working engineering log, not
polished documentation — it records what was tried, what worked, what
didn't, and why, so the reasoning doesn't have to be redone.

Agreed with the user 2026-09-27 after reviewing techno-edge's MiniMax-H3
speedup article (turbo steps, int8 VAE, sparse attention Sol-Attn/SLA/VSA)
and Sol-Attn. Item #3 (int8 video VAE) was implemented first (2026-09-27):
the 36 block linears of the F32 transformer VAE decoder run on the M5 int8
kernels, `H3_VAE_INT8=0` opts out. Measured 512x512/124 frames/4 steps: VAE
decode 108.0s → 37.0s, VAE GPU peak 9.45 → 2.83 GiB, end-to-end 250–270s →
186s; PSNR vs F32 46.4 dB (worst frame 44.9), visually identical.

Items #1 and #2 were done next (2026-09-27, for v0.2.0): SpeedMode
quality/fast/fastest = exact / 45 layers + core reuse min(4, steps/5) / +
token reduction. Measured 512x512/39fr/20 steps: 232.6s / 82.3s / 73.7s.
Stacking 40 layers + core 6 + token reduction + int8 row FC2 (61.0s) smeared
the subject; int8 row FC2 alone gave no speedup, so it isn't used. PSNR vs
exact is ~12 dB for every approximation (trajectory changes composition) —
judge visually. LoRA entries got `recommendedSteps` (auto-detected from
"4step" names) driving steps.

Remaining backlog, in the recommended order:

## 1. Expose already-implemented engine speed knobs in the app/API

A speed preset: `dit_layers` 45/40, `core_reuse` 4/6, `token_reduction`,
`use_int8_row_fc2` (h3.h:77-92 calls these "validated"). `core_reuse>1` and
`denoise_reuse>1` are mutually exclusive (h3.c check). H3Engine.swift didn't
pass `token_reduction`/`use_int8_row_fc2` at all before this. Small effort.

**Status: done** — see item summary above (SpeedMode quality/fast/fastest).

## 2. Turbo LoRA step presets

4/8 steps — the user has FL2V turbo 4step v1.1/v1.2, 8step v1.0 and
`minimax_h3_ref2v_turbo_4step_v0.1` in `~/Documents/work/h3c/lora`. Add a
"recommended steps" field per LoRA entry. H3 is CFG-free already.

**Status: done** — see item summary above.

## 3. int8 video VAE

**Status: done** — see the top of this document.

## 4. Unfused LoRA

**Status: done** 2026-09-28, pushed as `fe0ad33`. GPU weight patching with
stochastic rounding, all compute modes, stacking; no fused caches anymore.
See the multi-LoRA-stacking work (verified QKV/FC1 layout facts, the
stochastic-rounding requirement, costs, design) recorded separately.

## 5. Sol-Attn-style block-sparse attention

arXiv 2607.24027. The DiT joint attention is MPSGraph's built-in SDPA
(`h3_gpu.m` `h3_gpu_sdpa`) — so this first needs a custom Metal
flash-attention kernel (reference: philipturner/metal-flash-attention, MIT,
same lineage as ccv/Draw Things). Algorithm: 64-token Q/KV blocks, proxy
score = mean(Q_blk)·mean(K_blk), per-query-block threshold
tau = mu + beta·sigma, skipped blocks approximated by a zeroth-order term,
dense for the first 20% of steps and the first layer (~85% sparsity, 2-3x
end-to-end on NVIDIA). Large effort (custom Metal flash-attention + sparse
selection kernel from scratch, real correctness risk).

**Status: in progress, extensively prototyped, not yet integrated.** This
item has had by far the most investigation. Summary of each round below.

### Micro-benchmark (2026-09-28)

Gating check the roadmap itself calls for, before writing the kernel: a
standalone tool linking `libh3.a`'s public `h3_gpu.h`, timing
`h3_gpu_sdpa_bf16` against the four block linears (qkv/out/fc1/fc2) in
isolation at real sequence lengths read off actual H3Spike runs
(`H3_DEBUG_ROWS` temporarily added to `h3_dit.c`, then reverted). Critically,
comparing SDPA against int8-quantized linears (`h3_gpu_linear_int8_bf16`,
matching the real M5 default fast path — `dit->int8_qkv`/`int8_mlp`/
`int8_attention_out`) rather than plain BF16 linears, since quantization
speeds up the linears but not SDPA, raising attention's *share* of the
faster path.

Measured attention_share of one DiT block (M5, int8 linears):

| Resolution / frames | sequence length | attention share |
|---|---|---|
| 512² / 39fr (app's short/default clip) | 3204 | 17.8% |
| 384² / 124fr | 5744 | 28.9% |
| 768² / 124fr | 21728 | 65.7% |

Confirms the FLOPs-estimate direction (attention share grows with sequence
length, dominates for long/large clips) but the short-clip number is lower
than the earlier ~25% FLOPs estimate. Ceiling from Sol-Attn's own ~85%
sparsity: ~15% total block speedup at 512²/39fr vs ~56% at 768²/124fr — so
this item has little payoff for the short 4-step Turbo LoRA clips the app
now leans on, and a lot for long/large T2V and Ref2V.

### The real bar turned out to be higher than expected

The user chose to proceed with #5 for long/large clips. Immediately found
something that changes the difficulty estimate a lot: `h3_gpu_sdpa`
(`h3_gpu.m` `h3_gpu_sdpa_graph`) is **not** a naive/manual MPSGraph
composition — it's MPSGraph's own native
`scaledDotProductAttentionWithQueryTensor:...` op, i.e. Apple's
vendor-optimized fused attention kernel. Measured its achieved throughput
against the BF16 dense linears (same dtype, same GPU, from the microbench
above): SDPA runs at 10.1–12.7 TFLOP/s vs 13.6–14.2 TFLOP/s for
`h3_gpu_linear_bf16` — i.e. **~75–90% of the practical BF16 matmul ceiling
on this M5**. There is no "free" dense-kernel win sitting there; matching
vendor SDPA at all, before any sparsity, is the real bar.

### v1: naive scalar kernel — 63x slower

Built and correctness-tested a first custom tiled online-softmax Metal
kernel (`h3_flash_attn_dense_bf16`, scratch only) matching
`h3_gpu_sdpa_bf16`'s row-major `[seq, heads, head_dim]` layout: `TILE_Q=32`
(one thread per query row, one simdgroup/threadgroup), `TILE_KV=16`,
threadgroup memory for K/V tiles + output accumulator (24 KiB). Verified
correct against a CPU (Accelerate `cblas_sgemm`) double-matmul softmax
reference at real scale (seq=3204, heads=56): `max_abs_err` 8e-6,
`max_rel_err` 0.37%, consistent with BF16 rounding — the tiling/indexing
math is right.

But its **measured performance was 1477 ms for the same 56-head/seq=3204
attention MPSGraph does in 23.3 ms — about 63x slower**, because it's pure
per-thread scalar dot products (32 threads/threadgroup, nowhere near this
GPU's SIMD/matrix-unit width), not `simdgroup_matrix` or the Metal 4 tensor
primitives (`metal_tensor`/`MetalPerformancePrimitives`) already used
elsewhere in this codebase for the NAX kernels.

**Lesson:** a kernel worth shipping needs matrix-unit-based tiling
(`simdgroup_matrix` or Metal 4 tensor ops) to get anywhere near vendor
density before sparsity can help at all — even Sol-Attn's ~85% skip ratio
only pays off once the per-block cost of the *computed* blocks is within a
small factor of vendor SDPA, not 63x off.

### v2: matrix-unit dense attention — down to 2.1x

Rebuilt as a 3-pass design instead of v1's fused per-thread scalar loop,
reusing `h3_linear_bf16_nax_r128` verbatim (the same Metal 4 `matmul2d`
TensorOps kernel this codebase already uses for the DiT's own projections
and for `h3_gpu_lora_delta_bf16`'s validated `b[rows,K] @ at[cols,K]^T` GEMM)
for both QK^T and P@V, plus one new small row-softmax kernel
(`h3_softmax_rows_bf16`) and a transpose kernel for V.

Confirmed empirically (`h3_linear_bf16_nax_r128` has no public API
documentation — `MetalPerformancePrimitives.h` is just a 12-line stub) that
its buffer layout is: `input`/`weight` row-major with the reduction
dimension (`input_dim`) contiguous, `output` row-major with `output_dim`
contiguous — i.e. exactly this project's natural `[rows, head_dim]`-
contiguous Q/K layout, no transpose needed for QK^T; P@V needs V
pre-transposed to `[head_dim, kv_seq]`. Constraint confirmed from
`h3_gpu_lora_delta_bf16`'s own guard: the N/`output_dim` tile must be an
exact multiple of 64 (kv_seq padded up, padding masked to
-inf-equivalent in the softmax kernel via a `valid_columns` cutoff so it
contributes exactly zero); the M/`rows` tile needs no padding (hardware
handles the ragged last row-tile, matching the kernel's own comment).

**Correctness:** validated at real scale (seq=3204, heads=56, head 0
spot-checked against the CPU reference) — `mean_abs_err` 2.1e-6,
`max_abs_err` 1.6e-5 (BF16-precision floor, same order as v1); the one 8%
"max relative error" reading is a near-zero-denominator artifact (occurred
at a true value of 0.000001), not a bug.

**Performance**, iterated down in two steps from v1's 63x:

- Naive per-head dispatch (56 heads x 4 kernels, commit+wait every head):
  78.1 ms (of which 13.8 ms is pure CPU repack-copy, a test-harness
  artifact de-interleaving per-head Q/K/V out of the model's
  `[seq,heads,head_dim]` layout into tightly-packed buffers the kernel
  needs).
- Batching all 56 heads into ONE command buffer (single commit+wait)
  instead of 56: **49.3 ms** — confirms most of the per-head cost was
  command-buffer submission/sync overhead, not compute.
- vs MPSGraph's native SDPA: 23.3 ms. **Gap is ~2.1x, down from 63x.**

This changes the outlook a lot: Sol-Attn's own ~85% block-skip ratio would
already net a real win at a 2.1x per-tile efficiency gap (0.15 × 2.1 ≈ 0.32
of vendor's dense time — still a ~3x reduction in attention time), and the
remaining gap has concrete, not-yet-tried levers: (a) a small Metal repack
kernel (or a strided tensor view straight into the interleaved layout,
avoiding repack entirely) instead of CPU memcpy per head, (b) fewer/larger
dispatches (batch heads within one matmul call rather than 4 dispatches x
56 heads), (c) fusing softmax into the QK^T or P@V pass to cut a kernel
launch and a global-memory round trip for `scores`.

### v3: QK^T block skipping — a real, validated win (1.19–1.26x)

Added mean(Q_blk)·mean(K_blk) proxy scoring, per-row-tile threshold
tau = mu + beta·sigma, and a diagonal±1 anchor (always-kept local block) on
top of v2, skipping the masked QK^T tiles' `matmul2d.run()` entirely
(`h3_linear_bf16_nax_r128_masked` checks a per-(row_tile,col_tile) byte mask
before running; `h3_softmax_rows_masked_bf16` treats masked 64-column
groups as excluded, same zero-contribution mechanism as the seq-padding
mask in v2). PV was left fully dense in this prototype.

The first attempt was a real lesson, not just an implementation bug:
computed mean(Q_tile) and mean(K_tile) fresh inside every
(row_tile,col_tile) threadgroup pair — O(row_tiles·col_tiles·tile_rows·
head_dim) instead of the O(seq·head_dim) it should be, since each tile's
mean only depends on one axis. Result: sparse was *slower* than dense
(94.6ms vs 77.7ms, then 53.8ms vs 49.5ms after batching command buffers)
despite skipping ~72% of QK^T tiles — the naive scalar proxy-scoring pass
cost more wall-clock than the tensor-core matmul it was gating.

> **General lesson for this hardware:** because the dense matmul is now
> genuinely fast (matrix-unit tiles, ~2x off vendor), any gating/scoring
> mechanism has to be comparably cheap (linear in seq, reuse partial results
> across tiles) or it eats the entire win and then some.

Fixed by splitting into `h3_block_means_bf16` (reduce each axis's tile-means
once, O(seq·head_dim)) and `h3_block_dot_means_bf16` (a tiny
O(row_tiles·col_tiles·head_dim) dot pass over the small mean buffers, e.g.
26×51×128 ≈ 170k multiply-adds — trivial next to a QK^T tile).

After the fix, measured on structured-synthetic Q/K (8 random "cluster"
directions assigned to 64-row chunks + noise, since iid random data has no
exploitable redundancy for a fair sparsity test), M5, seq=3204/heads=56,
batched command buffers both phases:

| beta | tiles kept | speedup (QK^T-sparse, PV still dense) |
|---|---|---|
| 0.0 | 29.1% | 1.20x |
| 0.3 | 27.6% | 1.20x |
| 0.6 | 27.5% | 1.22x |
| 0.9 | 27.5% | 1.20x |
| 1.5 | 18.1% | 1.26x |

Speedup plateaus well below the skip ratio (e.g. 82% skipped → only 26%
faster) — expected Amdahl's-law ceiling since PV (~half the FLOPs) stays
fully dense and fixed overhead (proxy scoring, softmax, dispatch) doesn't
shrink with the skip ratio. **This is real, validated evidence the
mechanism works and nets a genuine wall-clock win once the scoring pass
itself is properly cheap** — not just a paper estimate anymore.

Quality signal from this run is **not** trustworthy yet and needs real
model data, not synthetic: `mean_abs_err` (0.0014) exceeded `mean|reference|`
(0.00076) at beta=0.6, and the synthetic generator's 64-row "cluster"
granularity doesn't match the 128-row Q-tile granularity used for
scoring/masking (each Q tile spans two clusters, diluting its mean) —
likely explains at least part of the poor number, but this needs checking
against real DiT block activations before drawing any conclusion about
real video quality impact.

### v4/v5: PV sparsity — negative result

Tried to sparsify PV too. Net result is negative as implemented, a real and
useful finding, not yet a win. Since a single `h3_linear_bf16_nax_r128` call
always reduces over its full declared K extent internally (no
documented/known-safe partial-K slicing), avoided that risk by **compacting**
each row-tile's surviving 64-wide KV blocks into a tightly packed buffer
first (two new gather kernels: `h3_gather_column_blocks_bf16` for P/scores —
the KV axis is the contiguous column axis — and `h3_gather_row_blocks_bf16`
for V — the KV axis is the outer row axis), then reusing the existing dense
`h3_linear_bf16_nax_r128` with a genuinely smaller K = kept_count×64.

Correct (same 0.0014/0.009 error figures as v3, since QK^T-side masking
dominates the error signal), but per (head, row_tile) this needs 4 small
encoder calls (gather-P, gather-V, transpose, matmul) instead of dense PV's
2 (transpose, matmul) — 26 row_tiles × 56 heads × 4 = **5824 tiny
dispatches** vs 112.

Measured: QK-only sparse still 1.20x (40.9ms, matches earlier); adding PV
sparsity on top made it **slower than dense** — 74.5ms (0.66x) vs dense's
49.1ms. The delta (33.6ms over ~5700 extra encoder calls ≈ 5.9us/call) lines
up exactly with CPU-side encode+launch overhead, confirming it's a
dispatch-count problem, not a logic bug.

> **Same lesson as the QK^T scoring pass originally hit:** on this hardware,
> with dense tensor-core matmul already this fast, sparsity only pays off
> when the overhead per skipped-or-kept unit is kept very low — many small
> dispatches lose to a few big ones even when they do genuinely less total
> compute.

### v6: batched PV sparsity — improved, still a net loss

Tried the batched-across-row_tiles fix — dispatch count came down as
predicted, but PV sparsity still doesn't beat dense. Added
`h3_gather_column_blocks_batched_bf16` / `h3_gather_row_blocks_batched_bf16`
(grid now spans all of a head's row_tiles via a 3rd/2nd grid dimension,
using the padding sentinel `0xFFFF` → exact zero contribution),
`h3_transpose_batched_bf16` (transposes each row_tile's own sub-block
independently, row_tile kept as an outer stride — a single flat transpose
of the whole per-head buffer would have interleaved row_tiles wrongly), and
`h3_linear_bf16_nax_r128_batched_weight` (each `group.x`/row_tile reads its
own weight sub-matrix via a pointer offset, since normal linear layers
share one weight across all rows but every row_tile's compacted V^T here is
different KV blocks). Every row_tile's kept-block count is padded up to one
shared per-head max (`kept_max`), so this is 4 dispatches per head total,
the same order as dense PV's 2.

Measured (same beta=0.6, same structured data): **55.4ms (0.88x)** — much
better than v5's 74.5ms (0.66x), but still slower than dense's 48.6ms.

Diagnosed the remaining gap: `kept_max` (the per-head max used for padding)
averaged 19.8 blocks (38.8% of 51) vs the true average kept count of 14.0
blocks (27.5%) — a real but modest 1.41x "waste" from padding every
row_tile up to its head's worst-case row_tile. The bigger cost is elsewhere:
`h3_transpose_batched_bf16` is a naive one-thread-per-element kernel (no
vectorization/coalescing), and it plus the two gather passes now
move/rearrange real data volume comparable to what the smaller matmul
saves — so the compaction machinery's own overhead (not dispatch count
anymore) is what's eating the win.

**Net conclusion:** this gather-then-smaller-dense-matmul strategy for PV
sparsity is validated **not** to pay off on this hardware even after fixing
dispatch count. Would need either:

- genuinely partial-K slicing of `h3_linear_bf16_nax_r128` without any
  separate gather/transpose materializing step (needs verifying `.slice()`'s
  behavior for a non-zero, bounded-width K offset — unconfirmed/risky, this
  API has no documentation on this machine), or
- a real fused kernel that skips blocks inline during a single matmul pass
  rather than compacting data beforehand.

Both are materially larger, riskier undertakings than what was attempted
across v4–v6. **QK^T-only sparsity remains the validated, real win from
this work (1.19–1.26x); PV-side sparsity is currently a dead end via the
compaction approach.**

### External review (2026-09-28) — several corrections, verified true

Another AI reviewed this document (via a shi3z article plus this repo at
`0ebbdbf`) and raised points that materially change the plan. Verified each
one directly rather than taking them on faith:

1. **The "QK-only sparse wins 1.2x" framing was comparing against the wrong
   baseline.** That 1.2x is against *this prototype's own* dense v2
   baseline (~48–49ms), not against the app's actual production SDPA
   (23.3ms). QK-only sparse (40.9ms) is still **~1.76x slower** than what
   the app uses today. Wiring it into `h3_dit.c` now would be a
   regression, not a speedup, until the matrix-unit implementation itself
   beats 23.3ms dense, before any sparsity is even added.

2. **`MetalPerformancePrimitives` is documented — the "no documentation"
   claim above was wrong.** Only the 12-line umbrella header
   (`MetalPerformancePrimitives.h`) was checked; the real API docs are in
   sibling headers under the same framework, e.g. `MPPTensorOpsMatMul2d.h`
   (verified present in the Xcode 27 SDK on this machine, ~660 lines). It
   documents, with worked examples:
   - A **K-tile-loop pattern** (`matMulKLoop`) — pick a `tilek`, loop over
     K in chunks calling `matmul2d.run(tA, tB, tC)` repeatedly into the
     *same* destination tile. This is exactly the genuine partial-K
     accumulation v4–v6 avoided by gathering/compacting data into a
     smaller dense buffer instead. `matmul2d_descriptor` has an explicit
     `mode::multiply_accumulate` vs `mode::multiply`, confirming
     accumulation across calls is a real, intended mode.
   - **`cooperative_tensor`**: a GEMM destination that stays in per-thread
     registers (not written to device/threadgroup memory), so
     post-processing — e.g. softmax normalization — happens in-register
     before one final `.store()`. This is the documented mechanism for
     never materializing a large intermediate score matrix, which v2–v6's
     3–4 pass designs all did.

   A real fused kernel (K-tiled QK^T → in-register online softmax via
   `cooperative_tensor` → K-tiled accumulate into the PV output, skipping
   whole K-chunks by simply not calling `.run()` for them) is a documented,
   intended usage pattern here — not undocumented/risky territory.

3. **`liuliu/ccv` has a real, validated Metal Sol Attention implementation**
   (`NAInt8SolAttentionKernel`, `doc/na-int8-sol-attention.md`, confirmed
   by fetching it directly) using exactly this hardware's INT8 Neural
   Accelerator path, block routing (proxy-score based, tau threshold, a
   configurable local-block radius, and an exact "protected" token range —
   useful for never approximating text/audio/reference-conditioning
   tokens), and critically: **no intermediate token-sized FP32
   numerator/softmax-state buffer and no dense token-by-token score
   matrix** — it shares FP32 numerator and softmax state in registers
   across a 5-launch (B64) pipeline. Its own published numbers (Apple M5
   Max, synthetic, N=1/H=56/D=128, B64/Q64/tau=0.5/radius=1): 2.82x/2.99x/
   2.85x sparse speedup at 32768/65536/103982 tokens — **against ccv's own
   dense INT8 attention, not against MPSGraph SDPA** (the doc says this
   explicitly). H3 quality was not evaluated (synthetic-only benchmark).
   **Studying/porting this design is now the recommended starting point
   for any further work here**, not more tuning of v2–v6.

4. **The 49 GiB memory-footprint problem is real — verified by direct
   calculation, matches exactly.** v2–v6's batched designs materialize a
   full `[seq, kv_padded]` BF16 scores buffer per head, and the "batch all
   56 heads in one command buffer" trick that gave the speedups allocates
   all of them at once. At seq=21728 (768²/124fr — exactly where sparsity
   matters most): 0.88 GiB/head × 56 heads = **49.3 GiB**. This is a
   serious flaw independent of the speedup numbers: the current prototype
   architecture doesn't scale to the long/large clips it's meant to help.
   A streaming, per-KV-block design (per points 2–3) sidesteps this
   entirely.

5. **`h3_cache_set_enabled` (`h3.h`/`h3.c`) is real, already implemented,
   and never called from the app — verified by `grep`.** See the new item
   7 below.

6. **`core_reuse`'s existing reduction is bigger than MotionCache's
   headline number implies — verified by calling `h3_dit_reuse_schedule`
   directly.** See the reframed item 6 below.

7. Miowtion/Veda's own README (re-fetched 2026-09-28) now says it supports
   both FL2VA and Ref2VA checkpoints — the "T2VA only" note below was
   accurate when written but is stale. Its sparse kernel is still CUDA-only
   (FlashAttention-4 CuTe, vendored SM8x patch), so it stays lower priority
   for this Mac-only app regardless.

**Given 1–4, the recommended next step if resuming Sol-Attn is *not* "keep
tuning v2–v6" but a fresh attempt studying `ccv`'s
`na-int8-sol-attention.md` design and using the now-confirmed `matmul2d`
K-loop + `cooperative_tensor` APIs for a genuinely fused kernel that never
materializes a scores matrix — and to benchmark any candidate against the
app's actual 23.3ms production SDPA, not a from-scratch dense baseline,
before claiming a win.**

### What's left

- A fresh, fused-kernel attempt per the external review above, rather than
  further tuning of v2–v6's gather/compact approach.
- Zeroth-order approximation for skipped blocks (v2–v6 hard-exclude them,
  like padding; `ccv`'s design keeps a pooled-block summary contribution
  instead) — and re-verify the actual arXiv 2607.24027 algorithm details
  instead of working from this document's own summary of it.
- Real-data quality validation (PSNR + visual) using actual DiT block
  Q/K/V, not synthetic clusters.
- Wire into `h3_dit.c`'s block forward path behind an opt-in flag (never
  the default until end-to-end proven, and only once it beats 23.3ms).

Scratch sources, **not committed** (existed only under a session scratchpad
directory — rewrite from this description if resuming):

- `flash_attn.metal` / `flash_test.m` — v1, naive scalar kernel.
- `flash_v2.metal` / `flash_test2.m` — v2, nax-based dense 3-pass.
  `flash_v2.metal` also grew `h3_block_means_bf16` /
  `h3_block_dot_means_bf16` / `h3_linear_bf16_nax_r128_masked` /
  `h3_softmax_rows_masked_bf16` for QK^T sparsity, and the per-row_tile and
  batched-across-row_tiles gather/transpose/matmul kernel pairs for the PV
  compaction attempts.
- `flash_test3.m` — v3, QK^T-only sparsity, properly batched (the
  1.19–1.26x numbers).
- `flash_test4.m` — PV compaction per-row_tile, invalid per-head-sync
  comparison, superseded.
- `flash_test5.m` — the same per-row_tile PV compaction properly batched —
  the valid "PV sparsity loses badly" measurement, 0.66x.
- `flash_test6.m` — PV compaction batched across row_tiles too — the
  current, valid "PV sparsity still loses, less badly" measurement, 0.88x
  (dense 48.6ms / QK-only 40.9ms (1.19x) / QK+PV 55.4ms (0.88x), `kept_max`
  padding waste 1.41x).

### Alternative selector: Veda

Could plug into the same block-sparse kernel once it exists. arXiv
2605.30325, code `veda-sparse/Miowtion`, HF
`Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview` — tokens permuted
into 128-token 3D tiles, a 275M fp8 predictor picks the top-10% tile pairs
per layer/head; end-to-end 1.57x/2.21x/2.76x at 5/10/14s on RTX 4090. 12
fixed geometries, CUDA FA4 (CuTe) kernels only, no Metal path. Design the
kernel to take a block list so Sol-Attn (training-free, any mode) comes
first and Veda can be added later; the 3D tile permutation is worth
borrowing for Sol-Attn.

*Update 2026-09-28:* Miowtion's own README now says both FL2VA and Ref2VA
checkpoints are supported — the "T2VA only" framing above was accurate when
written but is stale. The CUDA-only sparse kernel is still the blocker for
this Mac app, so priority is unchanged (below Sol-Attn).

## 6. Adaptive DiT-reuse schedule

*Reframed 2026-09-28 from "port MotionCache/TeaCache" after checking both
against this codebase — see the external-review notes under item 5 above.*

`core_reuse`'s existing *fixed* schedule (`h3_dit.c` `h3_dit_reuse_schedule`)
already evaluates the DiT body only 6 times in 20 steps at `core_reuse=4`
(steps 0,4,8,12,16,19 — confirmed by calling the function directly), a
bigger reduction than MotionCache's own reported "20→14 calls, ~1.43x"
estimate — so porting MotionCache's specific cache values doesn't make
sense on top of what's already done.

What could still help: making *when* to re-evaluate adaptive (small
frame-to-frame change → reuse longer, large change → recompute sooner)
instead of the fixed interval, built on `core_reuse`'s own
residual-reuse/velocity-extrapolation machinery rather than importing
MotionCache's cache. `H3_REUSE_STEPS` env already exists for custom fixed
schedules. Not started. Medium effort.

## 7. Enable the app's own existing generation cache

*New item, found 2026-09-28 via external review, verified by `grep`.*

`h3_cache_set_enabled` (`h3.h`/`h3.c`) reuses the DiT/video-decoder/
conditioning state across `h3_generate` calls that share the same prepared
key (model, resolution, etc.) — `ctx->cache_enabled` defaults to 0 and is
never set to 1 anywhere in
`native/H3Spike/Sources/H3Engine/H3Engine.swift` or the rest of the app, so
this already-built reuse path is completely unused today.

Would speed up the interactive workflow (tweak prompt/seed, generate again
on the same model+resolution) for free, independent of Sol-Attn or any
other item here. Needs a size-aware eligibility/eviction policy, not
"always on" — a 24GB Mac can't just keep everything resident (same caveat
as the existing "resident" compute mode). Not started. Small-to-medium
effort — the reuse mechanism itself already exists and works; the work is
deciding the eviction policy.

---

## Lower priority, noted but not planned

MATLOWAI's fused model approximates Ref2VA as FL2VA + rank-1024 SVD delta so
one ~21 GB model/cache serves both modes (disk win, not speed).
FastVideo-FastH3-8-Step-V2 is a full diffusers-layout checkpoint (flat, not
FL2VA/Ref2VA subdirs) — would need a loader check before the Model Manager
could register it.

## Methodology note

Every approximate technique needs a fixed-seed quality comparison first:
the H3Spike CLI's `H3SPIKE_DUMP=<path>` writes raw RGB24 frames for PSNR
checks.
