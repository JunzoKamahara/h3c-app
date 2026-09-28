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

### Reproducible bench vs production SDPA (2026-09-28)

Cloned `liuliu/ccv` (commit `6a611be`, `unstable` branch) and got its own
official benchmark tools building. Two real findings, one negative and one
strongly positive:

**Update 2026-09-28, same day — the Sol/sparse compile blocker is fixed.**
A patch was proposed (by another AI reviewing this document, working from
the exact error text and a matching report at
[ml-explore/mlx#4533](https://github.com/ml-explore/mlx/issues/4533))
diagnosing the cause as a Metal address-space qualifier surviving into a
`decltype(...)` used as a `get_destination_cooperative_tensor` template
argument, fixed by wrapping each with `metal::remove_addrspace_t<...>` (7
call sites in `NAInt8SolAttentionKernel.cpp`'s runtime-generated shader
source, no change to math/precision/layout). Applied it, rebuilt
`libccv.a` and `na_int8_sol_attention_bench`, and it now **compiles and
runs successfully** — the original diagnosis below (kept for the record)
turned out to be too pessimistic; this was fixable from outside Apple's
compiler after all. See "Sol/sparse kernel, now working" further below for
results.

Original (now-resolved) negative finding, kept for the record:
`bin/mfa/na_int8_sol_attention_bench` already does exactly what's needed
(accepts real captured contiguous-FP16 `[1,T,H,128]` Q/K/V via a
`raw-prefix` argument, compares native/all-exact-sol/sparse-sol). Building
it needed `./configure --enable-mps` in `lib/` (an opt-in flag, not
auto-detected) and `wget`. But its runtime-generated Metal shader **failed
to compile** on this machine: `error: no matching member function for call
to 'get_destination_cooperative_tensor'`, for matmul stages that chain a
`cooperative_tensor` as the left operand of a subsequent `matmul2d` — the
exact "no materialized scores buffer" fusion that makes this design
attractive. Assumed at the time to be an SDK/OS version mismatch (this
machine reports SDK 27.0; ccv's own published numbers were measured on
"macOS 26.6.2"); that assumption was wrong, or at least not the whole
story — the real cause was the address-space issue above.

**Strongly positive — ccv's plain dense int8 attention kernel (no sparsity,
no SDK blocker) already beats this app's production SDPA.** A second,
simpler ccv tool, `bin/mfa/na_int8_attention_bench` (just
`NAInt8AttentionKernel`, no Sol/routing stages), built and ran cleanly.
Run at this app's real shape (D=128, Hq=Hk=56) and its three real
sequence lengths:

| sequence length | ccv int8 NAX (quantize+int8, avg) | this app's real MPSGraph SDPA | speedup |
|---|---|---|---|
| 3214 (~512²/39fr) | 18.9 ms | 23.3 ms | 1.23x |
| 5744 (384²/124fr) | 54.0 ms | 75.0 ms | 1.39x |
| 21728 (768²/124fr) | 840.8 ms | 1340.3 ms | 1.59x |

(Synthetic random inputs — this tool has no raw-data-loading option, but
attention throughput at a given dtype/shape doesn't depend on data
content, only quality does. ccv's own internal FP16 "baseline" in the same
run was noisy and much slower than either column and isn't a useful
reference; the comparison that matters is the int8 column against this
app's own independently-measured MPSGraph number, same shapes. ccv's own
sampled validation against an FP32 reference reported `max_abs_o` ≈
3.8e-4 at all three shapes.)

**This is the clearest, most decisive result of the whole investigation:**
a real, already-working, already-validated-by-its-authors int8 kernel
beats this app's actual production attention by 1.2–1.6x, growing with
sequence length — without needing any sparsity at all. This re-opens
Sol-Attn in a more promising form than "block-sparse or nothing":
integrating a plain int8-quantized dense attention kernel (`ccv`'s
`NAInt8AttentionKernel` or an equivalent built from the same
`matmul2d`/`cooperative_tensor` primitives) is likely worth doing on its
own, before Sol's block-skip logic (which is still blocked by the SDK
issue above, and would only add to this dense win once available).

A capture hook now exists to get real (not synthetic) attention data out
of this engine for a follow-up quality check: `h3_dit.c` gained
`debug_dump_attention_qkv()`, called right after QKV projection/RoPE
(the exact input the production SDPA call consumes), behind three
off-by-default env vars (`H3_DUMP_ATTENTION_QKV=<prefix>`, `_BLOCK=<n>`,
`_STEP=<n>`) — converts BF16 to FP16 and writes `<prefix>.{q,k,v}.bin` in
ccv's expected format. Verified against a real 512²/39fr/4-step
generation, and used below for a real-data run of the Sol/sparse kernel
(not yet fed into `na_int8_attention_bench`, which still needs a small
patch to load real files — the dense kernel above was only checked on
synthetic-shape data).

### Sol/sparse kernel, now working — fast, but not yet a proven quality win

With the address-space patch applied, `na_int8_sol_attention_bench` runs
end-to-end and reports its own internal `native_ms` (plain int8, same
kernel family as the dense-only result above), `exact_ms` (Sol's
all-blocks-exact path, no approximation — a correctness check, not meant
to be fast), and `sparse_ms` (real routing/approximation active), plus
relative-L2 error against an FP32 reference.

Synthetic random Q/K/V, this app's real shape (H=56, head_dim=128) and
three real sequence lengths, default routing (`tau=0.5`,
`local_block_radius=1`, `approximation_start=470`):

| sequence length | native (dense int8) | sparse (Sol routing) | vs. this app's SDPA | sparse relative L2 |
|---|---|---|---|---|
| 3214 | 19.4 ms | 13.4 ms | 1.74x | 0.199 |
| 5744 | 54.8 ms | 38.5 ms | 1.95x | 0.217 |
| 21728 | 829.9 ms | 474.6 ms | 2.82x | 0.238 |

Then the same benchmark against the **real captured Q/K/V** from
`debug_dump_attention_qkv()` (block 25, step 2 of an actual 512²/39fr
generation, seed 7 — exactly T=3214, H=56):

```
native_ms=19.4265 exact_ms=27.2384 sparse_ms=12.6208
paired_sparse_speedup=1.5467  (i.e. 1.85x vs. this app's 23.3ms SDPA)
exact_block_fraction=0.512701
exact_relative_l2=5.94e-06        <- int8 quantization alone: negligible
protected_relative_l2=5.00e-06    <- protected/local region: negligible
sparse_relative_l2=0.1274         <- routed-approximate blocks: NOT negligible
```

**Read this carefully — it is a real speed win with a real, unresolved
quality question, not an unqualified win yet.** The int8 quantization
itself is essentially lossless here (~6e-6 relative error, consistent
with the dense-only kernel's ~3.8e-4 `max_abs_o` figure from ccv's own
validation). The cost is entirely in Sol's block-approximation: on real
attention inputs it produces a **12.7% relative L2 error** in the full
output, versus a fraction of a percent for the quantization-only path.
That is a per-attention-call, per-block number after just one DiT block's
worth of attention — whether it is acceptable depends entirely on how it
compounds or washes out across 50 DiT blocks and a VAE decode, which has
not been tested. A 13% relative-L2 difference could be invisible after
denoising smooths it out, or it could show up as visible drift/artifacts;
this document is not going to guess which without actually rendering a
frame. The synthetic-data error (0.199–0.238) is higher still, confirming
error is data-dependent as expected — real (correlated, RoPE-structured)
attention inputs approximate somewhat better than i.i.d. random noise, but
not enormously so.

Net: the reproducible bench asked for at the top of this section now
exists and gives a clear, honest answer — **dense int8 alone is a safe
1.2–1.6x win with negligible quality cost; adding Sol's sparsity on top
pushes that to 1.7–2.8x but at a quality cost that is not yet validated
end-to-end and must not be treated as free.**

### Correction: the "12.7% error" above was measured against the wrong reference — real ground-truth comparison now done, numbers revised

**The `sparse_relative_l2`/`exact_relative_l2` figures quoted just above are
ccv's own internal self-consistency check (its Sol-exact-mode output vs.
its own plain dense-kernel output, both computed by ccv from the same
input) — not a comparison against this app's actual real SDPA output.**
That distinction turned out to matter a lot once a direct comparison was
attempted, because it surfaced a real bug in how the input was captured.

Building the direct comparison (a new small tool,
`h3_real_attention_compare.cpp`, calling `ccv_nnc_mfa_encode_attention`
and `ccv_nnc_mfa_encode_sol_attention` directly on this app's captured
real Q/K/V, then diffing the result against this app's own captured real
SDPA output) surfaced two real bugs, both now fixed:

1. **The QKV capture itself was wrong.** This app's default (M5, int8
   QKV projection) path leaves `query`/`key`/`value` in `[heads, rows,
   head_dim]` layout for the immediately-following SDPA call to consume
   as a producer/consumer optimization (`h3_gpu.m`'s
   `headMajorSDPAInputs` flag) — but `debug_dump_attention_qkv()` assumed
   plain `[rows, heads, head_dim]` and wrote the head-major buffer out
   unpermuted. Every earlier "real capture" result in this document (the
   1.85x/0.1274 figures above) was computed on this silently-transposed,
   effectively-scrambled data. Fixed by adding a small accessor,
   `h3_gpu_head_major_sdpa_inputs()`, and having the dump function
   transpose to row-major before writing whenever it reports true. The
   first symptom, run through the (also newly fixed) real-vs-real
   comparison tool below, was unmistakable: `relative_l2=1.37` against
   this app's real output — worse than uncorrelated noise, an immediate
   sign of a layout bug rather than a genuine quality problem.
2. **The attention-cache used for capture had a stale weight fingerprint**
   (`h3: warning: attention cache ... model fingerprint does not match
   the loaded checkpoint`) - H3Spike auto-points at
   `~/models/cache/dit_int8_v2.cache` by default, and it was quietly out
   of sync with the currently-loaded checkpoint. The warning doesn't
   block generation (it's non-fatal), so this could have kept silently
   giving wrong int8-quantized weights without a hard failure. Fixed by
   rebuilding the cache (`./build_attention_cache
   ~/models/MiniMax-H3/FL2VA/transformer ~/models/cache/dit_int8_v2.cache`)
   before recapturing.

With both fixed, and also dumping this app's own real SDPA output
alongside the Q/K/V (a new sibling hook, `debug_dump_attention_output()`,
gated behind the same env vars, only meaningful with
`H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1` since the attention-output
tensor has its own, independent head-major toggle), three real
block/step captures from an actual 512²/39fr/4-step generation (seed 7)
were compared directly - ccv's kernel output vs. this app's own real
production SDPA output, not against any reference ccv computes
internally:

| block, step | ccv dense int8 vs. real SDPA | ccv Sol sparse vs. real SDPA |
|---|---|---|
| 0, step 0 | relative L2 1.56%, PSNR 62.0 dB | relative L2 9.97%, PSNR 45.8 dB |
| 25, step 2 | relative L2 1.77%, PSNR 57.7 dB | relative L2 10.18%, PSNR 41.5 dB |
| 49, step 3 | relative L2 4.57%, PSNR 52.9 dB | relative L2 6.14%, PSNR 50.3 dB |

**This is a much stronger and more trustworthy result than the earlier
one, in both directions.** Dense int8 is confirmed, against real
ground truth (not a self-consistency check), to closely reproduce this
app's actual attention output across early/mid/late blocks - 52-62 dB
PSNR is generally in the range other quality-affecting approximations in
this codebase are judged acceptable at (compare the VAE int8 work's 44.9
dB worst-frame PSNR, judged visually identical). Sol sparse is real but
consistently worse - 42-50 dB PSNR, a genuine, non-trivial quality cost
that is smaller than the earlier (buggy-data) 12.7%/41.5 dB estimate
suggested at one point and larger at another, but now backed by a real
reference instead of an internal check. Whether 42-50 dB per-block PSNR
survives 50 blocks + VAE decode without becoming visible is still the
open question - this correction sharpens the number that question is
about, it doesn't answer it.

**Lesson for any future real-data validation in this codebase**: this
engine has several independent, opt-in "head-major" layout optimizations
(SDPA inputs via QKV projection, SDPA's own output via
`h3_gpu_sdpa_bf16_head_major_output`) that silently change which axis is
contiguous, purely as GPU producer/consumer optimizations invisible to
the math - a raw tensor dump is only meaningful if it explicitly checks
and corrects for these, and a wildly-uncorrelated (`relative_l2 > 1`)
comparison against a real reference is a strong, checkable signal that a
layout assumption is wrong before concluding anything about quality.

### What's left

- **End-to-end quality validation is now the actual blocker, not the Metal
  compile error (fixed above) or capture correctness (fixed above).**
  Three real block/step spot-checks now confirm both kernels' per-block
  behavior against real ground truth (dense: 52-62 dB PSNR; sparse: 42-50
  dB PSNR). What's still missing is a full generation run actually
  substituting one of these kernels for every block's SDPA call (not an
  offline per-block comparison against the untouched trajectory - once
  block N's output changes, block N+1 sees a different input, so error
  can compound or cancel in ways a per-block check can't show) and
  comparing the final decoded video against the real baseline (PSNR +
  visual).
- Dense int8's per-block numbers are strong enough that it's likely a safe
  default-quality win; if sparse's end-to-end quality doesn't hold up,
  dense int8 alone is still a clean win on its own, independent of Sol's
  routing logic.
- Zeroth-order approximation for skipped blocks (v2–v6 hard-exclude them,
  like padding; `ccv`'s design keeps a pooled-block summary contribution
  instead) — and re-verify the actual arXiv 2607.24027 algorithm details
  instead of working from this document's own summary of it.
- Wire into `h3_dit.c`'s block forward path behind an opt-in flag (never
  the default until end-to-end proven, and only once it beats 23.3ms).
- A fresh, fused own-kernel attempt per the external review above (rather
  than further tuning of v2–v6's gather/compact approach) only becomes
  worth doing if a from-scratch kernel is preferred over depending on
  `ccv`'s existing, now-working implementation.

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
