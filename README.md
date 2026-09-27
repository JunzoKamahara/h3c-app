# h3c-app

[English](README.md) | [日本語](README.ja.md)

A native macOS app that runs
[MiniMax-H3](https://huggingface.co/lightx2v/Minimax-h3-Turbo) — a
text/image/video-to-video-with-audio diffusion transformer — entirely on
Apple Silicon. Everything runs in-process against Metal/MPSGraph: no
Python, no PyTorch, no cloud calls — media in and out goes through native
AVFoundation/ImageIO. The underlying
C/Objective-C inference engine (`libh3.a`) is a fork of Salvatore
Sanfilippo's (antirez) [h3.c](https://github.com/antirez/h3.c); full
attribution is in [License](#license).

While the app is running it also serves a local JSON API on `127.0.0.1`,
so it can be driven from a script as well as from its own window — see
[API](#api).

## Download

The easiest way to install h3c-app is the prebuilt, signed and notarized
`.dmg` on the [Releases page](https://github.com/JunzoKamahara/h3c-app/releases/latest) —
no Xcode Command Line Tools or building from source needed. Download the
`.dmg`, open it, and drag `h3c-app` into `Applications`; Gatekeeper accepts
it on first launch with no "unidentified developer" warning.

To build from source instead, see [Building from source](#building-from-source)
below.

## Requirements

- Apple Silicon Mac, macOS 13+. An M5-class GPU (Metal 4 TensorOps) unlocks
  the fastest int8 paths; older Apple Silicon works but falls back to
  BF16/MPSGraph automatically.
- Xcode Command Line Tools (`clang`, `swift`, `ar`) — only needed when
  building from source; not required for the prebuilt `.dmg`.
- The MiniMax-H3 checkpoint from Hugging Face
  ([`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)) —
  `FL2VA/` alone is ~134 GiB (a ~37 GiB BF16 transformer plus the Qwen3-VL
  text encoder and video/audio VAEs); adding `Ref2VA/` for reference-
  conditioned generation brings it to ~268 GiB combined. The app's model
  manager can download either or both directly.
- Nothing else at runtime.

## Quick start

```sh
make -j8 libh3.a
cd native/H3Spike
./package_app.sh
open .build/h3c-app.app
```

On first launch, if no model is registered yet, the app offers to download
MiniMax-H3 directly from Hugging Face (see [Features](#features)) — or use
the model manager to point it at a checkpoint you already have.

With a supported GPU, use the compute-mode picker to build the int8
attention cache (fast path) — or pick the memory-saving SSD-streaming mode
to skip that step and stream the original BF16 weights instead.

## Features

- **Text-to-video+audio** (T2VA): a prompt alone produces a synchronized
  H.264 + AAC clip.
- **First/last-frame conditioning** (FL2VA): anchor a generation's opening
  and/or closing frame (the app's first/last-image mode, or the API's
  `first_frame_path`/`last_frame_path`).
- **Reference conditioning** (Ref2VA), ordered and mixable: images, videos
  (with or without embedded audio), and standalone audio files as
  style/content references (the app's reference-image/video/audio mode, or
  the API's `reference_paths`). An audio file can't be the only reference —
  it needs at least one image or video reference alongside it. References
  appear to the model in order as `<Picture N>`/`<Video N>`. All reference
  reads and generated-video writes go through native AVFoundation/ImageIO.
- **Memory-aware reference-video sizing**: a large reference video is
  automatically downscaled based on the machine's physical RAM
  (`h3_reference_max_pixels()` in [h3_host.c](h3_host.c)) before it reaches
  the model, rather than always targeting the same fixed cap regardless of
  what the machine can hold. See [Reference-conditioning cost](#reference-conditioning-cost-why-a-bigger-reference-is-not-free) below for why this matters.
- **Stackable LoRAs**: the model manager keeps a library of LoRA files;
  any of them (up to 8) can be switched on at once, each with its own
  strength, and they add up. Every published H3 layout loads as-is —
  diffusers/PEFT (`to_q`/`to_k`/`to_v`), ComfyUI
  (`diffusion_model.blocks.N...`), kohya (`lora_unet_...`) and plain native
  names, BF16/F16/F32, any rank, full or partial coverage — and the manager
  shows what each file touches (format, blocks, rank). Adapters work in all
  three compute modes with no fused cache files: the engine adds the
  stacked delta to each weight on the GPU as it is loaded or streamed in
  (see [LoRA](#lora)). A 4-step Turbo distillation LoRA (from
  [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo))
  works end to end; each LoRA can carry a recommended step count
  (auto-detected from names like `..._4step_...`), and switching such a
  Turbo LoRA on moves the generation to that many steps.
- **Three compute modes**, see [Compute modes](#compute-modes): a fast int8
  attention cache, a resident mode for a Mac with memory to spare that
  needs no cache file, and a slow-but-low-memory SSD-streaming mode.
- **Speed modes**: standard / fast / fastest, bundling the engine's own
  approximations - gate-ranked DiT block skipping (`dit_layers`),
  transformer-core reuse across steps (`core_reuse`, scaled to the step
  count) and token reduction - up to ~3.2x faster at 20 steps; see
  [Speed modes](#speed-modes).
- **int8 video VAE on M5**: the video VAE decoder's transformer linears run
  on the int8 TensorOps kernels, about 3x faster decode than F32 at 46 dB
  PSNR against it (`H3_VAE_INT8=0` restores F32).
- **Model manager**: register and switch between several H3 checkpoint
  directories and LoRA files instead of one fixed path, and download
  FL2VA/Ref2VA directly from Hugging Face with no external dependency (no
  Python/huggingface_hub) — a plain resumable URLSession downloader,
  offered automatically on first launch. See
  [ModelDownloader.swift](native/H3Spike/Sources/H3cApp/ModelDownloader.swift).
- **Embedded local API**: a plain HTTP/JSON API on `127.0.0.1`, served by
  the app itself over raw POSIX sockets (no Network.framework, no
  third-party server), driving the exact same job/engine state as the
  window. See [API](#api).

## Compute modes

The DiT's ~37 GiB of BF16 weights have to be served to the GPU somehow every
generation. Three mutually exclusive modes exist, all selectable from the
app's compute-mode picker (or the API's `compute_mode` field):

| | Int8 attention cache | Resident | SSD streaming |
|---|---|---|---|
| App / API value | Fast, int8 cache / `attentionCache` | Resident, no cache needed / `resident` | Memory-saving, SSD streaming / `ssdStreaming` |
| Setup | A one-time cache build (the app offers this in-window when needed; ~28s per checkpoint, writes an ~18 GiB int8 cache file) | None to disk — quantizes in place at load time instead (same ~28s cost, paid again on every launch/model switch) | None — reads the checkpoint as-is |
| Speed | Fastest measured path | Comparable to the cache path (no per-step disk I/O either) | Slower; a 22-frame/512-square clip measured 141s vs. ~78s for the cache path (both 20 steps) |
| Memory | int8-quantized weights streamed in double-buffered slots | Every DiT block resident at once — ~18 GiB int8-quantized on a tensor-capable GPU, or the full ~37 GiB BF16 otherwise | Only 2 DiT blocks resident (~2 GiB tracked storage) at a time |
| Requires | M5-class GPU (Metal 4 TensorOps / int8 path) | Any Apple Silicon GPU (falls back to BF16 residency without tensor hardware) — meant for a Mac with memory to spare | Any Apple Silicon GPU |
| LoRA | Yes | Yes | Yes |

Full detail on the cache format — versioning, `model_kind`/`model_id`
mismatch guards — is in [h3_attention_cache.c](h3_attention_cache.c). A
cache can also be built as a standalone step with
`build_attention_cache <FL2VA/transformer dir> <output file>` (`make
build_attention_cache`), or pointed at a model root to build both FL2VA and
Ref2VA caches in one pass.

Both an unbounded generation duration *and* an oversized reference video push
against the same ceiling here — see the next section.

### Library-level configuration

The engine itself (`h3_dit.c`) reads a number of environment variables
directly — `H3_ATTENTION_CACHE`, `H3_ATTENTION_CACHE_DIR`, `H3_INT8_STREAM_MLP`,
`H3_QWEN_PREFETCH*`, `H3_ZERO_COPY_WEIGHTS`, `H3_VAE_TILE_PIXELS`,
`H3_VAE_INT8`, `H3_DIT_COMMAND_BLOCKS`, `H3_PROFILE`, and a long tail of
`H3_DISABLE_*`/`H3_USE_SLOWER_*`-style A/B diagnostic switches. These apply
to anyone linking `libh3.a` directly; the app itself doesn't rely on them
for normal use; it drives the same underlying options through its own UI
and `H3GenerationParams`. They're documented at their point of use in the
source (start from [h3_dit.c](h3_dit.c), [h3_gpu.m](h3_gpu.m), and
[h3_attention_cache.c](h3_attention_cache.c)).

## Speed modes

Independent of the compute mode, the advanced settings (or the API's
`speed_mode`) choose how much the engine approximates:

| Mode | Engine settings | 512x512 / 39 frames / 20 steps on M5 |
|---|---|---|
| Standard (`quality`) | exact | 232.6 s |
| Fast (`fast`) | 45 of 50 DiT blocks, transformer core reused over 4 steps | 82.3 s (2.8x) |
| Fastest (`fastest`) | fast + token reduction | 73.7 s (3.2x) |

Both faster modes stayed sharp and coherent in side-by-side checks, but they
change the sampling trajectory, so the same seed gives a different
composition than standard. Stacking the engine's most aggressive values
(40 blocks, core reuse 6, token reduction, int8 row FC2) reached 61.0 s but
visibly smeared the subject, so the presets stop short of that. Core reuse
scales with the step count (steps / 5, at most 4), so a 4-step Turbo LoRA run
effectively keeps only the block skipping and token reduction, and it's
turned off whenever the separate whole-velocity `reuse` is above 1.

## LoRA

Adapters are never fused into a cache file. [h3_lora.c](h3_lora.c) reads
each file once per generation, maps its tensors onto the engine's own
weight layout (ComfyUI and diffusers store attention q/k/v contiguously,
the official checkpoint interleaves them per head; diffusers also swaps
the two halves of the SwiGLU `fc1`), and concatenates the stack along the
rank. Whenever a block's weights reach the GPU — once at load for resident
weights, every step for streamed ones (int8 cache, SSD) — one matmul per
projection adds `Σ strength_i · alpha_i/rank_i · B_i A_i` in place. The
delta is far below one int8 or BF16 step, so it is added with stochastic
rounding (round-to-nearest would silently drop most of it); the rounding
noise is deterministic, so the same seed reproduces the same video.

Covered: `qkv`, `out`, `fc1` and `fc2` of the 50 DiT blocks and the 2
token-refiner blocks. Anything else in a file (`adaln_proj`, `final_layer`,
…) is ignored, and the model manager shows how many such tensors there
were. Measured on M5 at 512x512 / 39 frames / 4 steps: the lightx2v Turbo
LoRA (rank 128, all 50 blocks) adds about 1.8 s per step with the int8
cache (7.5 → 9.3 s) and about 2 s with SSD streaming (11.9 → 13.9 s); stacking a
second, rank-16 LoRA on top costs nothing measurable. Its diffusers and
ComfyUI releases produce bit-identical videos.

## Reference-conditioning cost: why a bigger reference is not free

A Ref2VA reference isn't encoded once and cached — the DiT re-attends to it
on every layer, every denoising step, alongside the main video/audio
latents. Since attention cost scales at least quadratically with total
token count, a larger or longer reference multiplies the cost of the
*entire* generation, not just a one-time encoding pass. Measured cases with
large/long references have taken 4+ hours to complete (correctly, but
slowly). The memory-aware sizing described above reduces the risk of an
extreme case but doesn't change this scaling.

**If a generation with references is unexpectedly slow**: use a
smaller/shorter reference, fewer denoising steps, or a higher `reuse`
value. An extreme case is refused outright with a clear error (a 512 MiB
attention-mask cap in the MPSGraph fallback path, see [h3_gpu.m](h3_gpu.m))
rather than risking an out-of-memory crash.

## Repository layout

```
h3.c, h3_dit.c, h3_gpu.m, ...   Core inference engine (C + Objective-C/Metal), builds into libh3.a
h3.h                            Public C API surface (h3_load_dir, h3_generate, h3_build_attention_cache, ...)
h3_shaders.metal                All Metal compute kernels
h3_build_attention_cache.c      CLI wrapper around h3_build_attention_cache() -> build_attention_cache
h3_lora.c                       LoRA loading (format normalization) and GPU weight patching
tests/                          C test suite (make test / make parity)
native/H3Spike/                 Native macOS app (SwiftPM)
  Sources/CH3                   C shim exposing libh3.a's C API to Swift
  Sources/H3Engine              Swift async wrapper over the C API (AsyncThrowingStream-based progress/cancellation)
  Sources/H3cApp                The SwiftUI app itself (h3c-app.app), including ModelLibrary/ModelManagerView
                                 (registered models/LoRAs), ModelDownloader (Hugging Face downloads), and
                                 HTTPServer/GenerationViewModel+API (the embedded automation API)
  Sources/H3Spike                Minimal in-process spike/reference client for H3Engine, not the shipped app
  package_app.sh                Builds + bundles h3c-app.app; also signs/notarizes it, see below
  make_dmg.sh                   Packages the built h3c-app.app into a distributable .dmg
```

## Building from source

### Library

```sh
make -j8 libh3.a     # builds the C/Objective-C engine
make test             # deterministic host suite (+ Metal/MLX parity if fixtures are installed)
make parity            # just the Metal/MLX numerical checks
```

`make build_attention_cache` builds the standalone cache-preparation tool
mentioned above; it isn't part of the default `make` target.

### Native app

```sh
cd native/H3Spike
./package_app.sh
```

`libh3.a` isn't rebuilt by this script — run `make libh3.a` at the repo root
first, and again after changing engine code. `package_app.sh` then runs
`swift build -c release` and assembles `h3c-app.app`. It builds to a
scratch directory outside the repo (`${TMPDIR}h3c-app-build-scratch`)
rather than SwiftPM's default in-tree `.build`, because a repo that lives
under a synced folder (Google Drive, iCloud Drive, Dropbox, ...) can cause
that daemon to hold locks on SwiftPM's `build.db`, intermittently failing
the build with a spurious "disk I/O error" — if you see that, check whether
your checkout is inside a synced directory before assuming it's flaky.

#### Code signing and notarization

Signing and notarization are opt-in via two environment variables, so a
plain `./package_app.sh` with neither set produces an unsigned development
build:

```sh
H3C_SIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" \
H3C_NOTARY_PROFILE="some-keychain-profile" \
./package_app.sh
```

- `H3C_SIGN_IDENTITY` — a `Developer ID Application` identity from
  `security find-identity -v -p codesigning`. Requires enrolling in the
  Apple Developer Program and creating that certificate (Xcode → Settings →
  Accounts → Manage Certificates → `+` → Developer ID Application — no
  Xcode project is needed for this, just the account manager). Signing
  alone (no `H3C_NOTARY_PROFILE`) is enough to run the app locally with
  Hardened Runtime enabled.
- `H3C_NOTARY_PROFILE` — a profile name saved once via
  `xcrun notarytool store-credentials <profile> --apple-id ... --team-id ...`
  (needs an [app-specific password](https://appleid.apple.com), not your
  regular Apple ID password). With both variables set, the script also
  zips, submits for notarization, waits, and staples the ticket, so the
  resulting `.app` passes Gatekeeper (`spctl -a -vvv -t exec`) on any Mac.

There are no nested frameworks or embedded dylibs to worry about here —
`h3c-app`'s only linked libraries are Apple system frameworks and the
statically-linked `libh3.a`, so a single `codesign --deep` on the bundle is
sufficient.

#### Distributing as a .dmg

```sh
./make_dmg.sh
```

Packages the already-built `h3c-app.app` into a compressed `.dmg` with a
drag-to-Applications shortcut, using only `hdiutil` (no third-party
dmg-building tool). Run `package_app.sh` first. If `H3C_SIGN_IDENTITY` and
`H3C_NOTARY_PROFILE` are set, `make_dmg.sh` also signs, notarizes, and
staples the `.dmg` file itself — optional, since Gatekeeper's real check on
launch is against the `.app`'s own signature/staple, but it makes the `.dmg`
itself pass a Gatekeeper check too for a fully clean download experience.

## Testing

```sh
make test
make parity
```

`make test` runs the deterministic host suite and, when the (git-ignored)
MLX fixtures are installed under `misc/fixtures/`, also compiles the Metal
source at runtime and checks a toy H3 block against named MLX outputs —
intentionally at runtime, matching Iris, so it needs no Xcode offline Metal
toolchain. `make parity` runs just those Metal/MLX checks.

## API

While `h3c-app.app` is running it serves a plain JSON API on
`http://127.0.0.1:8420` — implemented over raw POSIX sockets (see
[HTTPServer.swift](native/H3Spike/Sources/H3cApp/HTTPServer.swift)), not
Network.framework or any third-party server, and with no separate process
to start or stop. It drives the exact same `GenerationViewModel`/engine
instance the window does (see
[GenerationViewModel+API.swift](native/H3Spike/Sources/H3cApp/GenerationViewModel+API.swift)).
There is only ever one job at a time, shared with the UI — pressing the
generate button in the window and a `POST /api/generate` compete for the
same slot, and whichever loses gets a clear `409`.

Because the client and server are always on the same Mac, media inputs are
plain filesystem paths, not uploads.

| Method | Path | Does |
|---|---|---|
| `POST` | `/api/generate` | Starts a job. JSON body fully replaces the current draft (see below); `202` once started, `400` on a bad request, `409` if a job is already running. |
| `GET` | `/api/status` | Engine/job state, progress fraction, phase/stage text, error message, and whether a result is ready. |
| `POST` | `/api/cancel` | Cancels the running job, if any. |
| `GET` | `/api/result/video` | Streams the current result as `video/mp4`; `404` if none, or once the next job's `generate()` call deletes it. |
| `GET` | `/api/models` | Registered H3 model directories (id, name, path, whether active). |
| `GET` | `/api/loras` | Registered LoRA files (id, name, path, strength, recommended steps, whether enabled). |

`POST /api/generate` body fields, all optional except `prompt`:

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | Required. |
| `size_profile` | `"square"` | One of `smallSquare`, `square`, `landscapeUpscaled`, `landscapeNative`, `portraitUpscaled`, `portraitNative` (see `SizeProfile` in [GenerationModels.swift](native/H3Spike/Sources/H3cApp/GenerationModels.swift)). |
| `seconds` | `5` | 1–15. |
| `steps` | `20`, or the first enabled Turbo LoRA's recommended steps | |
| `reuse` | `1` | 1–3. |
| `compute_mode` | the app's own default for this GPU | `attentionCache`, `resident`, or `ssdStreaming`. |
| `speed_mode` | `"quality"` | `quality`, `fast`, or `fastest` (see [Speed modes](#speed-modes)). |
| `seed` | random | |
| `first_frame_path` / `last_frame_path` | none | FL2VA anchors; cannot combine with `reference_paths`. |
| `reference_paths` | `[]` | Ordered Ref2VA references; image/video/audio is auto-detected per path. At least one image or video is required if any audio path is included. |
| `loras` | `[]` | The LoRA stack for this job: names from `GET /api/loras`, or `{"name": ..., "strength": 0.8}` objects (strength multiplies the adapter's trained scale; omitted keeps the entry's saved strength). Omitting it means no LoRA, even if some are switched on in the window. |
| `lora_name` / `lora_scale` | none | Older single-LoRA form of `loras`; ignored when `loras` is given. |

```sh
curl -X POST http://127.0.0.1:8420/api/generate -H "Content-Type: application/json" -d '{
  "prompt": "A red fox walks through fresh snow in a pine forest.",
  "size_profile": "square", "seconds": 5, "steps": 20
}'

curl http://127.0.0.1:8420/api/status
curl http://127.0.0.1:8420/api/result/video -o fox.mp4
```

## License

MIT — see [LICENSE](LICENSE). Third-party notices (a Metal kernel design
adapted from ccv's FlashAttention implementation) are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The inference engine
originates from [antirez/h3.c](https://github.com/antirez/h3.c) by
Salvatore Sanfilippo.
