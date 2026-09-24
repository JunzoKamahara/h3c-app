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

## Requirements

- Apple Silicon Mac, macOS 13+. An M5-class GPU (Metal 4 TensorOps) unlocks
  the fastest int8 paths; older Apple Silicon works but falls back to
  BF16/MPSGraph automatically.
- Xcode Command Line Tools (`clang`, `swift`, `ar`).
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
- **Reference conditioning** (Ref2VA), ordered and mixable: images and
  videos (with or without embedded audio) as style/content references (the
  app's reference-image/video mode, or the API's `reference_paths`). References
  appear to the model in order as `<Picture N>`/`<Video N>`. All reference
  reads and generated-video writes go through native AVFoundation/ImageIO.
- **Memory-aware reference-video sizing**: a large reference video is
  automatically downscaled based on the machine's physical RAM
  (`h3_reference_max_pixels()` in [h3_host.c](h3_host.c)) before it reaches
  the model, rather than always targeting the same fixed cap regardless of
  what the machine can hold. See [Reference-conditioning cost](#reference-conditioning-cost-why-a-bigger-reference-is-not-free) below for why this matters.
- **LoRA**: the app's model manager keeps a library of registered LoRA
  files (each with its own saved strength), fused into the model at load
  time under any residency mode (see [h3_lora.c](h3_lora.c)). A 4-step
  Turbo distillation LoRA (from
  [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo))
  works end to end. `build_lora_cache` (built via `make build_lora_cache`)
  additionally pre-fuses a LoRA into an int8 cache file offline, for
  anyone driving the engine directly rather than through the app.
- **Two compute modes**, see [Compute modes](#compute-modes): a fast int8
  attention cache, and a slow-but-low-memory SSD-streaming mode with no
  cache file needed.
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
generation. Two mutually exclusive modes exist, both selectable from the
app's compute-mode picker (or the API's `compute_mode` field):

| | Int8 attention cache | SSD streaming |
|---|---|---|
| App / API value | Fast, int8 cache / `attentionCache` | Memory-saving, SSD streaming / `ssdStreaming` |
| Setup | A one-time cache build (the app offers this in-window when needed; ~28s per checkpoint, writes an ~18 GiB int8 cache file) | None — reads the checkpoint as-is |
| Speed | Fastest measured path | Slower; a 22-frame/512-square clip measured 141s vs. ~78s for the cache path (both 20 steps) |
| Memory | int8-quantized weights streamed in double-buffered slots | Only 2 DiT blocks resident (~2 GiB tracked storage) at a time |
| Requires | M5-class GPU (Metal 4 TensorOps / int8 path) | Any Apple Silicon GPU |
| LoRA | Yes | No — the engine only fuses LoRA when loading resident/cache blocks |

Full detail on the cache format — versioning, `model_kind`/`model_id`
mismatch guards, LoRA fusion into a cache — is in
[h3_attention_cache.c](h3_attention_cache.c) and [h3_lora.c](h3_lora.c). A
cache can also be built as a standalone step with
`build_attention_cache <FL2VA/transformer dir> <output file>` (`make
build_attention_cache`), or pointed at a model root to build both FL2VA and
Ref2VA caches in one pass.

Both an unbounded generation duration *and* an oversized reference video push
against the same ceiling here — see the next section.

### Library-level configuration

The engine itself (`h3_dit.c`) reads a number of environment variables
directly — `H3_ATTENTION_CACHE`, `H3_ATTENTION_CACHE_DIR`, `H3_LORA_PATH`,
`H3_LORA_SCALE`, `H3_TOKEN_REFINER_LORA`, `H3_INT8_STREAM_MLP`,
`H3_QWEN_PREFETCH*`, `H3_ZERO_COPY_WEIGHTS`, `H3_VAE_TILE_PIXELS`,
`H3_DIT_COMMAND_BLOCKS`, `H3_PROFILE`, and a long tail of
`H3_DISABLE_*`/`H3_USE_SLOWER_*`-style A/B diagnostic switches. These apply
to anyone linking `libh3.a` directly; the app itself doesn't rely on them
for normal use; it drives the same underlying options through its own UI
and `H3GenerationParams`. They're documented at their point of use in the
source (start from [h3_dit.c](h3_dit.c), [h3_gpu.m](h3_gpu.m), and
[h3_attention_cache.c](h3_attention_cache.c)).

## Reference-conditioning cost: why a bigger reference is not free

This is the single most important thing to understand before feeding a large
reference video into Ref2VA. The DiT's own joint self-attention
(`h3_gpu_sdpa_bf16`) is **non-causal and attends over every token every
layer, every evaluated denoising step** — main video/audio latents *and*
reference-conditioning tokens together. A reference isn't encoded once and
cached; it's re-attended to by all 50 DiT blocks on every step. Since
attention cost scales at least quadratically with total token count, a
larger or longer reference video multiplies the cost of the *entire*
generation, not just a one-time encoding pass.

Measured on this basis: a 672×384/10s generation against a large reference
video took ~4 hours (but completed correctly); the same reference enlarged
further to 15s, and separately a 1344×768/15s case, both also completed in
over 4 hours. None of these are bugs — isolated benchmarking of
`h3_gpu_gqa_causal_bf16` and `h3_gpu_sdpa_bf16` in isolation ruled out the
attention kernels themselves as anomalously slow; the cost is architectural.
The memory-aware cap described above (`h3_reference_max_pixels`) reduces
the *risk* of an extreme case, but does not change this scaling — expect
large/long reference videos to be slow, and prefer a smaller/shorter
reference, fewer denoising steps, or a higher `reuse` value when iterating.

A related, harder failure this scaling can cause: the causal-attention
fallback that runs when a sequence exceeds the custom Metal kernel's
threadgroup-memory limit (`h3_gpu_gqa_mps`, an MPSGraph path that builds an
explicit `O(sequence²)` mask) is capped at a 512 MiB mask — beyond that,
generation is refused with a clear error rather than risking the machine
running out of memory (see [h3_gpu.m](h3_gpu.m)). The general lesson,
applicable anywhere else a cache is keyed by a caller-controlled size: **a
cache keyed by unbounded input size needs an explicit ceiling, checked
before the allocation, not just a comment saying the input is expected to
be small.**

## Repository layout

```
h3.c, h3_dit.c, h3_gpu.m, ...   Core inference engine (C + Objective-C/Metal), builds into libh3.a
h3.h                            Public C API surface (h3_load_dir, h3_generate, h3_build_attention_cache, ...)
h3_shaders.metal                All Metal compute kernels
h3_build_attention_cache.c      CLI wrapper around h3_build_attention_cache() -> build_attention_cache
h3_build_lora_cache.c           CLI tool that fuses a LoRA into an int8 cache offline -> build_lora_cache
tests/                          C test suite (make test / make parity)
native/H3Spike/                 Native macOS app (SwiftPM)
  Sources/CH3                   C shim exposing libh3.a's C API to Swift
  Sources/H3Engine              Swift async wrapper over the C API (AsyncThrowingStream-based progress/cancellation)
  Sources/H3cApp                The SwiftUI app itself (h3c-app.app), including ModelLibrary/ModelManagerView
                                 (registered models/LoRAs), ModelDownloader (Hugging Face downloads), and
                                 HTTPServer/GenerationViewModel+API (the embedded automation API)
  Sources/H3Spike                Minimal in-process spike/reference client for H3Engine, not the shipped app
  package_app.sh                Builds + bundles h3c-app.app; also signs/notarizes it, see below
```

## Building from source

### Library

```sh
make -j8 libh3.a     # builds the C/Objective-C engine
make test             # deterministic host suite (+ Metal/MLX parity if fixtures are installed)
make parity            # just the Metal/MLX numerical checks
```

`make build_attention_cache` and `make build_lora_cache` build the two
standalone cache-preparation tools mentioned above; neither is part of the
default `make` target.

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
| `GET` | `/api/loras` | Registered LoRA files (id, name, path, scale, whether active). |

`POST /api/generate` body fields, all optional except `prompt`:

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | Required. |
| `size_profile` | `"square"` | One of `smallSquare`, `square`, `landscapeUpscaled`, `landscapeNative`, `portraitUpscaled`, `portraitNative` (see `SizeProfile` in [GenerationModels.swift](native/H3Spike/Sources/H3cApp/GenerationModels.swift)). |
| `seconds` | `5` | 1–15. |
| `steps` | `20` | |
| `reuse` | `1` | 1–3. |
| `compute_mode` | the app's own default for this GPU | `attentionCache` or `ssdStreaming`. |
| `seed` | random | |
| `first_frame_path` / `last_frame_path` | none | FL2VA anchors; cannot combine with `reference_paths`. |
| `reference_paths` | `[]` | Ordered Ref2VA references; image vs. video is auto-detected per path. |
| `lora_name` | none | Must match a name from `GET /api/loras`; omitting it (or `""`) means no LoRA for this job, even if one was selected in the window. |
| `lora_scale` | the LoRA's own saved scale | Only meaningful with `lora_name`. |

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
