# h3c-app

[English](README.md) | [日本語](README.ja.md)

A native macOS app (plus a CLI and a local web GUI) that runs
[MiniMax-H3](https://huggingface.co/lightx2v/Minimax-h3-Turbo) — a
text/image/video-to-video-with-audio diffusion transformer — entirely on
Apple Silicon. Everything runs in-process against Metal/MPSGraph: no Python,
no PyTorch, no cloud calls, and (since this fork) no FFmpeg dependency either
— media in and out goes through native AVFoundation/ImageIO.

This is a fork of Salvatore Sanfilippo's (antirez)
[h3.c](https://github.com/antirez/h3.c). The C/Objective-C engine
(`libh3.a`) is shared unchanged across all three front ends; this fork adds
the native SwiftUI app, native (ffmpeg-free) reference-media decoding, a
memory-aware reference-video resolution cap, an int8-attention-cache builder
usable as a library call (not just a CLI tool), and Developer ID
signing/notarization for distributing the app.

## Three ways to run it

| Front end | Where | Best for |
|---|---|---|
| **Native app** (`h3c-app.app`) | `native/H3Spike/` | End users. Guided form (subject/shape/length/references), progress + ETA, preview, no flags to remember. |
| **CLI** (`./h3`) | repo root | Scripting, benchmarking, and the full flag/env-var surface — every capability below is reachable here first. |
| **Local web GUI** | `gui/server.py` | Browser access to the same engine without building the native app; wraps `./h3` as a subprocess. Standard-library-only Python, no `pip install`. |

## Requirements

- Apple Silicon Mac, macOS 13+. An M5-class GPU (Metal 4 TensorOps) unlocks
  the fastest int8 paths; older Apple Silicon works but falls back to
  BF16/MPSGraph automatically.
- Xcode Command Line Tools (`clang`, `swift`, `ar`).
- The MiniMax-H3 checkpoint from Hugging Face
  ([`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)) —
  `FL2VA/` alone is ~134 GiB (the ~37 GiB BF16 transformer plus the Qwen3-VL
  text encoder and video/audio VAEs); adding `Ref2VA/` for reference-
  conditioned generation brings it to ~268 GiB combined. The native app's
  "モデル管理" ("Model manager") can download either or both directly.
- Nothing else at runtime. FFmpeg/FFprobe are **not** required — they're only
  used by one optional cross-check test (see [Testing](#testing)).

## Quick start

### Native app

```sh
make -j8 libh3.a
cd native/H3Spike
./package_app.sh
open .build/h3c-app.app
```

The app looks for the model at a fixed path:
`~/Library/Application Support/h3c-analysis/MiniMax-H3`. Put (or symlink)
your checkpoint there, with `FL2VA/` and optionally `Ref2VA/` inside it,
before first launch.

On first run with a supported GPU, use the app's compute-mode picker to build
the int8 attention cache (fast path) — or pick "省メモリ（SSDストリーミング）"
to skip that step entirely and stream the original BF16 weights instead.

### CLI

```sh
make -j8
mkdir -p outputs
./h3 --info -d ./MiniMax-H3          # inspect the model/device, no generation
./h3 -d ./MiniMax-H3 -p "A red fox walks through fresh snow in a pine forest." \
  --width 512 --height 512 --frames 22 --steps 20 -o outputs/fox.mp4
```

Without `-p`, `./h3 -d ./MiniMax-H3` starts an interactive session (`!help`
for commands, `!ref-image`/`!first`/`!last` for conditioning, `!save` to
write the current result). Run `./h3 --help` for the complete flag list —
the [CLI flag reference](#cli-flag-reference) below covers the ones worth
knowing about first.

### Web GUI

```sh
make -j8
python3 gui/server.py --port 8420
```

Then open `http://localhost:8420`. It shells out to `./h3` per job and
streams its progress into the page.

## Features

- **Text-to-video+audio** (T2VA): a prompt alone produces a synchronized
  H.264 + AAC clip.
- **First/last-frame conditioning** (FL2VA): anchor a generation's opening
  and/or closing frame with `--first-frame`/`--last-frame` (or the app's
  "最初・最後の画像" mode).
- **Reference conditioning** (Ref2VA), ordered and mixable: images
  (`--ref-image`), video with or without its audio (`--ref-video` /
  `--ref-silent-video`), a video with replacement audio
  (`--ref-video-audio`), and standalone audio (`--ref-audio`). References
  appear to the model in argument order as `<Picture N>`/`<Video N>`.
  Reference video/image reads and all video/audio writes go through native
  AVFoundation/ImageIO — no FFmpeg subprocess involved.
- **Memory-aware reference-video sizing**: a large reference video is
  automatically downscaled based on the machine's physical RAM
  (`h3_reference_max_pixels()` in [h3_host.c](h3_host.c)) before it reaches
  the model, rather than always targeting the same fixed cap regardless of
  what the machine can hold. See [Reference-conditioning cost](#reference-conditioning-cost-why-a-bigger-reference-is-not-free) below for why this matters.
- **LoRA**: `H3_LORA_PATH` fuses a diffusers/peft-format adapter at load
  time under any residency mode (see [h3_lora.c](h3_lora.c)); `build_lora_cache`
  pre-fuses one into an int8 cache file offline. A 4-step Turbo distillation
  LoRA (from
  [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo))
  is supported end to end, including the native app's "Turbo" option.
- **Two compute modes**, see the next section: a fast int8 attention cache,
  and a slow-but-low-memory SSD-streaming mode with no cache file needed.
- **Model manager** (native app only): register and switch between several
  H3 checkpoint directories and LoRA files instead of one hardcoded path,
  and download FL2VA/Ref2VA directly from Hugging Face with no external
  dependency (no Python/huggingface_hub) — a plain resumable URLSession
  downloader, offered automatically on first launch. See
  [ModelDownloader.swift](native/H3Spike/Sources/H3cApp/ModelDownloader.swift).
- **Interactive terminal preview** (`--show`) on Kitty/Ghostty/iTerm2/WezTerm/
  Konsole, and `--profile` for per-phase Metal timing/memory diagnostics.

## Compute modes

The DiT's ~37 GiB of BF16 weights have to be served to the GPU somehow every
generation. Two mutually exclusive modes exist:

| | Int8 attention cache | SSD streaming |
|---|---|---|
| Flag / setting | `H3_ATTENTION_CACHE=path` (app: "高速（int8キャッシュ）") | `--ssd-streaming` (app: "省メモリ（SSDストリーミング）") |
| Setup | One-time `build_attention_cache` run (~28s CLI, writes an ~18 GiB int8 cache file per checkpoint) | None — reads the checkpoint as-is |
| Speed | Fastest measured path | Slower; a 22-frame/512-square clip measured 141s vs. ~78s for the cache path (both 20 steps) |
| Memory | int8-quantized weights streamed in double-buffered slots | Only 2 DiT blocks resident (~2 GiB tracked storage) at a time |
| Requires | M5-class GPU (Metal 4 TensorOps / int8 path) | Any Apple Silicon GPU |
| LoRA | Yes (`H3_LORA_PATH`, or a pre-fused cache) | No — the engine only fuses LoRA when loading resident/cache blocks |

Build a cache with `build_attention_cache <FL2VA/transformer dir> <output
file>`, or point it at a model root to build both FL2VA and Ref2VA caches in
one pass. Full detail — cache versioning, `model_kind`/`model_id` mismatch
guards, `H3_INT8_STREAM_MLP`, LoRA fusion into a cache — is in
[h3_attention_cache.c](h3_attention_cache.c) and
[h3_lora.c](h3_lora.c); the CLI wrapper is
[h3_build_attention_cache.c](h3_build_attention_cache.c).

Both an unbounded generation duration *and* an oversized reference video push
against the same ceiling here — see the next section.

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
`h3_gpu_gqa_causal_bf16` and `h3_gpu_sdpa_bf16` in isolation (see git history
around `dff0763` for the methodology) ruled out the attention kernels
themselves as anomalously slow; the cost is architectural. The memory-aware
cap described above (`h3_reference_max_pixels`) reduces the *risk* of an
extreme case, but does not change this scaling — expect large/long reference
videos to be slow, and prefer a smaller/shorter reference or a lower
`--reuse`/`--layers` preset when iterating.

A related, harder failure this scaling caused once: the causal-attention
fallback that runs when a sequence exceeds the custom Metal kernel's
threadgroup-memory limit (`h3_gpu_gqa_mps`, an MPSGraph path that builds an
explicit `O(sequence²)` mask) used to cache that mask per sequence length for
the GPU's lifetime with no size limit. A sufficiently long reference-driven
sequence made that cache allocate tens of gigabytes and crashed the whole
machine (a watchdog-timeout kernel panic, not just an app crash — see
`/Library/Logs/DiagnosticReports/*.panic` if you ever need to diagnose one).
[h3_gpu.m](h3_gpu.m) now refuses generation with a clear error instead, once
that fallback's mask would exceed 512 MiB, rather than allocating it. The
general lesson, applicable anywhere else a cache is keyed by a
caller-controlled size: **a cache keyed by unbounded input size needs an
explicit ceiling, checked before the allocation, not just a comment saying
the input is expected to be small.**

## Repository layout

```
h3.c, h3_dit.c, h3_gpu.m, ...   Core inference engine (C + Objective-C/Metal), builds into libh3.a and the CLI
h3.h                            Public C API surface (h3_load_dir, h3_generate, h3_build_attention_cache, ...)
h3_shaders.metal                All Metal compute kernels
main.c, h3_cli.c, linenoise.c   CLI argument parsing + interactive session front end -> ./h3
h3_build_attention_cache.c      CLI wrapper around h3_build_attention_cache() -> build_attention_cache
h3_build_lora_cache.c           CLI tool that fuses a LoRA into an int8 cache offline -> build_lora_cache
tests/                          C test suite (make test / make parity)
gui/                            Local web GUI (stdlib-only Python server + static frontend)
native/H3Spike/                 Native macOS app (SwiftPM)
  Sources/CH3                   C shim exposing libh3.a's C API to Swift
  Sources/H3Engine              Swift async wrapper over the C API (AsyncThrowingStream-based progress/cancellation)
  Sources/H3cApp                The SwiftUI app itself (h3c-app.app)
  Sources/H3Spike                Minimal in-process spike/reference client for H3Engine, not the shipped app
  package_app.sh                Builds + bundles h3c-app.app; also signs/notarizes it, see below
```

## Building from source

### Library and CLI

```sh
make -j8            # builds ./h3 and libh3.a
make test            # deterministic host suite (+ Metal/MLX parity if fixtures are installed)
make parity           # just the Metal/MLX numerical checks
```

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
plain `./package_app.sh` with neither set still produces the same unsigned
development build as before:

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

### Web GUI

```sh
make -j8
python3 gui/server.py --port 8420
```

No install step: it's plain standard-library Python, subprocessing `./h3`
per job and parsing its `\r%-25s %4d/%-4d` progress lines
(`cli_progress` in `h3_cli.c`) into JSON the page polls.

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

One test, `h3_av_mux_test`, cross-checks the native AVFoundation muxer
against real FFmpeg output and is skipped automatically if `ffmpeg` isn't on
`PATH` — this is the only place FFmpeg is used anywhere in this project now.

## CLI flag reference

The full list is in `./h3 --help`; these are the ones to reach for first.
Defaults: `--width 864 --height 480 --frames 56 --steps 20 --layers 50
--reuse 1`.

| Flag | Effect |
|---|---|
| `-d, --model-dir PATH` | MiniMax-H3 checkpoint root |
| `-p, --prompt TEXT` | Run once and exit; omit for an interactive session |
| `--width/--height N` | Output canvas (multiples of 32, product ≤ 768×1344) |
| `--frames N` / `--seconds N` | Duration — mutually exclusive; rounds up to a legal H3 temporal shape |
| `--steps N` | Denoising passes. 20 default; 4–7 for fast iteration; 50 as a close-reference oracle |
| `--reuse N` | Whole-denoiser reuse (extrapolates skipped steps): 1 exact, 2 fast, 3 aggressive |
| `--layers N` | Active DiT blocks: 50 exact, 45 fast, 40 aggressive (min 35) |
| `--core-reuse N` | Alternative to `--reuse`: keep the transformer residual, refresh only the patch/head each step |
| `--token-reduction` | Pairs horizontal video tokens in middle blocks; faster, can shift composition |
| `--render-width/--render-height N` | Run the model internally smaller, then upscale with vImage |
| `--ssd-streaming` | See [Compute modes](#compute-modes) |
| `--use-int8-row-fc2` | M5-only faster (less conservative) int8 FC2 path |
| `--first-frame` / `--last-frame PATH` | FL2VA anchor conditioning |
| `--ref-image` / `--ref-video` / `--ref-silent-video` / `--ref-video-audio` / `--ref-audio` | Ref2VA ordered references — see [Features](#features) |
| `--show` | Live terminal preview (Kitty/Ghostty/iTerm2/WezTerm/Konsole) |
| `--profile` | Per-phase Metal timing, memory, and dispatch-count report |
| `--info` | Inspect model/device without generating |

Environment variables (`H3_ATTENTION_CACHE`, `H3_ATTENTION_CACHE_DIR`,
`H3_LORA_PATH`, `H3_LORA_SCALE`, `H3_TOKEN_REFINER_LORA`,
`H3_INT8_STREAM_MLP`, `H3_QWEN_PREFETCH*`, `H3_ZERO_COPY_WEIGHTS`,
`H3_VAE_TILE_PIXELS`, `H3_DIT_COMMAND_BLOCKS`, `H3_PROFILE`, and a long tail
of `H3_DISABLE_*`/`H3_USE_SLOWER_*`-style A/B diagnostic switches) select
alternate code paths for benchmarking or numerical comparison rather than
end-user tuning. They're documented at their point of use in the source
(start from [h3_dit.c](h3_dit.c), [h3_gpu.m](h3_gpu.m), and
[h3_attention_cache.c](h3_attention_cache.c)) and in the commit history —
each one exists because a specific optimization needed a same-process
oracle to A/B against, not as a supported end-user surface.

## License

MIT — see [LICENSE](LICENSE). Third-party notices (a Metal kernel design
adapted from ccv's FlashAttention implementation) are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Upstream project:
[antirez/h3.c](https://github.com/antirez/h3.c).
