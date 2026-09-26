import Foundation

// Mirrors PROFILES in gui/server.py: only these validated combinations of
// output size and (optional) lower internal render size are offered, rather
// than letting arbitrary width/height reach h3_generate. Do not add sizes
// (720p, 16:9, ...) that aren't one of these confirmed profiles.
enum SizeProfile: String, CaseIterable, Identifiable {
    // Declaration order matters: profiles(for:).first is each shape's
    // default, so the previously-only upscaled variant stays first/default
    // and the native (pre-upscale, half-size) variant is the alternative.
    case smallSquare, square, landscapeUpscaled, landscapeNative, portraitUpscaled, portraitNative
    var id: String { rawValue }

    var label: String { "\(shape.label) \(resolutionLabel)" }

    var resolutionLabel: String {
        let d = dimensions
        return "\(d.width)×\(d.height)"
    }

    var shape: AspectShape {
        switch self {
        case .smallSquare, .square: return .square
        case .landscapeUpscaled, .landscapeNative: return .landscape
        case .portraitUpscaled, .portraitNative: return .portrait
        }
    }

    var dimensions: (width: Int32, height: Int32, renderWidth: Int32, renderHeight: Int32) {
        switch self {
        case .smallSquare: return (256, 256, 0, 0)
        case .square: return (512, 512, 0, 0)
        case .landscapeUpscaled: return (1344, 768, 672, 384)
        // Same aspect as landscapeUpscaled at half the size, generated
        // directly at output size (no internal upscale) - render size 0
        // means "exact output canvas" the same way square does.
        case .landscapeNative: return (672, 384, 0, 0)
        case .portraitUpscaled: return (768, 1344, 384, 672)
        case .portraitNative: return (384, 672, 0, 0)
        }
    }

    static func profiles(for shape: AspectShape) -> [SizeProfile] {
        allCases.filter { $0.shape == shape }
    }
}

// "画面の形": the spec asks for a semantic ratio choice (横長/正方形/縦長)
// rather than raw dimensions at the top level - only the three shapes that
// SizeProfile actually offers, never a fabricated 16:9/9:16.
enum AspectShape: String, CaseIterable, Identifiable {
    case landscape, square, portrait
    var id: String { rawValue }

    var label: String {
        switch self {
        case .landscape: return "横長"
        case .square: return "正方形"
        case .portrait: return "縦長"
        }
    }

    var systemImage: String {
        switch self {
        case .landscape: return "rectangle"
        case .square: return "square"
        case .portrait: return "rectangle.portrait"
        }
    }
}

// "作り方": text-to-video vs. starting from an image. A second choice
// (ImageInputMode) narrows which image-driven mode within "画像から".
enum CreationMethod: String, CaseIterable, Identifiable {
    case text, image
    var id: String { rawValue }

    var label: String {
        switch self {
        case .text: return "文章から"
        case .image: return "画像から"
        }
    }
}

// FL2VA (first/last frame) and Ref2VA (reference images/videos/audio) use
// different transformer checkpoints/caches and can't be mixed (see
// build_job() in gui/server.py). Ref2VA itself accepts still images,
// videos, and audio-only files as references (H3ReferenceKind in
// H3Engine.swift) - the case name predates video/audio support here and
// stays for compatibility.
enum ImageInputMode: String, CaseIterable, Identifiable {
    case firstLastFrame, referenceImage
    var id: String { rawValue }

    var label: String {
        switch self {
        case .firstLastFrame: return "最初・最後の画像"
        case .referenceImage: return "参照画像・動画・音声"
        }
    }
}

// How the DiT weights are served. The three are mutually exclusive in the
// engine, not just in the UI:
// - attentionCache: pre-quantized int8 cache streamed from disk (+ streamed
//   MLP). Fast, needs the cache file, an M5-class GPU (tensor ops gate the
//   int8 path, h3_gpu.m) and supports LoRA.
// - resident: every DiT block loaded fully into memory at once instead of
//   streamed - h3_dit.c's plain load_block() path, the same one
//   attentionCache falls back on when H3_ATTENTION_CACHE isn't set. Needs
//   no cache file (quantizes to int8 in place at load time on a tensor-
//   capable GPU, same ~28s cost as building a cache, just not saved to
//   disk; falls back to full BF16 residency on older GPUs), supports LoRA,
//   but needs enough RAM to hold that residency - meant for a Mac with
//   memory to spare, not the default choice.
// - ssdStreaming: the original BF16 checkpoint, two blocks resident at a
//   time. No cache or quantization needed and far less memory, but slower,
//   and the engine has no LoRA path for it (h3_dit.c only fuses LoRA when
//   loading resident/cache blocks).
enum ComputeMode: String, CaseIterable, Identifiable, Codable {
    case attentionCache, resident, ssdStreaming
    var id: String { rawValue }

    var label: String {
        switch self {
        case .attentionCache: return "高速（int8キャッシュ）"
        case .resident: return "常駐（大容量メモリ向け、キャッシュ不要）"
        case .ssdStreaming: return "省メモリ（SSDストリーミング）"
        }
    }

    var summaryLabel: String {
        switch self {
        case .attentionCache: return "int8キャッシュ"
        case .resident: return "常駐モード"
        case .ssdStreaming: return "SSDストリーミング"
        }
    }
}

// Immutable snapshot of exactly what a completed generation used - kept
// separate from the live, still-editable draft in GenerationViewModel so
// showing "設定を見る" or reusing it never depends on (and is never
// clobbered by) whatever the user has since typed. See invariant #1/#10 in
// the design spec.
struct ResolvedResult {
    let prompt: String
    let creationMethod: CreationMethod
    let imageInputMode: ImageInputMode?
    let sizeProfile: SizeProfile
    let requestedSeconds: Int
    let requestedFrames: Int
    let actualFrameCount: Int
    let fps: Int
    var actualDurationSeconds: Double?
    let steps: Int
    let denoiseReuse: Int
    let computeMode: ComputeMode
    let seed: UInt64
    let seedWasRandom: Bool
    let loraPath: String?
    let loraScale: Float?
    let deviceLine: String
    let completedAt: Date

    var seedDecimalString: String { String(seed) }

    var settingsSummaryText: String {
        var lines = [
            "動画の内容: \(prompt)",
            "作り方: \(creationMethod.label)" + (imageInputMode.map { "（\($0.label)）" } ?? ""),
            "画面の形: \(sizeProfile.label)",
            "長さ: 指定\(requestedSeconds)秒 / 実測\(actualDurationSeconds.map { String(format: "%.1f秒", $0) } ?? "不明")",
            "生成ステップ数: \(steps)",
            "ノイズ除去の再利用（reuse）: \(denoiseReuse)",
            "計算方式: \(computeMode.label)",
            "シード: \(seedDecimalString)" + (seedWasRandom ? "（毎回変える設定で決定）" : "（固定）"),
        ]
        if let loraPath {
            lines.append("追加モデル（LoRA）: \(URL(fileURLWithPath: loraPath).lastPathComponent)")
        }
        lines.append("動作環境: \(deviceLine)")
        return lines.joined(separator: "\n")
    }
}
