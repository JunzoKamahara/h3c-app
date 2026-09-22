// Async Swift wrapper around libh3.a (see h3.h). Turns the C callback-driven
// h3_generate into an AsyncThrowingStream so SwiftUI can consume progress
// and frame events directly, in-process - no subprocess, no HTTP polling,
// unlike gui/server.py's job model.
import CH3
import Foundation

public enum H3GenerationEvent: Sendable {
    case progress(phase: String, completed: Int, total: Int)
    case frame(index: Int, count: Int, width: Int, height: Int)
    case preview(step: Int, steps: Int)
    case finished(H3GenerationResult)
}

public struct H3DeviceInfo: Sendable {
    public let name: String
    public let architecture: String
    public let unifiedMemory: Bool
    public let metal4: Bool
    /// Highest Apple GPU family the device reports supporting (h3_metal.m
    /// probes MTLGPUFamilyApple1...10 and keeps the highest match). 10 is the
    /// family Apple introduced the Metal 4 hardware tensor/matmul units
    /// (Neural Accelerators) with, first shipping in M5 - this is the real
    /// capability the int8 attention cache path needs, not the device name.
    public let appleGPUFamily: Int
    /// True when this device has the hardware tensor units the int8
    /// attention cache path relies on - mirrors h3_gpu.m's
    /// h3_device_has_tensor_ops(), which checks -supportsFamily: rather than
    /// matching "M5" in the device's marketing name.
    public var hasTensorHardware: Bool { appleGPUFamily >= 10 }
}

public struct H3GenerationResult: Sendable {
    public let frames: Int
    public let fps: Int
    public let seed: UInt64
    public let outputPath: String
}

public enum H3ReferenceKind: Sendable {
    case image, video, audio, videoAudio

    var cValue: h3_reference_kind {
        switch self {
        case .image: return H3_REFERENCE_IMAGE
        case .video: return H3_REFERENCE_VIDEO
        case .audio: return H3_REFERENCE_AUDIO
        case .videoAudio: return H3_REFERENCE_VIDEO_AUDIO
        }
    }
}

public struct H3ReferenceInput: Sendable, Identifiable {
    public let id = UUID()
    public var kind: H3ReferenceKind
    public var path: String
    public var audioPath: String?
    public var includeEmbeddedAudio: Bool

    public init(kind: H3ReferenceKind = .image, path: String, audioPath: String? = nil,
                includeEmbeddedAudio: Bool = false) {
        self.kind = kind
        self.path = path
        self.audioPath = audioPath
        self.includeEmbeddedAudio = includeEmbeddedAudio
    }
}

public struct H3GenerationParams: Sendable {
    public var width: Int32 = 512
    public var height: Int32 = 512
    public var renderWidth: Int32 = 0
    public var renderHeight: Int32 = 0
    public var frames: Int32 = 56
    public var steps: Int32 = 20
    public var seed: UInt64 = 42
    public var ditLayers: Int32 = 50
    public var denoiseReuse: Int32 = 1
    public var coreReuse: Int32 = 1
    public var firstFrame: String?
    public var lastFrame: String?
    public var references: [H3ReferenceInput] = []
    // libh3.a reads this from the environment rather than h3_params (see
    // H3_ATTENTION_CACHE in h3_dit.c), so H3Engine sets it just before each
    // call instead of once at process startup - different generation modes
    // (default/Ref2VA) need different prebuilt caches.
    public var attentionCachePath: String?
    // H3_LORA_PATH / H3_LORA_SCALE (see h3_lora.c/h3_dit.c on the
    // int8-cache-lora branch): fuses one diffusers/peft-format LoRA adapter
    // into the DiT weights at load time - works for either the resident
    // BF16 path or, combined with attentionCachePath, the streamed int8
    // path (which then transparently materializes and reuses a fused H3AC
    // cache keyed by the LoRA file's hash). nil scale means auto-detect the
    // adapter's own alpha/rank metadata, falling back to 1.0.
    public var loraPath: String?
    public var loraScale: Float?
    // h3_params.ssd_streaming: keep only two original BF16 DiT blocks
    // resident and read the next from the checkpoint while the GPU runs the
    // current one. It's an alternative to attentionCachePath, not an
    // addition: it runs on the unquantized BF16 weights, so the int8 cache
    // (and H3_INT8_STREAM_MLP) don't apply, and LoRA fusion isn't wired into
    // its layer-loading path at all (h3_dit.c only fuses in load_block /
    // load_block_norms_and_mlp) - the engine would silently skip it. So when
    // this is set, generate() clears the cache and LoRA settings instead of
    // passing along a combination that quietly does the wrong thing.
    public var ssdStreaming = false
    public init() {}
}

/* Matches align_frames() in gui/server.py, itself matching h3_align_frame_count
 * (h3_host.c): the engine only accepts frame counts of the form 5 + 17*k, so a
 * caller offering a "seconds" control has to round up to one itself. Not
 * calling into libh3.a for this since h3_align_frame_count lives in the
 * internal h3_host.h, not the public h3.h surface CH3 exposes. */
public func h3AlignedFrameCount(seconds: Double) -> Int32 {
    var requested = max(5, Int((seconds * 24).rounded()))
    let remainder = (requested - 5) % 17
    if remainder != 0 {
        requested += 17 - remainder
    }
    return Int32(requested)
}

/* Calls `body` with a C string for `string`, or nil if `string` is nil. */
private func withOptionalCString<R>(_ string: String?, _ body: (UnsafePointer<CChar>?) -> R) -> R {
    if let string {
        return string.withCString(body)
    }
    return body(nil)
}

/* Builds the h3_reference array by nesting one withCString/withOptionalCString
 * per path so all of them stay valid for the single call to `body`, without
 * resorting to manual strdup/free bookkeeping. */
private func withReferenceArray<R>(_ references: [H3ReferenceInput], built: [h3_reference] = [],
                                    _ body: (UnsafeBufferPointer<h3_reference>) -> R) -> R {
    guard let reference = references.first else {
        return built.withUnsafeBufferPointer(body)
    }
    let rest = Array(references.dropFirst())
    return reference.path.withCString { pathC in
        withOptionalCString(reference.audioPath) { audioC in
            var next = built
            next.append(h3_reference(kind: reference.kind.cValue, path: pathC, audio_path: audioC,
                                      include_embedded_audio: reference.includeEmbeddedAudio ? 1 : 0))
            return withReferenceArray(rest, built: next, body)
        }
    }
}

public enum H3EngineError: Error, LocalizedError {
    case loadFailed(String)
    case generationFailed(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let message): return "Failed to load model: \(message)"
        case .generationFailed(let message): return "Generation failed: \(message)"
        case .cancelled: return "Generation cancelled"
        }
    }
}

private func fixedCString<T>(_ tuple: T) -> String {
    withUnsafeBytes(of: tuple) { raw in
        String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
    }
}

/* A plain polling flag: the generation thread checks it from inside the C
 * callbacks, the caller (main actor) sets it when the user cancels. */
private final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return value
    }
    func cancel() {
        lock.lock(); value = true; lock.unlock()
    }
}

/* h3_frame_callback/h3_progress_callback are plain C function pointers, so
 * they can't capture Swift state directly. This bundles the stream
 * continuation and the cancel flag behind the `callback_opaque` pointer
 * instead, recovered on the other side with Unmanaged. */
private final class GenerationBridge: @unchecked Sendable {
    let continuation: AsyncThrowingStream<H3GenerationEvent, Error>.Continuation
    let cancelFlag: CancelFlag
    init(continuation: AsyncThrowingStream<H3GenerationEvent, Error>.Continuation, cancelFlag: CancelFlag) {
        self.continuation = continuation
        self.cancelFlag = cancelFlag
    }
}

private let h3ProgressTrampoline: h3_progress_callback = { phase, completed, total, opaque in
    guard let opaque else { return 0 }
    let bridge = Unmanaged<GenerationBridge>.fromOpaque(opaque).takeUnretainedValue()
    let phaseName = phase.map { String(cString: $0) } ?? ""
    bridge.continuation.yield(.progress(phase: phaseName, completed: Int(completed), total: Int(total)))
    return bridge.cancelFlag.isCancelled ? 1 : 0
}

private let h3FrameTrampoline: h3_frame_callback = { framePtr, opaque in
    guard let framePtr, let opaque else { return 0 }
    let bridge = Unmanaged<GenerationBridge>.fromOpaque(opaque).takeUnretainedValue()
    let frame = framePtr.pointee
    if frame.denoise_step >= 0 {
        bridge.continuation.yield(.preview(step: Int(frame.denoise_step), steps: Int(frame.denoise_steps)))
    } else {
        bridge.continuation.yield(.frame(index: Int(frame.frame_index), count: Int(frame.frame_count),
                                          width: Int(frame.width), height: Int(frame.height)))
    }
    return bridge.cancelFlag.isCancelled ? 1 : 0
}

public final class H3Engine: @unchecked Sendable {
    private let ctx: OpaquePointer
    private var currentCancelFlag: CancelFlag?

    public init(modelDirectory: String) throws {
        guard let ctx = h3_load_dir(modelDirectory) else {
            throw H3EngineError.loadFailed("h3_load_dir returned NULL")
        }
        self.ctx = ctx
    }

    deinit {
        h3_free(ctx)
    }

    public var device: H3DeviceInfo? {
        guard let device = h3_device(ctx)?.pointee else { return nil }
        return H3DeviceInfo(
            name: fixedCString(device.name),
            architecture: fixedCString(device.architecture),
            unifiedMemory: device.unified_memory != 0,
            metal4: device.metal4 != 0,
            appleGPUFamily: Int(device.apple_gpu_family)
        )
    }

    public func cancelCurrentGeneration() {
        currentCancelFlag?.cancel()
    }

    public func generate(prompt: String, outputPath: String,
                          params: H3GenerationParams) -> AsyncThrowingStream<H3GenerationEvent, Error> {
        let ctx = self.ctx
        return AsyncThrowingStream { continuation in
            let cancelFlag = CancelFlag()
            self.currentCancelFlag = cancelFlag
            let bridge = GenerationBridge(continuation: continuation, cancelFlag: cancelFlag)
            let bridgeHandle = Unmanaged.passRetained(bridge)

            DispatchQueue.global(qos: .userInitiated).async {
                defer { bridgeHandle.release() }

                // A GUI app that's occluded or in the background is eligible
                // for App Nap, which lowers CPU/disk I/O priority - measured
                // at ~1.5x slower on this pipeline (see taskpolicy -b trace).
                // Generation is user-requested work, so opt out for its
                // duration.
                let activity = ProcessInfo.processInfo.beginActivity(
                    options: [.userInitiated, .idleSystemSleepDisabled],
                    reason: "Generating video")
                defer { ProcessInfo.processInfo.endActivity(activity) }

                if params.ssdStreaming {
                    // Exclusive with the int8 cache and with LoRA - see
                    // H3GenerationParams.ssdStreaming.
                    unsetenv("H3_ATTENTION_CACHE")
                    unsetenv("H3_INT8_STREAM_MLP")
                    unsetenv("H3_LORA_PATH")
                    unsetenv("H3_LORA_SCALE")
                } else {
                    if let attentionCachePath = params.attentionCachePath {
                        setenv("H3_ATTENTION_CACHE", attentionCachePath, 1)
                        // Same pairing gui/server.py always used with a cache:
                        // stream FC1/FC2 from it too instead of keeping ~10.8GiB
                        // of int8 MLP resident for every block, which on a 24GB
                        // machine turns into memory pressure on longer clips
                        // (and measured +10s of setup even on a short one).
                        setenv("H3_INT8_STREAM_MLP", "1", 1)
                    } else {
                        unsetenv("H3_ATTENTION_CACHE")
                        unsetenv("H3_INT8_STREAM_MLP")
                    }
                    if let loraPath = params.loraPath, !loraPath.isEmpty {
                        setenv("H3_LORA_PATH", loraPath, 1)
                        if let loraScale = params.loraScale {
                            setenv("H3_LORA_SCALE", String(loraScale), 1)
                        } else {
                            unsetenv("H3_LORA_SCALE")
                        }
                    } else {
                        unsetenv("H3_LORA_PATH")
                        unsetenv("H3_LORA_SCALE")
                    }
                }
                // gui/server.py also pins the Qwen text-encoder prefetch
                // depth to 1 (default is 3 on M5) to keep its footprint down.
                if getenv("H3_QWEN_PREFETCH_DEPTH") == nil {
                    setenv("H3_QWEN_PREFETCH_DEPTH", "1", 1)
                }

                var cParams = h3_params()
                cParams.width = params.width
                cParams.height = params.height
                cParams.render_width = params.renderWidth
                cParams.render_height = params.renderHeight
                cParams.frames = params.frames
                cParams.steps = params.steps
                cParams.seed = params.seed
                cParams.dit_layers = params.ditLayers
                cParams.denoise_reuse = params.denoiseReuse
                cParams.core_reuse = params.coreReuse
                cParams.ssd_streaming = params.ssdStreaming ? 1 : 0
                cParams.on_progress = h3ProgressTrampoline
                cParams.on_frame = h3FrameTrampoline
                cParams.callback_opaque = bridgeHandle.toOpaque()

                let result: UnsafeMutablePointer<h3_result>? = withReferenceArray(params.references) { refBuffer in
                    outputPath.withCString { outputPathC in
                        prompt.withCString { promptC in
                            withOptionalCString(params.firstFrame) { firstFrameC in
                                withOptionalCString(params.lastFrame) { lastFrameC in
                                    cParams.output_path = outputPathC
                                    cParams.first_frame = firstFrameC
                                    cParams.last_frame = lastFrameC
                                    cParams.references = refBuffer.baseAddress
                                    cParams.reference_count = refBuffer.count
                                    return h3_generate(ctx, promptC, &cParams)
                                }
                            }
                        }
                    }
                }

                if let result {
                    let genResult = H3GenerationResult(
                        frames: Int(result.pointee.frames),
                        fps: Int(result.pointee.fps),
                        seed: result.pointee.seed,
                        outputPath: outputPath
                    )
                    h3_result_free(result)
                    continuation.yield(.finished(genResult))
                    continuation.finish()
                } else {
                    let message = h3_last_error(ctx).map { String(cString: $0) } ?? "unknown error"
                    continuation.finish(throwing: cancelFlag.isCancelled
                        ? H3EngineError.cancelled : H3EngineError.generationFailed(message))
                }
            }
        }
    }
}
