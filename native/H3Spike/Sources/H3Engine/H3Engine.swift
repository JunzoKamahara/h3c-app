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
}

public struct H3GenerationResult: Sendable {
    public let frames: Int
    public let fps: Int
    public let seed: UInt64
    public let outputPath: String
}

public struct H3GenerationParams: Sendable {
    public var width: Int32 = 256
    public var height: Int32 = 256
    public var frames: Int32 = 41
    public var steps: Int32 = 8
    public var seed: UInt64 = 42
    public var ditLayers: Int32 = 50
    public var denoiseReuse: Int32 = 1
    public var coreReuse: Int32 = 1
    public init() {}
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
            metal4: device.metal4 != 0
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

                var cParams = h3_params()
                cParams.width = params.width
                cParams.height = params.height
                cParams.frames = params.frames
                cParams.steps = params.steps
                cParams.seed = params.seed
                cParams.dit_layers = params.ditLayers
                cParams.denoise_reuse = params.denoiseReuse
                cParams.core_reuse = params.coreReuse
                cParams.on_progress = h3ProgressTrampoline
                cParams.on_frame = h3FrameTrampoline
                cParams.callback_opaque = bridgeHandle.toOpaque()

                let result: UnsafeMutablePointer<h3_result>? = outputPath.withCString { outputPathC in
                    prompt.withCString { promptC in
                        cParams.output_path = outputPathC
                        return h3_generate(ctx, promptC, &cParams)
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
