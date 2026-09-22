// Minimal spike: link libh3.a directly into a Swift executable and drive
// h3_load_dir -> h3_generate in-process, receiving progress/frames through
// the C callbacks instead of parsing subprocess stdout (as gui/server.py
// and the CLI's terminal front end currently do).
import CH3
import Foundation

/* h3_device_info's `name`/`architecture` fields are fixed-size C char arrays,
 * imported into Swift as anonymous tuples. Reading them as raw bytes avoids
 * having to spell out the tuple's exact arity. */
func fixedCString<T>(_ tuple: T) -> String {
    withUnsafeBytes(of: tuple) { raw in
        String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
    }
}

final class GenerationContext {
    var framesReceived = 0
    var previewsReceived = 0
}

let traceStart = Date()
setvbuf(stdout, nil, _IOLBF, 0)

let progressCallback: h3_progress_callback = { phase, completed, total, _ in
    let phaseName = phase.map { String(cString: $0) } ?? "?"
    print(String(format: "[%7.2fs] [progress] ", Date().timeIntervalSince(traceStart)) + "\(phaseName) \(completed)/\(total)")
    return 0 // return non-zero from here to cancel generation
}

let frameCallback: h3_frame_callback = { framePtr, opaque in
    guard let framePtr, let opaque else { return 0 }
    let frame = framePtr.pointee
    let context = Unmanaged<GenerationContext>.fromOpaque(opaque).takeUnretainedValue()
    if frame.denoise_step >= 0 {
        context.previewsReceived += 1
        print("[preview] denoise step \(frame.denoise_step + 1)/\(frame.denoise_steps)")
    } else {
        context.framesReceived += 1
        print(String(format: "[%7.2fs] [frame] ", Date().timeIntervalSince(traceStart)) + "\(frame.frame_index + 1)/\(frame.frame_count)")
    }
    return 0
}

let arguments = CommandLine.arguments
let modelDir = arguments.count > 1 ? arguments[1]
    : (NSHomeDirectory() + "/Library/Application Support/h3c-analysis/MiniMax-H3")
let outputPath = arguments.count > 2 ? arguments[2] : "/tmp/h3spike_output.mp4"
let prompt = arguments.count > 3 ? arguments[3] : "A cat playing with a ball of yarn."

print("== h3c-app native spike ==")
print("Model dir: \(modelDir)")

guard let ctx = h3_load_dir(modelDir) else {
    print("h3_load_dir returned NULL")
    exit(1)
}
defer { h3_free(ctx) }

if let device = h3_device(ctx)?.pointee {
    print("Device: \(fixedCString(device.name)) (\(fixedCString(device.architecture)))")
    print("  unified memory: \(device.unified_memory != 0), metal4: \(device.metal4 != 0)")
}

if let model = h3_model(ctx)?.pointee {
    print("FL2VA transformer: \(model.fl2va_transformer.bytes) bytes, \(model.fl2va_transformer.tensors) tensors")
}

// Point at the existing validated attention cache from the h3c working
// checkout so this spike runs the same fast int8 path the CLI/GUI use by
// default, rather than the slower close-reference BF16 path.
let useSSD = ProcessInfo.processInfo.environment["H3SPIKE_SSD"] == "1"
if getenv("H3_ATTENTION_CACHE") == nil && !useSSD {
    setenv("H3_ATTENTION_CACHE", "/Users/kamahara/Documents/work/h3c-app/dit_int8_v2.cache", 1)
}

let context = GenerationContext()
let opaque = Unmanaged.passUnretained(context).toOpaque()

var params = h3_params()
let sizeArg = Int32(ProcessInfo.processInfo.environment["H3SPIKE_SIZE"] ?? "") ?? 256
params.width = sizeArg
params.height = sizeArg
params.frames = Int32(ProcessInfo.processInfo.environment["H3SPIKE_FRAMES"] ?? "") ?? 9
params.steps = Int32(ProcessInfo.processInfo.environment["H3SPIKE_STEPS"] ?? "") ?? 4
params.seed = 42
params.dit_layers = 50
params.ssd_streaming = useSSD ? 1 : 0
params.denoise_reuse = 1
params.core_reuse = 1
params.reference_image_size = H3_REFERENCE_IMAGE_MATCH
params.on_progress = progressCallback
params.on_frame = frameCallback
params.callback_opaque = opaque

print("Generating \(params.width)x\(params.height), \(params.frames) frames, \(params.steps) steps...")
let start = Date()

let result: UnsafeMutablePointer<h3_result>? = outputPath.withCString { outputPathC in
    prompt.withCString { promptC in
        params.output_path = outputPathC
        return h3_generate(ctx, promptC, &params)
    }
}

guard let result else {
    let error = h3_last_error(ctx).map { String(cString: $0) } ?? "unknown error"
    print("h3_generate failed: \(error)")
    exit(1)
}
defer { h3_result_free(result) }

let elapsed = Date().timeIntervalSince(start)
print(String(format: "Done in %.1fs", elapsed))
print("Result: \(result.pointee.frames) frames @ \(result.pointee.fps)fps, seed=\(result.pointee.seed)")
print("Frames delivered via on_frame: \(context.framesReceived), previews: \(context.previewsReceived)")
print("Output written to: \(outputPath)")
