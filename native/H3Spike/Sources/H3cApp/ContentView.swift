import AppKit
import AVKit
import Foundation
import H3Engine
import SwiftUI
import UniformTypeIdentifiers

// SwiftUI's `VideoPlayer` crashes on this OS build: its `_AVKit_SwiftUI`
// bridging type fails to resolve generic metadata for VideoPlayerView's
// superclass at runtime (Swift runtime bug, not app code - see
// getSuperclassMetadata in the crash report). Wrapping plain AppKit
// AVPlayerView ourselves avoids that code path entirely.
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        // AVKit's own transport bar only fades in on mouse hover and turned
        // out to be easy to miss; a small always-visible SwiftUI bar below
        // (see VideoPlaybackModel/SimpleVideoPlayer) replaces it entirely.
        view.controlsStyle = .none
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}

@MainActor
final class VideoPlaybackModel: ObservableObject {
    let player: AVPlayer
    @Published var isPlaying = false
    @Published var currentTime: Double = 0
    @Published var duration: Double = 1
    private var timeObserverToken: Any?

    private var endObserver: NSObjectProtocol?

    init(url: URL) {
        player = AVPlayer(url: url)
        timeObserverToken = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.1, preferredTimescale: 600), queue: .main
        ) { [weak self] time in
            MainActor.assumeIsolated { self?.currentTime = time.seconds }
        }
        Task {
            if let asset = player.currentItem?.asset,
               let loadedDuration = try? await asset.load(.duration) {
                self.duration = max(loadedDuration.seconds, 0.1)
            }
        }
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: player.currentItem, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.isPlaying = false
                self.player.seek(to: .zero)
            }
        }
        player.play()
        isPlaying = true
    }

    func togglePlayback() {
        isPlaying.toggle()
        isPlaying ? player.play() : player.pause()
    }

    func seek(to seconds: Double) {
        // A zero-tolerance seek is exact but slow enough per call that
        // repeated calls while dragging a slider mostly just queue up and
        // lag behind; a small tolerance keeps scrubbing responsive.
        let time = CMTime(seconds: seconds, preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: CMTime(seconds: 0.05, preferredTimescale: 600),
                    toleranceAfter: CMTime(seconds: 0.05, preferredTimescale: 600))
    }

    func cleanup() {
        if let timeObserverToken {
            player.removeTimeObserver(timeObserverToken)
        }
        timeObserverToken = nil
        if let endObserver {
            NotificationCenter.default.removeObserver(endObserver)
        }
        endObserver = nil
    }
}

struct SimpleVideoPlayer: View {
    @StateObject private var model: VideoPlaybackModel
    let sourceURL: URL
    let aspectRatio: CGFloat
    // While the user is dragging, the slider shows this instead of
    // model.currentTime - otherwise the periodic time observer (which lags
    // one seek behind while scrubbing) snaps the thumb back every ~0.1s and
    // dragging looks like it does nothing.
    @State private var isScrubbing = false
    @State private var scrubTime: Double = 0

    init(url: URL, aspectRatio: CGFloat) {
        sourceURL = url
        self.aspectRatio = aspectRatio
        _model = StateObject(wrappedValue: VideoPlaybackModel(url: url))
    }

    var body: some View {
        VStack(spacing: 6) {
            PlayerView(player: model.player)
                .aspectRatio(aspectRatio, contentMode: .fit)
                .frame(maxWidth: .infinity, maxHeight: 360)
            HStack {
                Button(action: model.togglePlayback) {
                    Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                }
                .buttonStyle(.borderless)
                Slider(
                    value: Binding(
                        get: { isScrubbing ? scrubTime : model.currentTime },
                        set: { newValue in
                            scrubTime = newValue
                            model.seek(to: newValue)
                        }
                    ),
                    in: 0 ... model.duration,
                    onEditingChanged: { editing in
                        isScrubbing = editing
                        if !editing { model.seek(to: scrubTime) }
                    }
                )
                Text(String(format: "%.1fs / %.1fs", model.currentTime, model.duration))
                    .font(.caption)
                    .monospacedDigit()
            }
            HStack {
                Button("Save As…") { saveVideo(from: sourceURL) }
                Spacer()
            }
        }
        .onDisappear { model.cleanup() }
    }
}

private func saveVideo(from sourceURL: URL) {
    let panel = NSSavePanel()
    panel.nameFieldStringValue = sourceURL.lastPathComponent
    panel.allowedContentTypes = [.mpeg4Movie]
    guard panel.runModal() == .OK, let destination = panel.url else { return }
    try? FileManager.default.removeItem(at: destination)
    do {
        try FileManager.default.copyItem(at: sourceURL, to: destination)
    } catch {
        NSAlert(error: error).runModal()
    }
}

private func chooseFile(allowedContentTypes: [UTType]) -> String? {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.canChooseFiles = true
    panel.allowedContentTypes = allowedContentTypes
    return panel.runModal() == .OK ? panel.url?.path : nil
}

private func chooseFiles(allowedContentTypes: [UTType]) -> [String] {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    panel.canChooseFiles = true
    panel.allowedContentTypes = allowedContentTypes
    return panel.runModal() == .OK ? panel.urls.map(\.path) : []
}

private let imageTypes: [UTType] = [.image]
// "safetensors" has no registered system UTI, so this synthesizes a dynamic
// one from the extension - still filters the open panel correctly.
private let safetensorsTypes: [UTType] = [UTType(filenameExtension: "safetensors") ?? .data]

// Mirrors PROFILES in gui/server.py: only these three validated combinations
// of output size and (optional) lower internal render size are offered,
// rather than letting arbitrary width/height reach h3_generate.
enum SizeProfile: String, CaseIterable, Identifiable {
    case smallSquare, square, landscapeUpscaled, portraitUpscaled
    var id: String { rawValue }

    var label: String {
        switch self {
        case .smallSquare: return "Square 256×256"
        case .square: return "Square 512×512"
        case .landscapeUpscaled: return "Landscape 1344×768"
        case .portraitUpscaled: return "Portrait 768×1344"
        }
    }

    var dimensions: (width: Int32, height: Int32, renderWidth: Int32, renderHeight: Int32) {
        switch self {
        case .smallSquare: return (256, 256, 0, 0)
        case .square: return (512, 512, 0, 0)
        case .landscapeUpscaled: return (1344, 768, 672, 384)
        case .portraitUpscaled: return (768, 1344, 384, 672)
        }
    }
}

private let secondsRange = 1 ... 15
private let stepsRange = 1 ... 100

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

// Mirrors DEFAULT_ATTENTION_CACHE / REF2VA_ATTENTION_CACHE in gui/server.py.
// Dev-time absolute paths, same caveat as the rest of this spike: these
// 19GB caches can't be bundled into a distributable .app.
private let repoRoot = "/Users/kamahara/Documents/work/h3c-app"
private let defaultAttentionCache = repoRoot + "/dit_int8_v2.cache"
private let ref2vaAttentionCache = repoRoot + "/dit_int8_v2_ref2va.cache"

// FL2VA (first/last frame) and Ref2VA (reference images) use different
// transformer checkpoints/caches and can't be mixed (see build_job() in
// gui/server.py) - picking a mode up front instead of showing both image
// pickers at once makes the invalid combination unreachable, rather than
// catching it after the fact when Generate is pressed.
enum GenerationMode: String, CaseIterable, Identifiable {
    case textToVideo, firstLastFrame, referenceImage
    var id: String { rawValue }

    var label: String {
        switch self {
        case .textToVideo: return "Text-to-Video"
        case .firstLastFrame: return "First/Last Frame"
        case .referenceImage: return "Reference Image (Ref2VA)"
        }
    }
}

@MainActor
final class GenerationViewModel: ObservableObject {
    @Published var prompt: String = "A cat playing with a ball of yarn."
    @Published var deviceLine: String = "Loading model..."
    @Published var phase: String = ""
    @Published var progressFraction: Double = 0
    @Published var framesDone: Int = 0
    @Published var framesTotal: Int = 0
    @Published var isGenerating = false
    @Published var errorMessage: String?
    @Published var resultURL: URL?
    @Published var resultAspectRatio: CGFloat = 1
    @Published var lastSeedUsed: UInt64?

    @Published var sizeProfile: SizeProfile = .square
    @Published var seconds: Int = 5
    @Published var steps: Int = 20
    @Published var seedText: String = "" {
        didSet {
            let digitsOnly = seedText.filter(\.isNumber)
            if digitsOnly != seedText { seedText = digitsOnly }
        }
    }

    @Published var mode: GenerationMode = .textToVideo {
        didSet {
            guard mode != oldValue else { return }
            switch mode {
            case .textToVideo:
                firstFramePath = nil
                lastFramePath = nil
                referenceImages = []
            case .firstLastFrame:
                referenceImages = []
            case .referenceImage:
                firstFramePath = nil
                lastFramePath = nil
            }
        }
    }
    @Published var firstFramePath: String?
    @Published var lastFramePath: String?
    @Published var referenceImages: [H3ReferenceInput] = []

    // H3_LORA_PATH/H3_LORA_SCALE (see H3GenerationParams.loraPath) - any
    // diffusers/peft-format adapter, not just a pre-baked Turbo cache. Works
    // for either mode as long as the file matches the transformer that mode
    // loads (FL2VA vs Ref2VA); nothing here can check that ahead of time.
    @Published var loraPath: String?
    @Published var loraScaleText: String = "" {
        didSet {
            let filtered = loraScaleText.filter { $0.isNumber || $0 == "." || $0 == "-" }
            if filtered != loraScaleText { loraScaleText = filtered }
        }
    }

    private var engine: H3Engine?
    private var generationTask: Task<Void, Never>?

    init() {
        // Sweep anything a previous run left behind (crash, force quit) -
        // generated previews are meant to be throwaway unless the user
        // explicitly uses Save As, which copies them out.
        Self.sweepStaleTempFiles()
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.deleteCurrentPreview() }
        }
    }

    private static func sweepStaleTempFiles() {
        let directory = NSTemporaryDirectory()
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: directory) else { return }
        for name in names where name.hasPrefix("h3c-app_") && name.hasSuffix(".mp4") {
            try? FileManager.default.removeItem(atPath: directory + name)
        }
    }

    private func deleteCurrentPreview() {
        if let resultURL {
            try? FileManager.default.removeItem(at: resultURL)
        }
    }

    func loadModel() {
        let modelDir = NSHomeDirectory() + "/Library/Application Support/h3c-analysis/MiniMax-H3"
        do {
            let engine = try H3Engine(modelDirectory: modelDir)
            self.engine = engine
            if let device = engine.device {
                deviceLine = "\(device.name) · \(device.architecture)"
            } else {
                deviceLine = "Model loaded (device info unavailable)"
            }
        } catch {
            errorMessage = error.localizedDescription
            deviceLine = "Model load failed"
        }
    }

    func pickLoRA() {
        if let path = chooseFile(allowedContentTypes: safetensorsTypes) { loraPath = path }
    }

    func pickFirstFrame() {
        if let path = chooseFile(allowedContentTypes: imageTypes) { firstFramePath = path }
    }

    func pickLastFrame() {
        if let path = chooseFile(allowedContentTypes: imageTypes) { lastFramePath = path }
    }

    func addReferenceImages() {
        for path in chooseFiles(allowedContentTypes: imageTypes) {
            referenceImages.append(H3ReferenceInput(kind: .image, path: path))
        }
    }

    func removeReferenceImages(at offsets: IndexSet) {
        referenceImages.remove(atOffsets: offsets)
    }

    func moveReferenceImages(from source: IndexSet, to destination: Int) {
        referenceImages.move(fromOffsets: source, toOffset: destination)
    }

    func generate() {
        guard let engine, !isGenerating else { return }
        // mode already keeps firstFramePath/lastFramePath and referenceImages
        // mutually exclusive - see GenerationMode.didSet.

        deleteCurrentPreview()
        isGenerating = true
        errorMessage = nil
        resultURL = nil
        lastSeedUsed = nil
        phase = ""
        progressFraction = 0
        framesDone = 0
        framesTotal = 0

        let outputPath = NSTemporaryDirectory() + "h3c-app_\(Int(Date().timeIntervalSince1970)).mp4"
        let dimensions = sizeProfile.dimensions
        var params = H3GenerationParams()
        params.width = dimensions.width
        params.height = dimensions.height
        params.renderWidth = dimensions.renderWidth
        params.renderHeight = dimensions.renderHeight
        params.frames = h3AlignedFrameCount(seconds: Double(seconds.clamped(to: secondsRange)))
        params.steps = Int32(steps.clamped(to: stepsRange))
        params.seed = UInt64(seedText) ?? UInt64.random(in: UInt64.min ... UInt64.max)
        params.firstFrame = firstFramePath
        params.lastFrame = lastFramePath
        params.references = referenceImages
        params.attentionCachePath = referenceImages.isEmpty ? defaultAttentionCache : ref2vaAttentionCache
        params.loraPath = loraPath
        params.loraScale = Float(loraScaleText)

        let promptCopy = prompt
        generationTask = Task {
            do {
                for try await event in engine.generate(prompt: promptCopy, outputPath: outputPath, params: params) {
                    switch event {
                    case .progress(let phase, let completed, let total):
                        self.phase = phase
                        self.progressFraction = total > 0 ? Double(completed) / Double(total) : 0
                    case .frame(let index, let count, _, _):
                        self.framesDone = index + 1
                        self.framesTotal = count
                    case .preview:
                        break
                    case .finished(let result):
                        self.phase = "Done: \(result.frames) frames @ \(result.fps)fps"
                        self.resultURL = URL(fileURLWithPath: result.outputPath)
                        self.resultAspectRatio = CGFloat(dimensions.width) / CGFloat(dimensions.height)
                        self.lastSeedUsed = result.seed
                    }
                }
            } catch {
                self.errorMessage = error.localizedDescription
            }
            self.isGenerating = false
        }
    }

    func cancel() {
        engine?.cancelCurrentGeneration()
    }
}

private struct FramePickerRow: View {
    let label: String
    @Binding var path: String?
    let choose: () -> Void

    var body: some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary).frame(width: 70, alignment: .leading)
            Text(path.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "None")
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button("Choose…", action: choose)
            if path != nil {
                Button("Clear") { path = nil }
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()

    var body: some View {
        ScrollView {
            formBody
                .padding(20)
                .frame(maxWidth: 640)
        }
        .frame(minWidth: 480, minHeight: 400, idealHeight: 780)
        .onAppear { viewModel.loadModel() }
    }

    private var formBody: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("h3c-app").font(.title2).bold()
            Text(viewModel.deviceLine).font(.caption).foregroundStyle(.secondary)

            TextField("Prompt", text: $viewModel.prompt, axis: .vertical)
                .lineLimit(2 ... 4)
                .textFieldStyle(.roundedBorder)

            Picker("Size", selection: $viewModel.sizeProfile) {
                ForEach(SizeProfile.allCases) { profile in
                    Text(profile.label).tag(profile)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 280)

            HStack(spacing: 20) {
                Stepper("Duration: \(viewModel.seconds)s", value: $viewModel.seconds, in: secondsRange)
                Stepper("Steps: \(viewModel.steps)", value: $viewModel.steps, in: stepsRange)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("LoRA (optional)").font(.caption).foregroundStyle(.secondary)
                        .frame(width: 90, alignment: .leading)
                    Text(viewModel.loraPath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "None")
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button("Choose…") { viewModel.pickLoRA() }
                    if viewModel.loraPath != nil {
                        Button("Clear") { viewModel.loraPath = nil }
                    }
                }
                if viewModel.loraPath != nil {
                    HStack {
                        Text("Scale (blank = auto)").font(.caption).foregroundStyle(.secondary)
                            .frame(width: 90, alignment: .leading)
                        TextField("auto", text: $viewModel.loraScaleText)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 80)
                    }
                    Text("Must match the loaded transformer (FL2VA vs Ref2VA) - the file isn't checked ahead of time.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("Seed (blank = random)").font(.caption).foregroundStyle(.secondary)
                TextField("random", text: $viewModel.seedText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
            }

            Picker("Mode", selection: $viewModel.mode) {
                ForEach(GenerationMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            switch viewModel.mode {
            case .textToVideo:
                EmptyView()

            case .firstLastFrame:
                FramePickerRow(label: "First frame", path: $viewModel.firstFramePath, choose: viewModel.pickFirstFrame)
                FramePickerRow(label: "Last frame", path: $viewModel.lastFramePath, choose: viewModel.pickLastFrame)

            case .referenceImage:
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Reference images").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button("Add…") { viewModel.addReferenceImages() }
                    }
                    if !viewModel.referenceImages.isEmpty {
                        List {
                            ForEach(Array(viewModel.referenceImages.enumerated()), id: \.element.id) { index, reference in
                                Text("\(index + 1). \(URL(fileURLWithPath: reference.path).lastPathComponent)")
                                    .font(.caption)
                            }
                            .onDelete(perform: viewModel.removeReferenceImages)
                            .onMove(perform: viewModel.moveReferenceImages)
                        }
                        .frame(height: min(CGFloat(viewModel.referenceImages.count) * 24 + 8, 120))
                    }
                }
            }

            HStack {
                Button(viewModel.isGenerating ? "Generating…" : "Generate") {
                    viewModel.generate()
                }
                .disabled(viewModel.isGenerating)

                if viewModel.isGenerating {
                    Button("Cancel", role: .destructive) {
                        viewModel.cancel()
                    }
                }
            }

            if viewModel.isGenerating || !viewModel.phase.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: viewModel.progressFraction)
                    Text(viewModel.phase).font(.caption)
                    if viewModel.framesTotal > 0 {
                        Text("Frame \(viewModel.framesDone)/\(viewModel.framesTotal)").font(.caption)
                    }
                    if let seed = viewModel.lastSeedUsed {
                        Text("Seed used: \(seed)").font(.caption)
                    }
                }
            }

            if let error = viewModel.errorMessage {
                Text(error).foregroundStyle(.red)
            }

            if let url = viewModel.resultURL {
                SimpleVideoPlayer(url: url, aspectRatio: viewModel.resultAspectRatio)
            }
        }
    }
}
