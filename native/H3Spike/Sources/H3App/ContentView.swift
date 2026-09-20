import AppKit
import AVKit
import Foundation
import H3Engine
import SwiftUI

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
        view.controlsStyle = .inline
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
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

    private var engine: H3Engine?
    private var generationTask: Task<Void, Never>?

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

    func generate() {
        guard let engine, !isGenerating else { return }
        isGenerating = true
        errorMessage = nil
        resultURL = nil
        phase = ""
        progressFraction = 0
        framesDone = 0
        framesTotal = 0

        let outputPath = NSTemporaryDirectory() + "h3app_\(Int(Date().timeIntervalSince1970)).mp4"
        var params = H3GenerationParams()
        params.width = 256
        params.height = 256
        params.frames = 41
        params.steps = 8

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

struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("h3c-app").font(.title2).bold()
            Text(viewModel.deviceLine).font(.caption).foregroundStyle(.secondary)

            TextField("Prompt", text: $viewModel.prompt, axis: .vertical)
                .lineLimit(2 ... 4)
                .textFieldStyle(.roundedBorder)

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
                }
            }

            if let error = viewModel.errorMessage {
                Text(error).foregroundStyle(.red)
            }

            if let url = viewModel.resultURL {
                PlayerView(player: AVPlayer(url: url))
                    .frame(minWidth: 320, minHeight: 320)
            }

            Spacer()
        }
        .padding(20)
        .frame(minWidth: 420, minHeight: 480)
        .onAppear { viewModel.loadModel() }
    }
}
