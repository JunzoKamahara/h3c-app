import AVKit
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
        // AVKit's own transport bar only fades in on mouse hover and turned
        // out to be easy to miss; a small always-visible SwiftUI bar below
        // (see VideoPlaybackModel/ResultPlayerView) replaces it entirely.
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
        // "新しい結果への切替時は先頭フレームで停止。ユーザーが再生を開始する。"
        // (design spec section 7) - don't autoplay.
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

// Playback only - export/settings/reuse live in ResultActionsView (see
// PreviewPane.swift), matching the design spec's split between "右ペイン
// 中央" (the player) and "右ペイン下" (result actions).
struct ResultPlayerView: View {
    @StateObject private var model: VideoPlaybackModel
    let aspectRatio: CGFloat
    // While the user is dragging, the slider shows this instead of
    // model.currentTime - otherwise the periodic time observer (which lags
    // one seek behind while scrubbing) snaps the thumb back every ~0.1s and
    // dragging looks like it does nothing.
    @State private var isScrubbing = false
    @State private var scrubTime: Double = 0

    init(url: URL, aspectRatio: CGFloat) {
        self.aspectRatio = aspectRatio
        _model = StateObject(wrappedValue: VideoPlaybackModel(url: url))
    }

    var body: some View {
        VStack(spacing: H3Spacing.sm) {
            PlayerView(player: model.player)
                .aspectRatio(aspectRatio, contentMode: .fit)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: H3Radius.stage))
            HStack {
                Button(action: model.togglePlayback) {
                    Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel(model.isPlaying ? "一時停止" : "再生")
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
                .accessibilityLabel("再生位置")
                Text(String(format: "%.1fs / %.1fs", model.currentTime, model.duration))
                    .font(.caption)
                    .monospacedDigit()
            }
        }
        .onDisappear { model.cleanup() }
    }
}
