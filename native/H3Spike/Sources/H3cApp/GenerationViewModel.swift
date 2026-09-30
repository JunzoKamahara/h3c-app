import AppKit
import AVFoundation
import Combine
import Foundation
import H3Engine

let secondsRange = 1 ... 15
let stepsRange = 3 ... 20
// h3.c rejects anything outside [1, 3] ("denoise reuse must be in [1, 3]");
// gui/server.py's looser 1-6 check just let the engine reject 4+ later.
let reuseRange = 1 ... 3
// The real, existing engine defaults (H3GenerationParams / H3_DEFAULT_STEPS)
// - used only to detect whether "詳細設定" has been changed from them, not
// as a claim that these are the only valid values.
private let defaultSteps = 20
// 1 = the close-reference path (h3.h); the old web GUI defaulted to 2 (the
// validated fast path) - kept at 1 here so output doesn't silently change.
private let defaultReuse = 1
private let defaultSizeProfile: SizeProfile = .square

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

// Mirrors DEFAULT_ATTENTION_CACHE / REF2VA_ATTENTION_CACHE in gui/server.py,
// but under this user's Application Support instead of a dev checkout path -
// these 19GB caches can't be bundled into a distributable .app, and a path
// under a specific developer's home directory would never resolve on any
// other machine. Each registered model gets its own cache subdirectory (see
// GenerationViewModel.attentionCacheDirectory(for:)) now that ModelLibrary
// allows more than one - validationMessage below just reports a cache
// missing until one is built with build_attention_cache, in-app, or dropped
// in by hand.
// Not private: ModelLibrary.swift (same module, different file) also needs
// this base path to build the default download destination it suggests to
// brand-new installs - see defaultH3ModelDownloadPath there.
let h3AppSupportDirectory: String = {
    let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?.path
        ?? (NSHomeDirectory() + "/Library/Application Support")
    return base + "/h3c-app"
}()
// Paired with legacyDefaultH3ModelPath (ModelLibrary.swift) - moved
// alongside it, from h3AppSupportDirectory + "/cache" to ~/models/cache, so
// this dev's ~36GB of caches live next to the model instead of in
// ~/Library.
private let legacyCacheDirectory = NSHomeDirectory() + "/models/cache"

enum EngineState: Equatable {
    case loading
    case ready
    case failed(String)
}

@MainActor
final class GenerationViewModel: ObservableObject {
    // MARK: Engine readiness (separate from job state and results - design
    // spec invariant: "エンジン準備状態・生成ジョブ状態・前回結果は別々に保持する")
    @Published private(set) var engineState: EngineState = .loading
    @Published private(set) var deviceLine: String = "モデルを準備しています…"
    @Published private(set) var modelDirectory: String = ""
    let library = ModelLibrary()

    // MARK: Draft - editable settings, never overwritten by a running job or
    // a past result (design spec invariant #1).
    @Published var prompt: String = "A cat playing with a ball of yarn."
    @Published var creationMethod: CreationMethod = .text
    @Published var imageInputMode: ImageInputMode = .firstLastFrame
    @Published var sizeProfile: SizeProfile = defaultSizeProfile
    @Published var seconds: Int = 5
    @Published var steps: Int = defaultSteps
    @Published var denoiseReuse: Int = defaultReuse
    // See ComputeMode: cache, resident and SSD streaming are mutually
    // exclusive; LoRA works with all three.
    @Published var computeMode: ComputeMode = .attentionCache
    // The int8 path (and so the cache) needs the Metal 4 hardware tensor
    // units first shipped in M5 - h3_gpu.m enables tensor ops based on the
    // device's actual supportsFamily: capability (H3DeviceInfo.hasTensorHardware),
    // not its marketing name. Set from the real device in loadModel().
    @Published private(set) var supportsInt8Cache = true
    var defaultComputeMode: ComputeMode { supportsInt8Cache ? .attentionCache : .ssdStreaming }
    @Published var speedMode: SpeedMode = .quality
    // Opt-in fast mode (experimental): ccv's int8 attention, M5 only. Off by
    // default, offered only where the engine reports it can run, and kept
    // separate from speedMode - no speed preset turns it on.
    @Published var fastAttention = false
    let fastAttentionAvailable = H3Engine.fastAttentionAvailable

    struct SpeedSettings {
        var ditLayers: Int32 = 50
        var coreReuse: Int32 = 1
        var tokenReduction = false
    }

    /// The engine options behind speedMode for the current draft. Measured on
    /// an M5 (512x512, 39 frames, 20 steps, int8 cache): 232.6s exact, 82.3s
    /// fast (2.8x), 73.7s fastest (3.2x), both still sharp and coherent.
    /// Stacking the engine's most aggressive values instead (40 layers, core
    /// reuse 6, token reduction, int8 row FC2) reached 61.0s but visibly
    /// smeared the subject, and int8 row FC2 alone bought nothing.
    ///
    /// Core reuse refreshes the transformer core only every N steps, so it's
    /// scaled to the step count (a 4-step Turbo run has nothing to reuse
    /// across) and dropped when the separate whole-velocity reuse is on -
    /// the engine rejects combining the two.
    var speedSettings: SpeedSettings {
        var settings = SpeedSettings()
        guard speedMode != .quality else { return settings }
        settings.ditLayers = 45
        settings.coreReuse = denoiseReuse > 1 ? 1 :
            Int32(max(1, min(4, steps.clamped(to: stepsRange) / 5)))
        settings.tokenReduction = speedMode == .fastest
        return settings
    }
    @Published var seedText: String = "" {
        didSet {
            let digitsOnly = seedText.filter(\.isNumber)
            if digitsOnly != seedText { seedText = digitsOnly }
        }
    }
    // Not cleared on mode switches - design spec section 6: "モード切替では
    // 画像を消さない...画像からに戻すと復元する。" generate() below decides
    // which of these actually reach the engine based on the *current* mode.
    @Published var firstFramePath: String?
    @Published var lastFramePath: String?
    @Published var referenceImages: [H3ReferenceInput] = []

    /// The LoRA stack is chosen from ModelLibrary's registered library
    /// (entries switched on there or in the form) and applies in every
    /// compute mode.
    var effectiveLoRAs: [ResolvedLoRA] {
        library.enabledLoRAs.map { ResolvedLoRA(name: $0.name, path: $0.path, strength: $0.strength) }
    }

    /// Where the active model's attention cache files live. Isolated per
    /// registered model (by id) so switching models never risks streaming
    /// one model's cache against another's weights. The one exception: the
    /// very first model this app ever used didn't have this isolation and
    /// already has real ~19 GiB caches built at the flat legacy path -
    /// reuse that path for exactly that entry so upgrading doesn't demand
    /// rebuilding them.
    private func attentionCacheDirectory(for model: H3ModelEntry) -> String {
        if model.path == legacyDefaultH3ModelPath,
           FileManager.default.fileExists(atPath: legacyCacheDirectory) {
            return legacyCacheDirectory
        }
        return h3AppSupportDirectory + "/cache/" + model.id.uuidString
    }

    /// The int8 cache this job would use - which one depends on whether it
    /// runs the Ref2VA (reference image) transformer, and on the active
    /// model's own cache directory.
    var currentAttentionCachePath: String {
        guard let model = library.activeModel else { return "" }
        let usesReferences = creationMethod == .image && imageInputMode == .referenceImage
        let directory = attentionCacheDirectory(for: model)
        return directory + (usesReferences ? "/dit_int8_v2_ref2va.cache" : "/dit_int8_v2.cache")
    }

    /// The transformer directory currentAttentionCachePath would be built
    /// from - same reference-mode check, matching build_attention_cache's
    /// own FL2VA/Ref2VA convention (h3_dit.c's detect_model_kind()).
    var currentTransformerDirectory: String {
        let usesReferences = creationMethod == .image && imageInputMode == .referenceImage
        return modelDirectory + (usesReferences ? "/Ref2VA/transformer" : "/FL2VA/transformer")
    }

    var attentionCacheMissing: Bool {
        !FileManager.default.fileExists(atPath: currentAttentionCachePath)
    }

    /// What a generation started now would actually request.
    var useFastAttention: Bool { fastAttention && fastAttentionAvailable }

    var hasAdvancedChanges: Bool {
        sizeProfile != defaultSizeProfile || steps != defaultSteps || denoiseReuse != defaultReuse
            || computeMode != defaultComputeMode || speedMode != .quality || fastAttention
            || !seedText.isEmpty
            || !library.enabledLoRAs.isEmpty
    }

    /// Max-resolution + long duration + SSD streaming measured as
    /// impractically slow on a 24GB M5 Mac: the run pushed past physical
    /// memory into swap (SSD streaming's own disk reads then compete with
    /// OS paging on the same disk), and a single denoise step never
    /// finished in over two and a half hours at 1344x768/15s. Smaller
    /// canvases (e.g. 512x512-class) complete in the expected few minutes
    /// with no swapping, so this only warns at the largest profiles with a
    /// double-digit duration - not a precise model, just a conservative
    /// flag for the one combination actually observed to be this bad.
    var isHeavySsdStreamingConfig: Bool {
        guard computeMode == .ssdStreaming else { return false }
        let isMaxResolution = sizeProfile == .landscapeUpscaled || sizeProfile == .portraitUpscaled
        return isMaxResolution && seconds >= 10
    }

    /// Resident mode holds every DiT block in memory at once (h3_dit.c's
    /// load_block() path, no streaming) - on a tensor-capable GPU that's
    /// int8-quantized (comparable to the ~18 GiB attention-cache file, just
    /// not written to disk); on any other GPU it stays the full ~37 GiB
    /// BF16 checkpoint. Both are on top of the text encoder and VAEs this
    /// process already holds - a rough, unmeasured threshold (unlike
    /// isHeavySsdStreamingConfig's benchmarked one) just to steer a
    /// memory-constrained Mac back to attentionCache/ssdStreaming instead.
    var isLowMemoryForResident: Bool {
        guard computeMode == .resident else { return false }
        let physicalGiB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return physicalGiB < (supportsInt8Cache ? 32.0 : 64.0)
    }

    // MARK: Attention cache build - separate from generation job state,
    // same invariant as engine/job/result: none of these overwrite each
    // other. Triggered from the "int8キャッシュが見つかりません" message
    // above instead of just telling the user to go run a CLI tool.
    @Published private(set) var isBuildingCache = false
    @Published private(set) var cacheBuildProgress: Double?
    @Published var cacheBuildError: String?
    private var cacheBuildTask: Task<Void, Never>?

    func buildMissingAttentionCache() {
        guard let engine, !isBuildingCache, let model = library.activeModel else { return }
        isBuildingCache = true
        cacheBuildProgress = nil
        cacheBuildError = nil
        try? FileManager.default.createDirectory(
            atPath: attentionCacheDirectory(for: model), withIntermediateDirectories: true)
        let transformerDirectory = currentTransformerDirectory
        let outputPath = currentAttentionCachePath
        cacheBuildTask = Task {
            do {
                for try await progress in engine.buildAttentionCache(
                    transformerDirectory: transformerDirectory, outputPath: outputPath) {
                    self.cacheBuildProgress = Double(progress.completedBlocks) /
                        Double(max(progress.totalBlocks, 1))
                }
            } catch is CancellationError {
                // user-initiated - no error text needed
            } catch {
                if case H3EngineError.cancelled = error {
                    // user-initiated - no error text needed
                } else {
                    self.cacheBuildError = "キャッシュの作成に失敗しました。（詳細: \(error.localizedDescription)）"
                }
            }
            self.isBuildingCache = false
            self.cacheBuildProgress = nil
        }
    }

    func cancelCacheBuild() {
        engine?.cancelCurrentCacheBuild()
        cacheBuildTask?.cancel()
    }

    // MARK: Job state
    @Published private(set) var isGenerating = false
    @Published private(set) var isCancelling = false
    @Published var phase: String = ""
    @Published var errorMessage: String?

    // MARK: Timing / progress - all derived by ProgressEstimator from the
    // engine's real phase events plus previously measured timings.
    @Published private(set) var elapsedSeconds: Double = 0
    @Published private(set) var estimatedRemainingSeconds: Double?
    @Published private(set) var estimatedFinishDate: Date?
    @Published private(set) var progressBarFraction: Double?
    @Published private(set) var stageTitle: String = ""
    @Published private(set) var stageDetail: String = ""

    private var estimator: ProgressEstimator?
    private var elapsedTimerTask: Task<Void, Never>?

    // MARK: Result - immutable once set, independent of the live draft.
    @Published private(set) var resultURL: URL?
    @Published private(set) var resultAspectRatio: CGFloat = 1
    @Published private(set) var lastResult: ResolvedResult?

    private var engine: H3Engine?
    private var generationTask: Task<Void, Never>?

    // MARK: Local automation API (see GenerationViewModel+API.swift) -
    // replaces the old Python gui/server.py entirely: while this app runs,
    // the same job/state a person drives through the form is also reachable
    // over HTTP, with no separate process or dependency to install.
    var apiServer: HTTPServer?
    @Published var apiServerStatus: String = "起動しています…"

    init() {
        // Sweep anything a previous run left behind (crash, force quit) -
        // generated previews are meant to be throwaway unless the user
        // explicitly exports, which copies them out.
        Self.sweepStaleTempFiles()
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.deleteCurrentPreview()
                self?.apiServer?.stop()
            }
        }
        startAPIServer()
        followTurboLoRASteps()
    }

    private var turboStepsSubscription: AnyCancellable?

    /// A distilled Turbo LoRA only works at the step count it was trained
    /// for, and base-model quality collapses at that count once the LoRA is
    /// gone - so the draft's steps follow the stack: enabling one with
    /// recommendedSteps adopts them (the first such entry in list order
    /// wins), and disabling it restores the default if the user hadn't
    /// changed them. Also runs at launch, since steps aren't persisted but
    /// the stack is.
    private func followTurboLoRASteps() {
        turboStepsSubscription = library.$loras
            .map { loras -> Int? in
                loras.first { $0.enabled && $0.recommendedSteps != nil }?.recommendedSteps
            }
            .removeDuplicates()
            .scan((previous: Int?.none, current: Int?.none)) { ($0.current, $1) }
            .sink { [weak self] change in
                guard let self else { return }
                if let steps = change.current {
                    self.steps = steps
                } else if let previous = change.previous, self.steps == previous {
                    self.steps = defaultSteps
                }
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

    /// Reloads the engine against the active model - called at launch and
    /// whenever ModelManagerView changes which registered model is active.
    func loadModel() {
        guard let active = library.activeModel else {
            modelDirectory = ""
            deviceLine = "モデルが登録されていません"
            engineState = .failed("「モデル管理」からMiniMax-H3のフォルダを追加してください。")
            return
        }
        let modelDir = active.path
        modelDirectory = modelDir
        engineState = .loading
        deviceLine = "モデルを準備しています…"
        do {
            let engine = try H3Engine(modelDirectory: modelDir)
            self.engine = engine
            if let device = engine.device {
                deviceLine = "\(device.name) · \(device.architecture)"
                supportsInt8Cache = device.hasTensorHardware
                if !supportsInt8Cache { computeMode = .ssdStreaming }
            } else {
                deviceLine = "モデルは読み込めましたが、GPU情報が取得できませんでした"
            }
            engineState = .ready
        } catch {
            deviceLine = "モデルの読み込みに失敗しました"
            engineState = .failed(error.localizedDescription)
        }
    }

    /// Switches ModelLibrary's active model (when `id` is non-nil) and
    /// reloads the engine against it - always reloads even if `id` is
    /// already active, since a caller may have just changed that same
    /// entry's path (e.g. ModelDownloadWizardView finishing a download into
    /// it). `id` is nil when ModelManagerView just removed the active model
    /// and needs a reload to reflect whatever (possibly nothing) is active
    /// now. Callers that want to skip a redundant reload for an unchanged
    /// selection (a plain row tap) should check that themselves first.
    func switchModel(to id: UUID?) {
        if let id {
            library.selectModel(id: id)
        }
        loadModel()
    }

    // MARK: Image pickers

    func pickFirstFrame() {
        if let path = chooseFile(allowedContentTypes: imageTypes) { firstFramePath = path }
    }

    func pickLastFrame() {
        if let path = chooseFile(allowedContentTypes: imageTypes) { lastFramePath = path }
    }

    func addReferenceImages() {
        for path in chooseFiles(allowedContentTypes: referenceMediaTypes) {
            let kind: H3ReferenceKind = isVideoFile(path: path) ? .video : (isAudioFile(path: path) ? .audio : .image)
            referenceImages.append(H3ReferenceInput(kind: kind, path: path))
        }
    }

    func removeReferenceImages(at offsets: IndexSet) {
        referenceImages.remove(atOffsets: offsets)
    }

    func moveReferenceImages(from source: IndexSet, to destination: Int) {
        referenceImages.move(fromOffsets: source, toOffset: destination)
    }

    func resetAdvancedSettings() {
        sizeProfile = defaultSizeProfile
        steps = defaultSteps
        denoiseReuse = defaultReuse
        computeMode = defaultComputeMode
        speedMode = .quality
        fastAttention = false
        seedText = ""
        library.setEnabledLoRAs([])
    }

    // MARK: Validation (UI-07: block generation with a locatable reason
    // instead of a generic disabled button)

    var validationMessage: String? {
        guard case .ready = engineState else { return nil } // covered by engine status instead
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "動画の内容を入力してください"
        }
        if creationMethod == .image {
            switch imageInputMode {
            case .firstLastFrame:
                if firstFramePath == nil { return "最初の画像を選んでください" }
            case .referenceImage:
                if referenceImages.isEmpty { return "参照画像・動画・音声を選んでください" }
                // h3.c rejects this combination outright ("reference audio
                // requires an image or video reference") - an audio file
                // can season a visual reference but can't carry a
                // generation on its own.
                let hasVisualReference = referenceImages.contains { $0.kind == .image || $0.kind == .video }
                if !hasVisualReference {
                    return "音声だけの参照はできません。画像か動画の参照も追加してください"
                }
            }
        }
        for lora in library.enabledLoRAs {
            if !FileManager.default.fileExists(atPath: lora.path) {
                return "追加モデル「\(lora.name)」のファイルが見つかりません。モデル管理で確認してください"
            }
            if case .failure = library.loraInfo(for: lora) {
                return "追加モデル「\(lora.name)」はMiniMax-H3用として読み込めません。モデル管理で確認してください"
            }
        }
        if computeMode == .attentionCache {
            if !supportsInt8Cache {
                return "このGPUではint8キャッシュを使えません。詳細設定の「計算方式」でSSDストリーミングを選んでください"
            }
            if attentionCacheMissing {
                return "int8キャッシュが見つかりません。詳細設定の「計算方式」から作成するか、SSDストリーミングに切り替えてください"
            }
        }
        return nil
    }

    var canGenerate: Bool {
        engineState == .ready && !isGenerating && !isBuildingCache && validationMessage == nil
    }

    // MARK: Summary text shown above the primary button, and in "設定を見る"

    var draftSummaryText: String {
        var parts = ["\(sizeProfile.label)", "\(seconds)秒"]
        if hasAdvancedChanges {
            parts.append("Steps \(steps)")
            if denoiseReuse != defaultReuse { parts.append("reuse \(denoiseReuse)") }
            if computeMode != defaultComputeMode { parts.append(computeMode.summaryLabel) }
            if speedMode != .quality { parts.append(speedMode.summaryLabel) }
            if fastAttention { parts.append("高速モード（試験的）") }
            if !seedText.isEmpty { parts.append("シード固定") }
            let loras = library.enabledLoRAs
            if !loras.isEmpty { parts.append("追加モデル: " + loras.map(\.name).joined(separator: " + ")) }
        }
        return parts.joined(separator: " ・ ")
    }

    // MARK: Generation

    func generate() {
        guard let engine, canGenerate else { return }

        deleteCurrentPreview()
        isGenerating = true
        isCancelling = false
        errorMessage = nil
        resultURL = nil
        phase = ""

        let outputPath = NSTemporaryDirectory() + "h3c-app_\(Int(Date().timeIntervalSince1970)).mp4"
        let dimensions = sizeProfile.dimensions
        let requestedSeconds = seconds.clamped(to: secondsRange)
        let requestedFrames = Int(h3AlignedFrameCount(seconds: Double(requestedSeconds)))
        let seedWasRandom = seedText.isEmpty

        // Which resolution the DiT actually runs at (the upscaled profiles
        // generate at renderWidth x renderHeight, then upscale).
        let ditPixels = dimensions.renderWidth > 0
            ? Double(dimensions.renderWidth) * Double(dimensions.renderHeight)
            : Double(dimensions.width) * Double(dimensions.height)
        estimator = ProgressEstimator(
            shape: ProgressEstimator.Shape(
                steps: steps.clamped(to: stepsRange),
                reuse: denoiseReuse.clamped(to: reuseRange),
                totalFrames: requestedFrames,
                ditUnits: Double(requestedFrames) * ditPixels,
                decodeUnits: Double(requestedFrames) * Double(dimensions.width) * Double(dimensions.height)),
            calibration: TimingCalibration.load(for: computeMode, speed: speedMode,
                                                fastAttention: useFastAttention),
            start: Date())
        publishTiming()
        startElapsedTimer()
        let resolvedSeed = UInt64(seedText) ?? UInt64.random(in: UInt64.min ... UInt64.max)

        // Design spec invariant #4: only the image state matching the
        // *current* mode reaches the engine - the rest stays in the draft,
        // untouched, for if the user switches back.
        let effectiveFirstFrame = (creationMethod == .image && imageInputMode == .firstLastFrame) ? firstFramePath : nil
        let effectiveLastFrame = (creationMethod == .image && imageInputMode == .firstLastFrame) ? lastFramePath : nil
        let effectiveReferences = (creationMethod == .image && imageInputMode == .referenceImage) ? referenceImages : []

        var params = H3GenerationParams()
        params.width = dimensions.width
        params.height = dimensions.height
        params.renderWidth = dimensions.renderWidth
        params.renderHeight = dimensions.renderHeight
        params.frames = Int32(requestedFrames)
        params.steps = Int32(steps.clamped(to: stepsRange))
        params.denoiseReuse = Int32(denoiseReuse.clamped(to: reuseRange))
        params.seed = resolvedSeed
        params.firstFrame = effectiveFirstFrame
        params.lastFrame = effectiveLastFrame
        params.references = effectiveReferences
        params.ssdStreaming = computeMode == .ssdStreaming
        params.attentionCachePath = computeMode == .attentionCache ? currentAttentionCachePath : nil
        let capturedLoRAs = effectiveLoRAs
        params.loras = capturedLoRAs.map { H3LoRAInput(path: $0.path, strength: $0.strength) }
        let speed = speedSettings
        params.ditLayers = speed.ditLayers
        params.coreReuse = speed.coreReuse
        params.tokenReduction = speed.tokenReduction
        // Decided here, once, for this generation: the value is copied into
        // params, so later toggles in the form can't affect a running job.
        params.fastAttention = useFastAttention

        let promptCopy = prompt
        let capturedMode = creationMethod
        let capturedImageMode = creationMethod == .image ? imageInputMode : nil
        let capturedSizeProfile = sizeProfile
        let capturedSteps = steps.clamped(to: stepsRange)
        let capturedReuse = denoiseReuse.clamped(to: reuseRange)
        let capturedComputeMode = computeMode
        let capturedSpeedMode = speedMode
        let capturedFastAttention = params.fastAttention
        let capturedDeviceLine = deviceLine

        generationTask = Task {
            do {
                for try await event in engine.generate(prompt: promptCopy, outputPath: outputPath, params: params) {
                    switch event {
                    case .progress(let phase, let completed, let total):
                        self.phase = phase
                        self.estimator?.handle(phase: phase, completed: completed, total: total, now: Date())
                        self.publishTiming()
                    case .frame:
                        break
                    case .preview:
                        break
                    case .finished(let result):
                        self.phase = "できあがりました"
                        self.estimator?.finishedCalibration(now: Date())
                            // Keyed by the path that actually ran, not the
                            // checkbox: a diagnostic H3_ATTENTION_BACKEND can
                            // route through ccv even with it off.
                            .save(for: capturedComputeMode, speed: capturedSpeedMode,
                                  fastAttention: result.ccvAttentionCalls > 0)
                        let url = URL(fileURLWithPath: result.outputPath)
                        self.resultURL = url
                        self.resultAspectRatio = CGFloat(dimensions.width) / CGFloat(dimensions.height)
                        self.lastResult = ResolvedResult(
                            prompt: promptCopy,
                            creationMethod: capturedMode,
                            imageInputMode: capturedImageMode,
                            sizeProfile: capturedSizeProfile,
                            requestedSeconds: requestedSeconds,
                            requestedFrames: requestedFrames,
                            actualFrameCount: result.frames,
                            fps: result.fps,
                            actualDurationSeconds: nil,
                            steps: capturedSteps,
                            denoiseReuse: capturedReuse,
                            computeMode: capturedComputeMode,
                            speedMode: capturedSpeedMode,
                            fastAttention: capturedFastAttention,
                            ccvAttentionCalls: result.ccvAttentionCalls,
                            ccvAttentionDirectCalls: result.ccvAttentionDirectCalls,
                            seed: result.seed,
                            seedWasRandom: seedWasRandom,
                            loras: capturedLoRAs,
                            deviceLine: capturedDeviceLine,
                            completedAt: Date()
                        )
                        self.loadActualDuration(for: url)
                    }
                }
            } catch is CancellationError {
                self.phase = "生成を中止しました"
            } catch {
                if case H3EngineError.cancelled = error {
                    self.phase = "生成を中止しました"
                } else {
                    self.errorMessage = Self.userFacingMessage(for: error)
                }
            }
            self.isGenerating = false
            self.isCancelling = false
            self.stopElapsedTimer()
        }
    }

    // 指定秒数と実際のメディア長は一致するとは限らない（design spec 7章:
    // 「長さの不一致」）ため、生成結果のフレーム数/fpsからの概算ではなく、
    // 実際に書き出されたファイルをAVFoundationで読んで確認する。
    private func loadActualDuration(for url: URL) {
        Task {
            let asset = AVURLAsset(url: url)
            if let duration = try? await asset.load(.duration) {
                self.lastResult?.actualDurationSeconds = duration.seconds
            }
        }
    }

    private static func userFacingMessage(for error: Error) -> String {
        // Best-effort mapping of known engine failure text to the design
        // spec's Japanese error copy (section 8) - anything unrecognized
        // falls through to the generic message with the real detail kept
        // alongside it rather than guessing at a specific cause.
        let raw = error.localizedDescription
        if raw.contains("out of memory") || raw.contains("insufficient memory") {
            return "この設定ではメモリが足りませんでした。もっと小さいサイズや短い長さをお試しください。（詳細: \(raw)）"
        }
        return "動画をつくれませんでした。（詳細: \(raw)）"
    }

    func cancel() {
        guard isGenerating else { return }
        isCancelling = true
        engine?.cancelCurrentGeneration()
    }

    // MARK: Timing

    private func startElapsedTimer() {
        elapsedTimerTask?.cancel()
        elapsedTimerTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                self.publishTiming()
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
    }

    private func stopElapsedTimer() {
        elapsedTimerTask?.cancel()
        elapsedTimerTask = nil
    }

    private func publishTiming() {
        guard let estimator else { return }
        let now = Date()
        let snapshot = estimator.snapshot(now: now)
        elapsedSeconds = snapshot.elapsed
        estimatedRemainingSeconds = snapshot.remaining
        estimatedFinishDate = snapshot.remaining.map { now.addingTimeInterval($0) }
        progressBarFraction = snapshot.fraction
        stageTitle = snapshot.stageTitle
        stageDetail = snapshot.detail
    }

    // MARK: Reuse

    /// "この設定を使う": copies a past result's request back into the draft
    /// without starting generation, restoring the original seed *policy*
    /// (random stays random) rather than pinning the exact value.
    func applyDraft(from result: ResolvedResult) {
        prompt = result.prompt
        creationMethod = result.creationMethod
        if let imageInputMode = result.imageInputMode { self.imageInputMode = imageInputMode }
        sizeProfile = result.sizeProfile
        seconds = result.requestedSeconds
        denoiseReuse = result.denoiseReuse
        computeMode = result.computeMode
        speedMode = result.speedMode
        fastAttention = result.fastAttention && fastAttentionAvailable
        seedText = result.seedWasRandom ? "" : result.seedDecimalString
        // The result only kept the paths/strengths actually used, not
        // library entry ids (which may since have been renamed or removed) -
        // best-effort match them back to still-registered LoRAs by path.
        var ids = Set<UUID>()
        for used in result.loras {
            guard let match = library.loras.first(where: { $0.path == used.path }) else { continue }
            ids.insert(match.id)
            library.setLoRAScale(id: match.id, scaleText: used.strength == 1 ? "" : "\(used.strength)")
        }
        library.setEnabledLoRAs(ids)
        // After the LoRAs: enabling a Turbo LoRA moves steps to its
        // recommended count, and the past result's own value should win.
        steps = result.steps
    }

    /// "同じシードを使う": pins the exact seed regardless of the original
    /// policy, for reproducing one specific past output.
    func useSameSeed(from result: ResolvedResult) {
        seedText = result.seedDecimalString
    }
}
