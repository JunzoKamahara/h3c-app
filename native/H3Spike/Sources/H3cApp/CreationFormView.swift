import H3Engine
import SwiftUI

private let promptExamples: [(label: String, text: String)] = [
    ("猫と毛糸", "A cat playing with a ball of yarn."),
    ("海辺", "Waves rolling onto a sandy beach at sunset."),
]

struct CreationFormView: View {
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var library: ModelLibrary
    @Binding var showingModelManager: Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var isAdvancedExpanded = false
    @State private var pendingExampleReplacement: String?

    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: H3Spacing.xl) {
                    Text("動画をつくる")
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(palette.textPrimary)

                    creationMethodSection
                    if viewModel.creationMethod == .image {
                        imageInputSection
                    }
                    promptSection
                    aspectSection
                    durationSection
                    advancedSection
                }
                .padding(H3Spacing.xl)
            }

            Divider()

            bottomBar
        }
        .background(palette.canvas)
    }

    // MARK: 作り方

    private var creationMethodSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            sectionHeading("作り方")
            Picker("作り方", selection: $viewModel.creationMethod) {
                ForEach(CreationMethod.allCases) { method in
                    Text(method.label).tag(method)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if viewModel.creationMethod == .image {
                Picker("画像の使い方", selection: $viewModel.imageInputMode) {
                    ForEach(ImageInputMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
        }
    }

    // MARK: 画像入力

    private var imageInputSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            switch viewModel.imageInputMode {
            case .firstLastFrame:
                Text("この画像から動き始めます。").font(.caption).foregroundStyle(palette.textSecondary)
                ImagePickerRow(label: "最初の画像", path: $viewModel.firstFramePath, choose: viewModel.pickFirstFrame)
                if viewModel.lastFramePath != nil {
                    ImagePickerRow(label: "最後の画像", path: $viewModel.lastFramePath, choose: viewModel.pickLastFrame)
                } else {
                    Button("最後の画像も指定…") { viewModel.pickLastFrame() }
                        .font(.caption)
                }

            case .referenceImage:
                Text("画像・動画・音声の特徴を参考にして動画をつくります。最初のフレームが同じになるとは限りません。音声だけを参照にすることはできないので、画像か動画と組み合わせて追加してください。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                HStack {
                    Text("参照画像・動画・音声").font(.caption).foregroundStyle(palette.textSecondary)
                    Spacer()
                    Button("ファイルを選ぶ…") { viewModel.addReferenceImages() }
                }
                if !viewModel.referenceImages.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(viewModel.referenceImages.enumerated()), id: \.element.id) { index, reference in
                            HStack {
                                Image(systemName: referenceIcon(for: reference.kind))
                                    .foregroundStyle(palette.textSecondary)
                                Text("\(index + 1). \(URL(fileURLWithPath: reference.path).lastPathComponent)")
                                    .font(.caption)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Spacer()
                                Button("取り除く") {
                                    viewModel.removeReferenceImages(at: IndexSet(integer: index))
                                }
                                .font(.caption)
                            }
                        }
                    }
                }
            }
        }
        .padding(H3Spacing.md)
        .background(palette.surfaceMuted)
        .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
    }

    private func referenceIcon(for kind: H3ReferenceKind) -> String {
        switch kind {
        case .video, .videoAudio: return "video"
        case .audio: return "waveform"
        case .image: return "photo"
        }
    }

    // MARK: 動画の内容 (プロンプト)

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            sectionHeading("動画の内容")
            ZStack(alignment: .topLeading) {
                TextEditor(text: $viewModel.prompt)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(minHeight: 132, maxHeight: 200)
                    .background(palette.surface)
                    .clipShape(RoundedRectangle(cornerRadius: H3Radius.editor))
                    .overlay(
                        RoundedRectangle(cornerRadius: H3Radius.editor)
                            .stroke(palette.border, lineWidth: 1)
                    )
                if viewModel.prompt.isEmpty {
                    Text("猫が毛糸玉を追いかける。窓からやわらかな光が差し込む。")
                        .font(.body)
                        .foregroundStyle(palette.textSecondary)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 14)
                        .allowsHitTesting(false)
                }
            }

            HStack(spacing: H3Spacing.sm) {
                Text("例を入れる:").font(.caption).foregroundStyle(palette.textSecondary)
                ForEach(promptExamples, id: \.label) { example in
                    Button(example.label) { requestExample(example.text) }
                        .font(.caption)
                }
            }

            if let message = viewModel.validationMessage, message.contains("動画の内容") {
                Text(message).font(.caption).foregroundStyle(palette.errorColor)
            }
        }
        .confirmationDialog(
            "入力をこの例に置き換えますか？",
            isPresented: Binding(
                get: { pendingExampleReplacement != nil },
                set: { if !$0 { pendingExampleReplacement = nil } }
            )
        ) {
            Button("置き換える") {
                if let text = pendingExampleReplacement { viewModel.prompt = text }
                pendingExampleReplacement = nil
            }
            Button("キャンセル", role: .cancel) { pendingExampleReplacement = nil }
        }
    }

    private func requestExample(_ text: String) {
        if viewModel.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            viewModel.prompt = text
        } else {
            pendingExampleReplacement = text
        }
    }

    // MARK: 画面の形

    private var aspectSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            sectionHeading("画面の形")
            HStack(spacing: H3Spacing.sm) {
                ForEach(AspectShape.allCases) { shape in
                    shapeButton(shape)
                }
            }
        }
    }

    private func shapeButton(_ shape: AspectShape) -> some View {
        let isSelected = viewModel.sizeProfile.shape == shape
        return Button {
            if let first = SizeProfile.profiles(for: shape).first {
                viewModel.sizeProfile = first
            }
        } label: {
            VStack(spacing: 4) {
                Image(systemName: shape.systemImage).font(.system(size: 18))
                Text(shape.label).font(.caption)
            }
            .frame(width: 72, height: 56)
            .background(isSelected ? palette.accentSoft : palette.surfaceMuted)
            .foregroundStyle(isSelected ? palette.accent : palette.textPrimary)
            .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    // MARK: 長さ

    private var durationSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            sectionHeading("長さ")
            Picker("長さ", selection: $viewModel.seconds) {
                ForEach(secondsRange, id: \.self) { value in
                    Text("\(value) 秒").tag(value)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 160)
        }
    }

    // MARK: 詳細設定

    private var advancedSection: some View {
        DisclosureGroup(isExpanded: $isAdvancedExpanded) {
            VStack(alignment: .leading, spacing: H3Spacing.md) {
                if SizeProfile.profiles(for: viewModel.sizeProfile.shape).count > 1 {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("解像度").font(.caption).foregroundStyle(palette.textSecondary)
                        Picker("解像度", selection: $viewModel.sizeProfile) {
                            ForEach(SizeProfile.profiles(for: viewModel.sizeProfile.shape)) { profile in
                                Text(profile.resolutionLabel).tag(profile)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 240)
                    }
                } else {
                    labeledValue("解像度", viewModel.sizeProfile.resolutionLabel)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("生成ステップ数").font(.caption).foregroundStyle(palette.textSecondary)
                    Picker("生成ステップ数", selection: $viewModel.steps) {
                        ForEach(stepsRange, id: \.self) { value in
                            Text("\(value)").tag(value)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 120)
                }

                computeModeSection

                reuseSection

                seedSection

                loraSection

                Button("詳細設定を既定に戻す") { viewModel.resetAdvancedSettings() }
                    .font(.caption)
            }
            .padding(.top, H3Spacing.sm)
        } label: {
            Text(viewModel.hasAdvancedChanges ? "詳細設定・変更あり" : "詳細設定")
                .font(.system(size: 14, weight: .semibold))
        }
    }

    private var computeModeSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("計算方式").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("計算方式", selection: $viewModel.computeMode) {
                ForEach(ComputeMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                        .disabled(mode == .attentionCache && !viewModel.supportsInt8Cache)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 260)
            switch viewModel.computeMode {
            case .attentionCache:
                Text("事前に作ったint8キャッシュを読みながら計算します。速く、追加モデル（LoRA）も使えます。キャッシュと、Tensor演算ユニットを搭載したGPU（M5以降）が必要です。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            case .ssdStreaming:
                Text("元のBF16モデルを、必要なブロックだけSSDから読みながら計算します。メモリは少なくて済みますが遅くなります。キャッシュは使わず、追加モデル（LoRA）は使えません。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if viewModel.supportsInt8Cache && viewModel.computeMode == .attentionCache {
                attentionCacheBuildSection
            }
            if viewModel.isHeavySsdStreamingConfig {
                Text("最大解像度・長い秒数・SSDストリーミングの組み合わせは、メモリ不足でスワップが発生し非常に遅くなることがあります（数時間かかる場合も）。解像度か秒数を下げることをおすすめします。")
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
        }
    }

    /// Shown under the compute-mode picker when the current mode/reference
    /// selection needs a cache that doesn't exist yet on disk - lets the
    /// user build it in-app instead of just being told it's missing.
    private var attentionCacheBuildSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            if viewModel.isBuildingCache {
                HStack {
                    ProgressView(value: viewModel.cacheBuildProgress)
                        .frame(maxWidth: 200)
                    Text(viewModel.cacheBuildProgress.map { "\(Int($0 * 50))/50 ブロック" } ?? "準備中…")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                    Spacer()
                    Button("中止") { viewModel.cancelCacheBuild() }
                        .font(.caption)
                }
                Text("キャッシュを作成しています。モデルの重みを読み込んで量子化するため、数十秒〜1分程度かかります。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            } else if viewModel.attentionCacheMissing {
                HStack {
                    Text("int8キャッシュがまだありません。")
                        .font(.caption)
                        .foregroundStyle(palette.errorColor)
                    Spacer()
                    Button("キャッシュを作成…") { viewModel.buildMissingAttentionCache() }
                        .font(.caption)
                }
            }
            if let cacheBuildError = viewModel.cacheBuildError {
                Text(cacheBuildError)
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
        }
    }

    private var reuseSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ノイズ除去の再利用（reuse）").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("ノイズ除去の再利用", selection: $viewModel.denoiseReuse) {
                ForEach(reuseRange, id: \.self) { value in
                    Text(reuseLabel(value)).tag(value)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 200)
            Text("N回に1回だけモデルを計算し、残りは前回までの結果から補って高速化します。大きいほど速く、品質は下がる可能性があります。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
            if viewModel.denoiseReuse > 1 && viewModel.steps <= 7 {
                Text("ステップ数が少ないと、実際にモデルを計算する回数が極端に少なくなります。")
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
        }
    }

    private func reuseLabel(_ value: Int) -> String {
        switch value {
        case 1: return "1（標準）"
        case 2: return "2（高速）"
        default: return "\(value)（さらに高速）"
        }
    }

    private var seedSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ランダムさ").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("ランダムさ", selection: Binding(
                get: { viewModel.seedText.isEmpty },
                set: { useRandom in viewModel.seedText = useRandom ? "" : "0" }
            )) {
                Text("毎回変える").tag(true)
                Text("固定する").tag(false)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 240)

            if !viewModel.seedText.isEmpty {
                TextField("シード値", text: $viewModel.seedText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 200)
            }
        }
    }

    private var loraSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("追加モデル（LoRA）").font(.caption).foregroundStyle(palette.textSecondary)
            HStack {
                Picker("追加モデル（LoRA）", selection: Binding(
                    get: { library.activeLoRAID },
                    set: { library.selectLoRA($0) }
                )) {
                    Text("なし").tag(UUID?.none)
                    ForEach(library.loras) { entry in
                        Text(entry.name).tag(UUID?.some(entry.id))
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(maxWidth: 200)
                Spacer()
                Button("管理…") { showingModelManager = true }
                    .font(.caption)
            }
            .disabled(viewModel.computeMode == .ssdStreaming)

            if viewModel.computeMode == .ssdStreaming {
                // Not just disabled: the engine wouldn't error, it would
                // silently not apply it, so say so where the selection is shown.
                Text(library.activeLoRA == nil
                     ? "SSDストリーミングでは追加モデル（LoRA）を使えません。"
                     : "SSDストリーミングでは適用されないため、この追加モデルは今回の生成に使われません。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            } else if let active = library.activeLoRA {
                HStack {
                    Text("強さ（空欄=自動）").font(.caption).foregroundStyle(palette.textSecondary)
                    TextField("自動", text: Binding(
                        get: { active.scaleText },
                        set: { library.setLoRAScale(id: active.id, scaleText: $0) }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                }
                Text("読み込むモデル（最初/最後の画像・参照画像・動画）に対応したファイルか、事前に確認できません。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
        }
    }

    // MARK: 下部固定領域

    private var bottomBar: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            if let message = viewModel.validationMessage {
                Text(message).font(.caption).foregroundStyle(palette.errorColor)
            } else {
                Text(viewModel.draftSummaryText)
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(2)
            }

            HStack(spacing: H3Spacing.sm) {
                Button {
                    viewModel.generate()
                } label: {
                    Text(viewModel.isGenerating ? "生成中…" : "動画をつくる")
                        .frame(maxWidth: .infinity)
                        .frame(height: H3ControlHeight.primary)
                }
                .buttonStyle(.borderedProminent)
                .tint(palette.accent)
                .disabled(!viewModel.canGenerate)
                .keyboardShortcut(.return, modifiers: .command)

                if viewModel.isGenerating {
                    Button("中止") { viewModel.cancel() }
                        .disabled(viewModel.isCancelling)
                }
            }

            Text("⌘Return で生成")
                .font(.caption2)
                .foregroundStyle(palette.textSecondary)
        }
        .padding(H3Spacing.xl)
        .background(palette.surface)
    }

    private func sectionHeading(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(palette.textPrimary)
    }

    private func labeledValue(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(palette.textSecondary)
            Text(value).font(.caption)
        }
    }
}

private struct ImagePickerRow: View {
    let label: String
    @Binding var path: String?
    let choose: () -> Void

    var body: some View {
        HStack {
            Text(label).font(.caption).frame(width: 84, alignment: .leading)
            Text(path.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "未選択")
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button("変更…", action: choose)
            if path != nil {
                Button("取り除く") { path = nil }
            }
        }
    }
}
