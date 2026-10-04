import H3Engine
import SwiftUI

// Shown in grey while the prompt is empty; Tab turns it into real text.
private let promptSuggestion = "A cat playing with a ball of yarn."

struct CreationFormView: View {
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var library: ModelLibrary
    @Binding var showingModelManager: Bool
    @ObservedObject private var presetStore = PresetStore.shared
    // Floating-panel placement, owned by ContentView: offset from the
    // bottom-centre home position, clamped to `bounds` (the window).
    @Binding var offset: CGSize
    @Binding var isCollapsed: Bool
    var bounds: CGSize
    @Environment(\.colorScheme) private var colorScheme
    @State private var dragBase: CGSize?

    private var palette: H3Palette { H3Palette(colorScheme) }

    static let panelWidth: CGFloat = 640

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            dragHandle
            if !isCollapsed {
                topRow
                if viewModel.creationMethod == .image {
                    imageInputSection
                }
                promptSection
            }
            bottomBar
        }
        .padding(.horizontal, H3Spacing.lg)
        .padding(.bottom, H3Spacing.lg)
        .padding(.top, 6)
        .frame(width: Self.panelWidth)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(palette.border, lineWidth: 1))
        .shadow(color: .black.opacity(0.25), radius: 18, y: 6)
    }

    // MARK: 移動・折りたたみ

    /// Drag to move the panel anywhere over the preview; double-click to
    /// put it back at the bottom centre.
    private var dragHandle: some View {
        ZStack {
            Capsule()
                .fill(palette.textSecondary.opacity(0.45))
                .frame(width: 44, height: 5)
            // Collapsed, the bottom row is the (much larger) open target.
            if !isCollapsed {
                HStack {
                    Spacer()
                    Button {
                        withAnimation(.easeInOut(duration: 0.15)) { isCollapsed = true }
                    } label: {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(palette.textSecondary)
                            .frame(width: 40, height: 22)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("パネルを折りたたむ")
                }
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 22)
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 2)
                .onChanged { value in
                    let base = dragBase ?? offset
                    dragBase = base
                    offset = clamped(CGSize(width: base.width + value.translation.width,
                                            height: base.height + value.translation.height))
                }
                .onEnded { _ in dragBase = nil }
        )
        .onTapGesture(count: 2) { withAnimation { offset = .zero } }
        .help("ドラッグで移動、ダブルクリックで元の位置へ")
    }

    private func clamped(_ proposed: CGSize) -> CGSize {
        let maxX = max(0, (bounds.width - Self.panelWidth) / 2 - 8)
        let maxUp = max(0, bounds.height - 160)
        return CGSize(width: min(max(proposed.width, -maxX), maxX),
                      height: min(max(proposed.height, -maxUp), 0))
    }

    // MARK: 作り方・プリセット・詳細設定

    private var presetMenu: some View {
        Menu {
            if presetStore.presets.isEmpty {
                Text("保存したプリセットはありません")
            } else {
                ForEach(presetStore.presets) { preset in
                    Button {
                        viewModel.applyPreset(preset)
                    } label: {
                        if preset.id == viewModel.activePresetID {
                            Label(preset.name, systemImage: "checkmark")
                        } else {
                            Text(preset.name)
                        }
                    }
                }
            }
            Divider()
            Button("現在の設定をプリセットとして保存…") {
                viewModel.presetPendingName = viewModel.presetFromCurrentSettings()
            }
            if !presetStore.presets.isEmpty {
                Menu("削除") {
                    ForEach(presetStore.presets) { preset in
                        Button(preset.name) { viewModel.deletePreset(id: preset.id) }
                    }
                }
            }
        } label: {
            // The preset in use, marked when the form has moved off it.
            if let active = viewModel.activePreset {
                Label(active.modified ? String(localized: "\(active.preset.name)・変更あり") : active.preset.name,
                      systemImage: "square.stack")
            } else {
                Label("プリセット", systemImage: "square.stack")
            }
        }
        .fixedSize()
        .help("形・大きさ・長さと詳細設定をまとめて保存・適用します（プロンプトは含みません）")
    }

    private var topRow: some View {
        HStack(spacing: H3Spacing.sm) {
            Picker("作り方", selection: $viewModel.creationMethod) {
                ForEach(CreationMethod.allCases) { method in
                    Text(method.label).tag(method)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 200)

            if viewModel.creationMethod == .image {
                Picker("画像の使い方", selection: $viewModel.imageInputMode) {
                    ForEach(ImageInputMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }

            Spacer(minLength: 0)

            presetMenu

            Button {
                viewModel.showingAdvancedSettings = true
            } label: {
                Label(viewModel.hasAdvancedChanges ? String(localized: "詳細設定・変更あり") : String(localized: "詳細設定"),
                      systemImage: "slider.horizontal.3")
            }
            .help("詳細設定を開く（⌘,）")
        }
    }

    // MARK: 画像入力

    private var imageInputSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            switch viewModel.imageInputMode {
            case .firstLastFrame:
                Text("この画像から動き始めます。").font(.caption).foregroundStyle(palette.textSecondary)
                ImagePickerRow(label: String(localized: "最初の画像"), path: $viewModel.firstFramePath, choose: viewModel.pickFirstFrame)
                if viewModel.lastFramePath != nil {
                    ImagePickerRow(label: String(localized: "最後の画像"), path: $viewModel.lastFramePath, choose: viewModel.pickLastFrame)
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
            // Composer layout: the basic shape/size/length choices and the
            // generate button live in a toolbar along the bottom edge of the
            // prompt box. Return generates, Shift+Return starts a new line.
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    PromptTextView(text: $viewModel.prompt, suggestion: promptSuggestion) {
                        if viewModel.canGenerate { viewModel.generate() }
                    }
                    .frame(height: 84)
                    if viewModel.prompt.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(promptSuggestion)
                                .font(.body)
                            Text("Tabで入力 ・ Enterで生成 ・ Shift+Enterで改行")
                                .font(.caption)
                        }
                        .foregroundStyle(palette.textSecondary.opacity(0.75))
                        .padding(.horizontal, 11)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                    }
                }
                HStack(spacing: H3Spacing.sm) {
                    aspectMenu
                    sizeMenu
                    durationMenu
                    // Several videos per press only with a project, where
                    // they are kept.
                    if viewModel.project != nil { batchMenu }
                    Spacer(minLength: 0)
                    generateButton
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 8)
            }
            .background(palette.surface)
            .clipShape(RoundedRectangle(cornerRadius: H3Radius.editor))
            .overlay(
                RoundedRectangle(cornerRadius: H3Radius.editor)
                    .stroke(palette.border, lineWidth: 1)
            )

        }
    }

    // MARK: 画面の形・長さ (プロンプト欄の下端)

    private var aspectMenu: some View {
        Menu {
            Picker("画面の形", selection: Binding(
                get: { viewModel.sizeProfile.shape },
                set: { shape in
                    // Keep 小/大 when the shape changes.
                    viewModel.sizeProfile = SizeProfile.profile(
                        for: shape, large: viewModel.sizeProfile.isLarge)
                }
            )) {
                ForEach(AspectShape.allCases) { shape in
                    Label(shape.label, systemImage: shape.systemImage).tag(shape)
                }
            }
            .pickerStyle(.inline)
        } label: {
            composerChip(systemImage: viewModel.sizeProfile.shape.systemImage,
                         text: viewModel.sizeProfile.shape.label)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("画面の形")
    }

    private var sizeMenu: some View {
        Menu {
            Picker("大きさ", selection: Binding(
                get: { viewModel.sizeProfile.isLarge },
                set: { large in
                    viewModel.sizeProfile = SizeProfile.profile(
                        for: viewModel.sizeProfile.shape, large: large)
                }
            )) {
                let shape = viewModel.sizeProfile.shape
                Text("小（\(SizeProfile.profile(for: shape, large: false).resolutionLabel)）").tag(false)
                Text("大（\(SizeProfile.profile(for: shape, large: true).resolutionLabel)）").tag(true)
            }
            .pickerStyle(.inline)
        } label: {
            composerChip(systemImage: "arrow.up.left.and.arrow.down.right",
                         text: viewModel.sizeProfile.isLarge ? String(localized: "大") : String(localized: "小"))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("大きさ（\(viewModel.sizeProfile.resolutionLabel)）")
    }

    private var durationMenu: some View {
        Menu {
            Picker("長さ", selection: $viewModel.seconds) {
                ForEach(secondsRange, id: \.self) { value in
                    Text("\(value) 秒").tag(value)
                }
            }
            .pickerStyle(.inline)
        } label: {
            composerChip(systemImage: "clock", text: String(localized: "\(viewModel.seconds)秒"))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("長さ")
    }

    private var batchMenu: some View {
        Menu {
            Picker("本数", selection: $viewModel.batchCount) {
                ForEach([1, 2, 3, 4, 5, 6, 8, 10, 15, 20], id: \.self) { value in
                    Text("\(value) 本").tag(value)
                }
            }
            .pickerStyle(.inline)
        } label: {
            composerChip(systemImage: "square.stack.3d.down.right",
                         text: String(localized: "\(viewModel.batchCount)本"))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("同じプロンプトと設定で、シードを変えて続けて生成し、プロジェクトに保存します（シード固定のときは1ずつ増やします）")
    }

    private func composerChip(systemImage: String, text: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: systemImage)
            Text(text)
            Image(systemName: "chevron.down").font(.system(size: 9, weight: .semibold))
        }
        .font(.callout)
        .foregroundStyle(palette.textPrimary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(palette.surfaceMuted)
        .clipShape(Capsule())
    }

    // MARK: 下部固定領域

    private var bottomBar: some View {
        HStack(spacing: H3Spacing.sm) {
            if isCollapsed {
                // Collapsed, the whole row (all but the generate button)
                // opens the panel - the chevron alone was too small a target.
                Button(action: expand) {
                    HStack(spacing: H3Spacing.sm) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(collapsedPromptLine)
                                .font(.callout)
                                .foregroundStyle(viewModel.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? palette.textSecondary : palette.textPrimary)
                                .lineLimit(1)
                            statusText.lineLimit(1)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        Image(systemName: "chevron.up")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(palette.textSecondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("クリックでパネルを広げる")

                // The prompt box (and its button) is hidden while collapsed.
                generateButton
            } else {
                statusText
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var statusText: some View {
        Group {
            // An empty prompt already shows its grey suggestion; no need
            // for a red error line as well.
            if let message = viewModel.validationMessage, !viewModel.promptIsEmpty {
                Text(message).foregroundStyle(palette.errorColor)
            } else {
                Text(viewModel.draftSummaryText).foregroundStyle(palette.textSecondary)
            }
        }
        .font(.caption)
    }

    private var collapsedPromptLine: String {
        let firstLine = viewModel.prompt
            .split(whereSeparator: \.isNewline)
            .first { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        return firstLine.map(String.init) ?? String(localized: "プロンプトを入力")
    }

    private func expand() {
        withAnimation(.easeInOut(duration: 0.15)) { isCollapsed = false }
    }

    /// Round icon at the prompt's bottom-right: generate, or stop while a
    /// generation runs. ⌘Return also generates.
    private var generateButton: some View {
        Button {
            if viewModel.isGenerating { viewModel.cancel() } else { viewModel.generate() }
        } label: {
            Image(systemName: viewModel.isGenerating ? "stop.fill" : "arrow.up")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 30, height: 30)
                .background(Circle().fill(generateEnabled ? palette.accent : palette.textSecondary.opacity(0.4)))
        }
        .buttonStyle(.plain)
        .disabled(!generateEnabled)
        .keyboardShortcut(.return, modifiers: .command)
        .help(viewModel.isGenerating ? String(localized: "中止") : String(localized: "動画をつくる（Enter / ⌘Return）"))
        .accessibilityLabel(viewModel.isGenerating ? String(localized: "中止") : String(localized: "動画をつくる"))
    }

    private var generateEnabled: Bool {
        viewModel.isGenerating ? !viewModel.isCancelling : viewModel.canGenerate
    }
}

private struct ImagePickerRow: View {
    let label: String
    @Binding var path: String?
    let choose: () -> Void

    var body: some View {
        HStack {
            Text(label).font(.caption).frame(width: 84, alignment: .leading)
            Text(path.map { URL(fileURLWithPath: $0).lastPathComponent } ?? String(localized: "未選択"))
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
