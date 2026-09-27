import SwiftUI

/// The dialog behind the toolbar's "モデル管理" button and the engine-status
/// popover's "モデル管理…" link. Manages ModelLibrary's two independent
/// lists - registered H3 checkpoint directories and registered LoRA files -
/// side by side in one sheet, since both are "pick a file/folder once,
/// reuse it by name later" in the same shape.
struct ModelManagerView: View {
    @ObservedObject var library: ModelLibrary
    /// Called after the active model changes (a direct selection, or the
    /// previously-active model being removed) so GenerationViewModel can
    /// reload the engine. Not called for LoRA changes - the stack only takes
    /// effect at the next generate() call, no engine reload needed.
    var onActiveModelChange: (UUID?) -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme
    @State private var section: Section = .models
    @State private var renamingModelID: UUID?
    @State private var renamingLoRAID: UUID?
    @State private var renameText: String = ""
    @State private var showingDownloadWizard = false

    private var palette: H3Palette { H3Palette(colorScheme) }

    private enum Section: String, CaseIterable, Identifiable {
        case models, loras
        var id: String { rawValue }
        var label: String { self == .models ? "H3モデル" : "LoRA" }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("モデル管理").font(.system(size: 18, weight: .semibold))
                Spacer()
                Button("閉じる") { dismiss() }
            }
            .padding(H3Spacing.lg)

            Picker("", selection: $section) {
                ForEach(Section.allCases) { item in Text(item.label).tag(item) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, H3Spacing.lg)
            .padding(.bottom, H3Spacing.md)

            Divider()

            switch section {
            case .models: modelsSection
            case .loras: lorasSection
            }
        }
        .frame(width: 560, height: 460)
        .background(palette.canvas)
    }

    // MARK: H3 models

    private var modelsSection: some View {
        VStack(spacing: 0) {
            if library.models.isEmpty {
                emptyState("登録されたH3モデルはありません。")
            } else {
                List {
                    ForEach(library.models) { model in
                        modelRow(model)
                    }
                }
                .listStyle(.inset)
            }
            HStack {
                Text("FL2VA（と、あれば任意でRef2VA）を含むフォルダを登録してください。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                Spacer()
                Button("ダウンロードして追加…") { showingDownloadWizard = true }
                Button("フォルダを追加…") { addModel() }
            }
            .padding(H3Spacing.md)
        }
        .sheet(isPresented: $showingDownloadWizard) {
            ModelDownloadWizardView(
                suggestedDestination: defaultH3ModelDownloadPath,
                onCompleted: { path in
                    let wasEmpty = library.models.isEmpty
                    library.addModel(path: path)
                    if wasEmpty { onActiveModelChange(library.activeModelID) }
                    showingDownloadWizard = false
                },
                onCancelled: { showingDownloadWizard = false }
            )
        }
    }

    private func modelRow(_ model: H3ModelEntry) -> some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: model.id == library.activeModelID ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(model.id == library.activeModelID ? palette.accent : palette.textSecondary)
            VStack(alignment: .leading, spacing: 2) {
                if renamingModelID == model.id {
                    TextField("名前", text: $renameText, onCommit: {
                        library.renameModel(id: model.id, name: renameText)
                        renamingModelID = nil
                    })
                    .textFieldStyle(.roundedBorder)
                } else {
                    Text(model.name).font(.body)
                }
                Text(model.path)
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(modelStatusText(model))
                    .font(.caption2)
                    .foregroundStyle(palette.textSecondary)
            }
            Spacer()
            Menu {
                Button("名前を変更") {
                    renameText = model.name
                    renamingModelID = model.id
                }
                Button("削除", role: .destructive) {
                    let wasActive = model.id == library.activeModelID
                    library.removeModel(id: model.id)
                    if wasActive { onActiveModelChange(library.activeModelID) }
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
        .contentShape(Rectangle())
        .onTapGesture {
            // switchModel(to:) always reloads (needed after a download
            // finishes into the already-active entry) - guard the redundant
            // case here instead, so re-tapping the current row is a no-op.
            if model.id != library.activeModelID { onActiveModelChange(model.id) }
        }
        .padding(.vertical, 4)
    }

    private func modelStatusText(_ model: H3ModelEntry) -> String {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        func has(_ subpath: String) -> Bool {
            fm.fileExists(atPath: model.path + subpath, isDirectory: &isDir) && isDir.boolValue
        }
        return [
            has("/FL2VA") ? "FL2VAあり" : "FL2VAなし",
            has("/Ref2VA") ? "Ref2VAあり" : "Ref2VAなし",
        ].joined(separator: " ・ ")
    }

    private func addModel() {
        guard let path = chooseDirectory() else { return }
        library.addModel(path: path)
    }

    // MARK: LoRA

    private var lorasSection: some View {
        VStack(spacing: 0) {
            if library.loras.isEmpty {
                emptyState("登録されたLoRAはありません。")
            } else {
                List {
                    ForEach(library.loras) { entry in
                        loraRow(entry)
                    }
                    .onMove { library.moveLoRAs(fromOffsets: $0, toOffset: $1) }
                }
                .listStyle(.inset)
            }
            HStack {
                Text("チェックしたLoRAを重ねて適用します（最大\(maxStackedLoRAs)個、ドラッグで並べ替え）。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                Spacer()
                Button("追加…") { addLoRA() }
            }
            .padding(H3Spacing.md)
        }
    }

    private func loraRow(_ entry: LoRAEntry) -> some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Toggle("", isOn: Binding(
                get: { entry.enabled },
                set: { library.setLoRAEnabled(id: entry.id, $0) }
            ))
            .toggleStyle(.checkbox)
            .labelsHidden()
            .disabled(!entry.enabled && library.enabledLoRAs.count >= maxStackedLoRAs)
            VStack(alignment: .leading, spacing: 2) {
                if renamingLoRAID == entry.id {
                    TextField("名前", text: $renameText, onCommit: {
                        library.renameLoRA(id: entry.id, name: renameText)
                        renamingLoRAID = nil
                    })
                    .textFieldStyle(.roundedBorder)
                } else {
                    Text(entry.name).font(.body)
                }
                Text(entry.path)
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                loraInfoText(entry)
            }
            Spacer()
            HStack(spacing: 4) {
                Text("強さ").font(.caption2).foregroundStyle(palette.textSecondary)
                TextField("1", text: Binding(
                    get: { entry.scaleText },
                    set: { library.setLoRAScale(id: entry.id, scaleText: $0) }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(width: 56)
                .help("学習時の効き具合に対する倍率。空欄は1。")
                Text("ステップ").font(.caption2).foregroundStyle(palette.textSecondary)
                TextField("—", text: Binding(
                    get: { entry.recommendedSteps.map(String.init) ?? "" },
                    set: { library.setLoRARecommendedSteps(id: entry.id, text: $0) }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(width: 40)
                .help("Turbo（蒸留）LoRAの学習ステップ数。オンにすると生成ステップ数がこれに切り替わります。空欄は通常のLoRA。")
            }
            Menu {
                Button("名前を変更") {
                    renameText = entry.name
                    renamingLoRAID = entry.id
                }
                Button("削除", role: .destructive) { library.removeLoRA(id: entry.id) }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
        .padding(.vertical, 4)
    }

    /// What h3_lora_inspect read from the file's header: layout, how much
    /// of the DiT it touches, rank - or why it can't be used.
    private func loraInfoText(_ entry: LoRAEntry) -> some View {
        let text: String
        var isError = false
        switch library.loraInfo(for: entry) {
        case .success(let info):
            var parts = [info.format,
                         "\(info.blocks)/50ブロック" + (info.refinerBlocks > 0 ? "+refiner" : ""),
                         info.rankMin == info.rankMax ? "rank \(info.rankMax)" : "rank \(info.rankMin)–\(info.rankMax)"]
            if info.unsupported > 0 { parts.append("未対応\(info.unsupported)件は無視") }
            text = parts.joined(separator: " ・ ")
        case .failure(let error):
            text = FileManager.default.fileExists(atPath: entry.path)
                ? "MiniMax-H3用として読み込めません: \(error.localizedDescription)"
                : "ファイルが見つかりません"
            isError = true
        }
        return Text(text)
            .font(.caption2)
            .foregroundStyle(isError ? palette.errorColor : palette.textSecondary)
            .lineLimit(2)
    }

    private func addLoRA() {
        guard let path = chooseFile(allowedContentTypes: safetensorsTypes) else { return }
        library.addLoRA(path: path)
    }

    private func emptyState(_ text: String) -> some View {
        VStack {
            Spacer()
            Text(text).font(.callout).foregroundStyle(palette.textSecondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }
}
