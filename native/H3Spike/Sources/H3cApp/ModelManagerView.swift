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
    /// reload the engine. Not called for LoRA changes - those only take
    /// effect at the next generate() call, no engine reload needed.
    var onActiveModelChange: (UUID?) -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme
    @State private var section: Section = .models
    @State private var renamingModelID: UUID?
    @State private var renamingLoRAID: UUID?
    @State private var renameText: String = ""

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
                Button("追加…") { addModel() }
            }
            .padding(H3Spacing.md)
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
        .onTapGesture { onActiveModelChange(model.id) }
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
                    loraRow(nil)
                    ForEach(library.loras) { entry in
                        loraRow(entry)
                    }
                }
                .listStyle(.inset)
            }
            HStack {
                Text(".safetensors形式のLoRAアダプタファイルを登録してください。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                Spacer()
                Button("追加…") { addLoRA() }
            }
            .padding(H3Spacing.md)
        }
    }

    /// nil represents "追加モデルなし" - always shown first, selectable like
    /// any registered entry, so the manager's own list is a complete picture
    /// of every possible selection state, not just the registered files.
    private func loraRow(_ entry: LoRAEntry?) -> some View {
        HStack(spacing: H3Spacing.sm) {
            let isActive = entry?.id == library.activeLoRAID
            Image(systemName: isActive ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(isActive ? palette.accent : palette.textSecondary)
            VStack(alignment: .leading, spacing: 2) {
                if let entry, renamingLoRAID == entry.id {
                    TextField("名前", text: $renameText, onCommit: {
                        library.renameLoRA(id: entry.id, name: renameText)
                        renamingLoRAID = nil
                    })
                    .textFieldStyle(.roundedBorder)
                } else {
                    Text(entry?.name ?? "なし").font(.body)
                }
                if let entry {
                    Text(entry.path)
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer()
            if let entry {
                HStack(spacing: 4) {
                    Text("強さ").font(.caption2).foregroundStyle(palette.textSecondary)
                    TextField("自動", text: Binding(
                        get: { entry.scaleText },
                        set: { library.setLoRAScale(id: entry.id, scaleText: $0) }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 56)
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
        }
        .contentShape(Rectangle())
        .onTapGesture { library.selectLoRA(entry?.id) }
        .padding(.vertical, 4)
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
