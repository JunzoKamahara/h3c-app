import Foundation

/// This developer's own hand-placed model checkout - not inside Application
/// Support like a fresh download would be, so it needs its own literal path
/// here to be recognized. Originally at .../h3c-analysis/MiniMax-H3 (the
/// app's old name), then moved to ~/models/MiniMax-H3 to get the ~270GB
/// model and its caches out of ~/Library. Kept as an exact-path match so
/// ModelLibrary can seed an entry for it (if it exists) and
/// GenerationViewModel can reuse its already-built, multi-GB attention
/// caches instead of demanding a rebuild under a new per-model cache
/// directory. Never used to suggest a destination for a new download - see
/// defaultH3ModelDownloadPath for that.
let legacyDefaultH3ModelPath = NSHomeDirectory() + "/models/MiniMax-H3"

/// Suggested destination for a model a brand-new install downloads, under
/// this app's own current name - unlike legacyDefaultH3ModelPath above,
/// which only ever refers to a specific pre-existing folder from before the
/// app was renamed.
let defaultH3ModelDownloadPath = h3AppSupportDirectory + "/MiniMax-H3"

struct H3ModelEntry: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    var path: String

    init(id: UUID = UUID(), name: String, path: String) {
        self.id = id
        self.name = name
        self.path = path
    }
}

struct LoRAEntry: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    var path: String
    /// Empty means "auto": the engine detects a scale from the adapter's
    /// own alpha/rank metadata instead (h3_lora_detect_scale in h3_lora.c).
    var scaleText: String

    init(id: UUID = UUID(), name: String, path: String, scaleText: String = "") {
        self.id = id
        self.name = name
        self.path = path
        self.scaleText = scaleText
    }
}

/// Registered H3 checkpoint directories and LoRA files, operated on through
/// ModelManagerView. Persisted in UserDefaults - this only stores small
/// JSON (names/paths/ids), never the multi-GB model or cache files
/// themselves, which stay wherever the user pointed the library at.
@MainActor
final class ModelLibrary: ObservableObject {
    @Published private(set) var models: [H3ModelEntry] = []
    @Published private(set) var activeModelID: UUID?
    @Published private(set) var loras: [LoRAEntry] = []
    @Published private(set) var activeLoRAID: UUID?

    private let defaults = UserDefaults.standard
    private static let modelsKey = "H3ModelLibrary.models"
    private static let activeModelKey = "H3ModelLibrary.activeModelID"
    private static let lorasKey = "H3ModelLibrary.loras"
    private static let activeLoRAKey = "H3ModelLibrary.activeLoRAID"

    var activeModel: H3ModelEntry? { models.first { $0.id == activeModelID } }
    var activeLoRA: LoRAEntry? { loras.first { $0.id == activeLoRAID } }

    init() {
        models = Self.decodeArray(H3ModelEntry.self, key: Self.modelsKey)
        loras = Self.decodeArray(LoRAEntry.self, key: Self.lorasKey)
        activeModelID = UUID(uuidString: defaults.string(forKey: Self.activeModelKey) ?? "")
        activeLoRAID = UUID(uuidString: defaults.string(forKey: Self.activeLoRAKey) ?? "")

        // Upgrading from before this feature existed: seed the one path the
        // app used to hardcode, so an existing setup (and any already-built
        // attention caches - see GenerationViewModel.attentionCacheDirectory(for:))
        // keeps working without the user having to re-add it by hand. Only
        // done when that exact folder is actually there - a brand-new
        // install (e.g. from the distributed .dmg, on another Mac) has
        // nothing at that path and should see an empty library instead,
        // so the download wizard offers defaultH3ModelDownloadPath rather
        // than this app's old name.
        if models.isEmpty, FileManager.default.fileExists(atPath: legacyDefaultH3ModelPath) {
            let seeded = H3ModelEntry(name: "MiniMax-H3", path: legacyDefaultH3ModelPath)
            models = [seeded]
            activeModelID = seeded.id
            persistModels()
            persistActiveModel()
        }
    }

    // MARK: Models

    func addModel(path: String, name: String? = nil) {
        let entry = H3ModelEntry(name: name ?? (path as NSString).lastPathComponent, path: path)
        models.append(entry)
        if activeModelID == nil {
            activeModelID = entry.id
            persistActiveModel()
        }
        persistModels()
    }

    func renameModel(id: UUID, name: String) {
        guard let index = models.firstIndex(where: { $0.id == id }), !name.isEmpty else { return }
        models[index].name = name
        persistModels()
    }

    /// Used after ModelDownloadWizardView finishes downloading into a
    /// possibly-different folder than the entry originally pointed at.
    func setModelPath(id: UUID, path: String) {
        guard let index = models.firstIndex(where: { $0.id == id }) else { return }
        models[index].path = path
        persistModels()
    }

    func removeModel(id: UUID) {
        models.removeAll { $0.id == id }
        if activeModelID == id {
            activeModelID = models.first?.id
            persistActiveModel()
        }
        persistModels()
    }

    func selectModel(id: UUID) {
        guard models.contains(where: { $0.id == id }) else { return }
        activeModelID = id
        persistActiveModel()
    }

    // MARK: LoRA

    func addLoRA(path: String, name: String? = nil) {
        let entry = LoRAEntry(name: name ?? (path as NSString).lastPathComponent, path: path)
        loras.append(entry)
        persistLoRAs()
    }

    func renameLoRA(id: UUID, name: String) {
        guard let index = loras.firstIndex(where: { $0.id == id }), !name.isEmpty else { return }
        loras[index].name = name
        persistLoRAs()
    }

    func setLoRAScale(id: UUID, scaleText: String) {
        guard let index = loras.firstIndex(where: { $0.id == id }) else { return }
        let filtered = scaleText.filter { $0.isNumber || $0 == "." }
        loras[index].scaleText = filtered
        persistLoRAs()
    }

    func removeLoRA(id: UUID) {
        loras.removeAll { $0.id == id }
        if activeLoRAID == id {
            activeLoRAID = nil
            persistActiveLoRA()
        }
        persistLoRAs()
    }

    /// nil deselects - "追加モデルなし" is a valid, common choice.
    func selectLoRA(_ id: UUID?) {
        activeLoRAID = id
        persistActiveLoRA()
    }

    // MARK: Persistence

    private func persistModels() { Self.encodeArray(models, key: Self.modelsKey) }
    private func persistLoRAs() { Self.encodeArray(loras, key: Self.lorasKey) }

    private func persistActiveModel() {
        defaults.set(activeModelID?.uuidString, forKey: Self.activeModelKey)
    }

    private func persistActiveLoRA() {
        defaults.set(activeLoRAID?.uuidString, forKey: Self.activeLoRAKey)
    }

    private static func decodeArray<T: Decodable>(_ type: T.Type, key: String) -> [T] {
        guard let data = UserDefaults.standard.data(forKey: key),
              let value = try? JSONDecoder().decode([T].self, from: data) else { return [] }
        return value
    }

    private static func encodeArray<T: Encodable>(_ value: [T], key: String) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }
}
