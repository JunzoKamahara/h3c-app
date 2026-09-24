import AppKit
import UniformTypeIdentifiers

let imageTypes: [UTType] = [.image]
// Ref2VA reference slots accept either a still image or a video
// (H3ReferenceKind.image/.video) - h3_av_reader.m decodes both natively now.
let referenceMediaTypes: [UTType] = [.image, .movie]
// "safetensors" has no registered system UTI, so this synthesizes a dynamic
// one from the extension - still filters the open panel correctly.
let safetensorsTypes: [UTType] = [UTType(filenameExtension: "safetensors") ?? .data]

func chooseFile(allowedContentTypes: [UTType]) -> String? {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.canChooseFiles = true
    panel.allowedContentTypes = allowedContentTypes
    return panel.runModal() == .OK ? panel.url?.path : nil
}

func chooseFiles(allowedContentTypes: [UTType]) -> [String] {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    panel.canChooseFiles = true
    panel.allowedContentTypes = allowedContentTypes
    return panel.runModal() == .OK ? panel.urls.map(\.path) : []
}

/// For registering an H3 checkpoint directory (e.g. a MiniMax-H3 snapshot
/// containing FL2VA/Ref2VA) in ModelLibrary.
func chooseDirectory() -> String? {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.canCreateDirectories = false
    return panel.runModal() == .OK ? panel.url?.path : nil
}

/// Which H3ReferenceKind a picked reference file should be tagged as -
/// image or video, by extension (via the same UTType conformance check
/// referenceMediaTypes filters the open panel with).
func isVideoFile(path: String) -> Bool {
    let ext = (path as NSString).pathExtension
    guard let type = UTType(filenameExtension: ext) else { return false }
    return type.conforms(to: .movie)
}
