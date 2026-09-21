import AppKit
import UniformTypeIdentifiers

let imageTypes: [UTType] = [.image]
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
