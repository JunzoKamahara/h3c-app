import Foundation
import SwiftUI

@main
struct H3App: App {
    init() {
        // libh3.a resolves "h3_shaders.metal" relative to the process's
        // working directory (see h3.c). When running from inside an .app
        // bundle, LaunchServices doesn't guarantee any particular cwd, so
        // point it at Contents/Resources ourselves if the shader is there.
        if let resourcePath = Bundle.main.resourcePath,
           FileManager.default.fileExists(atPath: resourcePath + "/h3_shaders.metal") {
            FileManager.default.changeCurrentDirectoryPath(resourcePath)
        }
        // H3_ATTENTION_CACHE is no longer set here: GenerationViewModel now
        // picks the right cache (default/Ref2VA/Turbo) per job and passes it
        // through H3GenerationParams.attentionCachePath instead, since which
        // one is correct depends on that job's mode.
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}
