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

        // Dev-time default: use the validated int8 attention cache (symlinked
        // into the repo root, see dit_int8_v2.cache) instead of the slower
        // close-reference BF16 path libh3.a falls back to when this is unset.
        // `open`/LaunchServices launches don't inherit the shell's
        // environment, so without this the app silently runs uncached.
        // TODO: replace with a real setting once packaging is figured out -
        // this 19GB cache can't be bundled into the .app itself.
        if getenv("H3_ATTENTION_CACHE") == nil {
            setenv("H3_ATTENTION_CACHE", "/Users/kamahara/Documents/work/h3c-app/dit_int8_v2.cache", 1)
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}
