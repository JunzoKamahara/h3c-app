import Foundation
import SwiftUI

@main
struct H3cApp: App {
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
        // The Swift type/module is named H3cApp (Swift module names
        // can't contain a hyphen), but every user-visible name - this
        // window's title, the app/menu-bar name from Info.plist, the
        // bundle and executable filenames from package_app.sh - is
        // h3c-app, matching the actual product/repo name.
        WindowGroup("h3c-app") {
            ContentView()
        }
        // .contentSize forced the window to match the form's full ideal
        // height, which now runs taller than the screen with every control
        // added - .automatic lets it size within screen bounds and be
        // resized by hand, with ContentView's own ScrollView covering the
        // rest. defaultSize keeps the initial window at a sane size instead
        // of AppKit's own default (which ended up stretched full-width).
        .windowResizability(.automatic)
        .defaultSize(width: 640, height: 700)
        // Without this, AppKit cascades each new window progressively
        // down/right from the last - after many relaunches during
        // development the window had drifted low enough to push its bottom
        // (Save As...) past the dock. Centering keeps it reachable.
        .defaultPosition(.center)
    }
}
