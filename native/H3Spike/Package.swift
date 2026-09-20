// swift-tools-version:5.9
import Foundation
import PackageDescription

// Resolve the h3.c engine repo root (two levels up from this package: native/H3Spike -> native -> repo root)
// so the linker flags below work regardless of which directory `swift build` is invoked from.
let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let repoRoot = packageDir.deletingLastPathComponent().deletingLastPathComponent().path

let h3LinkerSettings: [LinkerSetting] = [
    .unsafeFlags(["-L\(repoRoot)", "-lh3", "-licucore"]),
    .linkedFramework("Foundation"),
    .linkedFramework("Metal"),
    .linkedFramework("MetalPerformanceShaders"),
    .linkedFramework("MetalPerformanceShadersGraph"),
    .linkedFramework("Accelerate"),
]

let package = Package(
    name: "H3Spike",
    platforms: [.macOS(.v13)],
    targets: [
        .target(
            name: "CH3",
            path: "Sources/CH3"
        ),
        .target(
            name: "H3Engine",
            dependencies: ["CH3"],
            path: "Sources/H3Engine"
        ),
        .executableTarget(
            name: "H3Spike",
            dependencies: ["CH3"],
            path: "Sources/H3Spike",
            linkerSettings: h3LinkerSettings
        ),
        .executableTarget(
            name: "H3App",
            dependencies: ["H3Engine"],
            path: "Sources/H3App",
            linkerSettings: h3LinkerSettings
        ),
    ]
)
