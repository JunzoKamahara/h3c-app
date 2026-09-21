import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct PreviewPane: View {
    @ObservedObject var viewModel: GenerationViewModel
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            header
            centerContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            if viewModel.resultURL != nil, let result = viewModel.lastResult {
                ResultActionsView(viewModel: viewModel, result: result)
            }
        }
        .padding(H3Spacing.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.canvas)
    }

    private var header: some View {
        HStack {
            Text("プレビュー").font(.system(size: 14, weight: .semibold)).foregroundStyle(palette.textPrimary)
            Text(statusLabel).font(.caption).foregroundStyle(palette.textSecondary)
            Spacer()
        }
    }

    private var statusLabel: String {
        if viewModel.isCancelling { return "中止しています" }
        if viewModel.isGenerating { return "動画を生成しています" }
        if viewModel.resultURL != nil { return "できあがりました" }
        if let message = viewModel.errorMessage { return message }
        return ""
    }

    @ViewBuilder
    private var centerContent: some View {
        if let url = viewModel.resultURL {
            ResultPlayerView(url: url, aspectRatio: viewModel.resultAspectRatio)
        } else if viewModel.isGenerating {
            generatingView
        } else {
            emptyStateView
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: H3Spacing.md) {
            Image(systemName: "video")
                .font(.system(size: 32))
                .foregroundStyle(palette.textSecondary)
            Text("ここに動画が表示されます").font(.body).foregroundStyle(palette.textSecondary)
            Text("左側で内容を決めて、動画をつくりましょう。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
                    .multilineTextAlignment(.center)
                    .padding(.top, H3Spacing.sm)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.stage)
        .clipShape(RoundedRectangle(cornerRadius: H3Radius.stage))
    }

    private var generatingView: some View {
        VStack(spacing: H3Spacing.md) {
            ProgressView(value: viewModel.progressBarFraction)
                .frame(maxWidth: 260)

            VStack(spacing: 2) {
                Text(viewModel.stageTitle).font(.callout)
                Text(viewModel.stageDetail)
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(palette.textSecondary)
            }

            VStack(spacing: 2) {
                Text("経過 \(formatElapsed(viewModel.elapsedSeconds))")
                    .font(.caption)
                    .monospacedDigit()
                if let remaining = viewModel.estimatedRemainingSeconds,
                   let finishDate = viewModel.estimatedFinishDate {
                    Text("残り約 \(formatElapsed(remaining))（完了予定 \(formatClockTime(finishDate))）")
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(palette.textSecondary)
                } else {
                    Text("残り時間を見積り中…")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.stage)
        .clipShape(RoundedRectangle(cornerRadius: H3Radius.stage))
    }
}

private struct ResultActionsView: View {
    @ObservedObject var viewModel: GenerationViewModel
    let result: ResolvedResult
    @State private var showingSettings = false
    @State private var exportConfirmation: String?
    @State private var lastExportedURL: URL?

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.xs) {
            HStack(spacing: H3Spacing.sm) {
                Button("動画を書き出す…") { exportVideo() }
                    .keyboardShortcut("s", modifiers: .command)

                Button("設定を見る") { showingSettings = true }
                    .popover(isPresented: $showingSettings) {
                        SettingsDetailView(viewModel: viewModel, result: result)
                    }

                Button("この設定を使う") { viewModel.applyDraft(from: result) }

                Spacer()

                if let lastExportedURL {
                    Button("Finderで表示") {
                        NSWorkspace.shared.activateFileViewerSelecting([lastExportedURL])
                    }
                    .font(.caption)
                }
            }
            if let exportConfirmation {
                Text(exportConfirmation).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func exportVideo() {
        guard let sourceURL = viewModel.resultURLForExport else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = sourceURL.lastPathComponent
        panel.allowedContentTypes = [.mpeg4Movie]
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        try? FileManager.default.removeItem(at: destination)
        do {
            try FileManager.default.copyItem(at: sourceURL, to: destination)
            lastExportedURL = destination
            exportConfirmation = "保存しました: \(destination.lastPathComponent)"
        } catch {
            exportConfirmation = "選んだ場所に保存できませんでした（詳細: \(error.localizedDescription)）"
        }
    }
}

private extension GenerationViewModel {
    // Export needs the still-live temp file, which is intentionally not
    // otherwise exposed as a stable public URL (design spec: distinguish
    // the engine's temp output from a user-chosen save location).
    var resultURLForExport: URL? { resultURL }
}

private struct SettingsDetailView: View {
    @ObservedObject var viewModel: GenerationViewModel
    let result: ResolvedResult

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            Text("使用した設定").font(.headline)
            ScrollView {
                Text(result.settingsSummaryText)
                    .font(.caption)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 260)

            HStack {
                Button("設定をコピー") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(result.settingsSummaryText, forType: .string)
                }
                Button("同じシードを使う") { viewModel.useSameSeed(from: result) }
            }
        }
        .padding(H3Spacing.lg)
        .frame(minWidth: 360)
    }
}
