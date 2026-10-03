import SwiftUI

/// First-run (and "モデル管理" > "ダウンロードして追加…") setup flow: fetches
/// MiniMax-H3's real file list from Hugging Face, checks disk space, then
/// downloads FL2VA (and optionally Ref2VA) with ModelDownloader. Hands the
/// destination path back via `onCompleted` once done - the caller decides
/// whether that updates an existing ModelLibrary entry or adds a new one,
/// this view only knows about the download itself.
struct ModelDownloadWizardView: View {
    let onCompleted: (String) -> Void
    let onCancelled: () -> Void

    @StateObject private var downloader = ModelDownloader()
    @Environment(\.colorScheme) private var colorScheme
    @State private var destination: String
    @State private var includeRef2VA = false

    init(suggestedDestination: String, onCompleted: @escaping (String) -> Void, onCancelled: @escaping () -> Void) {
        self.onCompleted = onCompleted
        self.onCancelled = onCancelled
        _destination = State(initialValue: suggestedDestination)
    }

    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.lg) {
            Text("MiniMax-H3 モデルのダウンロード")
                .font(.system(size: 18, weight: .semibold))

            existingFolderRow

            Divider()

            destinationRow

            Toggle("Ref2VA も含める（参照画像・動画機能。おおよそ倍のサイズになります）", isOn: $includeRef2VA)
                .disabled(!canEditOptions)

            Divider()

            statusSection

            Spacer(minLength: 0)

            footer
        }
        .padding(H3Spacing.xl)
        .frame(width: 520, height: 420)
        .background(palette.canvas)
        .onDisappear { downloader.cancel() }
    }

    /// Lets a user who already has FL2VA/Ref2VA on disk (e.g. copied from
    /// another Mac, or downloaded outside this app) register that folder
    /// directly instead of downloading again - the same "フォルダを追加…"
    /// escape hatch ModelManagerView offers, surfaced here too since this
    /// view is also what greets a first launch with no model configured.
    private var existingFolderRow: some View {
        HStack {
            Text("FL2VA（と、あればRef2VA）を含むフォルダが既にある場合は、ダウンロードせずに登録できます。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
            Spacer()
            Button("フォルダを追加…") { addExistingFolder() }
                .disabled(!canEditOptions)
        }
    }

    private func addExistingFolder() {
        guard let path = chooseDirectory() else { return }
        onCompleted(path)
    }

    private var canEditOptions: Bool {
        switch downloader.state {
        case .idle, .failed: return true
        default: return false
        }
    }

    private var destinationRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("保存先").font(.caption).foregroundStyle(palette.textSecondary)
            HStack {
                Text(destination)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button("変更…") {
                    if let path = chooseDirectory() { destination = path + "/MiniMax-H3" }
                }
                .disabled(!canEditOptions)
            }
        }
    }

    @ViewBuilder
    private var statusSection: some View {
        switch downloader.state {
        case .idle:
            Text("Hugging Face（MiniMaxAI/MiniMax-H3）から直接ダウンロードします。FL2VAだけで約130GB、Ref2VAも含めると約260GBと非常に大きいので、時間と空き容量に余裕がある回線・ディスクで実行してください。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)

        case .listing:
            HStack {
                ProgressView().controlSize(.small)
                Text("ファイル一覧を確認しています…").font(.caption).foregroundStyle(palette.textSecondary)
            }

        case .ready:
            Text("合計 \(formatGB(downloader.totalBytes))（\(downloader.totalFiles)ファイル）。「ダウンロード開始」を押してください。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)

        case .downloading:
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(downloader.downloadedBytes), total: Double(max(downloader.totalBytes, 1)))
                Text([
                    "\(formatGB(downloader.downloadedBytes)) / \(formatGB(downloader.totalBytes))",
                    String(localized: "\(downloader.filesCompleted)/\(downloader.totalFiles)ファイル"),
                    formatSpeed(downloader.bytesPerSecond),
                ].joined(separator: summarySeparator))
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                Text("中断しても、次回は続きから再開します。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }

        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(palette.errorColor)

        case .completed:
            Text("ダウンロードが完了しました。").font(.caption).foregroundStyle(palette.textSecondary)
        }
    }

    private var footer: some View {
        HStack {
            Button("あとで") { onCancelled() }
                .font(.caption)
            Spacer()
            switch downloader.state {
            case .idle:
                Button("サイズを確認…") { downloader.listFiles(includeRef2VA: includeRef2VA) }
                    .buttonStyle(.borderedProminent)
            case .listing:
                EmptyView()
            case .ready:
                Button("ダウンロード開始") { downloader.startDownload(destination: destination) }
                    .buttonStyle(.borderedProminent)
            case .downloading:
                Button("中止") { downloader.cancel() }
            case .failed:
                Button("再開") { downloader.startDownload(destination: destination) }
                    .buttonStyle(.borderedProminent)
            case .completed:
                Button("完了") { onCompleted(destination) }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private func formatGB(_ bytes: Int64) -> String {
        String(format: "%.1fGB", Double(bytes) / 1_073_741_824)
    }

    private func formatSpeed(_ bytesPerSecond: Double) -> String {
        guard bytesPerSecond > 0 else { return String(localized: "計測中…") }
        return String(format: "%.1fMB/s", bytesPerSecond / 1_048_576)
    }
}
