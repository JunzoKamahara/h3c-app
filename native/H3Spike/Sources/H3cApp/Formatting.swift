import Foundation

func formatElapsed(_ seconds: Double) -> String {
    let total = max(Int(seconds.rounded()), 0)
    let minutes = total / 60
    let secs = total % 60
    return minutes > 0 ? String(format: "%d:%02d", minutes, secs) : String(localized: "\(secs)秒")
}

/// Joins short items in one-line summaries (" ・ " in Japanese).
let summarySeparator = String(localized: " ・ ")

private let clockFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.timeStyle = .short
    formatter.dateStyle = .none
    return formatter
}()

func formatClockTime(_ date: Date) -> String {
    clockFormatter.string(from: date)
}
