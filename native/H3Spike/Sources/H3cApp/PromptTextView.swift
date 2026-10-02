import AppKit
import SwiftUI

/// The prompt editor: Return submits, Shift+Return inserts a newline, and
/// Tab on an empty prompt accepts the grey suggestion as real text.
/// AppKit-backed because SwiftUI's TextEditor can't tell the two apart on
/// macOS 13. Return while an input method is composing (e.g. Japanese
/// conversion) still just commits the conversion - the input method
/// consumes that key before it becomes an insertNewline: command.
struct PromptTextView: NSViewRepresentable {
    @Binding var text: String
    var suggestion: String = ""
    var onSubmit: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        guard let textView = scrollView.documentView as? NSTextView else { return scrollView }
        textView.delegate = context.coordinator
        textView.drawsBackground = false
        textView.isRichText = false
        textView.allowsUndo = true
        textView.font = NSFont.preferredFont(forTextStyle: .body)
        textView.textContainerInset = NSSize(width: 6, height: 8)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.string = text
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = scrollView.documentView as? NSTextView else { return }
        // Don't disturb an in-progress input-method composition.
        if textView.string != text && !textView.hasMarkedText() {
            textView.string = text
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PromptTextView
        init(_ parent: PromptTextView) { self.parent = parent }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
        }

        func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            if selector == #selector(NSResponder.insertTab(_:)) {
                guard textView.string.isEmpty, !parent.suggestion.isEmpty else { return false }
                textView.string = parent.suggestion
                textView.setSelectedRange(NSRange(location: (parent.suggestion as NSString).length, length: 0))
                parent.text = parent.suggestion
                return true
            }
            guard selector == #selector(NSResponder.insertNewline(_:)) else { return false }
            if NSApp.currentEvent?.modifierFlags.contains(.shift) == true {
                textView.insertNewlineIgnoringFieldEditor(nil)
            } else {
                parent.onSubmit()
            }
            return true
        }
    }
}
