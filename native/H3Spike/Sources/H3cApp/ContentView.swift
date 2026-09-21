import SwiftUI

// Two-pane creation workspace (design spec section 3): a fixed-width left
// pane for the form, and a flexible right pane for the preview. Only the
// form scrolls - the primary button and preview stay in view.
struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()

    var body: some View {
        HSplitView {
            CreationFormView(viewModel: viewModel)
                .frame(minWidth: 320, idealWidth: 352, maxWidth: 400)

            PreviewPane(viewModel: viewModel)
                .frame(minWidth: 400)
        }
        .frame(minWidth: 920, minHeight: 640)
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Text("h3c-app").font(.headline)
            }
            ToolbarItem(placement: .automatic) {
                EngineStatusButton(viewModel: viewModel)
            }
        }
        .onAppear { viewModel.loadModel() }
    }
}
