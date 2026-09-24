import SwiftUI

// Two-pane creation workspace (design spec section 3): a fixed-width left
// pane for the form, and a flexible right pane for the preview. Only the
// form scrolls - the primary button and preview stay in view.
struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()
    @State private var showingModelManager = false

    var body: some View {
        HSplitView {
            CreationFormView(
                viewModel: viewModel, library: viewModel.library,
                showingModelManager: $showingModelManager
            )
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
                Button {
                    showingModelManager = true
                } label: {
                    Label("モデル管理", systemImage: "square.stack.3d.up")
                }
            }
            ToolbarItem(placement: .automatic) {
                EngineStatusButton(
                    viewModel: viewModel, library: viewModel.library,
                    showingModelManager: $showingModelManager
                )
            }
        }
        .sheet(isPresented: $showingModelManager) {
            ModelManagerView(library: viewModel.library) { id in
                viewModel.switchModel(to: id)
            }
        }
        .onAppear { viewModel.loadModel() }
    }
}
