import SwiftUI

// Two-pane creation workspace (design spec section 3): a fixed-width left
// pane for the form, and a flexible right pane for the preview. Only the
// form scrolls - the primary button and preview stay in view.
struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()
    @State private var showingModelManager = false
    @State private var showingDownloadWizard = false
    // Auto-offer the download wizard once per launch when the engine can't
    // load a model - not on every relaunch attempt the user might trigger
    // via "もう一度試す", and not re-shown after they dismiss it once.
    @State private var hasOfferedDownloadWizard = false

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
        .sheet(isPresented: $showingDownloadWizard) {
            ModelDownloadWizardView(
                suggestedDestination: viewModel.library.activeModel?.path ?? defaultH3ModelDownloadPath,
                onCompleted: { path in
                    if let id = viewModel.library.activeModelID {
                        viewModel.library.setModelPath(id: id, path: path)
                        viewModel.switchModel(to: id)
                    } else {
                        viewModel.library.addModel(path: path)
                        if let newID = viewModel.library.activeModelID { viewModel.switchModel(to: newID) }
                    }
                    showingDownloadWizard = false
                },
                onCancelled: { showingDownloadWizard = false }
            )
        }
        .onAppear { viewModel.loadModel() }
        .onChange(of: viewModel.engineState) { newValue in
            if case .failed = newValue, !hasOfferedDownloadWizard {
                hasOfferedDownloadWizard = true
                showingDownloadWizard = true
            }
        }
    }
}
