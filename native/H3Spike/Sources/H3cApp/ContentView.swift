import SwiftUI

// The preview fills the window; the composer (作り方, prompt, shape / size /
// length, generate) floats in front of it at the bottom centre and can be
// dragged elsewhere or collapsed. 詳細設定 is a separate dialog. Each window
// has its own state: a new window starts like a fresh launch, from the
// preset used last.
struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()
    @State private var composerOffset: CGSize = .zero
    @State private var composerCollapsed = false
    @State private var presetName = ""
    @State private var showingModelManager = false
    @State private var showingDownloadWizard = false
    // Auto-offer the download wizard once per launch when the engine can't
    // load a model - not on every relaunch attempt the user might trigger
    // via "もう一度試す", and not re-shown after they dismiss it once.
    @State private var hasOfferedDownloadWizard = false

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                PreviewPane(viewModel: viewModel,
                            bottomInset: composerCollapsed ? 60 : 260)

                CreationFormView(
                    viewModel: viewModel, library: viewModel.library,
                    showingModelManager: $showingModelManager,
                    offset: $composerOffset, isCollapsed: $composerCollapsed,
                    bounds: geometry.size
                )
                .offset(composerOffset)
                .padding(.bottom, 20)
            }
        }
        .frame(minWidth: 760, minHeight: 600)
        // Once a generation starts (button, Return, ⌘Return or the HTTP
        // API), get the composer out of the way of the preview.
        .onChange(of: viewModel.isGenerating) { generating in
            if generating { withAnimation(.easeInOut(duration: 0.15)) { composerCollapsed = true } }
        }
        .alert("プリセットとして保存", isPresented: Binding(
            get: { viewModel.presetPendingName != nil },
            set: { if !$0 { viewModel.presetPendingName = nil } }
        )) {
            TextField("名前", text: $presetName)
            Button("保存") {
                if let preset = viewModel.presetPendingName {
                    viewModel.savePreset(preset, named: presetName)
                }
                presetName = ""
                viewModel.presetPendingName = nil
            }
            Button("キャンセル", role: .cancel) {
                presetName = ""
                viewModel.presetPendingName = nil
            }
        } message: {
            Text("形・大きさ・長さと詳細設定を保存します（プロンプトは含みません）。同じ名前のプリセットは上書きされます。")
        }
        .sheet(isPresented: $viewModel.showingAdvancedSettings) {
            AdvancedSettingsView(
                viewModel: viewModel, library: viewModel.library,
                showingModelManager: $showingModelManager
            )
        }
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
        .focusedSceneObject(viewModel)
        .onAppear {
            viewModel.loadModel()
            viewModel.applyLastUsedPreset()
        }
        .onChange(of: viewModel.engineState) { newValue in
            if case .failed = newValue, !hasOfferedDownloadWizard {
                hasOfferedDownloadWizard = true
                showingDownloadWizard = true
            }
        }
    }
}
