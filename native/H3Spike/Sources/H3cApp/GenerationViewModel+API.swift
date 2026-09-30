import Foundation
import H3Engine

/// The local automation API this app exposes while it's running - the
/// direct replacement for the old Python gui/server.py, which this app no
/// longer ships (removed along with the Python dependency it needed). The
/// difference in shape isn't incidental: gui/server.py had to accept
/// browser file uploads because its client was a browser; a script driving
/// this API runs on the same Mac, so it can just pass filesystem paths.
///
/// POST /api/generate fully replaces the current draft from its JSON body
/// (same fields the form itself edits) and, if valid, starts a job exactly
/// as pressing "動画をつくる" would - there is only ever one job at a time,
/// shared with the UI, so a request while one is already running gets 409.
/// GET /api/status polls progress. GET /api/result/video streams the
/// current result until the next generate() call deletes it.
let h3APIPort: UInt16 = 8420

extension GenerationViewModel {
    func startAPIServer() {
        let server = HTTPServer(port: h3APIPort) { [weak self] request in
            guard let self else { return .error(500, "app is shutting down") }
            return await self.handleAPIRequest(request)
        }
        do {
            try server.start()
            apiServer = server
            apiServerStatus = "http://127.0.0.1:\(h3APIPort)"
        } catch {
            apiServerStatus = "起動できませんでした（\(error)）"
        }
    }

    func handleAPIRequest(_ request: HTTPRequest) async -> HTTPResponse {
        switch (request.method, request.path) {
        case ("GET", "/api/status"):
            return statusResponse()
        case ("POST", "/api/generate"):
            return handleGenerateRequest(request)
        case ("POST", "/api/cancel"):
            cancel()
            return .json(200, ["ok": true])
        case ("GET", "/api/result/video"):
            return resultVideoResponse()
        case ("GET", "/api/models"):
            return .json(200, library.models.map {
                ["id": $0.id.uuidString, "name": $0.name, "path": $0.path, "active": $0.id == library.activeModelID]
            })
        case ("GET", "/api/loras"):
            return .json(200, library.loras.map {
                ["id": $0.id.uuidString, "name": $0.name, "path": $0.path,
                 "strength": $0.strength, "recommended_steps": $0.recommendedSteps.map { $0 as Any } ?? NSNull(),
                 "enabled": $0.enabled]
            })
        default:
            return .error(404, "no such endpoint: \(request.method) \(request.path)")
        }
    }

    private func statusResponse() -> HTTPResponse {
        let engineStateText: String
        switch engineState {
        case .loading: engineStateText = "loading"
        case .ready: engineStateText = "ready"
        case .failed(let message): engineStateText = "failed: \(message)"
        }
        var body: [String: Any] = [
            "engine_state": engineStateText,
            "device": deviceLine,
            "is_generating": isGenerating,
            "is_cancelling": isCancelling,
            "phase": phase,
            "elapsed_seconds": elapsedSeconds,
            "stage_title": stageTitle,
            "stage_detail": stageDetail,
            "result_available": resultURL != nil,
        ]
        if let estimatedRemainingSeconds { body["estimated_remaining_seconds"] = estimatedRemainingSeconds }
        if let progressBarFraction { body["progress_fraction"] = progressBarFraction }
        if let errorMessage { body["error_message"] = errorMessage }
        return .json(200, body)
    }

    private func resultVideoResponse() -> HTTPResponse {
        guard let resultURL else { return .error(404, "no result available yet") }
        return .file(atPath: resultURL.path, contentType: "video/mp4")
    }

    private func handleGenerateRequest(_ request: HTTPRequest) -> HTTPResponse {
        guard let json = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] else {
            return .error(400, "expected a JSON object body")
        }
        guard let requestedPrompt = (json["prompt"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !requestedPrompt.isEmpty else {
            return .error(400, "\"prompt\" is required")
        }

        // Every field below is set from this request (defaulting when
        // omitted), not merged onto whatever the UI's form last held - a
        // script driving this endpoint shouldn't have to know or care what
        // a human last typed into the window.
        prompt = requestedPrompt

        if let profileRaw = json["size_profile"] as? String {
            guard let profile = SizeProfile(rawValue: profileRaw) else {
                return .error(400, "unknown size_profile \(profileRaw) - expected one of: "
                    + SizeProfile.allCases.map(\.rawValue).joined(separator: ", "))
            }
            sizeProfile = profile
        } else {
            sizeProfile = .square
        }

        seconds = (json["seconds"] as? Int) ?? 5
        denoiseReuse = (json["reuse"] as? Int) ?? 1

        if let modeRaw = json["compute_mode"] as? String {
            guard let mode = ComputeMode(rawValue: modeRaw) else {
                return .error(400, "unknown compute_mode \(modeRaw) - expected attentionCache, resident, or ssdStreaming")
            }
            computeMode = mode
        } else {
            computeMode = defaultComputeMode
        }

        if let speedRaw = json["speed_mode"] as? String {
            guard let speed = SpeedMode(rawValue: speedRaw) else {
                return .error(400, "unknown speed_mode \(speedRaw) - expected quality, fast, or fastest")
            }
            speedMode = speed
        } else {
            speedMode = .quality
        }

        if let fast = json["fast_attention"] {
            guard let fast = fast as? Bool else {
                return .error(400, "\"fast_attention\" must be true or false")
            }
            if fast && !fastAttentionAvailable {
                return .error(400, "fast_attention is not available on this build/device")
            }
            fastAttention = fast
        } else {
            fastAttention = false
        }

        if let seed = json["seed"] {
            seedText = "\(seed)".filter(\.isNumber)
        } else {
            seedText = ""
        }

        let firstFrame = json["first_frame_path"] as? String
        let lastFrame = json["last_frame_path"] as? String
        let referencePaths = (json["reference_paths"] as? [String]) ?? []

        if firstFrame != nil || lastFrame != nil {
            if !referencePaths.isEmpty {
                return .error(400, "first_frame_path/last_frame_path cannot be combined with reference_paths")
            }
            creationMethod = .image
            imageInputMode = .firstLastFrame
            firstFramePath = firstFrame
            lastFramePath = lastFrame
        } else if !referencePaths.isEmpty {
            creationMethod = .image
            imageInputMode = .referenceImage
            referenceImages = referencePaths.map { path in
                let kind: H3ReferenceKind = isVideoFile(path: path) ? .video : (isAudioFile(path: path) ? .audio : .image)
                return H3ReferenceInput(kind: kind, path: path)
            }
        } else {
            creationMethod = .text
        }

        // "loras": [{"name": ..., "strength": 0.8}, ...] (or bare names)
        // replaces the stack; the older single "lora_name"/"lora_scale"
        // pair still works. Neither means no LoRA. Strength multiplies the
        // adapter's own alpha/rank scale; omitted keeps the entry's own.
        var requested: [(name: String, strength: Any?)] = []
        if let list = json["loras"] as? [Any] {
            for item in list {
                if let name = item as? String {
                    requested.append((name, nil))
                } else if let object = item as? [String: Any], let name = object["name"] as? String {
                    requested.append((name, object["strength"]))
                } else {
                    return .error(400, "each \"loras\" item must be a name or {\"name\", \"strength\"}")
                }
            }
        } else if let loraName = json["lora_name"] as? String, !loraName.isEmpty {
            requested.append((loraName, json["lora_scale"]))
        }
        if requested.count > maxStackedLoRAs {
            return .error(400, "at most \(maxStackedLoRAs) LoRAs can be stacked")
        }
        var stack: [(entry: LoRAEntry, strength: Any?)] = []
        for (name, strength) in requested {
            guard let entry = library.loras.first(where: { $0.name == name }) else {
                return .error(400, "unknown LoRA \(name) - see GET /api/loras")
            }
            if let strength, !(strength is NSNumber) {
                return .error(400, "strength for \(name) must be a number")
            }
            stack.append((entry, strength))
        }
        for (entry, strength) in stack {
            if let strength { library.setLoRAScale(id: entry.id, scaleText: "\(strength)") }
        }
        library.setEnabledLoRAs(Set(stack.map { $0.entry.id }))

        // After the LoRAs are settled: enabling a Turbo LoRA moves the draft
        // to its recommended steps (followTurboLoRASteps), which an explicit
        // "steps" in the request still overrides.
        let turboSteps = library.enabledLoRAs.first { $0.recommendedSteps != nil }?.recommendedSteps
        steps = (json["steps"] as? Int) ?? turboSteps ?? defaultStepsForAPI

        guard canGenerate else {
            return .error(409, validationMessage ?? "a generation is already running")
        }
        generate()
        return .json(202, ["ok": true])
    }
}

/// h3.h's own default - kept here rather than importing CH3 into this
/// small API file just for one constant.
private let defaultStepsForAPI = 20
