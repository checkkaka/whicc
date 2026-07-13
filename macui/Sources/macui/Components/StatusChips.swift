import SwiftUI

/// Bottom status strip showing ASR backend (with audio source toggle),
/// translation reachability, and Hermes reachability. Mirrors the
/// legacy overlay's persistent chips.
///
/// The ASR chip merges two concerns: backend name (Nemotron / Qwen3)
/// and audio source icon (speaker for system audio, mic for built-in
/// mic). Clicking the chip cycles between system audio and mic — the
/// change is written to lang_config.json + a SIGHUP is sent to whicc.py
/// which hot-swaps the active AudioSource without restarting ASR.
///
/// Note: we previously had a separate `AudioSourceController.swift`
/// that forked the audio subprocess; that predated the `src/audio.py`
/// refactor which moved audio capture into whicc.py. The old
/// controller is gone — this chip now mutates `state.audioSource`
/// directly and signals whicc.py via SIGHUP.
struct StatusChips: View {
    @ObservedObject var state: OverlayState
    @ObservedObject var langConfig: LangConfig
    /// Click handler — macui passes a closure that writes lang_config.json
    /// and signals whicc.py. Kept out of StatusChips so the chip stays a
    /// pure view of state.
    var onCycleAudioSource: () -> Void = {}

    var body: some View {
        HStack(spacing: 8) {
            asrChip
            chip(key: "文", label: translationLabel, dot: translationDot)
            chip(key: "Hermes", label: hermesLabel, dot: hermesDot)
        }
        .padding(.horizontal, 6)
        .frame(height: Palette.controlHeight)
        .background(Capsule().fill(Palette.controlFill))
    }

    // MARK: ASR chip

    /// ASR chip with the audio-source icon on the left, the backend
    /// name on the right, and a colored dot in the middle reflecting
    /// the audio child's runtime state.
    private var asrChip: some View {
        Button {
            onCycleAudioSource()
        } label: {
            HStack(spacing: 3) {
                // Source icon — speaker when capturing system audio,
                // mic when capturing the built-in microphone. Tinted
                // by runtime state so the user can read the status at
                // a glance.
                Image(systemName: state.audioSource.icon)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(asrTint)
                    .frame(width: 12, height: 12)
                Circle().fill(asrDot).frame(width: 6, height: 6)
                Text("ASR")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(Palette.textTertiary)
                Text(asrLabel)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
            }
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
        }
        .buttonStyle(.plain)
        .help(asrHelpText)
    }

    /// Icon tint reflects the audio subprocess lifecycle:
    ///   - white:  child is alive and writing PCM segments
    ///   - orange: child is being (re)started (incl. source switch,
    ///             model warmup). Triggered when `state.audioSource`
    ///             was just changed and we're awaiting whicc.py
    ///             confirmation via SIGHUP.
    ///   - red:    child died or failed to signal ready
    private var asrTint: Color {
        // During ASR warmup (orange statusText) the icon dims.
        if state.statusText != nil, state.statusColor == .orange {
            return Palette.textSecondary
        }
        return Palette.textPrimary
    }

    /// Dot color: green = ready, orange = starting/error mid-switch.
    /// We use the same logic as asrTint for parity (single source of
    /// truth for ASR lifecycle cues).
    private var asrDot: Color {
        if state.statusText != nil, state.statusColor == .orange {
            return .orange
        }
        return .green
    }

    private var asrLabel: String {
        // Mid model-switch the backend name fades to the target
        // so the user can see which model is loading.
        if let s = state.statusText, state.statusColor == .orange {
            if s.contains("Qwen3") { return "→ Qwen3" }
            if s.contains("Nemotron") { return "→ Nemotron" }
        }
        return state.asrBackend == "qwen3" ? "Qwen3" : "Nemotron"
    }

    private var asrHelpText: String {
        let current: String
        switch state.audioSource {
        case .system: current = "系统声音"
        case .mic: current = "麦克风"
        case .application:
            let name = langConfig.audioAppDisplayName
            current = name.isEmpty ? "指定应用" : name
        }
        let target: String
        switch state.audioSource {
        case .system: target = "麦克风"
        case .mic, .application: target = "系统声音"
        }
        return "ASR: \(asrLabel) · 当前采集：\(current)\n点击切到\(target)"
    }

    // MARK: Generic chip

    private func chip(key: String, label: String, dot: Color) -> some View {
        HStack(spacing: 3) {
            Circle().fill(dot).frame(width: 6, height: 6)
            Text(key)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundColor(Palette.textTertiary)
            Text(label)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
        }
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
    }

    // MARK: Translation chip

    private var translationLabel: String {
        if langConfig.translationUrl.isEmpty { return "Local" }
        let url = langConfig.translationUrl
            .replacingOccurrences(of: "http://", with: "")
            .replacingOccurrences(of: "https://", with: "")
        if url.hasPrefix("127.") || url.hasPrefix("localhost") { return "Local" }
        if let host = url.split(separator: ":").first { return String(host) }
        return "LAN"
    }

    private var translationDot: Color {
        switch langConfig.translationReachable {
        case .some(true):  return .green
        case .some(false): return .red
        case .none:        return .gray
        }
    }

    // MARK: Hermes chip

    private var hermesLabel: String {
        if langConfig.hermesHost.isEmpty { return "Off" }
        let h = langConfig.hermesHost
        if h.contains("127.0.0.1") || h.contains("localhost") { return "Local" }
        if let host = h.split(separator: ".").first { return String(host) }
        return "LAN"
    }

    private var hermesDot: Color {
        switch langConfig.hermesReachable {
        case .some(true):  return .green
        case .some(false): return .red
        case .none:        return .gray
        }
    }
}
