import SwiftUI

/// Top-of-window chrome. Two stacked plates:
///
///   • `HUDBar`        — interactive controls (opacity, font, accent, …)
///   • `StatusChips`   — read-only ASR / translation / Hermes status
///
/// Hides when the panel is not in focus and the user is not hovering.
///
/// The plate is mounted on top of the subtitle stage via
/// `.overlay(alignment: .top)` from `ContentView` so it never
/// participates in the subtitle's layout pass — see ContentView for
/// the rationale (selection stability). Visibility here is a
/// conditional `if`, with `.transition(.opacity)` for the fade. The
/// mount/unmount is wrapped in a `withAnimation` from the caller to
/// drive the transition; SwiftUI will not animate the transition
/// without that.
struct HUDView: View {
    @ObservedObject var state: OverlayState
    @ObservedObject var langConfig: LangConfig
    @Binding var style: OverlayStyle
    var onOpenSettings: () -> Void

    private var chromeVisible: Bool {
        // Hide while the startup banner is showing — the banner
        // covers the HUD's footprint anyway, and exposing both
        // looks busy during the loading pings. Banner is dismissed
        // when the first real subtitle arrives.
        state.isChromeVisible
            && state.isWindowActive
            && !state.startupBannerVisible
    }

    /// 切换音频采集源: 写 lang_config.json + 发 SIGHUP 给 whicc.py
    /// 触发热切换。SIGHUP 期间 statusText 切到 "切换音频源…"(橙色
    /// 指示), 等 whicc.py 写回 status 状态或下一次 ASR partial
    /// 到达时橙色自然消失。
    private func cycleAudioSource() {
        // HUD chip 只在 system ↔ mic 之间循环；指定应用模式点一下回到系统音频。
        let next: AudioSource = (state.audioSource == .system) ? .mic : .system
        state.audioSource = next
        langConfig.setAudioSource(next.rawValue)
        state.setTransientStatus("切换音频源…", color: .orange)
        // SIGHUP whicc.py
        BackendShutdown.signalWhiccForAudioSwitch()
    }

    var body: some View {
        if chromeVisible {
            VStack(spacing: 6) {
                HUDBar(
                    state: state,
                    langConfig: langConfig,
                    style: $style,
                    onOpenSettings: onOpenSettings
                )
                StatusChips(
                    state: state,
                    langConfig: langConfig,
                    onCycleAudioSource: { cycleAudioSource() }
                )
            }
            .padding(.horizontal, Palette.hudHPadding)
            .padding(.vertical, Palette.hudVPadding)
            .glassPlate()
            .padding(.horizontal, 12)
            .padding(.top, 0)
            .frame(maxWidth: .infinity, alignment: .top)
            .transition(.opacity)
        }
    }
}