import AppKit
import Combine
import SwiftUI

/// Owns the `SubtitlePanel`, its background layer, hover tracker, and
/// the key-window observers. The actual subtitle UI is hosted via an
/// `NSHostingView` whose root is `ContentView`.
@MainActor
final class OverlayWindowController: NSObject {

    let state: OverlayState
    let langConfig: LangConfig
    let glossaryState: GlossaryState
    let eventAgentState: EventAgentState
    let modelState: ModelState

    /// Called when the user clicks the settings icon in the HUD.
    var onOpenSettings: (() -> Void)?

    private(set) var panel: SubtitlePanel?
    private var bgObserver: AnyCancellable?

    init(state: OverlayState,
         langConfig: LangConfig,
         glossaryState: GlossaryState,
         eventAgentState: EventAgentState,
         modelState: ModelState) {
        self.state = state
        self.langConfig = langConfig
        self.glossaryState = glossaryState
        self.eventAgentState = eventAgentState
        self.modelState = modelState
    }

    // MARK: - Show / Close

    func show(using config: OverlayConfig) throws {
        let screen = try resolveScreen()
        let rect = makeOverlayRect(config: config, on: screen)

        let panel = makePanel(rect: rect)
        let host = NSHostingView(
            rootView: ContentView(
                state: state,
                langConfig: langConfig,
                glossaryState: glossaryState,
                eventAgent: eventAgentState,
                onOpenSettings: { [weak self] in self?.onOpenSettings?() },
                onHoverChanged: { [weak self] inside in
                    guard let self else { return }
                    // Wrap the visibility change in withAnimation so
                    // HUDView's .transition(.opacity) actually fires
                    // (SwiftUI only animates transitions inside an
                    // animation context). Without this the HUD would
                    // pop in/out instantly, which feels jarring and
                    // also makes the subtitle baseline twitch under
                    // the user's mouse during drag-select.
                    withAnimation(.easeOut(duration: 0.18)) {
                        self.state.isChromeVisible = inside
                    }
                    self.applyChromeVisibility(isVisible: inside)
                }
            )
        )
        host.frame = NSRect(origin: .zero, size: rect.size)
        host.autoresizingMask = [.width, .height]
        panel.contentView = host

        installKeyWindowObservers(for: panel)

        // Drive panel-level background color from state. The NSPanel's
        // backing store paints the area behind the SwiftUI content view,
        // which is precisely what we want — and it stays out of the
        // SwiftUI NSView tree so AppKit stops complaining about
        // "unknown subviews" inserted below the hosting view.
        panel.backgroundColor = NSColor.black.withAlphaComponent(state.bgOpacity)
        bgObserver = state.$bgOpacity
            .receive(on: DispatchQueue.main)
            .sink { [weak panel] value in
                panel?.backgroundColor = NSColor.black.withAlphaComponent(value)
            }

        self.panel = panel
        applyChromeVisibility(isVisible: false)
        panel.orderFrontRegardless()
        panel.makeKey()
    }

    func close() {
        bgObserver?.cancel()
        bgObserver = nil
        panel?.close()
        panel = nil
    }

    // MARK: - Setup helpers

    private func resolveScreen() throws -> NSScreen {
        // Prefer the screen that physically contains the (0, 0) point.
        // That's the built-in laptop display when one is connected;
        // macOS coordinates are global across all screens, and the
        // built-in is always the one that includes the menu bar at
        // the origin. Using NSScreen.main (the one with keyboard
        // focus) here made the panel pop up on the wrong screen when
        // the user had an external 4K display connected — focus started
        // on the laptop, the main screen then shifted to the 4K,
        // and the panel landed off-screen at negative Y because
        // the rect was computed from a different screen's
        // visibleFrame than the one we ended up on.
        let origin = NSPoint.zero
        if let primary = NSScreen.screens.first(where: { $0.frame.contains(origin) }) {
            return primary
        }
        if let main = NSScreen.main { return main }
        if let first = NSScreen.screens.first { return first }
        throw OverlayWindowError.noScreenAvailable
    }

    private func makeOverlayRect(config: OverlayConfig, on screen: NSScreen) -> NSRect {
        let visible = screen.visibleFrame
        // 宽度沿用 --w 百分比（窗体可以很宽，让字幕横排）
        let width = max(visible.width * config.wPct / 100, Palette.minWindowWidth)
        // 高度不取 --h 百分比——开窗时用默认高度（比最小高度大，能同时放下 HUD
        // 与一行 committed 字幕）。用户仍可拖窗口边缘缩小到 `minWindowHeight`。
        let height = max(Palette.minWindowHeight, Palette.defaultWindowHeight)
        let x = visible.origin.x + visible.width * config.xPct / 100
        let y = visible.origin.y + visible.height * config.yPct / 100
        return NSRect(x: x, y: y, width: width, height: height)
    }

    private func makePanel(rect: NSRect) -> SubtitlePanel {
        let panel = SubtitlePanel(
            contentRect: rect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable,
                        .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panel.title = ""
        panel.titlebarAppearsTransparent = true
        panel.titleVisibility = .hidden
        panel.titlebarSeparatorStyle = .none

        panel.isFloatingPanel = true
        panel.tabbingMode = .disallowed
        panel.backgroundColor = NSColor(white: 0, alpha: 0.001)
        panel.isOpaque = false
        panel.hasShadow = true
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.level = .statusBar
        panel.collectionBehavior = [.fullScreenAuxiliary, .ignoresCycle, .canJoinAllSpaces]
        // Don't enable `isMovableByWindowBackground` — it would intercept
        // every mouseDown, including HUD sliders, as a window-drag gesture.
        // The panel's standard title bar (which we hide visually) still
        // accepts Cmd+drag, and users can grab the corner to resize.
        panel.isMovableByWindowBackground = false
        panel.minSize = NSSize(width: Palette.minWidth, height: Palette.minHeight)
        // `panel.minSize` is *advisory* on some macOS 26 resize paths,
        // so install a hard clamp in `windowWillResize`. The clamp
        // returns a corrected size; AppKit keeps the existing
        // origin (i.e. the edge the user grabbed), so there's no
        // y-axis jump.
        panel.resizeClamp = { [weak panel] proposed in
            guard panel != nil else { return proposed }
            let minSize = NSSize(
                width: Palette.minWindowWidth,
                height: Palette.minWindowHeight
            )
            return NSSize(
                width: max(proposed.width, minSize.width),
                height: max(proposed.height, minSize.height)
            )
        }
        // Follow normal macOS window semantics: the red button hides the
        // overlay window, while Cmd+Q / Quit terminates the app and runs
        // backend cleanup from applicationWillTerminate.
        return panel
    }

    private func installKeyWindowObservers(for panel: NSPanel) {
        NotificationCenter.default.addObserver(
            forName: NSWindow.didBecomeKeyNotification,
            object: panel, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                // Same reason as the hover handler: wrap in
                // withAnimation so the HUD's transition fires
                // smoothly instead of popping, which would shift the
                // subtitle baseline by 1-2 pixels and break an
                // in-progress drag-select.
                withAnimation(.easeOut(duration: 0.18)) {
                    self?.state.isWindowActive = true
                }
            }
        }
        NotificationCenter.default.addObserver(
            forName: NSWindow.didResignKeyNotification,
            object: panel, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                withAnimation(.easeOut(duration: 0.18)) {
                    self?.state.isWindowActive = false
                }
            }
        }
        // (No didResize clamp here: `panel.minSize` already blocks the
        // window from being resized below the floor. Re-clamping inside
        // the resize loop caused a y-axis jump because we were forcing
        // the bottom-anchor (`frame.maxY - newSize.height`) on every
        // didResize, which fought with AppKit's own resize logic when
        // the user dragged the top edge. AppKit's enforcement is good
        // enough on macOS 26.)
    }

    private func applyChromeVisibility(isVisible: Bool) {
        // Only the SwiftUI HUD chrome (HUDBar + StatusChips) reacts to
        // hover. System traffic-light buttons keep standard macOS
        // behavior: close hides the overlay, minimize goes to Dock.
    }
}

// MARK: - Errors

enum OverlayWindowError: LocalizedError {
    case noScreenAvailable
    case frameViewUnavailable

    var errorDescription: String? {
        switch self {
        case .noScreenAvailable:    return "No screen available for overlay window."
        case .frameViewUnavailable: return "Failed to access NSWindow frame view."
        }
    }
}
