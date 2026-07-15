import AppKit

/// Floating panel that can become key (so it receives hover/keyboard)
/// but does not steal main-window status from the app behind it.
///
/// Acts as its own `NSWindowDelegate` so the resize hook is wired up
/// automatically. `OverlayWindowController` installs a
/// `resizeClamp` closure that the delegate forwards to, which gives
/// us a clean place to enforce the minimum frame size during a
/// user-driven resize (without falling out of AppKit's own anchor
/// logic, which is what was causing the y-axis jump with the old
/// `didResizeNotification` approach).
///
/// The red close button follows standard macOS semantics: it hides the
/// overlay window. Backend cleanup belongs to the explicit app quit path
/// (`Cmd+Q` / Quit menu), where `applicationWillTerminate` fires.
final class SubtitlePanel: NSPanel, NSWindowDelegate {
    /// Closure invoked on every user-driven resize *before* AppKit
    /// applies the new size. Return a size to clamp it; return
    /// `proposedFrameSize` unchanged to let the resize go through.
    var resizeClamp: ((_ proposedFrameSize: NSSize) -> NSSize)?

    /// Optional hook invoked right before the panel is closed.
    var onWillClose: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    override func becomeKey() {
        super.becomeKey()
        // SwiftUI bridges `Button` to NSButton, which accepts first
        // responder by default. When the panel becomes key — either
        // at launch (via `makeKey()`) or after the user clicks
        // elsewhere and back — AppKit would otherwise route keyboard
        // focus to the first focusable Button it finds, which happens
        // to be the `-` button in the font-size stepper. That gives
        // the user a phantom focus ring on a control they didn't ask
        // for, and routes Space/Return presses to the wrong control.
        //
        // Clear first responder. `nil` is a legal value for an
        // NSPanel — it means "no control is selected" — and no
        // focus ring is drawn. The user can still click any HUD
        // control to focus it explicitly.
        if firstResponder != nil {
            makeFirstResponder(nil)
        }
    }

    override init(contentRect: NSRect,
                  styleMask style: NSWindow.StyleMask,
                  backing bufferingType: NSWindow.BackingStoreType,
                  defer flag: Bool) {
        super.init(contentRect: contentRect,
                   styleMask: style,
                   backing: bufferingType,
                   defer: flag)
        delegate = self
    }

    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        resizeClamp?(frameSize) ?? frameSize
    }

    func windowWillClose(_ notification: Notification) {
        onWillClose?()
    }
}
