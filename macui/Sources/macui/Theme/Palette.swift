import SwiftUI

/// Centralized tokens for layout, spacing, radius, and shadow.
enum Palette {
    // Window
    // minWidth 500 = HUD 7 个控件 (~480pt) + 各 10pt 留白。
    static let minWindowWidth: CGFloat = 500
    // minHeight 68 = 1 行 committed 字幕 + 一行高度。
    //   HUD 和 draft 在此高度自动隐藏,窗口拉高时回来。
    static let minWindowHeight: CGFloat = 68
    // defaultHeight 240 = HUD + 上方稳定字幕 + 下方实时草稿均可读。
    //   用户仍能拖到 minWindowHeight，此时优先保留稳定字幕。
    static let defaultWindowHeight: CGFloat = 240

    // Backwards-compat aliases used in a few spots.
    static let minWidth: CGFloat = minWindowWidth
    static let minHeight: CGFloat = minWindowHeight

    // Padding
    static let hudHPadding: CGFloat = 12
    static let hudVPadding: CGFloat = 8
    static let subtitleHPadding: CGFloat = 16
    static let subtitleVPadding: CGFloat = 6

    // Radius
    static let hudCorner: CGFloat = 14
    static let controlCorner: CGFloat = 8

    // HUD control sizing
    static let controlHeight: CGFloat = 22

    // Stage
    // 显示 history 的最小 contentH。SubtitleStageView 还会在有实时
    // draft 时隐藏 history，确保 final + draft 优先可读。
    static let historyMinVisible: CGFloat = 160
    // 显示 draft 的最小 contentH(独立于 minWindowHeight)。
    //   70pt → 只要 final 下方还有一两行空间，就显示实时草稿。
    static let draftMinVisible: CGFloat = 70
    static let sourceMinVisible: CGFloat = 62
    static let historyMaxHeight: CGFloat = 1200

    // History row opacity decay
    static let historyBaseOpacity: Double = 0.45
    static let historyOpacityStep: Double = 0.06
    static let historyMinOpacity: Double = 0.15

    // Shadow (for subtitle text legibility)
    static let textShadowRadius: CGFloat = 16
    static let textShadowSoftRadius: CGFloat = 4

    // Glass overlay tints
    static let plateBorder: Color = .white.opacity(0.12)
    static let plateInner: Color = .white.opacity(0.05)
    static let plateTint: Color = .black.opacity(0.18)
    static let plateShadow: Color = .black.opacity(0.30)

    // HUD text colors (always explicit white-tinted for guaranteed contrast)
    static let textPrimary: Color = .white.opacity(0.96)
    static let textSecondary: Color = .white.opacity(0.72)
    static let textTertiary: Color = .white.opacity(0.48)

    // Control fill (frosted background)
    static let controlFill: Color = .white.opacity(0.08)
    static let controlFillSubtle: Color = .white.opacity(0.04)
    static let controlSelected: Color = .white.opacity(0.18)
}

// MARK: - Subtitle text coloring policy
//
// The slider only controls background opacity. The subtitle text color
// is always the user-picked accent (or white when the accent theme is
// `.theater`). The background is what changes, not the text — that
// keeps the user's color choice stable while they tune see-through.
//
// The accent-vs-white decision is based on the `OverlayStyle` (the
// user explicitly chose a white theme), not on the slider position.

extension OverlayState {
    /// `true` when the background is mostly transparent — used as a
    /// legacy hint for code that still wants to bump shadow opacity
    /// based on background transparency. The new shadow opacity is
    /// user-controlled via Settings (see `strongShadowOpacity` /
    /// `softShadowOpacity` properties), so this property is now only
    /// used in fallback paths.
    var hasLowBackground: Bool { bgOpacity < 0.22 }
}
