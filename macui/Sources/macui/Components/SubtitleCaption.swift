import SwiftUI

/// The committed subtitle — the most recent finalized line.
///
/// Honors `bilingualLayout` so the user can flip between translation-top
/// and source-top without restarting. When `showSource` is off, the
/// secondary line is dropped entirely.
///
/// Conforms to `Equatable` so SwiftUI can skip re-renders when none
/// of the inputs change. This is critical for drag-select stability
/// — without it, every unrelated state change (window focus, hover,
/// bg slider) would re-flow the layout and nudge the text baseline
/// 1-2px under the user's mouse.
struct SubtitleCaption: View, Equatable {
    let caption: OverlayCaption
    /// 已 resolved 的颜色——`state.resolvedAccent`，调用方负责在
    /// `style == .custom` 时把 `customColor` 传进来。组件本身不感知
    /// OverlayStyle，避免每个组件都重复自定义色的 fallback 逻辑。
    let accent: Color
    let showSource: Bool
    let srcFontSize: CGFloat
    let transFontSize: CGFloat
    let bilingualLayout: BilingualLayout
    let fontChoice: SubtitleFont
    let shadowStrong: Double
    let shadowSoft: Double
    let shadowStrongRadius: CGFloat
    let shadowSoftRadius: CGFloat

    static func == (lhs: SubtitleCaption, rhs: SubtitleCaption) -> Bool {
        lhs.caption == rhs.caption
            && lhs.accent == rhs.accent
            && lhs.showSource == rhs.showSource
            && lhs.srcFontSize == rhs.srcFontSize
            && lhs.transFontSize == rhs.transFontSize
            && lhs.bilingualLayout == rhs.bilingualLayout
            && lhs.fontChoice == rhs.fontChoice
            && lhs.shadowStrong == rhs.shadowStrong
            && lhs.shadowSoft == rhs.shadowSoft
            && lhs.shadowStrongRadius == rhs.shadowStrongRadius
            && lhs.shadowSoftRadius == rhs.shadowSoftRadius
    }

    var body: some View {
        BilingualStack(layout: bilingualLayout) {
            if !caption.translatedText.isEmpty {
                SubtitleText(
                    caption.translatedText,
                    font: fontChoice.font(size: transFontSize, weight: .bold),
                    color: primaryColor,
                    lineLimit: 4,
                    shadowStrong: shadowStrong,
                    shadowSoft: shadowSoft,
                    shadowStrongRadius: shadowStrongRadius,
                    shadowSoftRadius: shadowSoftRadius
                )
            }
        } source: {
            if showSource, !caption.sourceText.isEmpty {
                SubtitleText(
                    caption.sourceText,
                    font: fontChoice.font(size: srcFontSize, weight: .regular),
                    color: sourceColor,
                    lineLimit: 3,
                    shadowStrong: shadowStrong,
                    shadowSoft: shadowSoft,
                    shadowStrongRadius: shadowStrongRadius,
                    shadowSoftRadius: shadowSoftRadius
                )
            }
        }
        .padding(.horizontal, Palette.subtitleHPadding)
        .padding(.top, 0)
        .transition(.asymmetric(
            insertion: .opacity.combined(with: .move(edge: .bottom)),
            removal: .opacity
                .combined(with: .move(edge: .top))
                .combined(with: .scale(scale: 0.72))
        ))
    }

    // MARK: Color policy
//
// The slider controls background opacity only; it must never change
// the text color. The committed-caption text is the user-picked
// accent. A second, lighter line is drawn for the source.

    private var primaryColor: Color {
        accent.opacity(0.96)
    }
    private var sourceColor: Color {
        accent.opacity(0.58)
    }
}

// MARK: - Reusable subtitle line

/// One rendered subtitle line. The caller hands it a concrete
/// `Font` (built from `SubtitleFont`) so this view never has to
/// know which typeface the user picked.
struct SubtitleText: View {
    let text: String
    let font: Font
    let color: Color
    let lineLimit: Int
    let shadowStrong: Double
    let shadowSoft: Double
    let shadowStrongRadius: CGFloat
    let shadowSoftRadius: CGFloat

    init(_ text: String,
         font: Font,
         color: Color,
         lineLimit: Int,
         shadowStrong: Double,
         shadowSoft: Double,
         shadowStrongRadius: CGFloat = Palette.textShadowRadius,
         shadowSoftRadius: CGFloat = Palette.textShadowSoftRadius) {
        self.text = text
        self.font = font
        self.color = color
        self.lineLimit = lineLimit
        self.shadowStrong = shadowStrong
        self.shadowSoft = shadowSoft
        self.shadowStrongRadius = shadowStrongRadius
        self.shadowSoftRadius = shadowSoftRadius
    }

    var body: some View {
        Text(text)
            .font(font)
            .foregroundColor(color)
            .multilineTextAlignment(.center)
            .lineLimit(lineLimit)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity)
            .textSelection(.enabled)
            .shadow(color: .black.opacity(shadowStrong),
                    radius: shadowStrongRadius, x: 0, y: 6)
            .shadow(color: .black.opacity(shadowSoft),
                    radius: shadowSoftRadius, x: 0, y: 1)
    }
}
