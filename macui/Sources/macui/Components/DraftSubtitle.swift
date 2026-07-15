import SwiftUI

/// The live ASR/translation draft that streams underneath (or above) the
/// committed caption. Python 的 SSE 累计全文直接驱动刷新，SwiftUI 不再
/// 二次逐字播放，避免本地动画与后端 revision 互相追赶。
///
/// `Equatable` so SwiftUI can skip re-renders when none of the
/// inputs change (same drag-select rationale as `SubtitleCaption`).
struct DraftSubtitle: View, Equatable {
    let source: String
    let translated: String?
    let stablePrefixLen: Int
    let srcFontSize: CGFloat
    let transFontSize: CGFloat
    let bilingualLayout: BilingualLayout
    let showSource: Bool
    /// 已 resolved 的颜色——同 `SubtitleCaption.accent`，由调用方从
    /// `state.resolvedAccent` 传下来。
    let accent: Color
    let fontChoice: SubtitleFont

    static func == (lhs: DraftSubtitle, rhs: DraftSubtitle) -> Bool {
        // 输入直接驱动 body；完整比较可让 `.equatable()` 安全跳过 HUD 等
        // 无关状态导致的重复布局。
        lhs.source == rhs.source
            && lhs.translated == rhs.translated
            && lhs.stablePrefixLen == rhs.stablePrefixLen
            && lhs.srcFontSize == rhs.srcFontSize
            && lhs.transFontSize == rhs.transFontSize
            && lhs.bilingualLayout == rhs.bilingualLayout
            && lhs.showSource == rhs.showSource
            && lhs.accent == rhs.accent
            && lhs.fontChoice == rhs.fontChoice
    }

    var body: some View {
        // BilingualStack spacing 用 0：draft 双语两行(译 + 源)之间不留
        // 间距，避免"draft 译文出现把 draft 原文往下推一截"的插队感。
        //
        // 关键：两个 child 都**总是**渲染（用 Color.clear 占位 if 内容空），
        // 不让 VStack 的 child 数量从 1 变 2。BilingualStack 内部 VStack
        // child 数变化会改变整体高度 → draft slot 高度变 → 父 VStack 里
        // draft 整体下推（用户报告的"插队"）。固定 child 数 = 固定高度。
        BilingualStack(layout: bilingualLayout, spacing: 0) {
            if let trans = translated, !trans.isEmpty {
                SubtitleText(
                    trans,
                    font: fontChoice.font(size: max(srcFontSize, transFontSize * 0.75), weight: .semibold),
                    color: draftTransColor,
                    lineLimit: 3,
                    shadowStrong: 0.40,
                    shadowSoft: 0.40
                )
            } else {
                // 译文尚未到达时,占位 1 行 line height,保持 BilingualStack
                // 内部 child 数量恒定,draft 整体高度不变。
                Color.clear.frame(height: max(max(srcFontSize, transFontSize * 0.75) * 1.18, 18))
            }
        } source: {
            // Match the committed-caption policy: if the user toggled
            // source visibility off, the draft's source row is hidden too.
            if showSource, !source.isEmpty {
                draftSourceRow(displaySource: source)
            } else if showSource {
                // 源被显示但内容空时占位——保持 BilingualStack child 数量恒定。
                Color.clear.frame(height: max(srcFontSize * 0.92 * 1.18, 18))
            }
            // showSource == false: BilingualStack 内部 source 闭包返回
            // EmptyView,实际只 1 个 child(译),整体高度 = 1 行——这是
            // 跟 committed 行为对齐(committed 也只在 showSource 时显示
            // 源行)。
        }
        .padding(.horizontal, Palette.subtitleHPadding)
        .padding(.bottom, Palette.subtitleVPadding)
    }

    // MARK: Pieces

    @ViewBuilder
    private func draftSourceRow(displaySource: String) -> some View {
        // macOS 26 弃用 `Text + Text`,改用 AttributedString 分段染色。
        let attr = makeAttributedSource(displaySource: displaySource)
        Text(attr)
            .multilineTextAlignment(.center)
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .bottom)
            .shadow(color: .black.opacity(0.40),
                    radius: 6, x: 0, y: 2)
    }

    /// 拆 helper 是因为 @ViewBuilder 函数体内不能有 var/赋值副作用。
    private func makeAttributedSource(displaySource: String) -> AttributedString {
        let stable = String(displaySource.prefix(min(stablePrefixLen, displaySource.count)))
        let mutable = String(displaySource.dropFirst(stable.count))
        var attr = AttributedString()
        var stableRun = AttributedString(stable)
        stableRun.foregroundColor = draftStableColor
        attr.append(stableRun)
        var mutableRun = AttributedString(mutable)
        mutableRun.foregroundColor = draftMutableColor
        attr.append(mutableRun)
        attr.font = fontChoice.font(size: srcFontSize * 0.92, weight: .regular)
        return attr
    }

    // MARK: Color policy
//
// Draft text always uses the chosen accent. The background-opacity
// slider does not affect text color.

    private var draftStableColor: Color {
        accent.opacity(0.72)
    }
    private var draftMutableColor: Color {
        accent.opacity(0.50)
    }
    private var draftTransColor: Color {
        accent.opacity(0.66)
    }

}
