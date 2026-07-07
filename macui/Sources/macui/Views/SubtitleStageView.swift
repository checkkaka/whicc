import SwiftUI

/// Owns the vertical layout of the subtitle region:
///
///     ┌──────────────────────────────┐
///     │      history (scrollable)    │  ← optional
///     │                              │
///     │     committed caption        │  ← large
///     │     draft caption (stream)   │  ← smaller
///     └──────────────────────────────┘
///
/// Honors `Palette.historyMinVisible` / `Palette.sourceMinVisible` so
/// the layout adapts gracefully to short windows.
///
/// `Equatable` + the `.equatable()` modifier below mean this view
/// only re-renders when one of the subtitle fields actually
/// changes. `OverlayState` has ~20 `@Published` fields; without
/// this gate, toggling `isWindowActive` or `isChromeVisible` would
/// trigger a full layout pass on the subtitle stage and nudge the
/// baseline 1-2px under the user's mouse during drag-select.
struct SubtitleStageView: View, Equatable {
    let committed: OverlayCaption?
    let draftSourceText: String?
    let draftTranslatedText: String?
    let draftStablePrefixLen: Int
    let history: [OverlayCaption]
    let showSource: Bool
    let showHistory: Bool
    let bilingualLayout: BilingualLayout
    let transFontSize: CGFloat
    let srcFontSize: CGFloat
    /// 已 resolved 的颜色——`state.resolvedAccent`，调用方负责在
    /// `style == .custom` 时把 `customColor` 传进来。
    let accent: Color
    let fontChoice: SubtitleFont
    let shadowStrong: Double
    let shadowSoft: Double
    let shadowStrongRadius: CGFloat
    let shadowSoftRadius: CGFloat
    /// 用户在 Settings 调节外观时 = true，字幕区显示项目简介作为视觉
    /// 参考；其他时候字幕区干净（即使没真实字幕也不显示占位）。
    let showIdlePreview: Bool
    /// Top inset reserved for the HUD plate (measured upstream). The
    /// stage leaves this region empty so subtitle glyphs don't slide
    /// under the HUD.
    let hudHeight: CGFloat

    static func == (lhs: SubtitleStageView, rhs: SubtitleStageView) -> Bool {
        lhs.committed?.id == rhs.committed?.id
            && lhs.committed?.sourceText == rhs.committed?.sourceText
            && lhs.committed?.translatedText == rhs.committed?.translatedText
            && lhs.draftSourceText == rhs.draftSourceText
            && lhs.draftTranslatedText == rhs.draftTranslatedText
            && lhs.draftStablePrefixLen == rhs.draftStablePrefixLen
            && lhs.history.map(\.id) == rhs.history.map(\.id)
            && lhs.showSource == rhs.showSource
            && lhs.showHistory == rhs.showHistory
            && lhs.bilingualLayout == rhs.bilingualLayout
            && lhs.transFontSize == rhs.transFontSize
            && lhs.srcFontSize == rhs.srcFontSize
            && lhs.accent == rhs.accent
            && lhs.fontChoice == rhs.fontChoice
            && lhs.shadowStrong == rhs.shadowStrong
            && lhs.shadowSoft == rhs.shadowSoft
            && lhs.shadowStrongRadius == rhs.shadowStrongRadius
            && lhs.shadowSoftRadius == rhs.shadowSoftRadius
            && lhs.showIdlePreview == rhs.showIdlePreview
            && lhs.hudHeight == rhs.hudHeight
    }

    var body: some View {
        GeometryReader { geo in
            let availableH = geo.size.height
            let contentH = max(0, availableH - hudHeight)
            let showHistoryNow = showHistory
                && !history.isEmpty
                && contentH >= Palette.historyMinVisible
            let historyMaxH: CGFloat = showHistoryNow
                ? min(max((contentH - 140) * 1, 40), Palette.historyMaxHeight)
                : 0

            VStack(spacing: 0) {
                // HUD plate sits on top of the stage (see ContentView's
                // .overlay(alignment: .top)). Reserve its measured
                // height so subtitle glyphs never slide underneath.
                Color.clear.frame(height: hudHeight)

                if showHistoryNow {
                    HistorySection(
                        history: history,
                        accent: accent,
                        maxHeight: historyMaxH,
                        bilingualLayout: bilingualLayout,
                        showSource: showSource,
                        srcFontSize: srcFontSize,
                        transFontSize: transFontSize,
                        fontChoice: fontChoice
                    )
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                subtitleStage(availableH: availableH)
                    .padding(.horizontal, Palette.subtitleHPadding)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
    }

    @ViewBuilder
    private func subtitleStage(availableH: CGFloat) -> some View {
        // contentH = availableH - hudHeight。hideDraft 决策:
        //   < draftMinVisible → 隐藏 draft(避免刚露 1/4 行的"半截"状态)
        //   ≥ draftMinVisible → 显示 draft + committed,各自 fit-content
        //     (DraftSubtitle 内 lineLimit 截断,SwiftUI VStack 自然堆叠)
        // 之前用 committedHeight upper-bound (~236pt) 减 contentH 算
        // draftHeight,导致 contentH < 236 时 draft 永远不显示 — 已改。
        let contentH = max(0, availableH - hudHeight)
        let hideDraft = contentH < Palette.draftMinVisible
        VStack(spacing: 0) {
            // 没正式字幕时显示占位字幕——作为"参考信息"卡让用户看到
            // 当前外观设置 (字体/字号/颜色/阴影) 的实时效果。
            // 仅当 `showIdlePreview == true` 时显示（用户在 Settings 调
            // 节外观 5 秒内），平时字幕区干净。
            let displayedCaption: OverlayCaption? = committed
                ?? (showIdlePreview ? Self.idlePreviewCaption() : nil)
            if let cap = displayedCaption {
                SubtitleCaption(
                    caption: cap,
                    accent: accent,
                    showSource: showSource,
                    srcFontSize: srcFontSize,
                    transFontSize: transFontSize,
                    bilingualLayout: bilingualLayout,
                    fontChoice: fontChoice,
                    shadowStrong: shadowStrong,
                    shadowSoft: shadowSoft,
                    shadowStrongRadius: shadowStrongRadius,
                    shadowSoftRadius: shadowSoftRadius
                )
                    .frame(maxWidth: .infinity, alignment: .top)
                    .id(cap.id)
            }
            if !hideDraft {
                // 不给 draft 固定/最大 frame——让 DraftSubtitle 自己按内容
                // 算 fit-content 大小（BilingualStack + lineLimit(2/3) 自然
                // 截断，不会撑爆窗口）。之前两种尝试都有问题：
                //   - .frame(height:) 给固定高度，BilingualStack 默认 .center
                //     alignment 让 draft 内容在大框里上下漂移（窗口越大越靠下）
                //   - .frame(maxHeight:, alignment: .top) 让 BilingualStack 两个
                //     child 互相挤压，超 maxHeight 时后插入的 child 被压成 0 高
                //     消失（用户报告：y=1 显示译文时 draft 原文高度变 y=0）
                // fit-content 是 HIG 推荐方式：subtitles 自己按 lineLimit 算高度，
                // 父 VStack 自然堆叠不挤压。
                draftSlot
            }
        }
        .frame(maxWidth: .infinity, alignment: .top)
    }

    @ViewBuilder
    private var draftSlot: some View {
        let hasSource = (draftSourceText?.isEmpty == false)
            && draftSourceText != committed?.sourceText
        let hasTrans = (draftTranslatedText?.isEmpty == false)
        if hasSource || hasTrans {
            DraftSubtitle(
                source: draftSourceText ?? "",
                translated: draftTranslatedText,
                stablePrefixLen: draftStablePrefixLen,
                srcFontSize: srcFontSize,
                bilingualLayout: bilingualLayout,
                showSource: showSource,
                accent: accent,
                fontChoice: fontChoice
            )
        } else {
            Color.clear
        }
    }

    /// 没正式字幕时显示的占位字幕。中文译位 + 英文源位（跟正常双语
    /// 字幕布局一致），让用户看到当前外观 (字体/字号/颜色/阴影) 的
    /// 实时效果——作为"调节外观时的视觉参考"。
    ///
    /// Source key 用 `idle-preview` 避开真实字幕 ID 命名空间，OverlayState
    /// 收到 init- 这种 key 已经过滤了，但 idle- 也加一道保险：
    /// history 写入路径不会跑这条（committed 是 nil）。
    ///
    /// 内容是项目简介，跟用户 GitHub 信息呼应——让没字幕的 idle 状态
    /// 也传递"在做什么"的信息。
    static func idlePreviewCaption() -> OverlayCaption {
        OverlayCaption(
            id: "idle-preview",
            sourceText: "@Cyberteng https://github.com/nbzz/whicc Transcribe all languages and Chinese dialects and translate to any language",
            translatedText: "@Cyberteng https://github.com/nbzz/whicc 识别全球语言，翻译成各国语言",
            mode: "reset_full",
            translateMs: 0
        )
    }
}