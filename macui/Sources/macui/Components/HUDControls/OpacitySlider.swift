import SwiftUI

/// Background-opacity slider with the standard SF Symbol.
struct OpacitySlider: View {
    @ObservedObject var state: OverlayState

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: "circle.lefthalf.filled")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(Palette.textSecondary)
            Slider(value: $state.bgOpacity, in: 0.05...1.0)
                .controlSize(.mini)
                .tint(Palette.textPrimary)
                .frame(width: 72)
        }
        .help("字幕窗口背景透明度")
        .hudControl()
    }
}
