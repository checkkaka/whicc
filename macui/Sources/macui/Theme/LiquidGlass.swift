import SwiftUI

// MARK: - Glass plate shell
//
// A self-contained view modifier that produces the macOS 26 Liquid Glass
// "floating plate" look. The plate is the chrome that hosts HUD
// controls. On macOS 26+ we use `GlassEffectContainer` to coalesce
// neighbouring glass shapes for performance and morphing; on macOS 15
// the same `ultraThinMaterial` + tint stack renders directly (frosted
// material, no liquid morphing).

struct GlassPlate: ViewModifier {
    var corner: CGFloat = Palette.hudCorner

    private var plateFill: some View {
        ZStack {
            RoundedRectangle(cornerRadius: corner, style: .continuous)
                .fill(.ultraThinMaterial)
            RoundedRectangle(cornerRadius: corner, style: .continuous)
                .fill(Palette.plateTint)
        }
    }

    func body(content: Content) -> some View {
        content
            .background {
                if #available(macOS 26.0, *) {
                    GlassEffectContainer(spacing: 0) {
                        plateFill
                    }
                } else {
                    plateFill
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: corner, style: .continuous)
                    .stroke(Palette.plateBorder, lineWidth: 0.8)
            }
            .overlay {
                RoundedRectangle(cornerRadius: corner - 1, style: .continuous)
                    .inset(by: 1)
                    .stroke(Palette.plateInner, lineWidth: 0.5)
            }
            .shadow(color: Palette.plateShadow, radius: 18, x: 0, y: 8)
    }
}

extension View {
    /// Apply the standard Liquid Glass chrome to a HUD container.
    func glassPlate(corner: CGFloat = Palette.hudCorner) -> some View {
        modifier(GlassPlate(corner: corner))
    }
}

// MARK: - Control chip
//
// Capsule background used by every interactive control inside the HUD.
// `interactive(true)` lets the user click through the glass.

struct HUDControlBackground: ViewModifier {
    var selected: Bool = false

    func body(content: Content) -> some View {
        content
            .padding(.horizontal, 6)
            .frame(height: Palette.controlHeight)
            .background {
                // 共享 HUD plate 的 frosted glass。未选态用 4% 半透明白作
                // 微背景(给 chip 视觉边界,hover 才看得到这是按钮);
                // 选中态 18% 半透明白作选中 tint。早期 8% 半透明白在 HUD
                // glass 上太重,看起来像 chip 自己有不透明背景,反而盖住
                // 了 Liquid Glass 的 frosted 效果。
                Capsule(style: .continuous)
                    .fill(selected
                          ? Palette.controlSelected
                          : Palette.controlFillSubtle)
            }
            .contentShape(Capsule())
    }
}

extension View {
    func hudControl(selected: Bool = false) -> some View {
        modifier(HUDControlBackground(selected: selected))
    }
}
