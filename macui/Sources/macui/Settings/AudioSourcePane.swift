import AppKit
import SwiftUI

/// 音频来源设置：全部系统音频 / 麦克风 / 指定应用。
///
/// 指定应用只保存 Bundle ID（不存 PID）；写盘后发 SIGHUP 让 whicc.py
/// 热切换。应用列表来自 NSWorkspace.runningApplications。
struct AudioSourcePane: View {
    @ObservedObject var langConfig: LangConfig
    @ObservedObject var overlayState: OverlayState

    @State private var apps: [AppRow] = []
    @State private var statusHint: String = ""

    struct AppRow: Identifiable, Hashable {
        let id: String  // bundle id
        let name: String
        let icon: NSImage?
        let preferred: Bool
    }

    // 业务语义：设置页优先展示的常见媒体/浏览器 Bundle ID。
    private static let preferredBundleIds: Set<String> = [
        "com.google.Chrome",
        "com.google.Chrome.canary",
        "com.microsoft.edgemac",
        "com.apple.Safari",
        "com.apple.QuickTimePlayerX",
        "org.videolan.vlc",
        "com.spotify.client",
        "com.apple.Music",
        "com.hnc.Discord",
        "com.tinyspeck.slackmacgap",
        "us.zoom.xos",
    ]

    var body: some View {
        SettingsDetailContainer {
            VStack(alignment: .leading, spacing: 18) {
                SettingsCard {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "info.circle")
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("音频来源")
                                .font(.system(size: 12, weight: .semibold))
                            Text("指定应用模式只捕获该应用的声音；目标退出后会等待重新启动，不会自动改回全部系统音频。需要「系统音频录制」权限。")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

                modeCard
                if langConfig.audioSource == AudioSource.application.rawValue {
                    appPickerCard
                }
                statusCard
            }
        }
        .onAppear { refreshApps() }
        .onReceive(
            NotificationCenter.default.publisher(
                for: NSWorkspace.didLaunchApplicationNotification)
        ) { _ in refreshApps() }
        .onReceive(
            NotificationCenter.default.publisher(
                for: NSWorkspace.didTerminateApplicationNotification)
        ) { _ in refreshApps() }
    }

    // MARK: - Mode

    private var modeCard: some View {
        SettingsCard {
            SettingsSectionHeader(icon: "waveform", title: "采集模式")
            VStack(alignment: .leading, spacing: 8) {
                modeRow(
                    title: "全部系统音频",
                    subtitle: "捕获本机正在播放的所有声音",
                    selected: langConfig.audioSource == AudioSource.system.rawValue
                ) {
                    applyMode(.system)
                }
                modeRow(
                    title: "麦克风",
                    subtitle: "使用内置或默认输入设备",
                    selected: langConfig.audioSource == AudioSource.mic.rawValue
                ) {
                    applyMode(.mic)
                }
                modeRow(
                    title: "指定应用",
                    subtitle: "只捕获所选应用产生的音频",
                    selected: langConfig.audioSource == AudioSource.application.rawValue
                ) {
                    applyMode(.application)
                }
            }
        }
    }

    private func modeRow(
        title: LocalizedStringKey,
        subtitle: LocalizedStringKey,
        selected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                    .font(.system(size: 14))
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.primary)
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - App picker

    private var appPickerCard: some View {
        SettingsCard {
            SettingsSectionHeader(
                icon: "app.badge",
                title: "目标应用",
                tint: .accentColor,
                trailing: {
                    Button {
                        refreshApps()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .buttonStyle(.plain)
                    .help("刷新应用列表")
                }
            )

            if apps.isEmpty {
                Text("当前没有可选择的运行中应用")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            } else {
                Picker("应用", selection: appSelection) {
                    ForEach(apps) { app in
                        HStack(spacing: 8) {
                            if let icon = app.icon {
                                Image(nsImage: icon)
                                    .resizable()
                                    .frame(width: 16, height: 16)
                            }
                            Text(app.name)
                            Text(app.id)
                                .foregroundStyle(.secondary)
                                .font(.system(size: 10, design: .monospaced))
                        }
                        .tag(app.id)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)

                if !langConfig.audioAppBundleId.isEmpty {
                    Text(langConfig.audioAppBundleId)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var appSelection: Binding<String> {
        Binding(
            get: {
                let current = langConfig.audioAppBundleId
                if apps.contains(where: { $0.id == current }) {
                    return current
                }
                return apps.first?.id ?? ""
            },
            set: { bundleId in
                guard let row = apps.first(where: { $0.id == bundleId }) else { return }
                selectApp(row)
            }
        )
    }

    // MARK: - Status

    private var statusCard: some View {
        SettingsCard {
            SettingsSectionHeader(icon: "info.circle", title: "状态")
            Text(statusText)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if !statusHint.isEmpty {
                Text(statusHint)
                    .font(.system(size: 11))
                    .foregroundStyle(.orange)
            }
        }
    }

    private var statusText: String {
        switch langConfig.audioSource {
        case AudioSource.mic.rawValue:
            return "当前采集：麦克风"
        case AudioSource.application.rawValue:
            let name = langConfig.audioAppDisplayName
            if name.isEmpty {
                return "当前采集：指定应用（尚未选择）"
            }
            let running = apps.contains { $0.id == langConfig.audioAppBundleId }
            if running {
                return "当前采集：\(name)"
            }
            return "等待 \(name) 启动"
        default:
            return "当前采集：全部系统音频"
        }
    }

    // MARK: - Actions

    private func applyMode(_ mode: AudioSource) {
        if mode == .application {
            if langConfig.audioAppBundleId.isEmpty, let first = apps.first {
                langConfig.setAudioApplication(bundleId: first.id, displayName: first.name)
            }
            if langConfig.audioAppBundleId.isEmpty {
                statusHint = "请先选择一个正在运行的应用"
                refreshApps()
                return
            }
        }
        statusHint = ""
        langConfig.setAudioSource(mode.rawValue)
        overlayState.audioSource = mode
        // 调用 BackendShutdown：通知 whicc.py 热切换音频源。
        BackendShutdown.signalWhiccForAudioSwitch()
    }

    private func selectApp(_ row: AppRow) {
        langConfig.setAudioApplication(bundleId: row.id, displayName: row.name)
        if langConfig.audioSource != AudioSource.application.rawValue {
            langConfig.setAudioSource(AudioSource.application.rawValue)
            overlayState.audioSource = .application
        }
        statusHint = ""
        BackendShutdown.signalWhiccForAudioSwitch()
    }

    private func refreshApps() {
        let selfBundle = Bundle.main.bundleIdentifier
        let running = NSWorkspace.shared.runningApplications
        var rows: [AppRow] = []
        var seen = Set<String>()
        for app in running {
            guard app.activationPolicy == .regular,
                  let bid = app.bundleIdentifier,
                  !bid.isEmpty,
                  bid != selfBundle,
                  !seen.contains(bid),
                  app.isTerminated == false
            else { continue }
            seen.insert(bid)
            rows.append(
                AppRow(
                    id: bid,
                    name: app.localizedName ?? bid,
                    icon: app.icon,
                    preferred: Self.preferredBundleIds.contains(bid)
                )
            )
        }
        rows.sort {
            if $0.preferred != $1.preferred { return $0.preferred && !$1.preferred }
            return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
        apps = rows

        // 若当前选中的应用不在列表里，保留 Bundle ID（等待重启），不强行清空。
        if langConfig.audioSource == AudioSource.application.rawValue,
           langConfig.audioAppBundleId.isEmpty,
           let first = rows.first
        {
            langConfig.setAudioApplication(bundleId: first.id, displayName: first.name)
        }
    }
}
