import Foundation

/// Terminates the local whicc backend processes (dev mode).
///
/// BackendLauncher handles termination for packaged .app; this file is
/// the dev-mode counterpart (pkill the 4 python backends the user
/// launched themselves via `swift run` or similar).
///
/// macui is the sole UI, so quitting it should also stop the
/// translation daemon (otherwise the user has to remember to kill it).
///
/// `terminateLocalBackend()` fires `SIGTERM` against the four
/// processes the backend is built from. They're stopped in
/// reverse-startup order so the dependency chain unwinds cleanly:
/// glossary refresher first (it has no inbound consumers), then
/// translation, then ASR, then the audio source.
///
/// Called from both `applicationWillTerminate` (⌘W / ⌘Q menu paths)
/// and the panel's `windowWillClose` (red close button path)
/// so whichever way the user exits, the backend dies with the UI.
enum BackendShutdown {
    /// 给 whicc.py 发 SIGHUP 触发热切换 audio source (system ↔ mic)。
    /// 不杀进程,只是让 whicc.py 重新读 lang_config.json 的
    /// audio_source 键 + 重建 AudioSource。HUD ASR chip 的
    /// onCycleAudioSource 闭包调用。
    ///
    /// pkill 找不到 whicc.py (backend 没启动) 时返回 1,这里忽略。
    @discardableResult
    static func signalWhiccForAudioSwitch() -> Int32 {
        return kill(pattern: "whicc.py", signal: 1)  // SIGHUP
    }

    static func terminateLocalBackend() {
        let patterns = [
            "whicc-audio",
            "glossary_refresher",
            "translate_stream",
            "whicc.py",
        ]
        for pattern in patterns {
            // SIGTERM (-15), not SIGKILL — let the Python side flush
            // its log files and close its model handles cleanly.
            // `pkill` returns 1 when no match, which is fine.
            _ = kill(pattern: pattern, signal: 15)
        }
    }

    @discardableResult
    private static func kill(pattern: String, signal: Int32) -> Int32 {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        task.arguments = ["-\(signal)", "-f", pattern]
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
            task.waitUntilExit()
            return task.terminationStatus
        } catch {
            return -1
        }
    }

    /// 杀 + 重启 translate_stream.py。macui ServerPane 「保存并重启」按钮调。
    /// 行为:
    ///   1. pkill -f translate_stream (SIGTERM,等进程清理日志)
    ///   2. 启动新进程(用 venv python,参数跟 BackendLauncher 同款)
    ///   3. stdout+stderr 写到 /tmp/translate-stream.log
    ///
    /// 返回值: 启动成功 true,失败 false (stderr 写到 log)。
    ///
    /// 路径解析走 AppPaths:开发模式用项目 venv,打包模式用 .app bundle
    /// 内嵌的 venv。Thread.sleep 0.5s 让旧进程有时间 flush 日志 — 会卡
    /// UI 一会,用户量少可接受。
    @MainActor
    @discardableResult
    static func restartTranslateStream() -> Bool {
        // 1. 杀旧进程 (SIGTERM,跟 terminateLocalBackend 一致)
        _ = kill(pattern: "translate_stream", signal: 15)
        // 给 0.5s 让旧进程 flush 日志
        Thread.sleep(forTimeInterval: 0.5)

        // 2. 启动新进程
        // stdout+stderr 写到 translate-stream.log (truncate 模式 — 每次
        // 重启会清空旧 log,这是用户主动行为,期望干净)
        // C6 fix 同款哲学:stdout / stderr 用独立 FileHandle(独立 fd)。
        // Python 进程如果某天引入多线程或异步写,共用 fd 会让 stderr 行
        // 被 stdout 缓冲从中间切开 — 日志可读性灾难。translate_stream.py
        // 现在是单线程顺序写,但兜底用独立 handle 更稳。
        // FileHandle 必须活到 Process.run() 之后 — Process 持有 fd 引用,
        // 直到 Process deinit 才释放 fd。
        let stdoutPath = "/tmp/translate-stream.log"
        let stderrPath = "/tmp/translate-stream.err.log"
        // truncate 旧 logs
        for path in [stdoutPath, stderrPath] {
            do {
                _ = try? Data().write(to: URL(fileURLWithPath: path))
            }
        }
        guard let stdoutHandle = FileHandle(forWritingAtPath: stdoutPath),
              let stderrHandle = FileHandle(forWritingAtPath: stderrPath) else {
            fputs("[BackendShutdown] FAILED to open \(stdoutPath) / \(stderrPath) for log\n", stderr)
            return false
        }
        stdoutHandle.seekToEndOfFile()
        stderrHandle.seekToEndOfFile()

        let task = Process()
        task.executableURL = URL(fileURLWithPath: AppPaths.pythonExecutable)
        // --events 路径必须跟 BackendLauncher 启动 whicc.py 时用的
        // events.jsonl 路径一致 (runDir + "/events.jsonl")。之前这里
        // 写死 "/tmp/whicc-events.jsonl" 是 dev 模式的路径,
        // 打包模式下 BackendLauncher 用 runDir/events.jsonl,导致
        // 用户的"保存并重启翻译服务"按钮拉起的新 translate_stream
        // 监听了一个永远没人写的文件 → 看上去"没生效",实际上必须
        // 重启整个 app (BackendLauncher 才会按 runDir 路径拉起)。
        //
        // 翻译 URL 不传 CLI 参数 — 跟 BackendLauncher 同款源头治理:
        // translate_stream 启动时从 lang_config.json 读用户配的
        // translation_url / translation_fallback_url。
        task.arguments = [
            "\(AppPaths.srcDir)/translate_stream.py",
            "--events", AppPaths.runDir + "/events.jsonl",
            "--out-dir", AppPaths.runDir,
            "--glossary", AppPaths.glossaryPath,
            "--mode", "partial",
            "--target-lang", "auto",
        ]
        task.standardOutput = stdoutHandle
        task.standardError = stderrHandle
        do {
            try task.run()
            // stdoutHandle / stderrHandle 由 Process 持有(strong ref),
            // 直到 Process deinit 才释放 fd。UI 返回 true,不需要立即关。
            return true
        } catch {
            fputs("[BackendShutdown] failed to start translate_stream: \(error)\n", stderr)
            try? stdoutHandle.close()
            try? stderrHandle.close()
            return false
        }
    }
}