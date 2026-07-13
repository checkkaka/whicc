# 指定应用音频捕获

whicc 在「全部系统音频 / 麦克风」之外，支持按 **Bundle ID** 只捕获单个 macOS 应用的声音。

## 配置

写入 `/tmp/whicc-out/lang_config.json`：

```json
{
  "audio_source": "application",
  "audio_app_bundle_id": "com.google.Chrome",
  "audio_app_display_name": "Google Chrome"
}
```

- 只持久化 Bundle ID，不存 PID。
- 目标应用退出后保持等待，**不会**自动回退到全部系统音频。
- 设置页「音频来源」可切换；HUD chip 仍只在 system ↔ mic 间循环。

## 链路

1. Swift 设置页用 `NSWorkspace.runningApplications` 列出应用并写配置，然后 SIGHUP `whicc.py`。
2. Python `process_resolver.py` 用 `lsappinfo` + `proc_pidpath` 按 **`.app` 路径前缀** 收集主进程与 Helper（区分 Chrome / Canary / Edge）。
3. `audio.py` 以 `--include-processes <pid…>` 启动 `bin/audiotee`（空格分隔多 PID）。
4. PCM 仍为 16 kHz mono s16le → float32，ASR 路径不变。

## AudioTee 补丁

`patches/audiotee/overlay/` 在 `bin/build_audiotee.sh` 构建时覆盖上游 pin `56ac954`：

- 软跳过尚未产生 Audio Object 的 PID（至少一个有效则继续）。
- 全部无效时 exit code **2**（等待出声；禁止当成「捕获全部」）。
- stdin NDJSON 热重配：`{"cmd":"set_include_processes","pids":[…]}`（同进程重建 tap，缓解 macOS 26 TCC kill+respawn 问题）。

在 macOS 上重新编译并提交二进制：

```bash
./bin/build_audiotee.sh
# 然后 git add bin/audiotee
```

## 日志关键字

- `[AudioSource]` / `[ProcessResolver]` / `[AudioTee]`
- status 事件：`audio_app_waiting`、`audio_app_waiting_audio`、`audio_app_capturing`、`audio_app_receiving_pcm`

## 测试

```bash
python3 -m pip install pytest -q
PYTHONPATH=src python3 -m pytest tests/ -q
```

实机验证矩阵见开发计划：QuickTime / Chrome（多 Helper）/ Safari 隔离、模式来回切、权限撤销、应用退出再开。
