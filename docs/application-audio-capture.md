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

## 并发模型（audio.py）

- **supervisor 线程是唯一 spawner**：`_spawn_shared_unlocked` 幂等（同过滤签名直接复用；不同签名先 kill、睡 1s 让 TCC 回收再 spawn），任何时刻至多一个 audiotee。
- **_watch_pids 线程不 spawn**：只做目标退出 kill、PID 变化防抖 + stdin 热重配；热重配不可用时 kill 交给 supervisor respawn。
- supervisor kill 前做**所有权校验**（`_shared_proc is proc`），不误杀其他线程刚起的新进程。
- 目标未出声（exit 2 / 无 Audio Object）抛 `AudioteeWaitingError` → 等待重试，不计入失败、不回退全系统音频；whicc 主循环对等待态豁免 12s 无数据看门狗。
- SIGHUP swap 用非阻塞锁，避免 handler 在主线程自我死锁；同 mode 同 bundle 早退，但 source 进入 failed 态时放行重建。

## AudioTee 补丁

`patches/audiotee/overlay/` 在 `bin/build_audiotee.sh` 构建时覆盖上游 pin `56ac954`：

- 软跳过尚未产生 Audio Object 的 PID（至少一个有效则继续）。
- 全部无效时 exit code **2**（等待出声；禁止当成「捕获全部」）。
- stdin NDJSON 热重配：`{"cmd":"set_include_processes","pids":[…]}`（同进程重建 tap，缓解 macOS 26 TCC kill+respawn 问题）；空 PID 列表被拒绝，绝不扩大为全系统捕获。
- 每条命令在 stderr 回 `{"message_type":"reconfigure","status":"ok|waiting|error",…}`；Python 侧必须等到 ack 才认定成功——旧版未打补丁的二进制不回 ack，2 秒超时后自动永久回退 kill+respawn 路径。

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
