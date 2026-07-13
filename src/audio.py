"""Audio sources: microphone (sounddevice) and system audio (audiotee subprocess).

Both sources behave the same way: capture in the background, push float32 [-1, 1]
mono chunks into self.queue, and put a SENTINEL(None) when the stream ends. The
downstream ASR loop (whicc.py) consumes via queue.get() directly — in-memory,
no disk round-trip.

设计参考 livecaption (six-ddc/livecaption) 的 audio.py——单一 Python 进程
内部多线程,音频采集跟 ASR 通过内存 queue.Queue 解耦,不再依赖外部 Swift 二进制
长期驻守。

历史:早期版本经 SegDirWriter 把 queue 数据写成 /tmp/whicc-seg 段文件,再由
主循环轮询读回(0.15s 轮询 + 1s 段聚合的额外延迟,且 /tmp 被系统清理会断链)。
现在 live 模式(system/mic)主循环直接消费 source.queue;SEG_DIR 文件协议仅保留
为 whicc.py --audio-source segdir 的离线评估入口(tools/whicc_file_audio.py 投喂)。
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from config import SAMPLE_RATE, SYSTEM_AUDIO_STALL_SEC

SENTINEL = None  # putting this on the queue signals the audio stream has ended


class AudioteeWaitingError(RuntimeError):
    """application 模式:目标应用未出声(audiotee 无法建 tap),等待重试而非失败。"""


def build_audiotee_cmd(audiotee_path: str, include_pids: list[int] | None = None) -> list[str]:
    """构造 audiotee 启动命令。

    AudioTee 数组参数是「一个 flag + 空格分隔多个值」,不能重复传 flag。
    """
    cmd = [audiotee_path, "--sample-rate", str(SAMPLE_RATE)]
    if include_pids:
        cmd += ["--include-processes", *[str(p) for p in include_pids]]
    return cmd


class AudioSource(ABC):
    """Audio 源基类：后台采集,float32 [-1,1] mono chunks 进 self.queue。

    子类实现 start() / stop()——具体的麦克风 vs 系统声音采集。whicc.py 主线程
    从 source.queue.get() 读 chunks,不知道也不关心 source 是 sounddevice 还是
    audiotee 子进程。
    """

    def __init__(self, label: str):
        self.label = label
        # maxsize caps memory; 队列满了丢最旧的 live frame,不阻塞采集侧
        self.queue: queue.Queue = queue.Queue(maxsize=200)
        self._stop = threading.Event()

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def _offer(self, samples: np.ndarray) -> None:
        """Enqueue a live frame: 队列满了丢最旧,不阻塞采集侧。"""
        while True:
            try:
                self.queue.put_nowait(samples)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self.queue.get_nowait()

    def _put_sentinel(self) -> None:
        """Enqueue SENTINEL, 队列满时丢最旧腾位。"""
        while True:
            try:
                self.queue.put_nowait(SENTINEL)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self.queue.get_nowait()


class MicSource(AudioSource):
    """麦克风源:sounddevice 库回调,纯 Python,无外部二进制。

    sounddevice 在 PortAudio 音频线程上跑回调,只 enqueue,不做处理。
    block_ms 控制 latency,默认 100ms = 1600 samples @ 16kHz。
    """

    def __init__(self, label: str = "mic", device: int | str | None = None, block_ms: int = 100):
        super().__init__(label)
        self.device = device
        self.blocksize = int(SAMPLE_RATE * block_ms / 1000)
        self._stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "sounddevice 没装。装: pip install sounddevice\n"
                "macOS 上 PortAudio 跟 sounddevice 一起装:"
                " brew install portaudio && pip install sounddevice"
            )

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if self._stop.is_set():
                return
            # indata shape = (frames, channels), 我们只要 mono
            self._offer(indata[:, 0].copy().astype(np.float32))

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        print(f"[audio] MicSource started: device={self.device or 'default'}, "
              f"block_ms={1000 * self.blocksize // SAMPLE_RATE}", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._put_sentinel()
        print(f"[audio] MicSource stopped", flush=True)


class SystemAudioSource(AudioSource):
    """系统音频源:audiotee 子进程(stdout 给原始 PCM,stderr 给 NDJSON 状态)。

    audiotee 是 makeusabrew/audiotee 的 Swift 项目,Core Audio process tap。
    --sample-rate 16000 时输出固定 s16le mono。sounddevice 没法抓系统声音,
    必须走 audiotee 这种二进制方式——但 audiotee 是 build-once-run-anywhere,
    放 ./bin/audiotee 而不是 /tmp/ 就能避免 macOS 清理问题。

    supervisor 线程做断流看门狗:健康的 tap 即使静音也持续输出零字节流,
    5+ 秒完全无数据 = tap 已死(实测诱因:切换默认输出设备,tap 还挂在旧设备
    上 IO 停转),此时杀掉重启 audiotee 重新 tap 当前设备。

    Audiotee 子进程管理 (跨 swap 共享): stop() 不杀子进程,只让 _pump
    退出读循环。下次 start() 复用同一个 Popen 实例 + 同一个 stdout
    pipe。原因:macOS 26 的 Core Audio process tap 注册跟进程 PID 绑定;
    kill+respawn 时新进程被 TCC 静默拒绝授权,返回 0 字节流 → audio
    ~8s 静音警告 + 30s stall 杀 whicc.py。保留同一 audiotee 子进程让
    Core Audio tap 保持注册,swap 时只换 _pump 读循环。

    application 模式 (bundle_id 非空):
      - 按 Bundle ID 解析相关 PID (含 Chromium Helper);
      - audiotee 启动参数 --include-processes <pid…>;
      - 目标未运行 / 尚未出声 (exit 2) 时等待重试,不回退系统音频;
      - PID 集合变化时优先 stdin NDJSON 热重配 tap,失败再 kill+respawn;
      - 与「全部系统音频」模式不共享同一 audiotee 实例 (过滤集不同)。
    """

    # ── 模块级共享: 跨 SystemAudioSource 实例 + 跨 SIGHUP swap 复用 ──
    _shared_proc: subprocess.Popen | None = None
    _shared_lock = threading.Lock()
    # 业务语义：当前共享 audiotee 的过滤签名；变化时必须重建进程。
    _shared_filter_key: tuple | None = None

    # 业务语义：application 模式 PID 轮询间隔（秒）。
    _PID_POLL_SEC = 1.5
    # 业务语义：PID 集合变化后重建捕获的防抖窗口（秒）。
    _PID_DEBOUNCE_SEC = 0.75
    # 业务语义：目标应用未出声 / audiotee exit 2 时的重试间隔（秒）。
    _WAIT_RETRY_SEC = 1.0

    def __init__(
        self,
        audiotee_path: str,
        label: str = "system",
        include_pids: list[int] | None = None,
        bundle_id: str | None = None,
        display_name: str | None = None,
        status_callback=None,
    ):
        super().__init__(label)
        self.audiotee_path = audiotee_path
        self.include_pids = list(include_pids or [])
        # 业务语义：非空则进入指定应用捕获；保存 Bundle ID 而非 PID。
        self.bundle_id = bundle_id or None
        self.display_name = display_name or bundle_id or "application"
        # 业务语义：向 whicc EventLogger 回传捕获状态（可选）。
        self.status_callback = status_callback
        self._zero_warned = False  # 权限警告:只打印一次
        self._stderr_thread: threading.Thread | None = None
        self._supervisor_thread: threading.Thread | None = None
        self._pid_watch_thread: threading.Thread | None = None
        self._active_pids: set[int] = set(self.include_pids)
        # 业务语义：supervisor 三连败后置位;whicc 的 swap 早退需放行同配置重建。
        self.failed = False
        # stdin 热重配 ack:_read_stderr 收到 reconfigure 消息时置事件。
        self._reconfig_event = threading.Event()
        self._reconfig_status: str | None = None
        # 业务语义：旧版未打补丁的 audiotee 不回 ack;检测到后停用热重配。
        self._reconfig_unsupported = False
        # 状态去重:等待循环每秒解析一次,相同状态只发一次,防 events.jsonl 刷屏。
        self._last_status_key: tuple | None = None

    @property
    def waiting_for_app(self) -> bool:
        """application 模式且当前没有活着的 audiotee = 正在等待目标应用/出声。
        whicc 主循环用它豁免 12s 无数据看门狗(等待是正常态,不该重启进程)。
        """
        if not self.bundle_id:
            return False
        proc = SystemAudioSource._shared_proc
        return proc is None or proc.poll() is not None

    def _emit_status(self, status: str, **extra) -> None:
        key = (status, repr(sorted((k, repr(v)) for k, v in extra.items())))
        if key == self._last_status_key:
            return
        self._last_status_key = key
        print(f"[AudioSource] {status} {extra}", flush=True)
        if self.status_callback is not None:
            with contextlib.suppress(Exception):
                self.status_callback(status, **extra)

    def start(self) -> None:
        if not os.path.isfile(self.audiotee_path):
            raise RuntimeError(
                f"audiotee 不存在: {self.audiotee_path}\n"
                "运行: ./bin/build_audiotee.sh 编译并放在 ./bin/audiotee"
            )
        self._stop.clear()
        if self.bundle_id:
            # 调用 process_resolver：按 Bundle ID 解析当前可捕获 PID。
            # 不在 start() 里无限等待——否则会堵住 whicc 模型加载。
            # 目标未运行时由 _pid_watch_thread / _supervise 等待重试。
            ready = self._resolve_and_set_pids()
            if ready:
                self._emit_status(
                    "audio_app_starting",
                    application=self.display_name,
                    bundle_id=self.bundle_id,
                    pids=sorted(self._active_pids),
                )
            else:
                self._emit_status(
                    "audio_app_waiting",
                    application=self.display_name,
                    bundle_id=self.bundle_id,
                )

        filter_key = self._filter_key()
        with SystemAudioSource._shared_lock:
            alive = (
                SystemAudioSource._shared_proc is not None
                and SystemAudioSource._shared_proc.poll() is None
            )
            same_filter = SystemAudioSource._shared_filter_key == filter_key
            if alive and same_filter and not self.bundle_id:
                # 仅「全部系统音频」跨 swap 复用同一 audiotee (TCC 友好)。
                print(f"[audio] SystemAudioSource reusing audiotee "
                      f"subprocess pid={SystemAudioSource._shared_proc.pid}",
                      flush=True)
            elif self.bundle_id and not self._active_pids:
                # 目标应用尚未运行:不启动 audiotee,避免误捕获全部系统音频。
                if alive:
                    self._kill_shared_proc_unlocked()
                print(
                    f"[AudioSource] defer audiotee until "
                    f"{self.display_name} is running",
                    flush=True,
                )
            else:
                # 单次尝试:目标未出声(AudioteeWaitingError)交给 supervisor
                # 等待重试,绝不在 start() 里持锁 sleep 堵住模型加载/SIGHUP。
                try:
                    self._spawn_shared_unlocked()
                except AudioteeWaitingError:
                    pass

        self._supervisor_thread = threading.Thread(
            target=self._supervise, daemon=True, name=f"audiotee-sup-{self.label}"
        )
        self._supervisor_thread.start()
        if self.bundle_id:
            self._pid_watch_thread = threading.Thread(
                target=self._watch_pids, daemon=True, name="audiotee-pid-watch"
            )
            self._pid_watch_thread.start()
        print(
            f"[audio] SystemAudioSource started: {self.audiotee_path} "
            f"label={self.label} bundle_id={self.bundle_id!r} "
            f"pids={sorted(self._active_pids)}",
            flush=True,
        )

    def _filter_key(self) -> tuple:
        if self.bundle_id:
            return ("application", self.bundle_id, tuple(sorted(self._active_pids)))
        if self.include_pids:
            return ("include", tuple(sorted(self.include_pids)))
        return ("system",)

    def _resolve_and_set_pids(self) -> bool:
        """单次解析 Bundle ID → PID;成功返回 True。

        绝不在这里阻塞等待 —— 等待重试节奏由 supervisor 控制,
        目标未运行时保持等待、不回退系统音频。
        """
        from process_resolver import format_pid_set, resolve_bundle_processes

        resolved = resolve_bundle_processes(
            self.bundle_id,
            display_name_hint=self.display_name,
        )
        if resolved and resolved.process_identifiers:
            self.display_name = resolved.display_name or self.display_name
            self._active_pids = set(resolved.process_identifiers)
            self.include_pids = sorted(self._active_pids)
            print(
                f"[ProcessResolver] bundle_id={resolved.bundle_identifier} "
                f"main_path={resolved.bundle_path} "
                f"resolved_count={len(self._active_pids)} "
                f"pids={format_pid_set(self._active_pids)}",
                flush=True,
            )
            return True
        self._emit_status(
            "audio_app_waiting",
            application=self.display_name,
            bundle_id=self.bundle_id,
        )
        return False

    def _spawn_shared_unlocked(self) -> None:
        """Spawn 模块级 audiotee。调用方须已持有 _shared_lock。

        幂等:_shared_proc 已存活且过滤签名一致时直接返回(防止 supervisor
        与 _watch_pids 双重 spawn 出两个 audiotee);签名不同则先杀旧再起新。
        单次尝试:application 模式目标未出声导致的启动失败抛
        AudioteeWaitingError,由调用方(supervisor)按等待语义重试,
        不在锁内 sleep 循环(否则会堵住 start()/SIGHUP swap)。
        """
        existing = SystemAudioSource._shared_proc
        if existing is not None and existing.poll() is None:
            if SystemAudioSource._shared_filter_key == self._filter_key():
                return
            self._kill_shared_proc_unlocked()
            # macOS 26: kill 后立即 respawn 的 audiotee 可能被 TCC 静默
            # 拒绝(只回零字节)。在 kill 与 spawn 之间留 1s 让 Core Audio
            # 回收 tap 注册 —— 等待必须放这里而不是调用方,才真正隔开两个进程。
            time.sleep(1.0)
        pids = sorted(self._active_pids) if self._active_pids else list(self.include_pids)
        # 调用 build_audiotee_cmd：拼装含进程过滤的启动命令。
        cmd = build_audiotee_cmd(self.audiotee_path, pids)
        print(f"[AudioTee] action=start cmd={' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        time.sleep(0.3)
        if proc.poll() is not None:
            code = proc.returncode
            err = (proc.stderr.read() or b"").decode("utf-8", "replace")[:500]
            # 失败进程的管道显式关掉,不等 GC。
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                with contextlib.suppress(Exception):
                    if pipe:
                        pipe.close()
            # patched audiotee: exit 2 = 尚未出声;上游未打补丁时也会因
            # PID 无 Audio Object 启动失败——application 模式视作等待。
            if self.bundle_id and code not in (None, 0):
                self._emit_status(
                    "audio_app_waiting_audio",
                    application=self.display_name,
                    bundle_id=self.bundle_id,
                    pids=pids,
                    exit_code=code,
                )
                raise AudioteeWaitingError(
                    f"target app not emitting audio yet (exit {code})"
                )
            raise RuntimeError(
                f"audiotee failed to start (exit {code}): {err.strip()}"
            )
        SystemAudioSource._shared_proc = proc
        SystemAudioSource._shared_filter_key = self._filter_key()
        self._reconfig_unsupported = False
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, args=(proc,), daemon=True
        )
        self._stderr_thread.start()
        self._emit_status(
            "audio_app_capturing" if self.bundle_id else "audio_system_capturing",
            application=self.display_name if self.bundle_id else "system",
            bundle_id=self.bundle_id or "",
            pids=pids,
            pid_count=len(pids),
        )

    def _spawn_shared(self) -> None:
        with SystemAudioSource._shared_lock:
            self._spawn_shared_unlocked()

    # 业务语义：等待 patched audiotee 回 reconfigure ack 的超时（秒）。
    # 旧版未打补丁的二进制不回 ack → 超时视为不支持,回退 kill+respawn。
    _RECONFIG_ACK_SEC = 2.0

    def _try_reconfigure(self, pids: set[int]) -> bool:
        """通过 stdin NDJSON 热更新 include-processes。

        必须等到 stderr 回 `{"message_type":"reconfigure","status":"ok"}`
        才算成功——旧版未打补丁的 audiotee 不读 stdin,写入永远"成功"但
        tap 不变,若不等 ack 会造成过滤集永久漂移。ack 超时/非 ok 一律
        返回 False,由调用方 kill+respawn。
        """
        if self._reconfig_unsupported:
            return False
        proc = SystemAudioSource._shared_proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return False
        payload = json.dumps(
            {"cmd": "set_include_processes", "pids": sorted(pids)}
        ) + "\n"
        self._reconfig_event.clear()
        self._reconfig_status = None
        try:
            proc.stdin.write(payload.encode("utf-8"))
            proc.stdin.flush()
        except OSError as e:
            print(f"[AudioTee] reconfigure write failed: {e}", flush=True)
            return False
        if not self._reconfig_event.wait(self._RECONFIG_ACK_SEC):
            # 无 ack = 旧二进制(不读 stdin);记住并永久回退 respawn 路径。
            self._reconfig_unsupported = True
            print(
                "[AudioTee] reconfigure ack timeout — binary likely unpatched, "
                "falling back to respawn",
                flush=True,
            )
            return False
        if self._reconfig_status != "ok":
            print(
                f"[AudioTee] reconfigure rejected status={self._reconfig_status}",
                flush=True,
            )
            return False
        self._active_pids = set(pids)
        self.include_pids = sorted(pids)
        SystemAudioSource._shared_filter_key = self._filter_key()
        print(
            f"[AudioTee] action=reconfigure included_pids={sorted(pids)}",
            flush=True,
        )
        return True

    def _watch_pids(self) -> None:
        """定期检查目标应用 PID 集合;变化后防抖再热重配。

        职责边界:本线程绝不 spawn audiotee — spawn 只属于 _supervise
        (单一 spawner,杜绝双 audiotee 并存)。这里只做三件事:
          1. 目标退出 → kill 当前捕获(supervisor 转入等待);
          2. PID 集合变化 → 防抖后优先 stdin 热重配;
          3. 热重配不可用 → kill 当前捕获,supervisor 用新 PID respawn。
        """
        last_stable = set(self._active_pids)
        pending: set[int] | None = None
        pending_since = 0.0
        while not self._stop.is_set():
            time.sleep(self._PID_POLL_SEC)
            if self._stop.is_set():
                break
            # 调用 process_resolver：检测 Helper 增减 / 应用退出重启。
            from process_resolver import format_pid_set, resolve_bundle_processes

            resolved = resolve_bundle_processes(
                self.bundle_id, display_name_hint=self.display_name
            )
            if resolved is None or not resolved.process_identifiers:
                self._emit_status(
                    "audio_app_waiting",
                    application=self.display_name,
                    bundle_id=self.bundle_id,
                )
                # 目标退出:清空 PID 并杀掉当前过滤捕获;不回退系统音频。
                # supervisor 看到 _active_pids 为空会保持等待。
                self._active_pids = set()
                self.include_pids = []
                with SystemAudioSource._shared_lock:
                    self._kill_shared_proc_unlocked()
                last_stable = set()
                pending = None
                continue

            new_pids = set(resolved.process_identifiers)
            self.display_name = resolved.display_name or self.display_name
            if new_pids == last_stable:
                pending = None
                continue
            now = time.monotonic()
            if pending != new_pids:
                pending = new_pids
                pending_since = now
                print(
                    f"[ProcessResolver] pid_set_changed=true "
                    f"old={format_pid_set(last_stable)} "
                    f"new={format_pid_set(new_pids)}",
                    flush=True,
                )
                continue
            if now - pending_since < self._PID_DEBOUNCE_SEC:
                continue
            # 防抖窗口结束,应用新 PID 集合。
            proc_alive = (
                SystemAudioSource._shared_proc is not None
                and SystemAudioSource._shared_proc.poll() is None
            )
            if proc_alive and self._try_reconfigure(new_pids):
                self._emit_status(
                    "audio_app_capturing",
                    application=self.display_name,
                    bundle_id=self.bundle_id,
                    pids=sorted(new_pids),
                    pid_count=len(new_pids),
                )
            else:
                # 热重配不可用:更新 PID 后 kill,由 supervisor respawn。
                self._active_pids = set(new_pids)
                self.include_pids = sorted(new_pids)
                if proc_alive:
                    with SystemAudioSource._shared_lock:
                        self._kill_shared_proc_unlocked()
            last_stable = set(new_pids)
            pending = None

    def _supervise(self) -> None:
        """唯一的 audiotee spawner + pump 循环。

        - _shared_proc 不存在/已死 → (application 模式先解析 PID)spawn;
          目标未运行/未出声 → 等待重试,不计失败、不回退系统音频。
        - pump 返回(EOF/stall/stop)后,只有当 _shared_proc 仍是自己 pump
          的那个 proc 时才 kill —— 其他线程(_watch_pids/新 start)已接管
          时不误杀健康进程(C3)。
        """
        failures = 0
        while not self._stop.is_set():
            proc = SystemAudioSource._shared_proc
            proc_alive = proc is not None and proc.poll() is None
            if not proc_alive:
                # ── spawn 阶段(单一 spawner) ──
                if self.bundle_id:
                    # 调用 process_resolver：目标可能刚启动/刚出声,刷新 PID。
                    if not self._resolve_and_set_pids():
                        time.sleep(self._WAIT_RETRY_SEC)
                        continue
                try:
                    self._spawn_shared()
                    failures = 0
                except AudioteeWaitingError:
                    # 目标未出声:等待重试,不计失败。
                    time.sleep(self._WAIT_RETRY_SEC)
                    continue
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    if failures >= 3:
                        print(
                            f"\n[warn] 重启 audiotee 失败 ({e}); 此音频源已停。",
                            file=sys.stderr,
                            flush=True,
                        )
                        self._emit_status(
                            "audio_app_failed" if self.bundle_id
                            else "audio_system_failed",
                            error=str(e),
                            application=self.display_name,
                            bundle_id=self.bundle_id or "",
                        )
                        self.failed = True
                        break
                    time.sleep(2.0)
                    continue
                proc = SystemAudioSource._shared_proc
                if proc is None:
                    continue

            reason = self._pump(proc)
            if self._stop.is_set():
                break
            code = proc.poll()
            # 所有权校验:只有 _shared_proc 仍是自己 pump 的 proc 才 kill;
            # 否则说明 _watch_pids 或新 start() 已经换过进程,直接接管新进程。
            with SystemAudioSource._shared_lock:
                if SystemAudioSource._shared_proc is proc:
                    self._kill_shared_proc_unlocked()
                else:
                    continue
            if self.bundle_id and code == 2:
                # patched audiotee 运行中目标停声退出:进入等待,不告警。
                self._emit_status(
                    "audio_app_waiting_audio",
                    application=self.display_name,
                    bundle_id=self.bundle_id,
                )
                time.sleep(self._WAIT_RETRY_SEC)
                continue
            print(
                f"\n[warn] system audio {reason}; restarting audiotee — 如果输出"
                "设备切换了,捕获会自动 tap 到新设备。",
                file=sys.stderr,
                flush=True,
            )
            # 回到循环顶部统一走 spawn 阶段。
        self._put_sentinel()

    def _pump(self, proc: subprocess.Popen) -> str:
        """Forward the shared audiotee process's PCM into the queue until it ends.

        Returns why it ended: "stream ended (audiotee exited)" or "stalled (no data for
        Ns"; "stopped" when stop() was requested. Reads via select with a timeout
        instead of a plain blocking read, so a wedged tap is detected rather than
        blocking forever.
        """
        if proc is None:
            return "no process (start failed earlier?)"
        fd = proc.stdout.fileno()
        remainder = b""
        frames_seen = 0
        saw_audio = False
        last_data = time.monotonic()
        print(f"[audio] _pump start (proc={proc.pid}, fd={fd})", flush=True)
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                if time.monotonic() - last_data >= SYSTEM_AUDIO_STALL_SEC:
                    # 再查一次 fd 是否真挂掉(proc.poll 失败则进程死了)
                    if proc.poll() is not None:
                        return f"stream ended (audiotee exited, code={proc.returncode})"
                    return f"stalled (no data for {SYSTEM_AUDIO_STALL_SEC:.0f}s)"
                continue
            buf = os.read(fd, 4096)
            if not buf:
                return "stream ended (audiotee exited)"
            last_data = time.monotonic()
            buf = remainder + buf
            # s16le: 2 bytes per sample, carry a half-sample to the next round
            n = len(buf) - (len(buf) % 2)
            chunk, remainder = buf[:n], buf[n:]
            if not chunk:
                continue
            pcm = np.frombuffer(chunk, dtype="<i2")
            # 没权限时 Core Audio 静默返回全 0——8s+ 全 0 大概率是权限问题
            if not saw_audio:
                if int(np.abs(pcm).max(initial=0)) > 30:
                    saw_audio = True
                    print(f"[audio] _pump: first non-zero audio received (max={int(np.abs(pcm).max())})", flush=True)
                    if self.bundle_id:
                        self._emit_status(
                            "audio_app_receiving_pcm",
                            application=self.display_name,
                            bundle_id=self.bundle_id,
                        )
                else:
                    frames_seen += len(pcm)
                    if not self._zero_warned and frames_seen > SAMPLE_RATE * 8:
                        self._zero_warned = True
                        print(
                            "\n[warn] 系统音频捕获 ~8s 全是静音。如果实际有声音,"
                            "终端 app 几乎肯定没给「屏幕与系统录制」权限。"
                            "macOS 15+ 在「系统设置 → 隐私与安全性 → 屏幕与系统录制」"
                            "往下滚到「仅系统音频录制」子区(不是顶部那个),"
                            "加入终端 app 并打开开关,然后完全退出重启终端。"
                            "指定应用模式也需要同一「系统音频录制」权限。",
                            file=sys.stderr,
                            flush=True,
                        )
            else:
                # 看到非零数据后重置 frames_seen,防止后续有零帧
                # 又触发 warn。
                frames_seen = 0
            # 转 f32 + 平移到 [-1, 1]
            f32 = pcm.astype("<f4") / 32768.0
            self._offer(f32)
        return "stopped"

    def _kill_shared_proc_unlocked(self) -> None:
        proc = SystemAudioSource._shared_proc
        if proc is None:
            SystemAudioSource._shared_filter_key = None
            return
        with contextlib.suppress(Exception):
            if proc.stdin:
                proc.stdin.close()
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=2)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()
        SystemAudioSource._shared_proc = None
        SystemAudioSource._shared_filter_key = None

    def _kill_shared_proc(self) -> None:
        with SystemAudioSource._shared_lock:
            self._kill_shared_proc_unlocked()

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("message_type")
            if mtype == "metadata":
                enc = (msg.get("data") or {}).get("encoding", "")
                # 一直请求 16k => s16le 期望。如果出现 f32,警告,音频会乱。
                if enc and "f32" in enc:
                    print(
                        f"[warn] audiotee 输出 {enc},但解析器假设 s16le;"
                        "音频会变噪音。检查 --sample-rate 是否生效。",
                        file=sys.stderr,
                        flush=True,
                    )
            elif mtype == "reconfigure":
                print(
                    f"[AudioTee] reconfigure status={msg.get('status')} "
                    f"detail={msg.get('detail')}",
                    flush=True,
                )
                # 唤醒 _try_reconfigure 的 ack 等待。
                self._reconfig_status = msg.get("status")
                self._reconfig_event.set()

    def stop(self) -> None:
        # 注意:不杀 audiotee 子进程(全部系统音频模式)!只让 _pump 退出读循环。
        # application 模式会在下次不同 filter 启动时强制重建。
        self._stop.set()
        self._put_sentinel()
        # 等 supervise 线程退出,避免旧 _pump 还在读 pipe 时 start
        # 创建新的 _pump 同时读同一个 pipe (read-once 竞争)。
        if self._supervisor_thread is not None:
            self._supervisor_thread.join(timeout=2.0)
            if self._supervisor_thread.is_alive():
                print("[audio] warn: supervisor thread still alive after stop()",
                      flush=True)
        if self._pid_watch_thread is not None:
            self._pid_watch_thread.join(timeout=2.0)
            if self._pid_watch_thread.is_alive():
                print("[audio] warn: pid-watch thread still alive after stop()",
                      flush=True)
        print(f"[audio] SystemAudioSource stopped (audiotee kept alive)",
              flush=True)



def make_source(
    mode: str,
    audiotee_path: str | None = None,
    mic_device: int | str | None = None,
    bundle_id: str | None = None,
    display_name: str | None = None,
    status_callback=None,
) -> AudioSource:
    """根据 mode 构造 AudioSource。

    mode:
      - "system": 全部系统声音(audiotee)
      - "mic":    麦克风(sounddevice)
      - "application": 指定应用(audiotee --include-processes)
    """
    path = audiotee_path or "./bin/audiotee"
    if mode == "system":
        return SystemAudioSource(audiotee_path=path, label="system",
                                 status_callback=status_callback)
    if mode == "application":
        if not bundle_id:
            raise ValueError("application mode requires bundle_id")
        return SystemAudioSource(
            audiotee_path=path,
            label="application",
            bundle_id=bundle_id,
            display_name=display_name,
            status_callback=status_callback,
        )
    if mode == "mic":
        return MicSource(device=mic_device)
    raise ValueError(f"unknown audio mode: {mode!r}")