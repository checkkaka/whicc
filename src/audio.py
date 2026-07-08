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
    """

    # ── 模块级共享: 跨 SystemAudioSource 实例 + 跨 SIGHUP swap 复用 ──
    _shared_proc: subprocess.Popen | None = None
    _shared_lock = threading.Lock()

    def __init__(
        self,
        audiotee_path: str,
        label: str = "system",
        include_pids: list[int] | None = None,
    ):
        super().__init__(label)
        self.audiotee_path = audiotee_path
        self.include_pids = include_pids or []
        self._zero_warned = False  # 权限警告:只打印一次
        self._stderr_thread: threading.Thread | None = None
        self._supervisor_thread: threading.Thread | None = None

    def start(self) -> None:
        if not os.path.isfile(self.audiotee_path):
            raise RuntimeError(
                f"audiotee 不存在: {self.audiotee_path}\n"
                "运行: ./bin/build_audiotee.sh 编译并放在 ./bin/audiotee"
            )
        # 复用模块级 _shared_proc — 跨 SystemAudioSource 实例共享
        # 同一个 audiotee 子进程,保留 macOS Core Audio tap 注册。
        with SystemAudioSource._shared_lock:
            if (SystemAudioSource._shared_proc is not None
                    and SystemAudioSource._shared_proc.poll() is None):
                print(f"[audio] SystemAudioSource reusing audiotee "
                      f"subprocess pid={SystemAudioSource._shared_proc.pid}",
                      flush=True)
            else:
                self._spawn_shared()
        # _stop 重置: 上次 stop() 设过的清掉,让 _pump 重新读
        self._stop.clear()
        # 启 supervise 线程 — 每次 start 都新启一个 (旧线程 _pump
        # 退出 break 后整个 supervise 函数已 return)
        self._supervisor_thread = threading.Thread(
            target=self._supervise, daemon=True, name=f"audiotee-sup-{self.label}"
        )
        self._supervisor_thread.start()
        print(f"[audio] SystemAudioSource started: {self.audiotee_path}",
              flush=True)

    def _spawn_shared(self) -> None:
        """Spawn 模块级 audiotee 子进程。只在 _shared_proc 死了/为空时调。"""
        cmd = [self.audiotee_path, "--sample-rate", str(SAMPLE_RATE)]
        for pid in self.include_pids:
            cmd += ["--include-processes", str(pid)]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        time.sleep(0.3)
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode("utf-8", "replace")[:500]
            raise RuntimeError(
                f"audiotee failed to start (exit {proc.returncode}): {err.strip()}"
            )
        SystemAudioSource._shared_proc = proc
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, args=(proc,), daemon=True
        )
        self._stderr_thread.start()

    def _supervise(self) -> None:
        """Pump the SHARED audiotee process until it dies or stops.
        On stall: kill + respawn (also through _spawn_shared so the
        shared reference stays consistent).
        """
        failures = 0
        while not self._stop.is_set():
            reason = self._pump(SystemAudioSource._shared_proc)
            if self._stop.is_set():
                break
            self._kill_shared_proc()
            print(
                f"\n[warn] system audio {reason}; restarting audiotee — 如果输出"
                "设备切换了,捕获会自动 tap 到新设备。",
                file=sys.stderr,
                flush=True,
            )
            try:
                self._spawn_shared()
                failures = 0
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures >= 3:
                    print(
                        f"\n[warn] 重启 audiotee 失败 ({e}); 此音频源已停。",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                time.sleep(2.0)
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
                else:
                    frames_seen += len(pcm)
                    if not self._zero_warned and frames_seen > SAMPLE_RATE * 8:
                        self._zero_warned = True
                        print(
                            "\n[warn] 系统音频捕获 ~8s 全是静音。如果实际有声音,"
                            "终端 app 几乎肯定没给「屏幕与系统录制」权限。"
                            "macOS 15+ 在「系统设置 → 隐私与安全性 → 屏幕与系统录制」"
                            "往下滚到「仅系统音频录制」子区(不是顶部那个),"
                            "加入终端 app 并打开开关,然后完全退出重启终端。",
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

    def _kill_shared_proc(self) -> None:
        proc = SystemAudioSource._shared_proc
        if proc is None:
            return
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=2)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()

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

    def stop(self) -> None:
        # 注意:不杀 audiotee 子进程!只让 _pump 退出读循环。
        # 跨 SystemAudioSource 实例共享 _shared_proc,Core Audio tap 注册
        # 保持。下个 start() 复用同一个子进程 stdout pipe。
        self._stop.set()
        self._put_sentinel()
        # 等 supervise 线程退出,避免旧 _pump 还在读 pipe 时 start
        # 创建新的 _pump 同时读同一个 pipe (read-once 竞争)。
        if self._supervisor_thread is not None:
            self._supervisor_thread.join(timeout=2.0)
        print(f"[audio] SystemAudioSource stopped (audiotee kept alive)",
              flush=True)



def make_source(mode: str, audiotee_path: str | None = None,
                mic_device: int | str | None = None) -> AudioSource:
    """根据 mode 构造 AudioSource。

    mode:
      - "system": 系统声音(audiotee)
      - "mic":    麦克风(sounddevice)
    """
    if mode == "system":
        return SystemAudioSource(audiotee_path=audiotee_path or "./bin/audiotee")
    elif mode == "mic":
        return MicSource(device=mic_device)
    else:
        raise ValueError(f"unknown audio mode: {mode!r}")