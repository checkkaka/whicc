"""Smoke tests for audio.make_source application wiring (no audiotee binary)."""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

import audio  # noqa: E402


def test_make_source_application_requires_bundle_id():
    with pytest.raises(ValueError, match="bundle_id"):
        audio.make_source("application", audiotee_path="/tmp/missing-audiotee")


def test_make_source_application_sets_label_and_bundle():
    import audio

    src = audio.make_source(
        "application",
        audiotee_path="/tmp/missing-audiotee",
        bundle_id="com.google.Chrome",
        display_name="Google Chrome",
    )
    assert src.label == "application"
    assert src.bundle_id == "com.google.Chrome"
    assert src.display_name == "Google Chrome"


def test_make_source_system_label():
    import audio

    src = audio.make_source("system", audiotee_path="/tmp/missing-audiotee")
    assert src.label == "system"
    assert src.bundle_id is None


def test_build_audiotee_cmd_multi_pid_shape():
    """Multi-PID must be one flag + many values (AudioTee array option)."""
    import audio

    cmd = audio.build_audiotee_cmd("/bin/audiotee", [10, 11, 12])
    idx = cmd.index("--include-processes")
    assert cmd[idx + 1 :] == ["10", "11", "12"]
    # flag 只出现一次
    assert cmd.count("--include-processes") == 1
    assert cmd[:3] == ["/bin/audiotee", "--sample-rate", "16000"]


def test_build_audiotee_cmd_no_pids():
    import audio

    cmd = audio.build_audiotee_cmd("/bin/audiotee")
    assert "--include-processes" not in cmd


def test_waiting_for_app_property():
    """application 模式无活跃 audiotee = 等待态;system 模式恒 False。"""
    import audio

    app_src = audio.make_source(
        "application",
        audiotee_path="/tmp/missing-audiotee",
        bundle_id="com.example.Foo",
    )
    sys_src = audio.make_source("system", audiotee_path="/tmp/missing-audiotee")
    # 无共享 audiotee 进程时:application 等待中,system 不算等待。
    audio.SystemAudioSource._shared_proc = None
    assert app_src.waiting_for_app is True
    assert sys_src.waiting_for_app is False


def test_reconfigure_requires_ack(monkeypatch):
    """stdin 写成功但无 ack(旧二进制)必须判失败并停用热重配。"""
    import io

    import audio

    src = audio.make_source(
        "application",
        audiotee_path="/tmp/missing-audiotee",
        bundle_id="com.example.Foo",
    )

    class FakeProc:
        stdin = io.BytesIO()

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(audio.SystemAudioSource, "_shared_proc", FakeProc())
    monkeypatch.setattr(audio.SystemAudioSource, "_RECONFIG_ACK_SEC", 0.05)
    assert src._try_reconfigure({1, 2}) is False
    assert src._reconfig_unsupported is True
    # 已标记不支持后直接短路
    assert src._try_reconfigure({3}) is False


def test_reconfigure_closed_stdin_returns_false(monkeypatch):
    """stall 重启与 PID watcher 竞争时，已关闭 stdin 不能杀掉 watcher。"""
    import io
    import audio

    src = audio.make_source(
        "application", audiotee_path="/tmp/missing-audiotee",
        bundle_id="com.example.Foo",
    )

    class FakeProc:
        stdin = io.BytesIO()

        @staticmethod
        def poll():
            return None

    proc = FakeProc()
    proc.stdin.close()
    monkeypatch.setattr(audio.SystemAudioSource, "_shared_proc", proc)
    assert src._try_reconfigure({1}) is False


def test_audio_source_never_offers_pcm_after_stop_and_sentinel_is_idempotent():
    class Source(audio.AudioSource):
        def start(self):
            self._begin_stream()

        def stop(self):
            self._stop.set()
            self._put_sentinel()

    source = Source("test")
    source.start()
    source.stop()
    source._offer(np.ones(4, dtype=np.float32))
    source._put_sentinel()

    assert source.queue.get_nowait() is None
    with pytest.raises(queue.Empty):
        source.queue.get_nowait()


def test_system_audio_stop_joins_pump_before_publishing_sentinel(monkeypatch):
    events = []

    class FinishedThread:
        def join(self, timeout):
            events.append("join")

        def is_alive(self):
            return False

    source = audio.SystemAudioSource("/tmp/fake-audiotee")
    source._supervisor_thread = FinishedThread()
    monkeypatch.setattr(
        source, "_put_sentinel", lambda: events.append("sentinel"))

    source.stop()

    assert events == ["join", "sentinel"]
