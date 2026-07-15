from __future__ import annotations

import importlib.util
import json
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "whicc_file_audio.py"
SPEC = importlib.util.spec_from_file_location("whicc_file_audio", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_feeder_keeps_partial_tail(tmp_path):
    pcm = tmp_path / "audio.pcm"
    pcm.write_bytes(b"0123456789")
    assert list(MODULE.iter_audio_chunks(pcm, 8, 0)) == [
        b"01234567", b"89",
    ]


def test_feeder_appends_requested_silence(tmp_path):
    pcm = tmp_path / "audio.pcm"
    pcm.write_bytes(b"0123456789")
    assert b"".join(MODULE.iter_audio_chunks(pcm, 8, 6)) == (
        b"0123456789" + bytes(6)
    )


def test_feeder_declares_eof_without_deleting_written_segments(
        tmp_path, monkeypatch):
    pcm = tmp_path / "audio.pcm"
    pcm.write_bytes(bytes(16))
    segdir = tmp_path / "segments"
    monkeypatch.setattr(MODULE, "SEG_DIR", str(segdir))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        MODULE.sys, "argv",
        ["whicc_file_audio.py", str(pcm), "--chunk-ms", "1"],
    )

    MODULE.main()

    assert (segdir / "seg-000000.pcm").exists()
    metadata = json.loads(
        (segdir / "seg-000000.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["sequence"] == 0
    assert metadata["capture_end_mono_ns"] > 0
    assert (segdir / MODULE.SEG_DONE_FILE).read_text(
        encoding="utf-8") == "done\n"
