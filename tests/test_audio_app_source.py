"""Smoke tests for audio.make_source application wiring (no audiotee binary)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def test_make_source_application_requires_bundle_id():
    import audio

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


def test_include_processes_cli_shape():
    """Multi-PID must be one flag + many values (AudioTee array option)."""
    pids = [10, 11, 12]
    cmd = ["audiotee", "--sample-rate", "16000", "--include-processes", *[str(p) for p in pids]]
    assert cmd[cmd.index("--include-processes") + 1 :] == ["10", "11", "12"]
