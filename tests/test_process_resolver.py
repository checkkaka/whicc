"""Unit tests for process_resolver (no macOS / lsappinfo required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from process_resolver import (  # noqa: E402
    format_pid_set,
    resolve_bundle_processes,
)


def test_resolve_single_process_under_bundle_path():
    paths = {
        100: "/Applications/QuickTime Player.app/Contents/MacOS/QuickTime Player",
        200: "/Applications/Safari.app/Contents/MacOS/Safari",
    }

    resolved = resolve_bundle_processes(
        "com.apple.QuickTimePlayerX",
        path_for_pid=paths.get,
        list_pids=lambda: list(paths),
        bundle_path_hint="/Applications/QuickTime Player.app",
        display_name_hint="QuickTime Player",
        main_pid_hint=100,
    )
    assert resolved is not None
    assert resolved.process_identifiers == frozenset({100})
    assert resolved.display_name == "QuickTime Player"


def test_resolve_chrome_helpers_by_bundle_prefix():
    chrome = "/Applications/Google Chrome.app"
    canary = "/Applications/Google Chrome Canary.app"
    paths = {
        10: f"{chrome}/Contents/MacOS/Google Chrome",
        11: f"{chrome}/Contents/Frameworks/Google Chrome Framework.framework/"
        "Versions/1/Helpers/Google Chrome Helper.app/Contents/MacOS/Google Chrome Helper",
        12: f"{chrome}/Contents/Frameworks/Google Chrome Framework.framework/"
        "Versions/1/Helpers/Google Chrome Helper (Renderer).app/"
        "Contents/MacOS/Google Chrome Helper (Renderer)",
        20: f"{canary}/Contents/MacOS/Google Chrome Canary",
        21: f"{canary}/Contents/Frameworks/Google Chrome Framework.framework/"
        "Versions/1/Helpers/Google Chrome Helper.app/Contents/MacOS/Google Chrome Helper",
        30: "/Applications/Safari.app/Contents/MacOS/Safari",
    }

    resolved = resolve_bundle_processes(
        "com.google.Chrome",
        path_for_pid=paths.get,
        list_pids=lambda: list(paths),
        bundle_path_hint=chrome,
        display_name_hint="Google Chrome",
        main_pid_hint=10,
    )
    assert resolved is not None
    assert resolved.process_identifiers == frozenset({10, 11, 12})
    # Must not include Canary helpers that share the Helper process name.
    assert 20 not in resolved.process_identifiers
    assert 21 not in resolved.process_identifiers


def test_resolve_missing_bundle_returns_none():
    resolved = resolve_bundle_processes(
        "com.example.Missing",
        path_for_pid=lambda _pid: None,
        list_pids=lambda: [],
        bundle_path_hint="",
        main_pid_hint=0,
    )
    assert resolved is None


def test_resolve_dedupes_pids():
    paths = {
        5: "/Apps/Foo.app/Contents/MacOS/Foo",
    }
    resolved = resolve_bundle_processes(
        "com.example.Foo",
        path_for_pid=paths.get,
        list_pids=lambda: [5, 5, 5],
        bundle_path_hint="/Apps/Foo.app",
        display_name_hint="Foo",
        main_pid_hint=5,
    )
    assert resolved is not None
    assert resolved.process_identifiers == frozenset({5})


def test_format_pid_set_sorted():
    assert format_pid_set({3, 1, 2}) == "1,2,3"


def test_prefix_collision_not_matched():
    """'/Foo.app Backup' must not match '/Foo.app'."""
    paths = {
        1: "/Applications/Foo.app/Contents/MacOS/Foo",
        2: "/Applications/Foo.app Backup/Contents/MacOS/Foo",
    }
    resolved = resolve_bundle_processes(
        "com.example.Foo",
        path_for_pid=paths.get,
        list_pids=lambda: list(paths),
        bundle_path_hint="/Applications/Foo.app",
        display_name_hint="Foo",
        main_pid_hint=1,
    )
    assert resolved is not None
    assert resolved.process_identifiers == frozenset({1})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
