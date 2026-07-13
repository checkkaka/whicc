"""手填翻译场景注入契约测试。

覆盖：
  - set_scene_context → build_messages / build_delta_messages 真正注入 prompt
  - 清空后不再注入
  - read_user_scene 读 lang_config.json 的 scene 键
  - _strip_scene_echo 剥离场景回显
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

import translate_stream as ts  # noqa: E402
import translator_hy_mt2 as hy  # noqa: E402


def test_build_messages_includes_scene_when_set():
    hy.set_scene_context("AI访谈")
    try:
        msgs = hy.build_messages("Hello world", target_lang="Simplified Chinese")
        content = msgs[0]["content"]
        assert "翻译场景：AI访谈" in content
        assert "Hello world" in content
    finally:
        hy.set_scene_context("")


def test_build_messages_omits_scene_when_empty():
    hy.set_scene_context("")
    msgs = hy.build_messages("Hello world", target_lang="Simplified Chinese")
    content = msgs[0]["content"]
    assert "翻译场景：" not in content


def test_build_delta_messages_includes_scene():
    hy.set_scene_context("NBA总决赛")
    try:
        msgs = hy.build_delta_messages("and the score", target_lang="Simplified Chinese")
        content = msgs[0]["content"]
        assert "翻译场景：NBA总决赛" in content
        assert "and the score" in content
    finally:
        hy.set_scene_context("")


def test_build_messages_english_prompt_scene():
    """源文为英文且目标为非中文时走英文 prompt 分支。"""
    hy.set_scene_context("tech keynote")
    try:
        msgs = hy.build_messages("Hello", target_lang="Japanese")
        content = msgs[0]["content"]
        assert "Translation scene: tech keynote" in content
    finally:
        hy.set_scene_context("")


def test_read_user_scene_ok(tmp_path: Path):
    cfg = tmp_path / "lang_config.json"
    cfg.write_text(json.dumps({"scene": "  AI访谈  ", "target_lang": "auto"}),
                   encoding="utf-8")
    assert ts.read_user_scene(str(tmp_path)) == "AI访谈"


def test_read_user_scene_missing_file(tmp_path: Path):
    assert ts.read_user_scene(str(tmp_path)) == ""


def test_read_user_scene_missing_key(tmp_path: Path):
    cfg = tmp_path / "lang_config.json"
    cfg.write_text(json.dumps({"target_lang": "zh"}), encoding="utf-8")
    assert ts.read_user_scene(str(tmp_path)) == ""


def test_read_user_scene_non_string(tmp_path: Path):
    cfg = tmp_path / "lang_config.json"
    cfg.write_text(json.dumps({"scene": 123}), encoding="utf-8")
    assert ts.read_user_scene(str(tmp_path)) == ""


def test_strip_scene_echo_removes_prefix_and_scene():
    hy.set_scene_context("AI访谈")
    try:
        cleaned, hit = hy._strip_scene_echo("翻译场景：AI访谈\n这是译文")
        assert hit is True
        assert "AI访谈" not in cleaned or cleaned.strip() == "这是译文"
        assert "这是译文" in cleaned
    finally:
        hy.set_scene_context("")


def test_strip_scene_echo_noop_when_no_scene():
    hy.set_scene_context("")
    cleaned, hit = hy._strip_scene_echo("普通译文")
    assert hit is False
    assert cleaned == "普通译文"


def test_get_scene_context_roundtrip():
    hy.set_scene_context("  WWDC  ")
    try:
        assert hy.get_scene_context() == "WWDC"
    finally:
        hy.set_scene_context("")
