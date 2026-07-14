"""TranslationState prompt-leak drops must work without worker-only fields."""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

translator_stub = types.ModuleType("translator_hy_mt2")
translator_stub.HyMT2Translator = object
translator_stub.classify_update = lambda old, new: {
    "mode": "reset_full",
    "delta_source_text": new,
    "shared_prefix_len": 0,
}
translator_stub.detect_language = lambda text: "en"
translator_stub.detect_glossary_hits = lambda source_text, glossary, target_lang: {}
translator_stub.set_scene_context = lambda scene: None
translator_stub._strip_prompt_leak = lambda text: ("", True) if "PROMPT_LEAK" in text else (text, False)
translator_stub._is_bad_output = lambda text, target_lang: not bool(text.strip())
sys.modules.setdefault("translator_hy_mt2", translator_stub)

languages_stub = types.ModuleType("languages")
languages_stub.normalize_target_language = lambda lang: types.SimpleNamespace(prompt_name=lang, code=lang)
languages_stub.TargetLanguage = object
sys.modules.setdefault("languages", languages_stub)

import translate_stream as ts  # noqa: E402


class _FullLeakTranslator:
    glossary = {}
    glossary_path = None

    def translate(self, source_text: str, target_lang: str):
        return "PROMPT_LEAK", 1.0


class _DeltaLeakTranslator:
    glossary = {}
    glossary_path = None

    def translate_delta(self, delta: str, **kwargs):
        return "PROMPT_LEAK", 1.0, {}

    def translate(self, source_text: str, target_lang: str):
        return "PROMPT_LEAK", 1.0


def _event(text: str = "hello") -> dict:
    return {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1.0, "text": text}


def test_translation_state_full_prompt_leak_returns_error_without_worker_attrs():
    state = ts.TranslationState(target_lang="Simplified Chinese")
    counts = {"translated": 0, "errors": 0, "deltas": 0, "fallbacks": 0}

    out = state.translate_final(
        _FullLeakTranslator(),
        "hello",
        {"mode": "reset_full", "delta_source_text": "hello", "shared_prefix_len": 0},
        _event(),
        counts,
    )

    assert out["event_type"] == "translation_error"
    assert "prompt_leak" in out["error"]
    assert counts["leak_drops"] == 1


def test_translation_state_delta_prompt_leak_falls_back_to_error_without_worker_attrs():
    state = ts.TranslationState(target_lang="Simplified Chinese")
    state.last_source_text = "hello"
    state.last_translated_text = "你好"
    counts = {"translated": 0, "errors": 0, "deltas": 0, "fallbacks": 0}

    out = state.translate_final(
        _DeltaLeakTranslator(),
        "hello world",
        {"mode": "append_only", "delta_source_text": " world", "shared_prefix_len": 5},
        _event("hello world"),
        counts,
    )

    assert out["event_type"] == "translation_error"
    assert "prompt_leak" in out["error"]
    assert counts["leak_drops"] == 2
