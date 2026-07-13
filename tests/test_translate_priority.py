"""阶段二/三：翻译优先级队列、partial 全量流式、final 复用。"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


class MockTranslator:
    """不连真实 LM Studio 的翻译器桩。"""

    def __init__(self):
        self.glossary = {"en2zh": {}, "zh2en": {}}
        self.glossary_path = None
        self.temperature = 0.7
        self.top_p = 0.6
        self.top_k = 20
        self.repetition_penalty = 1.05
        self.max_new_tokens = 80
        self.calls: list[tuple] = []
        self.translate_calls: list[str] = []
        self._sig_extra = ""
        self._fail_streaming = False
        self._bad_output = False
        self._block_until: threading.Event | None = None
        self._started = threading.Event()
        self._backend = MagicMock()
        self._backend.base_url = "http://mock-vllm:8000"
        self._backend.model_id = "mock-model"
        self._backend._active_model_id = "mock-model"

    def build_request_signature(self, source_text: str,
                                target_lang: str = "Simplified Chinese",
                                context=None) -> str:
        # 与真实签名同形：任一输入变化 → miss；_sig_extra 模拟模型/参数变化
        payload = {
            "source_text": source_text,
            "target_lang": target_lang,
            "context": context,
            "extra": self._sig_extra,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "model": self._backend._active_model_id,
            "base_url": self._backend.base_url,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def translate_streaming(self, source_text: str, on_token,
                            context=None,
                            target_lang: str = "Simplified Chinese",
                            cancel_check=None):
        self._started.set()
        if self._block_until is not None:
            while not self._block_until.is_set():
                if cancel_check and cancel_check():
                    from translator_hy_mt2 import TranslationCancelled
                    raise TranslationCancelled("cancelled")
                time.sleep(0.01)
            if cancel_check and cancel_check():
                from translator_hy_mt2 import TranslationCancelled
                raise TranslationCancelled("cancelled")
        if self._fail_streaming:
            raise RuntimeError("mock stream fail")
        zh = f"译:{source_text}"
        if self._bad_output:
            zh = "here is the translation: bad"
        # 通过阻塞/取消后再记账，避免被取消的请求污染断言
        self.calls.append(("stream", source_text, target_lang))
        on_token(zh, zh)
        return zh, 12.5

    def translate_delta_streaming(self, *args, **kwargs):
        raise AssertionError("partial 主路径不得调用 translate_delta_streaming")

    def translate(self, source_text: str, target_lang: str = "Simplified Chinese",
                  context=None):
        self.translate_calls.append(source_text)
        self.calls.append(("full", source_text, target_lang))
        return f"全量:{source_text}", 33.0


def _make_worker(translator=None, priority_enabled=True):
    import translate_stream as ts

    translator = translator or MockTranslator()
    f_trans = io.StringIO()
    f_zh = io.StringIO()
    f_bi = io.StringIO()
    state = ts.TranslationState(target_lang="Simplified Chinese")
    worker = ts.TranslateWorker(
        translator, state, {},
        f_trans, f_zh, f_bi, threading.Lock(),
        {"translated": 0, "errors": 0, "deltas": 0, "fallbacks": 0},
        True,
        priority_enabled=priority_enabled,
    )
    return worker, translator, f_trans


def test_source_key_prefers_event_field():
    import translate_stream as ts

    ev = {"source_key": "stable-1", "seg_start": 0, "seg_end": 1, "audio_end_sec": 9}
    assert ts._source_key(ev) == "stable-1"
    ev2 = {"seg_start": 1, "seg_end": 2, "audio_end_sec": 3.5}
    assert ts._source_key(ev2) == "1-2-3.5"


def test_final_has_priority_over_partial():
    """final 优先级 0，应先于同队列的 partial 被取出。"""
    worker, _translator, _ = _make_worker()
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1, "revision": 1,
         "audio_start_sec": 0, "text": "hello"},
        "hello", "k1",
    )
    worker.dispatch_final(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello world", "accepted": True},
        "hello world", "k2",
    )
    first = worker._queue.get_nowait()
    second = worker._queue.get_nowait()
    assert first[0] == 0 and first[2]["mode"] == "final"
    assert second[0] == 1 and second[2]["mode"] == "partial"


def test_partial_keeps_only_latest_revision_per_key():
    """同 source_key 只保留最新 revision；旧 revision 被跳过。"""
    worker, translator, _ = _make_worker()
    translator._block_until = threading.Event()
    worker.start()

    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "a"},
        "a", "same-key",
    )
    assert translator._started.wait(2)
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 2, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "ab"},
        "ab", "same-key",
    )
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 3, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "abc"},
        "abc", "same-key",
    )
    time.sleep(0.05)
    translator._block_until.set()

    deadline = time.time() + 3
    while time.time() < deadline:
        stream_srcs = [c[1] for c in translator.calls if c[0] == "stream"]
        if "abc" in stream_srcs:
            break
        time.sleep(0.02)
    time.sleep(0.08)
    worker.stop()

    stream_srcs = [c[1] for c in translator.calls if c[0] == "stream"]
    assert stream_srcs == ["abc"], stream_srcs


def test_request_signature_changes_on_any_input():
    import translator_hy_mt2 as th
    from translator_hy_mt2 import HyMT2Translator

    # 不连网：直接测签名函数（用 stub backend 字段）
    tr = HyMT2Translator.__new__(HyMT2Translator)
    tr.glossary = {"en2zh": {"Foo": "福"}, "zh2en": {}}
    tr.glossary_path = None
    tr.temperature = 0.7
    tr.top_p = 0.6
    tr.top_k = 20
    tr.repetition_penalty = 1.05
    tr.max_new_tokens = 80
    tr._backend = MagicMock()
    tr._backend.base_url = "http://a:8000"
    tr._backend.model_id = "m1"
    tr._backend._active_model_id = "m1"

    th.set_scene_context("scene-A")
    base = tr.build_request_signature("Hello Foo", target_lang="Simplified Chinese")

    # 源文变化
    assert tr.build_request_signature("Hello Bar") != base
    # 目标语言变化
    assert tr.build_request_signature("Hello Foo", target_lang="English") != base
    # 场景变化
    th.set_scene_context("scene-B")
    assert tr.build_request_signature("Hello Foo") != base
    th.set_scene_context("scene-A")
    # 术语表变化
    tr.glossary = {"en2zh": {"Foo": "福福"}, "zh2en": {}}
    assert tr.build_request_signature("Hello Foo") != base
    tr.glossary = {"en2zh": {"Foo": "福"}, "zh2en": {}}
    # 采样参数变化
    tr.temperature = 0.1
    assert tr.build_request_signature("Hello Foo") != base
    tr.temperature = 0.7
    tr.top_k = 5
    assert tr.build_request_signature("Hello Foo") != base
    tr.top_k = 20
    # 模型/地址变化
    tr._backend._active_model_id = "m2"
    assert tr.build_request_signature("Hello Foo") != base
    tr._backend._active_model_id = "m1"
    tr._backend.base_url = "http://b:8000"
    assert tr.build_request_signature("Hello Foo") != base
    # 恢复后应与 base 一致
    tr._backend.base_url = "http://a:8000"
    assert tr.build_request_signature("Hello Foo") == base
    th.set_scene_context("")


def test_cancelled_partial_not_reused():
    worker, translator, f_trans = _make_worker()
    translator._block_until = threading.Event()
    worker.start()

    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello"},
        "hello", "k-cancel",
    )
    assert translator._started.wait(2)
    # final 入队应立刻取消同 key partial
    worker.dispatch_final(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello", "accepted": True},
        "hello", "k-cancel",
    )
    translator._block_until.set()

    deadline = time.time() + 3
    while not translator.translate_calls and time.time() < deadline:
        time.sleep(0.02)
    worker.stop()

    # 取消的 partial 不可复用 → 必须全量重译
    assert translator.translate_calls == ["hello"]
    lines = [json.loads(x) for x in f_trans.getvalue().splitlines() if x.strip()]
    finals = [x for x in lines if x.get("event_type") == "translation_final"]
    assert finals
    assert finals[-1].get("final_reused_partial") is not True


def test_failed_partial_not_reused():
    worker, translator, f_trans = _make_worker()
    translator._fail_streaming = True
    worker.start()
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello"},
        "hello", "k-fail",
    )
    deadline = time.time() + 2
    while not translator.calls and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.05)
    worker.dispatch_final(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello", "accepted": True},
        "hello", "k-fail",
    )
    deadline = time.time() + 3
    while not translator.translate_calls and time.time() < deadline:
        time.sleep(0.02)
    worker.stop()
    assert translator.translate_calls == ["hello"]
    lines = [json.loads(x) for x in f_trans.getvalue().splitlines() if x.strip()]
    finals = [x for x in lines if x.get("event_type") == "translation_final"]
    assert finals and finals[-1].get("final_reused_partial") is not True


def test_invalid_partial_output_not_reused():
    worker, translator, f_trans = _make_worker()
    translator._bad_output = True
    worker.start()
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello"},
        "hello", "k-bad",
    )
    deadline = time.time() + 2
    while not translator.calls and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.05)
    worker.dispatch_final(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello", "accepted": True},
        "hello", "k-bad",
    )
    deadline = time.time() + 3
    while not translator.translate_calls and time.time() < deadline:
        time.sleep(0.02)
    worker.stop()
    assert translator.translate_calls == ["hello"]


def test_final_reuses_matching_partial():
    worker, translator, f_trans = _make_worker()
    worker.start()
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello"},
        "hello", "k-hit",
    )
    deadline = time.time() + 2
    while not translator.calls and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.05)
    worker.dispatch_final(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello", "accepted": True},
        "hello", "k-hit",
    )
    deadline = time.time() + 3
    lines = []
    while time.time() < deadline:
        lines = [json.loads(x) for x in f_trans.getvalue().splitlines() if x.strip()]
        if any(x.get("event_type") == "translation_final" for x in lines):
            break
        time.sleep(0.02)
    worker.stop()

    finals = [x for x in lines if x.get("event_type") == "translation_final"]
    assert finals, lines
    hit = finals[-1]
    assert hit.get("final_reused_partial") is True
    assert hit.get("translate_ms") == 0
    assert hit.get("partial_translate_ms") == 12.5
    assert hit.get("translated_full_text") == "译:hello"
    assert translator.translate_calls == []  # 未全量重译


def test_length_finish_reason_not_reused():
    """finish_reason=length 的截断 partial 禁止复用。"""
    worker, translator, f_trans = _make_worker()
    # 直接写入截断快照，绕过流式桩
    sig = translator.build_request_signature("hello")
    worker._partial_snapshots["k-len"] = {
        "source_text": "hello",
        "translated_text": "半截译",
        "completed": True,
        "output_valid": True,
        "request_signature": sig,
        "translate_ms": 10.0,
        "finish_reason": "length",
    }
    worker.partial_cache["k-len"] = ("hello", "半截译")
    assert worker._can_reuse_partial("k-len", "hello", "Simplified Chinese") is None

    worker.start()
    worker.dispatch_final(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello", "accepted": True},
        "hello", "k-len",
    )
    deadline = time.time() + 3
    while not translator.translate_calls and time.time() < deadline:
        time.sleep(0.02)
    worker.stop()
    assert translator.translate_calls == ["hello"]
    lines = [json.loads(x) for x in f_trans.getvalue().splitlines() if x.strip()]
    finals = [x for x in lines if x.get("event_type") == "translation_final"]
    assert finals and finals[-1].get("final_reused_partial") is not True


def test_length_finish_reason_clears_partial_cache():
    """流式完成但 finish_reason=length 时必须清掉 partial_cache。"""
    worker, translator, _f_trans = _make_worker()
    translator.last_finish_reason = "length"
    worker.start()
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello"},
        "hello", "k-len-cache",
    )
    deadline = time.time() + 2
    while not translator.calls and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.05)
    worker.stop()
    assert "k-len-cache" not in worker.partial_cache
    snap = worker._partial_snapshots.get("k-len-cache")
    assert snap is not None
    assert snap.get("output_valid") is False
    assert snap.get("finish_reason") == "length"


def test_cancelled_partial_clears_partial_cache():
    """取消后 partial_cache 条目被清除，避免脏缓存。"""
    worker, translator, f_trans = _make_worker()
    translator._block_until = threading.Event()
    worker.start()
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello"},
        "hello", "k-cache-clear",
    )
    assert translator._started.wait(2)
    worker.dispatch_final(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello", "accepted": True},
        "hello", "k-cache-clear",
    )
    translator._block_until.set()
    deadline = time.time() + 3
    while not translator.translate_calls and time.time() < deadline:
        time.sleep(0.02)
    worker.stop()
    assert "k-cache-clear" not in worker.partial_cache


def test_signature_mismatch_forces_full_retranslate():
    worker, translator, f_trans = _make_worker()
    worker.start()
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello"},
        "hello", "k-sig",
    )
    deadline = time.time() + 2
    while not translator.calls and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.05)
    # 模拟请求签名输入变化（模型/参数）
    translator._sig_extra = "changed"
    worker.dispatch_final(
        {"seg_start": 0, "seg_end": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hello", "accepted": True},
        "hello", "k-sig",
    )
    deadline = time.time() + 3
    while not translator.translate_calls and time.time() < deadline:
        time.sleep(0.02)
    worker.stop()
    assert translator.translate_calls == ["hello"]
    lines = [json.loads(x) for x in f_trans.getvalue().splitlines() if x.strip()]
    finals = [x for x in lines if x.get("event_type") == "translation_final"]
    assert finals and finals[-1].get("final_reused_partial") is not True


def test_cli_defaults_match_official_params():
    """CLI --top-k / --repetition-penalty 对齐 Hy-MT2 官方推荐。"""
    import argparse
    import translate_stream as ts

    # 复用 main 里同名参数的默认值（不跑 main）
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    # 从源码抽检：模块级常量或 argparse 默认已改为官方值
    src = Path(ts.__file__).read_text(encoding="utf-8")
    assert 'add_argument("--top-k", type=int, default=20)' in src
    assert 'add_argument("--repetition-penalty", type=float, default=1.05)' in src


def test_partial_token_events_carry_full_text_never_commit():
    worker, translator, f_trans = _make_worker()
    worker.start()
    worker.dispatch_partial(
        {"seg_start": 0, "seg_end": 1, "revision": 1, "audio_end_sec": 1,
         "audio_start_sec": 0, "text": "hi"},
        "hi", "k-tok",
    )
    deadline = time.time() + 2
    while time.time() < deadline:
        if f_trans.getvalue().strip():
            break
        time.sleep(0.02)
    worker.stop()
    lines = [json.loads(x) for x in f_trans.getvalue().splitlines() if x.strip()]
    partials = [x for x in lines if x.get("event_type") == "translation_partial"]
    assert partials
    for p in partials:
        assert "translated_full_text" in p
        assert p["translated_full_text"]  # 完整累计
    # 未 commit 到 zh/bi（worker 内部 f_zh 仍空）
    assert worker.f_zh.getvalue() == ""
