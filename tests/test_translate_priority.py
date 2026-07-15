from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path
import json

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import translate_stream as ts  # noqa: E402
from translator_hy_mt2 import VLLMBackend  # noqa: E402


class FakeTranslator:
    def __init__(self, *, block_partial=False, final_error=False):
        self.block_partial = block_partial
        self.final_error = final_error
        self.partial_started = threading.Event()
        self.release_partial = threading.Event()
        self.final_started = threading.Event()
        self.translate_calls = 0
        self.finish_reason = "stop"
        self.final = None
        self.stream_read_timeout = None
        self.cancel_calls = 0

    def fork(self):
        self.final = FakeTranslator(final_error=self.final_error)
        return self.final

    def request_signature(self, source, target):
        return f"sig:{source}:{target}"

    def translate_streaming(self, source, on_token, target_lang,
                            should_cancel=None):
        self.partial_started.set()
        if self.block_partial:
            self.release_partial.wait(2)
        on_token("译", "译文")
        return "译文", 10.0

    def translate(self, source, target_lang):
        self.translate_calls += 1
        self.final_started.set()
        if self.final_error:
            raise RuntimeError("final failed")
        return "译文", 10.0

    def cancel_streaming(self):
        self.cancel_calls += 1

    def set_stream_read_timeout(self, seconds):
        self.stream_read_timeout = seconds


def worker(translator):
    output = io.StringIO()
    return ts.TranslateWorker(
        translator, ts.TranslationState(), {}, output, io.StringIO(),
        io.StringIO(), threading.Lock(),
        {"translated": 0, "errors": 0, "deltas": 0, "fallbacks": 0}, True,
    )


def test_only_partial_connection_gets_short_read_timeout():
    translator = FakeTranslator()
    worker(translator)
    assert translator.stream_read_timeout == ts.PARTIAL_READ_TIMEOUT_SEC
    assert translator.final.stream_read_timeout is None


def event(revision=1):
    return {"source_key": "k", "revision": revision, "seg_start": 0,
            "seg_end": 1, "audio_start_sec": 0, "audio_end_sec": 1}


def test_translation_input_uses_partial_worker_without_becoming_ui_partial():
    assert ts._is_partial_input({"event_type": "partial"}) is True
    assert ts._is_partial_input({"event_type": "translation_input"}) is True
    assert ts._is_partial_input({"event_type": "final"}) is False


def test_final_starts_while_partial_has_no_tokens():
    translator = FakeTranslator(block_partial=True)
    w = worker(translator)
    w.start()
    try:
        w.dispatch_partial(event(), "hello", "k")
        assert translator.partial_started.wait(.2)
        w.dispatch_final(event(), "hello", "k")
        assert translator.final.final_started.wait(.1)
    finally:
        translator.release_partial.set()
        w.stop()


def test_final_cancels_partial_before_sse_request_is_open(monkeypatch):
    translator = FakeTranslator()
    partial_preflight = threading.Event()
    release_preflight = threading.Event()
    partial_calls = [0]
    original_streaming = translator.translate_streaming

    def counted_streaming(*args, **kwargs):
        partial_calls[0] += 1
        return original_streaming(*args, **kwargs)

    translator.translate_streaming = counted_streaming

    def blocked_glossary(active_translator, *_args):
        if active_translator is translator:
            partial_preflight.set()
            release_preflight.wait(1)

    monkeypatch.setattr(ts, "_log_glossary_hits", blocked_glossary)
    w = worker(translator)
    w.start()
    try:
        w.dispatch_partial(event(), "hello", "k")
        assert partial_preflight.wait(.2)
        w.dispatch_final(event(), "hello", "k")
        assert translator.final.final_started.wait(.2)
        release_preflight.set()
    finally:
        release_preflight.set()
        w.stop()

    assert partial_calls[0] == 0


def test_final_reuses_only_exact_completed_partial():
    translator = FakeTranslator()
    w = worker(translator)
    w._do_partial(event(), "hello", "k")
    out = w._do_final(event(), "hello", "k")
    assert out["final_reused_partial"] is True
    assert translator.final.translate_calls == 0

    w._do_partial(event(2), "hello again", "k")
    out = w._do_final(event(3), "hello again", "k")
    assert out["final_reused_partial"] is False
    assert translator.final.translate_calls == 1


def test_reused_final_keeps_partial_finish_reason():
    translator = FakeTranslator()
    w = worker(translator)
    w.partial_cache["k"] = {
        "source_text": "hello", "translated_text": "译文", "revision": 1,
        "request_signature": "sig:hello:Simplified Chinese",
        "completed": True, "output_valid": True, "translate_ms": 8,
        "finish_reason": "stop",
    }

    out = w._do_final(event(), "hello", "k")

    assert out["final_reused_partial"] is True
    assert out["finish_reason"] == "stop"


def test_manual_length_cache_cannot_bypass_strict_reuse_gate():
    translator = FakeTranslator()
    w = worker(translator)
    w.partial_cache["k"] = {
        "source_text": "hello", "translated_text": "截断草稿", "revision": 1,
        "request_signature": "sig:hello:Simplified Chinese",
        "completed": True, "output_valid": True, "translate_ms": 8,
        "finish_reason": "length",
    }

    out = w._do_final(event(), "hello", "k")

    assert out["final_reused_partial"] is False
    assert translator.final.translate_calls == 1


def test_partial_cache_records_actual_request_signature():
    translator = FakeTranslator()
    translator.last_request_signature = "actual-request-body"
    w = worker(translator)
    w._do_partial(event(), "hello", "k")
    assert w.partial_cache["k"]["request_signature"] == "actual-request-body"


def test_worker_metrics_use_backend_actual_http_start_timestamp():
    translator = FakeTranslator()
    translator.request_start_mono_ns = 123_456
    w = worker(translator)

    w._do_partial(event(), "hello", "k")
    metric = next(json.loads(line) for line in w.f_trans.getvalue().splitlines()
                  if json.loads(line)["event_type"] == "translation_metrics")
    assert metric["http_request_start_mono_ns"] == 123_456

    translator.final.request_start_mono_ns = 234_567
    out = w._do_final(event(2), "hello final", "final-key")
    assert out["http_request_start_mono_ns"] == 234_567


def test_stream_eof_without_terminal_is_not_a_completed_translation():
    class TruncatedResponse:
        def iter_lines(self):
            return iter([
                'data: {"choices":[{"delta":{"content":"译"},'
                '"finish_reason":null}]}'
            ])

    backend = object.__new__(VLLMBackend)
    backend._active_endpoint = "chat"
    backend.last_finish_reason = ""

    with pytest.raises(RuntimeError, match="terminal"):
        backend._consume_sse(TruncatedResponse(), lambda *_args: None)


def test_length_truncated_partial_is_visible_but_never_reused():
    translator = FakeTranslator()
    translator.finish_reason = "length"
    w = worker(translator)

    w._do_partial(event(), "hello", "k")

    assert w.partial_cache["k"]["output_valid"] is False
    partials = [json.loads(line) for line in w.f_trans.getvalue().splitlines()
                if json.loads(line)["event_type"] == "translation_partial"]
    assert partials[-1]["translated_full_text"] == "译文"
    assert partials[-1]["partial_complete"] is True
    out = w._do_final(event(), "hello", "k")
    assert out["final_reused_partial"] is False
    assert translator.final.translate_calls == 1


@pytest.mark.parametrize("reason", ["content_filter", "tool_calls", "incomplete", ""])
def test_non_normal_partial_finish_reason_is_never_reused(reason):
    translator = FakeTranslator()
    translator.finish_reason = reason
    w = worker(translator)
    w._do_partial(event(), "hello", "k")
    assert w.partial_cache["k"]["output_valid"] is False


def test_short_stream_emits_one_complete_cumulative_draft(monkeypatch):
    clock = [0]

    class BurstTranslator(FakeTranslator):
        def fork(self):
            self.final = FakeTranslator()
            return self.final

        def translate_streaming(self, source, on_token, target_lang,
                                should_cancel=None):
            full = ""
            for piece in "实时翻译草稿":
                full += piece
                clock[0] += 100_000_000
                on_token(piece, full)
            return full, 10.0

    monkeypatch.setattr(ts.time, "monotonic_ns", lambda: clock[0])
    w = worker(BurstTranslator())
    w._do_partial(event(), "hello", "k")
    partials = [json.loads(line) for line in w.f_trans.getvalue().splitlines()
                if json.loads(line)["event_type"] == "translation_partial"]
    drafts = [item["translated_full_text"] for item in partials]
    assert drafts == ["实时翻译草稿"]
    assert partials[0]["partial_complete"] is True


def test_streaming_tokens_need_eight_chars_and_completion_marks_snapshot(
        monkeypatch):
    clock = [0]

    class IncrementalTranslator(FakeTranslator):
        def translate_streaming(self, source, on_token, target_lang,
                                should_cancel=None):
            full = ""
            for piece in "一二三四五六七八九":
                full += piece
                clock[0] += 100_000_000
                on_token(piece, full)
            return full, 900.0

    monkeypatch.setattr(ts.time, "monotonic_ns", lambda: clock[0])
    w = worker(IncrementalTranslator())
    w._do_partial(event(), "hello", "k")

    partials = [json.loads(line) for line in w.f_trans.getvalue().splitlines()
                if json.loads(line)["event_type"] == "translation_partial"]
    assert [(item["translated_full_text"], item["partial_complete"])
            for item in partials] == [("一二三四五六七八", False),
                                      ("一二三四五六七八九", True)]


def test_visible_partial_cadence_is_shared_across_revisions(monkeypatch):
    clock = [1_000_000_000]

    class RevisionTranslator(FakeTranslator):
        def translate_streaming(self, source, on_token, target_lang,
                                should_cancel=None):
            translated = f"完整译文{source}"
            on_token(translated, translated)
            return translated, 10.0

    monkeypatch.setattr(ts.time, "monotonic_ns", lambda: clock[0])
    w = worker(RevisionTranslator())

    w._do_partial(event(1), "one", "k")
    clock[0] += 500_000_000
    w._do_partial(event(2), "one two", "k")
    clock[0] += 600_000_000
    w._do_partial(event(3), "one two three", "k")

    partials = [json.loads(line) for line in w.f_trans.getvalue().splitlines()
                if json.loads(line)["event_type"] == "translation_partial"]
    assert [item["revision"] for item in partials] == [1, 3]
    # 被展示节流压掉的 revision 仍完成推理并更新严格复用 cache。
    assert w.partial_cache["k"]["revision"] == 3
    out = w._do_final(event(3), "one two three", "k")
    assert out["final_reused_partial"] is True


def test_newer_partial_allows_active_revision_to_finish_without_cancel(
        monkeypatch):
    monkeypatch.setattr(ts, "PARTIAL_VISIBLE_INTERVAL_NS", 0)
    translator = FakeTranslator(block_partial=True)
    w = worker(translator)
    w.start()
    try:
        w.dispatch_partial(event(1), "hello", "k")
        assert translator.partial_started.wait(.2)
        cancel_before = translator.cancel_calls
        w.dispatch_partial(event(2), "hello world", "k")
        assert translator.cancel_calls == cancel_before
        translator.release_partial.set()
        deadline = time.time() + 1
        while time.time() < deadline:
            partials = [
                json.loads(line) for line in w.f_trans.getvalue().splitlines()
                if json.loads(line)["event_type"] == "translation_partial"
            ]
            if len(partials) >= 2:
                break
            time.sleep(.01)
    finally:
        w.stop()

    assert [(item["revision"], item["source_text"], item["partial_complete"])
            for item in partials] == [
        (1, "hello", True),
        (2, "hello world", True),
    ]
    assert w.partial_cache["k"]["revision"] == 2


def test_new_revision_suppresses_old_tokens_but_keeps_complete_snapshot(
        monkeypatch):
    monkeypatch.setattr(ts, "PARTIAL_VISIBLE_INTERVAL_NS", 0)
    class SlowTranslator(FakeTranslator):
        def __init__(self):
            super().__init__()
            self.first_started = threading.Event()
            self.release_first = threading.Event()
            self.completed = []
            self.cancel_states = []

        def translate_streaming(self, source, on_token, target_lang,
                                should_cancel=None):
            if source == "hello":
                self.first_started.set()
                self.release_first.wait(1)
            self.cancel_states.append(bool(should_cancel and should_cancel()))
            if should_cancel and should_cancel():
                raise RuntimeError("transport cancelled")
            on_token("译", "稳定译文")
            self.completed.append(source)
            return "稳定译文", 500.0

    translator = SlowTranslator()
    w = worker(translator)
    w.start()
    try:
        w.dispatch_partial(event(1), "hello", "k")
        assert translator.first_started.wait(.2)
        w.dispatch_partial(event(2), "hello world", "k")
        translator.release_first.set()
        deadline = time.time() + 1
        while len(translator.completed) < 2 and time.time() < deadline:
            time.sleep(.01)
    finally:
        translator.release_first.set()
        w.stop()

    assert translator.cancel_calls == 1  # 仅 stop() 收尾，不是新 revision
    assert translator.cancel_states == [False, False]
    assert translator.completed == ["hello", "hello world"]
    partials = [json.loads(line) for line in w.f_trans.getvalue().splitlines()
                if json.loads(line)["event_type"] == "translation_partial"]
    assert {(item["revision"], item["source_text"], item["partial_complete"])
            for item in partials} == {
        (1, "hello", True),
        (2, "hello world", True),
    }


def test_partial_cannot_reappear_after_same_revision_final(monkeypatch):
    class ReturnWithoutTokens(FakeTranslator):
        def translate_streaming(self, source, on_token, target_lang,
                                should_cancel=None):
            return "译文", 10.0

    translator = ReturnWithoutTokens()
    entered_postcheck = threading.Event()
    release_postcheck = threading.Event()
    calls = [0]
    original_strip = ts._strip_prompt_leak

    def block_first_strip(text):
        calls[0] += 1
        if calls[0] == 1:
            entered_postcheck.set()
            release_postcheck.wait(1)
        return original_strip(text)

    monkeypatch.setattr(ts, "_strip_prompt_leak", block_first_strip)
    w = worker(translator)
    w.start()
    try:
        w.dispatch_partial(event(), "hello", "k")
        assert entered_postcheck.wait(.2)
        w.dispatch_final(event(), "hello", "k")
        deadline = time.time() + 1
        while "translation_final" not in w.f_trans.getvalue() and time.time() < deadline:
            time.sleep(.01)
        release_postcheck.set()
    finally:
        release_postcheck.set()
        w.stop()

    written = [json.loads(line) for line in w.f_trans.getvalue().splitlines()]
    assert any(item["event_type"] == "translation_final" for item in written)
    assert not any(item["event_type"] == "translation_partial" for item in written)


def test_unexpected_final_worker_exception_still_commits_source_only():
    translator = FakeTranslator()
    w = worker(translator)

    def broken_signature(*_args, **_kwargs):
        raise RuntimeError("signature exploded")

    w.final_translator.request_signature = broken_signature
    w.start()
    try:
        source_event = {**event(), "text": "hello", "accepted": True}
        w.dispatch_final(source_event, "hello", "k")
        deadline = time.time() + 1
        while "translation_final" not in w.f_trans.getvalue() and time.time() < deadline:
            time.sleep(.01)
    finally:
        w.stop()

    written = [json.loads(line) for line in w.f_trans.getvalue().splitlines()]
    assert [item["event_type"] for item in written] == [
        "translation_error", "translation_final"
    ]
    assert written[-1]["fallback_reason"] == "translation_failed_source_only"


def test_auxiliary_text_write_failure_cannot_overwrite_valid_final():
    class FailingAux:
        def write(self, _text):
            raise OSError("aux disk full")

        def flush(self):
            pass

    translator = FakeTranslator()
    output = io.StringIO()
    w = ts.TranslateWorker(
        translator, ts.TranslationState(), {}, output, FailingAux(),
        io.StringIO(), threading.Lock(),
        {"translated": 0, "errors": 0, "deltas": 0, "fallbacks": 0}, True,
    )
    w.start()
    try:
        source_event = {**event(), "text": "hello", "accepted": True}
        w.dispatch_final(source_event, "hello", "k")
        deadline = time.time() + 1
        while "translation_final" not in output.getvalue() and time.time() < deadline:
            time.sleep(.01)
    finally:
        w.stop()

    written = [json.loads(line) for line in output.getvalue().splitlines()]
    finals = [item for item in written
              if item["event_type"] == "translation_final"]
    assert len(finals) == 1
    assert finals[0]["translated_full_text"] == "译文"
    assert not any(item["event_type"] == "translation_error" for item in written)


@pytest.mark.parametrize("changed", ["source_text", "revision", "request_signature", "completed", "output_valid"])
def test_each_strict_reuse_miss_condition_forces_full_translation(changed):
    translator = FakeTranslator()
    w = worker(translator)
    snapshot = {
        "source_text": "hello", "translated_text": "译文", "revision": 1,
        "request_signature": "sig:hello:Simplified Chinese",
        "completed": True, "output_valid": True, "translate_ms": 8,
    }
    snapshot[changed] = {
        "source_text": "other", "revision": 2,
        "request_signature": "different", "completed": False,
        "output_valid": False,
    }[changed]
    w.partial_cache["k"] = snapshot
    out = w._do_final(event(), "hello", "k")
    assert out["final_reused_partial"] is False
    assert translator.final.translate_calls == 1


@pytest.mark.parametrize("changed", ["source_text", "revision", "request_signature", "completed", "output_valid"])
def test_strict_reuse_miss_never_falls_back_to_partial_when_full_fails(changed):
    translator = FakeTranslator(final_error=True)
    w = worker(translator)
    snapshot = {
        "source_text": "hello", "translated_text": "旧草稿", "revision": 1,
        "request_signature": "sig:hello:Simplified Chinese",
        "completed": True, "output_valid": True, "translate_ms": 8,
    }
    snapshot[changed] = {
        "source_text": "other", "revision": 2,
        "request_signature": "different", "completed": False,
        "output_valid": False,
    }[changed]
    w.partial_cache["k"] = snapshot

    out = w._do_final(event(), "hello", "k")

    assert out["event_type"] == "translation_error"
    assert translator.final.translate_calls == 1


def test_wrong_language_partial_is_not_emitted_or_reused():
    class WrongLanguageTranslator(FakeTranslator):
        def translate_streaming(self, source, on_token, target_lang,
                                should_cancel=None):
            on_token("English", "English only")
            return "English only", 10.0

    translator = WrongLanguageTranslator()
    w = worker(translator)
    w._do_partial(event(), "hello", "k")
    partials = [json.loads(line) for line in w.f_trans.getvalue().splitlines()
                if json.loads(line)["event_type"] == "translation_partial"]
    assert partials == []
    assert w.partial_cache["k"]["output_valid"] is False

    out = w._do_final(event(), "hello", "k")
    assert out["final_reused_partial"] is False
    assert translator.final.translate_calls == 1


def test_failed_final_translation_commits_source_only_caption():
    translator = FakeTranslator(final_error=True)
    w = worker(translator)
    source_event = {**event(), "text": "hello", "accepted": True}
    out = w._do_final(source_event, "hello", "k")

    w._write_final_output(out, source_event)

    written = [json.loads(line) for line in w.f_trans.getvalue().splitlines()]
    assert [item["event_type"] for item in written] == [
        "translation_error", "translation_final"
    ]
    assert written[1]["source_text"] == "hello"
    assert written[1]["translated_full_text"] == ""
    assert written[1]["fallback_reason"] == "translation_failed_source_only"


def test_prompt_leak_final_commits_source_only_caption(monkeypatch):
    w = worker(FakeTranslator())
    source_event = {**event(), "text": "hello", "accepted": True}
    monkeypatch.setattr(ts, "_strip_prompt_leak",
                        lambda _text: ("", True))

    out = w._do_final(source_event, "hello", "k")
    assert out["event_type"] == "translation_error"
    w._write_final_output(out, source_event)

    written = [json.loads(line) for line in w.f_trans.getvalue().splitlines()]
    assert [item["event_type"] for item in written] == [
        "translation_error", "translation_final"
    ]
    assert written[1]["translated_full_text"] == ""
    assert written[1]["fallback_reason"] == "translation_failed_source_only"


@pytest.mark.parametrize("translated, invalid_check", [
    ("", "empty"),
    ("Here is the translation: 你好", "explanation"),
    ("English only", "wrong_language"),
])
def test_invalid_full_final_never_commits_bad_translation(
        translated, invalid_check, monkeypatch):
    w = worker(FakeTranslator())
    w.final_translator.translate = lambda *_args, **_kwargs: (translated, 1.0)
    if invalid_check == "wrong_language":
        monkeypatch.setattr(ts, "_is_bad_output", lambda *_args: True)
    source_event = {**event(), "text": "hello", "accepted": True}

    out = w._do_final(source_event, "hello", "k")

    assert out["event_type"] == "translation_error"
    assert invalid_check in out["error"]


def test_stop_waits_for_final_queue_before_output_files_can_close():
    class ThreadProbe:
        def __init__(self):
            self.timeouts = []

        def join(self, timeout=None):
            self.timeouts.append(timeout)

        def is_alive(self):
            return False

    w = worker(FakeTranslator())
    partial_thread = ThreadProbe()
    final_thread = ThreadProbe()
    w._partial_thread = partial_thread
    w._final_thread = final_thread

    w.stop()

    assert partial_thread.timeouts == [5]
    assert final_thread.timeouts == [None]


def test_final_fifo_has_no_loss_or_reordering():
    translator = FakeTranslator()
    w = worker(translator)
    w.start()
    try:
        for index in range(3):
            e = {**event(), "source_key": f"k{index}"}
            w.dispatch_final(e, f"source {index}", f"k{index}")
        deadline = time.time() + 1
        while w.counts["translated"] < 3 and time.time() < deadline:
            time.sleep(.01)
    finally:
        w.stop()
    events = [json.loads(line) for line in w.f_trans.getvalue().splitlines()]
    assert [item["source_key"] for item in events] == ["k0", "k1", "k2"]


def test_http_client_ignores_malformed_no_proxy_after_explicit_proxy_resolution(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,::1,::1/128")
    monkeypatch.setenv("no_proxy", "127.0.0.1,::1,::1/128")
    backend = VLLMBackend.__new__(VLLMBackend)
    backend._client = None
    backend._read_timeout = 90.0
    client = backend._http_client()
    try:
        assert client._trust_env is False
    finally:
        client.close()


def test_cancel_streaming_closes_pending_http_client():
    class Client:
        closed = False

        def close(self):
            self.closed = True

    backend = VLLMBackend.__new__(VLLMBackend)
    backend._stream_lock = threading.Lock()
    backend._active_stream_response = None
    backend._client = Client()
    client = backend._client

    backend.cancel_streaming()

    assert client.closed is True
    assert backend._client is None
