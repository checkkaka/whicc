from __future__ import annotations

import queue
import json
import ast
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import audio  # noqa: E402
import whicc  # noqa: E402
from nemotron_stream import NemotronStream, native_chunk_frames  # noqa: E402
from nemotron_stream import aligned_token_segments  # noqa: E402


def test_audio_queue_marks_drop_and_sequence():
    class Source(audio.AudioSource):
        def start(self):
            pass

        def stop(self):
            pass

    source = Source("test")
    source.queue = queue.Queue(maxsize=1)
    source._offer(np.zeros(4, dtype=np.float32))
    source._offer(np.ones(4, dtype=np.float32))
    packet = source.queue.get_nowait()
    assert packet.sequence == 1
    assert packet.dropped_before is True
    assert packet.capture_end_mono_ns > 0


def test_live_queue_reports_sentinel_after_draining_audio():
    q = queue.Queue()
    audio_chunk = np.arange(4, dtype=np.float32)
    q.put(audio_chunk)
    q.put(None)

    chunks, ended = whicc.drain_audio_queue(q, first_timeout=0.01)

    assert ended is True
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], audio_chunk)


def test_segdir_done_marker_reports_true_eof(tmp_path, monkeypatch):
    monkeypatch.setattr(whicc, "SEG_DIR", str(tmp_path))
    payload = np.arange(4, dtype=np.float32).tobytes()
    (tmp_path / "seg-000000.pcm").write_bytes(payload)
    (tmp_path / "seg-000000.meta.json").write_text(
        json.dumps({"capture_end_mono_ns": 123456789, "sequence": 7}),
        encoding="utf-8",
    )
    (tmp_path / whicc.SEG_DONE_FILE).write_text("done\n", encoding="utf-8")

    chunks, next_seg, ended = whicc.read_segments(0)

    assert len(chunks) == 1
    assert chunks[0].data == payload
    assert chunks[0].capture_end_mono_ns == 123456789
    assert chunks[0].sequence == 7
    assert next_seg == 1
    assert ended is True


def test_stream_end_flush_retries_after_final_failure_without_new_audio():
    gate = whicc.StreamEndFlush()
    gate.observe(ended=True, real_audio=False)

    first = gate.silence_packet(
        has_pending_speech=True, silence_submit_sec=1.2,
        current_silence_sec=0.0, now=10.0)
    assert first is not None
    assert np.all(first == 0)

    gate.mark_failure(now=10.0)
    assert gate.retry_wait(now=10.1) == pytest.approx(0.15)
    assert gate.silence_packet(
        has_pending_speech=True, silence_submit_sec=1.2,
        current_silence_sec=1.22, now=10.1) is None
    retry = gate.silence_packet(
        has_pending_speech=True, silence_submit_sec=1.2,
        current_silence_sec=1.22,
        now=gate.retry_after_mono)
    assert retry is not None
    assert len(retry) == int(0.02 * whicc.SAMPLE_RATE)


def test_old_stream_eof_blocks_new_source_until_old_speech_is_flushed():
    gate = whicc.StreamEndFlush()
    gate.observe(ended=True, real_audio=True)

    # 下一轮即使新 source 已有数据，也不能撤销旧 EOF；主循环必须先补
    # 静音提交 old speech，再读取新 source。
    gate.observe(ended=False, real_audio=True)
    assert gate.pending is True
    assert gate.blocks_source_read(has_pending_audio=True) is True
    assert gate.blocks_source_read(has_pending_audio=False) is False


def test_audio_source_handoff_drains_retired_queue_before_new_source():
    events = []

    class FakeSource:
        def __init__(self, name, initial=()):
            self.name = name
            self.queue = queue.Queue()
            for item in initial:
                self.queue.put(item)

        def start(self):
            events.append(f"start:{self.name}")

        def stop(self):
            events.append(f"stop:{self.name}")
            self.queue.put(None)

    old_pcm = np.array([1.0], dtype=np.float32)
    new_pcm = np.array([2.0], dtype=np.float32)
    old = FakeSource("old", [old_pcm])
    new = FakeSource("new", [new_pcm])
    handoff = whicc.AudioSourceHandoff()

    current = handoff.activate(old, new)

    assert current is new
    assert events == ["stop:old", "start:new"]
    first_queue, retired = handoff.queue_for_read(current.queue)
    chunks, ended = whicc.drain_audio_queue(first_queue, first_timeout=0.01)
    handoff.finish_read(first_queue, ended=ended)
    assert retired is True
    assert ended is True
    assert len(chunks) == 1 and np.array_equal(chunks[0], old_pcm)

    second_queue, retired = handoff.queue_for_read(current.queue)
    chunks, ended = whicc.drain_audio_queue(second_queue, first_timeout=0.01)
    assert retired is False
    assert ended is False
    assert len(chunks) == 1 and np.array_equal(chunks[0], new_pcm)


def test_audio_source_handoff_keeps_old_source_running_if_new_start_fails():
    events = []

    class OldSource:
        def __init__(self):
            self.queue = queue.Queue()

        def start(self):
            events.append("start:old")

        def stop(self):
            events.append("stop:old")
            self.queue.put(None)

    class BrokenSource:
        queue = queue.Queue()

        def start(self):
            events.append("start:new")
            raise RuntimeError("permission denied")

        def stop(self):
            events.append("stop:new")

    old = OldSource()
    original_queue = old.queue
    handoff = whicc.AudioSourceHandoff()
    with pytest.raises(RuntimeError, match="permission denied"):
        handoff.activate(old, BrokenSource())

    assert events == ["stop:old", "start:new", "stop:new", "start:old"]
    assert handoff.has_retired is True
    assert old.queue is not original_queue
    retired_queue, is_retired = handoff.queue_for_read(old.queue)
    assert is_retired is True
    assert retired_queue is not original_queue


def test_audio_source_handoff_republishes_already_consumed_old_boundary():
    class AlreadyEndedSource:
        def __init__(self):
            self.queue = queue.Queue()
            self.queue.put(None)
            assert self.queue.get_nowait() is None  # supervisor EOF 已被消费

        def stop(self):
            # 幂等 sentinel 状态会让真实 AudioSource.stop() 不再 put。
            pass

    class NewSource:
        def __init__(self):
            self.queue = queue.Queue()

        def start(self):
            pass

    old = AlreadyEndedSource()
    new = NewSource()
    handoff = whicc.AudioSourceHandoff()

    assert handoff.activate(old, new) is new
    retired_queue, is_retired = handoff.queue_for_read(new.queue)
    chunks, ended = whicc.drain_audio_queue(
        retired_queue, first_timeout=0.01)

    assert is_retired is True
    assert chunks == []
    assert ended is True


def test_deferred_audio_swap_signal_only_sets_and_consumes_flag_once():
    request = whicc.DeferredAudioSwap()

    request.request()
    request.request()  # 连续 SIGHUP 合并，最终配置由安全点重新读取

    assert request.consume() is True
    assert request.consume() is False


def test_failed_audio_source_can_be_reselected_with_same_config():
    source = type("Source", (), {
        "label": "application",
        "bundle_id": "com.example.Player",
        "failed": True,
    })()

    assert whicc.audio_source_needs_swap(
        source, "application", "com.example.Player") is True


def test_normalize_segments_accepts_objects_and_dicts():
    class Segment:
        text = "hello"
        start = 0.1
        end = 0.7

    assert whicc.normalize_segments([Segment(), {"text": "!", "start": .7, "end": .8}]) == [
        {"text": "hello", "start": 0.1, "end": 0.7},
        {"text": "!", "start": 0.7, "end": 0.8},
    ]


def test_partial_and_final_share_canonical_text_and_revision():
    revisions = whicc.SourceRevision("source-1")
    raw = "chatgpt is useful."

    partial_text = whicc.canonical_asr_text(raw)
    _, partial_revision = revisions.update(partial_text)
    reject_reason, final_text = whicc.filter_result({
        "text": raw,
        "language": "en",
        "avg_logprob": -0.3,
        "avg_compression": 0.0,
    })
    _, final_revision = revisions.update(final_text)

    assert reject_reason is None
    assert partial_text == final_text == "ChatGPT is useful."
    assert partial_revision == final_revision == 1


def test_partial_event_gate_emits_first_and_latest_at_readable_cadence():
    gate = whicc.PartialEventGate(interval_ns=1_000_000_000)

    assert gate.should_emit("run:0", "hello", now_ns=100) is True
    assert gate.should_emit("run:0", "hello", now_ns=200) is False
    assert gate.should_emit("run:0", "hello world", now_ns=500_000_100) is False
    assert gate.should_emit("run:0", "hello world again",
                            now_ns=1_000_000_100) is True
    # 新句首个草稿不继承上一句冷却时间。
    assert gate.should_emit("run:1", "next", now_ns=1_000_000_101) is True


def test_event_logger_can_emit_translation_only_partial(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = whicc.EventLogger(str(path))
    logger.log_partial(
        0, 1, 0.0, 1.0, "hello",
        source_key="run:0", revision=2,
        event_type="translation_input",
    )
    logger.close()

    event = json.loads(path.read_text().strip())
    assert event["event_type"] == "translation_input"
    assert event["revision"] == 2


def test_strip_boundary_subword_overlap_keeps_whole_repeated_words():
    assert whicc.strip_boundary_subword_overlap(
        "but Tate has been yeah there's", "'s not any flaw"
    ) == "not any flaw"
    assert whicc.strip_boundary_subword_overlap(
        "it was very", "very very hard"
    ) == "very very hard"
    assert whicc.strip_boundary_subword_overlap(
        "now Tadapacha", "acha firmly in control"
    ) == "firmly in control"


def test_async_model_load_failure_is_not_reported_as_ready_success(monkeypatch):
    state = whicc.AsyncModelLoadState()
    monkeypatch.setattr(
        whicc, "_get_qwen3_model",
        lambda _path: (_ for _ in ()).throw(RuntimeError("broken weights")),
    )

    whicc._async_load_model("qwen3", "/tmp/qwen", state)

    assert state.ready.is_set() is True
    assert state.success is False
    assert "broken weights" in state.error


def test_async_model_load_success_is_explicit(monkeypatch):
    state = whicc.AsyncModelLoadState()
    monkeypatch.setattr(whicc, "_get_qwen3_model", lambda _path: object())
    monkeypatch.setattr(whicc, "_warmup_model", lambda *_args: None)

    whicc._async_load_model("qwen3", "/tmp/qwen", state)

    assert state.ready.is_set() is True
    assert state.success is True
    assert state.error == ""


def test_trim_transcription_result_uses_segment_end_time():
    result = {
        "text": "First sentence. Second sentence.",
        "segments": [
            {"text": "First sentence. ", "start": 0.1, "end": 1.2},
            {"text": "Second sentence.", "start": 1.3, "end": 2.5},
        ],
        "language": "en",
        "avg_logprob": -0.2,
    }
    trimmed = whicc.trim_transcription_result(result, 1.2)
    assert trimmed["text"] == "First sentence."
    assert trimmed["segments"] == [result["segments"][0]]
    assert trimmed["language"] == "en"


def test_finalize_native_result_reuses_stream_text_at_exact_cut(monkeypatch):
    clock = iter([100, 180])
    monkeypatch.setattr(whicc.time, "monotonic_ns", lambda: next(clock))

    class Stream:
        finalized = False

        def finalize(self):
            self.finalized = True

        def result_dict(self):
            return {
                "text": "First sentence. Second sentence.",
                "segments": [
                    {"text": "First sentence. ", "start": .1, "end": 1.2},
                    {"text": "Second sentence.", "start": 1.3, "end": 2.5},
                ],
            }

    stream = Stream()
    result = whicc.finalize_native_result(
        stream, audio_time_sec=1.2, configured_language="auto",
        finalize_stream=False)

    assert stream.finalized is False
    assert result["text"] == "First sentence."
    assert result["segments"] == [
        {"text": "First sentence. ", "start": .1, "end": 1.2}
    ]
    assert result["language"] == "en"
    assert result["_asr_request_start_mono_ns"] == 100
    assert result["_asr_complete_mono_ns"] == 180


def test_finalize_native_result_returns_none_without_timestamped_cut():
    class Stream:
        def finalize(self):
            pass

        def result_dict(self):
            return {"text": "words", "segments": []}

    assert whicc.finalize_native_result(
        Stream(), audio_time_sec=1.0, configured_language="auto",
        finalize_stream=False) is None


def test_finalize_native_result_keeps_final_padding_for_unsplit_chunk():
    """整块 final 的末 token 可略超 PCM 时长，不能像句中切点一样裁掉。"""
    class Stream:
        finalized = False

        def finalize(self):
            self.finalized = True

        def result_dict(self):
            return {
                "text": "And it's like a lifetime of.",
                "segments": [
                    {"text": "And it's like a lifetime of.",
                     "start": 0.08, "end": 2.08},
                ],
            }

    stream = Stream()
    result = whicc.finalize_native_result(
        stream, audio_time_sec=None, configured_language="auto",
        finalize_stream=True)

    assert stream.finalized is True
    assert result["text"] == "And it's like a lifetime of."
    assert result["segments"][0]["end"] == 2.08


def test_native_snapshot_only_destructively_finalizes_real_eof():
    source = inspect.getsource(whicc.main)
    assert re.search(
        r"native_final_snapshot\(\s*None,\s*finalize_stream=synthetic_eof\)",
        source)
    assert re.search(
        r"native_final_snapshot\(\s*split_sec,\s*finalize_stream=False\)",
        source)


def test_native_max_chunk_waits_for_right_context_then_splits_at_quality_boundary():
    assert whicc.native_guarded_max_split_sec(10.55, 10.0, .56) == 0
    assert whicc.native_guarded_max_split_sec(10.56, 10.0, .56) == 10.0
    assert whicc.native_guarded_max_split_sec(12.0, 10.0, 0) == 0


def test_native_max_split_moves_back_to_complete_word_boundary():
    segments = [
        {"text": " Tadapacha", "start": 8.8, "end": 9.4},
        {"text": " firm", "start": 9.4, "end": 9.85},
        {"text": "ly", "start": 9.85, "end": 10.05},
        {"text": " in", "start": 10.05, "end": 10.2},
    ]
    assert whicc.find_native_word_split_sec(
        segments, target_sec=10.0, max_lookback_sec=1.5) == pytest.approx(9.4)
    assert whicc.find_native_word_split_sec(
        segments, target_sec=10.0, max_lookback_sec=.7) == pytest.approx(9.4)
    split_sec, force_quality = whicc.resolve_native_max_split(
        segments, target_sec=10.0, max_lookback_sec=1.5)
    assert split_sec == pytest.approx(9.4)
    assert force_quality is False


def test_native_max_split_without_word_boundary_forces_quality_batch():
    assert whicc.resolve_native_max_split(
        [{"text": "firmly", "start": 9.85, "end": 10.05}],
        target_sec=10.0,
    ) == (10.0, True)


def test_speech_bounds_use_20ms_frames():
    samples = np.concatenate([
        np.zeros(320, dtype=np.float32),
        np.full(640, 0.2, dtype=np.float32),
        np.zeros(320, dtype=np.float32),
    ])
    assert whicc.find_speech_bounds_sec(samples, threshold=0.01) == (0.02, 0.06)


def test_speech_bounds_convert_to_capture_monotonic_time():
    samples = np.concatenate([
        np.zeros(320, dtype=np.float32),
        np.full(640, 0.2, dtype=np.float32),
        np.zeros(320, dtype=np.float32),
    ])
    # 音频共 80ms，capture_end=1s；有效语音位于音频内 20ms~60ms。
    assert whicc.find_speech_bounds_mono_ns(
        samples, capture_end_mono_ns=1_000_000_000, threshold=0.01
    ) == (940_000_000, 980_000_000)


def test_submission_speech_bounds_exclude_previous_sentence_overlap():
    overlap = np.full(3200, 0.2, dtype=np.float32)  # 上一句的 200ms 语音
    current = np.concatenate([
        np.zeros(320, dtype=np.float32),
        np.full(640, 0.2, dtype=np.float32),
        np.zeros(320, dtype=np.float32),
    ])
    submitted = np.concatenate([overlap, current])

    # current 共 80ms，结束于 1s；本句有效语音应是 940~980ms，不能被
    # overlap 中的上一句语音拉回到 720ms。
    assert whicc.find_submission_speech_bounds_mono_ns(
        submitted, capture_end_mono_ns=1_000_000_000, threshold=0.01,
        overlap_sample_count=len(overlap),
    ) == (940_000_000, 980_000_000)


def test_prepare_submission_audio_tracks_exact_overlap_for_failure_restore():
    overlap = np.array([-2.0, -1.0], dtype=np.float32)
    current = np.arange(6, dtype=np.float32)

    submitted, overlap_count = whicc.prepare_submission_audio(
        current, overlap, apply_overlap=True)

    assert overlap_count == len(overlap)
    assert np.array_equal(submitted[:overlap_count], overlap)
    # final 推理失败时只恢复 current，绝不能把上一句 overlap 再塞回去。
    assert np.array_equal(submitted[overlap_count:], current)


def test_fallback_split_keeps_tail_context_without_polluting_remainder():
    overlap = np.array([-2.0, -1.0], dtype=np.float32)
    first_part = np.arange(4, dtype=np.float32)

    submitted, overlap_count = whicc.prepare_split_submission_audio(
        first_part, overlap, native_active=False, native_fallback=True)

    assert overlap_count == 2
    assert np.array_equal(submitted, np.concatenate([overlap, first_part]))
    native_submitted, native_overlap = whicc.prepare_split_submission_audio(
        first_part, overlap, native_active=True, native_fallback=False)
    assert native_overlap == 0
    assert np.array_equal(native_submitted, first_part)
    assert np.array_equal(submitted[overlap_count:], first_part)


def test_probe_result_carries_real_asr_timing_for_final_reuse():
    result = {
        "text": "hello",
        "_asr_request_start_mono_ns": 100_000_000,
        "_asr_complete_mono_ns": 145_000_000,
    }
    assert whicc.precomputed_asr_timing(result) == (
        100_000_000, 145_000_000, 45.0)
    assert whicc.precomputed_asr_timing({"text": "legacy"}) is None


def test_native_commit_failure_keeps_exact_raw_remainder_and_forces_batch():
    class ExplodingStream:
        def commit_through(self, _seconds):
            raise RuntimeError("commit failed")

        def reset(self, *_args):
            raise RuntimeError("reset failed too")

    original = np.arange(12, dtype=np.float32)
    submitted_prefix = original[:7]
    raw_remainder = original[7:]

    fallback, protected_remainder, error = whicc.sync_native_after_submit(
        ExplodingStream(), commit_sec=0.7, language="en", right_context=6,
        backend_after_submit="nemotron", native_active_before_submit=True,
        current_fallback=False, raw_remainder=raw_remainder,
    )

    assert fallback is True
    assert "commit failed" in str(error)
    assert np.array_equal(protected_remainder, raw_remainder)
    assert np.array_equal(
        np.concatenate([submitted_prefix, protected_remainder]), original)


def test_all_successful_submit_paths_use_protected_native_sync():
    source = inspect.getsource(whicc.main)
    assert source.count("sync_native_after_submit(") == 3
    assert "native_stream.commit_through(" not in source


def test_remainder_vad_keeps_short_speech_for_next_chunk():
    remainder = np.concatenate([
        np.full(640, 0.2, dtype=np.float32),
        np.zeros(320, dtype=np.float32),
    ])
    assert whicc.carry_remainder_vad(
        remainder, threshold=0.01,
        has_speech=False, silence_streak=0.0, speech_accumulated=0.0,
    ) == pytest.approx((True, 0.02, 0.04))


def test_packet_vad_checks_every_20ms_not_only_the_tail():
    packet = np.concatenate([
        np.full(320, 0.2, dtype=np.float32),
        np.zeros(1280, dtype=np.float32),
    ])
    assert whicc.update_vad_state(
        packet, threshold=0.01,
        has_speech=False, silence_streak=0.0, speech_accumulated=0.0,
    ) == pytest.approx((True, 0.08, 0.02))


def test_partial_hallucination_filter():
    assert whicc.is_valid_partial_text("The.") is False
    assert whicc.is_valid_partial_text("The quick brown fox") is True


def test_punct_end_requires_same_text_from_two_probes():
    candidate, stable = whicc.update_punct_end_stability(None, "We are still talking.")
    assert candidate == "We are still talking."
    assert stable is False

    candidate, stable = whicc.update_punct_end_stability(
        candidate, "We are still talking and adding words."
    )
    assert candidate == "We are still talking and adding words."
    assert stable is False

    candidate, stable = whicc.update_punct_end_stability(candidate, candidate)
    assert stable is True

    candidate, stable = whicc.update_punct_end_stability(
        "First sentence.", "First sentence. Next thought"
    )
    assert candidate is None
    assert stable is False

    assert whicc.update_punct_end_stability(
        "Question?", "Question? Next thought"
    ) == ("Question?", True)

    # `?` 首次出现也不能立即提交：decoder 可能在下一轮把问号后移。
    assert whicc.update_punct_end_stability(
        None, "Question?"
    ) == ("Question?", False)
    assert whicc.update_punct_end_stability(
        None, "完成了。"
    ) == ("完成了。", False)

    candidate, stable = whicc.update_punct_end_stability(
        None, "Could AI help people stop?"
    )
    assert stable is False
    candidate, stable = whicc.update_punct_end_stability(
        candidate, "Could AI help people stop feeling misunderstood?"
    )
    assert (candidate, stable) == (
        "Could AI help people stop feeling misunderstood?", False
    )
    assert whicc.update_punct_end_stability(candidate, candidate) == (
        candidate, True
    )

    assert whicc.update_punct_end_stability(candidate, "No punctuation") == (None, False)


def test_append_only_native_punctuation_waits_for_next_observation():
    candidate, stable = whicc.update_punct_end_stability(
        None, "Complete sentence."
    )
    assert (candidate, stable) == ("Complete sentence.", False)
    assert whicc.update_punct_end_stability(
        candidate, "Complete sentence."
    ) == ("Complete sentence.", True)


def test_native_punctuation_requires_decoder_progress_not_repeated_feed():
    candidate, stable = whicc.update_native_punct_stability(
        None, False, "Version 21.",
        previous_generation=0, current_generation=1,
    )
    assert (candidate, stable) == ("Version 21.", False)

    # 100ms feed 尚未攒够下一个 560ms encoder chunk，不能把相同文本
    # 当成第二次 ASR 观察并提前在小数点处切句。
    candidate, stable = whicc.update_native_punct_stability(
        candidate, stable, "Version 21.",
        previous_generation=1, current_generation=1,
    )
    assert (candidate, stable) == ("Version 21.", False)

    candidate, stable = whicc.update_native_punct_stability(
        candidate, stable, "Version 21.4 is live",
        previous_generation=1, current_generation=2,
    )
    assert (candidate, stable) == (None, False)


@pytest.mark.parametrize("first, extended", [
    ("Version 21.", "Version 21.4 is live"),
    ("I met Dr.", "I met Dr. Smith"),
])
def test_ascii_period_prefix_is_not_a_stable_sentence_boundary(first, extended):
    assert whicc.update_punct_end_stability(first, extended) == (None, False)


def test_stable_ascii_period_does_not_add_a_second_pause_wait():
    candidate, stable = whicc.update_punct_end_stability(None, "I met Dr.")
    candidate, stable = whicc.update_punct_end_stability(candidate, "I met Dr.")
    assert stable is True
    assert whicc.punctuation_pause_ready(candidate, 0.0) is True
    assert whicc.punctuation_pause_ready(candidate, whicc.PUNCT_SUBMIT_SEC) is True
    assert whicc.punctuation_pause_ready("Question?", 0.0) is True


def test_submit_chunk_updates_outer_punctuation_state():
    tree = ast.parse(inspect.getsource(whicc.main))
    submit = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "submit_chunk")
    names = {name for node in submit.body if isinstance(node, ast.Nonlocal)
             for name in node.names}
    assert "prev_ended_with_punct" in names


def test_submit_chunk_failure_restores_audio_at_all_three_call_sites():
    tree = ast.parse(inspect.getsource(whicc.main))
    submit = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "submit_chunk")
    false_returns = [node for node in ast.walk(submit)
                     if isinstance(node, ast.Return)
                     and isinstance(node.value, ast.Constant)
                     and node.value.value is False]
    assert false_returns

    source = inspect.getsource(whicc.main)
    assert source.count("submitted = submit_chunk(") == 3
    assert source.count("restore_unsubmitted_audio(") >= 4  # 定义 + 3 个失败分支


def test_source_revision_is_stable_until_text_changes():
    revisions = whicc.SourceRevision("source-1")
    assert revisions.update("hello") == ("source-1", 1)
    assert revisions.update("hello") == ("source-1", 1)
    assert revisions.update("hello world") == ("source-1", 2)


def test_default_run_id_is_unique_per_backend_process():
    assert whicc.resolve_run_id("explicit", pid=10, mono_ns=20) == "explicit"
    assert whicc.resolve_run_id("", pid=10, mono_ns=20) == "run-10-20"
    assert whicc.resolve_run_id("", pid=11, mono_ns=20) != \
        whicc.resolve_run_id("", pid=10, mono_ns=20)


def test_probe_snapshot_uses_its_own_capture_end_timestamp():
    assert whicc.select_capture_end_mono_ns(
        uses_probe_snapshot=True, probe_capture_end=100, latest_capture_end=200
    ) == 100
    assert whicc.select_capture_end_mono_ns(
        uses_probe_snapshot=False, probe_capture_end=100, latest_capture_end=200
    ) == 200


def test_stale_probe_snapshot_preserves_all_later_pcm_as_next_remainder():
    probe = np.arange(1600, dtype=np.float32)
    tail = np.arange(1600, 2240, dtype=np.float32)
    current = np.concatenate([probe, tail])

    submitted, remainder, reused = whicc.split_probe_snapshot_audio(
        current, probe)

    assert reused is True
    assert np.array_equal(submitted, probe)
    assert np.array_equal(remainder, tail)
    assert np.array_equal(np.concatenate([submitted, remainder]), current)


def test_right_context_config_and_cli_priority(tmp_path):
    path = tmp_path / "lang_config.json"
    assert whicc.load_latency_config(str(path)) == {
        "nemotron_right_context": 3,
        "translation_priority_enabled": True,
        "probe_partial_enabled": True,
        "nemotron_native_streaming_enabled": True,
    }
    path.write_text(json.dumps({"nemotron_right_context": 3}), encoding="utf-8")
    assert whicc.load_latency_config(str(path))["nemotron_right_context"] == 3
    assert whicc.load_latency_config(str(path), 13)["nemotron_right_context"] == 13
    path.write_text(json.dumps({"nemotron_right_context": 99}), encoding="utf-8")
    assert whicc.load_latency_config(str(path))["nemotron_right_context"] == 3


def test_nemotron_chunk_mapping_and_commit_preserves_remainder():
    assert [native_chunk_frames(v) for v in (3, 6, 13)] == [4, 7, 14]
    stream = NemotronStream(None, right_context=6)
    stream.feed(np.arange(16000, dtype=np.float32))
    stream.commit_through(.75)
    assert len(stream.uncommitted_samples) == 4000


def test_native_commit_preserves_model_caches_and_offsets_visible_tokens():
    from mlx_audio.stt.models.nemo.alignment import AlignedToken

    stream = NemotronStream(None, right_context=6)
    stream.feed(np.arange(32000, dtype=np.float32))
    attn_cache = [object()]
    conv_cache = [object()]
    mel_cache = object()
    decoder_hidden = object()
    last_raw = object()
    stream._attn_cache = attn_cache
    stream._conv_cache = conv_cache
    stream._mel_cache = mel_cache
    stream._decoder_hidden = decoder_hidden
    stream._last_token = 42
    stream._last_raw_sample = last_raw
    stream._tokens = [AlignedToken(1, start=.2, duration=.2, text=" old")]

    stream.commit_through(1.0)
    # RNNT 右上下文可能在 commit 后才吐出时间戳落在切点前的 token；它在
    # 提交当时并不存在，必须按追加索引保留，不能再被动态时间过滤吞掉。
    stream._tokens.extend([
        AlignedToken(2, start=.9, duration=.05, text=" late"),
        AlignedToken(3, start=1.2, duration=.2, text=" new"),
    ])

    assert stream._attn_cache is attn_cache
    assert stream._conv_cache is conv_cache
    assert stream._mel_cache is mel_cache
    assert stream._decoder_hidden is decoder_hidden
    assert stream._last_token == 42
    assert stream._last_raw_sample is last_raw
    assert len(stream.uncommitted_samples) == 16000
    assert stream.audio_time_sec == pytest.approx(1.0)
    assert [segment["text"] for segment in stream.result_dict()["segments"]] == [
        "late", " new"]
    assert stream.result_dict()["segments"][0]["start"] == 0
    assert stream.result_dict()["segments"][1]["start"] == pytest.approx(.2)


def test_native_commit_drops_only_late_prefix_that_duplicates_committed_tokens():
    from mlx_audio.stt.models.nemo.alignment import AlignedToken

    stream = NemotronStream(None, right_context=6)
    stream.feed(np.arange(32000, dtype=np.float32))
    stream._tokens = [
        AlignedToken(1, start=.7, duration=.05, text=" there"),
        AlignedToken(2, start=.8, duration=.05, text="'"),
        AlignedToken(3, start=.8, duration=.05, text="s"),
    ]
    stream.commit_through(1.0)
    # RNNT 随后又吐出已提交尾部的副本；它们时间落在切点前且 token ID
    # 与已提交后缀完全相同，应只去掉副本，保留真正的新词。
    stream._tokens.extend([
        AlignedToken(2, start=.8, duration=.05, text="'"),
        AlignedToken(3, start=.8, duration=.05, text="s"),
        AlignedToken(4, start=1.1, duration=.05, text=" not"),
    ])

    assert stream.text == "not"
    assert [segment["text"] for segment in stream.result_dict()["segments"]] == [
        "not"
    ]


def test_native_commit_compacts_pcm_frontend_and_preemphasis_buffers():
    stream = NemotronStream(None, right_context=6)
    stream.feed(np.arange(32000, dtype=np.float32))
    stream.model = type("Model", (), {
        "preprocessor_config": type("Config", (), {
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
        })(),
    })()
    stream._preemphasized = np.arange(32000, dtype=np.float32)
    stream._frontend_mel = np.zeros((1, 100, 80), dtype=np.float32)
    stream._frontend_mel_frames = 100
    stream._mel_consumed = 80

    stream.commit_through(.75)

    assert len(stream._pcm) == 20000
    assert stream._committed_samples == 0
    assert stream._pcm_sample_origin == 12000
    assert stream._frontend_mel.shape[1] == 20
    assert stream._mel_frame_origin == 80
    assert stream._mel_consumed == 0
    # 下一 STFT 帧从 100*160-200=15800 开始，只保留该点之后的输入。
    assert stream._preemphasis_sample_origin == 15800
    assert len(stream._preemphasized) == 16200


def test_native_stream_close_releases_model_reference():
    stream = NemotronStream(None, right_context=6)
    stream.model = object()
    stream.close()
    assert stream.model is None
    assert len(stream.uncommitted_samples) == 0


def test_successful_batch_fallback_keeps_remainder_in_quality_recovery_mode():
    class TrackingStream:
        def __init__(self):
            self.reset_args = None

        def reset(self, language, right_context):
            self.reset_args = (language, right_context)

    stream = TrackingStream()
    remainder = np.arange(8, dtype=np.float32)

    fallback, protected, error = whicc.sync_native_after_submit(
        stream, commit_sec=1.0, language="en", right_context=6,
        backend_after_submit="nemotron", native_active_before_submit=False,
        current_fallback=True, raw_remainder=remainder,
    )

    assert error is None
    assert fallback is True
    assert stream.reset_args == ("en", 6)
    assert np.array_equal(protected, remainder)


def test_successful_batch_fallback_without_remainder_can_resume_native():
    class TrackingStream:
        def __init__(self):
            self.reset_args = None

        def reset(self, language, right_context):
            self.reset_args = (language, right_context)

    stream = TrackingStream()
    fallback, protected, error = whicc.sync_native_after_submit(
        stream, commit_sec=1.0, language="en", right_context=6,
        backend_after_submit="nemotron", native_active_before_submit=False,
        current_fallback=True, raw_remainder=np.array([], dtype=np.float32),
    )

    assert error is None
    assert fallback is False
    assert stream.reset_args == ("en", 6)
    assert protected.size == 0


def test_native_token_timestamps_split_at_punctuation_not_stream_tail():
    class Token:
        def __init__(self, text, start, duration):
            self.text = text
            self.start = start
            self.duration = duration

    segments = aligned_token_segments([
        Token(" Hello", 0.1, 0.4),
        Token(",", 0.5, 0.08),
        Token(" Ta", 0.58, 0.32),
    ])
    assert segments[0]["text"] == "Hello"
    assert whicc.find_audio_split_sec("Hello, Ta", 1.2, segments,
                                      punct_set={","}) == pytest.approx(0.58)


def test_native_punctuation_split_defers_when_next_word_shares_timestamp_frame():
    """同一 80ms 帧含句号和下一词时不能在词内切开。"""
    segments = [
        {"text": " place", "start": 6.40, "end": 6.56},
        {"text": ".", "start": 7.76, "end": 7.84},
        {"text": " O", "start": 7.76, "end": 7.84},
        {"text": "o", "start": 7.76, "end": 7.84},
        {"text": "h", "start": 7.84, "end": 7.92},
    ]

    assert whicc.find_audio_split_sec(
        " place. Ooh", 8.4, segments, punct_set={"."}
    ) == 0.0
