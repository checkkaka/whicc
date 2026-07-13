"""阶段一/五：度量、切点、source_key/revision、配置默认值。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def test_audio_chunk_carries_capture_timestamp():
    import audio

    samples = np.zeros(1600, dtype=np.float32)
    before = time.monotonic_ns()
    chunk = audio.AudioChunk.from_samples(samples)
    after = time.monotonic_ns()
    assert isinstance(chunk, audio.AudioChunk)
    assert chunk.samples is samples or np.array_equal(chunk.samples, samples)
    assert before <= chunk.captured_mono_ns <= after


def test_offer_wraps_ndarray_as_audio_chunk():
    import audio

    src = audio.MicSource()
    samples = np.ones(100, dtype=np.float32)
    src._offer(samples)
    item = src.queue.get_nowait()
    assert isinstance(item, audio.AudioChunk)
    assert np.array_equal(item.samples, samples)
    assert item.captured_mono_ns > 0


def test_drain_audio_queue_unwraps_chunks():
    import queue
    import audio
    import whicc

    q = queue.Queue()
    a = np.ones(10, dtype=np.float32)
    b = np.ones(20, dtype=np.float32) * 2
    q.put(audio.AudioChunk.from_samples(a))
    q.put(audio.AudioChunk.from_samples(b))
    out = whicc.drain_audio_queue(q, 0.01)
    assert len(out) == 2
    assert np.array_equal(out[0], a)
    assert np.array_equal(out[1], b)


def test_nemotron_sentences_to_segments():
    import whicc

    class Sent:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    segs = whicc.sentences_to_segments([Sent("Hello ", 0.0, 0.5), Sent("world.", 0.5, 1.2)])
    assert segs == [
        {"text": "Hello ", "start": 0.0, "end": 0.5},
        {"text": "world.", "start": 0.5, "end": 1.2},
    ]


def test_find_audio_split_uses_segments_when_present():
    import whicc

    text = "Hello world. More"
    segments = [
        {"text": "Hello world.", "start": 0.0, "end": 1.1},
        {"text": " More", "start": 1.1, "end": 1.8},
    ]
    split_sec, method = whicc.find_audio_split_sec(text, 1.8, segments)
    assert method == "segments"
    assert split_sec == pytest.approx(1.1)


def test_find_audio_split_falls_back_to_char_ratio():
    import whicc

    text = "Hello world. More"
    split_sec, method = whicc.find_audio_split_sec(text, 10.0, None)
    assert method == "char_ratio"
    assert split_sec > 0


def test_source_revision_is_stable_until_text_changes():
    import whicc

    revisions = whicc.SourceRevision("source-1")
    assert revisions.update("hello") == ("source-1", 1)
    assert revisions.update("hello") == ("source-1", 1)
    assert revisions.update("hello world") == ("source-1", 2)


def test_load_latency_config_defaults_and_cli_override(tmp_path):
    import whicc

    path = tmp_path / "lang_config.json"
    path.write_text(json.dumps({"nemotron_right_context": 3}), encoding="utf-8")
    assert whicc.load_latency_config(str(path))["nemotron_right_context"] == 3
    assert whicc.load_latency_config(str(path), 13)["nemotron_right_context"] == 13
    path.write_text(json.dumps({"nemotron_right_context": 99}), encoding="utf-8")
    assert whicc.load_latency_config(str(path))["nemotron_right_context"] == 6
    assert whicc.load_latency_config(str(tmp_path / "missing.json"))["nemotron_right_context"] == 6


def test_event_logger_writes_mono_timestamps(tmp_path):
    import whicc

    path = tmp_path / "events.jsonl"
    logger = whicc.EventLogger(str(path))
    logger.log_partial(0, 1, 0.0, 1.0, "hi", source_key="k1", revision=1, is_probe=True)
    logger.close()
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["source_key"] == "k1"
    assert row["revision"] == 1
    assert row["is_probe"] is True
    assert row["event_mono_ns"] > 0
    assert row["event_wall_ms"] > 0


def test_nemotron_stream_chunk_binding_and_feed():
    import numpy as np
    from nemotron_stream import NemotronStream, RIGHT_CONTEXT_CHUNK_FRAMES

    assert RIGHT_CONTEXT_CHUNK_FRAMES[3] == 4
    assert RIGHT_CONTEXT_CHUNK_FRAMES[6] == 7
    assert RIGHT_CONTEXT_CHUNK_FRAMES[13] == 14

    calls = []

    def fake_generate(pcm, language="auto", right_context=6):
        calls.append((len(pcm), right_context))
        return {"text": f"words-{len(calls)}"}

    stream = NemotronStream(generate_fn=fake_generate, right_context=6)
    # 不足一个原生 chunk：不触发
    assert stream.feed(np.zeros(1000, dtype=np.float32)) is None
    # 凑够 560ms ≈ 8960 samples
    snap = stream.feed(np.zeros(9000, dtype=np.float32))
    assert snap is not None
    assert snap.changed is True
    assert snap.text.startswith("words-")
    assert calls and calls[0][1] == 6


def test_nemotron_stream_reset_changes_chunk_size():
    from nemotron_stream import NemotronStream

    stream = NemotronStream(right_context=13)
    assert stream.chunk_frames == 14
    stream.reset(right_context=3)
    assert stream.chunk_frames == 4
