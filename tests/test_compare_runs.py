import json
import subprocess
import sys
from pathlib import Path

from tools.compare_runs import adjacent_repeated_fragments, normalize_english


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "compare_runs.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_compare(reference_dir: Path, candidate_dir: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(TOOL), str(reference_dir), str(candidate_dir)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_compare_final_coverage_distance_missing_span_and_counts(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()

    _write_jsonl(reference / "events.jsonl", [
        {"event_type": "partial", "text": "ignored partial"},
        {"event_type": "final", "text": "Alpha beta gamma delta"},
        {"event_type": "error", "message": "reference ASR error"},
    ])
    _write_jsonl(candidate / "events.jsonl", [
        {"event_type": "final", "text": "Alpha gamma delta"},
        {"event_type": "error", "message": "candidate ASR error 1"},
        {"event_type": "error", "message": "candidate ASR error 2"},
    ])
    _write_jsonl(reference / "translation_events.jsonl", [
        {
            "event_type": "translation_final",
            "source_text": "One two three four five six",
            "translated_full_text": "一二三四五六",
        },
        {"event_type": "translation_error", "source_text": "ignored"},
    ])
    _write_jsonl(candidate / "translation_events.jsonl", [
        {
            "event_type": "translation_final",
            "source_text": "One two six",
            "translated_full_text": "一二六",
        },
    ])

    report = _run_compare(reference, candidate)

    assert report["asr_final"]["reference"]["final_count"] == 1
    assert report["asr_final"]["reference"]["error_count"] == 1
    assert report["asr_final"]["candidate"]["final_count"] == 1
    assert report["asr_final"]["candidate"]["error_count"] == 2
    asr_delta = report["asr_final"]["candidate_vs_reference"]
    assert asr_delta["english_coverage"] == 0.75
    assert asr_delta["word_edit_distance"] == 1
    assert asr_delta["normalized_word_edit_distance"] == 0.25
    assert asr_delta["longest_missing_span"] == {
        "start_token": 1,
        "token_count": 1,
        "text": "beta",
    }

    assert report["translation_final"]["reference"]["final_count"] == 1
    assert report["translation_final"]["reference"]["error_count"] == 1
    assert report["translation_final"]["candidate"]["final_count"] == 1
    assert report["translation_final"]["candidate"]["error_count"] == 0
    translation_delta = report["translation_final"]["candidate_vs_reference"]
    assert translation_delta["english_coverage"] == 0.5
    assert translation_delta["word_edit_distance"] == 3
    assert translation_delta["normalized_word_edit_distance"] == 0.5
    assert translation_delta["longest_missing_span"] == {
        "start_token": 2,
        "token_count": 3,
        "text": "three four five",
    }


def test_normalization_and_adjacent_repeat_detection_are_word_based():
    tokens = normalize_english(
        "We NEED, we need stable output—output; don’t stop at V2."
    )

    assert tokens == [
        "we", "need", "we", "need", "stable", "output", "output",
        "don't", "stop", "at", "v2",
    ]
    assert adjacent_repeated_fragments(tokens) == [
        {
            "start_token": 0,
            "token_count": 2,
            "repeat_count": 2,
            "text": "we need",
        },
        {
            "start_token": 5,
            "token_count": 1,
            "repeat_count": 2,
            "text": "output",
        },
    ]


def test_startup_finals_are_excluded_but_all_error_events_are_counted(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for run_dir in (reference, candidate):
        _write_jsonl(run_dir / "events.jsonl", [
            {"event_type": "final", "source_key": "sentence-1", "text": "hello"},
        ])
        _write_jsonl(run_dir / "translation_events.jsonl", [
            {
                "event_type": "translation_final",
                "source_key": "init-1",
                "source_text": "whicc is listening",
            },
            {
                "event_type": "translation_final",
                "source_key": "monitor-1",
                "source_text": "backend restarted",
            },
            {
                "event_type": "translation_final",
                "source_key": "sentence-1",
                "source_text": "hello",
            },
            # 初始化失败可能还没有 source_key/source_text，也必须计入错误数量。
            {"event_type": "translation_error", "message": "translator unavailable"},
        ])

    report = _run_compare(reference, candidate)

    assert report["translation_final"]["reference"]["final_count"] == 1
    assert report["translation_final"]["reference"]["english_token_count"] == 1
    assert report["translation_final"]["reference"]["error_count"] == 1


def test_malformed_jsonl_is_ignored_and_input_files_are_not_modified(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    reference_events = reference / "events.jsonl"
    candidate_events = candidate / "events.jsonl"
    reference_events.write_text(
        '{"event_type":"final","text":"alpha beta"}\n{"unfinished":',
        encoding="utf-8",
    )
    candidate_events.write_text(
        '{"event_type":"final","text":"alpha"}\nnot-json\n',
        encoding="utf-8",
    )
    before = {
        reference_events: reference_events.read_bytes(),
        candidate_events: candidate_events.read_bytes(),
    }

    report = _run_compare(reference, candidate)

    assert report["asr_final"]["candidate_vs_reference"]["english_coverage"] == 0.5
    assert {path: path.read_bytes() for path in before} == before
