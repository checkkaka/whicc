import json
import subprocess
import sys
from pathlib import Path

from tools.youtube_regression_report import (
    build_report,
    documented_parameter_matrix,
    parse_vtt,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "youtube_regression_report.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_parse_vtt_merges_multiline_cues_and_tracks_tokens(tmp_path):
    vtt = tmp_path / "sample.vtt"
    vtt.write_text("""WEBVTT

00:00:01.000 --> 00:00:02.000
Can AI
be trusted?

00:00:03.000 --> 00:00:04.500
We had voice.
""", encoding="utf-8")

    cues = parse_vtt(vtt)

    assert [cue.text for cue in cues] == ["Can AI be trusted?", "We had voice."]
    assert [(cue.token_start, cue.token_end) for cue in cues] == [(0, 4), (4, 7)]


def test_report_flags_missing_caption_repetition_and_long_final(tmp_path):
    vtt = tmp_path / "sample.vtt"
    vtt.write_text("""WEBVTT

00:00:01.000 --> 00:00:02.000
Alpha beta.

00:00:03.000 --> 00:00:04.000
Gamma delta.
""", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "events.jsonl", [
        {
            "event_type": "final",
            "text": "Alpha beta alpha beta",
            "chunk_sec": 11.0,
        },
    ])

    report = build_report(vtt, run_dir, max_sentence_sec=10.56)

    assert report["asr"]["english_coverage"] == 0.5
    assert report["asr"]["missing_caption_blocks"] == [{
        "start_sec": 3.0,
        "end_sec": 4.0,
        "text": "Gamma delta.",
    }]
    assert report["segmentation"]["too_long_finals"][0]["duration_sec"] == 11.0
    assert report["segmentation"]["adjacent_repeated_fragments"][0]["text"] == "alpha beta"
    assert report["gates"] == {
        "no_complete_caption_missing": False,
        "no_too_long_final": False,
        "no_adjacent_repetition": False,
    }


def test_documented_matrix_is_small_and_keeps_56_0_hidden():
    matrix = documented_parameter_matrix()

    assert [item["att_context_size"] for item in matrix["nemotron_main"]] == [
        [56, 3], [56, 6], [56, 13],
    ]
    assert matrix["nemotron_hidden_exploration"][0]["att_context_size"] == [56, 0]
    assert matrix["apple_asr"]["silence_submit_sec"] == [1.2, 1.0]
    assert matrix["apple_asr"]["punct_end_min_chunk_sec_en"] == [3.0, 2.4, 2.0]


def test_cli_outputs_json_report(tmp_path):
    vtt = tmp_path / "sample.vtt"
    vtt.write_text("""WEBVTT

00:00:01.000 --> 00:00:02.000
Alpha beta.
""", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "events.jsonl", [
        {"event_type": "final", "text": "Alpha beta.", "chunk_sec": 1.0},
    ])

    result = subprocess.run(
        [sys.executable, str(TOOL), "--reference-vtt", str(vtt),
         "--run-dir", str(run_dir)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["gates"]["no_complete_caption_missing"] is True
