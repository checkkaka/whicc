import json
import sys

from tools.latency_report import main, report


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_end_to_end_drafts_use_ui_apply_time(tmp_path):
    _write_jsonl(tmp_path / "events.jsonl", [
        {
            "event_type": "partial",
            "source_key": "sentence-1",
            "speech_start_mono_ns": 1_000_000_000,
            "speech_end_mono_ns": 3_000_000_000,
            "event_mono_ns": 2_000_000_000,
        },
    ])
    _write_jsonl(tmp_path / "translation_events.jsonl", [
        {
            "event_type": "translation_partial",
            "source_key": "sentence-1",
            "event_mono_ns": 2_500_000_000,
        },
    ])
    _write_jsonl(tmp_path / "ui_metrics.jsonl", [
        {
            "event_type": "source_draft",
            "source_key": "sentence-1",
            "ui_apply_mono_ns": 4_000_000_000,
        },
        {
            "event_type": "translation_draft",
            "source_key": "sentence-1",
            "ui_apply_mono_ns": 5_000_000_000,
        },
        {
            "event_type": "translation_final",
            "source_key": "sentence-1",
            "ui_apply_mono_ns": 6_000_000_000,
        },
    ])

    metrics = report(tmp_path)

    assert metrics["开口→首个原文草稿"] == [3.0]
    assert metrics["开口→首个翻译草稿"] == [4.0]
    assert metrics["句末→稳定译文UI应用"] == [3.0]


def test_missing_ui_drafts_are_reported_unavailable(tmp_path, monkeypatch, capsys):
    _write_jsonl(tmp_path / "events.jsonl", [
        {
            "event_type": "partial",
            "source_key": "sentence-1",
            "speech_start_mono_ns": 1_000_000_000,
            "event_mono_ns": 2_000_000_000,
        },
    ])
    _write_jsonl(tmp_path / "translation_events.jsonl", [
        {
            "event_type": "translation_partial",
            "source_key": "sentence-1",
            "event_mono_ns": 2_500_000_000,
        },
    ])
    monkeypatch.setattr(sys, "argv", ["latency_report.py", "--out-dir", str(tmp_path)])

    main()

    output = capsys.readouterr().out
    assert "开口→首个原文草稿: unavailable（缺少 UI 应用记录）" in output
    assert "开口→首个翻译草稿: unavailable（缺少 UI 应用记录）" in output


def test_translation_stages_separate_final_and_partial(tmp_path, monkeypatch, capsys):
    partial_stamps = {
        "translation_enqueue_mono_ns": 1_000_000_000,
        "http_request_start_mono_ns": 3_000_000_000,
        "first_token_mono_ns": 4_000_000_000,
        "translation_complete_mono_ns": 6_000_000_000,
    }
    _write_jsonl(tmp_path / "translation_events.jsonl", [
        {"event_type": "translation_metrics", **partial_stamps},
        # 草稿输出会重复携带时间戳，不应重复计入阶段指标。
        {"event_type": "translation_partial", **partial_stamps},
        {
            "event_type": "translation_final",
            "translation_enqueue_mono_ns": 10_000_000_000,
            "http_request_start_mono_ns": 10_100_000_000,
            "first_token_mono_ns": 20_000_000_000,
            "translation_complete_mono_ns": 21_000_000_000,
        },
    ])
    monkeypatch.setattr(sys, "argv", ["latency_report.py", "--out-dir", str(tmp_path)])

    metrics = report(tmp_path)
    main()

    assert metrics["final翻译入队→HTTP开始"] == [0.1]
    assert metrics["partial翻译入队→HTTP开始"] == [2.0]
    assert metrics["partial HTTP开始→首token"] == [1.0]
    assert metrics["partial 首token→翻译完成"] == [2.0]
    assert "翻译入队→HTTP开始" not in metrics
    assert "final翻译入队→HTTP开始: n=1 P50=0.100s P95=0.100s" in capsys.readouterr().out


def test_stable_tail_stage_decomposition_joins_asr_translation_and_ui(tmp_path):
    _write_jsonl(tmp_path / "events.jsonl", [{
        "event_type": "final",
        "source_key": "sentence-1",
        "speech_start_mono_ns": 1_000_000_000,
        "speech_end_mono_ns": 3_000_000_000,
        "event_mono_ns": 4_000_000_000,
    }])
    _write_jsonl(tmp_path / "translation_events.jsonl", [{
        "event_type": "translation_final",
        "source_key": "sentence-1",
        "translated_full_text": "译文",
        "translation_enqueue_mono_ns": 4_100_000_000,
        "http_request_start_mono_ns": 4_200_000_000,
        "translation_complete_mono_ns": 5_000_000_000,
    }])
    _write_jsonl(tmp_path / "ui_metrics.jsonl", [{
        "event_type": "translation_final",
        "source_key": "sentence-1",
        "ui_apply_mono_ns": 5_050_000_000,
    }])

    metrics = report(tmp_path)

    assert metrics["句末→ASR final事件"] == [1.0]
    assert metrics["ASR final事件→翻译入队"] == [0.1]
    assert metrics["final HTTP开始→翻译完成"] == [0.8]
    assert metrics["翻译完成→UI应用"] == [0.05]


def test_forced_max_chunk_is_not_counted_as_sentence_end_latency(tmp_path):
    _write_jsonl(tmp_path / "events.jsonl", [
        {
            "event_type": "final",
            "source_key": "natural-end",
            "submit_reason": "silence",
            "speech_end_mono_ns": 3_000_000_000,
            "event_mono_ns": 4_000_000_000,
        },
        {
            "event_type": "final",
            "source_key": "forced-cut",
            "submit_reason": "max_chunk",
            "speech_end_mono_ns": 5_000_000_000,
            "event_mono_ns": 7_000_000_000,
        },
    ])
    _write_jsonl(tmp_path / "translation_events.jsonl", [
        {
            "event_type": "translation_final",
            "source_key": "natural-end",
            "translated_full_text": "自然句末",
        },
        {
            "event_type": "translation_final",
            "source_key": "forced-cut",
            "translated_full_text": "强制切块",
        },
    ])
    _write_jsonl(tmp_path / "ui_metrics.jsonl", [
        {
            "event_type": "translation_final",
            "source_key": "natural-end",
            "ui_apply_mono_ns": 5_000_000_000,
        },
        {
            "event_type": "translation_final",
            "source_key": "forced-cut",
            "ui_apply_mono_ns": 8_000_000_000,
        },
    ])

    metrics = report(tmp_path)

    assert metrics["句末→稳定译文UI应用"] == [2.0]
    assert metrics["强制切块→稳定译文UI应用"] == [3.0]
    assert metrics["句末→ASR final事件"] == [1.0]
    assert metrics["强制切块→ASR final事件"] == [2.0]


def test_failed_source_only_finals_do_not_improve_stable_latency(
        tmp_path, monkeypatch, capsys):
    _write_jsonl(tmp_path / "events.jsonl", [
        {
            "event_type": "final",
            "source_key": key,
            "speech_end_mono_ns": 3_000_000_000,
        }
        for key in ("success", "failed-by-reason", "failed-by-mode")
    ])
    _write_jsonl(tmp_path / "translation_events.jsonl", [
        {
            "event_type": "translation_final",
            "source_key": "success",
            "source_update_mode": "full_translate",
            "fallback_reason": "",
        },
        {
            "event_type": "translation_final",
            "source_key": "failed-by-reason",
            "source_update_mode": "full_translate",
            "fallback_reason": "translation_failed_source_only",
        },
        {
            "event_type": "translation_final",
            "source_key": "failed-by-mode",
            "source_update_mode": "translation_failed_source_only",
            "fallback_reason": "",
        },
    ])
    _write_jsonl(tmp_path / "ui_metrics.jsonl", [
        {
            "event_type": "translation_final",
            "source_key": "success",
            "ui_apply_mono_ns": 5_000_000_000,
        },
        {
            "event_type": "translation_final",
            "source_key": "failed-by-reason",
            "ui_apply_mono_ns": 3_100_000_000,
        },
        {
            "event_type": "translation_final",
            "source_key": "failed-by-mode",
            "ui_apply_mono_ns": 3_200_000_000,
        },
    ])
    monkeypatch.setattr(sys, "argv", ["latency_report.py", "--out-dir", str(tmp_path)])

    metrics = report(tmp_path)
    main()

    assert metrics["句末→稳定译文UI应用"] == [2.0]
    output = capsys.readouterr().out
    assert "翻译失败数: 2（translation_failed_source_only）" in output
    assert "翻译失败率: 66.67%（2/3，分母=已关联 ASR final）" in output


def test_unavailable_passthrough_is_failure_and_startup_final_not_denominator(
        tmp_path, monkeypatch, capsys):
    _write_jsonl(tmp_path / "events.jsonl", [
        {
            "event_type": "final", "accepted": True,
            "source_key": "success", "speech_end_mono_ns": 1_000_000_000,
        },
        {
            "event_type": "final", "accepted": True,
            "source_key": "unavailable", "speech_end_mono_ns": 1_000_000_000,
        },
    ])
    _write_jsonl(tmp_path / "translation_events.jsonl", [
        {
            "event_type": "translation_final", "source_key": "success",
            "translated_full_text": "成功", "source_update_mode": "full_translate",
        },
        {
            "event_type": "translation_final", "source_key": "unavailable",
            "translated_full_text": "",
            "source_update_mode": "no_translator_passthrough",
            "fallback_reason": "translator_unavailable",
        },
        {
            "event_type": "translation_final", "source_key": "init-123",
            "translated_full_text": "whicc 正在聆听",
            "source_update_mode": "reset_full",
        },
    ])
    _write_jsonl(tmp_path / "ui_metrics.jsonl", [
        {
            "event_type": "translation_final", "source_key": "success",
            "ui_apply_mono_ns": 2_000_000_000,
        },
        {
            "event_type": "translation_final", "source_key": "unavailable",
            "ui_apply_mono_ns": 1_100_000_000,
        },
    ])
    monkeypatch.setattr(sys, "argv", ["latency_report.py", "--out-dir", str(tmp_path)])

    metrics = report(tmp_path)
    main()

    assert metrics["句末→稳定译文UI应用"] == [1.0]
    output = capsys.readouterr().out
    assert "翻译失败数: 1（translator_unavailable）" in output
    assert "翻译失败率: 50.00%（1/2，分母=已关联 ASR final）" in output
