#!/usr/bin/env python3
"""按 source_key 关联 ASR / 翻译 / UI 事件,输出延迟分位数报表。

唯一验收口径:
  - 开口 → 首个原文/翻译草稿
  - 句末 → translation_final 在 UI 应用完成

用法:
  python3 tools/latency_report.py \\
    --events /tmp/whicc-out/events.jsonl \\
    --translations /tmp/whicc-out/translation_events.jsonl \\
    --ui /tmp/whicc-out/ui_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[p - 1]


def _ms(ns_delta: float | None) -> float | None:
    if ns_delta is None:
        return None
    return ns_delta / 1_000_000.0


def _fmt(ms: float | None) -> str:
    if ms is None:
        return "-"
    return f"{ms:.1f}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description="Whicc latency report")
    parser.add_argument("--events", required=True)
    parser.add_argument("--translations", required=True)
    parser.add_argument("--ui", default="")
    args = parser.parse_args()

    events = _load_jsonl(Path(args.events))
    translations = _load_jsonl(Path(args.translations))
    ui_rows = _load_jsonl(Path(args.ui)) if args.ui else []

    by_key: dict[str, dict] = defaultdict(dict)
    for ev in events:
        key = ev.get("source_key") or f"{ev.get('seg_start')}-{ev.get('seg_end')}-{ev.get('audio_end_sec')}"
        et = ev.get("event_type")
        if et == "partial" and "first_partial" not in by_key[key]:
            by_key[key]["first_partial"] = ev
        if et == "final":
            by_key[key]["final"] = ev
        if et == "partial" and ev.get("is_probe") and "first_probe" not in by_key[key]:
            by_key[key]["first_probe"] = ev

    for ev in translations:
        key = ev.get("source_key") or ""
        if not key:
            continue
        et = ev.get("event_type")
        if et == "translation_partial" and "first_trans_partial" not in by_key[key]:
            by_key[key]["first_trans_partial"] = ev
        if et == "translation_final":
            by_key[key]["trans_final"] = ev

    for ev in ui_rows:
        key = ev.get("source_key") or ""
        if not key:
            continue
        kind = ev.get("metric")
        if kind and kind not in by_key[key]:
            by_key[key][kind] = ev

    speech_to_draft: list[float] = []
    speech_to_trans_draft: list[float] = []
    end_to_ui_final: list[float] = []
    queue_wait: list[float] = []
    ttft: list[float] = []
    translate_ms: list[float] = []

    for key, bag in by_key.items():
        probe = bag.get("first_probe") or bag.get("first_partial")
        final = bag.get("final")
        tpartial = bag.get("first_trans_partial")
        tfinal = bag.get("trans_final")
        ui_final = bag.get("ui_final_applied")

        if probe and probe.get("speech_start_mono_ns") and probe.get("event_mono_ns"):
            speech_to_draft.append(probe["event_mono_ns"] - probe["speech_start_mono_ns"])
        if tpartial and probe and probe.get("speech_start_mono_ns") and tpartial.get("event_mono_ns"):
            speech_to_trans_draft.append(tpartial["event_mono_ns"] - probe["speech_start_mono_ns"])
        if ui_final and final and final.get("speech_end_mono_ns") and ui_final.get("event_mono_ns"):
            end_to_ui_final.append(ui_final["event_mono_ns"] - final["speech_end_mono_ns"])
        if tfinal:
            if tfinal.get("translation_enqueued_mono_ns") and tfinal.get("translation_request_started_mono_ns"):
                queue_wait.append(
                    tfinal["translation_request_started_mono_ns"]
                    - tfinal["translation_enqueued_mono_ns"]
                )
            if tfinal.get("translation_request_started_mono_ns") and tfinal.get("translation_first_token_mono_ns"):
                ttft.append(
                    tfinal["translation_first_token_mono_ns"]
                    - tfinal["translation_request_started_mono_ns"]
                )
            if tfinal.get("translate_ms") is not None:
                translate_ms.append(float(tfinal["translate_ms"]) * 1_000_000.0)

    def report(name: str, values: list[float]) -> None:
        p50 = _ms(_pct(values, 50))
        p95 = _ms(_pct(values, 95))
        print(f"{name}: n={len(values)}  P50={_fmt(p50)}  P95={_fmt(p95)}")

    print("=== Whicc latency report (source_key aligned) ===")
    report("开口→首个原文草稿", speech_to_draft)
    report("开口→首个翻译草稿", speech_to_trans_draft)
    report("句末→UI稳定字幕", end_to_ui_final)
    report("翻译队列等待", queue_wait)
    report("HTTP TTFT", ttft)
    report("translate_ms", translate_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
