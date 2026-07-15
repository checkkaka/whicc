#!/usr/bin/env python3
"""关联 ASR、翻译与 UI JSONL，以 UI 实际应用时间输出端到端延迟。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FAILED_SOURCE_ONLY = "translation_failed_source_only"
TRANSLATOR_UNAVAILABLE = "translator_unavailable"


def untranslated_final_reason(event: dict) -> str | None:
    """返回“仅原文 final”的失败类别；正常稳定译文返回 None。"""
    if event.get("event_type") != "translation_final":
        return None
    fallback = event.get("fallback_reason")
    mode = event.get("source_update_mode")
    if fallback == TRANSLATOR_UNAVAILABLE or mode == "no_translator_passthrough":
        return TRANSLATOR_UNAVAILABLE
    if fallback == FAILED_SOURCE_ONLY or mode == FAILED_SOURCE_ONLY:
        return FAILED_SOURCE_ONLY
    # 防御未知的仅原文路径；旧事件可能没有 translated_full_text 字段，
    # 不能把字段缺失误判为失败。
    if ("translated_full_text" in event
            and not str(event.get("translated_full_text") or "").strip()
            and str(event.get("source_text") or "").strip()):
        return "empty_translation"
    return None


def is_failed_source_only_final(event: dict) -> bool:
    """兼容旧调用名：任何没有稳定译文的 final 都算失败。"""
    return untranslated_final_reason(event) is not None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def report(out_dir: Path) -> dict[str, list[float]]:
    asr = read_jsonl(out_dir / "events.jsonl")
    translations = read_jsonl(out_dir / "translation_events.jsonl")
    ui = read_jsonl(out_dir / "ui_metrics.jsonl")
    untranslated_final_keys = {
        event.get("source_key")
        for event in translations
        if is_failed_source_only_final(event) and event.get("source_key")
    }
    asr_finals = {
        event.get("source_key"): event
        for event in asr
        if event.get("event_type") == "final" and event.get("source_key")
    }
    translation_finals = {
        event.get("source_key"): event
        for event in translations
        if (event.get("event_type") == "translation_final"
            and event.get("source_key")
            and event.get("source_key") not in untranslated_final_keys)
    }

    speech = {}
    for event in asr:
        key = event.get("source_key")
        if key:
            current = speech.setdefault(key, {})
            if event.get("speech_start_mono_ns"):
                current.setdefault("start", event["speech_start_mono_ns"])
            if event.get("speech_end_mono_ns"):
                current["end"] = event["speech_end_mono_ns"]

    first_asr = {}
    first_translation = {}
    for event in ui:
        key = event.get("source_key")
        applied = event.get("ui_apply_mono_ns", 0)
        if event.get("event_type") == "source_draft" and key and applied:
            first_asr.setdefault(key, applied)
        if event.get("event_type") == "translation_draft" and key and applied:
            first_translation.setdefault(key, applied)

    metrics = {
        "开口→首个原文草稿": [],
        "开口→首个翻译草稿": [],
        "句末→稳定译文UI应用": [],
        "强制切块→稳定译文UI应用": [],
        "句末→ASR final事件": [],
        "强制切块→ASR final事件": [],
        "ASR final事件→翻译入队": [],
        "final翻译入队→HTTP开始": [],
        "final HTTP开始→翻译完成": [],
        "翻译完成→UI应用": [],
        "partial翻译入队→HTTP开始": [],
        "partial HTTP开始→首token": [],
        "partial 首token→翻译完成": [],
    }
    for key, stamp in speech.items():
        start = stamp.get("start", 0)
        if start and first_asr.get(key):
            metrics["开口→首个原文草稿"].append((first_asr[key] - start) / 1e9)
        if start and first_translation.get(key):
            metrics["开口→首个翻译草稿"].append((first_translation[key] - start) / 1e9)
    for key, event in asr_finals.items():
        speech_end = event.get("speech_end_mono_ns", 0)
        final_event = event.get("event_mono_ns", 0)
        if speech_end and final_event:
            label = ("强制切块→ASR final事件"
                     if event.get("submit_reason") in {"max_chunk", "soft_max_split"}
                     else "句末→ASR final事件")
            metrics[label].append(
                (final_event - speech_end) / 1e9)
    for event in ui:
        if event.get("event_type") != "translation_final":
            continue
        if event.get("source_key") in untranslated_final_keys:
            continue
        end = speech.get(event.get("source_key"), {}).get("end", 0)
        if end and event.get("ui_apply_mono_ns"):
            final = asr_finals.get(event.get("source_key"), {})
            label = ("强制切块→稳定译文UI应用"
                     if final.get("submit_reason") in {"max_chunk", "soft_max_split"}
                     else "句末→稳定译文UI应用")
            metrics[label].append(
                (event["ui_apply_mono_ns"] - end) / 1e9)
        translation = translation_finals.get(event.get("source_key"), {})
        complete = translation.get("translation_complete_mono_ns", 0)
        if complete and event.get("ui_apply_mono_ns"):
            metrics["翻译完成→UI应用"].append(
                (event["ui_apply_mono_ns"] - complete) / 1e9)
    for event in translations:
        enqueue = event.get("translation_enqueue_mono_ns", 0)
        request = event.get("http_request_start_mono_ns", 0)
        if event.get("event_type") == "translation_final":
            if event.get("source_key") in untranslated_final_keys:
                continue
            asr_final = asr_finals.get(event.get("source_key"), {})
            asr_event = asr_final.get("event_mono_ns", 0)
            if asr_event and enqueue:
                metrics["ASR final事件→翻译入队"].append(
                    (enqueue - asr_event) / 1e9)
            if enqueue and request:
                metrics["final翻译入队→HTTP开始"].append((request - enqueue) / 1e9)
            complete = event.get("translation_complete_mono_ns", 0)
            if request and complete:
                metrics["final HTTP开始→翻译完成"].append(
                    (complete - request) / 1e9)
            continue
        # translation_partial 会重复携带时间戳；translation_metrics 才是
        # 每次 partial 请求唯一的完整阶段记录。
        if event.get("event_type") != "translation_metrics":
            continue
        first = event.get("first_token_mono_ns", 0)
        complete = event.get("translation_complete_mono_ns", 0)
        if enqueue and request:
            metrics["partial翻译入队→HTTP开始"].append((request - enqueue) / 1e9)
        if request and first:
            metrics["partial HTTP开始→首token"].append((first - request) / 1e9)
        if first and complete:
            metrics["partial 首token→翻译完成"].append((complete - first) / 1e9)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/whicc-out")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    for name, values in report(out_dir).items():
        p50, p95 = percentile(values, .5), percentile(values, .95)
        if p50 is None:
            if name in ("开口→首个原文草稿", "开口→首个翻译草稿", "句末→稳定译文UI应用"):
                print(f"{name}: unavailable（缺少 UI 应用记录）")
            else:
                print(f"{name}: 无数据")
        else:
            print(f"{name}: n={len(values)} P50={p50:.3f}s P95={p95:.3f}s")
    asr_final_keys = {
        event.get("source_key")
        for event in read_jsonl(out_dir / "events.jsonl")
        if event.get("event_type") == "final" and event.get("source_key")
    }
    # 只统计能关联到真实 ASR final 的译文；init/notice/monitor 等 UI
    # 启动事件不属于翻译样本，不能稀释失败率。
    finals = [
        event for event in read_jsonl(out_dir / "translation_events.jsonl")
        if (event.get("event_type") == "translation_final"
            and event.get("source_key") in asr_final_keys)
    ]
    reasons: dict[str, int] = {}
    for event in finals:
        reason = untranslated_final_reason(event)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    failed = sum(reasons.values())
    if not reasons:
        detail = "无"
    elif len(reasons) == 1:
        detail = next(iter(reasons))
    else:
        detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(reasons.items()))
    print(f"翻译失败数: {failed}（{detail}）")
    if finals:
        print(
            f"翻译失败率: {failed / len(finals):.2%}"
            f"（{failed}/{len(finals)}，分母=已关联 ASR final）"
        )
    else:
        print("翻译失败率: unavailable（0/0，分母=已关联 ASR final）")


if __name__ == "__main__":
    main()
