#!/usr/bin/env python3
"""只读比较同一录音的两个 Whicc 运行目录。

英文指标分别基于 ASR ``final.text`` 与业务翻译
``translation_final.source_text``。翻译启动提示和后端监控通知不属于录音，
会在存在 ``source_key`` 时通过 ASR final 关联过滤。
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence


_ENGLISH_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_NON_UTTERANCE_KEY_PREFIXES = ("init-", "monitor-", "idle-", "notice-")


def read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL；忽略被并发追加截断或本身损坏的单行。"""
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def normalize_english(text: str) -> list[str]:
    """把英文规范化为小写词序列，保留单词内部撇号。"""
    normalized = unicodedata.normalize("NFKC", str(text)).replace("’", "'")
    return _ENGLISH_TOKEN_RE.findall(normalized.casefold())


def _levenshtein(reference: Sequence[str], candidate: Sequence[str]) -> int:
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for row_index, reference_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, candidate_token in enumerate(candidate, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1]
                + (reference_token != candidate_token),
            ))
        previous = current
    return previous[-1]


def _lcs_reference_matches(
    reference: Sequence[str], candidate: Sequence[str],
) -> set[int]:
    """返回一条确定性 LCS 中已被候选覆盖的基准词下标。"""
    reference_count = len(reference)
    candidate_count = len(candidate)
    lengths = [
        [0] * (candidate_count + 1)
        for _ in range(reference_count + 1)
    ]
    for reference_index in range(reference_count - 1, -1, -1):
        for candidate_index in range(candidate_count - 1, -1, -1):
            if reference[reference_index] == candidate[candidate_index]:
                lengths[reference_index][candidate_index] = (
                    lengths[reference_index + 1][candidate_index + 1] + 1
                )
            else:
                lengths[reference_index][candidate_index] = max(
                    lengths[reference_index + 1][candidate_index],
                    lengths[reference_index][candidate_index + 1],
                )

    matched: set[int] = set()
    reference_index = 0
    candidate_index = 0
    while reference_index < reference_count and candidate_index < candidate_count:
        if reference[reference_index] == candidate[candidate_index]:
            matched.add(reference_index)
            reference_index += 1
            candidate_index += 1
        elif (lengths[reference_index + 1][candidate_index]
              >= lengths[reference_index][candidate_index + 1]):
            reference_index += 1
        else:
            candidate_index += 1
    return matched


def _longest_missing_span(
    reference: Sequence[str], matched_indices: set[int],
) -> dict[str, int | str]:
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    for index in range(len(reference)):
        if index in matched_indices:
            current_length = 0
            continue
        if current_length == 0:
            current_start = index
        current_length += 1
        if current_length > best_length:
            best_start = current_start
            best_length = current_length
    return {
        "start_token": best_start,
        "token_count": best_length,
        "text": " ".join(reference[best_start:best_start + best_length]),
    }


def adjacent_repeated_fragments(tokens: Sequence[str]) -> list[dict]:
    """找出相邻连续重复的最小词块，避免把普通的远距离复现算成抖动。"""
    fragments: list[dict] = []
    start = 0
    while start < len(tokens):
        found: tuple[int, int] | None = None
        max_width = min(32, (len(tokens) - start) // 2)
        for width in range(1, max_width + 1):
            block = tokens[start:start + width]
            if block != tokens[start + width:start + 2 * width]:
                continue
            repeat_count = 2
            while (tokens[start + repeat_count * width:
                          start + (repeat_count + 1) * width] == block):
                repeat_count += 1
            found = (width, repeat_count)
            break
        if found is None:
            start += 1
            continue
        width, repeat_count = found
        fragments.append({
            "start_token": start,
            "token_count": width,
            "repeat_count": repeat_count,
            "text": " ".join(tokens[start:start + width]),
        })
        start += width * repeat_count
    return fragments


def compare_token_sequences(
    reference: Sequence[str], candidate: Sequence[str],
) -> dict:
    matched = _lcs_reference_matches(reference, candidate)
    edit_distance = _levenshtein(reference, candidate)
    reference_count = len(reference)
    return {
        "english_coverage": (
            len(matched) / reference_count if reference_count else None
        ),
        "word_edit_distance": edit_distance,
        "normalized_word_edit_distance": (
            edit_distance / reference_count if reference_count else None
        ),
        "longest_missing_span": _longest_missing_span(reference, matched),
    }


def _event_text(events: Iterable[dict], field: str) -> tuple[str, int]:
    texts: list[str] = []
    count = 0
    for event in events:
        count += 1
        text = str(event.get(field) or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts), count


def _asr_stream(run_dir: Path) -> tuple[list[str], dict, set[str]]:
    events = read_jsonl(run_dir / "events.jsonl")
    finals = [event for event in events if event.get("event_type") == "final"]
    text, final_count = _event_text(finals, "text")
    tokens = normalize_english(text)
    keys = {
        str(event["source_key"])
        for event in finals
        if event.get("source_key")
    }
    summary = {
        "final_count": final_count,
        "error_count": sum(
            event.get("event_type") == "error" for event in events
        ),
        "english_token_count": len(tokens),
        "adjacent_repeated_fragments": adjacent_repeated_fragments(tokens),
    }
    return tokens, summary, keys


def _is_business_translation(event: dict, asr_final_keys: set[str]) -> bool:
    key = str(event.get("source_key") or "")
    if key.startswith(_NON_UTTERANCE_KEY_PREFIXES):
        return False
    if asr_final_keys and key:
        return key in asr_final_keys
    return bool(str(event.get("source_text") or "").strip())


def _translation_stream(
    run_dir: Path, asr_final_keys: set[str],
) -> tuple[list[str], dict]:
    events = read_jsonl(run_dir / "translation_events.jsonl")
    finals = [
        event for event in events
        if (event.get("event_type") == "translation_final"
            and _is_business_translation(event, asr_final_keys))
    ]
    # error 可能发生在 source_key/source_text 建立前；若像 final 一样做关联
    # 过滤，会把“翻译器不可用”这类最重要的失败静默漏掉。
    errors = [
        event for event in events
        if event.get("event_type") == "translation_error"
    ]
    text, final_count = _event_text(finals, "source_text")
    tokens = normalize_english(text)
    return tokens, {
        "final_count": final_count,
        "error_count": len(errors),
        "english_token_count": len(tokens),
        "adjacent_repeated_fragments": adjacent_repeated_fragments(tokens),
    }


def _stream_report(
    reference_tokens: list[str],
    candidate_tokens: list[str],
    reference_summary: dict,
    candidate_summary: dict,
) -> dict:
    return {
        "reference": reference_summary,
        "candidate": candidate_summary,
        "candidate_vs_reference": compare_token_sequences(
            reference_tokens, candidate_tokens,
        ),
    }


def compare_run_dirs(reference_dir: Path, candidate_dir: Path) -> dict:
    """构建比较结果；不修改两个运行目录中的任何文件。"""
    reference_asr, reference_asr_summary, reference_keys = _asr_stream(
        reference_dir
    )
    candidate_asr, candidate_asr_summary, candidate_keys = _asr_stream(
        candidate_dir
    )
    reference_translation, reference_translation_summary = _translation_stream(
        reference_dir, reference_keys
    )
    candidate_translation, candidate_translation_summary = _translation_stream(
        candidate_dir, candidate_keys
    )
    return {
        "reference_dir": str(reference_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "asr_final": _stream_report(
            reference_asr,
            candidate_asr,
            reference_asr_summary,
            candidate_asr_summary,
        ),
        "translation_final": _stream_report(
            reference_translation,
            candidate_translation,
            reference_translation_summary,
            candidate_translation_summary,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较同一录音的两个 Whicc 运行目录（仅输出 JSON，不写文件）",
    )
    parser.add_argument("reference_dir", type=Path, help="基准运行目录")
    parser.add_argument("candidate_dir", type=Path, help="候选运行目录")
    args = parser.parse_args()
    for label, run_dir in (
        ("reference_dir", args.reference_dir),
        ("candidate_dir", args.candidate_dir),
    ):
        if not run_dir.is_dir():
            parser.error(f"{label} 不是目录: {run_dir}")
    print(json.dumps(
        compare_run_dirs(args.reference_dir, args.candidate_dir),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
