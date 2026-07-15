#!/usr/bin/env python3
"""用 YouTube 英文字幕评估 Whicc 固定录音回归。

只做离线报告：不启动后端、不改配置。参数矩阵只列文档明确支持、
且当前项目能解释的档位，避免扫无意义组合。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_runs import (
    _lcs_reference_matches,
    adjacent_repeated_fragments,
    compare_token_sequences,
    normalize_english,
)
from tools.latency_report import percentile, read_jsonl, report as latency_report


_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d\d:\d\d:\d\d\.\d{3}) --> "
    r"(?P<end>\d\d:\d\d:\d\d\.\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Cue:
    start_sec: float
    end_sec: float
    text: str
    token_start: int
    token_end: int


def _seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt(path: Path) -> list[Cue]:
    """解析 WebVTT cue，合并同 cue 内多行文本并记录 reference token 范围。"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cues: list[Cue] = []
    token_cursor = 0
    index = 0
    while index < len(lines):
        match = _TIMESTAMP_RE.match(lines[index].strip())
        if not match:
            index += 1
            continue
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(_TAG_RE.sub("", lines[index].strip()))
            index += 1
        text = " ".join(text_lines)
        tokens = normalize_english(text)
        cues.append(Cue(
            start_sec=_seconds(match.group("start")),
            end_sec=_seconds(match.group("end")),
            text=text,
            token_start=token_cursor,
            token_end=token_cursor + len(tokens),
        ))
        token_cursor += len(tokens)
    return cues


def documented_parameter_matrix() -> dict:
    return {
        "nemotron_main": [
            {"att_context_size": [56, 3], "latency_ms": 320,
             "role": "低延迟候选"},
            {"att_context_size": [56, 6], "latency_ms": 560,
             "role": "默认平衡候选"},
            {"att_context_size": [56, 13], "latency_ms": 1120,
             "role": "质量回退"},
        ],
        "nemotron_hidden_exploration": [
            {"att_context_size": [56, 0], "latency_ms": 80,
             "role": "隐藏探索；不进 UI、不改默认"},
        ],
        "apple_asr": {
            "silence_submit_sec": [1.2, 1.0],
            "punct_end_min_chunk_sec_en": [3.0, 2.4, 2.0],
        },
        "native_streaming": [False, True],
        "translation": "Hy-MT2 参数固定；ASR/断句稳定后再单独 A/B",
        "sources": [
            "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
            "https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/nemotron_asr/README.md",
            "https://github.com/Blaizzy/mlx-audio/issues/773",
        ],
    }


def _final_events(run_dir: Path) -> list[dict]:
    return [
        event for event in read_jsonl(run_dir / "events.jsonl")
        if event.get("event_type") == "final"
    ]


def _final_text(events: list[dict]) -> str:
    return " ".join(str(event.get("text") or "").strip()
                    for event in events).strip()


def _missing_cues(cues: list[Cue], matched: set[int]) -> list[dict]:
    missing = []
    for cue in cues:
        token_count = cue.token_end - cue.token_start
        if token_count <= 0:
            continue
        covered = sum(index in matched
                      for index in range(cue.token_start, cue.token_end))
        if covered == 0:
            missing.append({
                "start_sec": cue.start_sec,
                "end_sec": cue.end_sec,
                "text": cue.text,
            })
    return missing


def _segmentation_quality(events: list[dict],
                          *, max_sentence_sec: float) -> dict:
    too_short = []
    too_long = []
    durations = []
    token_counts = []
    all_tokens: list[str] = []
    for index, event in enumerate(events):
        text = str(event.get("text") or "").strip()
        tokens = normalize_english(text)
        all_tokens.extend(tokens)
        token_counts.append(len(tokens))
        duration = float(event.get("chunk_sec") or 0.0)
        if duration:
            durations.append(duration)
        if 0 < len(tokens) < 3 and not text.rstrip().endswith(("?", "!", "？", "！")):
            too_short.append({"index": index, "text": text})
        if duration > max_sentence_sec:
            too_long.append({
                "index": index,
                "duration_sec": duration,
                "text": text,
                "submit_reason": event.get("submit_reason"),
            })
    return {
        "final_count": len(events),
        "token_count_p50": percentile(token_counts, .5),
        "chunk_sec_p50": percentile(durations, .5),
        "chunk_sec_p95": percentile(durations, .95),
        "too_short_finals": too_short,
        "too_long_finals": too_long,
        "adjacent_repeated_fragments": adjacent_repeated_fragments(all_tokens),
    }


def build_report(vtt_path: Path, run_dir: Path,
                 *, max_sentence_sec: float = 10.56) -> dict:
    cues = parse_vtt(vtt_path)
    reference_tokens = normalize_english(" ".join(cue.text for cue in cues))
    events = _final_events(run_dir)
    candidate_tokens = normalize_english(_final_text(events))
    matched = _lcs_reference_matches(reference_tokens, candidate_tokens)
    missing_cues = _missing_cues(cues, matched)
    latency = latency_report(run_dir)
    return {
        "reference_vtt": str(vtt_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "reference": {
            "cue_count": len(cues),
            "token_count": len(reference_tokens),
            "first_cue_sec": cues[0].start_sec if cues else None,
            "last_cue_sec": cues[-1].end_sec if cues else None,
        },
        "asr": {
            **compare_token_sequences(reference_tokens, candidate_tokens),
            "candidate_token_count": len(candidate_tokens),
            "missing_caption_blocks": missing_cues,
        },
        "segmentation": _segmentation_quality(
            events, max_sentence_sec=max_sentence_sec,
        ),
        "latency": {
            name: {
                "n": len(values),
                "p50_sec": percentile(values, .5),
                "p95_sec": percentile(values, .95),
            }
            for name, values in latency.items()
        },
        "parameter_matrix": documented_parameter_matrix(),
        "gates": {
            "no_complete_caption_missing": len(missing_cues) == 0,
            "no_too_long_final": not _segmentation_quality(
                events, max_sentence_sec=max_sentence_sec,
            )["too_long_finals"],
            "no_adjacent_repetition": not adjacent_repeated_fragments(
                candidate_tokens
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用 YouTube 英文 VTT 评估 Whicc ASR/断句/延迟回归",
    )
    parser.add_argument("--reference-vtt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-sentence-sec", type=float, default=10.32,
                        help="断句硬上限；默认 10s + [56,3] 320ms")
    parser.add_argument("--matrix-only", action="store_true",
                        help="只输出文档支持的参数矩阵")
    args = parser.parse_args()
    if args.matrix_only:
        print(json.dumps(documented_parameter_matrix(),
                         ensure_ascii=False, indent=2))
        return
    if not args.reference_vtt.is_file():
        parser.error(f"reference VTT 不存在: {args.reference_vtt}")
    if not args.run_dir.is_dir():
        parser.error(f"run dir 不存在: {args.run_dir}")
    print(json.dumps(
        build_report(args.reference_vtt, args.run_dir,
                     max_sentence_sec=args.max_sentence_sec),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
