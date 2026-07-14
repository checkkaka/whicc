#!/usr/bin/env python3
"""固定 ASR 输入独立比较 Hy-MT2 5/1.02 与 20/1.05。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from translator_hy_mt2 import HyMT2Translator, set_scene_context  # noqa: E402


def rewrite_ratio(previous: str, current: str) -> float:
    if not previous:
        return 0.0
    prefix = 0
    for left, right in zip(previous, current):
        if left != right:
            break
        prefix += 1
    return 1 - prefix / max(len(previous), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="JSONL：source_key/revision/source_text")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--target-lang", default="Simplified Chinese")
    parser.add_argument("--glossary", default=str(ROOT / "src" / "glossary.json"))
    parser.add_argument("--scene", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="hy_mt2_ab_results.jsonl")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = Path(args.output)
    set_scene_context(args.scene)
    results = []
    configs = (("A", 5, 1.02), ("B", 20, 1.05))
    translators = {
        label: HyMT2Translator(
            model_id=args.model, vllm_url=args.url, top_k=top_k,
            repetition_penalty=penalty, glossary_path=args.glossary)
        for label, top_k, penalty in configs
    }
    # 每个 group/run 都维持自己的 revision 链；不能让同一输入的 run=1
    # 错拿 run=0 输出作 previous。A/B 逐样本交错并轮换先后，降低预热、
    # 前缀缓存和服务状态随执行顺序造成的系统偏差。
    previous_by_group_key_run = {}
    for row_index, row in enumerate(rows):
        source = row["source_text"]
        key = row.get("source_key") or f"row-{row_index}"
        for run in range(args.runs):
            order = (configs if (row_index * args.runs + run) % 2 == 0
                     else tuple(reversed(configs)))
            for label, top_k, penalty in order:
                translator = translators[label]
                first_token = [0]
                started = time.monotonic_ns()

                def on_token(_piece, _full):
                    if not first_token[0]:
                        first_token[0] = time.monotonic_ns()

                translated, elapsed = translator.translate_streaming(
                    source, on_token=on_token, context=row.get("context"),
                    target_lang=args.target_lang)
                previous_key = (label, key, run)
                result = {
                    **row, "group": label, "run": run,
                    "top_k": top_k, "repetition_penalty": penalty,
                    "translated_text": translated,
                    "ttft_ms": ((first_token[0] - started) / 1e6
                                if first_token[0] else None),
                    "translate_ms": elapsed,
                    "rewrite_ratio": rewrite_ratio(
                        previous_by_group_key_run.get(previous_key, ""),
                        translated),
                    "finish_reason": translator.finish_reason,
                }
                results.append(result)
                previous_by_group_key_run[previous_key] = translated
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    for label in ("A", "B"):
        group = [r for r in results if r["group"] == label]
        elapsed = [r["translate_ms"] for r in group]
        rewrites = [r["rewrite_ratio"] for r in group]
        print(f"{label}: n={len(group)} latency_median={statistics.median(elapsed):.1f}ms "
              f"rewrite_mean={statistics.fmean(rewrites):.3f}")
    print(f"明细已写入 {output}；数字/专名/术语/语义和盲评需人工标注后决定是否改默认值。")


if __name__ == "__main__":
    main()
