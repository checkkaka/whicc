#!/usr/bin/env python3
"""同一音频按不同 feed 块回放，并与同档离线 generate 比较。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nemotron_stream import NemotronStream  # noqa: E402


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def feed_chunks(stream: NemotronStream, audio: np.ndarray,
                pattern: tuple[int, ...] | list[int]) -> None:
    offset = index = 0
    while offset < len(audio):
        size = pattern[index % len(pattern)]
        stream.feed(audio[offset:offset + size])
        offset += size
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    from mlx_audio.stt import load_model
    from mlx_audio.stt.utils import load_audio

    model = load_model(args.model)
    audio = np.asarray(load_audio(args.audio, 16000))
    patterns = ([1600], [2048], [997, 2111, 1603])
    failed = False
    for right in (3, 6, 13):
        offline = model.generate(
            __import__("mlx.core").core.array(audio),
            language=None, att_context_size=[56, right]).text
        for pattern in patterns:
            stream = NemotronStream(model, right_context=right)
            feed_chunks(stream, audio, pattern)
            actual = stream.finalize()
            match = normalized(actual) == normalized(offline)
            failed |= not match
            print(f"[56,{right}] feed={pattern} match={match} text={actual!r}")

    # 连续两次 commit 后，流内 PCM 必须逐样本等于最后切点后的原音频；
    # commit 不能改变完整 token 序列，只能隐藏已提交前缀。保留上下文后的
    # 续文不应再与“孤立 remainder”比较，后者正是旧 reset/replay 路径。
    right = 6
    total_sec = len(audio) / 16000
    first_feed_sec = min(20.0, total_sec * .4)
    first_cut_sec = min(10.0, first_feed_sec * .5)
    second_feed_sec = min(40.0, total_sec * .8)
    second_cut_global_sec = min(30.0, second_feed_sec * .75)
    first_feed = int(first_feed_sec * 16000)
    first_cut = int(first_cut_sec * 16000)
    second_feed = int(second_feed_sec * 16000)
    second_cut_global = int(second_cut_global_sec * 16000)
    if first_cut < first_feed < second_cut_global < second_feed:
        reference = NemotronStream(model, right_context=right)
        feed_chunks(reference, audio[:first_feed], [1600])
        feed_chunks(reference, audio[first_feed:second_feed], [2048])
        feed_chunks(reference, audio[second_feed:], [997, 2111, 1603])
        reference.finalize()

        stream = NemotronStream(model, right_context=right)
        feed_chunks(stream, audio[:first_feed], [1600])
        stream.commit_through(first_cut_sec)
        pcm_match_1 = np.array_equal(
            stream.uncommitted_samples, audio[first_cut:first_feed])
        feed_chunks(stream, audio[first_feed:second_feed], [2048])
        stream.commit_through((second_cut_global - first_cut) / 16000)
        pcm_match_2 = np.array_equal(
            stream.uncommitted_samples, audio[second_cut_global:second_feed])
        feed_chunks(stream, audio[second_feed:], [997, 2111, 1603])
        actual = stream.finalize()
        token_match = [token.id for token in stream._tokens] == [
            token.id for token in reference._tokens]
        reference.commit_through(second_cut_global_sec)
        text_match = normalized(actual) == normalized(reference.text)
        failed |= not (pcm_match_1 and pcm_match_2
                       and token_match and text_match)
        print("commit replay "
              f"pcm1={pcm_match_1} pcm2={pcm_match_2} "
              f"tokens={token_match} text={text_match}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
