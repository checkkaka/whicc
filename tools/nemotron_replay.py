#!/usr/bin/env python3
"""NemotronStream 与同档 generate 回放对比脚手架。

用法（Apple Silicon + 已加载模型）:
  python3 tools/nemotron_replay.py --wav sample.wav --right-context 6

云端 Linux 无 MLX 时仅校验接口与 chunk 绑定；真正 WER/CER 门禁需本机跑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default="")
    parser.add_argument("--right-context", type=int, default=6, choices=[3, 6, 13])
    parser.add_argument("--chunk-ms", type=int, default=100,
                        help="模拟实时喂入粒度(ms)")
    args = parser.parse_args()

    from nemotron_stream import NemotronStream, RIGHT_CONTEXT_CHUNK_FRAMES
    import numpy as np

    print(f"right_context={args.right_context} "
          f"native_chunk_frames={RIGHT_CONTEXT_CHUNK_FRAMES[args.right_context]} "
          f"native_chunk_ms={RIGHT_CONTEXT_CHUNK_FRAMES[args.right_context]*80}")

    if not args.wav:
        # 无音频：只验证接口
        stream = NemotronStream(
            right_context=args.right_context,
            generate_fn=lambda pcm, language="auto", right_context=6: {
                "text": f"len={len(pcm)}"
            },
        )
        block = int(16000 * args.chunk_ms / 1000)
        for _ in range(20):
            stream.feed(np.zeros(block, dtype=np.float32))
        snap = stream.finalize()
        print(f"interface_ok text={snap.text!r} audio_sec={snap.audio_sec:.3f}")
        return 0

    try:
        import soundfile as sf
    except ImportError:
        print("需要 soundfile 读 wav: pip install soundfile", file=sys.stderr)
        return 2

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        print(f"需要 16kHz wav，当前 {sr}", file=sys.stderr)
        return 2

    try:
        from mlx_audio.stt import load_model
        import whicc
        model = load_model(whicc.DEFAULT_MODEL)
    except Exception as e:
        print(f"无法加载 MLX 模型，仅接口模式可用: {e}", file=sys.stderr)
        return 3

    # 同档 offline generate
    offline = whicc._do_transcribe_nemotron(
        audio, language="auto", right_context=args.right_context)
    offline_text = (offline.get("text") or "").strip()

    stream = NemotronStream(
        model=model, language="auto", right_context=args.right_context)
    block = int(16000 * args.chunk_ms / 1000)
    last = None
    for i in range(0, len(audio), block):
        snap = stream.feed(audio[i:i + block])
        if snap is not None:
            last = snap
    last = stream.finalize() if last is None else stream.finalize()
    stream_text = (last.text if last else "").strip()

    print("--- offline ---")
    print(offline_text)
    print("--- stream ---")
    print(stream_text)
    print("--- equal ---", offline_text == stream_text)
    return 0 if offline_text == stream_text else 1


if __name__ == "__main__":
    sys.exit(main())
