#!/usr/bin/env python3
"""实时 PCM 回放器：完整投喂尾块，可选追加静音触发 final。"""
import argparse
import json
import sys
import os
import signal
import time
from pathlib import Path

SEG_DIR = "/tmp/whicc-seg"
SEG_DONE_FILE = ".whicc-done"
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 4
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE


def iter_audio_chunks(pcm_path: str | Path, chunk_bytes: int,
                      tail_silence_bytes: int):
    """按固定块返回完整文件，并用静音补接最后半块；绝不丢余数。"""
    pending = b""
    with open(pcm_path, "rb") as source:
        while True:
            data = source.read(chunk_bytes - len(pending))
            if not data:
                break
            pending += data
            if len(pending) == chunk_bytes:
                yield pending
                pending = b""

    remaining = max(0, tail_silence_bytes)
    while remaining:
        take = min(chunk_bytes - len(pending), remaining)
        pending += bytes(take)
        remaining -= take
        if len(pending) == chunk_bytes:
            yield pending
            pending = b""
    if pending:
        yield pending

def main():
    parser = argparse.ArgumentParser(description="按实时时序向 /tmp/whicc-seg 投喂 f32le PCM")
    parser.add_argument("input", help="mono 16kHz float32 little-endian PCM")
    parser.add_argument("--chunk-ms", type=int, default=1000,
                        help="投喂块大小（默认 1000ms；端到端延迟测试建议 100ms）")
    parser.add_argument("--tail-silence-sec", type=float, default=0,
                        help="末尾追加静音，触发正常 silence final")
    args = parser.parse_args()
    pcm_path = args.input
    if not os.path.exists(pcm_path):
        print(f"文件不存在: {pcm_path}", file=sys.stderr)
        sys.exit(1)
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms 必须大于 0")

    file_size = os.path.getsize(pcm_path)
    chunk_bytes = max(BYTES_PER_SAMPLE,
                      BYTES_PER_SECOND * args.chunk_ms // 1000)
    chunk_bytes -= chunk_bytes % BYTES_PER_SAMPLE
    tail_silence_bytes = int(max(0, args.tail_silence_sec) * BYTES_PER_SECOND)
    duration_sec = (file_size + tail_silence_bytes) / BYTES_PER_SECOND

    os.makedirs(SEG_DIR, exist_ok=True)
    for f in os.listdir(SEG_DIR):
        if (f.endswith(".pcm") or f.endswith(".meta.json")
                or f == SEG_DONE_FILE):
            try:
                os.unlink(os.path.join(SEG_DIR, f))
            except OSError:
                pass

    print(f"whicc-audio: OK", file=sys.stderr, flush=True)

    running = True
    def on_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    for seg_idx, data in enumerate(iter_audio_chunks(
            pcm_path, chunk_bytes, tail_silence_bytes)):
        if not running:
            break
        # chunk 表示这段音频在现实时间中刚采集完成；先按其时长等待，再将
        # 单调时间戳和 PCM 原子暴露给消费端。旧实现先写后睡，会把每块的
        # 起点误记成终点，使 100ms 回放的尾延迟系统性少算约 100ms。
        time.sleep(len(data) / BYTES_PER_SECOND)
        if not running:
            break
        capture_end_mono_ns = time.monotonic_ns()
        meta_path = os.path.join(
            SEG_DIR, f"seg-{seg_idx:06d}.meta.json")
        meta_tmp = meta_path + ".tmp"
        with open(meta_tmp, "w", encoding="utf-8") as metadata:
            json.dump({
                "capture_end_mono_ns": capture_end_mono_ns,
                "sequence": seg_idx,
            }, metadata, separators=(",", ":"))
        os.replace(meta_tmp, meta_path)
        path = os.path.join(SEG_DIR, f"seg-{seg_idx:06d}.pcm")
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as out:
            out.write(data)
        os.replace(tmp_path, path)

    # 不删除已写段：ASR 可能正忙于推理。原子 done marker 只声明生产端
    # 不会再写新块；消费端必须读完最后一个连续编号后才把它当 EOF。
    done_path = os.path.join(SEG_DIR, SEG_DONE_FILE)
    done_tmp = done_path + ".tmp"
    with open(done_tmp, "w", encoding="utf-8") as marker:
        marker.write("done\n")
    os.replace(done_tmp, done_path)
    print(f"回放投喂完成: {duration_sec:.3f}s", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
