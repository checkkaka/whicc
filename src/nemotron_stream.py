"""Nemotron 跨 PCM 持久流式包装。

不修改 site-packages。公开接口固定为:
  feed(pcm) -> StreamSnapshot | None
  commit_through(audio_time_sec)
  finalize()
  reset(language, right_context)

Linux/无 MLX 环境可加载本模块做接口与配置测试；真实 cache-aware
推理在 Apple Silicon + mlx_audio 可用时启用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# 右上下文 → 原生 chunk 帧数（每帧 80ms）
RIGHT_CONTEXT_CHUNK_FRAMES = {
    3: 4,    # 320ms
    6: 7,    # 560ms
    13: 14,  # 1.12s
}


@dataclass
class StreamSnapshot:
    """一次 feed 后的累计识别快照。"""

    text: str
    audio_sec: float
    tokens: list[Any] = field(default_factory=list)
    changed: bool = False
    # 切点用时间戳；空列表时主循环退化为字符比例切点
    segments: list[dict] = field(default_factory=list)


def _sentences_to_segments(sentences) -> list[dict]:
    """把 Nemotron sentences 规范为 {text,start,end}（避免依赖 whicc 循环导入）。"""
    out: list[dict] = []
    for sent in sentences or []:
        text = getattr(sent, "text", None)
        if text is None and isinstance(sent, dict):
            text = sent.get("text", "")
        start = getattr(sent, "start", None)
        if start is None and isinstance(sent, dict):
            start = sent.get("start", 0.0)
        end = getattr(sent, "end", None)
        if end is None and isinstance(sent, dict):
            end = sent.get("end", 0.0)
        out.append({
            "text": text or "",
            "start": float(start or 0.0),
            "end": float(end or 0.0),
        })
    return out


class NemotronStream:
    """跨实时 PCM 输入持久保存 encoder/RNNT 状态的流式包装。

    重要：当前默认路径不是真 cache-aware 流式。无 mlx stream_encode
    时用增量缓冲 + 整段 generate 回退（仍按原生 chunk 边界触发）。
    真机启用前须用 tools/nemotron_replay.py 过门禁；开关默认保持 false。
    """

    def __init__(self, model: Any = None, *, language: str = "auto",
                 right_context: int = 6, generate_fn=None):
        if right_context not in RIGHT_CONTEXT_CHUNK_FRAMES:
            right_context = 6
        self.model = model
        self.language = language
        self.right_context = right_context
        self.chunk_frames = RIGHT_CONTEXT_CHUNK_FRAMES[right_context]
        self.chunk_sec = self.chunk_frames * 0.08
        self._generate_fn = generate_fn
        self._pcm: list = []
        self._pcm_samples = 0
        self._committed_samples = 0
        self._text = ""
        self._last_emitted = ""
        self._sample_rate = 16000

    def reset(self, language: str | None = None,
              right_context: int | None = None) -> None:
        """语言切换/丢帧/异常时重置全部流状态。"""
        if language is not None:
            self.language = language
        if right_context is not None:
            if right_context not in RIGHT_CONTEXT_CHUNK_FRAMES:
                right_context = 6
            self.right_context = right_context
            self.chunk_frames = RIGHT_CONTEXT_CHUNK_FRAMES[right_context]
            self.chunk_sec = self.chunk_frames * 0.08
        self._pcm = []
        self._pcm_samples = 0
        self._committed_samples = 0
        self._text = ""
        self._last_emitted = ""

    def feed(self, pcm) -> Optional[StreamSnapshot]:
        """喂入 float32 mono PCM；凑够原生 chunk 才可能返回快照。"""
        import numpy as np

        if pcm is None or len(pcm) == 0:
            return None
        arr = np.asarray(pcm, dtype=np.float32)
        self._pcm.append(arr)
        self._pcm_samples += len(arr)
        available_sec = (self._pcm_samples - self._committed_samples) / self._sample_rate
        if available_sec + 1e-9 < self.chunk_sec:
            return None
        return self._decode_available()

    def commit_through(self, audio_time_sec: float) -> StreamSnapshot:
        """断句后提交边界前文本；保留边界后 PCM/状态（回退实现丢弃前缀）。"""
        import numpy as np

        cut = max(0, int(audio_time_sec * self._sample_rate))
        if cut <= 0:
            return StreamSnapshot(text=self._text, audio_sec=0.0, changed=False)
        all_pcm = np.concatenate(self._pcm) if self._pcm else np.array([], dtype=np.float32)
        kept = all_pcm[cut:]
        self._pcm = [kept] if len(kept) else []
        self._pcm_samples = len(kept)
        self._committed_samples = 0
        # 回退路径：提交后清空已发射文本，后续 feed 重新识别剩余音频
        prev = self._text
        self._text = ""
        self._last_emitted = ""
        return StreamSnapshot(text=prev, audio_sec=audio_time_sec, changed=bool(prev))

    def finalize(self) -> StreamSnapshot:
        """冲刷剩余缓冲。"""
        snap = self._decode_available(force=True)
        if snap is None:
            return StreamSnapshot(text=self._text, audio_sec=self._pcm_samples / self._sample_rate)
        return snap

    def _decode_available(self, force: bool = False) -> Optional[StreamSnapshot]:
        import numpy as np

        if not self._pcm:
            return None
        all_pcm = np.concatenate(self._pcm)
        if len(all_pcm) == 0:
            return None
        if not force:
            need = int(self.chunk_sec * self._sample_rate)
            if len(all_pcm) < need:
                return None
        text = self._text
        segments: list[dict] = []
        if self._generate_fn is not None:
            try:
                # 调用注入的 generate：便于单测与无 MLX 环境回退
                result = self._generate_fn(all_pcm, language=self.language,
                                           right_context=self.right_context)
                if isinstance(result, dict):
                    text = (result.get("text") or "").strip()
                    # 优先用现成 segments；否则从 sentences 转换，供精确切点
                    segs = result.get("segments")
                    if segs:
                        segments = list(segs)
                    else:
                        segments = _sentences_to_segments(result.get("sentences"))
                else:
                    text = str(result).strip()
            except Exception as e:
                # 上抛让主循环 disable_native_stream，避免静默卡在空结果
                raise RuntimeError(f"NemotronStream generate failed: {e}") from e
        elif self.model is not None and hasattr(self.model, "generate"):
            try:
                import mlx.core as mx
                audio = mx.array(all_pcm)
                r = self.model.generate(
                    audio,
                    language=None if self.language in ("auto", "") else self.language,
                    att_context_size=[56, self.right_context],
                )
                text = (r.text or "").strip()
                sentences = r.sentences if hasattr(r, "sentences") else []
                segments = _sentences_to_segments(sentences)
            except Exception as e:
                # 上抛让主循环 disable_native_stream，避免静默卡在空结果
                raise RuntimeError(f"NemotronStream generate failed: {e}") from e
        changed = text != self._last_emitted
        self._text = text
        self._last_emitted = text
        if not changed and not force:
            return None
        return StreamSnapshot(
            text=text,
            audio_sec=len(all_pcm) / self._sample_rate,
            changed=changed,
            segments=segments,
        )
