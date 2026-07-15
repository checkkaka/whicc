"""Nemotron 0.4.4 cache-aware streaming state without patching site-packages."""

from __future__ import annotations

import numpy as np


RIGHT_CONTEXTS = {3: 4, 6: 7, 13: 14}


def native_chunk_frames(right_context: int) -> int:
    if right_context not in RIGHT_CONTEXTS:
        raise ValueError("right_context must be 3, 6, or 13")
    return RIGHT_CONTEXTS[right_context]


def aligned_token_segments(tokens, *, time_offset_sec: float = 0.0) -> list[dict]:
    """保留 token 时间戳，并与 AlignedResult.text 的首尾 strip 对齐。"""
    segments = [
        {
            "text": token.text,
            "start": max(0.0, float(token.start) - time_offset_sec),
            "end": max(0.0, float(token.start + token.duration)
                       - time_offset_sec),
        }
        for token in tokens
    ]
    if segments:
        # SentencePiece 首 token 常解码成 " Hello"，而 mlx-audio 的
        # AlignedResult 会 strip 整段文本；不对齐会让标点字符索引提前
        # 落到前一 token，切点约早一个 80ms 帧。
        segments[0]["text"] = segments[0]["text"].lstrip()
        segments[-1]["text"] = segments[-1]["text"].rstrip()
    return segments


class NemotronStream:
    """跨 feed 保存 encoder/RNNT 状态；commit 仅推进可见时间轴。"""

    def __init__(self, model, language: str = "auto", right_context: int = 6):
        self.model = model
        self.reset(language, right_context)

    def reset(self, language: str = "auto", right_context: int = 6):
        self.language = language or "auto"
        self.right_context = right_context
        self.att_context_size = [56, right_context]
        self.chunk_frames = native_chunk_frames(right_context)
        self._pcm = np.array([], dtype=np.float32)
        self._preemphasized = None
        self._last_raw_sample = None
        self._frontend_mel = None
        self._frontend_mel_frames = 0
        self._mel_consumed = 0
        self._mel_cache = None
        self._subsample_emitted = 0
        self._attn_cache = []
        self._conv_cache = []
        self._last_token = None
        self._decoder_hidden = None
        self._tokens = []
        self._global_time = 0
        # 只有 encoder/RNNT 真正消费新帧后才递增。主循环不能把一次
        # 没有新 chunk 的 feed 当成第二次 ASR 观察，否则会把 21.4
        # 在只看到 "21." 时提前确认成句末。
        self._decode_generation = 0
        if self.model is not None:
            layers = len(self.model.encoder.layers)
            self._attn_cache = [None] * layers
            self._conv_cache = [None] * layers
            self._last_token = self.model.blank_id
        return self

    def close(self) -> None:
        """释放模型强引用并清空流状态，供单模型语言切换回收显存。"""
        language, right_context = self.language, self.right_context
        self.model = None
        self.reset(language, right_context)

    @property
    def decode_generation(self) -> int:
        """已实际推进的 encoder/RNNT chunk 数。"""
        return self._decode_generation

    @property
    def uncommitted_samples(self) -> np.ndarray:
        self._ensure_commit_epoch()
        return self._pcm[self._committed_samples:].copy()

    @property
    def audio_time_sec(self) -> float:
        self._ensure_commit_epoch()
        sample_rate = (self.model.preprocessor_config.sample_rate
                       if self.model is not None else 16000)
        return max(0, len(self._pcm) - self._committed_samples) / sample_rate

    def feed(self, pcm) -> str:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size:
            self._pcm = np.concatenate([self._pcm, samples])
        if self.model is not None:
            self._append_preemphasis(samples)
            self._append_mel(final=False)
            self._process_available(final=False)
        return self.text

    def finalize(self) -> str:
        if self.model is not None:
            self._append_mel(final=True)
            self._process_available(final=True)
        return self.text

    def commit_through(self, audio_time_sec: float) -> str:
        """逻辑提交切点；保留 encoder、conv 与 RNNT 状态继续解码。

        物理 reset/replay 会丢掉 56 帧左上下文和 decoder hidden，造成跨切点
        短语被吞。这里仅推进可见 PCM/时间轴，模型缓存保持原样。
        """
        self._ensure_commit_epoch()
        sample_rate = (self.model.preprocessor_config.sample_rate
                       if self.model is not None else 16000)
        visible_samples = max(0, len(self._pcm) - self._committed_samples)
        cut = max(0, min(visible_samples, int(audio_time_sec * sample_rate)))
        physical_drop = self._committed_samples + cut
        if physical_drop:
            self._pcm = self._pcm[physical_drop:].copy()
            self._pcm_sample_origin += physical_drop
        self._committed_samples = 0
        self._time_origin_sec += cut / sample_rate
        # 只跳过 commit 当时已经存在且完全落在切点前的 token。右上下文
        # 可能稍后才追加一个时间戳较早的 token；它不在此索引内，必须保留。
        while (self._visible_token_index < len(self._tokens)
               and float(self._tokens[self._visible_token_index].start
                         + self._tokens[self._visible_token_index].duration)
               <= self._time_origin_sec + 1e-6):
            self._visible_token_index += 1
        # encoder/RNNT cache 保留，但前端只需留下未消费 mel 与下一 STFT
        # 帧所需的原始窗口；否则一小时直播会让三个数组线性增长。
        if self._frontend_mel is not None and self._mel_consumed:
            consumed = min(self._mel_consumed, self._frontend_mel.shape[1])
            self._frontend_mel = self._frontend_mel[:, consumed:]
            self._mel_frame_origin += consumed
            self._mel_consumed = 0
        if self._preemphasized is not None and self.model is not None:
            args = self.model.preprocessor_config
            next_frame_start = max(
                0, self._frontend_mel_frames * args.hop_length
                - args.n_fft // 2)
            drop = min(
                len(self._preemphasized),
                max(0, next_frame_start - self._preemphasis_sample_origin),
            )
            if drop:
                self._preemphasized = self._preemphasized[drop:]
                self._preemphasis_sample_origin += drop
        return self.text

    def _ensure_commit_epoch(self) -> None:
        """reset 会替换 cache 列表；据此惰性清零逻辑提交游标。"""
        if getattr(self, "_commit_cache_identity", None) is self._attn_cache:
            return
        self._commit_cache_identity = self._attn_cache
        self._committed_samples = 0
        self._time_origin_sec = 0.0
        self._visible_token_index = 0
        self._pcm_sample_origin = 0
        self._preemphasis_sample_origin = 0
        self._mel_frame_origin = 0

    def _visible_tokens(self):
        self._ensure_commit_epoch()
        return self._tokens[self._visible_token_index:]

    def _stable_visible_tokens(self):
        """去掉 RNNT 在 commit 后迟到补出的已提交 token 副本。

        只检查时间仍落在切点前的前缀，并要求 token ID 与已提交尾部精确
        一致；真正的新迟到 token（ID 不同）仍会保留，避免用字符模糊去重
        吞掉合法重复词。
        """
        tokens = self._visible_tokens()
        if not tokens or self._visible_token_index <= 0:
            return tokens
        late_count = 0
        for token in tokens:
            if float(token.start + token.duration) > self._time_origin_sec + 1e-6:
                break
            late_count += 1
        if not late_count:
            return tokens
        committed = self._tokens[:self._visible_token_index]
        limit = min(late_count, len(committed))
        for size in range(limit, 0, -1):
            if ([token.id for token in committed[-size:]]
                    == [token.id for token in tokens[:size]]):
                return tokens[size:]
        return tokens

    @property
    def text(self) -> str:
        tokens = self._stable_visible_tokens()
        if not tokens:
            return ""
        from mlx_audio.stt.models.nemo.alignment import sentences_to_result, tokens_to_sentences
        return sentences_to_result(tokens_to_sentences(tokens)).text

    def result_dict(self) -> dict:
        tokens = self._stable_visible_tokens()
        if not tokens:
            return {"text": "", "segments": []}
        from mlx_audio.stt.models.nemo.alignment import sentences_to_result, tokens_to_sentences
        result = sentences_to_result(tokens_to_sentences(tokens))
        return {
            "text": result.text,
            "segments": aligned_token_segments(
                tokens, time_offset_sec=self._time_origin_sec),
        }

    def _process_available(self, final: bool) -> None:
        mel = self._frontend_mel
        if mel is None:
            return
        total = mel.shape[1]
        chunk_mel = self.chunk_frames * self.model.encoder.args.subsampling_factor
        while total - self._mel_consumed >= chunk_mel:
            end = self._mel_consumed + chunk_mel
            self._encode_decode(mel[:, self._mel_consumed:end], is_final=False)
        if final and self._mel_consumed < total:
            self._encode_decode(mel[:, self._mel_consumed:total], is_final=True)

    def _append_preemphasis(self, samples: np.ndarray) -> None:
        if not samples.size:
            return
        import mlx.core as mx

        raw = mx.array(samples)
        alpha = float(self.model.preprocessor_config.preemph or 0)
        if alpha > 0:
            first = (raw[:1] if self._last_raw_sample is None
                     else raw[:1] - alpha * self._last_raw_sample)
            emphasized = mx.concatenate(
                [first, raw[1:] - alpha * raw[:-1]], axis=0)
        else:
            emphasized = raw
        self._preemphasized = (emphasized if self._preemphasized is None
                               else mx.concatenate([self._preemphasized, emphasized]))
        self._last_raw_sample = raw[-1:]

    def _append_mel(self, *, final: bool) -> None:
        """只计算新稳定 STFT 帧；final 时补右侧 center padding。"""
        if self._preemphasized is None:
            return
        self._ensure_commit_epoch()
        import mlx.core as mx
        from mlx_audio.utils import STR_TO_WINDOW_FN, hanning, mel_filters

        args = self.model.preprocessor_config
        left_pad = args.n_fft // 2
        total_samples = self._pcm_sample_origin + len(self._pcm)
        if final:
            target_frames = 1 + total_samples // args.hop_length
        else:
            target_frames = max(
                0, 1 + (total_samples - left_pad) // args.hop_length)
        if target_frames <= self._frontend_mel_frames:
            return

        count = target_frames - self._frontend_mel_frames
        # STFT 第 k 帧读取 pre-emphasis 的
        # [k*hop-left_pad, k*hop-left_pad+n_fft)。commit 可能已经物理
        # 丢掉旧样本，因此按绝对采样偏移拼出当前所需的滚动 region。
        region_start = (self._frontend_mel_frames * args.hop_length
                        - left_pad)
        region_end = ((target_frames - 1) * args.hop_length
                      - left_pad + args.n_fft)
        buffer_start = self._preemphasis_sample_origin
        buffer_end = buffer_start + len(self._preemphasized)
        slice_start = max(region_start, buffer_start)
        slice_end = min(region_end, buffer_end)
        parts = []
        if slice_start > region_start:
            parts.append(mx.zeros(
                (slice_start - region_start,),
                dtype=self._preemphasized.dtype))
        if slice_end > slice_start:
            local_start = slice_start - buffer_start
            local_end = slice_end - buffer_start
            parts.append(self._preemphasized[local_start:local_end])
        if region_end > slice_end:
            parts.append(mx.zeros(
                (region_end - slice_end,),
                dtype=self._preemphasized.dtype))
        region = parts[0] if len(parts) == 1 else mx.concatenate(parts)
        frames = mx.as_strided(
            region, shape=(count, args.n_fft),
            strides=(args.hop_length, 1))

        window_fn = STR_TO_WINDOW_FN.get(args.window)
        window = window_fn(args.win_length) if window_fn else hanning(args.win_length)
        if window.shape[0] < args.n_fft:
            side = args.n_fft - window.shape[0]
            window = mx.concatenate([
                mx.zeros((side // 2,), dtype=window.dtype), window,
                mx.zeros((side - side // 2,), dtype=window.dtype),
            ])
        power = mx.square(mx.abs(mx.fft.rfft(frames * window))).astype(
            self._preemphasized.dtype)
        filters = mel_filters(
            args.sample_rate, args.n_fft, args.features,
            norm="slaney", mel_scale="slaney").astype(power.dtype)
        mel = mx.log(filters @ power.T + mx.array(
            args.log_zero_guard_value, dtype=power.dtype)).T[None, :, :]
        self._frontend_mel = (mel if self._frontend_mel is None
                              else mx.concatenate([self._frontend_mel, mel], axis=1))
        self._frontend_mel_frames = target_frames

    def _encode_decode(self, mel_chunk, *, is_final: bool) -> None:
        import mlx.core as mx
        from mlx_audio.stt.models.nemo.alignment import AlignedToken
        from mlx_audio.stt.models.nemotron_asr import tokenizer as tok
        from mlx_audio.stt.models.nemotron_asr.streaming import (
            _PRE_ENCODE_MEL_CACHE,
            _stream_block,
        )

        enc = self.model.encoder
        sf = enc.args.subsampling_factor
        left_cache = self.att_context_size[0]
        conv_left = enc.args.conv_kernel_size - 1
        cache_len = 0 if self._mel_cache is None else self._mel_cache.shape[1]
        window = (mel_chunk if self._mel_cache is None
                  else mx.concatenate([self._mel_cache, mel_chunk], axis=1))
        sub = enc.pre_encode(
            window, mx.array([window.shape[1]], dtype=mx.int32)
        )[0]
        absolute_consumed = self._mel_frame_origin + self._mel_consumed
        end = absolute_consumed + mel_chunk.shape[1]
        base = (absolute_consumed - cache_len) // sf
        lo = self._subsample_emitted - base
        hi = sub.shape[1] if is_final else (end // sf - base)
        self._mel_consumed += mel_chunk.shape[1]
        self._mel_cache = window[:, -_PRE_ENCODE_MEL_CACHE:]
        if hi <= lo:
            self._subsample_emitted = base + max(lo, hi)
            return
        self._subsample_emitted = base + hi
        hidden = sub[:, lo:hi]
        for index, block in enumerate(enc.layers):
            hidden, self._attn_cache[index], self._conv_cache[index] = _stream_block(
                block, hidden, enc.pos_enc,
                self._attn_cache[index], self._conv_cache[index],
                left_cache, conv_left,
            )
        prompted = self.model.apply_prompt(hidden, self.language)
        frame_sec = (self.model.encoder_config.subsampling_factor
                     * self.model.preprocessor_config.hop_length
                     / self.model.preprocessor_config.sample_rate)
        time_index = 0
        new_symbols = 0
        while time_index < prompted.shape[1]:
            feature = prompted[:, time_index:time_index + 1]
            current = (mx.array([[self._last_token]], dtype=mx.int32)
                       if self._last_token != self.model.blank_id else None)
            decoder_output, (h, c) = self.model.decoder(current, self._decoder_hidden)
            decoder_output = decoder_output.astype(feature.dtype)
            proposed_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
            pred = int(mx.argmax(self.model.joint(feature, decoder_output)))
            if pred != self.model.blank_id:
                self._last_token = pred
                self._decoder_hidden = proposed_hidden
                if not tok.is_special_token(pred, self.model.vocabulary):
                    self._tokens.append(AlignedToken(
                        pred,
                        start=(self._global_time + time_index) * frame_sec,
                        duration=frame_sec,
                        text=tok.decode([pred], self.model.vocabulary),
                    ))
                new_symbols += 1
                if self.model.max_symbols is not None and new_symbols >= self.model.max_symbols:
                    time_index += 1
                    new_symbols = 0
            else:
                time_index += 1
                new_symbols = 0
        self._global_time += prompted.shape[1]
        self._decode_generation += 1
