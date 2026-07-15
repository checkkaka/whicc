#!/usr/bin/env python3
"""whicc - 系统音频实时语音识别,nemotron / qwen3 streaming 直出原文

用法:
    python3 whicc.py [--save-wav DIR] [--dump-raw] [--dump-filtered] [--stats]
    python3 whicc.py --run-id v6_001 --events-jsonl runs/v6_001/events.jsonl --output-text runs/v6_001/final.txt \
        --min-chunk-sec 2.0 --max-chunk-sec 4.5 --overlap-sec 0.5 ...
"""
import sys
import os
import re
import json
import time
import argparse
import subprocess
import shlex
import shutil
import hashlib
import threading
import signal
import queue as queue_mod
from collections import Counter, deque
from dataclasses import dataclass
import numpy as np

# --------------- 配置 ---------------
import sys

# 让 python3 /path/to/src/whicc.py 这种直接调用方式能 import 同目录的
# config.py / audio.py。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# HF 镜像加速(设置页"下载加速"开关):whicc.py 的模型加载在本地缺模型
# 时会走 huggingface_hub 下载 fallback。HF_ENDPOINT 在 huggingface_hub
# **import 时**固化,必须在任何 hf/mlx_audio import 之前设好。
try:
    with open("/tmp/whicc-out/lang_config.json", encoding="utf-8") as _f:
        if json.load(_f).get("hf_mirror_enabled"):
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
except (OSError, json.JSONDecodeError):
    pass

from config import SEG_DIR, SAMPLE_RATE, BYTES_PER_SAMPLE, SEG_DURATION_SEC
# audiotee 路径：项目内 ./bin/audiotee（持久化,不会像 /tmp 那样被清掉）。
# 之前用的 /tmp/whicc-audio/.build/debug/whicc-audio 路径,二进制
# 容易被系统清理,导致字幕链断。现在改用 ./bin/audiotee,跟 livecaption
# (six-ddc/livecaption) 的设计一致——单文件 Python 进程,内部多线程,
# 音频采集跟 ASR 通过 queue.Queue 解耦。
AUDIO_BIN = "./bin/audiotee"
# 默认 ASR 模型: nemotron streaming (中英共用,英文效果更准;中文会自动切 qwen3)
DEFAULT_MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
# 兼容老的项目内 models/ 路径（仅做兜底，新代码用 --models-dir 走
# ~/Library/Application Support/whicc/models/）。保留以防用户从老
# 启动脚本启动时崩——但默认指向不存在的相对路径，强制使用 --models-dir。
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
QWEN3_MODEL = "mlx-community/Qwen3-ASR-0.6B-4bit"  # 中文备用 ASR

from model_state import (  # noqa: E402  (放在常量后避免循环 import)
    read_model_state,
    resolve_model_id, resolve_models_dir, resolve_chinese_model_id,
)

# 模型预设：不同模型的最佳默认参数
# max_chunk 兜底值: Qwen3 streaming 用 12s, nemotron streaming 用 15s。
MODEL_PRESETS = {
    "Qwen3-ASR":  {"no_speech": 0.50, "min_chunk": 2.0, "max_chunk": 12.0},
    "nemotron":   {"no_speech": 0.50, "min_chunk": 2.0, "max_chunk": 15.0},
}

def _detect_backend(model_path: str) -> str:
    """检测模型后端类型：'nemotron' 或 'qwen3'。"""
    name = os.path.basename(model_path).lower()
    if "nemotron" in name:
        return "nemotron"
    return "qwen3"

def _is_model_complete(local_path: str) -> bool:
    """本地模型目录是否完整可用。

    残缺目录(下载中断留下的 config/tokenizer 但没有权重)用
    os.path.isdir 判定会被当成有效模型 → load_model 抛
    "No weight files found" → 整个 ASR 链路起不来。判定标准:
    - model_downloader.py 下载成功后写的 <dir>.complete 标记;或
    - 目录里有权重文件(safetensors/npz — 兼容用户手工拷贝的模型)。
    """
    if not os.path.isdir(local_path):
        return False
    if os.path.exists(local_path + ".complete"):
        return True
    try:
        names = os.listdir(local_path)
    except OSError:
        return False
    return any(n.endswith((".safetensors", ".npz")) for n in names)

# 自适应 chunk (对齐 livecaption 策略,2026-06-28)
#   min=2.0, soft_max=5.0, max=10.0 (硬上限), silence=1.2s
#   稳定句末标点直接提交；0.6s 仅保留给已提交标点句后的静音路径。
#   核心改进: SILENCE_SUBMIT_SEC 0.4→1.2 — 思考停顿不再被当句尾
#   SOFT_MAX_SEC=5.0 — 长演讲累积够立刻在标点切,不强制切半
# 字幕触发参数 (平衡"快速显示" vs "完整句不切碎"):
# - PUNCT_END_MIN_CHUNK_SEC_EN=3.0 / _ZH=5.0: 累积够才检查标点,中文阈值更高
#   (中文 ASR streaming 容易幻觉句末标点,需要更多上下文才稳)。
# - SOFT_MAX_SEC = 5.0: 累积 5s 还没切 → 强制在中间标点切,防止字幕延迟
#   触发阈值的语言分层:
#   - 英文场景: 3s 后检查；`.` 跨两次观察防缩写/小数，其他强标点首次确认
#   - 中文场景: ASR 容易"幻觉"句末标点 (3s 仅 10-15 字时频繁误判),需要更长阈值
#   PUNCT_END_MIN_CHUNK_SEC_EN / _ZH 在 probe ASR 拿到 language 后动态选。
PUNCT_END_MIN_CHUNK_SEC_EN = 3.0   # 英文 (Nemotron 在英文 streaming 上 3s 标点可靠)
PUNCT_END_MIN_CHUNK_SEC_ZH = 5.0   # 中文 (Qwen3-ASR 中文 streaming 需要更多上下文)
STRONG_END_PUNCT = "。！？.!?"   # 强句末标点: 切这里一定是完整句
# 中文额外字符数校验: 标点出现位置的前一个字符往前数 >= 12 字符才算完整句。
# 短句 (< 12 字) 即使末尾有 `。` 也疑似 ASR 幻觉 (e.g. "美联。" 3 字就是截断)。
MIN_CHARS_BEFORE_PUNCT_ZH = 12
MIN_CHARS_BEFORE_PUNCT_EN = 8

MIN_CHUNK_SEC = 2.0
MAX_CHUNK_SEC = 10.0   # 兜底值,避免无限累积
SOFT_MAX_SEC = 5.0     # 软最大值:长句(无标点)兜底。新触发器优先切句末标点
SILENCE_THRESHOLD = 0.01  # RMS
PUNCT_SUBMIT_SEC = 0.6   # 句末标点结尾时更快提交
SILENCE_SUBMIT_SEC = 1.2  # 中间停顿等更久,避免思考停顿被切半
POLL_INTERVAL = 0.15       # 段文件轮询间隔
OVERLAP_SEC = 0.3          # chunk 间重叠

# 过滤
ALLOWED_LANG = {"en", "zh", "ja", "ko", "de", "fr", "es", "pt", "it"}
MIN_LOGPROB = -1.0
MAX_COMPRESSION = 2.4
MIN_SPEECH_SEC = 0.5       # 太短的 chunk 直接丢弃

# ASR 调优
NO_SPEECH_THRESHOLD = 0.45
INITIAL_PROMPT = ""

HALLUCINATION_PHRASES = [
    "субтитры", "dimatorzok", "подписал", "перевод",
    "subscribe", "like and subscribe", "thanks for watching",
]

# 短文本幻觉：长度 ≤ 阈值 且不包含有意义内容的转录
HALLUCINATION_SHORT_MAX_LEN = 5


def is_valid_partial_text(text: str) -> bool:
    """partial 的轻量幻觉门禁；不改变 final 的完整过滤阈值。"""
    stripped = (text or "").strip()
    short = stripped.rstrip(".!?")
    if not stripped or (len(short) <= 4 and short.lower() in {
        "the", "see", "you", "a", "oh", "so", "and", "but", "is"
    }):
        return False
    return not is_hallucination(stripped)


def update_punct_end_stability(candidate: str | None,
                               text: str | None) -> tuple[str | None, bool]:
    """ASCII 句点跨两次观察确认；无歧义强标点首次即可确认。"""
    stripped = (text or "").strip()
    if candidate:
        if stripped == candidate:
            return candidate, True
        # `.` 也可能是 Dr. / 21.4；后续一旦扩展就作废旧候选。
        # !? 及中日韩句末符号没有这种歧义，可在前缀保留时确认。
        if candidate[-1:] != "." and stripped.startswith(candidate):
            return candidate, True
    if not stripped or stripped[-1:] not in STRONG_END_PUNCT:
        return None, False
    return stripped, stripped[-1:] != "."


def update_native_punct_stability(candidate: str | None, stable: bool,
                                  text: str | None, *,
                                  previous_generation: int,
                                  current_generation: int
                                  ) -> tuple[str | None, bool]:
    """只在 native encoder/RNNT 真推进后把文本计作一次新观察。"""
    if current_generation == previous_generation:
        return candidate, stable
    return update_punct_end_stability(candidate, text)


def punctuation_pause_ready(candidate: str | None,
                            silence_streak: float) -> bool:
    """候选已完成标点稳定性确认，不再叠加第二段静音等待。"""
    _ = silence_streak  # 保留参数，避免主循环和历史测试调用面分叉。
    return bool(candidate)


def native_guarded_max_split_sec(chunk_sec: float, max_chunk_sec: float,
                                 right_context_sec: float) -> float:
    """等右上下文到齐后，仍在原 max_chunk 边界提交稳定 token。"""
    if (right_context_sec > 0
            and chunk_sec + 1e-9 >= max_chunk_sec + right_context_sec):
        return max_chunk_sec
    return 0.0


def find_native_word_split_sec(segments, *, target_sec: float,
                               max_lookback_sec: float = 1.5) -> float:
    """把硬切点移到最近的新词 token 起点，避免 `firm`/`ly` 分句。"""
    lower = max(0.0, target_sec - max_lookback_sec)
    candidate = 0.0
    for segment in normalize_segments(segments):
        text = segment.get("text", "")
        start = float(segment.get("start", 0.0) or 0.0)
        if lower <= start <= target_sec and text[:1].isspace():
            candidate = start
    return candidate


def resolve_native_max_split(segments, *, target_sec: float,
                             max_lookback_sec: float = 1.5
                             ) -> tuple[float, bool]:
    """返回 native 硬上限切点及是否必须用 ``[56,13]`` 批处理。

    有可靠空格起始 token 时回退到完整单词；否则仍在原 max_chunk
    边界切开，并由调用方对前半段走质量档批处理，禁止继续无限累积。
    """
    word_split = find_native_word_split_sec(
        segments, target_sec=target_sec,
        max_lookback_sec=max_lookback_sec)
    return (word_split, False) if word_split else (target_sec, True)


def sync_native_after_submit(
        native_stream, *, commit_sec: float, language: str,
        right_context: int, backend_after_submit: str,
        native_active_before_submit: bool, current_fallback: bool,
        raw_remainder: np.ndarray,
        ) -> tuple[bool, np.ndarray, Exception | None]:
    """final 成功后 best-effort 同步 native 状态，并始终保护原始 remainder。

    健康流用 `commit_through()` 逻辑提交并保留模型缓存。批处理 fallback 后若仍有
    remainder，则继续用 `[56,13] + overlap` 完成这段边界恢复；直接以短
    remainder 重启 native 会丢失跨切点词。只有 remainder 清空后才恢复
    native。MLX 任一步失败都保留原 PCM，并维持批处理质量兜底。
    """
    protected = np.asarray(raw_remainder, dtype=np.float32).reshape(-1).copy()
    if native_stream is None:
        return current_fallback, protected, None
    try:
        if backend_after_submit == "nemotron":
            if native_active_before_submit:
                native_stream.commit_through(commit_sec)
                fallback = False
            else:
                native_stream.reset(language, right_context)
                # 短 remainder 不足以独立建立可靠的 encoder/RNNT 上下文；
                # 保持批处理恢复，直到一次提交真正清空 remainder。
                fallback = bool(protected.size)
        else:
            native_stream.reset(language, right_context)
            fallback = False
        return fallback, protected, None
    except Exception as exc:  # noqa: BLE001 - native 失败必须降级而非丢 PCM
        try:
            native_stream.reset(language, right_context)
        except Exception:
            pass
        return True, protected, exc

# Prompt 模式
PROMPT_MODE_FIXED = "fixed"
PROMPT_MODE_TAIL = "tail"
PROMPT_MODE_NONE = "none"

# Reject reason 枚举
REJECT_SHORT_CHUNK = "short_chunk"
REJECT_HALLUCINATION = "hallucination"
REJECT_LANG = "lang"
REJECT_LOGPROB = "logprob"
REJECT_COMPRESSION = "compression"
REJECT_EMPTY = "empty_text"

# --------------- 过滤 ---------------

# 常见 ASR 转写错误修正 (大小写、专有名词)
CORRECTIONS = {
    "chatgpt": "ChatGPT",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "claude": "Claude",
    "dario": "Dario",
    "daniela": "Daniela",
    "amodei": "Amodei",
    "altman": "Altman",
    "stripe": "Stripe",
    "baidu": "Baidu",
    "palantir": "Palantir",
    "karnofsky": "Karnofsky",
    "hegseth": "Hegseth",
    "maduro": "Maduro",
    "oppenheimer": "Oppenheimer",
    "calypso": "Calypso",
    "mythos": "Mythos",
    "glasswing": "Glasswing",
    "precita": "Precita",
}

# 句首重复词清理 (ASR streaming 偶尔会重复短语)
SENTENCE_START_REPEATS = [
    r'^(I said,?\s+I said,?\s+)',
    r'^(and so,?\s+and so,?\s+)',
    r'^(but you know,?\s+but you know,?\s+)',
    r'^(so,?\s+so,?\s+so,?\s+)',
]

def postprocess(text: str) -> str:
    """修正常见 ASR 转写错误"""
    # 句首重复清理
    for pattern in SENTENCE_START_REPEATS:
        text = re.sub(pattern, lambda m: m.group(1)[len(m.group(1))//2:], text, flags=re.IGNORECASE)

    words = text.split()
    fixed = []
    for w in words:
        low = w.lower().strip('.,!?;:')
        if low in CORRECTIONS:
            # 提取词首非字母前缀和词尾非字母后缀
            first = next((i for i, c in enumerate(w) if c.isalpha()), len(w))
            last = next((i for i, c in enumerate(reversed(w)) if c.isalpha()), len(w))
            prefix = w[:first]
            suffix = w[len(w)-last:] if last > 0 else ''
            fixed.append(prefix + CORRECTIONS[low] + suffix)
        else:
            fixed.append(w)
    return ' '.join(fixed)


def canonical_asr_text(text: str) -> str:
    """partial/final 共用的源文规范化，保证同文本不虚增 revision。"""
    return postprocess((text or "").strip())

def is_hallucination(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if len(t) > 8:
        for i in range(1, min(6, len(t) // 3)):
            if t[:i] * 3 in t:
                return True
    low = t.lower()
    for p in HALLUCINATION_PHRASES:
        if p in low:
            return True
    # word-level 长句重复检测（抓 "This is a conversation about AI..." 类幻觉）
    words = low.split()
    if len(words) >= 12:
        for n in range(3, min(9, len(words) // 3 + 1)):
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            if not ngrams:
                continue
            counts = Counter(ngrams)
            most_common, most_count = counts.most_common(1)[0]
            if most_count >= 3:
                return True
    return False

# --------------- Nemotron / Qwen3 ---------------
# (曾经这里有 save_wav — 每次推理先把 float32 量化成 int16 写临时 WAV
# 再让模型读回。现在音频数组直传 generate(见 do_transcribe),整条管线
# 零磁盘往返,且保留 float32 全精度,不再有 int16 量化损失。)

_qwen3_model = None  # 预加载的 Qwen3 模型（懒初始化）

# 启动期模型加载错误（RepositoryNotFoundError / 网络 / disk） — 用这
# 个标记告诉 main() "模型加载失败,切回 Nemotron 默认"。
_model_load_failed = False
_model_load_error_msg = ""


def _get_qwen3_model(model_path: str):
    """懒加载 Qwen3 ASR 模型，只加载一次。失败时把全局 _model_load_failed 置上,
    main() 会回退到 Nemotron 默认而不是让进程 crash 掉。"""
    global _qwen3_model, _model_load_failed, _model_load_error_msg
    if _qwen3_model is None:
        from mlx_audio.stt import load_model
        try:
            _qwen3_model = load_model(model_path)
        except Exception as e:
            _model_load_failed = True
            _model_load_error_msg = f"Qwen3 model load failed: {model_path} - {e}"
            print(f"[model-load] {_model_load_error_msg}", file=sys.stderr, flush=True)
            raise
    return _qwen3_model


def _unload_qwen3():
    global _qwen3_model
    _qwen3_model = None


class AsyncModelLoadState:
    """异步模型加载的显式结果；ready 不再隐含 success。"""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.success = False
        self.error = ""

    def reset(self) -> None:
        self.success = False
        self.error = ""
        self.ready.clear()


def _async_load_model(which: str, model_path: str,
                      state: AsyncModelLoadState):
    """后台加载并 warmup；只有真实成功才允许主循环切换后端。"""
    try:
        if which == "qwen3":
            _get_qwen3_model(model_path)
            _warmup_model(model_path, "qwen3")
        else:
            _get_nemotron_model(model_path)
            _warmup_model(model_path, "nemotron")
        state.success = True
    except Exception as e:
        state.error = str(e)
        print(f"[lang-switch] 异步加载失败: {e}", file=sys.stderr, flush=True)
    finally:
        state.ready.set()


def _warmup_model(model_path: str, which: str) -> None:
    """对刚加载的模型做一次空推理 warmup（吸收 Metal kernel 编译延迟）。
    数组直传,不经临时 WAV。"""
    samples = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    try:
        if which == "qwen3":
            _do_transcribe_qwen3(samples, language="auto", model_path=model_path)
        else:
            _do_transcribe_nemotron(samples, language="auto", model_path=model_path)
    except Exception:
        pass


def do_transcribe(audio, language: str = "en",
                  model: str = DEFAULT_MODEL,
                  backend: str = "nemotron",
                  nemotron_right_context: int = 13) -> dict:
    """转录（非流式）。backend: 'nemotron' 或 'qwen3'。

    audio: WAV 路径(str) 或 float32 mono 16kHz 的 np.ndarray —
    两个后端的 generate 都原生支持数组输入(Nemotron 收 mx.array,
    Qwen3 直接收 ndarray),数组直传省掉"每次推理先写 WAV 再读回"
    的磁盘往返(探针每 0.6s 一次)。
    """
    if backend == "qwen3":
        return _do_transcribe_qwen3(audio, language=language, model_path=model)
    return _do_transcribe_nemotron(
        audio, language=language, model_path=model,
        right_context=nemotron_right_context,
    )


def _do_transcribe_qwen3(audio, language: str = "en", model_path: str = "") -> dict:
    """Qwen3 ASR 转录。audio: WAV 路径或 float32 16kHz ndarray
    (Qwen3ASRModel.generate 原生接受 ndarray,采样率假定 16k 与
    load_audio 默认一致)。"""
    m = _get_qwen3_model(model_path)
    r = m.generate(audio, language=language, verbose=False)

    text = r.text.strip() if r.text else ""
    # Qwen3 返回 language=['en']（列表），取第一个 — 但这个字段经常不准 (e.g. 中文
    # audio 仍返回 "en"),需要从文本字符判断。
    raw_lang = r.language[0] if isinstance(r.language, list) else (r.language or language)
    if language in ("auto", ""):
        # auto 模式下从文本判断: cjk > 30% → zh,否则用 raw_lang / "en"
        cjk_count = sum(1 for c in text if "一" <= c <= "鿿")
        if cjk_count > max(3, len(text) * 0.3):
            lang = "zh"
        else:
            lang = "en" if raw_lang in ("auto", "") else raw_lang
    else:
        # 显式指定了 language (e.g. "zh"),信任用户配置
        lang = "en" if raw_lang in ("auto", "") else raw_lang
    segs = r.segments or []

    # Qwen3 segments 没有 avg_logprob/compression_ratio/no_speech_prob
    # 用文本长度和 segment 数量做粗略估计
    avg_lp = -0.3 if text else -99    # Qwen3 没有 logprob，给个合理的默认值
    avg_cr = 0.0
    avg_nsp = 0.0 if text else 1.0

    return {"text": text, "language": lang, "avg_logprob": avg_lp,
            "avg_compression": avg_cr, "no_speech_prob": avg_nsp,
            "segments": normalize_segments(segs)}



def filter_result(result: dict) -> tuple:
    """判断转录结果是否通过过滤。返回 (reject_reason, postprocessed_text)，通过时 reject_reason=None"""
    text = result["text"]
    lang = result["language"]
    if not text:
        return REJECT_EMPTY, None
    # 单字幻觉：整句话就一个常见词（如 "The." "See." "You."），不是句子中有 the
    stripped = text.strip().rstrip(".!?")
    if len(stripped) <= 4 and stripped.lower() in {"the", "see", "you", "a", "oh", "so", "and", "but", "is"}:
        return REJECT_HALLUCINATION, None
    if lang not in ALLOWED_LANG:
        return REJECT_LANG, None
    if result["avg_logprob"] < MIN_LOGPROB:
        return REJECT_LOGPROB, None
    if result["avg_compression"] > MAX_COMPRESSION:
        return REJECT_COMPRESSION, None
    if is_hallucination(text):
        return REJECT_HALLUCINATION, None
    return None, canonical_asr_text(text)

# --------------- Nemotron ASR ---------------

_nemotron_model = None


def normalize_segments(items) -> list[dict]:
    """把 Qwen3 segments / Nemotron sentences 统一成切句所需结构。"""
    normalized = []
    for item in items or []:
        get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
        try:
            normalized.append({
                "text": str(get("text", "") or ""),
                "start": float(get("start", 0.0) or 0.0),
                "end": float(get("end", 0.0) or 0.0),
            })
        except (TypeError, ValueError):
            continue
    return normalized


def trim_transcription_result(result: dict, audio_time_sec: float) -> dict | None:
    """按真实 segment 终点裁剪结果；无可用时间戳时返回 None 触发批处理回退。"""
    segments = normalize_segments(result.get("segments"))
    kept = [segment for segment in segments
            if float(segment.get("end", 0) or 0) <= audio_time_sec + 1e-6]
    if not kept:
        return None
    trimmed = dict(result)
    trimmed["segments"] = kept
    trimmed["text"] = "".join(segment["text"] for segment in kept).strip()
    return trimmed if trimmed["text"] else None


def finalize_native_result(native_stream, *, audio_time_sec: float | None,
                           configured_language: str,
                           finalize_stream: bool) -> dict | None:
    """从持久 native 状态生成 final 快照，避免对同一 PCM 再跑整段 ASR。

    中间 split/silence/max_chunk 只读稳定快照，不能追加 EOF 右 padding 后
    再接真实音频；只有真实 EOF 才调用 ``finalize()``。split 提交仍按真实
    token 时间戳裁到切点；缺少时间戳时返回 ``None`` 走 `[56,13]` 回退。
    """
    request_start_ns = time.monotonic_ns()
    if finalize_stream:
        native_stream.finalize()
    result = dict(native_stream.result_dict())
    complete_ns = time.monotonic_ns()
    text = str(result.get("text", "") or "").strip()
    if not text:
        return None

    if configured_language not in ("", "auto"):
        language = configured_language
    else:
        language = _detect_qwen_lang(text) or "en"
    result.update({
        "text": text,
        "segments": normalize_segments(result.get("segments")),
        "language": language,
        # Nemotron native RNNT 与批处理 API 一样不提供这些 Whisper 指标；
        # 使用现有 Nemotron 批处理路径相同的中性值，保持过滤语义一致。
        "avg_logprob": float(result.get("avg_logprob", -0.3)),
        "avg_compression": float(result.get("avg_compression", 0.0)),
        "no_speech_prob": float(result.get("no_speech_prob", 0.0)),
        "_asr_request_start_mono_ns": request_start_ns,
        "_asr_complete_mono_ns": complete_ns,
    })
    if audio_time_sec is not None:
        return trim_transcription_result(result, audio_time_sec)
    return result


def _unload_nemotron():
    global _nemotron_model
    _nemotron_model = None

def _get_nemotron_model(model_path: str):
    """懒加载 Nemotron ASR 模型。失败时同样置 _model_load_failed。"""
    global _nemotron_model, _model_load_failed, _model_load_error_msg
    if _nemotron_model is None:
        from mlx_audio.stt import load_model
        try:
            _nemotron_model = load_model(model_path)
        except Exception as e:
            _model_load_failed = True
            _model_load_error_msg = f"Nemotron model load failed: {model_path} - {e}"
            print(f"[model-load] {_model_load_error_msg}", file=sys.stderr, flush=True)
            raise
    return _nemotron_model


def _do_transcribe_nemotron(audio, language: str = "en",
                            model_path: str = "", right_context: int = 13) -> dict:
    """Nemotron ASR 转录（非流式）。language='auto' 或空时自动检测。
    audio: WAV 路径或 float32 16kHz ndarray(generate 收 mx.array,
    ndarray 在这里转一层 — mx.array 包装零拷贝级开销)。"""
    m = _get_nemotron_model(model_path)
    if isinstance(audio, np.ndarray):
        import mlx.core as mx
        audio = mx.array(audio)
    lang_param = None if language in ("auto", "") else language
    r = m.generate(audio, language=lang_param,
                   att_context_size=[56, right_context])
    text = r.text.strip() if r.text else ""
    sentences = r.sentences if hasattr(r, 'sentences') else []
    avg_lp = -0.3
    avg_cr = 0.0
    avg_nsp = 0.0
    # Nemotron 不返回检测到的语言 — 从文本字符判断: 中文字符 > 30% 视为中文。
    # 这是必须的: 否则 probe ASR 永远返回 "en",trigger 永远用英文阈值 (3s),
    # 中文 3s 累积时 ASR 幻觉 `。` 会触发 mid-sentence 切割。
    if language not in ("auto", ""):
        detected_lang = language
    else:
        cjk_count = sum(1 for c in text if "一" <= c <= "鿿")
        detected_lang = "zh" if cjk_count > max(3, len(text) * 0.3) else "en"
    return {"text": text, "language": detected_lang, "avg_logprob": avg_lp,
            "avg_compression": avg_cr, "no_speech_prob": avg_nsp,
            "segments": normalize_segments(sentences)}


# --------------- 语言检测 ---------------

CJK_OBSERVE_LOW = 0.3    # 进入观察区
CJK_SWITCH_HIGH = 0.55   # 确认中文，触发切换

def _detect_qwen_lang(text: str) -> str | None:
    """检测是否应切换到 Qwen3。返回 'zh' / 'ja' / None。
    中文：CJK 字符占比 > 0，无日韩字符
    日文：含平假名或片假名
    """
    if not text:
        return None
    has_hira = any('ぁ' <= c <= 'ん' for c in text)
    has_kata = any('ァ' <= c <= 'ヶ' for c in text)
    has_hangul = any(('가' <= c <= '힣') or ('ㄱ' <= c <= 'ㅎ') for c in text)
    if has_hangul:
        return None  # 韩文不切
    if has_hira or has_kata:
        return 'ja'
    cn_count = sum(1 for c in text if '一' <= c <= '鿿')
    if cn_count / max(len(text), 1) > 0:
        return 'zh'
    return None

def _cjk_ratio(text: str) -> float:
    """返回文本中 CJK 字符的占比（用于中文观察区阈值判断）"""
    if not text:
        return 0.0
    cn_count = sum(1 for c in text if '一' <= c <= '鿿')
    return cn_count / len(text)

# --------------- 增量去重 ---------------

def incremental_text(old: str, new: str) -> str:
    """返回 new 相对于 old 的增量部分（word-level 最长公共前后缀匹配）"""
    if not old:
        return new
    old_words = old.split()
    new_words = new.split()
    best = 0
    for k in range(1, min(len(old_words), len(new_words)) + 1):
        if old_words[-k:] == new_words[:k]:
            best = k
    return " ".join(new_words[best:])


def strip_boundary_subword_overlap(previous: str, current: str) -> str:
    """只去掉跨 final 边界重复出现的上一词子词后缀。

    完整单词重复可能是原话，必须保留；只有重叠在 previous 中从单词内部
    开始、在 current 中恰好到词边界结束时才可证明是切点残留。
    """
    previous = previous or ""
    current = current or ""
    limit = min(24, len(previous), len(current))
    previous_folded = previous.casefold()
    current_folded = current.casefold()
    for size in range(limit, 1, -1):
        if previous_folded[-size:] != current_folded[:size]:
            continue
        previous_start = len(previous) - size
        starts_inside_previous_word = (
            previous_start > 0 and previous[previous_start - 1].isalnum())
        ends_at_current_word_boundary = (
            size == len(current) or not current[size].isalnum())
        if starts_inside_previous_word and ends_at_current_word_boundary:
            remainder = current[size:].lstrip()
            return remainder or current
    return current

# --------------- 软最大值断句辅助 ---------------

SENTENCE_END_PUNCT = set("。！？.!?")
MID_PUNCT = set("，、；：,;:")  # 中间标点：文字过长时也可作为切割点
SOFT_MAX_MIN_CHARS = 35  # 文字超过此长度时，中间标点也可切割
SOFT_MAX_ASR_COOLDOWN = 0.6  # 软最大值 ASR 重试冷却（秒）
SEG_DONE_FILE = ".whicc-done"
# 主循环内部专用标记：必须排在 EOF 前已经读取到的真实 PCM 之后处理。
# 不能直接用 ndarray 零静音代替，否则无法区分真实采集时钟与合成尾静音。
STREAM_END_PACKET = object()


@dataclass(frozen=True)
class SegmentChunk:
    """segdir PCM 及其生产端采集时钟；用于可复现的端到端回放。"""

    data: bytes
    capture_end_mono_ns: int
    sequence: int


class AudioSourceHandoff:
    """热切换时按 FIFO 保留已停止 source 的队列，直到 SENTINEL。"""

    def __init__(self) -> None:
        self._retired = deque()

    @property
    def has_retired(self) -> bool:
        return bool(self._retired)

    @staticmethod
    def _snapshot_with_boundary(source_queue):
        """复制全部残余 PCM，移除旧 sentinel，并追加唯一可达边界。"""
        # stop 已保证生产线程退出；在 Queue 自身 mutex 下做快照，既不会因
        # 满队列再丢一块 PCM，也能修复“旧 sentinel 已被消费”的复活场景。
        with source_queue.mutex:
            pending = [item for item in source_queue.queue if item is not None]
        retired = queue_mod.Queue()
        for item in pending:
            retired.put_nowait(item)
        retired.put_nowait(None)
        return retired

    def activate(self, old_source, new_source):
        """先完整停止旧 pump 再启动新源；失败则用新 queue 重启旧源。"""
        old_queue = old_source.queue
        old_source.stop()
        retired_queue = self._snapshot_with_boundary(old_queue)
        try:
            new_source.start()
        except Exception as activation_error:
            # start 可能已部分创建线程/设备，先 best-effort 收掉。
            try:
                new_source.stop()
            except Exception:  # noqa: BLE001
                pass
            # 旧 queue 以 SENTINEL 收尾并进入 retired FIFO；重启旧 source
            # 使用全新 queue，避免新 PCM 排在旧 SENTINEL 后又被边界 drain
            # 误处理。这样新源权限失败不会让字幕链永久停机。
            old_source.queue = queue_mod.Queue(
                maxsize=getattr(old_queue, "maxsize", 0))
            try:
                old_source.start()
            except Exception as rollback_error:
                # 回滚也失败时至少把完整旧尾音+边界留作当前 queue，让主
                # 循环能 final，不能卡死在已经消费过 sentinel 的空队列。
                old_source.queue = retired_queue
                raise RuntimeError(
                    f"{activation_error}; rollback old audio source failed: "
                    f"{rollback_error}") from activation_error
            self._retired.append(retired_queue)
            raise
        self._retired.append(retired_queue)
        return new_source

    def queue_for_read(self, current_queue):
        """始终先返回最老的 retired queue，避免跨 source 拼接 PCM。"""
        if self._retired:
            return self._retired[0], True
        return current_queue, False

    def finish_read(self, selected_queue, *, ended: bool) -> None:
        """读到 retired queue 的 SENTINEL 后才允许转向下一条队列。"""
        if (ended and self._retired
                and self._retired[0] is selected_queue):
            self._retired.popleft()


class DeferredAudioSwap:
    """SIGHUP 只置位；耗时/加锁的切换由主循环安全点执行。"""

    def __init__(self) -> None:
        self._pending = False

    def request(self) -> None:
        self._pending = True

    def consume(self) -> bool:
        pending = self._pending
        self._pending = False
        return pending


def audio_source_needs_swap(source, new_mode: str, bundle_id: str = "") -> bool:
    """同配置 source 进入 failed 态时也必须允许 SIGHUP 自助复活。"""
    if bool(getattr(source, "failed", False)):
        return True
    if getattr(source, "label", None) != new_mode:
        return True
    if new_mode == "application":
        return (getattr(source, "bundle_id", None) or "") != (bundle_id or "")
    return False


@dataclass
class StreamEndFlush:
    """只在明确 EOF/SENTINEL 后补静音，并对失败 final 退避重试。"""

    pending: bool = False
    failures: int = 0
    retry_after_mono: float = 0.0

    def observe(self, *, ended: bool, real_audio: bool) -> None:
        if ended:
            if not self.pending:
                self.failures = 0
                self.retry_after_mono = 0.0
            self.pending = True

    def blocks_source_read(self, *, has_pending_audio: bool) -> bool:
        """旧流尚有 PCM 时禁止读取新 source，避免跨源混成同一句。"""
        return self.pending and has_pending_audio

    def clear(self) -> None:
        self.pending = False
        self.failures = 0
        self.retry_after_mono = 0.0

    def retry_wait(self, *, now: float) -> float:
        """返回 EOF final 下一次允许重试前的剩余秒数。"""
        return max(0.0, self.retry_after_mono - now)

    def silence_packet(self, *, has_pending_speech: bool,
                       silence_submit_sec: float,
                       current_silence_sec: float = 0.0,
                       now: float) -> np.ndarray | None:
        if (not self.pending or not has_pending_speech
                or now < self.retry_after_mono):
            return None
        # 只补“尚缺的静音 + 一个 20ms VAD 帧”。final 失败恢复的 PCM
        # 已含上一轮合成静音，重试不能每次再膨胀完整 1.2s。
        seconds = max(
            0.02,
            float(silence_submit_sec) - max(0.0, current_silence_sec) + 0.02,
        )
        return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)

    def mark_failure(self, *, now: float) -> None:
        self.failures += 1
        delay = min(5.0, 0.25 * (2 ** min(self.failures - 1, 5)))
        self.retry_after_mono = now + delay

    def mark_success(self) -> None:
        self.failures = 0
        self.retry_after_mono = 0.0


def find_speech_bounds_sec(samples: np.ndarray, threshold: float,
                           frame_ms: int = 20) -> tuple[float, float] | None:
    """按 20ms 子帧返回首/末有效语音边界，未检测到语音时返回 None。"""
    frame_size = max(1, SAMPLE_RATE * frame_ms // 1000)
    active = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start:start + frame_size]
        if len(frame) and float(np.sqrt(np.mean(frame ** 2))) > threshold:
            active.append(start)
    if not active:
        return None
    return (active[0] / SAMPLE_RATE,
            min(len(samples), active[-1] + frame_size) / SAMPLE_RATE)


def find_speech_bounds_mono_ns(samples: np.ndarray, capture_end_mono_ns: int,
                               threshold: float) -> tuple[int, int]:
    """把 20ms VAD 边界换算为与采集时钟同源的单调时间。"""
    if not capture_end_mono_ns:
        return 0, 0
    bounds = find_speech_bounds_sec(samples, threshold)
    if not bounds:
        return 0, 0
    audio_start_ns = capture_end_mono_ns - int(
        len(samples) / SAMPLE_RATE * 1_000_000_000)
    return (audio_start_ns + int(bounds[0] * 1_000_000_000),
            audio_start_ns + int(bounds[1] * 1_000_000_000))


def find_submission_speech_bounds_mono_ns(
        samples: np.ndarray, capture_end_mono_ns: int, threshold: float, *,
        overlap_sample_count: int = 0) -> tuple[int, int]:
    """计算本句 VAD 边界；前置的上一句 overlap 不属于本句开口。"""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    skip = min(len(audio), max(0, int(overlap_sample_count)))
    return find_speech_bounds_mono_ns(
        audio[skip:], capture_end_mono_ns, threshold)


def prepare_submission_audio(
        current_samples: np.ndarray, tail_overlap: np.ndarray, *,
        apply_overlap: bool) -> tuple[np.ndarray, int]:
    """组装 final ASR 输入，并返回实际前置 overlap 样本数。

    失败恢复只恢复 `current_samples`；返回的计数同时用于延迟口径和 VAD
    排除上一句尾巴，避免布尔标记与实际 PCM 不一致。
    """
    current = np.asarray(current_samples, dtype=np.float32).reshape(-1)
    overlap = np.asarray(tail_overlap, dtype=np.float32).reshape(-1)
    if apply_overlap and current.size and overlap.size:
        return np.concatenate([overlap, current]), len(overlap)
    return current, 0


def prepare_split_submission_audio(
        first_part: np.ndarray, tail_overlap: np.ndarray, *,
        native_active: bool, native_fallback: bool
        ) -> tuple[np.ndarray, int]:
    """fallback 的句中切点也保留上句尾音上下文，remainder 仍保持原样。"""
    return prepare_submission_audio(
        first_part, tail_overlap,
        apply_overlap=(native_fallback and not native_active))


def precomputed_asr_timing(
        result: dict | None) -> tuple[int, int, float] | None:
    """读取探针随结果保存的真实 ASR 请求时间；旧缓存缺字段则不复用。"""
    if not result:
        return None
    try:
        start = int(result.get("_asr_request_start_mono_ns", 0) or 0)
        complete = int(result.get("_asr_complete_mono_ns", 0) or 0)
    except (TypeError, ValueError):
        return None
    if start <= 0 or complete < start:
        return None
    return start, complete, (complete - start) / 1_000_000


def update_vad_state(samples: np.ndarray, threshold: float, *,
                     has_speech: bool, silence_streak: float,
                     speech_accumulated: float,
                     frame_ms: int = 20) -> tuple[bool, float, float]:
    """逐 20ms 更新断句 VAD；不能只看较大采集块的最后 20ms。"""
    frame_size = max(1, SAMPLE_RATE * frame_ms // 1000)
    for start in range(0, len(samples), frame_size):
        frame = samples[start:start + frame_size]
        duration = len(frame) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(frame ** 2))) if len(frame) else 0.0
        if rms > threshold:
            has_speech = True
            silence_streak = 0.0
            speech_accumulated += duration
        else:
            silence_streak += duration
    return has_speech, silence_streak, speech_accumulated


def carry_remainder_vad(samples: np.ndarray, threshold: float, *,
                        has_speech: bool, silence_streak: float,
                        speech_accumulated: float) -> tuple[bool, float, float]:
    """切句后恢复 remainder 的逐帧 VAD 状态。"""
    return update_vad_state(
        samples, threshold, has_speech=has_speech,
        silence_streak=silence_streak,
        speech_accumulated=speech_accumulated)


@dataclass
class SourceRevision:
    """同一句的稳定 key 与仅在文本变化时递增的 revision。"""

    source_key: str
    revision: int = 0
    text: str = ""

    def update(self, text: str) -> tuple[str, int]:
        if text != self.text:
            self.text = text
            self.revision += 1
        return self.source_key, self.revision


class PartialEventGate:
    """限制可见草稿刷新频率；首版立即显示，同句后续最多每秒一版。"""

    def __init__(self, interval_ns: int = 1_000_000_000) -> None:
        self.interval_ns = interval_ns
        self._last_key = ""
        self._last_text = ""
        self._last_emit_ns = 0

    def should_emit(self, source_key: str, text: str, *, now_ns: int) -> bool:
        if source_key != self._last_key:
            self._last_key = source_key
            self._last_text = text
            self._last_emit_ns = now_ns
            return True
        if text == self._last_text:
            return False
        if now_ns - self._last_emit_ns < self.interval_ns:
            return False
        self._last_text = text
        self._last_emit_ns = now_ns
        return True


def resolve_run_id(configured: str, *, pid: int | None = None,
                   mono_ns: int | None = None) -> str:
    """为每次后端进程生成唯一前缀，避免重启后 source_key 串轮次。"""
    if configured:
        return configured
    return f"run-{pid if pid is not None else os.getpid()}-" \
           f"{mono_ns if mono_ns is not None else time.monotonic_ns()}"


def select_capture_end_mono_ns(*, uses_probe_snapshot: bool,
                               probe_capture_end: int,
                               latest_capture_end: int) -> int:
    """提交旧探针 PCM 时同步使用其采集末时刻，避免尾延迟被低估。"""
    if uses_probe_snapshot and probe_capture_end:
        return probe_capture_end
    return latest_capture_end


def split_probe_snapshot_audio(current: np.ndarray,
                               probe: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray, bool]:
    """提交旧探针快照，并把其后新增 PCM 原样留给下一句。"""
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    probe = np.asarray(probe, dtype=np.float32).reshape(-1)
    if (probe.size and current.size >= probe.size
            and np.array_equal(current[:probe.size], probe)):
        return probe.copy(), current[probe.size:].copy(), True
    # 理论上 probe 必为 current 前缀；若状态异常则提交当前整段，宁可
    # 延后切句也不能丢音频或复用不对应的探针转录。
    return current.copy(), np.array([], dtype=np.float32), False


def load_latency_config(path: str, cli_right_context: int | None = None) -> dict:
    """读取延迟功能开关；CLI 的右上下文覆盖配置文件。"""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        raw = {}
    configured = raw.get("nemotron_right_context", 3)
    right_context = cli_right_context if cli_right_context is not None else configured
    if right_context not in (3, 6, 13):
        right_context = 3
    return {
        "nemotron_right_context": right_context,
        "translation_priority_enabled": bool(raw.get("translation_priority_enabled", True)),
        "probe_partial_enabled": bool(raw.get("probe_partial_enabled", True)),
        "nemotron_native_streaming_enabled": bool(
            raw.get("nemotron_native_streaming_enabled", True)
        ),
    }

def find_audio_split_sec(text: str, total_sec: float, segments: list | None = None,
                         punct_set: set | None = None,
                         min_char_pos: int = 0) -> float:
    """找到 text 中 min_char_pos 之后最后一个标点对应的音频时间位置（秒）。
    优先使用 ASR segments 时间戳（精确），否则按字符比例估算。
    返回 0 表示没找到标点，不应分割。
    punct_set: 要搜索的标点集合，默认 SENTENCE_END_PUNCT
    min_char_pos: 忽略此位置之前的标点（避免切太短）
    """
    if punct_set is None:
        punct_set = SENTENCE_END_PUNCT
    last_punct_pos = -1
    for i in range(len(text) - 1, min_char_pos - 1, -1):
        if text[i] in punct_set:
            last_punct_pos = i
            break
    if last_punct_pos < 0:
        return 0.0

    # 方案 A: 用 ASR segments 时间戳 (精确)
    if segments:
        char_pos = 0
        for seg_index, seg in enumerate(segments):
            seg_text = seg.get("text", "")
            seg_start = char_pos
            seg_end = char_pos + len(seg_text)
            if last_punct_pos >= seg_start and last_punct_pos < seg_end:
                punct_end = float(seg.get("end", 0.0) or 0.0)
                # Nemotron 的 80ms 对齐帧可能同时包含句号和下一词的
                # SentencePiece token。此时按句号 end 切 PCM 会把下一词
                # 拦腰切开（例如 "Ooh" 变成 "Oo" + "h"）。这个时间戳
                # 没有可表达的安全边界，应放弃本次标点切句，继续等整词
                # 边界；final 的标点/静音质量阈值保持不变。
                for following in segments[seg_index + 1:]:
                    following_start = float(
                        following.get("start", punct_end) or 0.0)
                    if following_start >= punct_end - 1e-6:
                        break
                    if any(ch.isalnum()
                           for ch in following.get("text", "")):
                        return 0.0
                return punct_end
            char_pos = seg_end
        # fallback：标点在最后一个 segment
        if segments:
            return segments[-1].get("end", total_sec)

    # 方案 B：字符比例估算
    return total_sec * (last_punct_pos + 1) / len(text)

# --------------- 段文件消费 ---------------

def cleanup_seg_dir():
    os.makedirs(SEG_DIR, exist_ok=True)
    for f in os.listdir(SEG_DIR):
        if (f.endswith(".pcm") or f.endswith(".meta.json")
                or f == SEG_DONE_FILE):
            try:
                os.unlink(os.path.join(SEG_DIR, f))
            except OSError:
                pass

def read_segments(next_seg: int) -> tuple[list[SegmentChunk], int, bool]:
    """读取 PCM 与生产端时钟；done marker 且消费到文件尾时结束。"""
    chunks = []
    while True:
        path = os.path.join(SEG_DIR, f"seg-{next_seg:06d}.pcm")
        meta_path = os.path.join(
            SEG_DIR, f"seg-{next_seg:06d}.meta.json")
        if not os.path.exists(path):
            break
        try:
            with open(path, "rb") as f:
                data = f.read()
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            os.unlink(path)
            if os.path.exists(meta_path):
                os.unlink(meta_path)
        except (OSError, IOError, json.JSONDecodeError):
            break
        if data:
            chunks.append(SegmentChunk(
                data=data,
                capture_end_mono_ns=max(
                    0, int(metadata.get("capture_end_mono_ns", 0))),
                sequence=int(metadata.get("sequence", next_seg)),
            ))
        next_seg += 1
    next_path = os.path.join(SEG_DIR, f"seg-{next_seg:06d}.pcm")
    ended = (os.path.exists(os.path.join(SEG_DIR, SEG_DONE_FILE))
             and not os.path.exists(next_path))
    return chunks, next_seg, ended

def drain_audio_queue(q: "queue_mod.Queue",
                      first_timeout: float) -> tuple[list, bool]:
    """live 模式(system/mic)的读取路径:直接消费 AudioSource.queue。

    阻塞至多 first_timeout 等首个 chunk,拿到后无等待排空当前可用的
    全部 chunks。chunk 到达即返回(采集回调粒度 ~0.1s),替代旧的
    SegDirWriter→/tmp 段文件→read_segments 磁盘往返(1s 段聚合 +
    0.15s 轮询,每段最多 +1.15s 延迟,且 /tmp 被系统清理会断链)。

    SENTINEL(None) = 当前 source 流结束(audio swap 时旧 source 冲刷
    完毕)——丢弃并停止 drain,下一轮主循环读到的是(swap 后的)新
    source.queue。
    """
    chunks = []
    ended = False
    try:
        first = q.get(timeout=first_timeout)
        if first is None:
            return chunks, True
        chunks.append(first)
        while True:
            nxt = q.get_nowait()
            if nxt is None:
                ended = True
                break
            chunks.append(nxt)
    except queue_mod.Empty:
        pass
    return chunks, ended

# --------------- Event Logger ---------------

class EventLogger:
    """JSONL 结构化事件日志器"""

    def __init__(self, path: str):
        self.path = path
        # audio.py 的采集线程(status_callback)与主循环共用同一 logger,
        # 加锁防止 JSONL 行交错损坏(Swift EventWatcher 按行解析)。
        self._write_lock = threading.Lock()
        if path:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            self._f = open(path, "a", encoding="utf-8")
        else:
            self._f = None

    def write(self, event: dict):
        if self._f:
            event.setdefault("event_mono_ns", time.monotonic_ns())
            event.setdefault("event_wall_ms", int(time.time() * 1000))
            line = json.dumps(event, ensure_ascii=False) + "\n"
            with self._write_lock:
                self._f.write(line)
                self._f.flush()

    def log_status(self, status: str, **extra):
        event = {"event_type": "status", "status": status}
        event.update(extra)
        self.write(event)

    def log_reject(self, seg_start, seg_end, audio_start_sec, audio_end_sec, chunk_sec,
                   submit_wall_time, prompt_mode, prompt_chars, reject_reason,
                   submit_reason="", buffered_sec=0, speech_sec=0,
                   trailing_silence_sec=0, prompt_hash="", tail_source_chars=0,
                   language="", avg_logprob=0, avg_compression=0, no_speech_prob=0,
                   text=""):
        self.write({
            "event_type": "reject",
            "text": text,
            "seg_start": seg_start,
            "seg_end": seg_end,
            "audio_start_sec": round(audio_start_sec, 3),
            "audio_end_sec": round(audio_end_sec, 3),
            "chunk_sec": round(chunk_sec, 3),
            "submit_wall_time": round(submit_wall_time, 3),
            "prompt_mode": prompt_mode,
            "prompt_chars": prompt_chars,
            "reject_reason": reject_reason,
            "submit_reason": submit_reason,
            "buffered_sec": round(buffered_sec, 3),
            "speech_sec": round(speech_sec, 3),
            "trailing_silence_sec": round(trailing_silence_sec, 3),
            "prompt_hash": prompt_hash,
            "tail_source_chars": tail_source_chars,
            "language": language,
            "avg_logprob": round(avg_logprob, 4),
            "avg_compression": round(avg_compression, 4),
            "no_speech_prob": round(no_speech_prob, 4),
            "accepted": False,
        })

    def log_final(self, seg_start, seg_end, audio_start_sec, audio_end_sec, chunk_sec,
                  submit_wall_time, final_wall_time, transcribe_ms,
                  prompt_mode, prompt_chars, text,
                  submit_reason="", buffered_sec=0, speech_sec=0,
                  trailing_silence_sec=0, prompt_hash="", tail_source_chars=0,
                  asm_ms=0, postproc_ms=0,
                  language="", avg_logprob=0, avg_compression=0, no_speech_prob=0,
                  source_key="", revision=0, speech_start_mono_ns=0,
                  speech_end_mono_ns=0, asr_request_start_mono_ns=0,
                  asr_complete_mono_ns=0, capture_end_mono_ns=0):
        event = {
            "event_type": "final",
            "source_key": source_key,
            "revision": revision,
            "speech_start_mono_ns": speech_start_mono_ns,
            "speech_end_mono_ns": speech_end_mono_ns,
            "asr_request_start_mono_ns": asr_request_start_mono_ns,
            "asr_complete_mono_ns": asr_complete_mono_ns,
            "capture_end_mono_ns": capture_end_mono_ns,
            "seg_start": seg_start,
            "seg_end": seg_end,
            "audio_start_sec": round(audio_start_sec, 3),
            "audio_end_sec": round(audio_end_sec, 3),
            "chunk_sec": round(chunk_sec, 3),
            "submit_wall_time": round(submit_wall_time, 3),
            "final_wall_time": round(final_wall_time, 3),
            "transcribe_ms": round(transcribe_ms, 1),
            "asm_ms": round(asm_ms, 1),
            "postproc_ms": round(postproc_ms, 1),
            "relative_confirm_latency_sec": round(final_wall_time - submit_wall_time, 3),
            "prompt_mode": prompt_mode,
            "prompt_chars": prompt_chars,
            "submit_reason": submit_reason,
            "buffered_sec": round(buffered_sec, 3),
            "speech_sec": round(speech_sec, 3),
            "trailing_silence_sec": round(trailing_silence_sec, 3),
            "prompt_hash": prompt_hash,
            "tail_source_chars": tail_source_chars,
            "text": text,
            "language": language,
            "avg_logprob": round(avg_logprob, 4),
            "avg_compression": round(avg_compression, 4),
            "no_speech_prob": round(no_speech_prob, 4),
            "accepted": True,
        }
        self.write(event)

    def log_partial(self, seg_start: int, seg_end: int,
                    audio_start_sec: float, audio_end_sec: float,
                    text: str, *, source_key: str = "", revision: int = 0,
                    event_type: str = "partial",
                    speech_start_mono_ns: int = 0,
                    speech_end_mono_ns: int = 0,
                    is_probe: bool = False,
                    asr_request_start_mono_ns: int = 0,
                    asr_complete_mono_ns: int = 0,
                    capture_end_mono_ns: int = 0):
        self.write({
            "event_type": event_type,
            "source_key": source_key,
            "revision": revision,
            "speech_start_mono_ns": speech_start_mono_ns,
            "speech_end_mono_ns": speech_end_mono_ns,
            "asr_request_start_mono_ns": asr_request_start_mono_ns,
            "asr_complete_mono_ns": asr_complete_mono_ns,
            "capture_end_mono_ns": capture_end_mono_ns,
            "is_probe": is_probe,
            "seg_start": seg_start,
            "seg_end": seg_end,
            "audio_start_sec": round(audio_start_sec, 3),
            "audio_end_sec": round(audio_end_sec, 3),
            "text": text,
            "accepted": False,
        })

    def close(self):
        if self._f:
            self._f.close()

# --------------- Metrics ---------------

class Metrics:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.chunks = 0
        self.rejects = 0
        self.total_transcribe_ms = 0.0
        self.total_chunk_sec = 0.0
        self.last_report = time.monotonic()

    def record(self, transcribe_ms: float, chunk_sec: float):
        if not self.enabled:
            return
        self.chunks += 1
        self.total_transcribe_ms += transcribe_ms
        self.total_chunk_sec += chunk_sec
        now = time.monotonic()
        if now - self.last_report >= 30:
            self.report()
            self.last_report = now

    def record_reject(self):
        if self.enabled:
            self.rejects += 1

    def report(self):
        if not self.enabled or self.chunks == 0:
            return
        avg_t = self.total_transcribe_ms / self.chunks / 1000
        avg_c = self.total_chunk_sec / self.chunks
        print(f"\n[stats] chunks={self.chunks} rejects={self.rejects} "
              f"avg_transcribe={avg_t:.1f}s avg_chunk={avg_c:.1f}s",
              file=sys.stderr, flush=True)

# --------------- 主循环 ---------------

def main():
    parser = argparse.ArgumentParser(description="whicc - 系统音频实时语音识别")
    # 原有参数
    parser.add_argument("--save-wav", metavar="DIR", help="[未实现] 保存每个 chunk 的 WAV 到指定目录")
    parser.add_argument("--dump-raw", action="store_true", help="stderr 输出 ASR 原始结果")
    parser.add_argument("--dump-filtered", action="store_true", help="stderr 输出过滤后结果")
    parser.add_argument("--stats", action="store_true", help="输出性能指标摘要")
    # 新增：扫参参数
    parser.add_argument("--run-id", default="",
                        help="运行 ID；空值自动生成进程唯一 ID，避免重启后事件串句")
    parser.add_argument("--min-chunk-sec", type=float, default=MIN_CHUNK_SEC)
    parser.add_argument("--max-chunk-sec", type=float, default=MAX_CHUNK_SEC)
    parser.add_argument("--overlap-sec", type=float, default=OVERLAP_SEC)
    parser.add_argument("--silence-threshold", type=float, default=SILENCE_THRESHOLD)
    parser.add_argument("--silence-submit-sec", type=float, default=SILENCE_SUBMIT_SEC)
    parser.add_argument("--no-speech-threshold", type=float, default=NO_SPEECH_THRESHOLD)
    parser.add_argument("--temperature", default="0.0", help="温度，逗号分隔，如 '0.0' 或 '0.0,0.2'")
    parser.add_argument("--prompt-mode", choices=[PROMPT_MODE_FIXED, PROMPT_MODE_TAIL, PROMPT_MODE_NONE],
                        default=PROMPT_MODE_TAIL, help="prompt 策略（默认 tail）")
    parser.add_argument("--prompt-tail-chars", type=int, default=160, help="tail 模式截取确认文本字符数")
    parser.add_argument("--initial-prompt", default=INITIAL_PROMPT, help="Qwen3 ASR initial_prompt 基底文本")
    parser.add_argument("--language", default="en", help="源语言（默认 en；qwen3 可设为 auto 让模型自动识别）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"ASR 模型 ID 或本地路径（默认 {DEFAULT_MODEL}，"
                             f"但优先级低于 --model-state）")
    parser.add_argument("--model-state", default="",
                        help="[BackendLauncher 内部用] model_state.json 路径，"
                             "其中的 current_model 覆盖 --model")
    parser.add_argument("--models-dir", default="",
                        help="[BackendLauncher 内部用] 本地模型目录（~/Library/.../whicc/models/）")
    # 新增：结构化输出
    parser.add_argument("--events-jsonl", metavar="FILE", help="JSONL 事件日志路径")
    parser.add_argument("--output-text", metavar="FILE", help="final-only 输出文件路径")
    parser.add_argument("--audio-bin", default=AUDIO_BIN, help="音频捕获二进制路径（audiotee）")
    parser.add_argument("--audio-source", default="system",
                        choices=["system", "mic", "application", "segdir"],
                        help="音频源:system=全部系统声音(默认,audiotee),"
                             "mic=麦克风(sounddevice),"
                             "application=指定应用(audiotee --include-processes),"
                             "segdir=轮询 SEG_DIR 段文件(离线评估,"
                             "由 tools/whicc_file_audio.py 等外部进程投喂)")
    parser.add_argument("--audio-app-bundle-id", default="",
                        help="application 模式的目标 Bundle ID"
                             " (也可由 lang_config.json 的 audio_app_bundle_id 提供)")
    parser.add_argument("--audio-app-display-name", default="",
                        help="application 模式展示名"
                             " (也可由 lang_config.json 的 audio_app_display_name 提供)")
    parser.add_argument("--mic-device", default=None,
                        help="麦克风设备索引或名字(传给 sounddevice);默认系统默认")
    parser.add_argument("--dual-model", action="store_true",
                        help="预加载 Nemotron + Qwen3 双模型（中文秒切，耗内存）")
    parser.add_argument("--lang-config", default="/tmp/whicc-out/lang_config.json",
                        help="macui 与后端共享的配置文件")
    parser.add_argument("--nemotron-right-context", type=int, choices=[3, 6, 13],
                        default=None, help="Nemotron 右上下文；优先于配置文件")
    args = parser.parse_args()
    args.run_id = resolve_run_id(args.run_id)
    latency_cfg = load_latency_config(args.lang_config, args.nemotron_right_context)
    args.nemotron_right_context = latency_cfg["nemotron_right_context"]

    # 解析模型路径：优先级 model_state.json > --model 参数 > 内置默认
    # 与 review #1/#3 修过的 lang_config 模式一致——只动自己关心的字段。
    # models_dir 来源：--models-dir > model_state.json > 老项目内 MODEL_DIR（兜底）
    state = read_model_state(args.model_state) if args.model_state else {}
    models_dir = args.models_dir or resolve_models_dir(state, MODEL_DIR)
    if state:
        args.model = resolve_model_id(state)
        print(f"[model-state] 启动主模型={args.model} "
              f"(non_chinese_asr 槽位 > current_model > 默认)", flush=True)
    resolved_model = args.model

    def _local_model_path(model_id: str) -> str:
        # 之前用老 MODEL_DIR（项目内 ../models/）拼路径，导致 --models-dir
        # 形同虚设。修：用上面算出的 models_dir（--models-dir > model_state > 兜底）。
        return os.path.join(models_dir, model_id.replace("/", "--"))

    # 启动候选链(打包模式):主槽 → 中文槽 → 内置默认。主槽模型残缺
    # (下载中断,只有 config/tokenizer 没权重)时自动用下一个**完整**的
    # 本地模型顶上 — 之前 os.path.isdir 就当"本地模型存在",加载必失败,
    # 后端退出,用户已下载完好的另一个模型从没被试过,"说什么都没字幕"。
    model_candidates = [args.model]
    if state:
        for cid in (resolve_chinese_model_id(state, QWEN3_MODEL), DEFAULT_MODEL):
            if cid and cid not in model_candidates:
                model_candidates.append(cid)

    no_complete_local_model = False
    local_model = _local_model_path(args.model)
    if _is_model_complete(local_model):
        resolved_model = local_model
        print(f"使用本地模型: {resolved_model}", flush=True)
    elif os.path.isdir(args.model):
        # CLI 直接传本地目录(离线评估场景),尊重用户,不做完整性把关
        resolved_model = args.model
        print(f"使用本地模型: {resolved_model}", flush=True)
    elif state:
        # 打包模式:主槽不可用 → 候选链找完整的顶上。不回退成 HF ID —
        # 那会让 mlx 绕开 app 的下载管理直接联网拉 1GB+,启动卡死;
        # 模型下载统一归 model_downloader 管。
        if os.path.isdir(local_model):
            print(f"[model-state] 主模型残缺(未下载完),跳过: {local_model}",
                  flush=True)
        fb_id = next((cid for cid in model_candidates[1:]
                      if _is_model_complete(_local_model_path(cid))), None)
        if fb_id is not None:
            resolved_model = _local_model_path(fb_id)
            print(f"[model-state] 降级到已下载完的模型: {fb_id}", flush=True)
        else:
            # 没有任何完整本地模型 — logger 建立后统一 exit 3(等模型),
            # BackendLauncher 监控会弹"去设置页下载"指引。
            no_complete_local_model = True
    else:
        print(f"使用 HF 模型: {resolved_model}", flush=True)

    # 自动应用模型预设（CLI 参数优先，预设做默认值）
    preset = None
    for name, p in MODEL_PRESETS.items():
        if name in resolved_model:
            preset = p
            break
    if preset:
        if args.no_speech_threshold == 0.45:  # 还是默认值，用预设覆盖
            args.no_speech_threshold = preset["no_speech"]
        if args.min_chunk_sec == 2.0:
            args.min_chunk_sec = preset["min_chunk"]
        if args.max_chunk_sec == 5.5:
            args.max_chunk_sec = preset["max_chunk"]
        print(f"  预设: no_speech={args.no_speech_threshold}, chunk={args.min_chunk_sec}-{args.max_chunk_sec}s", flush=True)

    # 检测后端类型
    resolved_backend = _detect_backend(resolved_model)
    print(f"  后端: {resolved_backend}", flush=True)

    # 音频源:
    #  - system/mic/application: 进程内采集线程,主循环直接消费内存 queue
    #  - segdir:     外部进程写 SEG_DIR 段文件(tools/whicc_file_audio.py
    #                离线评估),主循环轮询 read_segments() — 文件协议只保留
    #                在这条评估入口
    from audio import CapturedAudio, make_source

    def _read_audio_app_config() -> tuple[str, str, str]:
        """从 lang_config.json 读 audio_source / Bundle ID / 展示名。

        返回 (mode, bundle_id, display_name)。文件缺失时回退 CLI 参数。
        """
        mode = args.audio_source
        bundle_id = args.audio_app_bundle_id or ""
        display_name = args.audio_app_display_name or ""
        try:
            with open(args.lang_config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mode = cfg.get("audio_source", mode) or mode
            bundle_id = cfg.get("audio_app_bundle_id", bundle_id) or bundle_id
            display_name = (
                cfg.get("audio_app_display_name", display_name) or display_name
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return mode, bundle_id, display_name

    startup_mode, startup_bundle, startup_name = _read_audio_app_config()
    # BackendLauncher 固定传 --audio-source system(argparse 默认值同为
    # system),只在这种情况下用 lang_config 恢复用户上次选择;显式传
    # --audio-source mic / segdir 的开发调试用法保持原语义不被覆盖。
    if args.audio_source == "system" and startup_mode in (
        "system", "mic", "application"
    ):
        args.audio_source = startup_mode
    use_segdir = args.audio_source == "segdir"
    audio_source = None

    # 初始化日志器（在模型加载前，这样 status 事件能被 overlay 接收）
    logger = EventLogger(args.events_jsonl)
    logger.log_status("loading_model")

    def _status_cb(status: str, **extra) -> None:
        logger.log_status(status, **extra)

    if not use_segdir:
        audio_source = make_source(
            mode=args.audio_source,
            audiotee_path=args.audio_bin,
            mic_device=args.mic_device,
            bundle_id=startup_bundle or None,
            display_name=startup_name or None,
            status_callback=_status_cb,
        )

    if no_complete_local_model:
        # 主槽/中文槽/默认全都没下载完 — 音频采集还没 start,直接退。
        # exit 3 = "等模型",BackendLauncher 按 code 弹下载指引,
        # 检测到新 .complete 出现后自动重启(见 waitedResourceReady)。
        print("[model-load] 本地无完整 ASR 模型(主槽/中文槽/默认均未下载完)",
              file=sys.stderr, flush=True)
        print("[model-load]   请在 macui 设置 → 模型 页下载", file=sys.stderr,
              flush=True)
        logger.log_status("model_load_failed")
        logger.log_status("no complete local model")
        sys.exit(3)

    # ── 启动加速: 音频采集先行 ──
    # 采集在模型加载/warmup(共 3-5s)期间并行进行,音频进 source.queue
    # 积累(容量 ~20s,远大于加载时长)。主循环开始时已有存量音频可
    # 处理 → 首条字幕出现时间提前 ≈ 整个模型加载时长。
    # 之前的顺序是 加载模型 → warmup → 启动采集,用户开 app 后说的
    # 前几秒话全部丢失。
    # status 事件序列(loading_model → ready → listening)保持不变,
    # macui 的 banner 逻辑无感知。
    if not use_segdir:
        try:
            audio_source.start()
        except RuntimeError as e:
            print(f"音频源启动失败: {e}", file=sys.stderr)
            sys.exit(1)

    def _exit_with_audio_cleanup(code: int):
        """模型加载失败退出前把已启动的采集收干净
        (audiotee 子进程 / sounddevice stream)。"""
        if audio_source is not None:
            try:
                audio_source.stop()
            except Exception:  # noqa: BLE001
                pass
        sys.exit(code)

    #   whicc.py 不应该 crash 整个进程。加载失败时沿候选链找下一个
    #   路径不同且完整的本地模型 — 之前 fallback 固定指回 nemotron
    #   默认路径,主模型本身就是 nemotron 时等于原地重试,必然二次
    #   失败 → 后端退出,已下载完好的 Qwen3 从没被试过。
    fallback_used = False
    try:
        if resolved_backend == "qwen3":
            print("正在加载 Qwen3 ASR 模型...", flush=True)
            _get_qwen3_model(resolved_model)
        else:  # nemotron (默认)
            print("正在加载 Nemotron ASR 模型...", flush=True)
            _get_nemotron_model(resolved_model)
    except Exception as e1:
        fallback_used = True
        fallback_path = next(
            (p for p in (_local_model_path(cid) for cid in model_candidates)
             if p != resolved_model and _is_model_complete(p)), None)
        if fallback_path is not None:
            fb_backend = _detect_backend(fallback_path)
            print(f"[fallback] 主模型加载失败,切到本地 {fb_backend}: "
                  f"{fallback_path}", flush=True)
            try:
                if fb_backend == "qwen3":
                    _get_qwen3_model(fallback_path)
                else:
                    _get_nemotron_model(fallback_path)
                resolved_model = fallback_path
                resolved_backend = fb_backend
                logger.log_status("model_fallback")
            except Exception as e2:
                # 连 fallback 都失败 — 致命错误, 但仍然 log 给 macui,
                # 不让进程静默 crash。
                print(
                    f"[model-load] FATAL both primary and fallback "
                    f"model load failed: {e2}",
                    file=sys.stderr,
                    flush=True,
                )
                logger.log_status("model_load_failed")
                logger.log_status(str(e2))
                # 不 raise — 让 whicc.py 干净退出留下 log 痕迹,
                # macui 能读到 model_load_failed status 提示用户。
                # exit 3 = "等模型"(未下载/残缺),不是程序故障 —
                # BackendLauncher 监控按 code 区分:3 → 提示用户去下载
                # + 检测到模型下载完成后立即重启;其他 → 崩溃重启。
                _exit_with_audio_cleanup(3)
        else:
            # 没有其他完整本地模型可退 → 让用户去 macui 下载
            print(f"[model-load] 主模型加载失败且无其他完整本地模型: {e1}",
                  file=sys.stderr, flush=True)
            print(f"[model-load]   请在 macui 设置里下载模型",
                  file=sys.stderr, flush=True)
            logger.log_status("model_load_failed")
            logger.log_status(str(e1))
            _exit_with_audio_cleanup(3)  # 3 = 等模型,见上方注释

    if fallback_used:
        print(f"[model-load] fallback 成功,继续运行", flush=True)

    # 两个切换目标都先解析成真实可用路径。旧逻辑从 Qwen3 启动时把
    # nemotron_model 留空，连续英文三句后会异步加载空路径并切坏后端。
    if resolved_backend == "nemotron":
        nemotron_model = resolved_model
    else:
        default_nemotron_local = _local_model_path(DEFAULT_MODEL)
        nemotron_model = (
            default_nemotron_local
            if _is_model_complete(default_nemotron_local) else ""
        )
    qwen3_model_id = resolve_chinese_model_id(state, QWEN3_MODEL)
    qwen3_local = os.path.join(models_dir, qwen3_model_id.replace("/", "--"))
    qwen3_path = (resolved_model if resolved_backend == "qwen3" else
                  (qwen3_local if os.path.isdir(qwen3_local)
                   else qwen3_model_id))
    qwen3_fallback = None
    zh_streak = 0           # 连续中文检测计数
    ja_streak = 0           # 连续日文检测计数
    en_streak = 0           # 连续非中日文计数（切回 Nemotron 需要更高阈值）
    pending_switch = None   # "to_qwen3" / "to_nemotron" / None（异步加载中）
    switch_state = AsyncModelLoadState()
    if resolved_backend == "nemotron":
        # 中文切换目标:优先 macui 槽位(chinese_asr),未配置回退内置常量
        # — 之前硬编码 QWEN3_MODEL,UI 的"中文语音识别"槽位选了也不生效。
        if qwen3_model_id != QWEN3_MODEL:
            print(f"[model-state] chinese_asr={qwen3_model_id}", flush=True)
        if args.dual_model:
            print(f"预加载 Qwen3 中文备用（双模型模式）: {qwen3_path}", flush=True)
            qwen3_fallback = _get_qwen3_model(qwen3_path)
        else:
            print(f"Qwen3 中文备用就绪（单模型模式，按需加载）: {qwen3_path}", flush=True)
            qwen3_fallback = True  # 标记可用，但不预加载

    # 模型 warmup：对空音频跑一次推理，吸收 Metal kernel 编译延迟
    # (数组直传,不再经临时 WAV 落盘)
    print("模型预热中...", flush=True)
    _warmup_samples = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    try:
        do_transcribe(_warmup_samples, language="en",
                      model=resolved_model, backend=resolved_backend,
                      nemotron_right_context=args.nemotron_right_context)
    except Exception:
        pass

    native_stream = None
    native_stream_fallback_until_commit = False
    if (latency_cfg["nemotron_native_streaming_enabled"]
            and resolved_backend == "nemotron"):
        try:
            from nemotron_stream import NemotronStream
            native_stream = NemotronStream(
                _get_nemotron_model(resolved_model),
                language=args.language,
                right_context=args.nemotron_right_context,
            )
            print(f"[nemotron-stream] enabled [56,{args.nemotron_right_context}]",
                  flush=True)
        except Exception as exc:
            # 原生流初始化失败后，本句话必须走计划约定的质量回退档。
            # 仅让 native_stream 保持 None 会继续使用用户选择的 3/6 档，
            # 与日志中的“回退批处理 [56,13]”不一致。
            native_stream_fallback_until_commit = True
            print(f"[nemotron-stream] 初始化失败，回退批处理 [56,13]: {exc}",
                  file=sys.stderr, flush=True)

    def deactivate_native_stream(*, release_model: bool) -> None:
        """离开 Nemotron 时清状态；单模型模式同时释放模型强引用。"""
        nonlocal native_stream, native_stream_fallback_until_commit
        if native_stream is not None:
            try:
                if release_model:
                    native_stream.close()
                    native_stream = None
                else:
                    native_stream.reset(args.language,
                                        args.nemotron_right_context)
            except Exception as exc:  # noqa: BLE001
                logger.log_status("nemotron_stream_reset_fallback",
                                  error=str(exc))
                if release_model:
                    native_stream = None
        native_stream_fallback_until_commit = False

    def activate_native_stream(model_path: str) -> None:
        """切回 Nemotron 时重建流；当前未提交 PCM 先由批处理保底。"""
        nonlocal native_stream, native_stream_fallback_until_commit
        if not latency_cfg["nemotron_native_streaming_enabled"]:
            return
        try:
            from nemotron_stream import NemotronStream
            model = _get_nemotron_model(model_path)
            if native_stream is not None and native_stream.model is model:
                native_stream.reset(args.language,
                                    args.nemotron_right_context)
            else:
                if native_stream is not None:
                    native_stream.close()
                native_stream = NemotronStream(
                    model, language=args.language,
                    right_context=args.nemotron_right_context)
            # 语言切换发生在 submit_chunk 内。外层可能刚按标点切出
            # remainder，新流没有听过它；保底到这段 PCM 的下一次 final，
            # 避免 native 草稿从下一句中间开始。
            native_stream_fallback_until_commit = True
        except Exception as exc:
            native_stream_fallback_until_commit = True
            logger.log_status("nemotron_stream_fallback", error=str(exc))

    # 注:BackendLauncher.waitForASRReady 扫日志找"模型就绪"关键词,
    # 这行文案不要改。音频采集已在模型加载前启动(启动加速),这里
    # 只是宣告主循环即将开始消费。
    print("模型就绪。启动系统音频捕获...\n", flush=True)
    logger.log_status("ready")

    text_out = open(args.output_text, "w", encoding="utf-8") if args.output_text else None

    metrics = Metrics(args.stats)

    # segdir 模式(离线评估)无进程内采集,清空段目录等外部投喂;
    # live 模式的 audio_source.start() 已提前到模型加载之前。
    if use_segdir:
        cleanup_seg_dir()
    logger.log_status("listening")

    # macui HUD ASR chip 点击切 audio source 时发 SIGHUP (pkill -1 -f
    # whicc.py)。handler 重新读 lang_config.json 的 audio_source 键
    # + swap audio_source。旧 source 先完全停止；其残余 PCM 被快照进
    # audio_handoff FIFO，主循环读完残余 chunks + 唯一 SENTINEL
    # 并完成旧句 final 后，才会读取新 source 的 queue。
    #
    # SIGHUP handler 只置位；真正切换在主循环进入 queue.get 前的安全点
    # 执行。signal 若打断 Queue 持锁区，handler 再碰同一非重入 mutex 会
    # 自死锁，因此 handler 内禁止 JSON IO、join、make_source 和 queue。
    import threading as _threading
    _audio_swap_lock = _threading.Lock()
    audio_handoff = AudioSourceHandoff()
    deferred_audio_swap = DeferredAudioSwap()

    def _swap_audio_source(
        new_mode: str,
        bundle_id: str = "",
        display_name: str = "",
    ) -> None:
        """停掉旧 audio source,启动新 source,替换 audio_source 引用。
        失败时 log error 但不崩溃 (whicc.py 继续跑 — 用户改错了能重试)。
        """
        nonlocal audio_source
        # 当前只从主循环安全点调用；保留非阻塞锁作为未来其他调用者的
        # 重入保护。连点产生的多个信号已由 DeferredAudioSwap 合并。
        if not _audio_swap_lock.acquire(blocking=False):
            logger.log_status("audio swap busy, ignored")
            return
        try:
            if audio_source is None:
                return  # segdir 模式没有进程内 source,不支持热切换
            if new_mode not in ("system", "mic", "application"):
                logger.log_status(f"unknown audio mode: {new_mode}")
                return
            if new_mode == "application" and not bundle_id:
                logger.log_status("application mode missing bundle_id")
                return
            # 同 mode + 同 bundle 无需重启;但 source 已进入 failed 态
            # (supervisor 三连败停机)时放行,让用户重选同一应用能复活捕获。
            source_failed = bool(getattr(audio_source, "failed", False))
            same_app = (
                new_mode == "application"
                and getattr(audio_source, "bundle_id", None) == bundle_id
                and audio_source.label == "application"
            )
            if (not source_failed) and audio_source.label == new_mode and (
                new_mode != "application" or same_app
            ):
                return
            try:
                old_source = audio_source
                # activate() 先停稳旧 pump 并快照尾音，再启动新 source；
                # 新源失败时旧 source 用新 queue 自动重启，旧尾音仍先 final。
                new_source = make_source(
                    mode=new_mode,
                    audiotee_path=args.audio_bin,
                    mic_device=args.mic_device,
                    bundle_id=bundle_id or None,
                    display_name=display_name or None,
                    status_callback=_status_cb,
                )
                audio_source = audio_handoff.activate(
                    old_source, new_source)
                if new_mode == "application":
                    logger.log_status(
                        f"audio source → application:{display_name or bundle_id}",
                        bundle_id=bundle_id,
                    )
                else:
                    logger.log_status(f"audio source → {new_mode}")
            except Exception as e:
                # 新源启动失败时 activate 已尝试重启旧 source；任何残余
                # PCM 仍有独立边界，不会与下一 source 混句或卡空退休队列。
                logger.log_status(f"audio swap to {new_mode} failed: {e}")
        finally:
            _audio_swap_lock.release()

    def _apply_deferred_audio_swap() -> None:
        """主循环安全点读取最终配置，并执行至多一次 source 切换。"""
        if audio_source is None:
            return  # segdir 模式(离线评估)没有进程内 source,不支持热切换
        try:
            new_mode, bundle_id, display_name = _read_audio_app_config()
            need_swap = audio_source_needs_swap(
                audio_source, new_mode, bundle_id)
            if need_swap:
                _swap_audio_source(new_mode, bundle_id, display_name)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.log_status(f"audio swap: bad config: {e}")

    def _sighup_audio_swap(_signum, _frame) -> None:
        """异步信号安全路径：只合并请求，不读取文件或接触 Queue。"""
        deferred_audio_swap.request()

    signal.signal(signal.SIGHUP, _sighup_audio_swap)

    overlap_samples = int(SAMPLE_RATE * args.overlap_sec)
    next_seg = 0            # segdir 模式的段文件游标
    samples_ingested = 0    # 已消费样本总数 → 虚拟段号 = // SAMPLE_RATE

    # 探针 ASR 缓存(soft_max/标点断句共享;submit 音频一致时复用免重转)
    sm_text = None          # 探针转录文本(strip 过)
    sm_segments = None      # 探针 segments(时间戳,find_audio_split_sec 用)
    sm_result = None        # 探针完整转录 dict,submit_chunk 复用
    sm_audio = np.array([], dtype=np.float32)  # 探针时点的音频快照,submit 复用避免后续 collected 增长导致 ASR 多识别尾巴
    sm_capture_end_mono_ns = 0  # 必须与 sm_audio 同步，不能拿后到 packet 的时间
    sm_language = "en"      # 默认英文,probe ASR 后会被覆盖
    stable_punct_candidate = None  # 已确认或等待下一次探针确认的句末前缀
    sm_punct_stable = False
    source_sequence = 0
    source_revision = SourceRevision(f"{args.run_id}:{source_sequence}")
    partial_event_gate = PartialEventGate()

    def advance_source_revision() -> None:
        nonlocal source_sequence, source_revision
        source_sequence += 1
        source_revision = SourceRevision(f"{args.run_id}:{source_sequence}")

    # 自适应 chunk 状态
    collected = []          # list[ndarray]
    collected_count = 0     # total samples
    has_speech = False
    silence_streak = 0.0
    speech_accumulated = 0.0  # 累积语音时长
    chunk_first_seg = 0     # 当前 chunk 的起始段号
    soft_max_remainder = np.array([], dtype=np.float32)  # 软最大值断句后的剩余音频
    soft_max_last_asr = 0.0   # 上次软最大值 ASR 的 monotonic 时间（0 = 未执行）
    latest_capture_end_mono_ns = 0
    last_audio_sequence = None

    # Partial/Final 状态
    confirmed_text = ""
    tail_buffer = ""   # 只来自已确认 final 的 tail prompt 文本
    prev_ended_with_punct = False  # 上一个 chunk 以句末标点结尾 → 下一个更快提交

    # 崩溃恢复计数
    restart_count = 0
    MAX_RESTARTS = 5

    # 静默卡死检测：None = 还没收到过任何数据,跳过卡死计时
    # (用户场景里长时间静音是正常的,启动等几秒才开始说话)。
    # 收到首段数据后变成 monotonic 时间戳,之后 30s 没新数据才退出。
    last_data_time = None
    stream_end_flush = StreamEndFlush()

    # 保存尾部 overlap
    tail_overlap = np.array([], dtype=np.float32)

    def save_tail(samples: np.ndarray) -> np.ndarray:
        """返回 samples 末尾 overlap_samples 长度的片段；overlap_samples<=0 时返回空数组"""
        if overlap_samples <= 0:
            return np.array([], dtype=np.float32)
        return samples[-overlap_samples:] if len(samples) > overlap_samples else samples

    def build_prompt() -> tuple[str, int, str]:
        """返回 (prompt_text, tail_source_chars, prompt_hash)"""
        base = args.initial_prompt
        if args.prompt_mode == PROMPT_MODE_NONE:
            p = ""
        elif args.prompt_mode == PROMPT_MODE_TAIL:
            tail = tail_buffer[-args.prompt_tail_chars:] if tail_buffer else ""
            p = base + " " + tail if tail else base
        else:  # fixed
            p = base
        h = hashlib.md5(p.encode()).hexdigest()[:8]
        tail_src = min(len(tail_buffer), args.prompt_tail_chars) if args.prompt_mode == PROMPT_MODE_TAIL else 0
        return p, tail_src, h

    def native_final_snapshot(audio_time_sec: float | None, *,
                              finalize_stream: bool) -> dict | None:
        """取得 native 快照；仅真实 EOF 补右 padding，异常由批处理接管。"""
        nonlocal native_stream_fallback_until_commit
        if (native_stream is None or resolved_backend != "nemotron"
                or native_stream_fallback_until_commit):
            return None
        try:
            result = finalize_native_result(
                native_stream, audio_time_sec=audio_time_sec,
                configured_language=args.language,
                finalize_stream=finalize_stream)
            if result is None:
                # 缺少完整文本或可靠切点同样属于不可复用流状态。本句改走
                # 质量档 `[56,13]`，同步阶段按 inactive reset，不能继续按
                # 用户选择的 3/6 档批处理并把不可用流当作健康状态。
                native_stream_fallback_until_commit = True
                logger.log_status("nemotron_stream_final_unusable_fallback")
            return result
        except Exception as exc:  # noqa: BLE001 - final 必须无损回退
            native_stream_fallback_until_commit = True
            try:
                native_stream.reset(args.language,
                                    args.nemotron_right_context)
            except Exception as reset_exc:  # noqa: BLE001
                logger.log_status("nemotron_stream_reset_fallback",
                                  error=str(reset_exc))
            logger.log_status("nemotron_stream_final_fallback",
                              error=str(exc))
            return None

    def submit_chunk(samples: np.ndarray, seg_start: int, seg_end: int,
                     submit_reason: str, speech_sec: float, trailing_silence_sec: float,
                     overlap_applied: bool = False,
                     overlap_sample_count: int = 0,
                     precomputed_result: dict | None = None,
                     force_quality_context: bool = False,
                     source_key: str = "", revision: int = 0,
                     speech_start_ns: int = 0, speech_end_ns: int = 0,
                     capture_end_mono_ns: int = 0) -> bool:
        """提交一段 final；返回 False 时调用方必须恢复原始未提交 PCM。"""
        nonlocal confirmed_text, tail_overlap, tail_buffer, prev_ended_with_punct
        nonlocal resolved_model, resolved_backend, zh_streak, ja_streak, en_streak
        nonlocal pending_switch, source_revision

        chunk_sec = len(samples) / SAMPLE_RATE
        audio_start_sec = seg_start * SEG_DURATION_SEC
        audio_end_sec = (seg_end + 1) * SEG_DURATION_SEC
        actual_overlap_samples = (
            min(len(samples), max(0, int(overlap_sample_count)))
            if overlap_applied else 0
        )
        actual_overlap_sec = actual_overlap_samples / SAMPLE_RATE
        if actual_overlap_samples:
            audio_start_sec = max(0.0, audio_start_sec - actual_overlap_sec)
        buffered_sec = chunk_sec - actual_overlap_sec
        if capture_end_mono_ns and (not speech_start_ns or not speech_end_ns):
            speech_start_ns, speech_end_ns = (
                find_submission_speech_bounds_mono_ns(
                    samples, capture_end_mono_ns, args.silence_threshold,
                    overlap_sample_count=actual_overlap_samples)
            )

        # 短 chunk 直接丢弃（用 buffered_sec 排除 overlap 虚增）
        if buffered_sec < MIN_SPEECH_SEC:
            metrics.record_reject()
            logger.log_reject(seg_start, seg_end, audio_start_sec, audio_end_sec,
                              chunk_sec, time.monotonic(), prompt_mode=args.prompt_mode,
                              prompt_chars=0, reject_reason=REJECT_SHORT_CHUNK,
                              submit_reason=submit_reason,
                              buffered_sec=buffered_sec,
                              speech_sec=speech_sec,
                              trailing_silence_sec=trailing_silence_sec)
            tail_overlap = save_tail(samples)
            return True

        # 阶段 1: prompt 组装(音频数组直传模型,不再写 WAV)
        try:
            t_asm_start = time.monotonic()
            prompt, tail_src, prompt_hash = build_prompt()
            t_asm_end = time.monotonic()
            asm_ms = (t_asm_end - t_asm_start) * 1000

            # 阶段 2: ASR 推理。precomputed_result 有两种可信来源：批处理
            # 探针对同一份 PCM 的快照，或持久 native 流的稳定快照。
            # 两者都透传真实推理时间并直接复用，避免 final 再跑整段 ASR。
            cached_timing = precomputed_asr_timing(precomputed_result)
            if precomputed_result is not None and cached_timing is not None:
                result = precomputed_result
                (asr_request_start_mono_ns,
                 asr_complete_mono_ns,
                 transcribe_ms) = cached_timing
            else:
                t_infer_start = time.monotonic()
                asr_request_start_mono_ns = time.monotonic_ns()
                right_context = (13 if (force_quality_context
                                         or native_stream_fallback_until_commit)
                                 else args.nemotron_right_context)
                try:
                    result = do_transcribe(
                        samples, language=args.language,
                        model=resolved_model, backend=resolved_backend,
                        nemotron_right_context=right_context)
                except Exception as primary_exc:
                    if resolved_backend != "nemotron" or right_context == 13:
                        raise
                    logger.log_status(
                        "nemotron_final_batch_retry_56_13",
                        error=str(primary_exc))
                    result = do_transcribe(
                        samples, language=args.language,
                        model=resolved_model, backend="nemotron",
                        nemotron_right_context=13)
                t_infer_end = time.monotonic()
                asr_complete_mono_ns = time.monotonic_ns()
                transcribe_ms = (t_infer_end - t_infer_start) * 1000
            metrics.record(transcribe_ms, buffered_sec)

            # partial/final 必须先走同一规范化。旧逻辑先用原文递增 revision，
            # final 再把 chatgpt→ChatGPT 等后处理，导致同一源文 revision 改变、
            # 严格复用永远 miss。
            t_post_start = time.monotonic()
            reject_reason, text = filter_result(result)
            if text is not None:
                text = strip_boundary_subword_overlap(confirmed_text, text)
                source_key, revision = source_revision.update(text)

            # 立刻写规范化后的 partial，翻译层可提前消费；完整过滤未通过的
            # 结果不下发，避免 final 随后拒绝但草稿已造成 UI 闪烁。
            if text and logger._f:
                if (is_valid_partial_text(text)
                        and partial_event_gate.should_emit(
                            source_key, text, now_ns=time.monotonic_ns())):
                    logger.log_partial(seg_start, seg_end,
                                       audio_start_sec, audio_end_sec,
                                       text, source_key=source_key,
                                       revision=revision,
                                       speech_start_mono_ns=speech_start_ns,
                                       speech_end_mono_ns=speech_end_ns,
                                       asr_request_start_mono_ns=(
                                           asr_request_start_mono_ns),
                                       asr_complete_mono_ns=asr_complete_mono_ns,
                                       capture_end_mono_ns=capture_end_mono_ns)

            if args.dump_raw and result["text"]:
                print(f"[raw] {result['text']}", file=sys.stderr, flush=True)

            if text is None:
                metrics.record_reject()
                if args.dump_filtered and reject_reason:
                    print(f"[reject:{reject_reason}] {result['text'][:60]}", file=sys.stderr, flush=True)
                logger.log_reject(seg_start, seg_end, audio_start_sec, audio_end_sec,
                                  chunk_sec, t_asm_start, args.prompt_mode, len(prompt),
                                  reject_reason or REJECT_SHORT_CHUNK,
                                  submit_reason=submit_reason,
                                  buffered_sec=buffered_sec,
                                  speech_sec=speech_sec,
                                  trailing_silence_sec=trailing_silence_sec,
                                  prompt_hash=prompt_hash,
                                  tail_source_chars=tail_src,
                                  language=result["language"],
                                  avg_logprob=result["avg_logprob"],
                                  avg_compression=result["avg_compression"],
                                  no_speech_prob=result["no_speech_prob"],
                                  text=result["text"])
                tail_overlap = save_tail(samples)
                return True

            if args.dump_filtered:
                print(f"[filtered] {text}", file=sys.stderr, flush=True)

            delta = incremental_text(confirmed_text, text)
            if delta:
                if text_out:
                    text_out.write(delta + "\n")
                    text_out.flush()

            # 更新 tail_buffer（只来自已确认 final）
            confirmed_text = text
            tail_buffer = text[-args.prompt_tail_chars:]
            # 标点感知：下个 chunk 更快提交
            prev_ended_with_punct = text.rstrip()[-1:] in "。！？.!?"

            # 稳定原文先落事件，再做任何模型切换。加载/warmup 即使需要几十
            # 秒或失败，也不能拖住一条已经识别完成的 final。
            t_post_end = time.monotonic()
            postproc_ms = (t_post_end - t_post_start) * 1000
            logger.log_final(seg_start, seg_end, audio_start_sec, audio_end_sec,
                             chunk_sec, t_asm_start, t_post_end, transcribe_ms,
                             args.prompt_mode, len(prompt), text,
                             submit_reason=submit_reason,
                             buffered_sec=buffered_sec,
                             speech_sec=speech_sec,
                             trailing_silence_sec=trailing_silence_sec,
                             prompt_hash=prompt_hash,
                             tail_source_chars=tail_src,
                             asm_ms=asm_ms,
                             postproc_ms=postproc_ms,
                             language=result["language"],
                             avg_logprob=result["avg_logprob"],
                             avg_compression=result["avg_compression"],
                             no_speech_prob=result["no_speech_prob"],
                             source_key=source_key,
                             revision=revision,
                             speech_start_mono_ns=speech_start_ns,
                             speech_end_mono_ns=speech_end_ns,
                             asr_request_start_mono_ns=asr_request_start_mono_ns,
                             asr_complete_mono_ns=asr_complete_mono_ns,
                             capture_end_mono_ns=capture_end_mono_ns)

            # 语言感知模型切换（观察区 + 连续计数 + 异步加载）
            qwen_lang = _detect_qwen_lang(text)
            ratio = _cjk_ratio(text)
            zh_streak = zh_streak + 1 if qwen_lang == "zh" and ratio >= CJK_OBSERVE_LOW else 0
            ja_streak = ja_streak + 1 if qwen_lang == "ja" else 0
            en_streak = en_streak + 1 if qwen_lang is None else 0
            cjk_streak = max(zh_streak, ja_streak)  # 中日文共用切换阈值

            if pending_switch:
                # ready 只表示线程结束；success=False 时保留当前模型/native
                # 状态，绝不能卸掉可用后端再切进加载失败的路径。
                if switch_state.ready.is_set():
                    if switch_state.success:
                        if pending_switch == "to_qwen3":
                            deactivate_native_stream(
                                release_model=not args.dual_model)
                            if not args.dual_model:
                                _unload_nemotron()
                                import gc; gc.collect()
                            resolved_model = qwen3_path
                            resolved_backend = "qwen3"
                            logger.log_status(
                                "已切换到 Qwen3 ASR", status_color="green")
                        elif pending_switch == "to_nemotron":
                            if not args.dual_model:
                                _unload_qwen3()
                                import gc; gc.collect()
                            resolved_model = nemotron_model
                            resolved_backend = "nemotron"
                            activate_native_stream(resolved_model)
                            logger.log_status(
                                "已切换回 Nemotron ASR", status_color="green")
                        print("[lang-switch] 模型加载完成，已切换",
                              file=sys.stderr, flush=True)
                    else:
                        logger.log_status(
                            "模型切换加载失败，保持当前 ASR",
                            status_color="red", error=switch_state.error)
                        print(f"[lang-switch] 加载失败，保持 {resolved_backend}: "
                              f"{switch_state.error}", file=sys.stderr,
                              flush=True)
                    pending_switch = None
                    switch_state.reset()
                    zh_streak = 0
                    ja_streak = 0
                    en_streak = 0

            elif (resolved_backend != "qwen3"
                  and (ja_streak >= 2 or zh_streak >= 2)):
                # 中/日文质量阈值保持不变；双模型可秒切，单模型一律在 final
                # 已落盘后异步 load+warmup，避免高置信中文同步卡 30–60s。
                detected = "日文" if ja_streak >= 2 else "中文"
                confidence = (f" ({ratio:.0%})" if detected == "中文" else "")
                if args.dual_model:
                    print(f"[lang-switch] 检测到{detected}{confidence}，切换到 Qwen3",
                          file=sys.stderr, flush=True)
                    deactivate_native_stream(release_model=False)
                    resolved_model = qwen3_path
                    resolved_backend = "qwen3"
                    logger.log_status(
                        f"检测到{detected}，已切换到 Qwen3", status_color="green")
                    zh_streak = 0
                    ja_streak = 0
                    en_streak = 0
                else:
                    print(f"[lang-switch] 检测到{detected}{confidence}，异步加载 Qwen3",
                          file=sys.stderr, flush=True)
                    logger.log_status(
                        f"检测到{detected}，正在加载 Qwen3...",
                        status_color="orange")
                    pending_switch = "to_qwen3"
                    switch_state.reset()
                    threading.Thread(
                        target=_async_load_model,
                        args=("qwen3", qwen3_path, switch_state),
                        daemon=True).start()

            elif resolved_backend == "qwen3" and en_streak >= 3:
                # 连续 3 句非中日文才切回；没有已下载 Nemotron 时保持 Qwen3，
                # 不允许把空路径交给加载线程。
                if not nemotron_model:
                    logger.log_status(
                        "未找到可用 Nemotron，保持 Qwen3 ASR",
                        status_color="orange")
                    en_streak = 0
                elif args.dual_model and _nemotron_model is not None:
                    print("[lang-switch] 检测到英文，切换回 Nemotron",
                          file=sys.stderr, flush=True)
                    resolved_model = nemotron_model
                    resolved_backend = "nemotron"
                    activate_native_stream(resolved_model)
                    logger.log_status(
                        "已切换回 Nemotron ASR", status_color="green")
                    en_streak = 0
                else:
                    # 从 Qwen3 作为主模型启动时，即使传了 --dual-model，
                    # Nemotron 也未必已经预载；未加载就仍走异步，不能在
                    # submit 返回前同步 load 或先改 resolved_backend。
                    print("[lang-switch] 检测到英文，异步加载 Nemotron",
                          file=sys.stderr, flush=True)
                    logger.log_status(
                        "检测到英文，正在加载 Nemotron...", status_color="orange")
                    pending_switch = "to_nemotron"
                    switch_state.reset()
                    threading.Thread(
                        target=_async_load_model,
                        args=("nemotron", nemotron_model, switch_state),
                        daemon=True).start()

        except Exception as e:
            print(f"[error] submit_chunk 异常，保留当前 PCM 待重试: {e}",
                  file=sys.stderr, flush=True)
            logger.write({
                "event_type": "error",
                "seg_start": seg_start,
                "seg_end": seg_end,
                "audio_start_sec": round(audio_start_sec, 3),
                "audio_end_sec": round(audio_end_sec, 3),
                "chunk_sec": round(chunk_sec, 3),
                "submit_reason": submit_reason,
                "speech_sec": round(speech_sec, 3),
                "trailing_silence_sec": round(trailing_silence_sec, 3),
                "prompt_mode": args.prompt_mode,
                "error": str(e),
            })
            return False

        # 只有成功提交/按质量规则拒绝才推进 overlap；异常 PCM 由调用方
        # 原样恢复，不能把它误当成已确认尾部。
        tail_overlap = save_tail(samples)
        return True

    def discard_collected_silence() -> None:
        """明确 EOF/纯静音时清空 chunk；不触碰任何已检测到的语音。"""
        nonlocal collected, collected_count, has_speech, silence_streak
        nonlocal speech_accumulated, sm_text, sm_segments, sm_result, sm_audio
        nonlocal sm_capture_end_mono_ns, stable_punct_candidate, sm_punct_stable
        nonlocal native_stream_fallback_until_commit, tail_overlap
        if collected:
            tail_overlap = save_tail(collected[-1])
        collected = []
        collected_count = 0
        has_speech = False
        silence_streak = 0.0
        speech_accumulated = 0.0
        sm_text = None
        sm_segments = None
        sm_result = None
        sm_audio = np.array([], dtype=np.float32)
        sm_capture_end_mono_ns = 0
        stable_punct_candidate = None
        sm_punct_stable = False
        if native_stream is not None:
            try:
                native_stream.reset(args.language,
                                    args.nemotron_right_context)
                native_stream_fallback_until_commit = False
            except Exception as exc:  # noqa: BLE001
                native_stream_fallback_until_commit = True
                logger.log_status("nemotron_stream_reset_fallback",
                                  error=str(exc))
        advance_source_revision()

    def restore_unsubmitted_audio(samples: np.ndarray) -> None:
        """ASR final 异常后恢复整段 PCM，并强制下一次走 [56,13] 批处理。"""
        nonlocal collected, collected_count, has_speech, silence_streak
        nonlocal speech_accumulated, sm_text, sm_result, sm_audio
        nonlocal sm_capture_end_mono_ns, stable_punct_candidate, sm_punct_stable
        nonlocal soft_max_last_asr, native_stream_fallback_until_commit
        restored = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        collected = [restored] if restored.size else []
        collected_count = len(restored)
        has_speech, silence_streak, speech_accumulated = carry_remainder_vad(
            restored, args.silence_threshold,
            has_speech=False, silence_streak=0.0,
            speech_accumulated=0.0)
        sm_text = None
        sm_result = None
        sm_audio = np.array([], dtype=np.float32)
        sm_capture_end_mono_ns = 0
        stable_punct_candidate = None
        sm_punct_stable = False
        soft_max_last_asr = 0.0
        if native_stream is not None:
            native_stream_fallback_until_commit = True
            try:
                native_stream.reset(args.language,
                                    args.nemotron_right_context)
            except Exception as exc:  # noqa: BLE001
                # collected 已先恢复；reset 再失败也只能禁用 native，不能让
                # 异常冒出导致进程重启并丢掉这段仅存在内存中的 PCM。
                logger.log_status("nemotron_stream_reset_fallback",
                                  error=str(exc))
        if stream_end_flush.pending:
            stream_end_flush.mark_failure(now=time.monotonic())

    try:
        while True:
            # 必须位于任何 queue.get/get_nowait 之前；信号处理器只置位，
            # 所有 stop/join/queue snapshot 都在这里的无锁主循环上下文执行。
            if deferred_audio_swap.consume():
                _apply_deferred_audio_swap()

            # 音频源的崩溃恢复由 audio.py 的 SystemAudioSource 内部
            # supervisor 线程处理（5s stall 自动重启 audiotee）。whicc.py
            # 只消费数据:live 模式读 source.queue,segdir 模式轮询段文件。
            # SIGINT/SIGTERM 时 audio_source.stop() 会优雅退出。
            #
            # 保留长 stall 总超时,防止 audio 源自己挂了重启又不成功时
            # whicc.py 无限空转。12s = supervisor 5s stall 重启一次
            # audiotee + 观察窗:救不回(macOS process tap 对 kill 后
            # respawn 的 audiotee 常静默拒绝授权,只给零数据)就快速换
            # 进程 — 新 whicc.py 进程 spawn 的 audiotee 能重新拿到授权
            # (实测),比等 30s 少断流 18s。
            now = time.monotonic()
            # 收到首段数据后才开始计时;启动期静音不退出。
            # application 模式等待目标应用启动/出声是正常态(可能持续数
            # 分钟),此期间豁免看门狗 —— 否则 12s 无数据就 exit(4) 重启
            # 整个后端,违反「保持等待,不回退」的语义。
            waiting_for_app = bool(
                audio_source is not None
                and getattr(audio_source, "waiting_for_app", False)
            )
            if waiting_for_app:
                last_data_time = None  # 等待结束后重新计时
            elif (not stream_end_flush.pending
                  and last_data_time is not None
                  and now - last_data_time > 12.0):
                print(f"\n[error] 音频源 12s 无数据(audiotee 重启也没救回),"
                      "退出换新进程重新授权 process tap。",
                      file=sys.stderr, flush=True)
                # exit 4 = 音频自愈重启(恢复性,不是故障)。BackendLauncher
                # 按 code 给"正在自动恢复"文案而不是"异常退出"。
                # SystemExit 会经 finally 完整清理(metrics/logger/audio)。
                sys.exit(4)

            # 切点后的剩余 PCM 必须先恢复，随后才能判断旧 stream 是否仍
            # 有待 final 的内容。否则 source swap 后会先读新 source，造成
            # 新旧音频混入同一句。
            if len(soft_max_remainder) > 0:
                collected.insert(0, soft_max_remainder)
                collected_count += len(soft_max_remainder)
                has_speech, silence_streak, speech_accumulated = carry_remainder_vad(
                    soft_max_remainder, args.silence_threshold,
                    has_speech=has_speech,
                    silence_streak=silence_streak,
                    speech_accumulated=speech_accumulated,
                )
                soft_max_remainder = np.array([], dtype=np.float32)

            # 读取新音频:
            #  - live(system/mic): 阻塞至多 POLL_INTERVAL 等内存 queue 的
            #    chunks,到达即处理(采集回调粒度 ~0.1s,无磁盘往返)
            #  - segdir: 0.15s 轮询 SEG_DIR 段文件(离线评估协议)
            pending_old_audio = bool(collected)
            if stream_end_flush.blocks_source_read(
                    has_pending_audio=pending_old_audio):
                # final 失败后的退避期也不能读新 source；queue 最多缓存 20s。
                # 本轮只重放 EOF 标记，由下方按“尚缺静音”补齐后重试。
                retry_wait = stream_end_flush.retry_wait(now=time.monotonic())
                if retry_wait > 0:
                    time.sleep(min(POLL_INTERVAL, retry_wait))
                    # 恢复的 PCM 已包含上一轮合成静音；退避未到时若继续走
                    # normal submit，silence_streak 会立刻再次触发 final，实际
                    # 变成每 150ms 重试。这里必须直接回到外层轮询。
                    continue
                new_arrays = [STREAM_END_PACKET]
                stream_ended = False
            else:
                if stream_end_flush.pending:
                    # 旧 stream 的所有 PCM 已 final/明确丢弃。segdir 的 done
                    # marker 表示整个受控回放结束；live 则清掉 gate 后读取新
                    # source，且新 source 的 sequence 从 0 重新开始。
                    if use_segdir:
                        break
                    stream_end_flush.clear()
                    last_audio_sequence = None

                if use_segdir:
                    time.sleep(POLL_INTERVAL)
                    raw_segs, next_seg, stream_ended = read_segments(next_seg)
                    new_arrays = []
                    for segment in raw_segs:
                        try:
                            new_arrays.append(CapturedAudio(
                                samples=np.frombuffer(
                                    segment.data, dtype=np.float32).copy(),
                                capture_end_mono_ns=(
                                    segment.capture_end_mono_ns
                                    or time.monotonic_ns()),
                                sequence=segment.sequence,
                                dropped_before=False,
                            ))
                        except (ValueError, BufferError) as e:
                            print(f"[warn] 损坏段文件已跳过: {e}",
                                  file=sys.stderr, flush=True)
                else:
                    selected_queue, reading_retired_queue = (
                        audio_handoff.queue_for_read(audio_source.queue)
                    )
                    new_arrays, stream_ended = drain_audio_queue(
                        selected_queue, POLL_INTERVAL)
                    audio_handoff.finish_read(
                        selected_queue, ended=stream_ended)

            real_audio_arrived = any(
                packet is not STREAM_END_PACKET for packet in new_arrays)
            stream_end_flush.observe(
                ended=stream_ended, real_audio=real_audio_arrived)
            if real_audio_arrived:
                last_data_time = time.monotonic()  # 真音频到达才重置看门狗
            # EOF 标记必须排在本轮已读真实 PCM 的末尾；这样最后一个无尾静音
            # 的 chunk 也会在同一轮被补静音 final，而不是等下一 source 的音频。
            if stream_ended:
                new_arrays.append(STREAM_END_PACKET)
            if not new_arrays:
                continue

            batch_started = False
            for packet in new_arrays:
                synthetic_eof = packet is STREAM_END_PACKET
                if synthetic_eof:
                    if collected and not has_speech:
                        # EOF 后只剩纯静音，无需等到 max_chunk；没有语音可丢。
                        discard_collected_silence()
                        stream_end_flush.mark_success()
                        continue
                    samples = stream_end_flush.silence_packet(
                        has_pending_speech=bool(collected and has_speech),
                        silence_submit_sec=args.silence_submit_sec,
                        current_silence_sec=silence_streak,
                        now=time.monotonic(),
                    )
                    if samples is None:
                        continue
                    logger.log_status(
                        "audio_stream_end_flush",
                        retry=stream_end_flush.failures,
                        synthetic_silence_sec=len(samples) / SAMPLE_RATE)
                    # 合成静音的 capture 时钟沿用最后一块真实 PCM 的终点，
                    # 再加合成时长。用处理当下的 now 会把 ASR 推理耗时误算成
                    # 音频时间，导致句末→上屏尾延迟失真。
                    if latest_capture_end_mono_ns:
                        latest_capture_end_mono_ns += int(
                            len(samples) / SAMPLE_RATE * 1_000_000_000)
                    else:
                        latest_capture_end_mono_ns = time.monotonic_ns()
                elif isinstance(packet, CapturedAudio):
                    samples = packet.samples
                    latest_capture_end_mono_ns = packet.capture_end_mono_ns
                    if (packet.dropped_before or
                            (last_audio_sequence is not None and
                             packet.sequence != last_audio_sequence + 1)):
                        logger.log_status("audio_sequence_gap",
                                          sequence=packet.sequence,
                                          dropped_before=packet.dropped_before)
                        if native_stream is not None:
                            native_stream_fallback_until_commit = True
                            try:
                                native_stream.reset(
                                    args.language,
                                    args.nemotron_right_context)
                            except Exception as exc:  # noqa: BLE001
                                logger.log_status(
                                    "nemotron_stream_reset_fallback",
                                    error=str(exc))
                    last_audio_sequence = packet.sequence
                else:
                    samples = packet
                    latest_capture_end_mono_ns = time.monotonic_ns()
                # 虚拟段号 = 已消费整秒数(1 段 ≡ 1 秒),与旧文件段号语义一致
                # — translate_stream._source_key 和日志对齐依赖 seg_start/
                # seg_end。segdir 模式每段恰 1s,虚拟号等于文件段号。
                # 新 chunk 起始段号（仅在 collected 为空时更新，remainder 不影响）
                if not collected and not batch_started:
                    chunk_first_seg = samples_ingested // SAMPLE_RATE
                batch_started = True
                samples_ingested += len(samples)

                collected.append(samples)
                collected_count += len(samples)

                native_active = (native_stream is not None
                                 and resolved_backend == "nemotron"
                                 and not native_stream_fallback_until_commit)
                if native_active:
                    try:
                        native_asr_start_ns = time.monotonic_ns()
                        previous_decode_generation = native_stream.decode_generation
                        streamed_text = native_stream.feed(samples).strip()
                        native_asr_complete_ns = time.monotonic_ns()
                        current_decode_generation = native_stream.decode_generation
                        if (streamed_text and streamed_text != sm_text
                                and is_valid_partial_text(streamed_text)):
                            streamed = native_stream.result_dict()
                            streamed.update({
                                "language": "zh" if _cjk_ratio(streamed_text) > .3 else "en",
                                "avg_logprob": -0.3,
                                "avg_compression": 0.0,
                                "no_speech_prob": 0.0,
                                "_asr_request_start_mono_ns": native_asr_start_ns,
                                "_asr_complete_mono_ns": native_asr_complete_ns,
                            })
                            sm_text = streamed_text
                            stable_punct_candidate, sm_punct_stable = (
                                update_native_punct_stability(
                                    stable_punct_candidate, sm_punct_stable,
                                    sm_text,
                                    previous_generation=previous_decode_generation,
                                    current_generation=current_decode_generation)
                            )
                            sm_segments = streamed["segments"]
                            sm_result = streamed
                            sm_audio = np.concatenate(collected).copy()
                            sm_capture_end_mono_ns = latest_capture_end_mono_ns
                            sm_language = streamed["language"]
                            draft_text = strip_boundary_subword_overlap(
                                confirmed_text, canonical_asr_text(sm_text))
                            draft_changed = draft_text != source_revision.text
                            key, revision = source_revision.update(draft_text)
                            draft_start_ns, draft_end_ns = find_speech_bounds_mono_ns(
                                sm_audio, sm_capture_end_mono_ns,
                                args.silence_threshold)
                            visible = partial_event_gate.should_emit(
                                key, draft_text,
                                now_ns=native_asr_complete_ns)
                            if visible or draft_changed:
                                logger.log_partial(
                                    chunk_first_seg,
                                    samples_ingested // SAMPLE_RATE,
                                    chunk_first_seg * SEG_DURATION_SEC,
                                    samples_ingested / SAMPLE_RATE,
                                    draft_text, source_key=key,
                                    revision=revision,
                                    event_type=("partial" if visible
                                                else "translation_input"),
                                    speech_start_mono_ns=draft_start_ns,
                                    speech_end_mono_ns=draft_end_ns,
                                    is_probe=False,
                                    asr_request_start_mono_ns=native_asr_start_ns,
                                    asr_complete_mono_ns=native_asr_complete_ns,
                                    capture_end_mono_ns=latest_capture_end_mono_ns,
                                )
                        elif streamed_text and streamed_text == sm_text:
                            # 只有 encoder/RNNT 真正消费了新 chunk，才算
                            # 第二次观察；普通 100ms feed 可能没有新推理。
                            stable_punct_candidate, sm_punct_stable = (
                                update_native_punct_stability(
                                    stable_punct_candidate, sm_punct_stable,
                                    sm_text,
                                    previous_generation=previous_decode_generation,
                                    current_generation=current_decode_generation)
                            )
                    except Exception as exc:
                        native_stream_fallback_until_commit = True
                        try:
                            native_stream.reset(args.language,
                                                args.nemotron_right_context)
                        except Exception as reset_exc:  # noqa: BLE001
                            logger.log_status(
                                "nemotron_stream_reset_fallback",
                                error=str(reset_exc))
                        logger.log_status("nemotron_stream_fallback", error=str(exc))

                # 采集块可能是 100/128/200ms；逐 20ms 子帧更新，避免短词
                # 位于块前部而末帧已静音时整块被误判为静音。
                has_speech, silence_streak, speech_accumulated = update_vad_state(
                    samples, args.silence_threshold,
                    has_speech=has_speech,
                    silence_streak=silence_streak,
                    speech_accumulated=speech_accumulated,
                )

            chunk_sec = collected_count / SAMPLE_RATE
            seg_end = samples_ingested // SAMPLE_RATE  # 当前虚拟段号(累计秒)

            # ---- 探针 ASR: 累积够 MIN_CHUNK_SEC (2s) 就跑,持续刷新 sm_text ----
            # ASCII `.` 必须跨两次观察稳定，避免把 Dr. / 21.4 当句末；
            # !? 及中日韩强句末符号首次可靠观察即可确认。
            # 软最大值切割在下面单独判断,共用 sm_text 缓存不重复跑推理。
            native_active = (native_stream is not None
                             and resolved_backend == "nemotron"
                             and not native_stream_fallback_until_commit)
            if not native_active and chunk_sec >= MIN_CHUNK_SEC and has_speech:
                need_asr = sm_text is None or (time.monotonic() - soft_max_last_asr) >= SOFT_MAX_ASR_COOLDOWN
                if need_asr:
                    all_sm = np.concatenate(collected)
                    sm_audio = all_sm.copy()
                    sm_capture_end_mono_ns = latest_capture_end_mono_ns
                    try:
                        # 数组直传 — 探针每 0.6s 跑一次,之前每次都
                        # save_wav 落盘再让模型读回,现在零磁盘往返。
                        probe_start_ns = time.monotonic_ns()
                        result_sm = do_transcribe(all_sm,
                                                  language=args.language,
                                                  model=resolved_model,
                                                  backend=resolved_backend,
                                                  nemotron_right_context=(
                                                      13 if native_stream_fallback_until_commit
                                                      else args.nemotron_right_context
                                                  ))
                        probe_done_ns = time.monotonic_ns()
                        # 与识别结果作为一个快照保存；final 复用时必须透传
                        # 探针真实耗时，不能把“缓存读取≈0ms”伪装成 ASR 延迟。
                        result_sm = dict(result_sm)
                        result_sm["_asr_request_start_mono_ns"] = probe_start_ns
                        result_sm["_asr_complete_mono_ns"] = probe_done_ns
                        sm_text = result_sm.get("text", "").strip()
                        sm_segments = result_sm.get("segments")
                        sm_result = result_sm  # 完整 dict,submit 音频一致时复用
                        # ASR 返回的 language (e.g. "zh" / "en") 用于分语言阈值
                        sm_language = result_sm.get("language", "en") or "en"
                        # Nemotron 返回 "zh" / "en" / "zh-CN" 等,Qwen3 返回 ["zh"] 列表
                        if isinstance(sm_language, list):
                            sm_language = sm_language[0] if sm_language else "en"
                        # 规范化: "zh-CN" / "zh-Hans" → "zh"
                        sm_language = sm_language.split("-")[0].lower()
                        soft_max_last_asr = time.monotonic()
                        if not sm_text:
                            sm_text = None
                            sm_result = None
                            sm_audio = np.array([], dtype=np.float32)
                            sm_capture_end_mono_ns = 0
                            stable_punct_candidate = None
                            sm_punct_stable = False
                        else:
                            stable_punct_candidate, sm_punct_stable = (
                                update_punct_end_stability(
                                    stable_punct_candidate, sm_text)
                            )
                        draft_text = strip_boundary_subword_overlap(
                            confirmed_text,
                            canonical_asr_text(sm_text or ""))
                        if (draft_text and latency_cfg["probe_partial_enabled"]
                                and is_valid_partial_text(draft_text)):
                            key, revision = source_revision.update(draft_text)
                            bounds = find_speech_bounds_sec(
                                all_sm, args.silence_threshold)
                            start_ns = end_ns = 0
                            if latest_capture_end_mono_ns and bounds:
                                audio_start_ns = latest_capture_end_mono_ns - int(
                                    len(all_sm) / SAMPLE_RATE * 1_000_000_000)
                                start_ns = audio_start_ns + int(bounds[0] * 1_000_000_000)
                                end_ns = audio_start_ns + int(bounds[1] * 1_000_000_000)
                            if partial_event_gate.should_emit(
                                    key, draft_text, now_ns=probe_done_ns):
                                logger.log_partial(
                                    chunk_first_seg, seg_end,
                                    chunk_first_seg * SEG_DURATION_SEC,
                                    (seg_end + 1) * SEG_DURATION_SEC,
                                    draft_text, source_key=key,
                                    revision=revision,
                                    speech_start_mono_ns=start_ns,
                                    speech_end_mono_ns=end_ns,
                                    is_probe=True,
                                    asr_request_start_mono_ns=probe_start_ns,
                                    asr_complete_mono_ns=probe_done_ns,
                                    capture_end_mono_ns=latest_capture_end_mono_ns,
                                )
                    except Exception as e:
                        print(f"[probe-asr] ASR 异常: {e}", file=sys.stderr, flush=True)
                        sm_text = None
                        sm_result = None
                        sm_audio = np.array([], dtype=np.float32)
                        sm_capture_end_mono_ns = 0
                        stable_punct_candidate = None
                        sm_punct_stable = False

            # ---- 标点感知断句: ASR 看到完整句立刻在标点位置切 ----
            # 与 soft_max 区别: 不要求 chunk_sec >= SOFT_MAX_SEC。
            # 只要稳定探针文本以强句末标点结尾 + 累积够阈值，就在标点的
            # 音频位置切，而不是切到 chunk 末尾 — 否则 submit_chunk
            # native 健康时不额外叠加 overlap，避免多听尾音产生
            # "Human centered... right? People are" 之类 mid-sentence 尾巴；
            # 只有流异常后的 `[56,13]` 质量恢复批次保留 0.3s
            # 边界上下文。用 find_audio_split_sec 的
            # segments 时间戳精确定位标点 → submit 前半,剩余音频 (标点后) 作为新
            # chunk 起始 (跟 soft_max 一样)。
            #
            # 分语言阈值 (PUNCT_END_MIN_CHUNK_SEC_EN=3 / _ZH=5),中文需要更多上下文。
            # 字符数校验 (MIN_CHARS_BEFORE_PUNCT_*): 中文 12 字以上,英文 8 字以上。
            _cur_lang = sm_language
            _min_chunk_for_punct = PUNCT_END_MIN_CHUNK_SEC_ZH if _cur_lang.startswith("zh") else PUNCT_END_MIN_CHUNK_SEC_EN
            _min_chars_before_punct = MIN_CHARS_BEFORE_PUNCT_ZH if _cur_lang.startswith("zh") else MIN_CHARS_BEFORE_PUNCT_EN
            _stable_punct_text = stable_punct_candidate or ""
            _chars_before_punct = len(_stable_punct_text) - 1
            if (chunk_sec >= _min_chunk_for_punct
                    and has_speech and sm_text
                    and sm_punct_stable
                    and _stable_punct_text[-1:] in STRONG_END_PUNCT
                    and punctuation_pause_ready(
                        _stable_punct_text, silence_streak)
                    and _chars_before_punct >= _min_chars_before_punct):
                split_sec = find_audio_split_sec(
                    _stable_punct_text, chunk_sec, sm_segments,
                    min_char_pos=0)  # 任意位置都允许,标点已经是强句末
                # 校验: 切点不能太短 (< 0.5s, 字幕太碎) 也不能太靠近结尾
                # (>= chunk_sec - 0.3s, 剩余音频几乎为空,失去切的意义)
                if split_sec > 0 and 0.5 <= split_sec <= chunk_sec - 0.3:
                    all_sm = np.concatenate(collected)
                    split_pos = int(split_sec * SAMPLE_RATE)
                    first_part = all_sm[:split_pos]
                    remainder_audio = all_sm[split_pos:]
                    print(f"[punct-split] 标点位置切 @{split_sec:.1f}s/{chunk_sec:.1f}s: "
                          f"{sm_text[:50]}... | 剩余 {len(remainder_audio)/SAMPLE_RATE:.1f}s",
                          file=sys.stderr, flush=True)
                    native_final = (native_final_snapshot(
                                        split_sec, finalize_stream=False)
                                    if native_active else None)
                    # snapshot 异常会置 fallback；同步阶段必须按失效流处理，
                    # 否则 reset 后的空流无法重放切点后的原始 PCM。
                    native_active_for_sync = (
                        native_active
                        and not native_stream_fallback_until_commit)
                    first_part_for_submit, split_overlap_count = (
                        prepare_split_submission_audio(
                            first_part, tail_overlap,
                            native_active=native_active_for_sync,
                            native_fallback=(
                                native_stream_fallback_until_commit),
                        )
                    )
                    submitted = submit_chunk(
                        first_part_for_submit, chunk_first_seg, seg_end,
                        submit_reason="punct_split",
                        speech_sec=speech_accumulated,
                        trailing_silence_sec=silence_streak,
                        overlap_applied=split_overlap_count > 0,
                        overlap_sample_count=split_overlap_count,
                        # native final 已按真实 token 终点裁剪；没有可用切点
                        # 或流异常时为 None，submit_chunk 自动批处理保底。
                        precomputed_result=native_final,
                        capture_end_mono_ns=(
                            latest_capture_end_mono_ns
                            - int((chunk_sec - split_sec) * 1_000_000_000)
                        ))
                    if not submitted:
                        restore_unsubmitted_audio(all_sm)
                        continue
                    stream_end_flush.mark_success()
                    # 先保存切点后的原始 PCM，再 best-effort 同步 MLX cache。
                    # commit/reset 失败只关闭 native 草稿，不能吞下一句开头。
                    soft_max_remainder = remainder_audio.copy()
                    # 下一句从切点后的 remainder 起始；同步推进绝对音频段号，
                    # 避免持久流每条 final 的 audio_start_sec 都错误回到 0。
                    chunk_first_seg = max(
                        chunk_first_seg,
                        (samples_ingested - len(soft_max_remainder))
                        // SAMPLE_RATE,
                    )
                    (native_stream_fallback_until_commit,
                     soft_max_remainder,
                     native_sync_error) = sync_native_after_submit(
                        native_stream, commit_sec=split_sec,
                        language=args.language,
                        right_context=args.nemotron_right_context,
                        backend_after_submit=resolved_backend,
                        native_active_before_submit=native_active_for_sync,
                        current_fallback=native_stream_fallback_until_commit,
                        raw_remainder=soft_max_remainder,
                    )
                    if native_sync_error is not None:
                        logger.log_status(
                            "nemotron_stream_commit_fallback",
                            error=str(native_sync_error))
                    advance_source_revision()
                    collected = []
                    collected_count = 0
                    has_speech = False
                    silence_streak = 0
                    speech_accumulated = 0
                    soft_max_last_asr = 0.0
                    sm_text = None
                    sm_result = None
                    sm_audio = np.array([], dtype=np.float32)
                    sm_capture_end_mono_ns = 0
                    stable_punct_candidate = None
                    sm_punct_stable = False
                    should_submit = False
                    should_discard = False
                    continue
                # 否则 split 不理想,fall through 到下面的 soft_max / punct_end 兜底

            native_lookahead_sec = (
                float(native_stream.chunk_frames) * 0.08
                if native_active and native_stream is not None else 0.0)
            guarded_max_split = native_guarded_max_split_sec(
                chunk_sec, args.max_chunk_sec, native_lookahead_sec)

            # ---- 软最大值 / native 硬上限断句 ----
            # 共用上面探针的 sm_text/sm_segments 缓存,不重复跑推理。
            # 找到合适的标点 → 在标点位置切 (前半提交 final, 后半作为新 chunk 继续)。
            # native 到 max_chunk + look-ahead 后即使没有文本/可靠 token
            # 边界也必须切；前 max_chunk 秒改走 [56,13]，尾部 PCM 原样保留。
            if (chunk_sec >= SOFT_MAX_SEC and has_speech
                    and (sm_text or guarded_max_split)):
                split_sec = (find_audio_split_sec(
                    sm_text, chunk_sec, sm_segments,
                    min_char_pos=SOFT_MAX_MIN_CHARS) if sm_text else 0)
                split_label = "整句"
                force_quality_batch = False
                if (split_sec == 0 and sm_text
                        and len(sm_text) >= SOFT_MAX_MIN_CHARS):
                    split_sec = find_audio_split_sec(
                        sm_text, chunk_sec, sm_segments,
                        punct_set=SENTENCE_END_PUNCT | MID_PUNCT,
                        min_char_pos=SOFT_MAX_MIN_CHARS)
                    split_label = "中间标点"
                min_split = 1.5 if split_label == "中间标点" else 0.5
                max_remainder = 0.5 if split_label == "中间标点" else 0.3
                split_invalid = (
                    split_sec == 0 or split_sec <= min_split
                    or split_sec > chunk_sec - max_remainder)
                if split_invalid and guarded_max_split:
                    # max_chunk 本身不变，只额外等当前档位的右上下文。这样
                    # 迟到 token 与 SentencePiece 子词能整体留在正确一侧；
                    # 无可靠整词边界则仍在 max_chunk 切，并强制质量档批处理。
                    split_sec, force_quality_batch = resolve_native_max_split(
                        sm_segments, target_sec=guarded_max_split,
                        max_lookback_sec=1.5)
                    split_label = ("右上下文质量回退" if force_quality_batch
                                   else "右上下文整词")
                    min_split = 0.5
                    max_remainder = 0.3
                    split_invalid = (
                        split_sec <= min_split
                        or split_sec > chunk_sec - max_remainder)
                if split_invalid:
                    if split_sec > 0:
                        print(f"[soft-max] 标点位置不佳 ({split_sec:.1f}s/{chunk_sec:.1f}s)，继续积累",
                              file=sys.stderr, flush=True)
                else:
                    all_sm = np.concatenate(collected)
                    split_pos = int(split_sec * SAMPLE_RATE)
                    first_part = all_sm[:split_pos]
                    remainder_audio = all_sm[split_pos:]
                    is_full = split_label == "整句"
                    reason = ("max_chunk" if split_label.startswith("右上下文")
                              else "soft_max" if is_full
                              else "soft_max_split")
                    print(f"[soft-max] {split_label}断句 @{split_sec:.1f}s: "
                          f"{(sm_text or '')[:50]} | 剩余 "
                          f"{len(remainder_audio)/SAMPLE_RATE:.1f}s",
                          file=sys.stderr, flush=True)
                    native_final = (native_final_snapshot(
                                        split_sec, finalize_stream=False)
                                    if native_active and not force_quality_batch
                                    else None)
                    native_active_for_sync = (
                        native_active
                        and not force_quality_batch
                        and not native_stream_fallback_until_commit)
                    first_part_for_submit, split_overlap_count = (
                        prepare_split_submission_audio(
                            first_part, tail_overlap,
                            native_active=native_active_for_sync,
                            native_fallback=(
                                native_stream_fallback_until_commit
                                or force_quality_batch),
                        )
                    )
                    submitted = submit_chunk(
                        first_part_for_submit, chunk_first_seg, seg_end,
                        submit_reason=reason,
                        speech_sec=speech_accumulated,
                        trailing_silence_sec=silence_streak,
                        # 已按真实 token 时间戳精确切点；健康 native 不叠加
                        # overlap，质量恢复批次则真实 prepend 并记录样本数。
                        overlap_applied=split_overlap_count > 0,
                        overlap_sample_count=split_overlap_count,
                        # 优先复用持久流状态；缺失可靠时间戳时批处理保底。
                        precomputed_result=native_final,
                        force_quality_context=force_quality_batch,
                        capture_end_mono_ns=(
                            latest_capture_end_mono_ns
                            - int((chunk_sec - split_sec) * 1_000_000_000)
                        ))
                    if not submitted:
                        restore_unsubmitted_audio(all_sm)
                        continue
                    stream_end_flush.mark_success()
                    soft_max_remainder = remainder_audio.copy()
                    # 与标点切句保持同一绝对时间轴。MLX cache 可以跨句保留，
                    # 但事件/UI 的字幕区间必须从 remainder 的真实起点继续。
                    chunk_first_seg = max(
                        chunk_first_seg,
                        (samples_ingested - len(soft_max_remainder))
                        // SAMPLE_RATE,
                    )
                    (native_stream_fallback_until_commit,
                     soft_max_remainder,
                     native_sync_error) = sync_native_after_submit(
                        native_stream, commit_sec=split_sec,
                        language=args.language,
                        right_context=args.nemotron_right_context,
                        backend_after_submit=resolved_backend,
                        native_active_before_submit=native_active_for_sync,
                        current_fallback=native_stream_fallback_until_commit,
                        raw_remainder=soft_max_remainder,
                    )
                    if native_sync_error is not None:
                        logger.log_status(
                            "nemotron_stream_commit_fallback",
                            error=str(native_sync_error))
                    advance_source_revision()
                    # 切点后的 PCM 无条件留给下一句；即便这是完整句末，尾部也
                    # 可能已经含下一句开头，丢弃会造成无法恢复的吞词。
                    collected = []
                    collected_count = 0
                    has_speech = False
                    silence_streak = 0
                    speech_accumulated = 0
                    soft_max_last_asr = 0.0
                    sm_text = None
                    sm_result = None
                    sm_audio = np.array([], dtype=np.float32)
                    sm_capture_end_mono_ns = 0
                    stable_punct_candidate = None
                    sm_punct_stable = False
                    should_submit = False
                    should_discard = False
                    continue

            # ---- Normal 提交判断 ----
            should_submit = False
            should_discard = False
            submit_reason = ""

            # 标点感知断句：上一句以句末标点结尾时，更快提交
            effective_silence = PUNCT_SUBMIT_SEC if prev_ended_with_punct else args.silence_submit_sec

            # 新触发器: 批探针句末在下一次结果中仍保持为前缀，且累积够
            # 阈值才切。native 只走上面的 token 时间戳切点，不在此提交整块。
            # - 分语言阈值: 英文 3.0s,中文 5.0s (中文 ASR 容易幻觉句末标点,需要更多上下文)
            # - 中文额外字符数校验: 标点前 >= 12 字才算完整句 (避免 "美联。" 3 字截断)
            # - STRONG_END_PUNCT: 排除 ,;: 等中间标点,只在强句末标点切
            # - overlap_applied=False: 不要在 submit 时叠加 tail_overlap。
            #   tail_overlap 是上一段 final 末尾的 0.3s,提交时叠上去会让 ASR
            #   多听一段音频 → 可能把 sm_text 之后的 "I have to" / "People are"
            #   等续接内容拉进来,产生 mid-sentence 结尾。
            #   probe ASR 用的是当前 chunk 音频,submit 不加 overlap 就跟 probe 看到的音频范围一致。
            punct_end_submit = False
            # 根据语言选阈值
            _cur_lang = sm_language
            _min_chunk_for_punct = PUNCT_END_MIN_CHUNK_SEC_ZH if _cur_lang.startswith("zh") else PUNCT_END_MIN_CHUNK_SEC_EN
            _min_chars_before_punct = MIN_CHARS_BEFORE_PUNCT_ZH if _cur_lang.startswith("zh") else MIN_CHARS_BEFORE_PUNCT_EN
            if (not native_active and not native_stream_fallback_until_commit
                    and has_speech
                    and chunk_sec >= _min_chunk_for_punct and sm_text
                    and sm_punct_stable
                    and sm_text.rstrip() == stable_punct_candidate
                    and punctuation_pause_ready(
                        stable_punct_candidate, silence_streak)):
                # 字符数校验: 标点前的字符数 (去掉尾部空白后)。中文常见 "。" 前有
                # 短句 (5-8 字),如果是 3 字就强烈疑似 ASR 幻觉 (e.g. "美联。")。
                _stripped = sm_text.rstrip()
                _chars_before_punct = len(_stripped) - 1  # 去掉末尾的标点
                if _chars_before_punct >= _min_chars_before_punct:
                    should_submit = True
                    submit_reason = "punct_end"
                    punct_end_submit = True
            elif has_speech and silence_streak >= effective_silence:
                should_submit = True
                submit_reason = "silence"
            elif chunk_sec >= args.max_chunk_sec:
                if has_speech:
                    # native 先等本档右上下文，再由上面的 split 路径在原
                    # max_chunk 边界提交；快照不可用时才走整块批处理回退。
                    if (not native_active
                            or (guarded_max_split and not sm_text)):
                        should_submit = True
                        submit_reason = "max_chunk"
                else:
                    should_discard = True

            if should_submit:
                # punct_end 路径: 用探针时点的 sm_audio (跟 probe ASR 完全相同的音频),
                # 而不是当前 collected (collected 累积到 submit 时多了 0.3s,
                # ASR 会多识别出 "...right? People are" 这种 mid-sentence 尾巴)
                # 提交音频与探针一致时把探针转录一并传下去,submit_chunk 免重转。
                uses_probe_snapshot = False
                probe_remainder = np.array([], dtype=np.float32)
                current_samples = (np.concatenate(collected) if collected
                                   else np.array([], dtype=np.float32))
                if native_active:
                    # 不用可能滞后的 sm_audio；读取当前持久流快照，避免对
                    # 同一 PCM 再跑批量 ASR。只有真实 EOF 才补右 padding。
                    all_samples = current_samples
                    overlap_applied = False
                    applied_overlap_count = 0
                    # silence/max_chunk/EOF 是整块提交，不按时间戳裁尾；仅
                    # 真实 EOF 补右 padding，中间提交必须保留缓存继续解码。
                    reuse_result = native_final_snapshot(
                        None, finalize_stream=synthetic_eof)
                    native_active_for_sync = (
                        native_active
                        and not native_stream_fallback_until_commit)
                    if not native_active_for_sync:
                        # native 快照不可用时，本句立刻按质量档批处理；
                        # 与其他 fallback 一样补回上句尾音上下文。
                        all_samples, applied_overlap_count = (
                            prepare_submission_audio(
                                current_samples, tail_overlap,
                                apply_overlap=len(tail_overlap) > 0)
                        )
                        overlap_applied = applied_overlap_count > 0
                elif punct_end_submit and len(sm_audio) > 0:
                    all_samples, probe_remainder, uses_probe_snapshot = (
                        split_probe_snapshot_audio(current_samples, sm_audio)
                    )
                    overlap_applied = False
                    applied_overlap_count = 0
                    reuse_result = sm_result if uses_probe_snapshot else None
                    native_active_for_sync = native_active
                else:
                    reuse_result = None
                    native_active_for_sync = native_active
                    if punct_end_submit:
                        overlap_applied = False
                        applied_overlap_count = 0
                        all_samples = current_samples
                    else:
                        overlap_applied = (len(tail_overlap) > 0
                                           and len(current_samples) > 0)
                        all_samples, applied_overlap_count = (
                            prepare_submission_audio(
                                current_samples, tail_overlap,
                                apply_overlap=overlap_applied)
                        )
                        overlap_applied = applied_overlap_count > 0
                        if (not overlap_applied and sm_result is not None
                                and len(sm_audio) == len(all_samples)):
                            # silence/max_chunk 提交的音频与探针时点完全一致
                            # (collected 只尾部 append,等长 ⇒ 同一段) → 复用
                            reuse_result = sm_result
                cur_speech = speech_accumulated
                cur_trailing = silence_streak
                submit_capture_end_mono_ns = select_capture_end_mono_ns(
                    uses_probe_snapshot=uses_probe_snapshot,
                    probe_capture_end=sm_capture_end_mono_ns,
                    latest_capture_end=latest_capture_end_mono_ns,
                )
                collected = []
                collected_count = 0
                has_speech = False
                silence_streak = 0
                speech_accumulated = 0
                sm_text = None
                sm_result = None
                sm_audio = np.array([], dtype=np.float32)
                sm_capture_end_mono_ns = 0
                stable_punct_candidate = None
                sm_punct_stable = False
                submitted = submit_chunk(
                    all_samples, chunk_first_seg, seg_end,
                    submit_reason=submit_reason,
                    speech_sec=cur_speech,
                    trailing_silence_sec=cur_trailing,
                    overlap_applied=overlap_applied,
                    overlap_sample_count=applied_overlap_count,
                    precomputed_result=reuse_result,
                    capture_end_mono_ns=submit_capture_end_mono_ns)
                if not submitted:
                    restore_unsubmitted_audio(current_samples)
                    continue
                stream_end_flush.mark_success()
                if probe_remainder.size:
                    # 探针完成后新到达的 PCM 属于下一句，立即按原顺序恢复到
                    # collected；不能等下一轮再插，否则同批后续 packet 会先
                    # 更新 VAD，造成时间顺序颠倒。
                    collected = [probe_remainder]
                    collected_count = len(probe_remainder)
                    chunk_first_seg = max(
                        0, (samples_ingested - len(probe_remainder)) // SAMPLE_RATE)
                    has_speech, silence_streak, speech_accumulated = (
                        carry_remainder_vad(
                            probe_remainder, args.silence_threshold,
                            has_speech=False, silence_streak=0.0,
                            speech_accumulated=0.0)
                    )
                (native_stream_fallback_until_commit,
                 protected_probe_remainder,
                 native_sync_error) = sync_native_after_submit(
                    native_stream, commit_sec=chunk_sec,
                    language=args.language,
                    right_context=args.nemotron_right_context,
                    backend_after_submit=resolved_backend,
                    native_active_before_submit=native_active_for_sync,
                    current_fallback=native_stream_fallback_until_commit,
                    raw_remainder=probe_remainder,
                )
                # sync helper 会复制 raw PCM；若未来调用顺序调整，仍以受保护
                # 的副本覆盖，避免 MLX 内部对输入引用产生副作用。
                if protected_probe_remainder.size:
                    collected = [protected_probe_remainder]
                    collected_count = len(protected_probe_remainder)
                if native_sync_error is not None:
                    logger.log_status(
                        "nemotron_stream_commit_fallback",
                        error=str(native_sync_error))
                advance_source_revision()

            elif should_discard:
                # 纯静音，丢弃但保留尾部 overlap
                if collected:
                    last = collected[-1]
                    tail_overlap = save_tail(last)
                collected = []
                collected_count = 0
                has_speech = False
                silence_streak = 0
                speech_accumulated = 0
                sm_text = None
                sm_result = None
                sm_audio = np.array([], dtype=np.float32)
                sm_capture_end_mono_ns = 0
                stable_punct_candidate = None
                sm_punct_stable = False
                if native_stream is not None:
                    try:
                        native_stream.reset(args.language,
                                            args.nemotron_right_context)
                        native_stream_fallback_until_commit = False
                    except Exception as exc:  # noqa: BLE001
                        native_stream_fallback_until_commit = True
                        logger.log_status(
                            "nemotron_stream_reset_fallback",
                            error=str(exc))
                advance_source_revision()

    except KeyboardInterrupt:
        print("\n退出。", flush=True)
    finally:
        metrics.report()
        logger.close()
        if text_out:
            text_out.close()
        # 关闭音频源——AudioSource.stop() 会把 audiotee 子进程 /
        # sounddevice stream 收掉。
        if audio_source is not None:
            try:
                audio_source.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[audio] stop() 异常: {e}", file=sys.stderr)
        if use_segdir:
            cleanup_seg_dir()

if __name__ == "__main__":
    main()
