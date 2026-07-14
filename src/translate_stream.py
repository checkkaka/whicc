#!/usr/bin/env python3
"""实时翻译消费者（完整源文草稿流 + 严格复用 + 命中式术语注入）。

架构：
  主线程读 events.jsonl → classify_update() 判定增量模式 → 翻译 → 输出
  partial 模式：final/partial 独立线程，partial 每 key 只保留最新

final-only 兼容模式仍保留增量分类：
  append_only     — 新文本只是旧文本的尾巴，只翻增量
  small_rewrite_tail — 前面一致，尾部小改写，翻增量
  reset_full      — 差异太大，整段重翻

用法：
  python3 src/translate_stream.py \
    --events runs/smoke5min/events.jsonl \
    --out-dir runs/smoke5min \
    --mode partial
"""

import argparse
import json
import os
import sys
import threading
import time
import datetime
from queue import Queue

from translator_hy_mt2 import (
    HyMT2Translator, classify_update, detect_language, detect_glossary_hits,
    set_scene_context, get_scene_context, _strip_prompt_leak, _is_bad_output,
)
from languages import normalize_target_language, TargetLanguage

# ponytail: partial 超过 1.5s 无响应即作废；若未来远端 TTFT 超过此值，改为可配置。
PARTIAL_READ_TIMEOUT_SEC = 1.5
# 单次 SSE 至少积累 500ms 再显示中途 token；短请求只显示完整结果。
PARTIAL_DRAFT_INTERVAL_NS = 500_000_000
# ASR 每约 560ms 改一次完整源文时，逐版重译会整段改写。每句首版立即
# 显示，此后所有 revision 共用 1s 展示门限；cache/final 仍处理每一版。
PARTIAL_VISIBLE_INTERVAL_NS = 1_000_000_000
PARTIAL_DRAFT_MIN_CHARS = 4


# ── 共享文件 I/O ─────────────────────────────────────────────────────────────
# lang_config.json 是 macui 设置界面和 Python 后端**共享**的文件。
# 写盘要保证原子性（写临时文件 → fsync → 原子替换），避免 macui 端读到
# 半写状态。也避免后端崩溃留下 .tmp 临时文件（崩溃前如果只完成了第一步
# 就死了，原文件还在，临时文件残留但下次启动会被清掉）。

def _atomic_write_json(path: str, obj) -> None:
    """原子地把 obj 写成 JSON 到 path。

    流程：写 path + ".tmp" → flush + fsync → rename 替换原文件。
    如果原文件存在，rename 在同 inode 上原地完成（保持 macui 端
    文件监视器有效）；如果不存在则创建新文件。
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_user_scene(out_dir: str) -> str:
    """从 lang_config.json 读取用户手填的翻译场景（键名 `scene`）。

    与 event_agent.clear_event / macui LangConfig.sceneText 约定一致。
    文件缺失、键缺失或非字符串时返回空串。
    捕 ValueError 覆盖 JSONDecodeError 与编码损坏的 UnicodeDecodeError。
    """
    path = os.path.join(out_dir, "lang_config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        scene = cfg.get("scene", "")
        return scene.strip() if isinstance(scene, str) else ""
    except (OSError, ValueError, TypeError):
        return ""


# ── 输出过滤 ────────────────────────────────────────────────────────────────────

_EXPLANATION_PATTERNS = [
    "here is the translation",
    "the translation is",
    "translated text",
    "翻译如下",
    "译文如下",
    "以下是翻译",
]

_CONTEXT_ECHO_PATTERNS = [
    "最近上下文",
    "最近的相关",
    "最近的情况",
    "最近的背景",
    "最近的热门",
    "仅供参考，不要重复",
    "仅供参考，请勿重复",
    "英文前文",
    "中文前文",
    "前文衔接",
]


def _is_explanation(text: str) -> bool:
    lower = text.strip().lower()
    if any(lower.startswith(p) for p in _EXPLANATION_PATTERNS):
        return True
    if any(p in text[:200] for p in _CONTEXT_ECHO_PATTERNS):
        return True
    return False


# ── 格式化 ──────────────────────────────────────────────────────────────────────

def _format_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _bilingual_line(event: dict, zh_text: str) -> str:
    t_start = _format_time(event.get("audio_start_sec", 0))
    t_end = _format_time(event.get("audio_end_sec", 0))
    en = event.get("text", "").strip()
    return f"[{t_start} → {t_end}]  {en}\n  → {zh_text}\n"


def _source_key(event: dict) -> str:
    if event.get("source_key"):
        return str(event["source_key"])
    return f"{event['seg_start']}-{event['seg_end']}-{event.get('audio_end_sec', 0)}"


def _is_partial_input(event: dict) -> bool:
    """可见 ASR 草稿和仅供翻译预热的输入都走 partial worker。"""
    return event.get("event_type") in {"partial", "translation_input"}


def _source_only_final(error_event: dict, source_event: dict) -> dict:
    """翻译失败时仍提交稳定原文；错误事件本身由调用方先写入。"""
    completed_ns = time.monotonic_ns()
    source_text = error_event.get("source_text") or source_event.get("text", "")
    return {
        "event_type": "translation_final",
        "source_key": error_event.get("source_key") or _source_key(source_event),
        "revision": source_event.get("revision", 0),
        "source_update_mode": "translation_failed_source_only",
        "source_text": source_text,
        "delta_source_text": source_text,
        "translated_delta_text": "",
        "translated_full_text": "",
        "translate_ms": 0,
        "shared_prefix_len": 0,
        "glossary_hits": [],
        "retried": False,
        "fallback_reason": "translation_failed_source_only",
        "final_reused_partial": False,
        "translation_enqueue_mono_ns": source_event.get("translation_enqueue_mono_ns", 0),
        "translation_complete_mono_ns": completed_ns,
        "event_mono_ns": completed_ns,
        "event_wall_ms": int(time.time() * 1000),
        "finish_reason": "error",
    }


def _validate_final_translation(text: str, target_lang: str) -> tuple[str, str]:
    """返回（清理后译文, 错误原因）；accepted final 不得因坏输出消失。"""
    cleaned, leak_hit = _strip_prompt_leak(text or "")
    if leak_hit:
        return cleaned, "prompt_leak"
    if not cleaned.strip():
        return cleaned, "empty_output"
    if _is_explanation(cleaned):
        return cleaned, "explanation_output"
    if _is_bad_output(cleaned, target_lang):
        return cleaned, "wrong_language_output"
    return cleaned, ""


def _finish_reason_allows_reuse(reason: str | None) -> bool:
    """strict reuse 只接受明确正常结束；截断/过滤/tool call 一律重译。"""
    normalized = str(reason or "").strip().lower()
    return normalized in {"stop", "done", "completed", "eos", "eos_token"}


def resolve_target_lang(source_text: str, target_lang: str | None) -> str:
    """根据源文本和目标语言配置，解析出实际的 target_lang prompt_name。

    target_lang=None 或 "auto" 时自动检测：中文→English，其他→Simplified Chinese。
    否则使用用户指定的语言。
    """
    if target_lang is None or target_lang == "auto":
        src_lang = detect_language(source_text)
        return "English" if src_lang == "zh" else "Simplified Chinese"
    return target_lang


def _log_glossary_hits(translator, source_text: str, target_lang: str):
    """检测并打印词库命中，同时更新命中计数。"""
    glossary = getattr(translator, "glossary", {})
    if not glossary:
        return
    hits = detect_glossary_hits(source_text, glossary, target_lang)
    if hits:
        terms = ", ".join(f"{k}→{v}" for k, v in list(hits.items())[:6])
        print(f"[glossary] 命中 {len(hits)} 个术语: {terms}", flush=True)
        # 更新命中元数据（内存中，定期刷盘）
        _track_hits(translator.glossary_path, glossary, hits)


_glossary_hits_buffer: dict[str, int] = {}
_glossary_flush_counter = 0
GLOSSARY_FLUSH_INTERVAL = 50  # 每 50 次命中刷盘一次


def _track_hits(glossary_path: str | None, glossary: dict, hits: dict):
    """在内存中累计命中次数，定期写回 glossary.json。

    只往 buffer 记增量，不直接改传进来的 glossary（那是 _load_glossary
    的缓存，只含 en2zh/zh2en，没有 _meta——直接 dump 会把磁盘上的
    _meta 整个抹掉，manual 来源信息丢失后术语会被 refresher 当
    unknown 过期删除）。
    """
    global _glossary_flush_counter
    for term in hits:
        _glossary_hits_buffer[term] = _glossary_hits_buffer.get(term, 0) + 1

    _glossary_flush_counter += len(hits)
    if _glossary_flush_counter >= GLOSSARY_FLUSH_INTERVAL:
        _flush_glossary_meta(glossary_path)
        _glossary_flush_counter = 0


def _flush_glossary_meta(glossary_path: str | None):
    """把 buffer 里的命中计数合并进磁盘最新词库后原子写回。

    读盘→合并→写盘，以磁盘为准：macui 手动增删、refresher 新增都
    不会被这次刷盘覆盖，词条与 _meta 全量保留，只更新命中字段。
    """
    if not glossary_path or not _glossary_hits_buffer:
        return
    try:
        import json as _json
        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                fresh = _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            fresh = {}
        if not isinstance(fresh, dict):
            fresh = {}
        meta = fresh.setdefault("_meta", {})
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        for term, count in _glossary_hits_buffer.items():
            info = meta.get(term)
            if not isinstance(info, dict):
                info = {"source": "unknown", "added": now_str, "hits": 0}
            info["last_used"] = now_str
            info["hits"] = int(info.get("hits", 0)) + count
            meta[term] = info
        tmp = glossary_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(fresh, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, glossary_path)
        _glossary_hits_buffer.clear()
    except Exception:
        pass


# ── 增量翻译状态 ────────────────────────────────────────────────────────────────

class TranslationState:
    """维护增量翻译的源/目标缓冲区。"""

    def __init__(self, target_lang: str | None = None):
        self.last_source_text: str = ""
        self.last_translated_text: str = ""
        self.last_source_key: str = ""
        self.target_lang = target_lang  # None = auto

    def classify(self, new_source: str) -> dict:
        """判定 new_source 相对于 last_source_text 的更新模式。"""
        return classify_update(self.last_source_text, new_source)

    def translate_final(self, translator, new_source: str, update_info: dict,
                        event: dict, counts: dict) -> dict | None:
        """增量翻译一条 final 事件。返回输出事件 dict，或 None 表示跳过。"""
        mode = update_info["mode"]
        delta = update_info["delta_source_text"]
        t0 = time.monotonic()

        if mode == "reset_full":
            return self._do_full(translator, new_source, event, counts, t0, mode)

        if not delta:
            # 文本没变化，跳过
            return None

        return self._do_delta(translator, delta, event, counts, t0, mode, update_info)

    def _do_delta(self, translator, delta, event, counts, t0, mode, update_info):
        prev_src_tail = self.last_source_text[-40:] if self.last_source_text else ""
        prev_tgt_tail = self.last_translated_text[-20:] if self.last_translated_text else ""
        target_lang = resolve_target_lang(delta, self.target_lang)
        try:
            delta_zh, translate_ms, meta = translator.translate_delta(
                delta, prev_source_tail=prev_src_tail, prev_target_tail=prev_tgt_tail,
                target_lang=target_lang,
            )
        except Exception as exc:
            counts["errors"] += 1
            return self._error_event(event, delta, str(exc))

        if _is_explanation(delta_zh):
            # 增量输出异常 → fallback 全量重翻
            counts["fallbacks"] += 1
            return self._do_full(translator, event.get("text", ""), event, counts, t0, "reset_full",
                                 fallback_reason=f"bad_delta_output: {delta_zh[:40]}")

        delta_zh, delta_leak_hit = _strip_prompt_leak(delta_zh)
        if delta_leak_hit:
            counts["leak_drops"] = counts.get("leak_drops", 0) + 1
            counts["fallbacks"] += 1
            return self._do_full(
                translator, event.get("text", ""), event, counts, t0,
                "reset_full", fallback_reason="bad_delta_output: prompt_leak")

        if mode == "append_only":
            merged_zh = self.last_translated_text + delta_zh
        elif mode == "small_rewrite_tail":
            # 跨语言字符比例不稳定，用整段拼接（重翻 delta 部分，拼上前缀）
            merged_zh = self.last_translated_text + delta_zh
        else:
            merged_zh = delta_zh

        # 增量合并后的整段也执行 final 门禁；坏输出回退全量重译，若全量
        # 仍失败则由调用方写 error + source-only final，不能吞掉原句。
        merged_zh, invalid_reason = _validate_final_translation(
            merged_zh, resolve_target_lang(event.get("text", ""), self.target_lang))
        if invalid_reason:
            if invalid_reason == "prompt_leak":
                counts["leak_drops"] = counts.get("leak_drops", 0) + 1
            counts["fallbacks"] += 1
            return self._do_full(
                translator, event.get("text", ""), event, counts, t0,
                "reset_full", fallback_reason=f"bad_delta_output: {invalid_reason}")

        self.last_source_text = event.get("text", "")
        self.last_translated_text = merged_zh
        self.last_source_key = _source_key(event)
        counts["translated"] += 1
        counts["deltas"] += 1

        return {
            "event_type": "translation_final",
            "source_key": _source_key(event),
            "source_update_mode": mode,
            "source_text": event.get("text", ""),
            "delta_source_text": delta,
            "translated_delta_text": delta_zh,
            "translated_full_text": merged_zh,
            "translate_ms": round(translate_ms, 1),
            "shared_prefix_len": update_info["shared_prefix_len"],
            "glossary_hits": meta.get("glossary_hits", []),
            "retried": meta.get("retried", False),
            "fallback_reason": "",
        }

    def _do_full(self, translator, new_source, event, counts, t0, mode,
                 fallback_reason=""):
        target_lang = resolve_target_lang(new_source, self.target_lang)
        _log_glossary_hits(translator, new_source, target_lang)
        try:
            zh, translate_ms = translator.translate(new_source, target_lang=target_lang)
        except Exception as exc:
            counts["errors"] += 1
            return self._error_event(event, new_source, str(exc))

        zh, invalid_reason = _validate_final_translation(zh, target_lang)
        if invalid_reason:
            if invalid_reason == "prompt_leak":
                counts["leak_drops"] = counts.get("leak_drops", 0) + 1
            counts["errors"] += 1
            return self._error_event(
                event, new_source, f"bad_full_output: {invalid_reason}")

        self.last_source_text = new_source
        self.last_translated_text = zh
        self.last_source_key = _source_key(event)

        if fallback_reason:
            counts["fallbacks"] += 1
            event_type = "translation_reset"
        else:
            event_type = "translation_final"

        counts["translated"] += 1
        return {
            "event_type": event_type,
            "source_key": _source_key(event),
            "source_update_mode": mode,
            "source_text": new_source,
            "delta_source_text": new_source,
            "translated_delta_text": zh,
            "translated_full_text": zh,
            "translate_ms": round(translate_ms, 1),
            "shared_prefix_len": 0,
            "glossary_hits": [],
            "retried": False,
            "fallback_reason": fallback_reason,
        }

    def _error_event(self, event, source_text, error_msg):
        return {
            "event_type": "translation_error",
            "source_key": _source_key(event),
            "source_seg_start": event.get("seg_start"),
            "source_seg_end": event.get("seg_end"),
            "source_text": source_text,
            "error": error_msg,
            "retriable": True,
        }


# ── 异步翻译工作线程 ──────────────────────────────────────────────────────────

class TranslateWorker:
    """final/partial 独立 worker：final FIFO，partial 每 key 只留最新。"""

    def __init__(self, translator, trans_state, partial_cache,
                 f_trans, f_zh, f_bi, out_lock, counts, is_partial_mode,
                 priority_enabled=True):
        self.translator = translator  # partial 专用连接
        self.final_translator = translator.fork() if hasattr(translator, "fork") else translator
        if hasattr(translator, "set_stream_read_timeout"):
            translator.set_stream_read_timeout(PARTIAL_READ_TIMEOUT_SEC)
        self.trans_state = trans_state
        self.partial_cache = partial_cache
        self.f_trans = f_trans
        self.f_zh = f_zh
        self.f_bi = f_bi
        self.out_lock = out_lock
        self.counts = counts
        self.is_partial_mode = is_partial_mode
        self.priority_enabled = priority_enabled
        self._legacy_lock = threading.Lock()
        self._final_queue: Queue = Queue()
        self._partial_pending: dict[str, tuple] = {}
        self._partial_cv = threading.Condition()
        self._cache_lock = threading.Lock()
        self._stopping = False
        self._active_partial_key: str | None = None
        self._active_partial_revision = 0
        # transport cancel 只允许 final/stop 触发；新 partial revision 仅把
        # 旧结果标 stale，不能掐断通常需 380–542ms 的在途 SSE。
        self._active_partial_transport_cancelled = False
        self._final_revision_by_key: dict[str, int] = {}
        self._last_visible_partial_ns_by_key: dict[str, int] = {}
        self._partial_thread = threading.Thread(target=self._run_partial, daemon=True)
        self._final_thread = threading.Thread(target=self._run_final, daemon=True)

    def start(self):
        self._partial_thread.start()
        self._final_thread.start()

    def stop(self):
        self._stopping = True
        self._final_queue.put(None)
        with self._partial_cv:
            self._partial_cv.notify_all()
        if hasattr(self.translator, "cancel_streaming"):
            self.translator.cancel_streaming()
        self._partial_thread.join(timeout=5)
        if self._partial_thread.is_alive():
            print("[warn] partial worker 未在 5s 内退出", flush=True)
        # sentinel 排在所有已入队 final 之后；必须 drain 完再返回，调用方
        # 随后会关闭 JSONL 文件。固定 120s 超时会在最坏三次 90s HTTP
        # 重试期间提前关文件，使迟到 final 永久丢失。
        self._final_thread.join()

    def dispatch_partial(self, event, source_text, key):
        """同 key 单槽 latest-only；新 revision 仅废弃旧结果，不断 SSE。"""
        event = dict(event)
        event["translation_enqueue_mono_ns"] = time.monotonic_ns()
        revision = int(event.get("revision", 0) or 0)
        with self._partial_cv:
            if self._final_revision_by_key.get(key, -1) >= revision:
                return
            old = self._partial_pending.get(key)
            if old is None or revision >= int(old[0].get("revision", 0) or 0):
                self._partial_pending[key] = (event, source_text, key)
                self._partial_cv.notify()
        # 新 revision 只把旧结果标 stale，让在途请求自然结束后马上处理
        # 单槽中的 latest。若每 320/560ms 都 close SSE，而翻译常需
        # 380–542ms，连续讲话时可能永远产不出首个草稿。真正的连接抢占
        # 只留给 final。

    def dispatch_final(self, event, source_text, key):
        """final 立即进入独立 FIFO，并取消同 key 的 partial。"""
        event = dict(event)
        event["translation_enqueue_mono_ns"] = time.monotonic_ns()
        with self._partial_cv:
            revision = int(event.get("revision", 0) or 0)
            self._final_revision_by_key[key] = max(
                revision, self._final_revision_by_key.get(key, -1))
            if self.priority_enabled:
                self._partial_pending.pop(key, None)
                cancel_active = (self._active_partial_key == key
                                 and self._active_partial_revision <= revision)
                if cancel_active:
                    self._active_partial_transport_cancelled = True
            else:
                cancel_active = False
            # final 已封口后该 key 不会再接受 partial；及时释放长会话状态。
            self._last_visible_partial_ns_by_key.pop(key, None)
        if (cancel_active and hasattr(self.translator, "cancel_streaming")):
            self.translator.cancel_streaming()
        self._final_queue.put((event, source_text, key))

    def _run_partial(self):
        while True:
            with self._partial_cv:
                self._partial_cv.wait_for(lambda: self._stopping or self._partial_pending)
                if self._stopping:
                    return
                key, item = next(iter(self._partial_pending.items()))
                del self._partial_pending[key]
                self._active_partial_key = key
                self._active_partial_revision = int(item[0].get("revision", 0) or 0)
                self._active_partial_transport_cancelled = False
            try:
                if self.priority_enabled:
                    self._do_partial(*item)
                else:
                    with self._legacy_lock:
                        self._do_partial(*item)
            except Exception as exc:
                print(f"\n[warn] partial worker error: {exc}", flush=True)
            finally:
                with self._partial_cv:
                    self._active_partial_key = None
                    self._active_partial_revision = 0
                    self._active_partial_transport_cancelled = False

    def _run_final(self):
        while True:
            item = self._final_queue.get()
            if item is None:
                return
            event, source_text, key = item
            try:
                if self.priority_enabled:
                    out_event = self._do_final(event, source_text, key)
                else:
                    with self._legacy_lock:
                        out_event = self._do_final(event, source_text, key)
            except Exception as exc:
                print(f"\n[warn] final worker error: {exc}", flush=True)
                # accepted ASR final 的最后一道兜底：签名、词库、缓存解析等
                # translate try 外的异常也必须写 error + source-only final。
                try:
                    self.counts["errors"] += 1
                    error_event = self.trans_state._error_event(
                        event, source_text, str(exc))
                    out_event = error_event
                except Exception as fallback_exc:
                    print(f"\n[warn] source-only final 构造失败: {fallback_exc}",
                          flush=True)
                    continue
            # 计算成功后只允许一次 canonical final 写入。辅助 txt 写盘失败
            # 由 writer 内部降级；绝不能再发同 key 的空 final 覆盖正确字幕。
            try:
                self._write_final_output(out_event, event)
            except Exception as write_exc:
                print(f"\n[warn] translation JSONL 写入失败: {write_exc}",
                      flush=True)

    def _do_partial(self, event, source_text, key):
        """始终用当前完整源文流式翻译；结果只能更新 draft。"""
        revision = int(event.get("revision", 0) or 0)
        with self._cache_lock:
            cached = self.partial_cache.get(key)
        if cached and cached.get("source_text") == source_text and cached.get("revision") == revision:
            return

        target_lang = resolve_target_lang(source_text, self.trans_state.target_lang)
        signature = self.translator.request_signature(source_text, target_lang)
        first_token_ns = 0
        last_draft_emit_ns = 0
        last_draft_text = ""
        request_start_ns = time.monotonic_ns()

        def has_newer_pending_locked() -> bool:
            pending = self._partial_pending.get(key)
            return bool(
                pending
                and int(pending[0].get("revision", 0) or 0) > revision)

        def is_stale() -> bool:
            """新 revision 只禁止旧草稿上屏/缓存，不关闭 HTTP 连接。"""
            with self._partial_cv:
                return (
                    self._final_revision_by_key.get(key, -1) >= revision
                    or has_newer_pending_locked()
                )

        def should_cancel_transport() -> bool:
            """只有 final/stop 能抢占 SSE，避免连续 revision 永远无草稿。"""
            with self._partial_cv:
                return (
                    self._stopping
                    or self._final_revision_by_key.get(key, -1) >= revision
                    or (self._active_partial_key == key
                        and self._active_partial_revision == revision
                        and self._active_partial_transport_cancelled)
                )

        def emit_partial(piece: str, cleaned: str, now_ns: int) -> bool:
            actual_request_start_ns = (
                getattr(self.translator, "request_start_mono_ns", 0)
                or request_start_ns)
            ev = {
                "event_type": "translation_partial",
                "source_key": key,
                "revision": revision,
                "source_text": source_text,
                "translated_full_text": cleaned,
                "is_streaming_token": True,
                "streaming_piece": piece,
                "translation_enqueue_mono_ns": event.get("translation_enqueue_mono_ns", 0),
                "http_request_start_mono_ns": actual_request_start_ns,
                "first_token_mono_ns": first_token_ns,
                "event_mono_ns": now_ns,
                "event_wall_ms": int(time.time() * 1000),
            }
            # 与 dispatch_final 共用 condition 锁，使“检查是否已 final + 写
            # partial”成为原子顺序：partial 要么先写、随后被 final 清理，
            # 要么在 final 已登记后被拒绝，绝不会在 final 后复活。
            with self._partial_cv:
                if (self._final_revision_by_key.get(key, -1) >= revision
                        or has_newer_pending_locked()):
                    return False
                last_visible_ns = self._last_visible_partial_ns_by_key.get(key)
                if (last_visible_ns is not None
                        and now_ns - last_visible_ns
                        < PARTIAL_VISIBLE_INTERVAL_NS):
                    return False
                with self.out_lock:
                    self.f_trans.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    self.f_trans.flush()
                self._last_visible_partial_ns_by_key[key] = now_ns
            return True

        def on_token(piece: str, full: str) -> None:
            nonlocal first_token_ns, last_draft_emit_ns, last_draft_text
            if is_stale():
                return
            now_ns = time.monotonic_ns()
            if not first_token_ns:
                first_token_ns = now_ns
            cleaned, leak_hit = _strip_prompt_leak(full)
            if leak_hit:
                return
            if (len(cleaned.strip()) < PARTIAL_DRAFT_MIN_CHARS
                    or now_ns - first_token_ns < PARTIAL_DRAFT_INTERVAL_NS
                    or (last_draft_emit_ns
                        and now_ns - last_draft_emit_ns < PARTIAL_DRAFT_INTERVAL_NS)
                    or cleaned == last_draft_text
                    or _is_explanation(cleaned)
                    or _is_bad_output(cleaned, target_lang)):
                return
            # 单次 SSE 只发送累计全文；跨 revision 的整段改写还会在
            # emit_partial() 里按 source_key 统一限为最多约 1 次/秒。
            if is_stale():
                return
            if emit_partial(piece, cleaned, now_ns):
                last_draft_emit_ns = now_ns
                last_draft_text = cleaned

        try:
            _log_glossary_hits(self.translator, source_text, target_lang)
            if is_stale():
                return
            translated, ms = self.translator.translate_streaming(
                source_text, on_token=on_token, target_lang=target_lang,
                should_cancel=should_cancel_transport)
            request_start_ns = (
                getattr(self.translator, "request_start_mono_ns", 0)
                or request_start_ns)
        except Exception as exc:
            print(f"\n[partial] 全量流式翻译已取消或失败: {exc}", flush=True)
            return

        if is_stale():
            return

        translated, leak_hit = _strip_prompt_leak(translated)
        draft_valid = (bool(translated.strip()) and not leak_hit
                       and not _is_explanation(translated)
                       and not _is_bad_output(translated, target_lang))
        finish_reason = getattr(self.translator, "finish_reason", "")
        reuse_valid = (draft_valid
                       and _finish_reason_allows_reuse(finish_reason))
        completed_ns = time.monotonic_ns()
        # 流结束时补最后一版，确保节流窗口中的尾部 token 不丢。
        if draft_valid and translated != last_draft_text:
            emit_partial("", translated, completed_ns)
        snapshot = {
            "source_text": source_text,
            "translated_text": translated,
            "revision": revision,
            # 优先记录实际发出的最后一个请求；若触发坏输出重试或去上下文
            # 重发，它会与预估签名不同，从而保守地阻止 final 误复用。
            "request_signature": (
                getattr(self.translator, "last_request_signature", "")
                or signature
            ),
            "completed": True,
            "output_valid": reuse_valid,
            "translate_ms": ms,
            "finish_reason": finish_reason,
            "translation_complete_mono_ns": completed_ns,
        }
        with self._partial_cv:
            if self._final_revision_by_key.get(key, -1) >= revision:
                return
            with self._cache_lock:
                current = self.partial_cache.get(key)
                if current is None or revision >= int(
                        current.get("revision", 0) or 0):
                    self.partial_cache[key] = snapshot

        metric = {
            "event_type": "translation_metrics",
            "source_key": key,
            "revision": revision,
            "translation_enqueue_mono_ns": event.get("translation_enqueue_mono_ns", 0),
            "http_request_start_mono_ns": request_start_ns,
            "first_token_mono_ns": first_token_ns,
            "translation_complete_mono_ns": completed_ns,
            "finish_reason": snapshot["finish_reason"],
            "event_mono_ns": completed_ns,
            "event_wall_ms": int(time.time() * 1000),
        }
        with self.out_lock:
            self.f_trans.write(json.dumps(metric, ensure_ascii=False) + "\n")
            self.f_trans.flush()

        with self.out_lock:
            sys.stdout.write(f"\r\033[K[partial] {translated}")
            sys.stdout.flush()

    def _do_final(self, event, source_text, key):
        """严格签名命中才复用，否则 final 用独立连接完整重译。"""
        with self._cache_lock:
            cached_entry = self.partial_cache.pop(key, None)
        had_cached_partial = cached_entry is not None
        # 热加载词库只改主 translator；final 独立连接必须复用同一快照，
        # 否则请求签名与输出风格会在同一句上分叉。
        if (hasattr(self.translator, "glossary")
                and hasattr(self.final_translator, "glossary")):
            self.final_translator.glossary = self.translator.glossary
        target_lang = resolve_target_lang(source_text, self.trans_state.target_lang)
        signature = self.final_translator.request_signature(source_text, target_lang)
        revision = int(event.get("revision", 0) or 0)
        can_reuse = bool(
            cached_entry
            and cached_entry.get("source_text") == source_text
            and int(cached_entry.get("revision", 0) or 0) == revision
            and cached_entry.get("request_signature") == signature
            and cached_entry.get("completed") is True
            and cached_entry.get("output_valid") is True
            and _finish_reason_allows_reuse(
                str(cached_entry.get("finish_reason", "")))
        )
        last_zh = cached_entry.get("translated_text", "") if cached_entry else ""

        _log_glossary_hits(self.final_translator, source_text, target_lang)
        fallback_reason = ""
        zh = ""
        translate_ms = 0.0
        request_start_ns = 0
        final_reused_partial = False

        if can_reuse:
            zh = last_zh
            translate_ms = float(cached_entry.get("translate_ms", 0) or 0)
            final_reused_partial = True
        else:
            request_start_ns = time.monotonic_ns()
            try:
                zh, translate_ms = self.final_translator.translate(
                    source_text, target_lang=target_lang
                )
                request_start_ns = (
                    getattr(self.final_translator,
                            "request_start_mono_ns", 0)
                    or request_start_ns)
            except Exception as exc:
                # 严格复用任一条件未命中时，partial 绝不能升级为 final；
                # 否则场景/术语/参数或 revision 已变化仍会提交旧译文。
                self.counts["errors"] += 1
                return self.trans_state._error_event(event, source_text, str(exc))

        # 3. final 统一门禁：prompt leak、空输出、解释性输出和目标语言
        # 错误都转成 error；writer 随后提交 source-only final，不能吞句。
        zh, invalid_reason = _validate_final_translation(zh, target_lang)
        if invalid_reason:
            with self.out_lock:
                print(
                    f"\r\033[K[warn] invalid final ({invalid_reason}) @ {key} "
                    f"src={source_text[:50]!r}",
                    flush=True,
                )
            if invalid_reason == "prompt_leak":
                self.counts["leak_drops"] = self.counts.get("leak_drops", 0) + 1
            self.counts["errors"] += 1
            return self.trans_state._error_event(
                event, source_text, f"bad_full_output: {invalid_reason}")

        # 4. 决定 source_update_mode — 调试用,UI 不依赖它。
        # - full_translate: 没有 partial (first-final 或 partial 被丢)
        # - full_translate_corrected_partial: 有 partial 且本次覆盖了它
        if final_reused_partial:
            mode = "partial_cache_hit"
        elif had_cached_partial and not fallback_reason:
            mode = "full_translate_corrected_partial"
        elif fallback_reason:
            mode = "full_translate_failed_use_partial"
        else:
            mode = "full_translate"

        # 5. 更新 trans_state 缓冲 — 下一个 final 的 delta 计算需要
        self.trans_state.last_source_text = source_text
        self.trans_state.last_translated_text = zh
        self.trans_state.last_source_key = key
        self.counts["translated"] += 1
        completed_ns = time.monotonic_ns()
        finish_reason = (str(cached_entry.get("finish_reason", ""))
                         if final_reused_partial and cached_entry
                         else getattr(self.final_translator,
                                      "finish_reason", ""))

        out_event = {
            "event_type": "translation_final",
            "source_key": key,
            "revision": revision,
            "source_update_mode": mode,
            "source_text": source_text,
            # delta_source_text / translated_delta_text 字段保留以兼容旧 UI,
            # 但语义上是"本次重译覆盖了之前所有 partial",而非 delta。
            "delta_source_text": source_text,
            "translated_delta_text": zh,
            "translated_full_text": zh,
            "translate_ms": round(translate_ms, 1),
            "shared_prefix_len": 0,
            "glossary_hits": [],
            "retried": False,
            "fallback_reason": fallback_reason,
            "final_reused_partial": final_reused_partial,
            "translation_enqueue_mono_ns": event.get("translation_enqueue_mono_ns", 0),
            "http_request_start_mono_ns": request_start_ns,
            "translation_complete_mono_ns": completed_ns,
            "event_mono_ns": completed_ns,
            "event_wall_ms": int(time.time() * 1000),
            "finish_reason": finish_reason,
        }
        return out_event

    def _write_final_output(self, out_event, event):
        """把 _do_final 返回的 out_event 写到所有 JSONL 文件 + stdout。
        抽出公共写盘逻辑,让 _do_final 内部不再关心 IO。
        """
        if out_event is None:
            return

        if out_event.get("event_type") == "translation_error":
            with self.out_lock:
                print(f"\n[warn] {out_event.get('error', '')[:60]}", flush=True)
                self.f_trans.write(json.dumps(out_event, ensure_ascii=False) + "\n")
                self.f_trans.flush()
            # UI 明确不提交 ASR final，而是等待 translation_final。运行中
            # 翻译失败时仍发 source-only final，避免整句从稳定字幕/历史消失；
            # 这不是复用 partial，译文保持空串，错误事件已单独保留。
            out_event = _source_only_final(out_event, event)

        zh_full = out_event["translated_full_text"]
        t_start = _format_time(event.get("audio_start_sec", 0))
        t_end = _format_time(event.get("audio_end_sec", 0))
        ms = out_event.get("translate_ms", 0)
        mode = out_event.get("source_update_mode", "")

        with self.out_lock:
            if self.is_partial_mode:
                sys.stdout.write("\n")
            label = f"[{mode}]" if mode else ""
            print(f"[final]{label} [{t_start}-{t_end}] {zh_full}  ({ms:.0f}ms)", flush=True)

            # translation_events.jsonl 是 UI 的唯一 canonical 通道，先写且
            # 失败时向上抛；后面的纯文本导出只是辅助产物，各自失败不得
            # 生成第二条 source-only final 覆盖已经提交的正确译文。
            self.f_trans.write(json.dumps(out_event, ensure_ascii=False) + "\n")
            self.f_trans.flush()
            try:
                self.f_zh.write(zh_full + "\n")
                self.f_zh.flush()
            except Exception as exc:
                print(f"[warn] translation_zh.txt 写入失败: {exc}",
                      flush=True)
            try:
                self.f_bi.write(_bilingual_line(event, zh_full))
                self.f_bi.flush()
            except Exception as exc:
                print(f"[warn] translation_bilingual.txt 写入失败: {exc}",
                      flush=True)


# ── 状态管理 ────────────────────────────────────────────────────────────────────

def _load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"byte_offset": 0}


def _save_state(path: str, state: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)


# ── 主循环 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="实时翻译消费者（增量翻译）")
    parser.add_argument("--events", required=True, help="whicc.py 的 events.jsonl 路径")
    parser.add_argument("--out-dir", required=True, help="翻译输出目录")
    # --model-id 默认值是 ""
    # 没在 UI 配 translation_model 时,不该给 vLLM 塞个默认模型 ID
    # (不同 vLLM 部署挂的模型不一样,塞错会导致 404/加载错模型)。
    # 下面 main() 里有"lang_config > CLI > model_id"优先级;这里空串
    # 表示"完全没默认值",CLI 显式 --model-id 才用。
    parser.add_argument("--model-id", default="",
                        help="默认翻译模型 ID (空=无默认值,推荐用"
                             " lang_config.json:translation_model 或"
                             " --translation-model)。")
    parser.add_argument("--translation-model", default="",
                        help="远端翻译节点(vLLM/LM Studio)的模型名称，发到 /v1/chat/completions "
                             "请求体的 model 字段。空 = 用 --model-id。也可由 lang_config.json 的"
                             " translation_model 键覆盖。")
    parser.add_argument("--vllm-url", default="",
                        help="远端翻译节点 URL。空 = 不在 CLI 设默认值,只走"
                             " lang_config.json 配的 translation_url /"
                             " translation_fallback_url。")
    parser.add_argument("--vllm-fallback-url", default="",
                        help="远端翻译回退 URL (本机 LM Studio 等)。"
                             "空 = 同上,只走 lang_config.json。")
    parser.add_argument("--glossary", default=os.path.join(os.path.dirname(__file__), "glossary.json"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.6)
    # 固定英文递增输入 A/B 通过后对齐 Hy-MT2 官方参数；partial/final
    # 共用同一 translator 配置，避免同一句草稿与稳定字幕风格突变。
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--poll-interval", type=float, default=0.05,
                        help="轮询间隔（秒，默认 0.05）")
    parser.add_argument("--max-new-tokens", type=int, default=80,
                        help="最大生成 token 数（默认 80）")
    parser.add_argument("--once", action="store_true", help="只处理已有事件，不持续 tail")
    parser.add_argument("--context-size", type=int, default=0,
                        help="传给翻译器的上下文条数（0=不传）")
    parser.add_argument("--mode", default="final", choices=["final", "partial"],
                        help="final=只翻 final 事件；partial=翻 partial 事件（同声传译模式）")
    parser.add_argument("--target-lang", default="auto",
                        help="目标语言（auto=自动检测，或语言名如 'Japanese', 'German', 'zh-cn'）")
    parser.add_argument("--force-enable", action="store_true",
                        help="绕过 lang_config.json 的 translation_enabled 检查 (用于 .app 打包模式)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    trans_events_path = os.path.join(args.out_dir, "translation_events.jsonl")
    zh_txt_path = os.path.join(args.out_dir, "translation_zh.txt")
    bilingual_path = os.path.join(args.out_dir, "translation_bilingual.txt")
    state_path = os.path.join(args.out_dir, "translation_state.json")

    # 从 lang_config.json 读取 translation_url（UI 配置优先）
    # 注意：lang_config.json 是 macui 设置界面和 Python 后端**共享**的文件。
    # macui 可能在其中存 scene_text、hermes_host 等其他键，Python 端不能
    # 整体覆盖（dict 全量 json.dump）—— 那会把其他键全删掉。
    # 标准做法：只读自己关心的键，写回时用 read-modify-write。
    lang_cfg = {}
    try:
        with open(os.path.join(args.out_dir, "lang_config.json"), "r") as f:
            lang_cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    configured_url = (lang_cfg.get("translation_url") or "").strip()
    configured_fb = (lang_cfg.get("translation_fallback_url") or "").strip()
    # 启动时加载手填场景（事件临时场景稍后由热重载覆盖）
    _startup_scene = (lang_cfg.get("scene") or "").strip() if isinstance(lang_cfg.get("scene"), str) else ""
    if _startup_scene:
        set_scene_context(_startup_scene)
        print(f"[translate] 手填场景已加载: {_startup_scene[:40]}", flush=True)
    else:
        set_scene_context("")
    # 优先级: lang_config.json (用户 UI 配) > CLI 参数
    # 现在 CLI 默认值是 "" 空串,
    # 用户没在 UI 配翻译节点 → vllm_url = "", 下面 candidates 是空,
    # translate_stream 干净退出 + 提示用户去 macui 配翻译。
    vllm_url = configured_url if configured_url else (args.vllm_url or "").strip()
    # 不要把 CLI 默认值写回 lang_config.json — 那是用户私有的局域网地址,
    # 写回去会让 Ethan 之类朋友看到你家的 vllm URL 当成默认。
    if not configured_url and vllm_url:
        # 用户通过 CLI 显式传了 --vllm-url, 把这个写回 lang_config.json
        # 默认值(我们在 BackendLauncher 不传了,这里要确保)。
        lang_cfg["translation_url"] = vllm_url
        try:
            _atomic_write_json(os.path.join(args.out_dir, "lang_config.json"), lang_cfg)
        except OSError:
            pass
    if configured_url:
        if not vllm_url.startswith("http"):
            vllm_url = f"http://{vllm_url}"
        print(f"[translate] 使用配置的翻译节点: {vllm_url}", flush=True)
    if configured_fb:
        if not configured_fb.startswith("http"):
            configured_fb = f"http://{configured_fb}"
        print(f"[translate] 使用配置的翻译回退: {configured_fb}", flush=True)

    # 翻译节点总开关:lang_config.json 的 translation_enabled 决定是否
    # 真的连远端翻译节点。默认 False — 即使填了 URL 也不会自动启用,
    # 用户必须显式开。关 = 翻译不可用 (不再回退到本地,本地 transformers
    # 后端已删除)。改完需重启 translate_stream 生效。
    translation_enabled = bool(lang_cfg.get("translation_enabled", False))
    if not translation_enabled and not args.force_enable:
        print(f"[translate] translation_enabled=False,翻译未启用,退出", flush=True)
        print(f"[translate] 请在 macui 设置 → 服务配置 → 启用远端翻译,"
              f"并配置翻译节点地址", flush=True)
        sys.exit(3)  # 3 = 等配置(非故障),BackendLauncher 监控按 code 区分提示
    if args.force_enable and not translation_enabled:
        print(f"[translate] --force-enable 覆盖 lang_config.json 的 enabled=False,"
              f"继续启动 (打包模式 .app 行为)", flush=True)
    print(f"[translate] translation_enabled=True,允许走远端翻译", flush=True)

    # 远端模型名称：lang_config.json 里的 translation_model 优先，
    # 否则用 --translation-model CLI，否则回退到 --model-id 默认值。
    # 这样 UI 配置能覆盖 CLI 默认，CLI 又能覆盖硬编码值。
    configured_model = (lang_cfg.get("translation_model") or "").strip()
    cli_model = (args.translation_model or "").strip()
    if configured_model:
        model_id = configured_model
        print(f"[translate] 使用配置的远端模型: {model_id}", flush=True)
    elif cli_model:
        model_id = cli_model
        print(f"[translate] 使用 CLI 指定的远端模型: {model_id}", flush=True)
    else:
        model_id = args.model_id
        if model_id:
            print(f"[translate] 使用 --model-id 默认模型: {model_id}", flush=True)
        else:
            # 完全没配置模型名 — 不给 vLLM 塞个写死的 Hy-MT2 (用户 vLLM 上可能
            # 挂着完全不同的模型,塞错会导致 404/load 错模型)。发请求时 model
            # 字段留空,让 vLLM 服务端用自己加载的默认模型。
            print(f"[translate] 未配置 translation_model / --translation-model / "
                  f"--model-id,将让 vLLM 服务端自行选择默认模型", flush=True)

    # Fallback 用的模型名: lang_config.json:translation_fallback_model
    # 未配置 → 跟主 URL 共用 model_id (向后兼容,旧用户零迁移)
    # 如果主 model_id 也是空 (用户没在 UI/CLI 配),fb 也保持空,
    # 由 vLLM 服务端自己选默认模型。
    configured_fb_model = (lang_cfg.get("translation_fallback_model") or "").strip()
    fb_model_id = configured_fb_model if configured_fb_model else model_id
    if configured_fb_model:
        print(f"[translate] 使用配置的 fallback 模型: {fb_model_id}", flush=True)
    elif model_id:
        print(f"[translate] fallback 模型未配置,沿用主模型: {fb_model_id}", flush=True)
    else:
        print(f"[translate] fallback 模型未配置,主模型也未配置,fb 留空", flush=True)

    # 翻译节点 fallback 链（按优先级，VLLMBackend 内部按序探活挑首个健康）:
    # 1. lang_config.json:translation_url          (UI 主 URL)
    # 2. lang_config.json:translation_fallback_url (UI fallback URL)
    # 3. --vllm-url CLI 参数 (外部脚本可能传,BackendLauncher 不传)
    # 4. --vllm-fallback-url CLI 参数 (同上)
    # CLI 的 --vllm-url / --vllm-fallback-url 默认值都是 ""
    # 只有任一来源配出 URL 才会进 candidates;都空就下面 if not candidates 退出。
    candidates: list[str] = []
    if vllm_url:
        candidates.append(vllm_url)
    if configured_fb and configured_fb not in candidates:
        candidates.append(configured_fb)
    cli_fb = (args.vllm_fallback_url or "").strip()
    if cli_fb and cli_fb not in candidates:
        candidates.append(cli_fb)

    if not candidates:
        print(f"[translate] 未配置任何翻译节点 URL (translation_url / "
              f"translation_fallback_url / --vllm-url / --vllm-fallback-url),退出", flush=True)
        # exit 3 = 等配置,不是故障。用户在设置页填好地址(lang_config
        # 保存)后,BackendLauncher 监控检测到配置文件更新会立即重启
        # 本进程,新 URL 生效 — 用户不用手动"保存并重启"。
        sys.exit(3)

    # per-URL model_map: fallback URL 用 fb_model_id (可能跟主 URL 不同)
    # 主 URL 用 model_id (fallback map 里没写)。backend 内部 _resolve_ipv4
    # 标准化 key,所以上层传原始 URL string 即可。
    model_map: dict[str, str] = {}
    if configured_fb and configured_fb_model:
        model_map[configured_fb] = fb_model_id
    if cli_fb and cli_fb != configured_fb and configured_fb_model:
        # CLI fallback (外部脚本可能传) 用配置的 fb model — 也算"本机"
        model_map[cli_fb] = fb_model_id

    # per-URL API key: 主/备节点各自的鉴权 key(macui 设置页配,可空)。
    # 非空时该 URL 的所有请求(探活 + 翻译)带 Authorization: Bearer 头。
    api_key_map: dict[str, str] = {}
    main_api_key = (lang_cfg.get("translation_api_key") or "").strip()
    fb_api_key = (lang_cfg.get("translation_fallback_api_key") or "").strip()
    if vllm_url and main_api_key:
        api_key_map[vllm_url] = main_api_key
    if configured_fb and fb_api_key:
        api_key_map[configured_fb] = fb_api_key
    if cli_fb and cli_fb != configured_fb and fb_api_key:
        api_key_map[cli_fb] = fb_api_key
    if api_key_map:
        print(f"[translate] 已配置 API key ({len(api_key_map)} 个节点)", flush=True)

    # 请求端点: auto(默认,404 自适应) / chat(/v1/chat/completions) /
    # responses(/v1/responses — GPT 系列流式端点,一些站点只提供它)
    endpoint = (lang_cfg.get("translation_endpoint") or "auto").strip().lower()
    if endpoint not in ("auto", "chat", "responses"):
        print(f"[translate] 未知 translation_endpoint '{endpoint}',回退 auto", flush=True)
        endpoint = "auto"
    print(f"[translate] 翻译请求端点: {endpoint}", flush=True)

    # __init__ 内部就做健康探活,失败抛异常(早期版本先建一个丢弃的实例
    # 探活再建正式的,启动时白跑两次 GET /v1/models,已合并)。
    #
    # 节点全不可达**不再退出** — 之前 sys.exit(1) 意味着用户后开
    # LM Studio 也没用,必须手动"保存并重启"。改为等待模式:每 15s
    # 重试,期间字幕照常显示原文(事件跳过不积压),连上自动恢复。
    # --once(离线评估)保留立即退出:评估跑批不该空等。
    def _try_build_translator():
        try:
            t = HyMT2Translator(
                model_id=model_id,
                vllm_url=candidates,
                model_map=model_map,
                glossary_path=args.glossary,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                api_key_map=api_key_map,
                endpoint=endpoint,
            )
            print(f"[translate] 已连接翻译服务,候选: {candidates}", flush=True)
            return t
        except Exception as e:
            print(f"[translate] 翻译节点暂不可达: {e}", flush=True)
            return None

    TRANSLATOR_RETRY_SEC = 15.0
    translator = _try_build_translator()
    last_translator_retry = time.monotonic()
    if translator is None:
        if args.once:
            print("[translate] --once 模式且节点不可达,退出", flush=True)
            sys.exit(1)
        print(f"[translate] 进入等待模式:每 {TRANSLATOR_RETRY_SEC:.0f}s 重试,"
              f"字幕先显示原文,翻译服务可达后自动恢复", flush=True)

    state = _load_state(state_path)
    byte_offset = state.get("byte_offset", 0)
    pending_partial_line = state.get("pending_partial_line", "")

    f_trans = open(trans_events_path, "a", encoding="utf-8")
    f_zh = open(zh_txt_path, "a", encoding="utf-8")
    f_bi = open(bilingual_path, "a", encoding="utf-8")

    if translator is None:
        # 一次性告知 UI 进入"仅原文"模式 — 正式字幕走上面的原文透传,
        # 但用户得知道译文为什么没了(走 translation_final 通道,跟
        # BackendLauncher.appendBackendNotice 同款形态)。
        _notice = {
            "event_type": "translation_final",
            "source_key": f"notice-{int(time.time() * 1000)}",
            "source_update_mode": "reset_full",
            "source_text": "Translation service unreachable — captions show "
                           "source only; auto-recovers once reachable",
            "translated_full_text": "⚠️ 翻译服务不可达,字幕暂只显示原文;"
                                    "连上后自动恢复(设置 → 服务配置 可检查地址)",
            "translate_ms": 0,
            "shared_prefix_len": 0,
            "glossary_hits": [],
            "retried": False,
            "fallback_reason": "",
        }
        f_trans.write(json.dumps(_notice, ensure_ascii=False) + "\n")
        f_trans.flush()

    counts = {"translated": 0, "errors": 0, "deltas": 0, "fallbacks": 0}
    is_partial_mode = args.mode == "partial"
    translation_priority_enabled = bool(
        lang_cfg.get("translation_priority_enabled", True)
    )
    out_lock = threading.Lock()

    # 解析目标语言
    initial_target_lang = None  # None = auto
    if args.target_lang.lower() != "auto":
        try:
            tl = normalize_target_language(args.target_lang)
            initial_target_lang = tl.prompt_name
            print(f"[translate] 目标语言: {tl.prompt_name} ({tl.code})", flush=True)
        except ValueError as e:
            print(f"[translate] 未知语言 '{args.target_lang}'，使用自动模式: {e}", flush=True)
    else:
        print(f"[translate] 目标语言: 自动（中文→英文，其他→中文）", flush=True)

    # lang_config.json 热重载路径
    lang_config_path = os.path.join(args.out_dir, "lang_config.json")

    trans_state = TranslationState(target_lang=initial_target_lang)
    partial_cache: dict[str, dict] = {}

    worker = None

    def _start_worker():
        """partial 模式的异步翻译线程。translator 就绪时才建 —
        等待模式下重连成功后由主循环补建。"""
        nonlocal worker
        if is_partial_mode and worker is None and translator is not None:
            worker = TranslateWorker(
                translator, trans_state, partial_cache,
                f_trans, f_zh, f_bi, out_lock, counts, is_partial_mode,
                translation_priority_enabled,
            )
            worker.start()
            print("[translate] 同声传译模式（完整源文流式草稿 + final 独立线程）", flush=True)

    _start_worker()

    def _flush_state():
        # 注: 早期版本还持久化 processed_keys/failed_keys 两个集合,但去重
        # 逻辑已改走 partial_cache,它们只增不读、每次全量序列化 — 长会话
        # 内存与磁盘 IO 无限上涨,已删除。旧 state 文件里的多余键被忽略。
        _save_state(state_path, {
            "byte_offset": byte_offset,
            "pending_partial_line": pending_partial_line,
        })

    def _write_output(event, out_event):
        """写翻译结果到文件和终端（final 模式同步调用）。"""
        if out_event is None or out_event.get("event_type") == "translation_error":
            return
        zh_full = out_event["translated_full_text"]
        t_start = _format_time(event.get("audio_start_sec", 0))
        t_end = _format_time(event.get("audio_end_sec", 0))
        ms = out_event.get("translate_ms", 0)
        mode = out_event.get("source_update_mode", "")
        label = f"[{mode}]" if mode else ""

        with out_lock:
            print(f"[final]{label} [{t_start}-{t_end}] {zh_full}  ({ms:.0f}ms)", flush=True)
            f_trans.write(json.dumps(out_event, ensure_ascii=False) + "\n")
            f_trans.flush()
            f_zh.write(zh_full + "\n")
            f_zh.flush()
            f_bi.write(_bilingual_line(event, zh_full))
            f_bi.flush()

    print(f"[translate] 开始消费 {args.events} (offset={byte_offset})", flush=True)

    # ── 词库热加载 ──
    _glossary_path = args.glossary
    try:
        _last_glossary_mtime = os.path.getmtime(_glossary_path)
    except OSError:
        _last_glossary_mtime = 0.0
    _glossary_check_counter = 0

    def _try_reload_glossary():
        nonlocal _last_glossary_mtime, _glossary_check_counter
        if translator is None:
            return  # 等待模式:重连成功后 mtime 会被重置强制重灌
        _glossary_check_counter += 1
        if _glossary_check_counter % 200 != 0:  # 每 200 次循环检查一次
            return
        try:
            mtime = os.path.getmtime(_glossary_path)
        except OSError:
            return
        if mtime <= _last_glossary_mtime:
            return
        try:
            with open(_glossary_path, "r", encoding="utf-8") as gf:
                new_glossary = json.load(gf)
            _base_glossary = new_glossary
            # 合并临时事件词库
            _merge_event_glossary()
            _last_glossary_mtime = mtime
            total = sum(len(v) for v in translator.glossary.values())
            print(f"[translate] 词库热加载完成（{total} 条）", flush=True)
        except Exception as exc:
            print(f"[translate] 词库加载失败: {exc}", flush=True)

    # ── 目标语言热重载 ──
    _last_lang_config_mtime = 0.0
    _lang_check_counter = 0

    # ── 临时事件词库/场景热重载 ──
    event_glossary_path = os.path.join(args.out_dir, "event_glossary.json")
    event_scene_path = os.path.join(args.out_dir, "event_scene.json")
    _last_event_glossary_mtime = 0.0
    _event_glossary_check_counter = 0
    _last_event_scene_mtime = 0.0
    _event_scene_check_counter = 0
    # 保存永久词库快照(等待模式下 translator 为 None,先空着 —
    # 重连成功时主循环会重新初始化并强制重灌词库)
    _base_glossary = dict(translator.glossary) if translator is not None else {}
    _event_active = False  # 当前是否有活跃事件
    _lang_check_counter = 0

    def _try_reload_lang():
        nonlocal _last_lang_config_mtime, _lang_check_counter
        _lang_check_counter += 1
        if _lang_check_counter % 200 != 0:
            return
        try:
            mtime = os.path.getmtime(lang_config_path)
        except OSError:
            return
        if mtime <= _last_lang_config_mtime:
            return
        try:
            with open(lang_config_path, "r", encoding="utf-8") as lf:
                cfg = json.load(lf)
            new_lang = cfg.get("target_lang", "auto")
            if new_lang.lower() == "auto":
                trans_state.target_lang = None
                print(f"[translate] 目标语言切换: 自动", flush=True)
            else:
                tl = normalize_target_language(new_lang)
                trans_state.target_lang = tl.prompt_name
                print(f"[translate] 目标语言切换: {tl.prompt_name} ({tl.code})", flush=True)
            # 手填场景热重载：事件激活期间不覆盖临时事件场景
            if not _event_active:
                new_scene = cfg.get("scene", "")
                new_scene = new_scene.strip() if isinstance(new_scene, str) else ""
                if new_scene != get_scene_context():
                    set_scene_context(new_scene)
                    if new_scene:
                        print(f"[translate] 手填场景切换: {new_scene[:40]}", flush=True)
                    else:
                        print("[translate] 手填场景已清空", flush=True)
            _last_lang_config_mtime = mtime
        except Exception as exc:
            print(f"[translate] 语言配置加载失败: {exc}", flush=True)

    # ── 临时事件词库合并 ──
    def _merge_event_glossary():
        """将 event_glossary 合并到 _base_glossary 上，不修改 _base_glossary。"""
        if translator is None:
            return  # 等待模式:重连后强制重灌时再合并
        event_gloss = {}
        try:
            with open(event_glossary_path, "r", encoding="utf-8") as ef:
                event_gloss = json.load(ef)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        merged = dict(_base_glossary)
        merged["en2zh"] = dict(merged.get("en2zh", {}))
        merged["zh2en"] = dict(merged.get("zh2en", {}))
        for en, value in event_gloss.get("en2zh", {}).items():
            if isinstance(value, dict):
                merged["en2zh"][en] = value.get("translation", en)
            else:
                merged["en2zh"][en] = value
        for zh, value in event_gloss.get("zh2en", {}).items():
            if isinstance(value, dict):
                merged["zh2en"][zh] = value.get("translation", zh)
            else:
                merged["zh2en"][zh] = value
        translator.glossary = merged

    # ── 临时事件词库热重载 ──
    def _try_reload_event_glossary():
        nonlocal _last_event_glossary_mtime, _event_glossary_check_counter
        if translator is None:
            return  # 等待模式:重连后 mtime 重置强制重灌
        _event_glossary_check_counter += 1
        if _event_glossary_check_counter % 200 != 0:
            return
        try:
            mtime = os.path.getmtime(event_glossary_path)
        except OSError:
            return
        if mtime <= _last_event_glossary_mtime:
            return
        _last_event_glossary_mtime = mtime
        _merge_event_glossary()
        total = sum(len(v) for v in translator.glossary.values())
        event_en = len(translator.glossary.get("en2zh", {})) - len(_base_glossary.get("en2zh", {}))
        event_zh = len(translator.glossary.get("zh2en", {})) - len(_base_glossary.get("zh2en", {}))
        if event_en > 0 or event_zh > 0:
            print(f"[event] 临时词库已加载 (en2zh+{event_en}, zh2en+{event_zh}, 总{total})", flush=True)

    # ── 临时事件场景热重载 ──
    def _try_reload_event_scene():
        nonlocal _last_event_scene_mtime, _event_scene_check_counter, _event_active
        if translator is None:
            return  # 等待模式:重连后 mtime 重置强制重灌
        _event_scene_check_counter += 1
        if _event_scene_check_counter % 200 != 0:
            return
        try:
            mtime = os.path.getmtime(event_scene_path)
        except OSError:
            return
        if mtime <= _last_event_scene_mtime:
            return
        _last_event_scene_mtime = mtime
        try:
            with open(event_scene_path, "r", encoding="utf-8") as sf:
                scene_data = json.load(sf)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        status = scene_data.get("status", "")
        if status == "applied":
            scene_text = scene_data.get("temp_scene_text", "")
            event_name = scene_data.get("event_name", "")
            expires_at = scene_data.get("expires_at", "")
            # 检查 TTL 是否过期
            if expires_at:
                try:
                    exp = datetime.datetime.fromisoformat(expires_at)
                    if datetime.datetime.now() > exp:
                        # TTL 过期，清除事件，恢复用户手填场景 + base glossary
                        print(f"[event] 事件 '{event_name}' TTL 过期，清除场景", flush=True)
                        _event_active = False
                        set_scene_context(read_user_scene(args.out_dir))
                        translator.glossary = {"en2zh": dict(_base_glossary.get("en2zh", {})),
                                               "zh2en": dict(_base_glossary.get("zh2en", {}))}
                        return
                except ValueError:
                    pass
            _event_active = True
            set_scene_context(scene_text)
            print(f"[event] 场景已应用: {event_name} → {scene_text[:40]}...", flush=True)
        elif status == "idle":
            if _event_active:
                _event_active = False
                # 调用恢复手填场景：事件清除后回落 lang_config.scene，而非清空
                set_scene_context(read_user_scene(args.out_dir))
                translator.glossary = {"en2zh": dict(_base_glossary.get("en2zh", {})),
                                       "zh2en": dict(_base_glossary.get("zh2en", {}))}
                print(f"[event] 事件已清除", flush=True)

    try:
        while True:
            # 等待模式:翻译服务不可达时限频重连,连上自动恢复(重灌
            # 词库 + 补建 worker),用户不用"保存并重启"。
            if translator is None and \
                    time.monotonic() - last_translator_retry >= TRANSLATOR_RETRY_SEC:
                last_translator_retry = time.monotonic()
                translator = _try_build_translator()
                if translator is not None:
                    _base_glossary = dict(translator.glossary)
                    _last_glossary_mtime = 0.0        # 强制重灌词库
                    _last_event_glossary_mtime = 0.0  # 强制重灌事件词库
                    _start_worker()

            _try_reload_glossary()
            _try_reload_lang()
            _try_reload_event_glossary()
            _try_reload_event_scene()
            try:
                fsize = os.path.getsize(args.events)
            except OSError:
                if args.once:
                    break
                time.sleep(args.poll_interval)
                continue

            if fsize < byte_offset:
                # 文件被截断/rotate/truncate — 之前累积的 offset 失效。
                # 重置到文件开头,避免永久卡死。
                print(f"[translate] events.jsonl was truncated/rotated "
                      f"(fsize={fsize} < byte_offset={byte_offset}), "
                      f"resetting offset", flush=True)
                byte_offset = 0
                # 也清空 saved state 里的旧 offset,持久层也跟上
                _flush_state()
            elif fsize == byte_offset:
                if args.once:
                    break
                time.sleep(args.poll_interval)
                continue

            with open(args.events, "rb") as ef:
                ef.seek(byte_offset)
                new_data = ef.read()
                new_byte_offset = ef.tell()

            chunk_text = pending_partial_line + new_data.decode("utf-8", errors="replace")
            raw_lines = chunk_text.splitlines(keepends=True)
            pending_partial_line = ""

            for raw_line in raw_lines:
                if not raw_line.endswith("\n"):
                    pending_partial_line = raw_line
                    break

                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("event_type")

                # 等待模式:服务不可达期间不翻译(offset 照常推进,不
                # 积压 — 实时字幕翻旧内容没有意义),但 ASR final 必须
                # **原文透传** — UI 正式字幕的唯一 commit 通道是
                # translation_final(OverlayState.applyFinal),之前这里
                # 直接 continue,以为"字幕区还有 ASR 原文 draft"顶着,
                # 实际 final 一发 draft 就走清理链,翻译节点不可达时
                # 用户一个字都看不到。译文发空串:SubtitleCaption 对空
                # 译文只渲染原文行,连上后自动恢复双语。
                if translator is None:
                    if etype == "final" and event.get("accepted", False):
                        src_text = event.get("text", "").strip()
                        if src_text:
                            passthrough = {
                                "event_type": "translation_final",
                                "source_key": _source_key(event),
                                "source_update_mode": "no_translator_passthrough",
                                "source_text": src_text,
                                "delta_source_text": src_text,
                                "translated_delta_text": "",
                                "translated_full_text": "",
                                "translate_ms": 0,
                                "shared_prefix_len": 0,
                                "glossary_hits": [],
                                "retried": False,
                                "fallback_reason": "translator_unavailable",
                            }
                            with out_lock:
                                f_trans.write(json.dumps(
                                    passthrough, ensure_ascii=False) + "\n")
                                f_trans.flush()
                    continue

                # ── partial 事件（partial 模式，异步翻译 + 显示）──
                if is_partial_mode and _is_partial_input(event):
                    key = _source_key(event)
                    source_text = event.get("text", "").strip()
                    if not source_text:
                        continue
                    cached = partial_cache.get(key)
                    if cached and cached.get("source_text") == source_text and \
                            int(cached.get("revision", 0) or 0) == int(event.get("revision", 0) or 0):
                        continue
                    worker.dispatch_partial(event, source_text, key)
                    continue

                # ── final 事件 ──
                if etype != "final":
                    continue
                if not event.get("accepted", False):
                    continue

                key = _source_key(event)
                # 不用 processed_keys 去重 — ASR 可能对同一段发多个 final（修正）
                # partial_cache 已经处理了重复翻译的问题

                source_text = event.get("text", "").strip()
                if not source_text:
                    continue

                if is_partial_mode and worker is not None:
                    # 异步：dispatch 给 worker，worker 内部做 classify + translate
                    worker.dispatch_final(event, source_text, key)
                else:
                    # 同步：主线程做 classify + translate + write
                    event = dict(event)
                    event["translation_enqueue_mono_ns"] = time.monotonic_ns()
                    update_info = trans_state.classify(source_text)
                    request_start_ns = time.monotonic_ns()
                    out_event = trans_state.translate_final(
                        translator, source_text, update_info, event, counts,
                    )
                    request_start_ns = (
                        getattr(translator, "request_start_mono_ns", 0)
                        or request_start_ns)
                    if out_event is not None:
                        completed_ns = time.monotonic_ns()
                        out_event.update({
                            "revision": event.get("revision", 0),
                            "translation_enqueue_mono_ns": event["translation_enqueue_mono_ns"],
                            "http_request_start_mono_ns": request_start_ns,
                            "translation_complete_mono_ns": completed_ns,
                            "event_mono_ns": completed_ns,
                            "event_wall_ms": int(time.time() * 1000),
                            "finish_reason": getattr(translator, "finish_reason", ""),
                        })
                    if out_event and out_event.get("event_type") == "translation_error":
                        with out_lock:
                            f_trans.write(json.dumps(out_event, ensure_ascii=False) + "\n")
                            f_trans.flush()
                        # final-only 模式同样由 translation_final 驱动 UI；
                        # 翻译失败时提交空译文的稳定原文，不能让整句消失。
                        _write_output(event, _source_only_final(out_event, event))
                        _flush_state()
                        continue
                    _write_output(event, out_event)

            byte_offset = new_byte_offset
            _flush_state()
            if args.once:
                break

    except KeyboardInterrupt:
        print("\n[translate] 退出。", flush=True)
    finally:
        if worker is not None:
            worker.stop()
        _flush_state()
        f_trans.close()
        f_zh.close()
        f_bi.close()
        print(f"[translate] 完成：{counts['translated']} 条翻译, "
              f"{counts['errors']} 条错误, "
              f"{counts['deltas']} 条增量, "
              f"{counts['fallbacks']} 次 fallback", flush=True)


if __name__ == "__main__":
    main()
