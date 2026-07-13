#!/usr/bin/env python3
"""实时翻译消费者（增量翻译 + 命中式术语注入）。

架构：
  主线程读 events.jsonl → classify_update() 判定增量模式 → 翻译 → 输出
  partial 模式：后台翻译线程异步处理，不阻塞主线程

增量模式：
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
import itertools
import json
import os
import sys
import threading
import time
import datetime
from queue import Queue, PriorityQueue, Empty

from translator_hy_mt2 import (
    HyMT2Translator, classify_update, detect_language, detect_glossary_hits,
    set_scene_context, get_scene_context, _strip_prompt_leak,
    TranslationCancelled,
)
from languages import normalize_target_language, TargetLanguage


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
    """优先用事件自带的稳定 source_key；否则回退 seg 拼接键。"""
    existing = event.get("source_key")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    return f"{event['seg_start']}-{event['seg_end']}-{event.get('audio_end_sec', 0)}"


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

        if mode == "append_only":
            merged_zh = self.last_translated_text + delta_zh
        elif mode == "small_rewrite_tail":
            # 跨语言字符比例不稳定，用整段拼接（重翻 delta 部分，拼上前缀）
            merged_zh = self.last_translated_text + delta_zh
        else:
            merged_zh = delta_zh

        # 最终 prompt leak 兜底：translator.translate_delta 内部已
        # _strip_prompt_leak，但只剥 delta_zh 开头。我们再检查 merged_zh
        # 整体——如果包含 `_PROMPT_LABEL_LEAKS` 列表里的关键字（"只输出翻译结果"、
        # "请只输出翻译"等），整条 final drop，避免把翻译器自己的 prompt
        # 文本当字幕显示。
        cleaned_zh, leak_hit = _strip_prompt_leak(merged_zh)
        if leak_hit:
            with self.out_lock:
                key = _source_key(event)
                print(
                    f"\r\033[K[warn] drop leak final (delta) @ {key} "
                    f"src={event.get('text', '')[:50]!r}",
                    flush=True,
                )
            self.counts["leak_drops"] = self.counts.get("leak_drops", 0) + 1
            return None
        merged_zh = cleaned_zh

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

        if _is_explanation(zh):
            counts["errors"] += 1
            return self._error_event(event, new_source, f"bad_full_output: {zh[:40]}")

        # 兜底 prompt leak 检测（_do_delta 也加了同样逻辑）。translator
        # 内部已剥 delta_zh 开头，但 merged_zh 整段里仍可能含 prompt
        # 关键字——drop final 避免把翻译器 prompt 当字幕显示。
        cleaned_zh, leak_hit = _strip_prompt_leak(zh)
        if leak_hit:
            with self.out_lock:
                key = _source_key(event)
                print(
                    f"\r\033[K[warn] drop leak final (full) @ {key} "
                    f"src={new_source[:50]!r}",
                    flush=True,
                )
            self.counts["leak_drops"] = self.counts.get("leak_drops", 0) + 1
            return None
        zh = cleaned_zh

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
    """后台翻译线程：partial 模式下异步处理翻译任务。

    优先级队列（translation_priority_enabled=True）：
      - final 优先级 0，全部保留按到达顺序
      - partial 优先级 1，同 source_key 只保留最新 revision（无 revision 则 0）
      - final 入队时立即取消同 key 进行中的 partial SSE
    关闭开关时回退到旧的 batch drain 丢弃逻辑。
    """

    _PRI_FINAL = 0
    _PRI_PARTIAL = 1

    def __init__(self, translator, trans_state, partial_cache,
                 f_trans, f_zh, f_bi, out_lock, counts, is_partial_mode,
                 priority_enabled: bool = True):
        self.translator = translator
        self.trans_state = trans_state
        self.partial_cache = partial_cache
        self.f_trans = f_trans
        self.f_zh = f_zh
        self.f_bi = f_bi
        self.out_lock = out_lock
        self.counts = counts
        self.is_partial_mode = is_partial_mode
        self.priority_enabled = bool(priority_enabled)
        self._seq = itertools.count()
        self._queue = PriorityQueue() if self.priority_enabled else Queue()
        self._auto_promote_timer: threading.Timer | None = None
        self._last_partial_key: str | None = None
        self._last_partial_zh: str | None = None
        self._last_partial_src: str | None = None
        self._last_partial_event: dict | None = None
        # partial 完成后的可复用快照（替代旧 _partial_state 增量拼接）
        self._partial_snapshots: dict[str, dict] = {}
        # 同 key 世代号：更新 revision / final 入队时递增，出队时比对丢弃过期项
        self._partial_generation: dict[str, int] = {}
        self._latest_partial_revision: dict[str, int] = {}
        self._inflight_cancel: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        if self.priority_enabled:
            # PriorityQueue 哨兵：优先级最高，尽快唤醒
            self._queue.put((-1, next(self._seq), None))
        else:
            self._queue.put(None)
        self._thread.join(timeout=120)
        if self._thread.is_alive():
            print("[warn] worker thread did not exit within 120s, closing files anyway", flush=True)

    def _bump_generation(self, key: str) -> int:
        """递增同 key 世代，使队列中更早的 partial 在出队时失效。"""
        gen = self._partial_generation.get(key, 0) + 1
        self._partial_generation[key] = gen
        return gen

    def _request_cancel(self, key: str) -> None:
        """业务目的：final / 更新 revision 立刻打断同 key 进行中的 partial SSE。"""
        with self._cancel_lock:
            ev = self._inflight_cancel.get(key)
            if ev is not None:
                ev.set()

    def _bind_cancel(self, key: str) -> threading.Event:
        """为即将开始的翻译绑定取消事件。"""
        ev = threading.Event()
        with self._cancel_lock:
            old = self._inflight_cancel.get(key)
            if old is not None:
                old.set()
            self._inflight_cancel[key] = ev
        return ev

    def _unbind_cancel(self, key: str, ev: threading.Event) -> None:
        with self._cancel_lock:
            if self._inflight_cancel.get(key) is ev:
                del self._inflight_cancel[key]

    def dispatch_partial(self, event, source_text, key):
        """异步处理 partial 事件（仅翻译 + 显示，不写文件）。"""
        try:
            revision = int(event.get("revision") or 0)
        except (TypeError, ValueError):
            revision = 0
        enqueued_ns = time.monotonic_ns()
        if self.priority_enabled:
            latest = self._latest_partial_revision.get(key, -1)
            if revision < latest:
                # 旧 revision 直接丢，避免占队列
                print(f"[translate] 丢弃过期 partial revision "
                      f"(key={key} rev={revision} < {latest})", flush=True)
                return
            self._latest_partial_revision[key] = revision
            gen = self._bump_generation(key)
            # 新 revision 抢占：取消同 key 进行中的流
            self._request_cancel(key)
            item = {
                "mode": "partial",
                "event": event,
                "source_text": source_text,
                "key": key,
                "revision": revision,
                "generation": gen,
                "translation_enqueued_mono_ns": enqueued_ns,
            }
            self._queue.put((self._PRI_PARTIAL, next(self._seq), item))
        else:
            self._queue.put(("partial", event, source_text, key, enqueued_ns))

    def dispatch_final(self, event, source_text, key):
        """异步处理 final 事件（翻译 + 显示 + 写文件）。"""
        enqueued_ns = time.monotonic_ns()
        if self.priority_enabled:
            # final 入队瞬间取消同 key partial，并作废排队中的 partial
            self._bump_generation(key)
            self._request_cancel(key)
            item = {
                "mode": "final",
                "event": event,
                "source_text": source_text,
                "key": key,
                "revision": int(event.get("revision") or 0) if event.get("revision") is not None else 0,
                "generation": self._partial_generation.get(key, 0),
                "translation_enqueued_mono_ns": enqueued_ns,
            }
            self._queue.put((self._PRI_FINAL, next(self._seq), item))
        else:
            self._queue.put(("final", event, source_text, key, enqueued_ns))

    def _run(self):
        if self.priority_enabled:
            self._run_priority()
        else:
            self._run_legacy()

    def _run_priority(self):
        """单优先级队列：用世代/版本比较替代 batch drain 丢弃。"""
        processed = 0
        dropped = 0
        while True:
            try:
                priority, _seq, item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                break
            mode = item["mode"]
            key = item["key"]
            if mode == "partial":
                # 版本比较：只处理当前世代的最新 partial
                if item.get("generation") != self._partial_generation.get(key):
                    dropped += 1
                    continue
            try:
                if mode == "partial":
                    self._do_partial(
                        item["event"], item["source_text"], key,
                        enqueued_ns=item.get("translation_enqueued_mono_ns"),
                        generation=item.get("generation"),
                    )
                else:
                    out_event = self._do_final(
                        item["event"], item["source_text"], key,
                        enqueued_ns=item.get("translation_enqueued_mono_ns"),
                    )
                    self._write_final_output(out_event, item["event"])
                processed += 1
            except Exception as exc:
                print(f"\n[warn] worker error ({processed} done): {exc}",
                      flush=True)
                import traceback; traceback.print_exc()
        if dropped > 0:
            print(f"[translate] 队列积压,丢弃 {dropped} 条过期 partial",
                  flush=True)
        print(f"[translate] worker exiting after {processed} items", flush=True)

    def _run_legacy(self):
        """关闭优先级开关时：接近旧逻辑的 batch drain 丢弃。"""
        processed = 0
        stopping = False
        while not stopping:
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                break
            batch = [item]
            while True:
                try:
                    nxt = self._queue.get_nowait()
                except Empty:
                    break
                if nxt is None:
                    stopping = True
                    break
                batch.append(nxt)
            finals = [it for it in batch if it[0] == "final"]
            final_keys = {it[3] for it in finals}
            latest_partial: dict[str, tuple] = {}
            for it in batch:
                if it[0] == "partial" and it[3] not in final_keys:
                    latest_partial[it[3]] = it
            dropped = len(batch) - len(finals) - len(latest_partial)
            if dropped > 0:
                print(f"[translate] 队列积压,丢弃 {dropped} 条过期 partial",
                      flush=True)
            for entry in finals + list(latest_partial.values()):
                mode, event, source_text, key = entry[0], entry[1], entry[2], entry[3]
                enqueued_ns = entry[4] if len(entry) > 4 else None
                try:
                    if mode == "partial":
                        self._do_partial(event, source_text, key,
                                         enqueued_ns=enqueued_ns)
                    else:
                        out_event = self._do_final(event, source_text, key,
                                                   enqueued_ns=enqueued_ns)
                        self._write_final_output(out_event, event)
                    processed += 1
                except Exception as exc:
                    print(f"\n[warn] worker error ({processed} done): {exc}",
                          flush=True)
                    import traceback; traceback.print_exc()
        print(f"[translate] worker exiting after {processed} items", flush=True)

    def _save_partial_snapshot(self, key: str, *, source_text: str,
                               translated_text: str, completed: bool,
                               output_valid: bool, request_signature: str,
                               translate_ms: float,
                               finish_reason: str | None = None) -> None:
        """业务目的：保存 partial 完成快照，供 final 在签名一致时复用。"""
        self._partial_snapshots[key] = {
            "source_text": source_text,
            "translated_text": translated_text,
            "completed": completed,
            "output_valid": output_valid,
            "request_signature": request_signature,
            "translate_ms": translate_ms,
            "finish_reason": finish_reason or "stop",
        }

    def _do_partial(self, event, source_text, key, enqueued_ns=None,
                    generation=None):
        """Partial: 当前完整源文 → 完整译文流式；token 事件带累计全文，绝不 commit。"""
        if self.partial_cache.get(key, (None,))[0] == source_text:
            return
        if (self.priority_enabled and generation is not None
                and generation != self._partial_generation.get(key)):
            return

        t = _format_time(event.get("audio_start_sec", 0))
        target_lang = resolve_target_lang(source_text, self.trans_state.target_lang)
        cancel_ev = self._bind_cancel(key)
        started_ns = time.monotonic_ns()
        first_token_ns: list[int | None] = [None]
        sig = ""
        try:
            sig = self.translator.build_request_signature(
                source_text, target_lang=target_lang)
        except Exception:
            sig = ""

        def _on_token_streaming(piece: str, full: str) -> None:
            # 每个 token 带完整累计译文；不写 zh/bi（不 commit）
            if first_token_ns[0] is None:
                first_token_ns[0] = time.monotonic_ns()
            if cancel_ev.is_set():
                raise TranslationCancelled("token callback cancelled")
            cleaned, leak_hit = _strip_prompt_leak(full)
            if leak_hit:
                return
            self.partial_cache[key] = (source_text, cleaned)
            ev = {
                "event_type": "translation_partial",
                "source_key": key,
                "source_text": source_text,
                "translated_full_text": cleaned,
                "is_streaming_token": True,
                "streaming_piece": piece,
                "translation_enqueued_mono_ns": enqueued_ns,
                "translation_request_started_mono_ns": started_ns,
                "translation_first_token_mono_ns": first_token_ns[0],
            }
            with self.out_lock:
                self.f_trans.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self.f_trans.flush()

        try:
            _log_glossary_hits(self.translator, source_text, target_lang)
            # 主路径一律全量流式；不再对 partial 调用 translate_delta_streaming
            full_zh, ms = self.translator.translate_streaming(
                source_text,
                on_token=_on_token_streaming,
                target_lang=target_lang,
                cancel_check=cancel_ev.is_set,
            )
        except TranslationCancelled:
            self.partial_cache.pop(key, None)
            self._save_partial_snapshot(
                key, source_text=source_text, translated_text="",
                completed=False, output_valid=False,
                request_signature=sig, translate_ms=0.0,
            )
            return
        except Exception as exc:
            print(f"\n[partial] 全量翻译失败: {exc}", flush=True)
            self.partial_cache.pop(key, None)
            self._write_failed_partial(key, source_text)
            self._save_partial_snapshot(
                key, source_text=source_text, translated_text="",
                completed=False, output_valid=False,
                request_signature=sig, translate_ms=0.0,
            )
            return
        finally:
            self._unbind_cancel(key, cancel_ev)

        completed_ns = time.monotonic_ns()
        cleaned_zh, leak_hit = _strip_prompt_leak(full_zh)
        bad = _is_explanation(cleaned_zh) or leak_hit or not cleaned_zh.strip()
        # length/incomplete 截断视为无效，禁止 final 复用半截译文
        fr = getattr(self.translator, "last_finish_reason", None) or "stop"
        output_valid = (not bad) and fr not in ("length", "incomplete", "max_tokens")
        self._save_partial_snapshot(
            key,
            source_text=source_text,
            translated_text=cleaned_zh if output_valid else "",
            completed=True,
            output_valid=output_valid,
            request_signature=sig,
            translate_ms=float(ms),
            finish_reason=fr,
        )
        if bad or not output_valid:
            # 业务目的：截断/脏译文不得留在 partial_cache，避免后续跳过重译或 fallback 半截
            self.partial_cache.pop(key, None)
            return

        # 补一条带完成时间戳的 partial（不 commit）
        done_ev = {
            "event_type": "translation_partial",
            "source_key": key,
            "source_text": source_text,
            "translated_full_text": cleaned_zh,
            "is_streaming_token": False,
            "translation_enqueued_mono_ns": enqueued_ns,
            "translation_request_started_mono_ns": started_ns,
            "translation_first_token_mono_ns": first_token_ns[0],
            "translation_completed_mono_ns": completed_ns,
        }
        with self.out_lock:
            self.f_trans.write(json.dumps(done_ev, ensure_ascii=False) + "\n")
            self.f_trans.flush()
            sys.stdout.write(f"\r\033[K[partial][{t}] {cleaned_zh}")
            sys.stdout.flush()

    def _can_reuse_partial(self, key: str, source_text: str,
                           target_lang: str) -> dict | None:
        """业务目的：判断 final 能否复用已完成且签名一致的 partial 快照。"""
        snap = self._partial_snapshots.get(key)
        if not snap:
            return None
        if not snap.get("completed") or not snap.get("output_valid"):
            return None
        if snap.get("source_text") != source_text:
            return None
        zh = (snap.get("translated_text") or "").strip()
        if not zh or _is_explanation(zh):
            return None
        cleaned, leak_hit = _strip_prompt_leak(zh)
        if leak_hit or not cleaned.strip():
            return None
        try:
            cur_sig = self.translator.build_request_signature(
                source_text, target_lang=target_lang)
        except Exception:
            return None
        if snap.get("request_signature") != cur_sig:
            return None
        # length/incomplete 截断译文禁止复用，避免半截进 final
        fr = snap.get("finish_reason") or "stop"
        if fr in ("length", "incomplete", "max_tokens"):
            return None
        return {**snap, "translated_text": cleaned}

    def _do_final(self, event, source_text, key, enqueued_ns=None):
        """Final: 签名+源文一致且 partial 正常完成时复用；否则全量重译。"""
        cached_entry = self.partial_cache.pop(key, None)
        target_lang = resolve_target_lang(source_text, self.trans_state.target_lang)
        started_ns = time.monotonic_ns()
        first_token_ns = None
        completed_ns = None

        reused = self._can_reuse_partial(key, source_text, target_lang)
        self._partial_snapshots.pop(key, None)

        if reused is not None:
            zh = reused["translated_text"]
            partial_ms = float(reused.get("translate_ms") or 0.0)
            completed_ns = time.monotonic_ns()
            self.trans_state.last_source_text = source_text
            self.trans_state.last_translated_text = zh
            self.trans_state.last_source_key = key
            self.counts["translated"] += 1
            return {
                "event_type": "translation_final",
                "source_key": key,
                "source_update_mode": "final_reused_partial",
                "source_text": source_text,
                "delta_source_text": source_text,
                "translated_delta_text": zh,
                "translated_full_text": zh,
                "translate_ms": 0,
                "partial_translate_ms": round(partial_ms, 1),
                "final_reused_partial": True,
                "shared_prefix_len": 0,
                "glossary_hits": [],
                "retried": False,
                "fallback_reason": "",
                "translation_enqueued_mono_ns": enqueued_ns,
                "translation_request_started_mono_ns": started_ns,
                "translation_first_token_mono_ns": started_ns,
                "translation_completed_mono_ns": completed_ns,
                "finish_reason": reused.get("finish_reason") or "stop",
            }

        # 全量重译
        _log_glossary_hits(self.translator, source_text, target_lang)
        fallback_reason = ""
        zh = ""
        translate_ms = 0.0
        cancel_ev = self._bind_cancel(key)

        try:
            zh, translate_ms = self.translator.translate(
                source_text, target_lang=target_lang
            )
            first_token_ns = time.monotonic_ns()  # 非流式：完成即首末 token
        except Exception as exc:
            self.counts["errors"] += 1
            # 失败兜底：若仍有同文 partial 缓存则用之
            if (cached_entry and cached_entry[0] == source_text
                    and cached_entry[1]):
                zh = cached_entry[1]
                fallback_reason = f"full_translate_failed_use_partial: {exc}"
                self.counts["fallbacks"] += 1
                print(
                    f"\r\033[K[warn] full translate failed @ {key}, "
                    f"using partial fallback: {exc}",
                    flush=True,
                )
            else:
                self._unbind_cancel(key, cancel_ev)
                return self._error_event(event, source_text, str(exc))
        finally:
            self._unbind_cancel(key, cancel_ev)

        completed_ns = time.monotonic_ns()
        cleaned_zh, leak_hit = _strip_prompt_leak(zh)
        if leak_hit:
            with self.out_lock:
                print(
                    f"\r\033[K[warn] drop leak final (full) @ {key} "
                    f"src={source_text[:50]!r}",
                    flush=True,
                )
            self.counts["leak_drops"] = self.counts.get("leak_drops", 0) + 1
            return None
        zh = cleaned_zh

        if fallback_reason:
            mode = "full_translate_failed_use_partial"
        else:
            mode = "full_translate"

        self.trans_state.last_source_text = source_text
        self.trans_state.last_translated_text = zh
        self.trans_state.last_source_key = key
        self.counts["translated"] += 1

        return {
            "event_type": "translation_final",
            "source_key": key,
            "source_update_mode": mode,
            "source_text": source_text,
            "delta_source_text": source_text,
            "translated_delta_text": zh,
            "translated_full_text": zh,
            "translate_ms": round(translate_ms, 1),
            "final_reused_partial": False,
            "shared_prefix_len": 0,
            "glossary_hits": [],
            "retried": False,
            "fallback_reason": fallback_reason,
            "translation_enqueued_mono_ns": enqueued_ns,
            "translation_request_started_mono_ns": started_ns,
            "translation_first_token_mono_ns": first_token_ns or completed_ns,
            "translation_completed_mono_ns": completed_ns,
            "finish_reason": getattr(self.translator, "last_finish_reason", None) or "stop",
        }

    def _write_failed_partial(self, key: str, source_text: str) -> None:
        """翻译失败时给 macui 发一个空 partial 事件,清掉 UI 上的旧 draft。
        H10 修:之前失败路径只 print,流式推到一半的 partial 永远留在 UI。
        """
        ev = {
            "event_type": "translation_partial",
            "source_key": key,
            "source_text": source_text,
            "translated_full_text": "",  # 空串 → macui 清掉 draft
            "is_streaming_token": False,
        }
        try:
            with self.out_lock:
                self.f_trans.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self.f_trans.flush()
        except Exception as exc:
            # 写盘失败 (磁盘满 / out_dir 不可写),不要让 _do_partial 抛错
            # 把 already-excepting 路径给覆盖了 — print 一下即可。
            print(f"[partial] 写失败事件失败: {exc}", flush=True)

    def _write_final_output(self, out_event, event):
        """把 _do_final 返回的 out_event 写到所有 JSONL 文件 + stdout。
        抽出公共写盘逻辑,让 _do_final 内部不再关心 IO。
        """
        if out_event is None:
            return

        if out_event.get("event_type") == "translation_error":
            with self.out_lock:
                print(f"\n[warn] {out_event.get('error', '')[:60]}", flush=True)
            return

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

            self.f_trans.write(json.dumps(out_event, ensure_ascii=False) + "\n")
            self.f_trans.flush()
            self.f_zh.write(zh_full + "\n")
            self.f_zh.flush()
            self.f_bi.write(_bilingual_line(event, zh_full))
            self.f_bi.flush()

    def _error_event(self, event, source_text, error_msg):
        return {
            "event_type": "translation_error",
            "source_key": _source_key(event) if event else "",
            "source_seg_start": event.get("seg_start") if event else None,
            "source_seg_end": event.get("seg_end") if event else None,
            "source_text": source_text,
            "error": error_msg,
            "retriable": True,
        }


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
    partial_cache: dict[str, tuple[str, str]] = {}

    worker = None

    def _start_worker():
        """partial 模式的异步翻译线程。translator 就绪时才建 —
        等待模式下重连成功后由主循环补建。"""
        nonlocal worker
        if is_partial_mode and worker is None and translator is not None:
            # 从 lang_config 读优先级队列开关（默认 True）
            priority_enabled = bool(lang_cfg.get("translation_priority_enabled", True))
            worker = TranslateWorker(
                translator, trans_state, partial_cache,
                f_trans, f_zh, f_bi, out_lock, counts, is_partial_mode,
                priority_enabled=priority_enabled,
            )
            worker.start()
            print(f"[translate] 同声传译模式（增量翻译 + 异步线程, "
                  f"priority={priority_enabled}）", flush=True)

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
                if is_partial_mode and etype == "partial":
                    key = _source_key(event)
                    source_text = event.get("text", "").strip()
                    if not source_text:
                        continue
                    if partial_cache.get(key, (None,))[0] == source_text:
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
                    update_info = trans_state.classify(source_text)
                    out_event = trans_state.translate_final(
                        translator, source_text, update_info, event, counts,
                    )
                    if out_event and out_event.get("event_type") == "translation_error":
                        with out_lock:
                            f_trans.write(json.dumps(out_event, ensure_ascii=False) + "\n")
                            f_trans.flush()
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
