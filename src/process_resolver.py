"""Resolve a macOS app Bundle ID to the set of related process PIDs.

Used by application-mode system-audio capture so AudioTee can
``--include-processes`` every process that may emit audio for the selected
app (Chrome main + Helpers, etc.).

Matching rules (in order of preference):

1. Locate the main app via ``lsappinfo`` (bundle id → bundle path + main PID).
2. Include every live process whose executable path is under that ``.app``
   bundle path (covers Chromium Helpers without matching Canary/Edge by name).
3. Fall back to main PID only when path enumeration is unavailable.

This module is intentionally pure-stdlib + subprocess so it stays off the
mlx/ASR import path. Unit tests inject fake process tables.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

# 业务语义：进程路径探测缓冲上限（macOS proc_pidpath）。
_PROC_PIDPATH_MAX = 4096

# 业务语义：常见媒体/浏览器 Bundle ID，应用列表排序时优先展示。
PREFERRED_BUNDLE_IDS: frozenset[str] = frozenset(
    {
        "com.google.Chrome",
        "com.google.Chrome.canary",
        "com.microsoft.edgemac",
        "com.apple.Safari",
        "com.apple.QuickTimePlayerX",
        "org.videolan.vlc",
        "com.spotify.client",
        "com.apple.Music",
        "com.hnc.Discord",
        "com.tinyspeck.slackmacgap",
        "us.zoom.xos",
    }
)


@dataclass(frozen=True)
class ResolvedApplicationProcesses:
    """Bundle ID 解析结果：展示名 + 当前可捕获 PID 集合。"""

    bundle_identifier: str
    display_name: str
    bundle_path: str
    process_identifiers: frozenset[int]


@dataclass(frozen=True)
class RunningApplicationInfo:
    """运行中应用的列表项（供设置页 / 调试用）。"""

    bundle_identifier: str
    display_name: str
    bundle_path: str
    process_identifier: int
    preferred: bool = False


ProcessPathFn = Callable[[int], str | None]
ListPidsFn = Callable[[], Iterable[int]]


def _proc_pidpath(pid: int) -> str | None:
    """调用 macOS libproc.proc_pidpath 取可执行路径；非 macOS 返回 None。"""
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        buf = ctypes.create_string_buffer(_PROC_PIDPATH_MAX)
        n = lib.proc_pidpath(ctypes.c_int(pid), buf, ctypes.c_uint32(_PROC_PIDPATH_MAX))
        if n <= 0:
            return None
        return buf.value.decode("utf-8", "replace")
    except OSError:
        return None


def _list_pids_via_ps() -> list[int]:
    """通过 ``ps -axo pid=`` 枚举本机 PID。"""
    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid="], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def _run_lsappinfo(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["lsappinfo", *args], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def _parse_lsappinfo_kv(blob: str) -> dict[str, str]:
    """Parse ``lsappinfo info`` style ``"key"="value"`` lines."""
    result: dict[str, str] = {}
    for raw in blob.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().strip('"')
        val = val.strip().strip('"')
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        result[key] = val
    return result


def _kv_first(kv: dict[str, str], *keys: str) -> str:
    """按别名顺序取第一个非空值。lsappinfo 实际键名随 macOS 版本变化
    (如 LSDisplayName / LSBundlePath / CFBundleIdentifier),全部列入别名。"""
    for key in keys:
        val = kv.get(key)
        if val:
            return val
    return ""


# lsappinfo info 输出键名别名（不同 macOS 版本/查询方式下观测到的变体）。
_NAME_KEYS = ("LSDisplayName", "display name", "name", "CFBundleName")
_BUNDLE_PATH_KEYS = ("LSBundlePath", "bundle path", "bundlepath", "CFBundlePath",
                     "executable path")
_BUNDLE_ID_KEYS = ("CFBundleIdentifier", "bundleID", "bundleid")
_PID_KEYS = ("pid", "PID")


def _lsappinfo_info(asn: str) -> dict[str, str]:
    """取单个 ASN 的完整 info(不用 -only,避免键名过滤踩版本差异)。"""
    return _parse_lsappinfo_kv(_run_lsappinfo("info", asn))


def find_app_by_bundle_id(bundle_identifier: str) -> tuple[str, str, int] | None:
    """Return ``(display_name, bundle_path, main_pid)`` for a running Bundle ID."""
    if not bundle_identifier:
        return None
    asn_blob = _run_lsappinfo("find", f"bundleid={bundle_identifier}")
    asns = re.findall(r"ASN:[^\s\"]+", asn_blob) or [
        line.strip() for line in asn_blob.splitlines() if line.strip()
    ]
    if not asns:
        return None
    kv = _lsappinfo_info(asns[0])
    bundle_path = _kv_first(kv, *_BUNDLE_PATH_KEYS)
    # executable path 形如 …/Foo.app/Contents/MacOS/Foo,截到 .app 根。
    if bundle_path and ".app/" in bundle_path and not bundle_path.endswith(".app"):
        bundle_path = bundle_path.split(".app/", 1)[0] + ".app"
    name = _kv_first(kv, *_NAME_KEYS) or bundle_identifier
    pid_raw = _kv_first(kv, *_PID_KEYS)
    try:
        pid = int(pid_raw)
    except ValueError:
        pid = 0
    if not bundle_path and pid <= 0:
        return None
    return name, bundle_path, pid


def resolve_bundle_processes(
    bundle_identifier: str,
    *,
    path_for_pid: ProcessPathFn | None = None,
    list_pids: ListPidsFn | None = None,
    bundle_path_hint: str | None = None,
    display_name_hint: str | None = None,
    main_pid_hint: int | None = None,
) -> ResolvedApplicationProcesses | None:
    """Resolve Bundle ID → related PID set (main + helpers under the .app path).

    Injectable ``path_for_pid`` / ``list_pids`` keep unit tests offline.
    """
    if not bundle_identifier:
        return None

    display_name = display_name_hint or bundle_identifier
    bundle_path = (bundle_path_hint or "").rstrip("/")
    main_pid = main_pid_hint or 0

    if not bundle_path or main_pid <= 0:
        found = find_app_by_bundle_id(bundle_identifier)
        if found is None and not bundle_path:
            return None
        if found is not None:
            display_name, bp, mp = found
            if not bundle_path:
                bundle_path = bp.rstrip("/")
            if main_pid <= 0:
                main_pid = mp

    if not bundle_path:
        pids = frozenset({main_pid} if main_pid > 0 else ())
        if not pids:
            return None
        return ResolvedApplicationProcesses(
            bundle_identifier=bundle_identifier,
            display_name=display_name,
            bundle_path="",
            process_identifiers=pids,
        )

    path_fn = path_for_pid or _proc_pidpath
    pid_iter = list_pids or _list_pids_via_ps
    prefix = bundle_path
    prefix_slash = prefix if prefix.endswith("/") else prefix + "/"

    matched: set[int] = set()
    if main_pid > 0:
        matched.add(main_pid)

    for pid in pid_iter():
        try:
            pid_i = int(pid)
        except (TypeError, ValueError):
            continue
        if pid_i <= 0:
            continue
        path = path_fn(pid_i)
        if not path:
            continue
        if path == prefix or path.startswith(prefix_slash):
            matched.add(pid_i)

    if not matched:
        return None

    return ResolvedApplicationProcesses(
        bundle_identifier=bundle_identifier,
        display_name=display_name,
        bundle_path=bundle_path,
        process_identifiers=frozenset(matched),
    )


def list_running_applications(
    *,
    exclude_bundle_ids: Iterable[str] | None = None,
) -> list[RunningApplicationInfo]:
    """List regular running GUI apps via ``lsappinfo`` (best-effort)."""
    exclude = set(exclude_bundle_ids or ())
    blob = _run_lsappinfo("list")
    asns = re.findall(r"ASN:[^\s\"]+", blob)
    apps: list[RunningApplicationInfo] = []
    seen: set[str] = set()
    for asn in asns:
        kv = _lsappinfo_info(asn)
        bid = _kv_first(kv, *_BUNDLE_ID_KEYS)
        if not bid or bid in exclude or bid in seen:
            continue
        bpath = _kv_first(kv, *_BUNDLE_PATH_KEYS)
        if bpath and ".app" not in bpath:
            continue
        try:
            pid = int(_kv_first(kv, *_PID_KEYS) or "0")
        except ValueError:
            pid = 0
        if pid <= 0:
            continue
        name = _kv_first(kv, *_NAME_KEYS) or bid
        seen.add(bid)
        apps.append(
            RunningApplicationInfo(
                bundle_identifier=bid,
                display_name=name,
                bundle_path=bpath,
                process_identifier=pid,
                preferred=bid in PREFERRED_BUNDLE_IDS,
            )
        )

    apps.sort(
        key=lambda a: (
            0 if a.preferred else 1,
            a.display_name.lower(),
            a.bundle_identifier,
        )
    )
    return apps


def format_pid_set(pids: Iterable[int]) -> str:
    return ",".join(str(p) for p in sorted(pids))


if __name__ == "__main__":  # pragma: no cover
    bid = sys.argv[1] if len(sys.argv) > 1 else "com.google.Chrome"
    resolved = resolve_bundle_processes(bid)
    if resolved is None:
        print(f"[ProcessResolver] not running: {bid}")
        sys.exit(1)
    print(
        f"[ProcessResolver] bundle_id={resolved.bundle_identifier} "
        f"name={resolved.display_name} path={resolved.bundle_path} "
        f"pids={format_pid_set(resolved.process_identifiers)} "
        f"count={len(resolved.process_identifiers)}"
    )
