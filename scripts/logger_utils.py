#!/usr/bin/env python3
"""
logger_utils.py — centralized, debounced, thread-safe logging with factories.
"""
import atexit
import time
from pathlib import Path
from typing import Optional, List, Dict
import threading
from datetime import datetime

# ───────────────────────── Internal registry ─────────────────────────
_REGISTRY_LOCK = threading.Lock()
_REGISTRY: List["DebouncedLogger"] = []

def _register_logger(logger: "DebouncedLogger"):
    with _REGISTRY_LOCK:
        _REGISTRY.append(logger)

def _flush_all():
    with _REGISTRY_LOCK:
        for lg in _REGISTRY:
            try:
                lg.periodic_flush(force=True)
            except Exception:
                pass



atexit.register(_flush_all)

# ───────────────────────── Debounced Logger ──────────────────────────
class DebouncedLogger:
    def __init__(self, log_path: Path, flush_interval_sec: int = 5, debug: bool = False, tee: bool = False):
        self.log_path = Path(log_path)
        self._lines: List[str] = []
        self._last_warn: Optional[str] = None
        self._repeat_warn: int = 0
        self._lock = threading.Lock()
        self._flush_interval = max(1, int(flush_interval_sec))
        self._last_flush = time.monotonic()
        self._debug_enabled = bool(debug)
        self._tee = bool(tee)
        _register_logger(self)

    @staticmethod
    def _ts() -> str:
        return datetime.now().replace(microsecond=0).isoformat()

    def _emit(self, line: str):
        self._lines.append(line)
        if self._tee:
            try:
                print(line, flush=False)
            except Exception:
                pass

    def _flush_repeat_warn(self):
        if self._last_warn is not None:
            if self._repeat_warn > 1:
                self._emit(f"{self._ts()} [WARN x{self._repeat_warn}] {self._last_warn}")
            else:
                self._emit(f"{self._ts()} [WARN] {self._last_warn}")
        self._last_warn = None
        self._repeat_warn = 0

    def debug(self, msg: str):
        if not self._debug_enabled:
            return
        with self._lock:
            self._flush_repeat_warn()
            self._emit(f"{self._ts()} [DEBUG] {msg}")

    def info(self, msg: str):
        with self._lock:
            self._flush_repeat_warn()
            self._emit(f"{self._ts()} [INFO] {msg}")

    def warn(self, msg: str):
        with self._lock:
            if msg == self._last_warn:
                self._repeat_warn += 1
            else:
                self._flush_repeat_warn()
                self._last_warn = msg
                self._repeat_warn = 1

    def error(self, msg: str):
        with self._lock:
            self._flush_repeat_warn()
            self._emit(f"{self._ts()} [ERROR] {msg}")

    def periodic_flush(self, force: bool = False):
        now = time.monotonic()
        with self._lock:
            if force or (now - self._last_flush) >= self._flush_interval:
                self._flush_repeat_warn()
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                if self._lines:
                    with open(self.log_path, "a") as f:
                        f.write("\n".join(self._lines) + "\n")
                    self._lines.clear()
                self._last_flush = now

    def close(self):
        self.periodic_flush(force=True)

# ───────────────────────── Logger factories ──────────────────────────
def _mk_logger(path: Path, flush_sec: int, debug: bool, tee: bool) -> DebouncedLogger:
    return DebouncedLogger(path, flush_interval_sec=flush_sec, debug=debug, tee=tee)

def get_capture_logger(cam_dir: Path, *, flush_sec: int = 5, debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(Path(cam_dir) / "logs" / "capture_rt.log", flush_sec, debug, tee)

def get_movement_logger(cam_dir: Path, *, flush_sec: int = 5, debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(Path(cam_dir) / "logs" / "hand_detection_rt.log", flush_sec, debug, tee)

def get_emotion_logger(cam_dir: Path, *, flush_sec: int = 5, debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(Path(cam_dir) / "logs" / "emotion_rt.log", flush_sec, debug, tee)

def get_preview_logger(out_dir: Path, *, flush_sec: int = 5, debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(Path(out_dir) / "logs" / "preview_rt.log", flush_sec, debug, tee)

# keep: per-cam speed-trigger logger
def get_speed_trigger_logger(cam_dir: Path, *, flush_sec: int = 5, debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(Path(cam_dir) / "logs" / "speed_trigger.txt", flush_sec, debug, tee)

# NEW: per-cam object-untouched trigger logger (no extension per spec)
def get_object_trigger_logger(cam_dir: Path, *, flush_sec: int = 5, debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(Path(cam_dir) / "logs" / "object_trigger", flush_sec, debug, tee)


# keep: per-cam speed-trigger logger
def get_speed_trigger_logger(cam_dir: Path, *, flush_sec: int = 5,
                             debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(Path(cam_dir) / "logs" / "speed_trigger.txt",
                      flush_sec, debug, tee)

# NEW: eye tracking logger
def get_eye_logger(cam_dir: Path, *, flush_sec: int = 5,
                   debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(
        Path(cam_dir) / "logs" / "eye_tracking_rt.log",
        flush_sec,
        debug,
        tee
    )

# NEW: per-cam object-untouched trigger logger
def get_object_trigger_logger(cam_dir: Path, *, flush_sec: int = 5,
                              debug: bool = False, tee: bool = False) -> DebouncedLogger:
    return _mk_logger(Path(cam_dir) / "logs" / "object_trigger",
                      flush_sec, debug, tee)

# ───────────────────────── Convenience helpers ───────────────────────
def log_exception(logger: DebouncedLogger, prefix: str, exc: BaseException):
    try:
        logger.error(f"{prefix}: {exc.__class__.__name__}: {exc}")
    except Exception:
        pass

def log_untouched_intervals_text(
    out_dir: Path,
    untouched_out: Dict[str, List[List[int]]],
    filename: str = "untouched_intervals_xyz.txt"
) -> str:
    """
    Write the untouched-intervals text file in the exact format used in the
    offline script, centralized here for reuse.

    Output format:
        📌 Untouched intervals (confirmed):
        red: [a, b], [c, d]
        green: None
        ...

    Returns the absolute path to the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    lines: List[str] = ["📌 Untouched intervals (confirmed):"]
    for obj, spans in untouched_out.items():
        if not spans:
            lines.append(f"{obj}: None")
        else:
            parts = [f"[{a}, {b}]" for a, b in spans]
            lines.append(f"{obj}: {', '.join(parts)}")

    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass

    return str(path)
