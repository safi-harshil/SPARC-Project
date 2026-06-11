#!/usr/bin/env python3
# notes_ui.py — Qt preview + notes dock (multiline, Shift+Enter, strict key isolation)
# + GUI hotkeys (SPACE pause/resume, ESC stop) when NOT typing

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from PySide6.QtCore import Qt, QTimer, QObject, QEvent, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QTextBrowser, QTextEdit, QPushButton, QSizePolicy
)

# ✅ pipeline control flags
from control_flags import pause_event, stop_event


def format_time_hms(sec: int) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class NoteTimes:
    draft_sec: int
    commit_sec: int


class _NoteKeyFilter(QObject):
    """
    - Enter: send note
    - Shift+Enter: newline
    - ESC: defocus input (do NOT propagate)
    Ensures pipeline hotkeys do NOT trigger while typing.
    """
    send_requested = Signal()
    defocus_requested = Signal()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            mods = event.modifiers()

            if key == Qt.Key_Escape:
                self.defocus_requested.emit()
                return True  # swallow ESC (typing-mode escape)

            if key in (Qt.Key_Return, Qt.Key_Enter):
                if mods & Qt.ShiftModifier:
                    return False  # newline allowed
                self.send_requested.emit()
                return True

            # swallow nothing else explicitly; focus prevents propagation to main window
            return False
        return False


class PreviewNotesWindow(QMainWindow):
    """
    Left: live preview image (your 2x2 grid).
    Right: notes panel with chat history + multiline input.
    UI shows COMMIT time only.
    File saves both DRAFT and COMMIT time (block + ---).

    Global hotkeys (when NOT typing):
      - SPACE: Pause/Resume pipeline
      - ESC  : Stop pipeline
      - q    : Close preview window only (pipeline continues)
    """

    def __init__(
        self,
        title: str,
        get_frame: Callable[[], Optional[np.ndarray]],
        get_timeline_seconds: Callable[[], int],
        notes_out_dir: Path,
        session_tag: Optional[str] = None,
        on_close_request: Optional[Callable[[], None]] = None,
        on_toggle_pause: Optional[Callable[[], None]] = None,
        on_reopen_hint: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.get_frame = get_frame
        self.get_timeline_seconds = get_timeline_seconds
        self.on_close_request = on_close_request
        self.on_toggle_pause = on_toggle_pause
        self.on_reopen_hint = on_reopen_hint

        notes_out_dir = Path(notes_out_dir)
        notes_out_dir.mkdir(parents=True, exist_ok=True)

        tag = session_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.notes_path = notes_out_dir / f"notes_{tag}.txt"
        if not self.notes_path.exists():
            self.notes_path.write_text("", encoding="utf-8")

        # draft latch
        self._draft_time_sec: Optional[int] = None
        self._was_empty = True

        # ---- UI layout ----
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: preview image
        self.preview_label = QLabel("Preview loading…", self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.preview_label, stretch=3)

        # Right: notes panel
        notes_panel = QWidget(self)
        notes_layout = QVBoxLayout(notes_panel)
        notes_layout.setContentsMargins(8, 8, 8, 8)

        self.current_time_lbl = QLabel("Current: 00:00", self)
        notes_layout.addWidget(self.current_time_lbl)

        self.chat = QTextBrowser(self)
        self.chat.setOpenExternalLinks(False)
        self.chat.setReadOnly(True)
        notes_layout.addWidget(self.chat, stretch=1)

        self.input = QTextEdit(self)
        self.input.setAcceptRichText(False)
        self.input.setPlaceholderText("Type note… (Enter=Send, Shift+Enter=Newline, Esc=Exit typing)")
        notes_layout.addWidget(self.input, stretch=0)

        btn_row = QHBoxLayout()
        self.send_btn = QPushButton("Send", self)
        btn_row.addStretch(1)
        btn_row.addWidget(self.send_btn)
        notes_layout.addLayout(btn_row)

        root.addWidget(notes_panel, stretch=1)

        # Key filter to implement Enter/Shift+Enter/Esc in the input
        self._filter = _NoteKeyFilter(self)
        self.input.installEventFilter(self._filter)
        self._filter.send_requested.connect(self.commit_note)
        self._filter.defocus_requested.connect(self.defocus_input)

        self.send_btn.clicked.connect(self.commit_note)
        self.input.textChanged.connect(self._on_text_changed)

        # timer to refresh UI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)  # ~33 fps update

        # make sure main window can receive keys when not typing
        self.setFocusPolicy(Qt.StrongFocus)
        self.preview_label.setFocusPolicy(Qt.ClickFocus)

    # ---------------- note logic ----------------
    def _on_text_changed(self):
        txt = self.input.toPlainText()
        empty = (len(txt.strip()) == 0)

        # latch draft time at first character
        if self._was_empty and not empty and self._draft_time_sec is None:
            self._draft_time_sec = int(self.get_timeline_seconds())

        # reset latch if cleared
        if empty:
            self._draft_time_sec = None

        self._was_empty = empty

    def defocus_input(self):
        self.input.clearFocus()
        self.setFocus(Qt.OtherFocusReason)

    def _append_to_file(self, times: NoteTimes, text: str):
        d = format_time_hms(times.draft_sec)
        c = format_time_hms(times.commit_sec)
        block = (
            f"[DRAFT {d}] [COMMIT {c}] NOTE:\n"
            f"{text.rstrip()}\n"
            f"---\n"
        )
        with self.notes_path.open("a", encoding="utf-8") as f:
            f.write(block)
            f.flush()

    def _append_to_ui(self, commit_sec: int, text: str):
        c = format_time_hms(commit_sec)
        safe = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
        )
        html = (
            f"<div style='margin:8px 0; padding:8px; border:1px solid #444; border-radius:10px;'>"
            f"<div style='font-size:12px; opacity:0.75;'>[{c}]</div>"
            f"<div style='margin-top:4px;'><b>NOTE:</b><br>{safe}</div>"
            f"</div>"
        )
        self.chat.append(html)
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def commit_note(self):
        text = self.input.toPlainText()
        if len(text.strip()) == 0:
            return

        commit_sec = int(self.get_timeline_seconds())
        draft_sec = self._draft_time_sec if self._draft_time_sec is not None else commit_sec

        self._append_to_file(NoteTimes(draft_sec=draft_sec, commit_sec=commit_sec), text)
        self._append_to_ui(commit_sec, text)

        self.input.clear()
        self._draft_time_sec = None
        self._was_empty = True

    # ---------------- UI tick/render ----------------
    def _tick(self):
        t = int(self.get_timeline_seconds())
        self.current_time_lbl.setText(f"Current: {format_time_hms(t)}")

        frame = self.get_frame()
        if frame is None:
            return

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)

        # cv2 uses BGR; Qt expects RGB
        rgb = frame[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    # ---------------- global hotkeys (NOT typing) ----------------
    def keyPressEvent(self, event):
        k = event.key()

        # If typing: keep natural text behavior (SPACE inserts space, etc.)
        # ESC is handled by _NoteKeyFilter (defocus + swallow).
        if self.input.hasFocus():
            super().keyPressEvent(event)
            return

        # SPACE = pause/resume pipeline
        if k == Qt.Key_Space:
            if pause_event.is_set():
                pause_event.clear()
                print("[▶] Resume (space - Notes UI)", flush=True)
            else:
                pause_event.set()
                print("[⏸] Pause (space - Notes UI)", flush=True)
            return

        # ESC = stop pipeline
        if k == Qt.Key_Escape:
            print("[⛔] ESC (Notes UI) → stopping.", flush=True)
            stop_event.set()
            return

        # 'q' closes ONLY the preview window (pipeline continues)
        if k == Qt.Key_Q:
            if self.on_close_request:
                self.on_close_request()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event):
        # closing GUI should behave like "q": close window only (pipeline continues)
        if self.on_close_request:
            self.on_close_request()
            event.ignore()
        else:
            event.accept()


# ──────────────────────────────────────────────────────────────────────────────
# ✅ REQUIRED BY realtime_capture.py
# Returns (app, window).
def start_notes_ui(preview_ref, out_dir: Path, logger=None, title: str = "Preview + Notes"):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    win = PreviewNotesWindow(
        title=title,
        get_frame=preview_ref.get_latest_grid,
        get_timeline_seconds=preview_ref.get_timeline_seconds,  # ✅ this pauses with pause_event
        notes_out_dir=Path(out_dir) / "notes",
        session_tag=None,
        on_close_request=preview_ref.close_window,
    )
    win.show()
    return app, win
