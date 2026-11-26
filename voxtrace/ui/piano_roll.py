"""Editable piano-roll with pitch track, key shading and breath lanes."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..notes import BreathRegion, KeyInfo, NoteEvent, is_black_key, midi_to_name
from . import theme
from .timeline import PLOT_LEFT, TimeView


class PianoRoll(QWidget):
    notesEdited = Signal(object, str)
    seekRequested = Signal(float)
    hoverChanged = Signal(str)
    selectionChanged = Signal(object)

    def __init__(self, view: TimeView, parent=None):
        super().__init__(parent)
        self.view = view
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.notes: List[NoteEvent] = []
        self.breaths: List[BreathRegion] = []
        self.pitch_times: Optional[np.ndarray] = None
        self.pitch_hz: Optional[np.ndarray] = None
        self.pitch_prob: Optional[np.ndarray] = None
        self.key = KeyInfo()
        self.mode = "vocal"
        self.snap_enabled = True
        self.snap_beats = 0.25
        self.selected: Optional[int] = None
        self._drag_mode: Optional[str] = None
        self._drag_note: Optional[NoteEvent] = None
        self._press_time = 0.0
        self._press_pitch = 60.0
        self._press_x = 0.0
        self._press_y = 0.0
        self._low = 55
        self._high = 84
        self._min_pitch_span = 12
        self._max_pitch_span = 48
        self._cursor_hover = Qt.CursorShape.ArrowCursor
        self.setToolTip(
            "滚轮：上下滑动音域   ·   Shift+滚轮：左右平移   ·   "
            "Ctrl+滚轮：时间缩放   ·   Ctrl+Shift+滚轮：音域缩放"
        )
        self._did_drag = False
        self._static_cache: Optional[QPixmap] = None
        view.viewChanged.connect(self._on_view_changed)
        view.playheadChanged.connect(lambda _: self.update())
        view.selectionChanged.connect(self.update)

    def invalidate_cache(self) -> None:
        self._static_cache = None

    def _on_view_changed(self) -> None:
        self._static_cache = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._static_cache = None
        super().resizeEvent(event)

    # ------------------------------------------------------------------ data
    def set_notes(
        self,
        notes: Sequence[NoteEvent],
        keep_selection: bool = False,
        fit_range: bool = True,
    ) -> None:
        self.notes = sorted([n.copy() for n in notes], key=lambda n: (n.start, n.pitch))
        if not keep_selection or self.selected is None or self.selected >= len(self.notes):
            self.selected = None
            self.selectionChanged.emit(None)
        if fit_range:
            self._fit_range_to_notes()
        self._static_cache = None
        self.update()

    def set_analysis(self, breaths, pitch_times=None, pitch_hz=None, pitch_prob=None, key: Optional[KeyInfo] = None) -> None:
        self.breaths = list(breaths)
        self.pitch_times = pitch_times
        self.pitch_hz = pitch_hz
        self.pitch_prob = pitch_prob
        if key is not None:
            self.key = key
        self._static_cache = None
        self.update()

    def set_key(self, key: KeyInfo) -> None:
        self.key = key
        self._static_cache = None
        self.update()

    def set_snap(self, enabled: bool, beats: float) -> None:
        self.snap_enabled = enabled
        self.snap_beats = beats
        self._static_cache = None
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in ("vocal", "bass"):
            mode = "vocal"
        self.mode = mode
        self._static_cache = None
        self.update()

    def _beat(self) -> float:
        return 60.0 / max(30.0, self.view.bpm)

    def _grid(self) -> float:
        return max(0.01, self._beat() * self.snap_beats) if self.snap_enabled else 0.001

    def fit_pitch_to_notes(self) -> None:
        """Recover the automatic vertical range after manual scrolling/zoom."""
        self._fit_range_to_notes()
        self.update()

    def _fit_range_to_notes(self) -> None:
        if self.notes:
            lo = min(n.pitch for n in self.notes)
            hi = max(n.pitch for n in self.notes)
        else:
            lo, hi = 55, 79
        lo -= 2
        hi += 2
        span = max(24, min(self._max_pitch_span, hi - lo + 1))
        center = (lo + hi) / 2.0
        low = int(round(center - span / 2.0))
        self._set_pitch_window(low, span)

    def _set_pitch_window(self, low: int, span: int) -> None:
        span = max(self._min_pitch_span, min(self._max_pitch_span, int(span)))
        low = max(21, min(108 - span + 1, int(low)))
        self._low = low
        self._high = low + span - 1
        self._static_cache = None

    def _pitch_span(self) -> int:
        return self._high - self._low + 1

    def scroll_pitch(self, semitones: int) -> None:
        """Move the visible pitch window up/down without changing zoom."""
        if not semitones:
            return
        self._set_pitch_window(self._low + int(semitones), self._pitch_span())
        self.update()

    def _zoom_pitch(self, factor: float, anchor_y: float) -> None:
        old_span = self._pitch_span()
        new_span = max(
            self._min_pitch_span,
            min(self._max_pitch_span, int(round(old_span * factor))),
        )
        if new_span == old_span:
            return
        anchor = self._pitch_at(anchor_y)
        fraction_from_bottom = (anchor - self._low) / max(1, old_span)
        new_low = int(round(anchor - fraction_from_bottom * new_span))
        self._set_pitch_window(new_low, new_span)
        self.update()

    def _ensure_pitch_visible(self, pitch: int, margin: int = 2) -> None:
        if pitch < self._low + margin:
            self.scroll_pitch(pitch - margin - self._low)
        elif pitch > self._high - margin:
            self.scroll_pitch(pitch + margin - self._high)

    # ------------------------------------------------------------------ geometry
    def _rows_height(self) -> float:
        return max(20.0, self.height() - 1.0)

    def _row_h(self) -> float:
        return self._rows_height() / (self._high - self._low + 1)

    def _y_for_pitch(self, pitch: float) -> float:
        """Top edge of the semitone row (integer pitch)."""
        row = self._row_h()
        return (self._high - pitch) * row

    def _y_center_for_pitch(self, pitch: float) -> float:
        return (self._high - pitch + 0.5) * self._row_h()

    def _pitch_at(self, y: float) -> float:
        row = self._row_h()
        return self._high + 0.5 - y / max(1e-9, row)

    def _rect_for_note(self, note: NoteEvent) -> QRectF:
        w = self.width()
        x0 = self.view.x_at(note.start, w)
        x1 = self.view.x_at(note.end, w)
        row = self._row_h()
        y = self._y_for_pitch(note.pitch)
        return QRectF(x0, y + 1.1, max(2.0, x1 - x0), max(2.0, row - 2.2))

    def _hit_note(self, x: float, y: float) -> Optional[int]:
        for idx in range(len(self.notes) - 1, -1, -1):
            rect = self._rect_for_note(self.notes[idx])
            if rect.adjusted(-1, -1, 1, 1).contains(QPointF(x, y)):
                return idx
        return None

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        dpr = max(1.0, self.devicePixelRatioF())
        if (
            self._static_cache is None
            or self._static_cache.width() != int(round(w * dpr))
            or self._static_cache.height() != int(round(h * dpr))
        ):
            self._render_static(w, h)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._static_cache is not None:
            p.drawPixmap(0, 0, self._static_cache)
        self._draw_playhead(p, w, h)
        p.end()

    def _render_static(self, w: int, h: int) -> None:
        dpr = max(1.0, self.devicePixelRatioF())
        cache = QPixmap(int(round(w * dpr)), int(round(h * dpr)))
        cache.setDevicePixelRatio(dpr)
        cache.fill(QColor("#0b0e12"))
        p = QPainter(cache)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(QRectF(0, 0, PLOT_LEFT, h), QColor("#0a0c10"))
        self._draw_rows(p, w, h)
        self._draw_grid(p, w, h)
        self._draw_breaths(p, w, h)
        self._draw_pitch(p, w, h)
        self._draw_notes(p)
        self._draw_keyboard(p, h)
        self._draw_legend(p, w)
        p.end()
        self._static_cache = cache

    def _draw_rows(self, p, w, h) -> None:
        row = self._row_h()
        scale = set()
        if self.key is not None:
            intervals = [0, 2, 4, 5, 7, 9, 11] if self.key.mode == "major" else [0, 2, 3, 5, 7, 8, 10]
            scale = {(self.key.tonic + i) % 12 for i in intervals}
        for pitch in range(self._low, self._high + 1):
            y = self._y_for_pitch(pitch)
            black = is_black_key(pitch)
            base = QColor("#0d1015") if black else QColor("#11151a")
            p.fillRect(QRectF(PLOT_LEFT, y, w - PLOT_LEFT, math.ceil(row) + 1), base)
            if pitch % 12 in scale:
                p.fillRect(QRectF(PLOT_LEFT, y, w - PLOT_LEFT, math.ceil(row) + 1), QColor(89, 216, 200, 6))
            if pitch % 12 == 0:
                p.setPen(QPen(QColor(206, 215, 224, 54), 1))
                p.drawLine(QPointF(PLOT_LEFT, round(y) + 0.5), QPointF(w, round(y) + 0.5))
            elif row >= 9:
                p.setPen(QPen(QColor(255, 255, 255, 8), 1))
                p.drawLine(QPointF(PLOT_LEFT, y + 0.5), QPointF(w, y + 0.5))
        p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
        p.drawLine(QPointF(PLOT_LEFT, 0), QPointF(PLOT_LEFT, h))

    def _draw_grid(self, p, w, h) -> None:
        beat = self._beat()
        span = self.view.span
        # Choose a musically sensible grid density.
        while beat * 4 / max(1e-9, span) * max(1, w - PLOT_LEFT) < 8:
            beat *= 4
        if beat / max(1e-9, span) * (w - PLOT_LEFT) > 60:
            beat = beat / 2
        start = math.floor(self.view.start / beat) * beat
        bar = beat * 4
        t = start
        while t <= self.view.end + beat:
            x = self.view.x_at(t, w)
            is_bar = abs((t / bar) - round(t / bar)) < 1e-5
            if is_bar:
                p.setPen(QPen(QColor(255, 255, 255, 30), 1))
            else:
                p.setPen(QPen(QColor(255, 255, 255, 12), 1))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            t += beat
        if self.snap_enabled:
            subdivision = max(0.01, beat * 0.25)
            if subdivision / max(1e-9, self.view.span) * (w - PLOT_LEFT) > 10:
                t = math.floor(self.view.start / subdivision) * subdivision
                p.setPen(QPen(QColor(255, 255, 255, 6), 1))
                while t <= self.view.end:
                    p.drawLine(QPointF(self.view.x_at(t, w), 0), QPointF(self.view.x_at(t, w), h))
                    t += subdivision

    def _draw_breaths(self, p, w, h) -> None:
        for region in self.breaths:
            x0 = self.view.x_at(region.start, w)
            x1 = self.view.x_at(region.end, w)
            if x1 < PLOT_LEFT or x0 > w:
                continue
            x0 = max(x0, PLOT_LEFT)
            x1 = min(x1, w)
            if x1 - x0 < 1:
                continue
            p.fillRect(QRectF(x0, 0, x1 - x0, h), QColor(117, 151, 177, 18))
            p.setPen(QPen(QColor(150, 184, 209, 50), 1))
            p.drawLine(QPointF(x0, 0), QPointF(x0, h))
            if x1 - x0 > 46:
                p.setPen(QColor(151, 181, 201, 108))
                p.setFont(QFont("Microsoft YaHei UI", 7))
                p.drawText(QRectF(x0 + 4, h - 15, x1 - x0 - 4, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "气声")

    def _draw_pitch(self, p, w, h) -> None:
        if self.pitch_times is None or self.pitch_hz is None:
            return
        times = np.asarray(self.pitch_times)
        hz = np.asarray(self.pitch_hz)
        prob = np.asarray(self.pitch_prob) if self.pitch_prob is not None else np.ones_like(hz)
        mask = (
            np.isfinite(hz)
            & (times >= self.view.start)
            & (times <= self.view.end)
            & (prob >= 0.30)
        )
        if not np.any(mask):
            return
        times = times[mask]
        midi = 69.0 + 12.0 * np.log2(np.maximum(hz[mask], 1e-9) / 440.0)
        max_points = max(240, int((w - PLOT_LEFT) * 2))
        if len(times) > max_points:
            keep = np.linspace(0, len(times) - 1, max_points).astype(np.int64)
            times = times[keep]
            midi = midi[keep]
        path = QPainterPath()
        started = False
        last_t = -10.0
        for t, note in zip(times, midi):
            if t - last_t > 0.08:
                started = False
            x = self.view.x_at(float(t), w)
            y = self._y_center_for_pitch(float(note))
            if not started:
                path.moveTo(x, y)
                started = True
            else:
                path.lineTo(x, y)
            last_t = float(t)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(89, 216, 200, 30), 4.0))
        p.drawPath(path)
        p.setPen(QPen(QColor(94, 232, 216, 200), 1.05))
        p.drawPath(path)

    def _draw_notes(self, p) -> None:
        w = self.width()
        for idx, note in enumerate(self.notes):
            rect = self._rect_for_note(note)
            if rect.right() < PLOT_LEFT or rect.left() > w:
                continue
            selected = idx == self.selected
            rect = rect.adjusted(0.4, 0.2, -0.2, -0.2)
            if selected:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(89, 216, 200, 40))
                p.drawRoundedRect(rect.adjusted(-1.8, -1.8, 1.8, 1.8), 4.5, 4.5)
            bass = self.mode == "bass"
            top = (QColor("#9fcbff") if bass else QColor("#ffd18c")) if not selected else QColor("#b9fff5")
            grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
            if selected:
                grad.setColorAt(0, QColor("#76e4d6"))
                grad.setColorAt(1, QColor("#2d938a"))
            elif bass:
                grad.setColorAt(0, QColor("#93c9ff"))
                grad.setColorAt(1, QColor("#3773bd"))
            else:
                grad.setColorAt(0, top)
                grad.setColorAt(1, QColor("#b9762f"))
            border = QColor("#fff0c9") if selected else (QColor(175, 216, 255, 230) if bass else QColor(255, 214, 154, 220))
            p.setPen(QPen(border, 1.0))
            p.setBrush(grad)
            p.drawRoundedRect(rect, 3.2, 3.2)
            if rect.height() >= 5:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, 40))
                p.drawRoundedRect(QRectF(rect.left() + 3, rect.top() + 1.2, max(0.0, rect.width() - 6), 1.0), 0.5, 0.5)
            if selected:
                p.setPen(QPen(QColor(230, 255, 252, 220), 1.3))
                p.drawLine(QPointF(rect.right() - 2.5, rect.top() + 3), QPointF(rect.right() - 2.5, rect.bottom() - 3))
            label = midi_to_name(note.pitch)
            if rect.width() > 27 and rect.height() >= 8:
                p.setPen(
                    QColor(7, 40, 37) if selected
                    else (QColor(10, 28, 48) if bass else QColor(35, 23, 10))
                )
                p.setFont(QFont("Bahnschrift", max(6, min(8, int(rect.height() * 0.55))), QFont.Weight.DemiBold))
                p.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)

    def _draw_keyboard(self, p, h) -> None:
        row = self._row_h()
        p.fillRect(QRectF(0, 0, PLOT_LEFT, h), QColor("#101318"))
        for pitch in range(self._low, self._high + 1):
            y = self._y_for_pitch(pitch)
            black = is_black_key(pitch)
            key_w = PLOT_LEFT - 16 if black else PLOT_LEFT - 1
            rect = QRectF(0, y + 0.3, key_w, max(1.0, row - 0.6))
            grad = QLinearGradient(rect.left(), 0, rect.right(), 0)
            if black:
                grad.setColorAt(0, QColor("#1a1d22"))
                grad.setColorAt(1, QColor("#0a0c0f"))
            else:
                grad.setColorAt(0, QColor("#c6c9cc"))
                grad.setColorAt(1, QColor("#8c9298"))
            p.setPen(QPen(QColor("#07090b"), 0.8))
            p.setBrush(grad)
            p.drawRect(rect)
            if pitch % 12 in {0, 2, 4, 5, 7, 9, 11}:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, 18))
                p.drawRect(QRectF(1, y + 1, 3, max(1.0, row - 2)))
        # C labels
        p.setFont(QFont("Bahnschrift", 6))
        for pitch in range(self._low, self._high + 1):
            if pitch % 12 != 0:
                continue
            y = self._y_for_pitch(pitch)
            p.setPen(QColor(34, 38, 43))
            p.drawText(QRectF(6, y + 0.5, PLOT_LEFT - 20, row), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, midi_to_name(pitch))
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawLine(QPointF(PLOT_LEFT - 0.5, 0), QPointF(PLOT_LEFT - 0.5, h))
        # Octave shading in scale.
        p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
        p.drawLine(QPointF(0, 0), QPointF(0, h))

    def _draw_playhead(self, p, w, h) -> None:
        x = self.view.x_at(self.view.playhead, w)
        if PLOT_LEFT <= x <= w:
            p.setPen(QPen(QColor(255, 181, 96, 40), 4))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            p.setPen(QPen(QColor(theme.ACCENT_BRIGHT), 1.0))
            p.drawLine(QPointF(x, 0), QPointF(x, h))

    def _draw_legend(self, p, w) -> None:
        # Tiny legend in the lower right; low contrast so it never fights notes.
        p.setFont(QFont("Microsoft YaHei UI", 7))
        note_color = QColor("#93c9ff") if self.mode == "bass" else QColor("#f0ab55")
        labels = [("音高曲线", QColor(94, 232, 216)), ("BASS 音符" if self.mode == "bass" else "AI 音符", note_color), ("气声" if self.mode != "bass" else "噪声", QColor(117, 151, 177))]
        x = w - 188
        y = self.height() - 14
        if x < PLOT_LEFT:
            return
        for text, color in labels:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(QRectF(x, y + 4, 4, 4))
            p.setPen(QColor(145, 154, 164, 140))
            p.drawText(QRectF(x + 7, y, 52, 12), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            x += 62

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, event) -> None:
        x = event.position().x()
        y = event.position().y()
        if event.button() == Qt.MouseButton.RightButton:
            hit = self._hit_note(x, y)
            if hit is not None:
                note = self.notes.pop(hit)
                self.selected = None
                self.selectionChanged.emit(None)
                self._static_cache = None
                self.update()
                self.notesEdited.emit([n.copy() for n in self.notes], f"删除音符 {note.name}")
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        hit = self._hit_note(x, y)
        if hit is None:
            self.selected = None
            self.selectionChanged.emit(None)
            self._drag_mode = None
            self.update()
            if x >= PLOT_LEFT:
                t = max(0.0, min(self.view.duration, self.view.time_at(x, self.width())))
                self.view.set_playhead(t)
                self.seekRequested.emit(t)
            return
        self.selected = hit
        self.selectionChanged.emit(self.notes[hit])
        note = self.notes[hit]
        rect = self._rect_for_note(note)
        self._press_time = self.view.time_at(x, self.width())
        self._press_pitch = self._pitch_at(y)
        self._press_x = x
        self._press_y = y
        self._drag_note = note.copy()
        self._did_drag = False
        if abs(x - rect.right()) <= 6:
            self._drag_mode = "resize_r"
        elif abs(x - rect.left()) <= 5:
            self._drag_mode = "resize_l"
        else:
            self._drag_mode = "move"
        self._static_cache = None
        self.update()

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        y = event.position().y()
        if self._drag_mode and self._drag_note and (event.buttons() & Qt.MouseButton.LeftButton):
            self._did_drag = True
            self._apply_drag(x, y, event.modifiers())
            self._static_cache = None
            self.update()
            return
        # Hover cursors and readout.
        if x < PLOT_LEFT:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        hit = self._hit_note(x, y)
        if hit is None:
            self.setCursor(Qt.CursorShape.CrossCursor)
            t = max(0.0, min(self.view.duration, self.view.time_at(x, self.width())))
            self.hoverChanged.emit(f"{midi_to_name(int(round(self._pitch_at(y))))}   ·   {t:06.3f}s")
            return
        note = self.notes[hit]
        rect = self._rect_for_note(note)
        if abs(x - rect.right()) <= 6 or abs(x - rect.left()) <= 5:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.hoverChanged.emit(f"{note.name}   ·   {note.start:06.3f} – {note.end:06.3f}s   ·   Vel {note.velocity}")

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_mode and self._drag_note and self._did_drag:
            mode = {"move": "移动", "resize_r": "调整长度", "resize_l": "调整起点"}.get(self._drag_mode, "编辑")
            self.notesEdited.emit([n.copy() for n in self.notes], f"{mode} {self._drag_note.name}")
        self._drag_mode = None
        self._drag_note = None
        self._did_drag = False

    def _apply_drag(self, x: float, y: float, modifiers) -> None:
        if self._drag_note is None or self.selected is None:
            return
        live = self.notes[self.selected]
        grid = self._grid()
        snap = self.snap_enabled and not (modifiers & Qt.KeyboardModifier.ShiftModifier)
        t = self.view.time_at(x, self.width())
        p = self._pitch_at(y)
        dt = t - self._press_time
        dp = p - self._press_pitch
        if self._drag_mode == "move":
            new_start = self._drag_note.start + dt
            if snap:
                new_start = round(new_start / grid) * grid
            length = self._drag_note.duration
            new_start = max(0.0, min(new_start, self.view.duration - length))
            live.start = new_start
            live.end = new_start + length
            live.pitch = int(np.clip(round(self._drag_note.pitch + dp), 21, 108))
        elif self._drag_mode == "resize_r":
            new_end = self._drag_note.end + dt
            if snap:
                new_end = round(new_end / grid) * grid
            new_end = max(self._drag_note.start + grid, min(new_end, self.view.duration))
            live.end = new_end
        elif self._drag_mode == "resize_l":
            new_start = self._drag_note.start + dt
            if snap:
                new_start = round(new_start / grid) * grid
            new_start = max(0.0, min(new_start, self._drag_note.end - grid))
            live.start = new_start

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._hit_note(event.position().x(), event.position().y()) is not None:
            return
        if event.position().x() < PLOT_LEFT:
            return
        grid = self._grid()
        beat = self._beat()
        t = self.view.time_at(event.position().x(), self.width())
        if self.snap_enabled:
            t = round(t / grid) * grid
        t = max(0.0, min(t, self.view.duration - grid))
        pitch = int(np.clip(round(self._pitch_at(event.position().y())), 21, 108))
        length = max(beat, grid * 2)
        if t + length > self.view.duration:
            length = max(grid, self.view.duration - t)
        note = NoteEvent(t, t + length, pitch, velocity=100, confidence=0.6, source="manual")
        self.notes.append(note)
        self.notes.sort(key=lambda n: (n.start, n.pitch))
        self.selected = next((i for i, n in enumerate(self.notes) if n is note), None)
        self.selectionChanged.emit(note)
        self._ensure_pitch_visible(note.pitch)
        self._static_cache = None
        self.update()
        self.notesEdited.emit([n.copy() for n in self.notes], f"添加音符 {note.name}")

    def wheelEvent(self, event) -> None:
        mods = event.modifiers()
        delta = event.angleDelta()
        dy = delta.y()
        dx = delta.x()
        if mods & Qt.KeyboardModifier.ControlModifier:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                factor = 0.84 if dy > 0 else 1.0 / 0.84
                self._zoom_pitch(factor, event.position().y())
                event.accept()
                return
            factor = 0.82 if dy > 0 else 1.0 / 0.82
            t = self.view.time_at(event.position().x(), self.width())
            self.view.zoom_at(factor, max(0.0, min(self.view.duration, t)))
        elif mods & Qt.KeyboardModifier.ShiftModifier or (abs(dx) > abs(dy) and dx):
            amount = dx if dx else dy
            self.view.pan(-amount / 120.0 * self.view.span * 0.16)
        else:
            # Normal wheel: move the pitch window up/down. Two semitones per
            # notch keeps the piano roll responsive without feeling jumpy.
            steps = dy / 120.0
            if abs(steps) < 0.05 and event.pixelDelta().y():
                steps = event.pixelDelta().y() / 40.0
            amount = int(round(steps * 2))
            if amount:
                self.scroll_pitch(amount)
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_PageUp:
            self.scroll_pitch(12)
            return
        if event.key() == Qt.Key.Key_PageDown:
            self.scroll_pitch(-12)
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.selected is not None:
            note = self.notes.pop(self.selected)
            self.selected = None
            self.selectionChanged.emit(None)
            self._static_cache = None
            self.update()
            self.notesEdited.emit([n.copy() for n in self.notes], f"删除音符 {note.name}")
            return
        if event.key() == Qt.Key.Key_Escape:
            self.selected = None
            self.selectionChanged.emit(None)
            self._static_cache = None
            self.update()
            return
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right) and self.selected is not None:
            direction = -1 if event.key() == Qt.Key.Key_Left else 1
            note = self.notes[self.selected]
            step = self._grid() if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else self._beat()
            note.start = max(0.0, min(note.start + direction * step, self.view.duration - note.duration))
            note.end = note.start + note.duration
            self._static_cache = None
            self.update()
            self.notesEdited.emit([n.copy() for n in self.notes], "微调音符位置")
            return
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and self.selected is not None:
            note = self.notes[self.selected]
            note.pitch = int(np.clip(note.pitch + (1 if event.key() == Qt.Key.Key_Up else -1), 21, 108))
            self._ensure_pitch_visible(note.pitch)
            self._static_cache = None
            self.update()
            self.notesEdited.emit([n.copy() for n in self.notes], "移调音符")
            return
        super().keyPressEvent(event)
