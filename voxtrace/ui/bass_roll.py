"""Read-only bass MIDI piano-roll lane, aligned to the main timeline."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..notes import KeyInfo, is_black_key, midi_to_name
from . import theme
from .timeline import PLOT_LEFT, TimeView


class BassRoll(QWidget):
    seekRequested = Signal(float)
    hoverChanged = Signal(str)

    def __init__(self, view: TimeView, parent=None):
        super().__init__(parent)
        self.view = view
        self.setMinimumHeight(86)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.notes: List = []
        self.key = KeyInfo()
        self._low = 28
        self._high = 52
        self._static: Optional[QPixmap] = None
        view.viewChanged.connect(self._invalidate)
        view.playheadChanged.connect(lambda _: self.update())
        view.selectionChanged.connect(lambda: self.update())

    def set_notes(self, notes: Sequence) -> None:
        self.notes = sorted((n.copy() for n in notes), key=lambda n: (n.start, n.pitch))
        self._update_range()
        self._static = None
        self.update()

    def set_key(self, key: KeyInfo) -> None:
        self.key = key
        self._static = None
        self.update()

    def refresh(self) -> None:
        self._static = None
        self.update()

    def _update_range(self) -> None:
        if self.notes:
            lo = min(n.pitch for n in self.notes)
            hi = max(n.pitch for n in self.notes)
        else:
            lo, hi = 28, 52
        lo -= 2
        hi += 2
        span = max(18, hi - lo + 1)
        span = min(span, 38)
        center = (lo + hi) / 2.0
        low = int(round(center - span / 2.0))
        low = max(21, min(107 - span, low))
        self._low, self._high = low, low + span - 1
        self._static = None

    def _invalidate(self) -> None:
        self._static = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._static = None
        super().resizeEvent(event)

    def _row_h(self) -> float:
        return max(1.0, self.height() / max(1, self._high - self._low + 1))

    def _y(self, pitch: float) -> float:
        return (self._high - pitch) * self._row_h()

    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        dpr = max(1.0, self.devicePixelRatioF())
        if (
            self._static is None
            or self._static.width() != int(round(w * dpr))
            or self._static.height() != int(round(h * dpr))
        ):
            self._render_static(w, h)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._static is not None:
            p.drawPixmap(0, 0, self._static)
        x = self.view.x_at(self.view.playhead, w)
        if PLOT_LEFT <= x <= w:
            p.setPen(QPen(QColor(255, 181, 96, 40), 4))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            p.setPen(QPen(QColor(theme.ACCENT_BRIGHT), 1.0))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        p.end()

    def _render_static(self, w: int, h: int) -> None:
        dpr = max(1.0, self.devicePixelRatioF())
        cache = QPixmap(int(round(w * dpr)), int(round(h * dpr)))
        cache.setDevicePixelRatio(dpr)
        cache.fill(QColor("#0a0e13"))
        p = QPainter(cache)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        row = self._row_h()
        scale = set()
        if self.key is not None:
            intervals = [0, 2, 4, 5, 7, 9, 11] if self.key.mode == "major" else [0, 2, 3, 5, 7, 8, 10]
            scale = {(self.key.tonic + i) % 12 for i in intervals}

        p.fillRect(QRectF(0, 0, PLOT_LEFT, h), QColor("#0b0e12"))
        for pitch in range(self._low, self._high + 1):
            y = self._y(pitch)
            black = is_black_key(pitch)
            p.fillRect(
                QRectF(PLOT_LEFT, y, max(1, w - PLOT_LEFT), math.ceil(row) + 1),
                QColor("#0d1118") if black else QColor("#121822"),
            )
            if pitch % 12 in scale:
                p.fillRect(
                    QRectF(PLOT_LEFT, y, max(1, w - PLOT_LEFT), math.ceil(row) + 1),
                    QColor(80, 160, 230, 8),
                )
            if pitch % 12 == 0:
                p.setPen(QPen(QColor(200, 215, 230, 45), 1))
                p.drawLine(QPointF(PLOT_LEFT, round(y) + 0.5), QPointF(w, round(y) + 0.5))
                p.setPen(QColor(150, 165, 180, 160))
                p.setFont(QFont("Bahnschrift", 6))
                p.drawText(QRectF(6, y, PLOT_LEFT - 12, max(8, row)), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, midi_to_name(pitch))

        # Beat grid, aligned with the main piano roll.
        beat = 60.0 / max(30.0, self.view.bpm)
        if beat / max(1e-9, self.view.span) * max(1, w - PLOT_LEFT) > 60:
            beat /= 2.0
        bar = beat * 4.0
        t = math.floor(self.view.start / beat) * beat
        while t <= self.view.end + beat:
            x = self.view.x_at(t, w)
            is_bar = abs(t / bar - round(t / bar)) < 1e-5
            p.setPen(QPen(QColor(255, 255, 255, 32 if is_bar else 13), 1))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            t += beat

        for note in self.notes:
            x0 = self.view.x_at(note.start, w)
            x1 = self.view.x_at(note.end, w)
            if x1 < PLOT_LEFT or x0 > w:
                continue
            y = self._y(note.pitch)
            rect = QRectF(x0, y + 0.8, max(2.0, x1 - x0), max(2.0, row - 1.6))
            grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
            grad.setColorAt(0, QColor("#8fc5ff"))
            grad.setColorAt(1, QColor("#3b78c9"))
            p.setPen(QPen(QColor(190, 227, 255, 230), 1))
            p.setBrush(grad)
            p.drawRoundedRect(rect, 2.4, 2.4)
            if rect.width() > 34 and rect.height() >= 6:
                p.setPen(QColor(13, 28, 48))
                p.setFont(QFont("Bahnschrift", max(6, min(8, int(rect.height() * 0.55))), QFont.Weight.DemiBold))
                p.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, note.name)
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawLine(QPointF(PLOT_LEFT - 0.5, 0), QPointF(PLOT_LEFT - 0.5, h))
        p.end()
        self._static = cache

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().x() >= PLOT_LEFT:
            t = max(0.0, min(self.view.duration, self.view.time_at(event.position().x(), self.width())))
            self.view.set_playhead(t)
            self.seekRequested.emit(t)

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if x < PLOT_LEFT:
            return
        t = self.view.time_at(x, self.width())
        y = event.position().y()
        pitch = int(round(self._high + 0.5 - y / max(1e-9, self._row_h())))
        self.hoverChanged.emit(f"BASS {midi_to_name(pitch)}   ·   {t:06.3f}s")

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 0.82 if event.angleDelta().y() > 0 else 1.0 / 0.82
            t = self.view.time_at(event.position().x(), self.width())
            self.view.zoom_at(factor, max(0.0, min(self.view.duration, t)))
        else:
            delta = event.angleDelta().y() or event.angleDelta().x()
            self.view.pan(-delta / 120.0 * self.view.span * 0.16)
        event.accept()
