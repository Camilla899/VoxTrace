"""Shared timeline state and time ruler."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme

PLOT_LEFT = 54


class TimeView(QObject):
    viewChanged = Signal()
    playheadChanged = Signal(float)
    selectionChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.duration = 1.0
        self.start = 0.0
        self.end = 1.0
        self.playhead = 0.0
        self.selection: Optional[Tuple[float, float]] = None
        self.bpm = 120.0

    @property
    def span(self) -> float:
        return max(1e-4, self.end - self.start)

    def set_duration(self, duration: float) -> None:
        self.duration = max(0.05, float(duration))
        self.start = 0.0
        self.end = self.duration
        self.playhead = 0.0
        self.selection = None
        self.viewChanged.emit()
        self.playheadChanged.emit(self.playhead)
        self.selectionChanged.emit()

    def set_view(self, start: float, end: float) -> None:
        start = max(0.0, min(float(start), self.duration - 0.03))
        end = max(start + 0.03, min(float(end), self.duration))
        if end - start < 0.03:
            end = min(self.duration, start + 0.03)
        if abs(start - self.start) < 1e-9 and abs(end - self.end) < 1e-9:
            return
        self.start, self.end = start, end
        self.viewChanged.emit()

    def fit(self) -> None:
        self.set_view(0.0, self.duration)

    def zoom_at(self, factor: float, anchor_time: float) -> None:
        if self.duration <= 0:
            return
        new_span = max(0.05, min(self.duration, self.span * factor))
        ratio = (anchor_time - self.start) / self.span
        new_start = anchor_time - new_span * ratio
        new_start = max(0.0, min(new_start, self.duration - new_span))
        self.set_view(new_start, new_start + new_span)

    def pan(self, seconds: float) -> None:
        span = self.span
        new_start = max(0.0, min(self.start + seconds, self.duration - span))
        self.set_view(new_start, new_start + span)

    def set_playhead(self, seconds: float) -> None:
        value = max(0.0, min(float(seconds), self.duration))
        if abs(value - self.playhead) < 1e-6:
            return
        self.playhead = value
        self.playheadChanged.emit(value)

    def set_selection(self, a: Optional[float], b: Optional[float]) -> None:
        if a is None or b is None:
            self.selection = None
        else:
            lo = max(0.0, min(a, b))
            hi = min(self.duration, max(a, b))
            if hi - lo < 0.005:
                self.selection = None
            else:
                self.selection = (lo, hi)
        self.selectionChanged.emit()

    def clear_selection(self) -> None:
        if self.selection is not None:
            self.selection = None
            self.selectionChanged.emit()

    def x_at(self, time_s: float, width: float) -> float:
        return PLOT_LEFT + (time_s - self.start) / self.span * max(0.0, width - PLOT_LEFT)

    def time_at(self, x: float, width: float) -> float:
        usable = max(1.0, width - PLOT_LEFT)
        return self.start + (x - PLOT_LEFT) / usable * self.span

    def in_plot(self, x: float, width: float) -> bool:
        return x >= PLOT_LEFT - 2 and x <= width


class TimeRuler(QWidget):
    seekRequested = Signal(float)

    def __init__(self, view: TimeView, parent=None):
        super().__init__(parent)
        self.view = view
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        view.viewChanged.connect(self.update)
        view.playheadChanged.connect(lambda _: self.update())
        view.selectionChanged.connect(self.update)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _tick_step(self) -> float:
        span = self.view.span
        candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]
        for step in candidates:
            if span / step <= 9:
                return step
        return 600

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()
        p.fillRect(0, 0, w, h, QColor(theme.BG))
        p.fillRect(0, h - 1, w, 1, QColor(theme.LINE_SOFT))
        # Left gutter visually anchors the ruler to the scope's dB strip.
        p.fillRect(0, 0, PLOT_LEFT, h, QColor(theme.PANEL))
        p.setPen(QColor(theme.FAINT))
        p.setFont(QFont("Bahnschrift", 7))
        p.drawText(QRectF(10, 0, PLOT_LEFT - 14, h - 1), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "TIME")

        sel = self.view.selection
        if sel:
            x0 = self.view.x_at(sel[0], w)
            x1 = self.view.x_at(sel[1], w)
            p.fillRect(QRectF(x0, 0, x1 - x0, h - 1), QColor(240, 162, 74, 18))

        step = self._tick_step()
        start = math.floor(self.view.start / step) * step
        t = start
        while t <= self.view.end + step * 0.01:
            if t >= self.view.start - 1e-9:
                x = self.view.x_at(t, w)
                major = abs(t / (step * 5) - round(t / (step * 5))) < 1e-6 or step >= 1
                p.setPen(QPen(QColor(theme.MUTED if major else theme.FAINT), 1))
                p.drawLine(QRectF(x, h - (8 if major else 5), 0, 8 if major else 5).topLeft(), QRectF(x, h - 1, 0, 1).topLeft())
                if major:
                    label = self._format_time(t, step)
                    p.setPen(QColor(theme.TEXT_SOFT))
                    p.setFont(QFont("Bahnschrift", 8))
                    if x + 66 <= w:
                        p.drawText(
                            QRectF(x + 4, 2, 64, h - 8),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            label,
                        )
                    else:
                        p.drawText(
                            QRectF(x - 64, 2, 60, h - 8),
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                            label,
                        )
            t += step

        # Playhead marker.
        x = self.view.x_at(self.view.playhead, w)
        if PLOT_LEFT <= x <= w:
            p.setPen(QPen(QColor(theme.ACCENT), 1.2))
            p.drawLine(QRectF(x, 0, 0, h).topLeft(), QRectF(x, h, 0, 0).topLeft())
            p.setBrush(QColor(theme.ACCENT))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([QPointF(x - 4, 0), QPointF(x + 4, 0), QPointF(x, 5)]))
        p.end()

    @staticmethod
    def _format_time(t: float, step: float) -> str:
        if step < 1:
            return f"{int(t // 60):02d}:{t % 60:06.3f}"
        return f"{int(t // 60):02d}:{int(round(t % 60)):02d}"

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._seek(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek(event.position().x())

    def _seek(self, x: float) -> None:
        if x >= PLOT_LEFT:
            t = self.view.time_at(x, self.width())
            self.view.set_playhead(t)
            self.seekRequested.emit(t)
