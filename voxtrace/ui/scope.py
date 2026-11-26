"""Waveform + log-frequency spectrogram viewer with pitch/breath overlays."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..audio_io import ANALYSIS_SR, AudioFile
from ..notes import BreathRegion
from . import theme
from .timeline import PLOT_LEFT, TimeView

COLORMAP_STOPS = [
    (0.00, (5, 7, 10)),
    (0.10, (9, 15, 25)),
    (0.24, (16, 31, 50)),
    (0.38, (26, 48, 70)),
    (0.51, (60, 69, 88)),
    (0.62, (117, 73, 79)),
    (0.73, (193, 111, 73)),
    (0.83, (239, 161, 74)),
    (0.93, (255, 210, 137)),
    (1.00, (255, 248, 225)),
]


def _build_lut() -> np.ndarray:
    positions = np.array([s[0] for s in COLORMAP_STOPS])
    colors = np.array([s[1] for s in COLORMAP_STOPS], dtype=np.float64)
    x = np.linspace(0, 1, 256)
    lut = np.empty((256, 4), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.clip(np.interp(x, positions, colors[:, channel]), 0, 255).astype(np.uint8)
    lut[:, 3] = 255
    return lut


_CMAP = _build_lut()


class Envelope:
    """High-resolution min/max/RMS envelope cached once per file."""

    def __init__(self, samples: np.ndarray, sr: int, block: int = 16):
        self.sr = sr
        self.block = block
        n = len(samples) // block * block
        if n < block:
            samples = np.pad(samples, (0, block - len(samples)))
            n = block
        data = samples[:n].reshape(-1, block)
        self.mins = data.min(axis=1).astype(np.float32)
        self.maxs = data.max(axis=1).astype(np.float32)
        self.rms = np.sqrt(np.mean(data**2, axis=1) + 1e-12).astype(np.float32)
        self.duration = len(samples) / sr

    def window(self, t0: float, t1: float, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = max(2, count)
        times = np.linspace(max(0.0, t0), min(self.duration, t1), count)
        idx = np.clip((times * self.sr / self.block).astype(np.int64), 0, len(self.mins) - 1)
        return self.mins[idx], self.maxs[idx], self.rms[idx]


class AudioScope(QWidget):
    seekRequested = Signal(float)
    hoverChanged = Signal(str)

    def __init__(self, view: TimeView, parent=None):
        super().__init__(parent)
        self.view = view
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.mode = "vocal"
        self.audio: Optional[AudioFile] = None
        self.envelope: Optional[Envelope] = None
        self.breaths: List[BreathRegion] = []
        self.pitch_times: Optional[np.ndarray] = None
        self.pitch_hz: Optional[np.ndarray] = None
        self.pitch_prob: Optional[np.ndarray] = None
        self._image: Optional[QImage] = None
        self._image_key: Optional[tuple] = None
        self._pitch_path: Optional[QPainterPath] = None
        self._pitch_path_key: Optional[tuple] = None
        self._static_cache: Optional[QPixmap] = None
        self._anchor: Optional[float] = None
        self._drag_start_x: Optional[float] = None
        self._dragging = False
        view.viewChanged.connect(self._invalidate)
        view.viewChanged.connect(self.update)
        view.playheadChanged.connect(lambda _: self.update())
        view.selectionChanged.connect(self._invalidate_selection)

    def set_mode(self, mode: str) -> None:
        if mode not in ("vocal", "bass"):
            mode = "vocal"
        if mode != self.mode:
            self.mode = mode
            self._static_cache = None
            self.update()

    def set_audio(self, audio: Optional[AudioFile]) -> None:
        self.audio = audio
        self.envelope = Envelope(audio.samples, audio.sample_rate) if audio else None
        self._image = None
        self._image_key = None
        self._static_cache = None
        self.update()

    def set_analysis(
        self,
        breaths: Sequence[BreathRegion],
        pitch_times=None,
        pitch_hz=None,
        pitch_prob=None,
    ) -> None:
        self.breaths = list(breaths)
        self.pitch_times = pitch_times
        self.pitch_hz = pitch_hz
        self.pitch_prob = pitch_prob
        self._pitch_path = None
        self._pitch_path_key = None
        self._static_cache = None
        self.update()

    def _invalidate(self) -> None:
        self._image = None
        self._pitch_path = None
        self._pitch_path_key = None
        self._static_cache = None

    def _invalidate_selection(self) -> None:
        # Selection is drawn as a lightweight overlay so dragging stays smooth.
        self.update()

    def resizeEvent(self, event) -> None:
        self._static_cache = None
        super().resizeEvent(event)

    def _layout(self) -> Tuple[int, int, int, int, int]:
        w = max(40, self.width())
        h = max(80, self.height())
        wave_h = int(np.clip((h - 16) * 0.27, 64, 140))
        gap = 1
        spec_top = wave_h + gap
        spec_h = max(40, h - spec_top - 8)
        return PLOT_LEFT, w, wave_h, spec_top, spec_h

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        dpr = max(1.0, self.devicePixelRatioF())
        if (
            self._static_cache is None
            or self._static_cache.width() != int(round(w * dpr))
            or self._static_cache.height() != int(round(h * dpr))
        ):
            self._render_static()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._static_cache is not None:
            p.drawPixmap(0, 0, self._static_cache)
        if self.audio is not None:
            left, _, wave_h, spec_top, spec_h = self._layout()
            self._draw_selection(p, left, w, wave_h, spec_top, spec_h)
            self._draw_playhead(p, left, w, wave_h, spec_top, spec_h)
        p.end()

    def _render_static(self) -> None:
        w, h = self.width(), self.height()
        left, _, wave_h, spec_top, spec_h = self._layout()
        dpr = max(1.0, self.devicePixelRatioF())
        cache = QPixmap(int(round(w * dpr)), int(round(h * dpr)))
        cache.setDevicePixelRatio(dpr)
        cache.fill(QColor(theme.BG))
        p = QPainter(cache)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.audio is None or self.envelope is None:
            self._draw_empty(p, left, w, wave_h, spec_top, spec_h)
        else:
            self._draw_section_backgrounds(p, left, w, wave_h, spec_top, spec_h)
            self._draw_waveform(p, left, w, wave_h)
            self._draw_spectrogram(p, left, w, spec_top, spec_h)
            self._draw_breaths(p, left, w, wave_h, spec_top, spec_h)
            self._draw_labels(p, left, w, wave_h, spec_top, spec_h)
        p.end()
        self._static_cache = cache

    def _draw_empty(self, p, left, w, wave_h, spec_top, spec_h) -> None:
        p.fillRect(QRectF(0, 0, w, self.height()), QColor(theme.BG))
        rect = QRectF(left + 24, 20, w - left - 48, self.height() - 40)
        pen = QPen(QColor(theme.LINE), 1.0, Qt.PenStyle.DashLine)
        pen.setDashPattern([3, 5])
        p.setPen(pen)
        p.setBrush(QColor(14, 17, 21))
        p.drawRoundedRect(rect, 4, 4)
        # Faint spectral decoration
        rng = np.random.default_rng(7)
        for i in range(64):
            x = rect.left() + rect.width() * (i + 0.5) / 64
            h = 6 + rng.random() * rect.height() * 0.32
            color = (
                QColor(79, 157, 255, 9 + int(rng.random() * 15))
                if self.mode == "bass"
                else QColor(240, 162, 74, 9 + int(rng.random() * 15))
            )
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(x, rect.center().y() - h / 2, 2, h), 1, 1)
        p.setPen(QColor(theme.TEXT_SOFT))
        p.setFont(QFont("Microsoft YaHei UI", 13, QFont.Weight.DemiBold))
        bass = self.mode == "bass"
        title = "打开或拖入一段 Bass 音频" if bass else "打开或拖入一段人声音频"
        subtitle = (
            "独立低音工作区 · 识别低音音符、调性与 BPM"
            if bass
            else "WAV · MP3 · FLAC · M4A · OGG     然后将自动识别人声旋律、调性与气声"
        )
        p.drawText(QRectF(rect.left(), rect.center().y() - 16, rect.width(), 30), Qt.AlignmentFlag.AlignCenter, title)
        p.setPen(QColor(theme.FAINT))
        p.setFont(QFont("Microsoft YaHei UI", 9))
        p.drawText(
            QRectF(rect.left(), rect.center().y() + 15, rect.width(), 26),
            Qt.AlignmentFlag.AlignCenter,
            subtitle,
        )

    def _draw_section_backgrounds(self, p, left, w, wave_h, spec_top, spec_h) -> None:
        p.fillRect(QRectF(0, 0, left, self.height()), QColor(theme.PANEL))
        p.fillRect(QRectF(left, 0, w - left, wave_h), QColor("#0f1217"))
        p.fillRect(QRectF(left, spec_top, w - left, spec_h), QColor("#06080b"))
        p.fillRect(QRectF(0, wave_h, w, 1), QColor(theme.LINE_SOFT))
        # Fine horizontal time grid, extremely subtle.
        p.setPen(QPen(QColor(255, 255, 255, 7), 1))
        step = self._grid_step()
        t = math.ceil(self.view.start / step) * step
        while t <= self.view.end:
            x = self.view.x_at(t, w)
            p.drawLine(QPointF(x, 0), QPointF(x, spec_top + spec_h))
            t += step

    def _draw_waveform(self, p, left, w, wave_h) -> None:
        width = max(2, int(w - left))
        mins, maxs, rms = self.envelope.window(self.view.start, self.view.end, width)
        mid = wave_h / 2.0
        amp = (wave_h / 2.0 - 8.0)
        mx = np.clip(maxs, -1, 1)
        mn = np.clip(mins, -1, 1)
        top = mid - mx * amp
        bottom = mid - mn * amp
        xs = np.arange(width, dtype=np.float64) + left

        path = QPainterPath()
        path.moveTo(xs[0], top[0])
        for x, y in zip(xs[1:], top[1:]):
            path.lineTo(x, y)
        for x, y in zip(xs[::-1], bottom[::-1]):
            path.lineTo(x, y)
        path.closeSubpath()
        grad = QLinearGradient(0, 2, 0, wave_h - 2)
        grad.setColorAt(0.0, QColor(104, 222, 209, 100))
        grad.setColorAt(0.5, QColor(64, 188, 178, 150))
        grad.setColorAt(1.0, QColor(104, 222, 209, 100))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawPath(path)

        # RMS core gives the waveform a finished, RX-like depth.
        r = np.clip(rms * 1.45, 0, 1) * amp
        inner = QPainterPath()
        inner.moveTo(xs[0], mid - r[0])
        for x, y in zip(xs[1:], mid - r[1:]):
            inner.lineTo(x, y)
        for x, y in zip(xs[::-1], (mid + r)[::-1]):
            inner.lineTo(x, y)
        inner.closeSubpath()
        p.setBrush(QColor(173, 255, 243, 74))
        p.drawPath(inner)
        p.setPen(QPen(QColor(25, 42, 44, 220), 1))
        p.drawLine(QPointF(left, mid), QPointF(w, mid))

    def _grid_step(self) -> float:
        span = max(1e-4, self.view.span)
        for step in [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120]:
            if span / step <= 12:
                return step
        return 300

    def _draw_spectrogram(self, p, left, w, spec_top, spec_h) -> None:
        plot_w = max(1, int(w - left))
        key = (round(self.view.start, 4), round(self.view.end, 4), plot_w, int(spec_h), id(self.audio))
        if self._image is None or self._image_key != key:
            self._image = self._render_spectrogram(plot_w, int(spec_h))
            self._image_key = key
        if self._image is not None:
            p.drawImage(QRectF(left, spec_top, plot_w, spec_h), self._image)
        # Dark fades at the top/bottom edges.
        grad = QLinearGradient(0, spec_top, 0, spec_top + 34)
        grad.setColorAt(0, QColor(4, 6, 9, 160))
        grad.setColorAt(1, QColor(4, 6, 9, 0))
        p.fillRect(QRectF(left, spec_top, plot_w, 34), grad)
        # Frequency guide lines
        high = self._f_high()
        low = 40.0
        p.setFont(QFont("Bahnschrift", 7))
        for f in (100, 200, 500, 1000, 2000, 4000, 8000):
            if f < low or f > high:
                continue
            y = spec_top + math.log(high / f) / math.log(high / low) * spec_h
            p.setPen(QPen(QColor(255, 255, 255, 14), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(left, y), QPointF(w, y))
            p.setPen(QColor(183, 192, 202, 145))
            p.drawText(QRectF(w - 48, y - 7, 42, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._freq_label(f))
        self._draw_pitch_curve(p, left, w, spec_top, spec_h, high, low)

    def _render_spectrogram(self, width: int, height: int) -> Optional[QImage]:
        if self.audio is None or width <= 2 or height <= 2:
            return None
        audio = self.audio.analysis_samples
        sr = ANALYSIS_SR
        n_fft = 2048
        times = np.linspace(self.view.start, self.view.end, width)
        pad = np.pad(audio, (n_fft // 2 + 1, n_fft // 2 + 1))
        indices = np.round(times * sr).astype(np.int64) + n_fft // 2
        indices = np.clip(indices, 0, max(0, len(pad) - n_fft))
        windows = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[indices]
        windows = windows * np.hanning(n_fft).astype(np.float32)
        spectrum = np.fft.rfft(windows, n=4096, axis=1)
        power = (np.abs(spectrum) ** 2).astype(np.float32) + 1e-12
        freqs = np.fft.rfftfreq(4096, 1.0 / sr)
        high = self._f_high()
        low = 40.0
        row_freqs = np.geomspace(high, low, height)
        bins = np.clip(np.searchsorted(freqs, row_freqs), 0, len(freqs) - 1)
        img = power[:, bins].T
        db = 10.0 * np.log10(img)
        # gentle high-frequency tilt so harmonics stay visible
        db += np.linspace(9.0, 0.0, height)[:, None]
        sample = db[:: max(1, height // 120), :: max(1, width // 160)]
        ceiling = float(np.percentile(sample, 99.85)) + 2.0
        floor = ceiling - 63.0
        norm = np.clip((db - floor) / max(1e-6, ceiling - floor), 0, 1) ** 0.78
        indices_lut = np.clip((norm * 255).astype(np.uint8), 0, 255)
        rgba = _CMAP[indices_lut]
        rgba = np.ascontiguousarray(rgba)
        image = QImage(rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
        return image

    def _draw_pitch_curve(self, p, left, w, spec_top, spec_h, high, low) -> None:
        if self.pitch_times is None or self.pitch_hz is None:
            return
        key = (round(self.view.start, 4), round(self.view.end, 4), int(w), int(spec_top), int(spec_h))
        if self._pitch_path is None or self._pitch_path_key != key:
            times = np.asarray(self.pitch_times)
            hz = np.asarray(self.pitch_hz)
            prob = np.asarray(self.pitch_prob) if self.pitch_prob is not None else np.ones_like(hz)
            mask = (
                np.isfinite(hz)
                & (hz >= low)
                & (hz <= high)
                & (times >= self.view.start)
                & (times <= self.view.end)
                & (prob >= 0.35)
            )
            path = QPainterPath()
            if np.any(mask):
                sel_times = times[mask]
                sel_hz = hz[mask]
                max_points = max(320, int((w - left) * 2))
                if len(sel_times) > max_points:
                    keep = np.linspace(0, len(sel_times) - 1, max_points).astype(np.int64)
                    sel_times = sel_times[keep]
                    sel_hz = sel_hz[keep]
                started = False
                last_t = -1e9
                for t, f in zip(sel_times, sel_hz):
                    if float(t) - last_t > 0.08:
                        started = False
                    x = self.view.x_at(float(t), w)
                    y = spec_top + math.log(high / float(f)) / math.log(high / low) * spec_h
                    if not started:
                        path.moveTo(x, y)
                        started = True
                    else:
                        path.lineTo(x, y)
                    last_t = float(t)
            self._pitch_path = path
            self._pitch_path_key = key
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(89, 216, 200, 40), 4.2))
        p.drawPath(self._pitch_path)
        p.setPen(QPen(QColor(107, 239, 223, 188), 1.15))
        p.drawPath(self._pitch_path)

    def _draw_breaths(self, p, left, w, wave_h, spec_top, spec_h) -> None:
        if not self.breaths:
            return
        for region in self.breaths:
            if region.end < self.view.start or region.start > self.view.end:
                continue
            x0 = self.view.x_at(region.start, w)
            x1 = self.view.x_at(region.end, w)
            if x1 - x0 < 1:
                x1 = x0 + 1
            rect = QRectF(x0, 0, x1 - x0, spec_top + spec_h)
            p.fillRect(rect, QColor(111, 148, 178, 21))
            p.setPen(QPen(QColor(136, 173, 199, 60), 1))
            p.drawLine(QPointF(x0 + 0.5, 0), QPointF(x0 + 0.5, rect.bottom()))
            p.drawLine(QPointF(x1 - 0.5, 0), QPointF(x1 - 0.5, rect.bottom()))
            if x1 - x0 > 46:
                p.setPen(QColor(155, 185, 205, 110))
                p.setFont(QFont("Bahnschrift", 6))
                p.drawText(QRectF(x0 + 4, 4, x1 - x0 - 6, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "BREATH")

    def _draw_selection(self, p, left, w, wave_h, spec_top, spec_h) -> None:
        sel = self.view.selection
        if not sel:
            return
        x0 = self.view.x_at(sel[0], w)
        x1 = self.view.x_at(sel[1], w)
        rect = QRectF(x0, 0, max(1.0, x1 - x0), spec_top + spec_h)
        p.fillRect(rect, QColor(240, 162, 74, 21))
        p.setPen(QPen(QColor(240, 162, 74, 170), 1))
        p.drawLine(QPointF(x0, 0), QPointF(x0, rect.bottom()))
        p.drawLine(QPointF(x1, 0), QPointF(x1, rect.bottom()))

    def _draw_playhead(self, p, left, w, wave_h, spec_top, spec_h) -> None:
        x = self.view.x_at(self.view.playhead, w)
        if x < left - 1 or x > w + 1:
            return
        p.setPen(QPen(QColor(255, 181, 96, 42), 5))
        p.drawLine(QPointF(x, 0), QPointF(x, spec_top + spec_h))
        p.setPen(QPen(QColor(theme.ACCENT_BRIGHT), 1.15))
        p.drawLine(QPointF(x, 0), QPointF(x, spec_top + spec_h))

    def _draw_labels(self, p, left, w, wave_h, spec_top, spec_h) -> None:
        p.setFont(QFont("Bahnschrift", 6))
        p.setPen(QColor(163, 173, 184, 135))
        p.drawText(QRectF(left + 9, 6, 120, 13), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "WAVEFORM")
        p.drawText(QRectF(left + 9, spec_top + 5, 140, 13), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "SPECTROGRAM  ·  LOG")
        # Left gutter: level scale.
        p.setFont(QFont("Bahnschrift", 7))
        p.setPen(QColor(theme.FAINT))
        for frac, label in [(0.12, "0"), (0.5, "-")]:
            y = frac * wave_h
            p.drawText(QRectF(6, y - 7, 36, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
        p.drawText(QRectF(6, wave_h + 6, 42, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "dBFS")
        p.drawText(QRectF(6, spec_top + 6, 42, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "Hz")
        p.setPen(QColor(183, 192, 202, 110))
        p.setFont(QFont("Bahnschrift", 6))
        p.drawText(QRectF(6, spec_top + spec_h - 18, 42, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "40")

    @staticmethod
    def _freq_label(f: int) -> str:
        return f"{f // 1000}k" if f >= 1000 else str(f)

    def _f_high(self) -> float:
        return 11000.0

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.audio is None:
            return
        x = event.position().x()
        if x < PLOT_LEFT:
            return
        t = max(0.0, min(self.view.duration, self.view.time_at(x, self.width())))
        self._anchor = t
        self._drag_start_x = x
        self._dragging = False
        self.view.set_selection(None, None)
        self.view.set_playhead(t)
        self.seekRequested.emit(t)
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if self._anchor is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            if abs(x - (self._drag_start_x or x)) > 3:
                self._dragging = True
            if self._dragging:
                t = max(0.0, min(self.view.duration, self.view.time_at(x, self.width())))
                self.view.set_selection(self._anchor, t)
        elif self.audio is not None and x >= PLOT_LEFT:
            t = self.view.time_at(x, self.width())
            y = event.position().y()
            _, _, wave_h, spec_top, spec_h = self._layout()
            if spec_top <= y <= spec_top + spec_h:
                frac = (y - spec_top) / max(1, spec_h)
                f = self._f_high() * (40.0 / self._f_high()) ** frac
                self.hoverChanged.emit(f"{self._fmt_time(t)}   ·   {f:,.0f} Hz")
            else:
                self.hoverChanged.emit(self._fmt_time(t))

    def mouseReleaseEvent(self, event) -> None:
        self._anchor = None
        self._drag_start_x = None
        self._dragging = False

    def wheelEvent(self, event) -> None:
        if self.audio is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 0.82 if delta > 0 else 1.0 / 0.82
            t = self.view.time_at(event.position().x(), self.width())
            self.view.zoom_at(factor, max(0.0, min(self.view.duration, t)))
        else:
            delta = event.angleDelta().y() or event.angleDelta().x()
            self.view.pan(-delta / 120.0 * self.view.span * 0.16)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if self.audio is not None and event.position().x() >= PLOT_LEFT:
            self.view.fit()

    @staticmethod
    def _fmt_time(t: float) -> str:
        minutes = int(t // 60)
        return f"{minutes:02d}:{t % 60:06.3f}"
