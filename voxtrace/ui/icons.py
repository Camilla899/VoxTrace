"""Small hand-drawn vector icon set, so the app has no external UI assets."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from .theme import ACCENT, TEXT


def _pen(painter: QPainter, color: str, width: float = 1.6) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    return pen


def draw_icon(painter: QPainter, name: str, rect: QRectF, color: str = TEXT) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    w = rect.width()
    h = rect.height()
    cx = rect.center().x()
    cy = rect.center().y()
    _pen(painter, color, max(1.25, w / 13.0))

    if name == "folder":
        path = QPainterPath()
        path.moveTo(rect.left() + w * 0.08, rect.top() + h * 0.28)
        path.lineTo(rect.left() + w * 0.40, rect.top() + h * 0.28)
        path.lineTo(rect.left() + w * 0.48, rect.top() + h * 0.40)
        path.lineTo(rect.right() - w * 0.08, rect.top() + h * 0.40)
        path.lineTo(rect.right() - w * 0.15, rect.bottom() - h * 0.15)
        path.lineTo(rect.left() + w * 0.10, rect.bottom() - h * 0.15)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "play":
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        path = QPainterPath()
        path.moveTo(cx - w * 0.20, cy - h * 0.27)
        path.lineTo(cx + w * 0.29, cy)
        path.lineTo(cx - w * 0.20, cy + h * 0.27)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "pause":
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        bw = w * 0.11
        painter.drawRoundedRect(QRectF(cx - w * 0.22, cy - h * 0.26, bw, h * 0.52), 1, 1)
        painter.drawRoundedRect(QRectF(cx + w * 0.11, cy - h * 0.26, bw, h * 0.52), 1, 1)
    elif name == "stop":
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(cx - w * 0.20, cy - h * 0.20, w * 0.40, h * 0.40), 1.5, 1.5)
    elif name == "skip_back":
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(cx - w * 0.28, cy - h * 0.25, w * 0.07, h * 0.5))
        path = QPainterPath()
        path.moveTo(cx + w * 0.25, cy - h * 0.25)
        path.lineTo(cx - w * 0.14, cy)
        path.lineTo(cx + w * 0.25, cy + h * 0.25)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "skip_forward":
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(cx + w * 0.21, cy - h * 0.25, w * 0.07, h * 0.5))
        path = QPainterPath()
        path.moveTo(cx - w * 0.25, cy - h * 0.25)
        path.lineTo(cx + w * 0.14, cy)
        path.lineTo(cx - w * 0.25, cy + h * 0.25)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "spark":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        path = QPainterPath()
        pts = []
        for i in range(8):
            angle = -math.pi / 2 + i * math.pi / 4
            radius = w * (0.44 if i % 2 == 0 else 0.13)
            pts.append(QPointF(cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path.closeSubpath()
        painter.drawPath(path)
        painter.setPen(_pen(painter, color, max(1.0, w / 20)))
        painter.drawEllipse(QPointF(cx + w * 0.30, cy - h * 0.30), w * 0.045, w * 0.045)
    elif name == "wave":
        path = QPainterPath()
        for i in range(19):
            x = rect.left() + w * 0.12 + i * w * 0.76 / 18
            k = math.sin(i * 0.85) * 0.30 + math.sin(i * 1.9) * 0.14
            y = cy + k * h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.drawPath(path)
    elif name == "bass":
        # Low-frequency waveform with a small dot, signalling the bass range.
        path = QPainterPath()
        for i in range(15):
            x = rect.left() + w * 0.14 + i * w * 0.70 / 14
            k = math.sin(i * 0.72) * 0.28
            y = cy + h * 0.08 + k * h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.drawPath(path)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(rect.right() - w * 0.18, cy - h * 0.27), w * 0.065, w * 0.065)
    elif name == "breath":
        for i in range(3):
            path = QPainterPath()
            y = rect.top() + h * (0.36 + i * 0.16)
            path.moveTo(rect.left() + w * 0.10, y)
            path.cubicTo(
                rect.left() + w * 0.38, y - h * 0.16,
                rect.left() + w * 0.56, y + h * 0.16,
                rect.right() - w * 0.10, y,
            )
            painter.drawPath(path)
    elif name == "key":
        painter.drawRoundedRect(QRectF(rect.left() + w * 0.10, rect.top() + h * 0.12, w * 0.80, h * 0.76), 2, 2)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(rect.left() + w * 0.17, rect.top() + h * 0.12, w * 0.20, h * 0.44))
        painter.drawRect(QRectF(rect.left() + w * 0.45, rect.top() + h * 0.12, w * 0.20, h * 0.44))
    elif name == "piano":
        painter.drawRoundedRect(QRectF(rect.left() + w * 0.08, rect.top() + h * 0.12, w * 0.84, h * 0.76), 2, 2)
        painter.drawLine(QPointF(rect.left() + w * 0.25, rect.top() + h * 0.48), QPointF(rect.left() + w * 0.25, rect.bottom() - h * 0.11))
        painter.drawLine(QPointF(rect.left() + w * 0.50, rect.top() + h * 0.48), QPointF(rect.left() + w * 0.50, rect.bottom() - h * 0.11))
        painter.drawLine(QPointF(rect.left() + w * 0.75, rect.top() + h * 0.48), QPointF(rect.left() + w * 0.75, rect.bottom() - h * 0.11))
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(rect.left() + w * 0.18, rect.top() + h * 0.12, w * 0.12, h * 0.38))
        painter.drawRect(QRectF(rect.left() + w * 0.43, rect.top() + h * 0.12, w * 0.12, h * 0.38))
        painter.drawRect(QRectF(rect.left() + w * 0.68, rect.top() + h * 0.12, w * 0.12, h * 0.38))
    elif name == "export":
        _pen(painter, color)
        painter.drawLine(QPointF(cx, rect.top() + h * 0.14), QPointF(cx, rect.top() + h * 0.63))
        path = QPainterPath()
        path.moveTo(cx - w * 0.19, rect.top() + h * 0.42)
        path.lineTo(cx, rect.top() + h * 0.62)
        path.lineTo(cx + w * 0.19, rect.top() + h * 0.42)
        painter.drawPath(path)
        path2 = QPainterPath()
        path2.moveTo(rect.left() + w * 0.14, rect.top() + h * 0.66)
        path2.lineTo(rect.left() + w * 0.14, rect.bottom() - h * 0.12)
        path2.lineTo(rect.right() - w * 0.14, rect.bottom() - h * 0.12)
        path2.lineTo(rect.right() - w * 0.14, rect.top() + h * 0.66)
        painter.drawPath(path2)
    elif name == "zoom_in" or name == "zoom_out":
        painter.drawEllipse(QRectF(rect.left() + w * 0.15, rect.top() + h * 0.12, w * 0.56, h * 0.56))
        painter.drawLine(QPointF(rect.left() + w * 0.62, rect.top() + h * 0.62), QPointF(rect.right() - w * 0.12, rect.bottom() - h * 0.12))
        painter.drawLine(QPointF(rect.left() + w * 0.32, rect.top() + h * 0.40), QPointF(rect.left() + w * 0.54, rect.top() + h * 0.40))
        if name == "zoom_in":
            painter.drawLine(QPointF(rect.left() + w * 0.43, rect.top() + h * 0.29), QPointF(rect.left() + w * 0.43, rect.top() + h * 0.51))
    elif name == "fit":
        _pen(painter, color)
        x0, y0 = rect.left() + w * 0.13, rect.top() + h * 0.13
        x1, y1 = rect.right() - w * 0.13, rect.bottom() - h * 0.13
        path = QPainterPath()
        path.moveTo(x0, y0 + h * 0.22)
        path.lineTo(x0, y0)
        path.lineTo(x0 + w * 0.22, y0)
        path.moveTo(x1 - w * 0.22, y0)
        path.lineTo(x1, y0)
        path.lineTo(x1, y0 + h * 0.22)
        path.moveTo(x1, y1 - h * 0.22)
        path.lineTo(x1, y1)
        path.lineTo(x1 - w * 0.22, y1)
        path.moveTo(x0 + w * 0.22, y1)
        path.lineTo(x0, y1)
        path.lineTo(x0, y1 - h * 0.22)
        painter.drawPath(path)
        painter.drawRect(QRectF(rect.left() + w * 0.33, rect.top() + h * 0.37, w * 0.34, h * 0.26))
    elif name == "volume":
        path = QPainterPath()
        path.moveTo(rect.left() + w * 0.10, cy - h * 0.15)
        path.lineTo(rect.left() + w * 0.30, cy - h * 0.15)
        path.lineTo(rect.left() + w * 0.50, cy - h * 0.36)
        path.lineTo(rect.left() + w * 0.50, cy + h * 0.36)
        path.lineTo(rect.left() + w * 0.30, cy + h * 0.15)
        path.lineTo(rect.left() + w * 0.10, cy + h * 0.15)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawArc(QRectF(rect.left() + w * 0.40, cy - h * 0.30, w * 0.40, h * 0.60), -55 * 16, 110 * 16)
    elif name == "trash":
        painter.drawLine(QPointF(rect.left() + w * 0.18, rect.top() + h * 0.25), QPointF(rect.right() - w * 0.18, rect.top() + h * 0.25))
        painter.drawLine(QPointF(cx - w * 0.12, rect.top() + h * 0.25), QPointF(cx - w * 0.08, rect.top() + h * 0.14))
        painter.drawLine(QPointF(cx + w * 0.12, rect.top() + h * 0.25), QPointF(cx + w * 0.08, rect.top() + h * 0.14))
        painter.drawRoundedRect(QRectF(rect.left() + w * 0.24, rect.top() + h * 0.29, w * 0.52, h * 0.55), 2, 2)
        painter.drawLine(QPointF(cx - w * 0.10, rect.top() + h * 0.40), QPointF(cx - w * 0.08, rect.bottom() - h * 0.26))
        painter.drawLine(QPointF(cx + w * 0.10, rect.top() + h * 0.40), QPointF(cx + w * 0.08, rect.bottom() - h * 0.26))
    elif name == "undo":
        path = QPainterPath()
        path.moveTo(rect.left() + w * 0.18, rect.top() + h * 0.62)
        path.cubicTo(rect.left() + w * 0.18, rect.top() + h * 0.20, rect.right() - w * 0.10, rect.top() + h * 0.18, rect.right() - w * 0.15, rect.top() + h * 0.55)
        path.cubicTo(rect.right() - w * 0.18, rect.top() + h * 0.75, rect.left() + w * 0.56, rect.top() + h * 0.80, rect.left() + w * 0.38, rect.top() + h * 0.68)
        painter.drawPath(path)
        p = QPainterPath()
        p.moveTo(rect.left() + w * 0.18, rect.top() + h * 0.34)
        p.lineTo(rect.left() + w * 0.18, rect.top() + h * 0.64)
        p.lineTo(rect.left() + w * 0.45, rect.top() + h * 0.58)
        painter.drawPath(p)
    elif name == "plus":
        painter.drawLine(QPointF(cx, rect.top() + h * 0.18), QPointF(cx, rect.bottom() - h * 0.18))
        painter.drawLine(QPointF(rect.left() + w * 0.18, cy), QPointF(rect.right() - w * 0.18, cy))
    elif name == "cursor":
        path = QPainterPath()
        path.moveTo(rect.left() + w * 0.25, rect.top() + h * 0.10)
        path.lineTo(rect.left() + w * 0.72, rect.top() + h * 0.55)
        path.lineTo(rect.left() + w * 0.49, rect.top() + h * 0.60)
        path.lineTo(rect.left() + w * 0.62, rect.bottom() - h * 0.12)
        path.lineTo(rect.left() + w * 0.48, rect.bottom() - h * 0.06)
        path.lineTo(rect.left() + w * 0.36, rect.top() + h * 0.66)
        path.closeSubpath()
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)
    elif name == "grid":
        for i in range(3):
            painter.drawLine(QPointF(rect.left() + w * 0.15, rect.top() + h * (0.25 + i * 0.25)), QPointF(rect.right() - w * 0.15, rect.top() + h * (0.25 + i * 0.25)))
        for i in range(3):
            painter.drawLine(QPointF(rect.left() + w * (0.25 + i * 0.25), rect.top() + h * 0.15), QPointF(rect.left() + w * (0.25 + i * 0.25), rect.bottom() - h * 0.15))
    elif name == "metronome":
        path = QPainterPath()
        path.moveTo(rect.left() + w * 0.28, rect.bottom() - h * 0.14)
        path.lineTo(rect.left() + w * 0.42, rect.top() + h * 0.16)
        path.lineTo(rect.left() + w * 0.58, rect.top() + h * 0.16)
        path.lineTo(rect.right() - w * 0.26, rect.bottom() - h * 0.14)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(QPointF(cx, rect.top() + h * 0.36), QPointF(rect.left() + w * 0.74, rect.top() + h * 0.24))
    elif name == "power":
        painter.drawArc(QRectF(rect.left() + w * 0.20, rect.top() + h * 0.24, w * 0.60, h * 0.60), 45 * 16, 270 * 16)
        painter.drawLine(QPointF(cx, rect.top() + h * 0.14), QPointF(cx, rect.top() + h * 0.46))
    else:
        painter.drawEllipse(QRectF(rect.left() + w * 0.25, rect.top() + h * 0.25, w * 0.5, h * 0.5))

    painter.restore()


def icon(name: str, color: str = TEXT, size: int = 20) -> QIcon:
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    draw_icon(painter, name, QRectF(2, 2, size * 2 - 4, size * 2 - 4), color)
    painter.end()
    pixmap.setDevicePixelRatio(2.0)
    return QIcon(pixmap)


def app_icon() -> QIcon:
    pixmap = QPixmap(256, 256)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#171b21"))
    p.drawRoundedRect(QRectF(8, 8, 240, 240), 48, 48)
    # spectrum bars
    for i in range(9):
        height = [28, 66, 112, 164, 116, 74, 144, 88, 42][i]
        p.setBrush(QColor(ACCENT if i in (2, 3, 6, 7) else "#6b4a2b"))
        p.drawRoundedRect(QRectF(58 + i * 16, 128 - height / 2, 7, height), 3.5, 3.5)
    p.setPen(QPen(QColor("#f6f2ea"), 10))
    p.drawArc(QRectF(63, 46, 130, 165), 200 * 16, 140 * 16)
    p.end()
    return QIcon(pixmap)
