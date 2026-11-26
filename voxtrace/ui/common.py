"""Reusable controls: animated toggle, parameter slider, module rail item."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .icons import draw_icon


class ToggleSwitch(QAbstractButton):
    """A compact animated switch; use `toggled` like a checkbox."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(38, 21)
        self._offset = 1.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = Property(float, get_offset, set_offset)

    def sizeHint(self):
        return self.minimumSizeHint()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        on = self.isChecked()
        track = QColor(theme.ACCENT if on else "#343b44")
        if not self.isEnabled():
            track.setAlpha(90)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        knob_x = 2.0 + (self.width() - self.height()) * self._offset
        knob = QRectF(knob_x, 2.0, self.height() - 4, self.height() - 4)
        p.setBrush(QColor("#f7f8f8" if on else "#aeb5be"))
        p.drawEllipse(knob)
        p.end()


class ParamSlider(QWidget):
    """Label + live value + slider, RX-style compact control."""

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str = "",
        formatter=None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._suffix = suffix
        self._formatter = formatter
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        self.label = QLabel(label)
        self.label.setObjectName("Muted")
        self.value_label = QLabel()
        self.value_label.setObjectName("Value")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.label, 1)
        top.addWidget(self.value_label, 0)
        layout.addLayout(top)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._on_value)
        self._update_label(value)

    def _on_value(self, value: int) -> None:
        self._update_label(value)
        self.valueChanged.emit(float(value))

    def _update_label(self, value: int) -> None:
        if self._formatter:
            text = self._formatter(value)
        else:
            text = f"{value}{self._suffix}"
        self.value_label.setText(text)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.slider.blockSignals(False)
        self._update_label(int(value))


class ModuleItem(QWidget):
    clicked = Signal(str)

    def __init__(
        self,
        key: str,
        icon_name: str,
        title: str,
        subtitle: str,
        shortcut: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.key = key
        self._active = False
        self._hover = False
        self.icon_name = icon_name
        self.shortcut = shortcut
        self.setFixedHeight(53)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self._title = title
        self._subtitle = subtitle

    def set_active(self, active: bool) -> None:
        if self._active != active:
            self._active = active
            self.update()

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit(self.key)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0, 0, self.width(), self.height())
        if self._active:
            p.fillRect(rect, QColor(31, 36, 42))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.ACCENT))
            p.drawRoundedRect(QRectF(0, 9, 3, self.height() - 18), 1.5, 1.5)
        elif self._hover:
            p.fillRect(rect, QColor(26, 30, 36))
        icon_color = theme.ACCENT if self._active else theme.MUTED
        draw_icon(p, self.icon_name, QRectF(17, 15, 21, 21), icon_color)
        title_color = QColor(theme.TEXT if self._active else theme.TEXT_SOFT)
        p.setPen(title_color)
        font = QFont("Microsoft YaHei UI", 9)
        font.setWeight(QFont.Weight.DemiBold if self._active else QFont.Weight.Normal)
        p.setFont(font)
        p.drawText(QRectF(50, 10, self.width() - 64, 19), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._title)
        p.setPen(QColor(theme.FAINT))
        sub_font = QFont("Bahnschrift", 7)
        sub_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        p.setFont(sub_font)
        p.drawText(QRectF(50, 28, self.width() - 64, 15), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._subtitle)
        if self.shortcut:
            p.setPen(QColor(theme.FAINT))
            p.setFont(QFont("Bahnschrift", 7))
            p.drawText(QRectF(self.width() - 36, 11, 22, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self.shortcut)


class StatTile(QWidget):
    def __init__(self, label: str, value: str = "—", suffix: str = "", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(48)
        self._label = label
        self._suffix = suffix
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("BigNumber")
        self.label = QLabel(label)
        self.label.setObjectName("Hint")
        layout.addWidget(self.value_label)
        layout.addWidget(self.label)

    def set_value(self, value) -> None:
        self.value_label.setText(f"{value}{self._suffix}")
