"""Left rail: independent Vocal and Bass workflows with integrated features."""

from __future__ import annotations

import os
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from . import theme
from .common import ModuleItem, ToggleSwitch
from .icons import app_icon, icon


class FeatureRow(QWidget):
    """A capability bundled inside the main vocal-to-MIDI module."""

    clicked = Signal(str)
    toggled = Signal(str, bool)

    def __init__(
        self,
        key: str,
        icon_name: str,
        title: str,
        status: str = "",
        switch: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.key = key
        self.icon_name = icon_name
        self._hover = False
        self.setFixedHeight(29)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 0, 6, 0)
        layout.setSpacing(7)
        self.icon_label = QLabel()
        self.icon_label.setPixmap(icon(icon_name, theme.MUTED, 13).pixmap(13, 13))
        self.icon_label.setFixedSize(14, 14)
        self.icon_label.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(self.icon_label)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"color: {theme.TEXT_SOFT}; font-size: 10px; background: transparent; border: none;"
        )
        layout.addWidget(self.title_label, 1)
        self.switch: Optional[ToggleSwitch] = None
        if switch:
            self.switch = ToggleSwitch()
            self.switch.setFixedSize(32, 18)
            self.switch.setStyleSheet("background: transparent;")
            self.switch.toggled.connect(lambda checked: self.toggled.emit(self.key, checked))
            layout.addWidget(self.switch, 0, Qt.AlignmentFlag.AlignVCenter)
        else:
            self.status_label = QLabel(status)
            self.status_label.setStyleSheet(
                f"color: {theme.FAINT}; font-size: 8px; background: transparent; border: none;"
                " font-family: Bahnschrift;"
            )
            layout.addWidget(self.status_label, 0)
        self.setToolTip(self._tooltip())

    def _tooltip(self) -> str:
        return {
            "spectrum": "波形与频谱已内置在中央工作区",
            "key": "调性检测在每次分析时自动运行",
            "breath": "气声抑制已并入人声转 MIDI 流程，切换后立即重筛音符",
            "piano": "跳到钢琴卷帘，直接编辑识别出的音符",
        }.get(self.key, "")

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._hover:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 8))
            p.drawRoundedRect(self.rect().adjusted(0, 1, 0, -1), 4, 4)
        p.end()

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


class Sidebar(QWidget):
    moduleSelected = Signal(str)
    presetSelected = Signal(str)
    featureClicked = Signal(str)
    breathChanged = Signal(bool)
    exportRequested = Signal()

    PRESETS = {
        "vocal": [("clean", "清唱"), ("whisper", "耳语"), ("rap", "说唱"), ("choir", "合唱")],
        "bass": [("bass_pick", "指弹"), ("bass_electric", "电贝斯"), ("bass_synth", "合成"), ("bass_upright", "低音提琴")],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(218)
        self._items: Dict[str, ModuleItem] = {}
        self._mode = "vocal"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Brand block -------------------------------------------------------
        brand = QWidget()
        brand.setFixedHeight(78)
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(17, 16, 12, 10)
        brand_layout.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(40, 40))
        logo.setFixedSize(40, 40)
        brand_layout.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.setContentsMargins(0, 1, 0, 0)
        name = QLabel("VOXTRACE")
        name.setObjectName("Brand")
        sub = QLabel("VOCAL / BASS → MIDI")
        sub.setObjectName("BrandSub")
        titles.addWidget(name)
        titles.addWidget(sub)
        brand_layout.addLayout(titles, 1)
        layout.addWidget(brand)

        top_line = QFrame()
        top_line.setFixedHeight(1)
        top_line.setStyleSheet(f"background: {theme.LINE_SOFT};")
        layout.addWidget(top_line)

        # One unified workspace --------------------------------------------
        module_box = QWidget()
        module_layout = QVBoxLayout(module_box)
        module_layout.setContentsMargins(0, 14, 0, 0)
        module_layout.setSpacing(0)
        section = QLabel("主工作区")
        section.setObjectName("SectionLabel")
        section.setContentsMargins(18, 0, 0, 8)
        module_layout.addWidget(section)
        module_specs = [
            ("vocal", "spark", "人声转 MIDI", "OWN FILE · PITCH · NOTES", "F"),
            ("bass", "bass", "Bass 转 MIDI", "OWN FILE · BASS · MONO", ""),
        ]
        for key, icon_name, title, subtitle, shortcut in module_specs:
            item = ModuleItem(key, icon_name, title, subtitle, shortcut)
            item.clicked.connect(self._on_module)
            self._items[key] = item
            module_layout.addWidget(item)

        # Bundled capabilities, not separate navigation modules.
        bundle = QFrame()
        bundle.setObjectName("BundleCard")
        bundle.setStyleSheet(
            f"QFrame#BundleCard {{ background: #12161b; border: 1px solid {theme.LINE_SOFT};"
            " border-radius: 0px 0px 5px 5px; border-top: none; }"
        )
        bundle_layout = QVBoxLayout(bundle)
        bundle_layout.setContentsMargins(9, 7, 9, 8)
        bundle_layout.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(6, 0, 6, 4)
        head.setSpacing(4)
        head_label = QLabel("集成功能")
        head_label.setStyleSheet(
            f"color: {theme.MUTED}; font-size: 8px; letter-spacing: 0.5px;"
            " background: transparent; border: none;"
        )
        head.addWidget(head_label)
        head.addStretch(1)
        head_hint = QLabel("ALL-IN-ONE")
        head_hint.setStyleSheet(
            f"color: {theme.FAINT}; font-family: Bahnschrift; font-size: 7px;"
            " letter-spacing: 1px; background: transparent; border: none;"
        )
        head.addWidget(head_hint)
        bundle_layout.addLayout(head)

        self.feature_rows: Dict[str, FeatureRow] = {}
        feature_specs = [
            ("spectrum", "wave", "频谱与波形", "内置", False),
            ("key", "key", "调性检测", "自动", False),
            ("breath", "breath", "气声抑制", "", True),
            ("piano", "piano", "钢琴卷帘", "编辑", False),
        ]
        for key, icon_name, title, status, switch in feature_specs:
            row = FeatureRow(key, icon_name, title, status, switch)
            row.clicked.connect(self._on_feature)
            if row.switch is not None:
                row.toggled.connect(lambda _key, checked: self.breathChanged.emit(checked))
            self.feature_rows[key] = row
            bundle_layout.addWidget(row)
        self.breath_switch = self.feature_rows["breath"].switch
        self.breath_switch.setChecked(True)
        wrap_bundle = QWidget()
        wrap_layout = QVBoxLayout(wrap_bundle)
        wrap_layout.setContentsMargins(18, 0, 18, 0)
        wrap_layout.addWidget(bundle)
        module_layout.addWidget(wrap_bundle)
        layout.addWidget(module_box)

        # Presets -----------------------------------------------------------
        layout.addSpacing(15)
        preset_box = QWidget()
        pv = QVBoxLayout(preset_box)
        pv.setContentsMargins(18, 0, 18, 0)
        pv.setSpacing(8)
        preset_head = QHBoxLayout()
        preset_head.setContentsMargins(0, 0, 0, 0)
        section2 = QLabel("快速预设")
        section2.setObjectName("SectionLabel")
        preset_head.addWidget(section2)
        preset_head.addStretch(1)
        preset_hint = QLabel("PRESETS")
        preset_hint.setObjectName("Faint")
        preset_hint.setStyleSheet("font-family: Bahnschrift; font-size: 7px; letter-spacing: 1.2px;")
        preset_head.addWidget(preset_hint)
        pv.addLayout(preset_head)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self._preset_buttons = []
        self._preset_keys = [key for key, _ in self.PRESETS["vocal"]]
        for idx, (key, label) in enumerate(self.PRESETS["vocal"]):
            button = QPushButton(label)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(26)
            button.setStyleSheet(
                f"QPushButton {{ background: {theme.PANEL_RAISED}; border: 1px solid {theme.LINE};"
                f" border-radius: 4px; color: {theme.TEXT_SOFT}; font-size: 10px; padding: 2px; }}"
                f"QPushButton:hover {{ border-color: {theme.ACCENT_DARK}; color: {theme.TEXT};"
                f" background: rgba(240,162,74,0.06); }}"
            )
            button.clicked.connect(lambda checked=False, i=idx: self._on_preset_index(i))
            self._preset_buttons.append(button)
            grid.addWidget(button, idx // 2, idx % 2)
        pv.addLayout(grid)
        self.export_btn = QPushButton("  导出 MIDI")
        self.export_btn.setObjectName("Outline")
        self.export_btn.setIcon(icon("export", theme.TEXT_SOFT, 14))
        self.export_btn.setFixedHeight(31)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("导出当前钢琴卷帘中的全部音符  (Ctrl+E)")
        self.export_btn.clicked.connect(self.exportRequested.emit)
        pv.addSpacing(3)
        pv.addWidget(self.export_btn)
        layout.addWidget(preset_box)

        layout.addStretch(1)

        # Engine status -----------------------------------------------------
        status = QFrame()
        status.setStyleSheet(
            f"QFrame#EngineCard {{ background: {theme.PANEL_RAISED}; border: 1px solid {theme.LINE};"
            " border-radius: 5px; }"
        )
        status.setObjectName("EngineCard")
        card = QVBoxLayout(status)
        card.setContentsMargins(12, 10, 12, 10)
        card.setSpacing(5)
        head2 = QHBoxLayout()
        head2.setContentsMargins(0, 0, 0, 0)
        head2.setSpacing(6)
        dot = QLabel()
        dot.setFixedSize(7, 7)
        dot.setStyleSheet(f"background: {theme.GREEN}; border-radius: 3px;")
        head2.addWidget(dot)
        online = QLabel("ENGINE READY")
        online.setStyleSheet(
            f"font-family: Bahnschrift; font-size: 8px; letter-spacing: 1.2px; color: {theme.GREEN}; border:none;"
        )
        head2.addWidget(online)
        head2.addStretch(1)
        card.addLayout(head2)
        self.engine_label = QLabel("RMVPE · Basic Pitch")
        self.engine_label.setStyleSheet(f"color: {theme.TEXT_SOFT}; font-size: 10px; border:none; background:transparent;")
        card.addWidget(self.engine_label)
        self.cores_label = QLabel(f"{min(8, os.cpu_count() or 4)} 核并行 · CPU")
        self.cores_label.setStyleSheet(f"color: {theme.FAINT}; font-size: 9px; border:none; background:transparent;")
        card.addWidget(self.cores_label)
        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setContentsMargins(16, 0, 16, 13)
        wrap_layout.addWidget(status)
        layout.addWidget(wrap)

        version = QLabel(f"v{__version__}   ·   LOCAL AI")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet(f"color: {theme.FAINT}; font-size: 8px; padding-bottom: 8px; font-family: Bahnschrift;")
        layout.addWidget(version)

        self.set_active("vocal")

    def _on_module(self, key: str) -> None:
        self.set_active(key)
        self.moduleSelected.emit(key)

    def _on_feature(self, key: str) -> None:
        if key == "breath":
            if self.breath_switch is not None:
                self.breath_switch.toggle()
            return
        self.featureClicked.emit(key)

    def _on_preset_index(self, index: int) -> None:
        if 0 <= index < len(self._preset_keys):
            self.presetSelected.emit(self._preset_keys[index])

    def set_active(self, key: str) -> None:
        for name, item in self._items.items():
            item.set_active(name == key)

    def set_mode(self, mode: str) -> None:
        if mode not in ("vocal", "bass"):
            mode = "vocal"
        self._mode = mode
        self.set_active(mode)
        self.feature_rows["breath"].setVisible(mode == "vocal")
        presets = self.PRESETS[mode]
        self._preset_keys = [key for key, _ in presets]
        for button, (_, label) in zip(self._preset_buttons, presets):
            button.setText(label)
        self.export_btn.setText("  导出 Bass MIDI" if mode == "bass" else "  导出 MIDI")
        self.export_btn.setToolTip(
            "导出当前 Bass 音符为 MIDI  (Ctrl+E)" if mode == "bass"
            else "导出当前钢琴卷帘中的全部音符  (Ctrl+E)"
        )
        self.set_engine_text(
            "Bass AI · pYIN" if mode == "bass" else "RMVPE · Basic Pitch"
        )

    def set_breath_checked(self, checked: bool) -> None:
        if self.breath_switch is None:
            return
        self.breath_switch.blockSignals(True)
        self.breath_switch.setChecked(checked)
        self.breath_switch.blockSignals(False)

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_btn.setEnabled(bool(enabled))

    def set_engine_text(self, text: str, cores: int = 0) -> None:
        self.engine_label.setText(text)
        if cores:
            self.cores_label.setText(f"{cores} 核并行 · CPU")
