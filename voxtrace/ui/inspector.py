"""Right-hand analysis inspector and result cards."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..notes import AnalysisResult, AnalysisSettings, NOTE_NAMES, FLAT_NAMES
from . import theme
from .common import ParamSlider, StatTile, ToggleSwitch
from .icons import icon


def hline() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {theme.LINE_SOFT}; border: none;")
    return line


class ConfidenceBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setFixedHeight(5)

    def set_value(self, value: float) -> None:
        self.value = max(0.0, min(1.0, float(value)))
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0, 1, self.width(), 3)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2a3038"))
        p.drawRoundedRect(r, 1.5, 1.5)
        if self.value > 0:
            grad = QLinearGradient(0, 0, self.width(), 0)
            grad.setColorAt(0, QColor(theme.ACCENT))
            grad.setColorAt(1, QColor(theme.ACCENT_BRIGHT))
            p.setBrush(grad)
            p.drawRoundedRect(QRectF(0, 1, self.width() * self.value, 3), 1.5, 1.5)
        p.end()


class Inspector(QWidget):
    analyzeRequested = Signal()
    exportRequested = Signal()
    paramChanged = Signal()
    breathChanged = Signal()
    settingsChanged = Signal()
    engineChanged = Signal(str)
    presetRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Inspector")
        self.setFixedWidth(300)
        self._busy = False
        self._result: Optional[AnalysisResult] = None
        self._mode = "vocal"
        self._mode_ranges = {"vocal": (43, 86), "bass": (28, 55)}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header.setObjectName("PanelHeader")
        header.setFixedHeight(54)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(18, 0, 16, 0)
        title = QLabel("检测与结果")
        title.setObjectName("SectionTitle")
        hl.addWidget(title)
        hl.addStretch(1)
        self.ai_badge = QLabel("AI")
        self.ai_badge.setObjectName("PillAccent")
        self.ai_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ai_badge.setFixedHeight(20)
        hl.addWidget(self.ai_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        body = QWidget()
        body.setObjectName("Inspector")
        self.body_layout = QVBoxLayout(body)
        self.body_layout.setContentsMargins(18, 14, 18, 14)
        self.body_layout.setSpacing(10)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # -- Engine ---------------------------------------------------------
        self.engine_section_label = self._label("转写引擎", "SectionLabel")
        self.body_layout.addWidget(self.engine_section_label)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("人声主旋律  ·  Vocal AI", "vocal")
        self.engine_combo.addItem("复音 / 和声  ·  Polyphonic", "poly")
        self.engine_combo.addItem("单声部清唱  ·  pYIN Mono", "mono")
        self.engine_combo.currentIndexChanged.connect(self._engine_changed)
        self.body_layout.addWidget(self.engine_combo)
        self.engine_desc = QLabel(
            "Basic Pitch + RMVPE 神经音高，针对单人主唱优化。"
        )
        self.engine_desc.setObjectName("Hint")
        self.engine_desc.setWordWrap(True)
        self.body_layout.addWidget(self.engine_desc)

        precision_head = QHBoxLayout()
        precision_head.setContentsMargins(0, 2, 0, 0)
        precision_label = QLabel("分析精度")
        precision_label.setObjectName("Muted")
        precision_head.addWidget(precision_label)
        precision_head.addStretch(1)
        self.precision_value = QLabel("标准")
        self.precision_value.setObjectName("AccentValue")
        precision_head.addWidget(self.precision_value)
        self.body_layout.addLayout(precision_head)

        self._precision = "balanced"
        self.precision_group = QButtonGroup(self)
        self.precision_group.setExclusive(True)
        segment_row = QHBoxLayout()
        segment_row.setContentsMargins(0, 0, 0, 0)
        segment_row.setSpacing(4)
        self._precision_tips_vocal = [
            ("fast", "快速", "快速：pYIN 音高 + 单遍转录，省去大模型加载  (Ctrl+1)"),
            ("balanced", "标准", "标准：RMVPE 神经音高 + Basic Pitch，速度与精度平衡  (Ctrl+2)"),
            ("precise", "精细", "精细：双阈值召回 + 神经音高补全，优先召回与边界精度  (Ctrl+3)"),
        ]
        self._precision_tips_bass = [
            ("fast", "快速", "快速：Basic Pitch 单遍转录 + pYIN 低频音高  (Ctrl+1)"),
            ("balanced", "标准", "标准：Basic Pitch + pYIN 低频校验，速度与精度平衡  (Ctrl+2)"),
            ("precise", "精细", "精细：双阈值召回 + pYIN 低频音高验证  (Ctrl+3)"),
        ]
        self.precision_buttons = {}
        for key, text, tip in self._precision_tips_vocal:
            button = QPushButton(text)
            button.setObjectName("Segment")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(tip)
            button.setProperty("precisionKey", key)
            if key == "balanced":
                button.setChecked(True)
            self.precision_group.addButton(button)
            self.precision_buttons[key] = button
            segment_row.addWidget(button, 1)
        self.precision_group.buttonClicked.connect(self._precision_clicked)
        self.body_layout.addLayout(segment_row)

        self.sensitivity = ParamSlider(
            "灵敏度", 0, 100, 58, formatter=lambda v: f"{v}%"
        )
        self.sensitivity.valueChanged.connect(self._param_changed)
        self.body_layout.addWidget(self.sensitivity)
        self.min_note = ParamSlider(
            "最短音符", 40, 400, 90, formatter=lambda v: f"{v} ms"
        )
        self.min_note.valueChanged.connect(self._param_changed)
        self.body_layout.addWidget(self.min_note)

        range_row = QHBoxLayout()
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.setSpacing(8)
        self.range_label = QLabel("人声音域")
        self.range_label.setObjectName("Muted")
        range_row.addWidget(self.range_label, 1)
        self.low_combo = QComboBox()
        self.high_combo = QComboBox()
        for midi in range(21, 97):
            name = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
            self.low_combo.addItem(name, midi)
            self.high_combo.addItem(name, midi)
        self.low_combo.setCurrentIndex(self.low_combo.findData(43))
        self.high_combo.setCurrentIndex(self.high_combo.findData(86))
        self.low_combo.setFixedWidth(58)
        self.high_combo.setFixedWidth(58)
        self.low_combo.currentIndexChanged.connect(self._range_changed)
        self.high_combo.currentIndexChanged.connect(self._range_changed)
        range_row.addWidget(self.low_combo)
        dash = QLabel("—")
        dash.setObjectName("Faint")
        range_row.addWidget(dash)
        range_row.addWidget(self.high_combo)
        self.body_layout.addLayout(range_row)

        self.body_layout.addWidget(hline())

        # -- Breath / performance ------------------------------------------
        self.breath_section_label = self._label("气声与噪声", "SectionLabel")
        self.body_layout.addWidget(self.breath_section_label)
        breath_row = QHBoxLayout()
        breath_row.setContentsMargins(0, 0, 0, 0)
        breath_row.setSpacing(10)
        labels = QVBoxLayout()
        labels.setSpacing(1)
        self.breath_title = QLabel("忽略气声 / 呼吸音")
        self.breath_title.setStyleSheet(f"color:{theme.TEXT_SOFT}; font-weight:600;")
        self.breath_sub = QLabel("从 MIDI 中剔除气口、换气与噪声误检")
        self.breath_sub.setObjectName("Hint")
        self.breath_sub.setWordWrap(True)
        labels.addWidget(self.breath_title)
        labels.addWidget(self.breath_sub)
        breath_row.addLayout(labels, 1)
        self.breath_toggle = ToggleSwitch()
        self.breath_toggle.setChecked(True)
        self.breath_toggle.toggled.connect(self._breath_toggled)
        breath_row.addWidget(self.breath_toggle, 0, Qt.AlignmentFlag.AlignTop)
        self.body_layout.addLayout(breath_row)
        self.breath_strength = ParamSlider(
            "气声抑制强度", 0, 100, 62, formatter=lambda v: f"{v}%"
        )
        self.breath_strength.valueChanged.connect(self._breath_strength_changed)
        self.body_layout.addWidget(self.breath_strength)
        self.noise_gate = ParamSlider(
            "噪声门 · 静音过滤", 0, 100, 70, formatter=lambda v: "关闭" if v <= 0 else f"{v}%"
        )
        self.noise_gate.setToolTip("过滤静音段、底噪和弱噪声产生的假音符；调高会更干净，但可能误删极弱唱音")
        self.noise_gate.valueChanged.connect(self._noise_gate_changed)
        self.body_layout.addWidget(self.noise_gate)

        perf_row = QHBoxLayout()
        perf_row.setContentsMargins(0, 0, 0, 0)
        perf_row.setSpacing(8)
        core_label = QLabel("并行核心")
        core_label.setObjectName("Muted")
        perf_row.addWidget(core_label, 1)
        self.cores_combo = QComboBox()
        cpu = os.cpu_count() or 4
        auto_cores = min(8, cpu)
        self.cores_combo.addItem(f"自动 · {auto_cores} 核", 0)
        for value in (2, 4, 6, 8, 12):
            if value <= cpu:
                self.cores_combo.addItem(f"{value} 线程", value)
        self.cores_combo.currentIndexChanged.connect(self._param_changed)
        self.cores_combo.setToolTip(f"多核神经网络推理 · {cpu} 个逻辑处理器可用")
        perf_row.addWidget(self.cores_combo, 1)
        self.body_layout.addLayout(perf_row)

        self.analyze_btn = QPushButton("  开始分析")
        self.analyze_btn.setObjectName("Primary")
        self.analyze_btn.setIcon(icon("spark", "#20160a", 16))
        self.analyze_btn.setFixedHeight(39)
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.analyzeRequested.emit)
        self.analyze_btn.setToolTip("分析人声并生成钢琴卷帘  (F)")
        self.body_layout.addWidget(self.analyze_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(3)
        self.progress.hide()
        self.body_layout.addWidget(self.progress)
        self.status_label = QLabel("等待音频 · 本地 AI 分析")
        self.status_label.setObjectName("Hint")
        self.body_layout.addWidget(self.status_label)

        self.body_layout.addWidget(hline())

        # -- Result ---------------------------------------------------------
        result_head = QHBoxLayout()
        result_head.setContentsMargins(0, 0, 0, 0)
        result_head.addWidget(self._label("分析结果", "SectionLabel"))
        result_head.addStretch(1)
        self.result_badge = QLabel("—")
        self.result_badge.setObjectName("Pill")
        result_head.addWidget(self.result_badge)
        self.body_layout.addLayout(result_head)

        self.key_card = QFrame()
        self.key_card.setStyleSheet(
            f"QFrame {{ background: {theme.PANEL_RAISED}; border: 1px solid {theme.LINE}; border-radius: 6px; }}"
            "QFrame QLabel { border: none; background: transparent; }"
        )
        key_layout = QVBoxLayout(self.key_card)
        key_layout.setContentsMargins(13, 12, 13, 11)
        key_layout.setSpacing(4)
        key_top = QHBoxLayout()
        key_top.setContentsMargins(0, 0, 0, 0)
        self.key_name = QLabel("—")
        self.key_name.setObjectName("KeyName")
        key_top.addWidget(self.key_name)
        key_top.addStretch(1)
        self.camelot = QLabel("—")
        self.camelot.setObjectName("Camelot")
        key_top.addWidget(self.camelot, 0, Qt.AlignmentFlag.AlignTop)
        key_layout.addLayout(key_top)
        self.key_mode = QLabel("等待分析")
        self.key_mode.setObjectName("KeyMode")
        key_layout.addWidget(self.key_mode)
        self.confidence_bar = ConfidenceBar()
        key_layout.addWidget(self.confidence_bar)
        key_bottom = QHBoxLayout()
        key_bottom.setContentsMargins(0, 2, 0, 0)
        self.confidence_label = QLabel("调性置信度")
        self.confidence_label.setObjectName("Hint")
        self.confidence_value = QLabel("—")
        self.confidence_value.setObjectName("Mono")
        key_bottom.addWidget(self.confidence_label, 1)
        key_bottom.addWidget(self.confidence_value)
        key_layout.addLayout(key_bottom)
        self.body_layout.addWidget(self.key_card)

        stats = QHBoxLayout()
        stats.setContentsMargins(0, 2, 0, 0)
        stats.setSpacing(6)
        self.notes_stat = StatTile("检测音符")
        self.breath_stat = StatTile("已过滤")
        self.time_stat = StatTile("用时", suffix="s")
        stats.addWidget(self.notes_stat, 1)
        stats.addWidget(self.breath_stat, 1)
        stats.addWidget(self.time_stat, 1)
        self.body_layout.addLayout(stats)

        self.export_btn = QPushButton("  导出 MIDI")
        self.export_btn.setObjectName("Outline")
        self.export_btn.setIcon(icon("export", theme.TEXT_SOFT, 15))
        self.export_btn.setFixedHeight(34)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.clicked.connect(self.exportRequested.emit)
        self.export_btn.setToolTip("将当前钢琴卷帘导出为标准 MIDI 文件  (Ctrl+E)")
        self.export_btn.setEnabled(False)
        self.body_layout.addWidget(self.export_btn)

        footer = QLabel("本地处理  ·  音频不会上传")
        footer.setObjectName("Hint")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body_layout.addSpacing(2)
        self.body_layout.addWidget(footer)

    @staticmethod
    def _label(text: str, object_name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(object_name)
        return label

    # ------------------------------------------------------------------ state
    def set_mode(self, mode: str) -> None:
        if mode not in ("vocal", "bass"):
            mode = "vocal"
        previous = self._mode
        if previous != mode:
            self._mode_ranges[previous] = (
                int(self.low_combo.currentData() or 43),
                int(self.high_combo.currentData() or 86),
            )
        self._mode = mode
        bass = mode == "bass"
        self.engine_section_label.setVisible(not bass)
        self.engine_combo.setVisible(not bass)
        if bass:
            self.engine_desc.setText(
                "Bass AI：Basic Pitch 提取低音音符，pYIN 低频跟踪校验音高、修正八度，保留单音贝斯线条。"
            )
            self.breath_section_label.setText("低频清理")
            self.range_label.setText("Bass 音域")
            self.ai_badge.setText("BASS")
            tips = self._precision_tips_bass
        else:
            self.engine_desc.setText("Basic Pitch + RMVPE 神经音高，针对单人主唱优化。")
            self.breath_section_label.setText("气声与噪声")
            self.range_label.setText("人声音域")
            self.ai_badge.setText("AI")
            tips = self._precision_tips_vocal
        for widget in (self.breath_title, self.breath_sub, self.breath_toggle, self.breath_strength):
            widget.setVisible(not bass)
        self.export_btn.setText("  导出 Bass MIDI" if bass else "  导出 MIDI")
        for key, _, tip in tips:
            self.precision_buttons[key].setToolTip(tip)
        low, high = self._mode_ranges[mode]
        self.low_combo.blockSignals(True)
        self.high_combo.blockSignals(True)
        low_index = self.low_combo.findData(low)
        high_index = self.high_combo.findData(high)
        if low_index >= 0:
            self.low_combo.setCurrentIndex(low_index)
        if high_index >= 0:
            self.high_combo.setCurrentIndex(high_index)
        self.low_combo.blockSignals(False)
        self.high_combo.blockSignals(False)
        self.analyze_btn.setToolTip("分析 Bass 并生成低频 MIDI  (F)" if bass else "分析人声并生成钢琴卷帘  (F)")

    def _engine_changed(self) -> None:
        engine = self.engine_combo.currentData()
        descriptions = {
            "vocal": "Basic Pitch + RMVPE 神经音高，针对单人主唱优化。",
            "poly": "Basic Pitch 复音转写，保留和声、叠唱与宽广音域。",
            "mono": "RMVPE 神经音高分割，适合清唱与说话。",
        }
        self.engine_desc.setText(descriptions.get(engine, ""))
        self.engineChanged.emit(str(engine))
        self.paramChanged.emit()

    def _precision_clicked(self, button: QPushButton) -> None:
        key = str(button.property("precisionKey") or "balanced")
        self._set_precision(key)
        self.paramChanged.emit()

    def _set_precision(self, key: str, emit: bool = False) -> None:
        if key not in self.precision_buttons:
            key = "balanced"
        self._precision = key
        for name, button in self.precision_buttons.items():
            button.setChecked(name == key)
        labels = {"fast": "快速", "balanced": "标准", "precise": "精细"}
        self.precision_value.setText(labels[key])
        if emit:
            self.paramChanged.emit()

    def set_precision(self, key: str) -> None:
        self._set_precision(key, emit=True)

    def _param_changed(self) -> None:
        self.paramChanged.emit()

    def _range_changed(self) -> None:
        if self.low_combo.currentData() >= self.high_combo.currentData():
            sender = self.sender()
            if sender is self.low_combo:
                self.high_combo.setCurrentIndex(min(self.high_combo.count() - 1, self.low_combo.currentIndex() + 12))
            else:
                self.low_combo.setCurrentIndex(max(0, self.high_combo.currentIndex() - 12))
        self._mode_ranges[self._mode] = (
            int(self.low_combo.currentData() or 43),
            int(self.high_combo.currentData() or 86),
        )
        self.paramChanged.emit()

    def _breath_toggled(self, checked: bool) -> None:
        self.breath_strength.setEnabled(checked)
        self.breathChanged.emit()
        self.paramChanged.emit()

    def _breath_strength_changed(self, value: float) -> None:
        self.breathChanged.emit()
        self.paramChanged.emit()

    def _noise_gate_changed(self, value: float) -> None:
        self.breathChanged.emit()
        self.paramChanged.emit()

    def current_settings(self) -> AnalysisSettings:
        threshold = 0.72 - (self.sensitivity.value() / 100.0) * 0.40
        engine = "bass" if self._mode == "bass" else str(self.engine_combo.currentData() or "vocal")
        return AnalysisSettings(
            engine=engine,
            precision=self._precision,
            onset_threshold=float(np.clip(threshold, 0.28, 0.72)),
            frame_threshold=0.30,
            min_note_ms=int(self.min_note.value()),
            min_midi=int(self.low_combo.currentData() or 43),
            max_midi=int(self.high_combo.currentData() or 86),
            ignore_breath=self.breath_toggle.isChecked() and self._mode != "bass",
            breath_strength=self.breath_strength.value() / 100.0,
            noise_gate=self.noise_gate.value() / 100.0,
            detect_bpm=True,
            pitch_refine=True,
            workers=int(self.cores_combo.currentData() or 0),
        )

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.analyze_btn.setEnabled(not busy)
        self.export_btn.setEnabled(not busy and self._result is not None and bool(self._result.notes))
        self.progress.setVisible(busy)
        if not busy:
            self.progress.setValue(0)
        self.analyze_btn.setText("  正在分析…" if busy else "  开始分析")

    def set_progress(self, value: int, message: str) -> None:
        self.progress.setValue(max(0, min(100, value)))
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {theme.ACCENT}; font-size: 10px;")

    def set_status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {theme.RED if error else theme.MUTED}; font-size: 10px;")

    def set_result(self, result: AnalysisResult) -> None:
        self._result = result
        key = result.key
        self.result_badge.setText(f"{result.engine_label.split('·')[0].strip()}")
        self.result_badge.setObjectName("PillTeal")
        self.repaint_badge()
        self.key_name.setText(f"{NOTE_NAMES[key.tonic % 12]} {key.chinese_mode}")
        flats = key.mode == "minor" and key.tonic in {1, 3, 8, 10}
        if flats:
            self.key_name.setText(f"{FLAT_NAMES[key.tonic % 12]} {key.chinese_mode}")
        self.camelot.setText(key.camelot)
        self.key_mode.setText(f"调号 {key.signature}  ·  {round(key.bpm)} BPM  ·  {key.camelot} 轮盘")
        self.confidence_bar.set_value(key.confidence / 100.0)
        self.confidence_value.setText(f"{key.confidence:.0f}%")
        self.notes_stat.set_value(len(result.notes))
        self.breath_stat.set_value(len(result.removed_breath_notes) + len(result.removed_noise_notes))
        self.time_stat.set_value(f"{result.elapsed:.2f}")
        self.export_btn.setEnabled(bool(result.notes) and not self._busy)
        precision_label = {"fast": "快速", "balanced": "标准", "precise": "精细"}.get(result.precision, "标准")
        filtered = len(result.removed_breath_notes) + len(result.removed_noise_notes)
        filter_text = f" · 过滤 {filtered}" if filtered else ""
        self.set_status(
            f"完成 · {len(result.notes)} 音符{filter_text} · {precision_label}精度 · "
            f"{result.cores_used} 核 · {result.elapsed:.2f} 秒"
        )

    def set_bpm(self, bpm: float) -> None:
        if self._result is None:
            return
        key = self._result.key
        key.bpm = float(bpm)
        self.key_mode.setText(f"调号 {key.signature}  ·  {bpm:.1f} BPM  ·  {key.camelot} 轮盘")

    def repaint_badge(self) -> None:
        self.result_badge.style().unpolish(self.result_badge)
        self.result_badge.style().polish(self.result_badge)

    def clear_result(self) -> None:
        self._result = None
        self.key_name.setText("—")
        self.camelot.setText("—")
        self.key_mode.setText("等待分析")
        self.confidence_bar.set_value(0)
        self.confidence_value.setText("—")
        self.notes_stat.set_value("—")
        self.breath_stat.set_value("—")
        self.time_stat.set_value("—")
        self.export_btn.setEnabled(False)
        self.result_badge.setText("—")
        self.result_badge.setObjectName("Pill")
        self.repaint_badge()

    def set_file_ready(self, name: str) -> None:
        self.set_status(f"已载入 · {name}")
        self.analyze_btn.setEnabled(True)

    def apply_preset(self, preset: str) -> None:
        if preset == "clean":
            self.engine_combo.setCurrentIndex(0)
            self._set_precision("balanced")
            self.sensitivity.set_value(58)
            self.min_note.set_value(90)
            self.breath_toggle.setChecked(True)
            self.breath_strength.set_value(62)
            self.noise_gate.set_value(70)
        elif preset == "whisper":
            self.engine_combo.setCurrentIndex(0)
            self._set_precision("precise")
            self.sensitivity.set_value(76)
            self.min_note.set_value(130)
            self.breath_toggle.setChecked(True)
            self.breath_strength.set_value(82)
            self.noise_gate.set_value(35)
        elif preset == "rap":
            self.engine_combo.setCurrentIndex(2)
            self._set_precision("fast")
            self.sensitivity.set_value(68)
            self.min_note.set_value(70)
            self.breath_toggle.setChecked(True)
            self.breath_strength.set_value(55)
            self.noise_gate.set_value(78)
        elif preset == "choir":
            self.engine_combo.setCurrentIndex(1)
            self._set_precision("precise")
            self.sensitivity.set_value(54)
            self.min_note.set_value(110)
            self.breath_toggle.setChecked(False)
            self.breath_strength.set_value(45)
            self.noise_gate.set_value(45)
        elif preset == "bass_pick":
            self._set_precision("balanced")
            self.sensitivity.set_value(60)
            self.min_note.set_value(85)
            self.noise_gate.set_value(60)
            self._set_mode_range(28, 55)
        elif preset == "bass_electric":
            self._set_precision("precise")
            self.sensitivity.set_value(68)
            self.min_note.set_value(65)
            self.noise_gate.set_value(68)
            self._set_mode_range(24, 60)
        elif preset == "bass_synth":
            self._set_precision("fast")
            self.sensitivity.set_value(55)
            self.min_note.set_value(70)
            self.noise_gate.set_value(55)
            self._set_mode_range(21, 55)
        elif preset == "bass_upright":
            self._set_precision("precise")
            self.sensitivity.set_value(64)
            self.min_note.set_value(110)
            self.noise_gate.set_value(50)
            self._set_mode_range(28, 60)

    def _set_mode_range(self, low: int, high: int) -> None:
        self._mode_ranges[self._mode] = (low, high)
        self.low_combo.blockSignals(True)
        self.high_combo.blockSignals(True)
        low_index = self.low_combo.findData(low)
        high_index = self.high_combo.findData(high)
        if low_index >= 0:
            self.low_combo.setCurrentIndex(low_index)
        if high_index >= 0:
            self.high_combo.setCurrentIndex(high_index)
        self.low_combo.blockSignals(False)
        self.high_combo.blockSignals(False)
