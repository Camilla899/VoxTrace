"""VoxTrace Studio main window."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QShortcut,
    QTransform,
)
from PySide6.QtMultimedia import QAudioDevice, QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..audio_io import AudioFile, SUPPORTED_EXTENSIONS, cleanup_cache, load_audio
from ..engine import refilter_breath
from ..midi_io import MidiFileData, read_midi_notes
from ..notes import AnalysisResult, KeyInfo, NoteEvent, midi_to_name, note_to_midi_bytes
from . import theme
from .bass_roll import BassRoll
from .icons import app_icon, icon
from .inspector import Inspector
from .piano_roll import PianoRoll
from .scope import AudioScope
from .sidebar import Sidebar
from .timeline import TimeRuler, TimeView
from .worker import AnalysisWorker


class Toast(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet(
            f"background: #20262e; color: {theme.TEXT}; border: 1px solid {theme.LINE};"
            "border-radius: 6px; padding: 8px 16px; font-size: 11px;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, error: bool = False, msec: int = 2600) -> None:
        self.setText(text)
        self.setStyleSheet(
            f"background: #20262e; color: {theme.RED if error else theme.TEXT};"
            f" border: 1px solid {theme.RED if error else theme.LINE};"
            " border-radius: 6px; padding: 8px 16px; font-size: 11px;"
        )
        self.adjustSize()
        parent = self.parentWidget()
        self.move((parent.width() - self.width()) // 2, parent.height() - self.height() - 74)
        self.raise_()
        self.show()
        self._timer.start(msec)


@dataclass
class ModeSession:
    """Independent audio, result, edit history and viewport for one workflow."""

    mode: str
    audio: Optional[AudioFile] = None
    result: Optional[AnalysisResult] = None
    notes: List[NoteEvent] = field(default_factory=list)
    undo: List[List[NoteEvent]] = field(default_factory=list)
    redo: List[List[NoteEvent]] = field(default_factory=list)
    view_start: float = 0.0
    view_end: float = 10.0
    playhead: float = 0.0
    bpm: float = 120.0
    position_ms: int = 0
    demo: bool = False


class MainWindow(QMainWindow):
    def __init__(self, auto_demo: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VoxTrace Studio  ·  人声 / Bass 转 MIDI")
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.resize(1480, 940)
        self.setMinimumSize(1180, 740)
        self._apply_dark_titlebar()

        self._sessions = {
            "vocal": ModeSession("vocal"),
            "bass": ModeSession("bass"),
        }
        self.mode = "vocal"
        self.audio: Optional[AudioFile] = self._sessions["vocal"].audio
        self.result: Optional[AnalysisResult] = self._sessions["vocal"].result
        self.notes: List[NoteEvent] = self._sessions["vocal"].notes
        self._undo: List[List[NoteEvent]] = self._sessions["vocal"].undo
        self._redo: List[List[NoteEvent]] = self._sessions["vocal"].redo
        self.bass_data: Optional[MidiFileData] = None
        self.worker: Optional[AnalysisWorker] = None
        self._busy = False
        self._closing = False
        self._seek_guard = False
        self._dirty = False
        self._updating_ui = False
        self._breath_timer = QTimer(self)
        self._breath_timer.setSingleShot(True)
        self._breath_timer.setInterval(70)
        self._breath_timer.timeout.connect(self._apply_breath_filter)
        self._demo_path = Path(__file__).resolve().parents[2] / "demo" / "voxtrace_demo.wav"

        self.view = TimeView(self)
        self._build_ui()
        self._build_player()
        self._connect_shortcuts()

        self._toast = Toast(self.centralWidget())
        self._toast.hide()
        QTimer.singleShot(0, self._apply_responsive_layout)

        if auto_demo and self._demo_path.exists():
            QTimer.singleShot(280, lambda: self._load_startup_demo())
        else:
            QTimer.singleShot(150, self._show_welcome)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.moduleSelected.connect(self._module_selected)
        self.sidebar.presetSelected.connect(self._preset_selected)
        self.sidebar.featureClicked.connect(self._feature_clicked)
        self.sidebar.breathChanged.connect(self._sidebar_breath_changed)
        self.sidebar.exportRequested.connect(self.export_midi)
        root_layout.addWidget(self.sidebar)

        center = QWidget()
        self.center_widget = center
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        root_layout.addWidget(center, 1)

        self._build_topbar(center_layout)
        self._build_metabar(center_layout)
        self._build_editor(center_layout)
        self._build_transport(center_layout)

        self.inspector = Inspector()
        self.inspector.analyzeRequested.connect(self.start_analysis)
        self.inspector.exportRequested.connect(self.export_midi)
        self.inspector.paramChanged.connect(self._settings_dirty)
        self.inspector.engineChanged.connect(self._engine_changed)
        self.inspector.breathChanged.connect(self._schedule_breath_filter)
        self.inspector.breath_toggle.toggled.connect(self._sync_sidebar_breath)
        self.sidebar.set_breath_checked(self.inspector.breath_toggle.isChecked())
        self.sidebar.set_mode("vocal")
        self.inspector.set_mode("vocal")
        root_layout.addWidget(self.inspector)

    def _build_topbar(self, parent_layout) -> None:
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(56)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(7)

        self.open_btn = QPushButton("  打开音频")
        self.open_btn.setObjectName("Outline")
        self.open_btn.setIcon(icon("folder", theme.TEXT_SOFT, 16))
        self.open_btn.setFixedHeight(32)
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_btn.clicked.connect(self.open_audio_dialog)
        self.open_btn.setToolTip("打开 WAV / MP3 / FLAC / M4A / OGG 音频  (Ctrl+O)")
        layout.addWidget(self.open_btn)

        self.export_top = QPushButton("  导出 MIDI")
        self.export_top.setObjectName("Ghost")
        self.export_top.setIcon(icon("export", theme.MUTED, 15))
        self.export_top.setFixedHeight(32)
        self.export_top.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_top.clicked.connect(self.export_midi)
        self.export_top.setEnabled(False)
        layout.addWidget(self.export_top)

        sep = QFrame()
        sep.setFixedSize(1, 20)
        sep.setStyleSheet(f"background: {theme.LINE_SOFT};")
        layout.addSpacing(3)
        layout.addWidget(sep)
        layout.addSpacing(3)

        self.undo_btn = self._tool_button("undo", "撤销  (Ctrl+Z)", self.undo)
        self.undo_btn.setEnabled(False)
        self.redo_btn = self._tool_button("undo", "重做  (Ctrl+Shift+Z)", self.redo)
        self.redo_btn.setEnabled(False)
        # Mirror the undo icon for redo.
        self.redo_btn.setIcon(self._mirrored_icon("undo", theme.TEXT_SOFT))
        layout.addWidget(self.undo_btn)
        layout.addWidget(self.redo_btn)

        layout.addStretch(1)

        self.tool_label = QLabel("编辑工具")
        self.tool_label.setObjectName("Hint")
        layout.addWidget(self.tool_label)
        self.snap_toggle = QToolButton()
        self.snap_toggle.setIcon(icon("grid", theme.TEXT_SOFT, 16))
        self.snap_toggle.setCheckable(True)
        self.snap_toggle.setChecked(True)
        self.snap_toggle.setFixedSize(28, 28)
        self.snap_toggle.setToolTip("吸附网格  (编辑时按住 Shift 可临时关闭)")
        self.snap_toggle.toggled.connect(self._snap_changed)
        layout.addWidget(self.snap_toggle)

        layout.addSpacing(8)
        self.zoom_out_btn = self._tool_button("zoom_out", "缩小  (Ctrl+-)", lambda: self._zoom(1.35))
        self.zoom_in_btn = self._tool_button("zoom_in", "放大  (Ctrl+=)", lambda: self._zoom(0.74))
        self.fit_btn = self._tool_button("fit", "显示全部（含音域）  (Ctrl+0)", self._fit_view)
        layout.addWidget(self.zoom_out_btn)
        layout.addWidget(self.zoom_in_btn)
        layout.addWidget(self.fit_btn)

        layout.addSpacing(8)
        self.analyze_top = QPushButton("  分析人声")
        self.analyze_top.setObjectName("Primary")
        self.analyze_top.setIcon(icon("spark", "#20160a", 16))
        self.analyze_top.setFixedHeight(32)
        self.analyze_top.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_top.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyze_top)

        parent_layout.addWidget(bar)

    def _build_metabar(self, parent_layout) -> None:
        bar = QWidget()
        bar.setObjectName("MetaBar")
        bar.setFixedHeight(44)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(17, 0, 15, 0)
        layout.setSpacing(9)

        self.file_dot = QLabel()
        self.file_dot.setFixedSize(6, 6)
        self.file_dot.setStyleSheet(f"background: {theme.ACCENT}; border-radius: 3px;")
        layout.addWidget(self.file_dot)
        self.file_name_label = QLabel("未载入音频")
        self.file_name_label.setObjectName("FileName")
        layout.addWidget(self.file_name_label)
        self.file_meta_label = QLabel("等待音频")
        self.file_meta_label.setObjectName("Hint")
        layout.addWidget(self.file_meta_label)
        self.demo_pill = QLabel("DEMO")
        self.demo_pill.setObjectName("PillAccent")
        self.demo_pill.hide()
        layout.addWidget(self.demo_pill)
        layout.addStretch(1)
        self.key_pill = QLabel("KEY —")
        self.key_pill.setObjectName("Pill")
        self.bpm_pill = QLabel("BPM —")
        self.bpm_pill.setObjectName("Pill")
        self.notes_pill = QLabel("NOTES —")
        self.notes_pill.setObjectName("Pill")
        self.engine_pill = QLabel("LOCAL AI")
        self.engine_pill.setObjectName("PillTeal")
        for pill in (self.key_pill, self.bpm_pill, self.notes_pill, self.engine_pill):
            layout.addWidget(pill)
        parent_layout.addWidget(bar)

    def _build_editor(self, parent_layout) -> None:
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)

        self.ruler = TimeRuler(self.view)
        self.ruler.seekRequested.connect(self.seek_to)
        wl.addWidget(self.ruler)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)

        self.scope = AudioScope(self.view)
        self.scope.seekRequested.connect(self.seek_to)
        self.scope.hoverChanged.connect(self._set_status)
        splitter.addWidget(self.scope)

        piano_panel = QWidget()
        pp = QVBoxLayout(piano_panel)
        pp.setContentsMargins(0, 0, 0, 0)
        pp.setSpacing(0)
        header = QWidget()
        header.setObjectName("PianoHeader")
        header.setFixedHeight(34)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 12, 0)
        hl.setSpacing(8)
        piano_icon = QLabel()
        piano_icon.setPixmap(icon("piano", theme.ACCENT, 15).pixmap(15, 15))
        hl.addWidget(piano_icon)
        self.piano_title = QLabel("钢琴卷帘")
        self.piano_title.setStyleSheet(f"color:{theme.TEXT}; font-size:11px; font-weight:600;")
        hl.addWidget(self.piano_title)
        self.piano_sub = QLabel("PIANO ROLL")
        self.piano_sub.setStyleSheet(f"color:{theme.FAINT}; font-family: Bahnschrift; font-size:8px; letter-spacing:1.2px;")
        hl.addWidget(self.piano_sub)
        self.piano_hint = QLabel("双击添加  ·  拖动编辑  ·  滚轮上下  ·  Delete 删除")
        self.piano_hint.setObjectName("Hint")
        hl.addSpacing(8)
        hl.addWidget(self.piano_hint)
        hl.addStretch(1)
        self.bass_import_btn = QPushButton("  BASS 参考")
        self.bass_import_btn.setObjectName("BassButton")
        self.bass_import_btn.setIcon(icon("wave", theme.BLUE, 13))
        self.bass_import_btn.setFixedHeight(26)
        self.bass_import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bass_import_btn.setToolTip("载入外部 Bass MIDI 作为参考，在下方独立泳道查看  (Ctrl+B)")
        self.bass_import_btn.clicked.connect(self.import_bass_midi_dialog)
        hl.addWidget(self.bass_import_btn)
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setObjectName("BpmSpin")
        self.bpm_spin.setRange(30.0, 300.0)
        self.bpm_spin.setDecimals(1)
        self.bpm_spin.setSingleStep(0.1)
        self.bpm_spin.setValue(120.0)
        self.bpm_spin.setPrefix("BPM ")
        self.bpm_spin.setFixedWidth(96)
        self.bpm_spin.setKeyboardTracking(False)
        self.bpm_spin.setToolTip("检测到的速度；可直接修正，钢琴窗网格与 MIDI 导出会同步")
        self.bpm_spin.valueChanged.connect(self._bpm_changed)
        hl.addWidget(self.bpm_spin)
        self.snap_combo = QComboBox()
        self.snap_combo.addItem("吸附 1/16", 0.25)
        self.snap_combo.addItem("吸附 1/8", 0.5)
        self.snap_combo.addItem("吸附 1/32", 0.125)
        self.snap_combo.addItem("不吸附", 0.0)
        self.snap_combo.setFixedWidth(94)
        self.snap_combo.currentIndexChanged.connect(self._snap_changed)
        hl.addWidget(self.snap_combo)
        self.note_readout = QLabel("—")
        self.note_readout.setObjectName("Hint")
        self.note_readout.setMinimumWidth(100)
        self.note_readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hl.addWidget(self.note_readout)
        pp.addWidget(header)

        self.piano = PianoRoll(self.view)
        self.piano.notesEdited.connect(self._notes_edited)
        self.piano.seekRequested.connect(self.seek_to)
        self.piano.hoverChanged.connect(self._set_status)
        self.piano.selectionChanged.connect(self._note_selected)
        pp.addWidget(self.piano, 1)

        self.bass_panel = QWidget()
        self.bass_panel.setObjectName("BassPanel")
        self.bass_panel.setFixedHeight(150)
        bass_layout = QVBoxLayout(self.bass_panel)
        bass_layout.setContentsMargins(0, 0, 0, 0)
        bass_layout.setSpacing(0)
        bass_header = QWidget()
        bass_header.setObjectName("BassHeader")
        bass_header.setFixedHeight(30)
        bh = QHBoxLayout(bass_header)
        bh.setContentsMargins(14, 0, 10, 0)
        bh.setSpacing(8)
        bass_dot = QLabel()
        bass_dot.setFixedSize(6, 6)
        bass_dot.setStyleSheet(f"background: {theme.BLUE}; border-radius: 3px;")
        bh.addWidget(bass_dot)
        bass_title = QLabel("BASS MIDI")
        bass_title.setStyleSheet(
            f"color: {theme.BLUE}; font-family: Bahnschrift; font-size: 9px;"
            " letter-spacing: 1.2px; font-weight: 600;"
        )
        bh.addWidget(bass_title)
        self.bass_file_label = QLabel("")
        self.bass_file_label.setObjectName("Hint")
        self.bass_file_label.setFixedWidth(170)
        bh.addWidget(self.bass_file_label)
        self.bass_track_combo = QComboBox()
        self.bass_track_combo.setFixedWidth(170)
        self.bass_track_combo.setToolTip("选择要查看的 MIDI 轨道")
        self.bass_track_combo.currentIndexChanged.connect(self._bass_track_changed)
        bh.addWidget(self.bass_track_combo)
        self.bass_info = QLabel("")
        self.bass_info.setObjectName("Hint")
        self.bass_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bh.addWidget(self.bass_info, 1)
        self.bass_clear_btn = QPushButton("清除")
        self.bass_clear_btn.setObjectName("DangerGhost")
        self.bass_clear_btn.setFixedHeight(22)
        self.bass_clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bass_clear_btn.clicked.connect(lambda: self.clear_bass_midi(show_status=True))
        bh.addWidget(self.bass_clear_btn)
        bass_layout.addWidget(bass_header)
        self.bass_roll = BassRoll(self.view)
        self.bass_roll.seekRequested.connect(self.seek_to)
        self.bass_roll.hoverChanged.connect(self._set_status)
        bass_layout.addWidget(self.bass_roll, 1)
        self.bass_panel.hide()
        pp.addWidget(self.bass_panel)

        splitter.addWidget(piano_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([440, 320])
        wl.addWidget(splitter, 1)
        parent_layout.addWidget(wrap, 1)

    def _build_transport(self, parent_layout) -> None:
        bar = QWidget()
        bar.setObjectName("Transport")
        bar.setFixedHeight(48)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(8)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(f"color: {theme.MUTED}; font-size: 10px;")
        self.status_label.setMinimumWidth(150)
        layout.addWidget(self.status_label, 0)

        layout.addStretch(1)
        self.skip_btn = self._transport_button("skip_back", "回到开头", self._to_start)
        layout.addWidget(self.skip_btn)
        self.play_btn = QPushButton()
        self.play_btn.setFixedSize(34, 34)
        self.play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_btn.setIcon(icon("play", "#1b1408", 17))
        self.play_btn.setIconSize(QSize(17, 17))
        self.play_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.ACCENT}; border: 1px solid {theme.ACCENT_BRIGHT};"
            " border-radius: 17px; padding: 0; }"
            f"QPushButton:hover {{ background: {theme.ACCENT_BRIGHT}; }}"
            f"QPushButton:disabled {{ background: #493a2a; border-color: #57432e; }}"
        )
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setToolTip("播放 / 暂停  (Space)")
        layout.addWidget(self.play_btn)
        self.stop_btn = self._transport_button("stop", "停止", self.stop_playback)
        layout.addWidget(self.stop_btn)

        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setObjectName("Mono")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setMinimumWidth(130)
        layout.addWidget(self.time_label)
        layout.addStretch(1)

        self.device_btn = QToolButton()
        self.device_btn.setObjectName("DeviceButton")
        self.device_btn.setIcon(icon("volume", theme.MUTED, 16))
        self.device_btn.setIconSize(QSize(16, 16))
        self.device_btn.setFixedSize(28, 28)
        self.device_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.device_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.device_btn.setToolTip("音频输出设备")
        self.device_btn.setAccessibleName("音频输出设备")
        layout.addWidget(self.device_btn)
        self.output_name_label = QLabel("音频输出")
        self.output_name_label.setStyleSheet(f"color: {theme.FAINT}; font-size: 9px;")
        self.output_name_label.setFixedWidth(102)
        self.output_name_label.setToolTip("当前音频输出设备")
        layout.addWidget(self.output_name_label)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(78)
        self.volume.setFixedWidth(70)
        layout.addWidget(self.volume)
        parent_layout.addWidget(bar)

    def _tool_button(self, icon_name: str, tooltip: str, callback) -> QToolButton:
        button = QToolButton()
        button.setIcon(icon(icon_name, theme.TEXT_SOFT, 16))
        button.setFixedSize(28, 28)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def _transport_button(self, icon_name: str, tooltip: str, callback) -> QToolButton:
        button = QToolButton()
        button.setIcon(icon(icon_name, theme.TEXT_SOFT, 17))
        button.setFixedSize(28, 28)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    @staticmethod
    def _mirrored_icon(icon_name: str, color: str):
        source = icon(icon_name, color, 16).pixmap(32, 32)
        mirrored = source.transformed(QTransform().scale(-1, 1))
        return QIcon(mirrored)

    def _build_player(self) -> None:
        self.audio_out = QAudioOutput(self)
        self.audio_out.setVolume(0.78)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda _: self._update_time_label())
        self.player.playbackStateChanged.connect(self._on_playback_state)
        try:
            self.player.errorOccurred.connect(self._on_player_error)
        except Exception:
            pass
        self.volume.valueChanged.connect(lambda value: self.audio_out.setVolume(value / 100.0))

        # Audio output is monitored continuously: Windows may change the
        # default endpoint when headphones / Bluetooth / HDMI are connected.
        self.media_devices = QMediaDevices(self)
        self.media_devices.audioOutputsChanged.connect(self._on_audio_outputs_changed)
        self._audio_device_signature = None
        self._output_device = QAudioDevice()
        self._follow_default = True
        self._pinned_device_key = ""
        self._audio_device_timer = QTimer(self)
        self._audio_device_timer.setInterval(1200)
        self._audio_device_timer.timeout.connect(self._poll_audio_devices)
        self._device_menu = QMenu(self)
        self.device_btn.setMenu(self._device_menu)
        self._restore_audio_preference()
        self._refresh_audio_devices(initial=True, force=True, announce=False)
        self._audio_device_timer.start()

    # ---------------------------------------------------------- audio devices
    @staticmethod
    def _device_key(device: QAudioDevice) -> str:
        return bytes(device.id()).hex()

    def _device_signature_now(self):
        devices = QMediaDevices.audioOutputs()
        default = QMediaDevices.defaultAudioOutput()
        return (
            tuple((self._device_key(d), d.description(), d.isDefault()) for d in devices),
            "" if default.isNull() else self._device_key(default),
        )

    def _restore_audio_preference(self) -> None:
        settings = QSettings("VoxTrace", "VoxTrace Studio")
        self._follow_default = bool(settings.value("audio/follow_default", True, type=bool))
        pinned = settings.value("audio/device_id", "", type=str)
        self._pinned_device_key = str(pinned or "")

    def _save_audio_preference(self) -> None:
        settings = QSettings("VoxTrace", "VoxTrace Studio")
        settings.setValue("audio/follow_default", self._follow_default)
        settings.setValue("audio/device_id", "" if self._follow_default else self._pinned_device_key)
        settings.sync()

    def _on_audio_outputs_changed(self) -> None:
        self._refresh_audio_devices(announce=True)

    def _poll_audio_devices(self) -> None:
        signature = self._device_signature_now()
        if signature != self._audio_device_signature:
            self._refresh_audio_devices(announce=True)

    def _resolve_audio_output(self, devices, default: QAudioDevice):
        fallback_reason = ""
        if self._follow_default:
            selected = default
            if selected.isNull() and devices:
                selected = devices[0]
            return selected, fallback_reason

        for device in devices:
            if self._device_key(device) == self._pinned_device_key:
                return device, fallback_reason

        # The explicitly selected endpoint disappeared (USB / Bluetooth unplug).
        self._follow_default = True
        self._pinned_device_key = ""
        self._save_audio_preference()
        fallback_reason = "音频输出设备已断开 · 已切换至系统默认"
        selected = default
        if selected.isNull() and devices:
            selected = devices[0]
        return selected, fallback_reason

    def _rebuild_output_menu(self, devices) -> None:
        menu = self._device_menu
        menu.clear()
        group = QActionGroup(menu)
        group.setExclusive(True)
        system_action = QAction("跟随系统默认", menu)
        system_action.setCheckable(True)
        system_action.setChecked(self._follow_default)
        system_action.triggered.connect(lambda checked=False: self._choose_audio_output("__system__"))
        group.addAction(system_action)
        menu.addAction(system_action)
        menu.addSeparator()

        if not devices:
            empty = QAction("未检测到音频输出设备", menu)
            empty.setEnabled(False)
            menu.addAction(empty)
            return

        for index, device in enumerate(devices):
            label = device.description() or f"音频设备 {index + 1}"
            if device.isDefault():
                label += "  ·  系统默认"
            action = QAction(label, menu)
            action.setCheckable(True)
            key = self._device_key(device)
            action.setChecked(not self._follow_default and key == self._pinned_device_key)
            action.triggered.connect(lambda checked=False, k=key: self._choose_audio_output(k))
            group.addAction(action)
            menu.addAction(action)
        self._output_action_group = group

    def _choose_audio_output(self, key: str) -> None:
        if key == "__system__":
            self._follow_default = True
            self._pinned_device_key = ""
        else:
            self._follow_default = False
            self._pinned_device_key = key
        self._save_audio_preference()
        self._refresh_audio_devices(force=True, announce=True)

    def _refresh_audio_devices(self, initial: bool = False, force: bool = False, announce: bool = False) -> None:
        devices = QMediaDevices.audioOutputs()
        default = QMediaDevices.defaultAudioOutput()
        signature = (
            tuple((self._device_key(d), d.description(), d.isDefault()) for d in devices),
            "" if default.isNull() else self._device_key(default),
        )
        self._audio_device_signature = signature
        selected, fallback_reason = self._resolve_audio_output(devices, default)
        self._rebuild_output_menu(devices)

        if selected.isNull():
            self._output_device = QAudioDevice()
            self.device_btn.setIcon(icon("volume", theme.RED, 16))
            self.device_btn.setToolTip("未检测到音频输出设备\n点击查看设备列表")
            self.output_name_label.setText("无输出设备")
            self.output_name_label.setStyleSheet(f"color: {theme.RED}; font-size: 9px;")
            self.output_name_label.setToolTip("未检测到音频输出设备")
            if announce and not initial:
                self._toast.show_message("未检测到音频输出设备", error=True)
            return

        changed = self._output_device.isNull() or self._device_key(selected) != self._device_key(self._output_device)
        if changed or force:
            self._apply_audio_device(
                selected,
                announce=announce and not initial,
                reason=fallback_reason,
            )
        else:
            self._set_device_button(selected)

    def _set_device_button(self, device: QAudioDevice) -> None:
        label = device.description() or "音频输出"
        short = label.split(" (")[0].strip() or label
        display = f"{short} · 系统" if self._follow_default else short
        elided = self.output_name_label.fontMetrics().elidedText(display, Qt.TextElideMode.ElideRight, 98)
        self.output_name_label.setText(elided)
        self.output_name_label.setStyleSheet(f"color: {theme.MUTED}; font-size: 9px;")
        self.output_name_label.setToolTip(f"当前音频输出：{label}")
        self.device_btn.setIcon(icon("volume", theme.MUTED if self._follow_default else theme.TEAL, 16))
        self.device_btn.setToolTip(f"音频输出：{label}\n点击切换输出设备")

    def _apply_audio_device(self, device: QAudioDevice, announce: bool = False, reason: str = "") -> None:
        if device.isNull():
            return
        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        position = self.player.position()
        if was_playing:
            self.player.pause()
        self.audio_out.setDevice(device)
        self._output_device = device
        self._set_device_button(device)
        if was_playing:
            QTimer.singleShot(140, lambda: self._resume_after_output_switch(position))
        if announce:
            label = device.description() or "音频输出"
            message = reason or f"音频输出已切换 · {label}"
            self._toast.show_message(message)
            self._set_status(message)
        self._refresh_audio_device_menu_checks()

    def _refresh_audio_device_menu_checks(self) -> None:
        if not hasattr(self, "_output_action_group"):
            return
        system_action = None
        for action in self._output_action_group.actions():
            if action.text() == "跟随系统默认":
                system_action = action
                break
        if system_action is not None:
            system_action.setChecked(self._follow_default)

    def _resume_after_output_switch(self, position: int) -> None:
        if self.audio is None:
            return
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.player.setPosition(max(0, min(position, int(self.audio.duration * 1000) - 1)))
            self.player.play()

    def _on_player_error(self, error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        text = message or "音频播放失败"
        self._toast.show_message(f"播放错误 · {text}", error=True)
        self._set_status(f"播放错误 · {text}")

    def _shortcut(self, sequence: str, callback) -> QShortcut:
        shortcut = QShortcut(QKeySequence(sequence), self)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(callback)
        return shortcut

    def _connect_shortcuts(self) -> None:
        self._shortcut("Ctrl+O", self.open_audio_dialog)
        self._shortcut("Ctrl+E", self.export_midi)
        self._shortcut("Ctrl+Z", self.undo)
        self._shortcut("Ctrl+Shift+Z", self.redo)
        self._shortcut("Ctrl+Y", self.redo)
        self._shortcut("Ctrl+0", self._fit_view)
        self._shortcut("Ctrl+=", lambda: self._zoom(0.74))
        self._shortcut("Ctrl+-", lambda: self._zoom(1.35))
        self._shortcut("Space", self.toggle_play)
        self._shortcut("F", self.start_analysis)
        self._shortcut("Ctrl+1", lambda: self._quick_precision("fast"))
        self._shortcut("Ctrl+2", lambda: self._quick_precision("balanced"))
        self._shortcut("Ctrl+3", lambda: self._quick_precision("precise"))
        self._shortcut("Ctrl+B", self.import_bass_midi_dialog)

    def _quick_precision(self, key: str) -> None:
        self.inspector.set_precision(key)
        labels = {"fast": "快速", "balanced": "标准", "precise": "精细"}
        self._toast.show_message(f"分析精度 · {labels.get(key, key)}")

    # ------------------------------------------------------------------ startup
    def _show_welcome(self) -> None:
        self._set_status("就绪 · 打开一段人声或 Bass，或直接拖入窗口")
        self.inspector.set_status("等待音频 · 人声 / Bass 本地 AI 分析")

    def _load_startup_demo(self) -> None:
        self.load_audio_path(self._demo_path, is_demo=True, auto_analyze=True)

    def _apply_dark_titlebar(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes

            hwnd = int(self.winId())
            value = ctypes.c_int(1)
            for attribute in (20, 19):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            pass

    # ------------------------------------------------------------------ loading
    def open_audio_dialog(self) -> None:
        if self._busy:
            self._toast.show_message("分析进行中，请稍候…")
            return
        start_dir = str(Path.home() / "Music")
        if not Path(start_dir).exists():
            start_dir = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开音频",
            start_dir,
            "音频文件 (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus *.aiff *.wma);;所有文件 (*.*)",
        )
        if path:
            self.load_audio_path(path)

    def load_audio_path(self, path: str | Path, is_demo: bool = False, auto_analyze: bool = False) -> None:
        if self._busy:
            self._toast.show_message("请等待当前分析完成")
            return
        try:
            self._set_status("正在载入音频…")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            audio = load_audio(path)
            QApplication.restoreOverrideCursor()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "无法打开音频", str(exc))
            self._set_status("载入失败")
            return
        self._flush_breath_timer()
        self.player.stop()
        self.player.setSource(QUrl())
        session = self._sessions[self.mode]
        session.audio = audio
        session.result = None
        session.notes = []
        session.undo = []
        session.redo = []
        session.view_start = 0.0
        session.view_end = audio.duration
        session.playhead = 0.0
        session.bpm = 120.0
        session.position_ms = 0
        session.demo = is_demo
        self.audio = audio
        self.result = None
        self.notes = session.notes
        self._undo = session.undo
        self._redo = session.redo
        if self.mode == "vocal":
            self.clear_bass_midi(show_status=False)
        self._dirty = False
        self.view.set_duration(audio.duration)
        self.view.bpm = 120.0
        self._updating_ui = True
        self.bpm_spin.setValue(120.0)
        self._updating_ui = False
        self.scope.set_audio(audio)
        self.scope.set_analysis([], None, None, None)
        self.piano.set_mode(self.mode)
        self.piano.set_notes([])
        self.piano.set_analysis([], None, None, None, KeyInfo())
        self.inspector.clear_result()
        self.inspector.set_file_ready(audio.display_name)
        self.player.setSource(QUrl.fromLocalFile(audio.playback_path))
        self.player.setPosition(0)
        self._update_file_header()
        self._update_mode_ui()
        self.key_pill.setText("KEY —")
        self.bpm_pill.setText("BPM —")
        self.notes_pill.setText("NOTES —")
        self.export_top.setEnabled(False)
        self.sidebar.set_export_enabled(False)
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.note_readout.setText("—")
        self.play_btn.setEnabled(True)
        self._update_time_label()
        target = "Bass" if self.mode == "bass" else "人声"
        self._set_status(f"{target} 工作区已载入 · {audio.duration:.2f} 秒 · 点击「分析 {target}」")
        if auto_analyze:
            QTimer.singleShot(420, self.start_analysis)

    # ------------------------------------------------------------------ analysis
    def start_analysis(self) -> None:
        if self.audio is None:
            self._toast.show_message("请先打开一段音频", error=True)
            return
        if self._busy:
            return
        self._flush_breath_timer()
        settings = self.inspector.current_settings()
        self._busy = True
        self._dirty = False
        self._set_busy(True)
        self.inspector.set_busy(True)
        target = "Bass" if self.mode == "bass" else "人声"
        self.inspector.set_progress(1, f"准备{target}分析…")
        self._set_status(f"分析中 · {target} · 多核神经网络转录")
        self.sidebar.set_engine_text(
            "Bass AI · pYIN" if self.mode == "bass" else "RMVPE · Basic Pitch",
            min(8, os.cpu_count() or 4),
        )
        self.worker = AnalysisWorker(self.audio, settings, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_analysis_done)
        self.worker.failed.connect(self._on_analysis_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_progress(self, value: int, message: str) -> None:
        self.inspector.set_progress(value, message)
        if value < 45:
            self._set_status(message)
        elif value < 88:
            self._set_status(message)

    def _apply_result_to_ui(self, result: AnalysisResult, notes: Optional[List[NoteEvent]] = None, fit: bool = True) -> None:
        self.result = result
        self.notes = [n.copy() for n in (notes if notes is not None else result.notes)]
        self._dirty = False
        self.view.bpm = result.key.bpm
        self._updating_ui = True
        self.bpm_spin.setValue(float(round(result.key.bpm, 1)))
        self._updating_ui = False
        self.piano.invalidate_cache()
        if fit:
            self.view.fit()
        self.scope.set_analysis(result.breaths, result.f0_times, result.f0_hz, result.f0_prob)
        self.piano.set_analysis(
            result.breaths, result.f0_times, result.f0_hz, result.f0_prob, result.key
        )
        self.piano.set_notes(self.notes)
        if self.bass_data is not None:
            self.bass_roll.set_key(result.key)
        self.inspector.set_result(result)
        self._update_result_meta()
        self.export_top.setEnabled(bool(self.notes))
        self.sidebar.set_export_enabled(bool(self.notes))
        self.note_readout.setText(f"{len(self.notes)} NOTES")

    def _clear_analysis_display(self) -> None:
        self.result = None
        self.notes = []
        self._undo.clear()
        self._redo.clear()
        self.piano.set_notes([])
        self.piano.set_analysis([], None, None, None, KeyInfo())
        self.scope.set_analysis([], None, None, None)
        self.inspector.clear_result()
        self.view.bpm = 120.0
        self._updating_ui = True
        self.bpm_spin.setValue(120.0)
        self._updating_ui = False
        self.key_pill.setText("KEY  —")
        self.key_pill.setObjectName("Pill")
        for pill in (self.key_pill, self.bpm_pill, self.notes_pill):
            pill.style().unpolish(pill)
            pill.style().polish(pill)
        self.bpm_pill.setText("BPM  —")
        self.notes_pill.setText("NOTES  —")
        precision_label = {"fast": "快速", "balanced": "标准", "precise": "精细"}.get(self.inspector._precision, "标准")
        self.engine_pill.setText(f"{'BASS' if self.mode == 'bass' else 'VOCAL'} AI · {precision_label}")
        self.export_top.setEnabled(False)
        self.sidebar.set_export_enabled(False)
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.note_readout.setText("—")
        self._update_time_label()

    def _save_active_session(self) -> None:
        session = self._sessions[self.mode]
        session.audio = self.audio
        session.result = self.result
        session.notes = self.notes
        session.undo = self._undo
        session.redo = self._redo
        if self.audio is not None:
            session.view_start = self.view.start
            session.view_end = self.view.end
            session.playhead = self.view.playhead
            session.bpm = self.view.bpm
            session.position_ms = max(0, self.player.position())

    def _update_file_header(self) -> None:
        session = self._sessions[self.mode]
        audio = session.audio
        if audio is None:
            bass = self.mode == "bass"
            self.file_name_label.setText("未载入 Bass 音频" if bass else "未载入人声音频")
            self.file_meta_label.setText("BASS 工作区 · 等待音频" if bass else "VOCAL 工作区 · 等待音频")
            self.demo_pill.hide()
            return
        self.file_name_label.setText(audio.display_name)
        self.file_meta_label.setText(
            f"{Path(audio.source_path).suffix[1:].upper()} · 44.1 kHz · MONO"
        )
        self.demo_pill.setVisible(session.demo)

    def _activate_session(self, mode: str) -> None:
        self.mode = mode
        session = self._sessions[mode]
        self.audio = session.audio
        self.result = session.result
        self.notes = session.notes
        self._undo = session.undo
        self._redo = session.redo
        self.sidebar.set_mode(mode)
        self.inspector.set_mode(mode)
        self.piano.set_mode(mode)
        self.player.stop()
        self.player.setSource(QUrl())
        self._seek_guard = True
        if session.audio is not None:
            self.scope.set_audio(session.audio)
            self.view.set_duration(session.audio.duration)
            self.view.bpm = session.bpm or 120.0
            self._updating_ui = True
            self.bpm_spin.setValue(float(round(self.view.bpm, 1)))
            self._updating_ui = False
            self.player.setSource(QUrl.fromLocalFile(session.audio.playback_path))
            if session.position_ms > 0:
                self.player.setPosition(
                    max(0, min(session.position_ms, int(session.audio.duration * 1000) - 1))
                )
            self.play_btn.setEnabled(not self._busy)
            end = min(session.view_end, session.audio.duration)
            start = max(0.0, min(session.view_start, end - 0.03))
            if end > start:
                self.view.set_view(start, end)
            else:
                self.view.fit()
            self.view.set_playhead(max(0.0, min(session.playhead, session.audio.duration)))
            self.inspector.set_file_ready(session.audio.display_name)
        else:
            self.scope.set_audio(None)
            self.scope.set_analysis([], None, None, None)
            self.view.set_duration(1.0)
            self.view.bpm = 120.0
            self.view.fit()
            self.view.set_playhead(0.0)
            self._updating_ui = True
            self.bpm_spin.setValue(120.0)
            self._updating_ui = False
            self.play_btn.setEnabled(False)
        self._seek_guard = False
        self._update_file_header()
        self._update_mode_ui()
        if session.result is not None:
            self._apply_result_to_ui(session.result, session.notes, fit=False)
            self.view.bpm = session.bpm or session.result.key.bpm
            self._updating_ui = True
            self.bpm_spin.setValue(float(round(self.view.bpm, 1)))
            self._updating_ui = False
            if session.audio is not None:
                end = min(session.view_end, session.audio.duration)
                start = max(0.0, min(session.view_start, end - 0.03))
                if end > start:
                    self.view.set_view(start, end)
                self.view.set_playhead(max(0.0, min(session.playhead, session.audio.duration)))
        else:
            self._clear_analysis_display()
            if session.notes:
                self.notes = session.notes
                self.piano.set_notes(self.notes)
                self.note_readout.setText(f"{len(self.notes)} NOTES")
            if session.audio is None:
                self.inspector.set_status("等待音频 · 拖入或点击打开")
            else:
                self.inspector.set_status(f"已载入 · {session.audio.display_name} · 等待分析")
        self.undo_btn.setEnabled(bool(self._undo))
        self.redo_btn.setEnabled(bool(self._redo))
        self._set_busy(False)
        self._update_time_label()

    def _switch_mode(self, mode: str) -> None:
        if mode not in ("vocal", "bass"):
            return
        if self._busy:
            self._toast.show_message("分析进行中，请稍候再切换模式", error=True)
            self.sidebar.set_mode(self.mode)
            return
        if mode == self.mode:
            self.sidebar.set_mode(mode)
            return
        self._flush_breath_timer()
        self._save_active_session()
        self._activate_session(mode)
        label = "Bass 转 MIDI" if mode == "bass" else "人声转 MIDI"
        if self.audio is None:
            self._set_status(f"{label} · 未载入音频，请打开该工作区的音频")
        elif self.result is None:
            self._set_status(f"{label} · 已切换到独立文件，点击「开始分析」")
        else:
            self._set_status(f"{label} · 独立文件 · {len(self.notes)} 音符")

    def _update_mode_ui(self) -> None:
        bass = self.mode == "bass"
        self.piano_title.setText("Bass 音符" if bass else "钢琴卷帘")
        self.piano_sub.setText("BASS PIANO ROLL" if bass else "PIANO ROLL")
        self.scope.set_mode(self.mode)
        self.analyze_top.setText("  分析 Bass" if bass else "  分析人声")
        self.sidebar.set_engine_text("Bass AI · pYIN" if bass else "RMVPE · Basic Pitch")
        if bass:
            self.bass_import_btn.hide()
            if self.bass_panel.isVisible():
                self.bass_panel.hide()
            self.engine_pill.setText("BASS AI")
        else:
            self.bass_import_btn.show()
            if self.bass_data is not None and self.audio is not None:
                self.bass_panel.show()
                self.piano.setMinimumHeight(150)
                self._update_bass_display()
            elif self.bass_panel.isVisible():
                self.bass_panel.hide()
                self.piano.setMinimumHeight(210)
            self.engine_pill.setText("VOCAL AI")
        self.window().setWindowTitle(
            f"VoxTrace Studio  ·  {'Bass' if bass else 'Vocal'}  ·  "
            f"{self.audio.display_name if self.audio is not None else '空工作区'}"
        )

    def _on_analysis_done(self, result: AnalysisResult) -> None:
        self._apply_result_to_ui(result)
        self._save_active_session()
        self._undo.clear()
        self._redo.clear()
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self._dirty = False
        precision_label = {"fast": "快速", "balanced": "标准", "precise": "精细"}.get(result.precision, "标准")
        self._set_status(
            f"完成 · {len(self.notes)} 音符 · 调性 {result.key.name} · {precision_label}精度 · "
            f"{result.cores_used} 核 · {result.elapsed:.2f}s"
        )
        self._set_busy(False)
        self.inspector.set_busy(False)
        self.sidebar.set_active(self.mode)
        mode_label = "Bass" if self.mode == "bass" else "人声"
        self._toast.show_message(
            f"已提取 {len(self.notes)} 个{mode_label}音符  ·  {result.key.name}  ·  "
            f"{precision_label}精度  ·  {result.elapsed:.2f} 秒"
        )

    def _on_analysis_failed(self, message: str) -> None:
        self._set_busy(False)
        self.inspector.set_busy(False)
        self.inspector.set_status(f"分析失败 · {message}", error=True)
        self._set_status("分析失败")
        QMessageBox.critical(self, "分析失败", message)
        self._toast.show_message("分析失败，请检查音频文件", error=True)

    def _on_worker_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.open_btn.setEnabled(not busy)
        self.analyze_top.setEnabled(not busy)
        idle_text = "  分析 Bass" if self.mode == "bass" else "  分析人声"
        self.analyze_top.setText("  分析中…" if busy else idle_text)
        self.play_btn.setEnabled(not busy and self.audio is not None)
        self.snap_toggle.setEnabled(not busy)
        if busy:
            self.export_top.setEnabled(False)
            self.sidebar.set_export_enabled(False)
        elif self.inspector._result is not None:
            self.export_top.setEnabled(bool(self.notes))
            self.sidebar.set_export_enabled(bool(self.notes))
        self.inspector.analyze_btn.setEnabled(not busy)
        self.inspector.set_busy(busy)

    def _update_result_meta(self) -> None:
        if not self.result:
            return
        key = self.result.key
        self.key_pill.setText(f"KEY  {key.name.upper()}  ·  {key.camelot}")
        self.key_pill.setObjectName("PillAccent")
        bpm_text = f"{key.bpm:.1f}".rstrip("0").rstrip(".")
        self.bpm_pill.setText(f"BPM  {bpm_text}")
        self.notes_pill.setText(f"NOTES  {len(self.notes)}")
        precision_label = {"fast": "快速", "balanced": "标准", "precise": "精细"}.get(self.result.precision, "标准")
        self.engine_pill.setText(
            f"{'BASS' if self.mode == 'bass' else 'VOCAL'} AI · {precision_label}"
        )
        for pill in (self.key_pill, self.bpm_pill, self.notes_pill):
            pill.style().unpolish(pill)
            pill.style().polish(pill)
        if self.mode == "bass":
            self.sidebar.set_engine_text(f"Bass AI · {self.result.cores_used} 核", self.result.cores_used)
        else:
            self.sidebar.set_engine_text(f"RMVPE · {self.result.cores_used} 核", self.result.cores_used)

    def _settings_dirty(self) -> None:
        if self._updating_ui:
            return
        self._dirty = True
        if self.result is not None and not self._busy:
            self.inspector.set_status("参数已更改 · 点击「开始分析」重新提取")
        elif self.audio is not None and not self._busy:
            self.inspector.set_status("参数就绪 · 点击「开始分析」")

    def _engine_changed(self, engine: str) -> None:
        labels = {
            "vocal": "RMVPE · Basic Pitch",
            "poly": "RMVPE · Poly",
            "mono": "RMVPE · Mono",
        }
        self.sidebar.set_engine_text(labels.get(engine, "Basic Pitch · pYIN"), min(8, os.cpu_count() or 4))
        self._settings_dirty()

    def _schedule_breath_filter(self) -> None:
        if self.result is not None and not self._busy:
            self._breath_timer.start()

    def _flush_breath_timer(self) -> None:
        if self._breath_timer.isActive():
            self._breath_timer.stop()
            self._apply_breath_filter()

    def _apply_breath_filter(self) -> None:
        if self.result is None or self._busy:
            return
        settings = self.inspector.current_settings()
        self.result.settings.ignore_breath = settings.ignore_breath
        self.result.settings.breath_strength = settings.breath_strength
        self.result.settings.noise_gate = settings.noise_gate
        refilter_breath(self.result, settings)
        self.notes = [n.copy() for n in self.result.notes]
        self.piano.set_notes(self.notes, fit_range=False)
        self.inspector.notes_stat.set_value(len(self.notes))
        filtered = len(self.result.removed_breath_notes) + len(self.result.removed_noise_notes)
        self.inspector.breath_stat.set_value(filtered)
        self.notes_pill.setText(f"NOTES  {len(self.notes)}")
        self.export_top.setEnabled(bool(self.notes))
        self.sidebar.set_export_enabled(bool(self.notes))
        self.note_readout.setText(f"{len(self.notes)} NOTES")
        noise_count = len(self.result.removed_noise_notes)
        breath_count = len(self.result.removed_breath_notes)
        if self.mode == "bass":
            self._set_status(
                f"过滤已更新 · 噪声/静音 {noise_count} · 保留 {len(self.notes)} 音符"
            )
        else:
            self._set_status(
                f"过滤已更新 · 噪声/静音 {noise_count} · 气声 {breath_count} · 保留 {len(self.notes)} 音符"
            )

    # ------------------------------------------------------------------ editing
    def _notes_edited(self, notes: List[NoteEvent], description: str) -> None:
        if not notes and self.audio is None:
            return
        self._undo.append([n.copy() for n in self.notes])
        if len(self._undo) > 50:
            self._undo.pop(0)
        self._redo.clear()
        self.notes = [n.copy() for n in notes]
        if self.result is not None:
            self.result.notes = [n.copy() for n in self.notes]
        self.piano.set_notes(self.notes, fit_range=False)
        self.note_readout.setText(f"{len(self.notes)} NOTES")
        self.inspector.notes_stat.set_value(len(self.notes))
        self.notes_pill.setText(f"NOTES  {len(self.notes)}")
        self.export_top.setEnabled(bool(self.notes))
        self.sidebar.set_export_enabled(bool(self.notes))
        self.inspector.export_btn.setEnabled(bool(self.notes) and not self._busy)
        self.undo_btn.setEnabled(True)
        self.redo_btn.setEnabled(False)
        self._set_status(f"已{description} · {len(self.notes)} 个音符")

    def _note_selected(self, note) -> None:
        if note is None:
            self.note_readout.setText(f"{len(self.notes)} NOTES")
        else:
            self.note_readout.setText(f"{note.name}  ·  {note.duration:.2f}s")

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append([n.copy() for n in self.notes])
        self.notes = self._undo.pop()
        self._apply_note_history("撤销")
        self.undo_btn.setEnabled(bool(self._undo))
        self.redo_btn.setEnabled(True)

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append([n.copy() for n in self.notes])
        self.notes = self._redo.pop()
        self._apply_note_history("重做")
        self.undo_btn.setEnabled(True)
        self.redo_btn.setEnabled(bool(self._redo))

    def _apply_note_history(self, label: str) -> None:
        if self.result is not None:
            self.result.notes = [n.copy() for n in self.notes]
        self.piano.set_notes(self.notes, fit_range=False)
        self.note_readout.setText(f"{len(self.notes)} NOTES")
        self.inspector.notes_stat.set_value(len(self.notes))
        self.notes_pill.setText(f"NOTES  {len(self.notes)}")
        self.export_top.setEnabled(bool(self.notes))
        self.sidebar.set_export_enabled(bool(self.notes))
        self.inspector.export_btn.setEnabled(bool(self.notes) and not self._busy)
        self._set_status(f"已{label}")

    # ------------------------------------------------------------------ actions
    def _fit_view(self) -> None:
        self.view.fit()
        self.piano.fit_pitch_to_notes()

    def _module_selected(self, key: str) -> None:
        if key not in ("vocal", "bass"):
            return
        self._switch_mode(key)
        if self.audio is None:
            label = "Bass 转 MIDI" if key == "bass" else "人声转 MIDI"
            self._toast.show_message(f"{label} · 这是独立工作区，请打开自己的音频")
        elif self.result is None:
            label = "Bass 转 MIDI" if key == "bass" else "人声转 MIDI"
            self._set_status(f"{label} · 已选择，点击「开始分析」查看结果")

    def _feature_clicked(self, key: str) -> None:
        if key == "spectrum":
            self.view.fit()
            self.scope.setFocus()
            self._set_status("频谱与波形 · 已显示完整音频")
        elif key == "key":
            if self.result:
                key_info = self.result.key
                self._toast.show_message(
                    f"{key_info.name}  ·  {key_info.camelot}  ·  置信度 {key_info.confidence:.0f}%"
                )
            else:
                self._toast.show_message("调性检测将在分析后自动显示")
        elif key == "piano":
            self.piano.setFocus()
            self._set_status("钢琴卷帘 · 双击添加，拖动编辑")

    def _sidebar_breath_changed(self, checked: bool) -> None:
        if self.inspector.breath_toggle.isChecked() != checked:
            self.inspector.breath_toggle.setChecked(checked)
        self._set_status("气声抑制已开启" if checked else "气声抑制已关闭")

    def _sync_sidebar_breath(self, checked: bool) -> None:
        self.sidebar.set_breath_checked(checked)

    def _preset_selected(self, preset: str) -> None:
        self._updating_ui = True
        self.inspector.apply_preset(preset)
        self._updating_ui = False
        if self.result is not None:
            self.inspector.set_status("预设已应用 · 点击「开始分析」重新提取")
        labels = {
            "clean": "清唱", "whisper": "耳语", "rap": "说唱", "choir": "合唱",
            "bass_pick": "指弹贝斯", "bass_electric": "电贝斯",
            "bass_synth": "合成贝斯", "bass_upright": "低音提琴",
        }
        self._toast.show_message(f"已应用预设 · {labels.get(preset, preset)}")

    # ------------------------------------------------------------------ bass MIDI
    def import_bass_midi_dialog(self) -> None:
        if self.mode == "bass":
            self._toast.show_message("Bass 转 MIDI 已直接显示在主钢琴卷帘中", error=True)
            return
        start_dir = str(Path.home() / "Music")
        if self.audio is not None:
            start_dir = str(Path(self.audio.source_path).parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "载入 Bass MIDI",
            start_dir,
            "MIDI 文件 (*.mid *.midi);;所有文件 (*.*)",
        )
        if path:
            self.load_bass_midi(path)

    def load_bass_midi(self, path: str | Path) -> None:
        if self.mode == "bass":
            self._toast.show_message("Bass 转 MIDI 工作区不使用参考 MIDI；请直接载入 Bass 音频")
            return
        try:
            data = read_midi_notes(path)
        except Exception as exc:
            QMessageBox.critical(self, "无法读取 MIDI", str(exc))
            self._set_status("Bass MIDI 载入失败")
            return
        self.bass_data = data
        self.bass_panel.show()
        self.piano.setMinimumHeight(150)
        self.bass_import_btn.setText("  BASS ✓")
        self.bass_import_btn.setToolTip(f"参考 MIDI 已载入：{data.filename}\n点击更换")
        fm = self.bass_file_label.fontMetrics()
        self.bass_file_label.setText(fm.elidedText(data.filename, Qt.TextElideMode.ElideMiddle, 160))
        self.bass_file_label.setToolTip(str(data.path))
        self.bass_track_combo.blockSignals(True)
        self.bass_track_combo.clear()
        self.bass_track_combo.addItem("全部轨道", -1)
        preferred = data.preferred_track_index()
        preferred_index = 0
        for track in data.tracks:
            label = f"{track.name}  ·  {len(track.notes)} notes"
            if track.is_drum:
                label += "  [鼓]"
            self.bass_track_combo.addItem(label, track.index)
            if track.index == preferred:
                preferred_index = self.bass_track_combo.count() - 1
        self.bass_track_combo.setCurrentIndex(preferred_index)
        self.bass_track_combo.blockSignals(False)
        if self.audio is None:
            self.view.set_duration(max(1.0, data.duration))
        if self.result is None:
            self.view.bpm = data.bpm
            self._updating_ui = True
            self.bpm_spin.setValue(float(round(data.bpm, 1)))
            self._updating_ui = False
            bpm_text = f"{data.bpm:.1f}".rstrip("0").rstrip(".")
            self.bpm_pill.setText(f"BPM  {bpm_text}")
        self.view.fit()
        self.bass_roll.set_key(self.result.key if self.result is not None else KeyInfo())
        self._update_bass_display()
        self._update_time_label()
        self._sync_bass_button()
        self._toast.show_message(f"已载入 Bass MIDI · {data.filename}")
        self._set_status(f"Bass MIDI · {len(data.all_notes)} 音符 · {data.bpm:.1f} BPM · 参考泳道")

    def _current_bass_notes(self):
        if self.bass_data is None:
            return []
        selected = self.bass_track_combo.currentData()
        if selected is None or selected == -1:
            notes = self.bass_data.all_notes
            if notes:
                return notes
            return sorted(
                (n.copy() for t in self.bass_data.tracks for n in t.notes),
                key=lambda n: (n.start, n.pitch),
            )
        track = self.bass_data.track_by_index(int(selected))
        return track.notes if track is not None else []

    def _bass_track_changed(self) -> None:
        self._update_bass_display()

    def _update_bass_display(self) -> None:
        notes = self._current_bass_notes()
        self.bass_roll.set_notes(notes)
        if self.result is not None:
            self.bass_roll.set_key(self.result.key)
        if notes:
            low = min(n.pitch for n in notes)
            high = max(n.pitch for n in notes)
            span = f"{midi_to_name(low)}–{midi_to_name(high)}"
            self.bass_info.setText(f"{span}  ·  {len(notes)} NOTES")
        else:
            self.bass_info.setText("无音符")
        self._sync_bass_button()

    def _sync_bass_button(self) -> None:
        if self.bass_data is None:
            self.bass_import_btn.setText("  BASS 参考")
            self.bass_import_btn.setToolTip("载入外部 Bass MIDI 作为参考，在下方独立泳道查看  (Ctrl+B)")
        else:
            self.bass_import_btn.setText("  BASS ✓")

    def clear_bass_midi(self, show_status: bool = True) -> None:
        self.bass_data = None
        self.bass_panel.hide()
        self.piano.setMinimumHeight(210)
        self.bass_roll.set_notes([])
        self.bass_track_combo.blockSignals(True)
        self.bass_track_combo.clear()
        self.bass_track_combo.blockSignals(False)
        self.bass_file_label.clear()
        self.bass_info.clear()
        self._sync_bass_button()
        if show_status:
            self._toast.show_message("已关闭 Bass MIDI 参考轨")
            self._set_status("Bass MIDI 参考轨已清除")

    def export_midi(self) -> None:
        if not self.notes:
            self._toast.show_message("当前没有可导出的音符", error=True)
            return
        stem = Path(self.audio.source_path).stem if self.audio else "VoxTrace"
        suffix = "_Bass" if self.mode == "bass" else "_Vocal"
        default = str(Path(self.audio.source_path).with_name(f"{stem}{suffix}_VoxTrace.mid")) if self.audio else f"{stem}{suffix}.mid"
        path, _ = QFileDialog.getSaveFileName(self, "导出 Bass MIDI" if self.mode == "bass" else "导出 MIDI", default, "MIDI 文件 (*.mid);;所有文件 (*.*)")
        if not path:
            return
        if not path.lower().endswith((".mid", ".midi")):
            path += ".mid"
        try:
            key = self.result.key if self.result else KeyInfo()
            bpm = self.view.bpm
            track_name = f"{stem} Bass" if self.mode == "bass" else stem
            data = note_to_midi_bytes(self.notes, bpm=bpm, track_name=track_name, key=key)
            Path(path).write_bytes(data)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self._toast.show_message(f"已导出 MIDI · {Path(path).name}  ({len(self.notes)} 音符)")
        self._set_status(f"MIDI 已导出 · {path}")

    def _bpm_changed(self, value: float) -> None:
        if self._updating_ui:
            return
        self.view.bpm = float(value)
        self.piano.invalidate_cache()
        self.piano.update()
        self.bass_roll.refresh()
        if self.result is not None:
            self.result.key.bpm = float(value)
            self.inspector.set_bpm(float(value))
        bpm_text = f"{value:.1f}".rstrip("0").rstrip(".")
        self.bpm_pill.setText(f"BPM  {bpm_text}")
        self._set_status(f"BPM 已设为 {bpm_text} · 钢琴窗网格与 MIDI 导出速度已同步")

    def _snap_changed(self) -> None:
        beats = float(self.snap_combo.currentData() or 0.0)
        enabled = self.snap_toggle.isChecked() and beats > 0
        self.piano.set_snap(enabled, beats if beats > 0 else 0.25)
        if hasattr(self, "snap_toggle"):
            self.snap_toggle.setChecked(enabled)
        step_names = {0.25: "1/16", 0.5: "1/8", 0.125: "1/32"}
        self._set_status(f"网格吸附 · {step_names.get(beats, '关闭')}")

    def _zoom(self, factor: float) -> None:
        center = self.view.playhead if self.view.selection is None else sum(self.view.selection) / 2
        self.view.zoom_at(factor, center)

    # ------------------------------------------------------------------ playback
    def toggle_play(self) -> None:
        if self.audio is None:
            return
        if self._output_device.isNull():
            self._toast.show_message("未检测到音频输出设备，请连接扬声器或耳机", error=True)
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.player.position() >= max(1, int(self.audio.duration * 1000) - 20):
                self.player.setPosition(0)
                self.view.set_playhead(0)
            elif self.view.playhead > 0.5 and abs(self.player.position() / 1000.0 - self.view.playhead) > 0.25:
                # After switching workspaces the media source is reloaded; keep
                # playback aligned with the restored per-mode playhead.
                self.player.setPosition(int(self.view.playhead * 1000))
            self.player.play()

    def stop_playback(self) -> None:
        self.player.stop()
        self.view.set_playhead(0)
        self.player.setPosition(0)
        self._update_time_label()

    def _to_start(self) -> None:
        self.seek_to(0.0)

    def seek_to(self, seconds: float) -> None:
        if self.audio is None:
            return
        self._seek_guard = True
        self.player.setPosition(int(max(0.0, min(seconds, self.audio.duration)) * 1000))
        self.view.set_playhead(seconds)
        self._seek_guard = False
        self._update_time_label()

    def _on_position(self, position: int) -> None:
        if self._seek_guard or self.audio is None:
            return
        t = position / 1000.0
        self.view.set_playhead(t)
        self._update_time_label()
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            if t > self.view.end - self.view.span * 0.03 or t < self.view.start:
                span = self.view.span
                new_start = max(0.0, min(t - span * 0.12, self.view.duration - span))
                self.view.set_view(new_start, new_start + span)

    def _on_playback_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_btn.setIcon(icon("pause" if playing else "play", "#1b1408", 17))
        self.play_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.TEAL if playing else theme.ACCENT};"
            f" border: 1px solid {theme.ACCENT_BRIGHT if not playing else theme.TEAL};"
            " border-radius: 17px; padding: 0; }"
            f"QPushButton:hover {{ background: {theme.ACCENT_BRIGHT if not playing else '#83efe2'}; }}"
        )

    def _update_time_label(self) -> None:
        dur = self.audio.duration if self.audio else 0.0
        pos = self.view.playhead
        self.time_label.setText(f"{self._fmt_time(pos)} / {self._fmt_time(dur)}")

    @staticmethod
    def _fmt_time(t: float) -> str:
        return f"{int(t // 60):02d}:{t % 60:05.2f}"

    def _set_status(self, message: str) -> None:
        if hasattr(self, "status_label"):
            self.status_label.setText(message)

    # ------------------------------------------------------------------ events
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                suffix = Path(url.toLocalFile()).suffix.lower()
                if suffix in SUPPORTED_EXTENSIONS or suffix in {".mid", ".midi"}:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            suffix = Path(path).suffix.lower()
            if suffix in SUPPORTED_EXTENSIONS:
                self.load_audio_path(path)
                event.acceptProposedAction()
                return
            if suffix in {".mid", ".midi"}:
                self.load_bass_midi(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout()
        if hasattr(self, "_toast") and self._toast.isVisible():
            self._toast.move(
                (self.centralWidget().width() - self._toast.width()) // 2,
                self.centralWidget().height() - self._toast.height() - 74,
            )

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "center_widget") or not hasattr(self, "output_name_label"):
            return
        narrow = self.center_widget.width() < 760
        self.output_name_label.setVisible(not narrow)
        self.piano_hint.setVisible(not narrow)
        if hasattr(self, "bass_file_label"):
            self.bass_file_label.setVisible(not narrow)
            self.bass_info.setVisible(not narrow)
            self.bass_track_combo.setFixedWidth(140 if narrow else 170)
        self.volume.setFixedWidth(54 if narrow else 70)
        self.status_label.setMinimumWidth(108 if narrow else 150)
        self.time_label.setMinimumWidth(118 if narrow else 130)

    def closeEvent(self, event) -> None:
        self._closing = True
        self._flush_breath_timer()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            if not self.worker.wait(8000):
                self.worker.terminate()
                self.worker.wait(1000)
        self.player.stop()
        cleanup_cache()
        super().closeEvent(event)
