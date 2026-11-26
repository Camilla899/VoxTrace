"""Visual language for VoxTrace Studio: deep graphite, warm amber, spectral teal."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase

# Core palette -----------------------------------------------------------------
BG = "#0e1013"
BG_DEEP = "#0a0c0f"
PANEL = "#15181d"
PANEL_RAISED = "#1b1f25"
PANEL_SOFT = "#20252c"
LINE = "#2a3038"
LINE_SOFT = "#22272e"
TEXT = "#e8eaed"
TEXT_SOFT = "#bac0c8"
MUTED = "#848c98"
FAINT = "#5f6772"
ACCENT = "#f0a24a"
ACCENT_BRIGHT = "#ffc36d"
ACCENT_DARK = "#b36f27"
TEAL = "#59d8c8"
TEAL_DARK = "#2d8b83"
BLUE = "#78a9ff"
RED = "#ff746f"
GREEN = "#7bcf8b"
PURPLE = "#b897ff"
SELECTION = "#f0a24a"
BREATH = "#7c93a8"

CHINESE_FONT = '"Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI"'
DISPLAY_FONT = '"Bahnschrift", "Segoe UI"'
MONO_FONT = '"Cascadia Mono", "Consolas", monospace'
CHEVRON_SVG = (Path(__file__).resolve().parent / "assets" / "chevron-down.svg").as_posix()

COLOR_MAP = {
    "bg": BG,
    "bg_deep": BG_DEEP,
    "panel": PANEL,
    "raised": PANEL_RAISED,
    "soft": PANEL_SOFT,
    "line": LINE,
    "text": TEXT,
    "soft_text": TEXT_SOFT,
    "muted": MUTED,
    "faint": FAINT,
    "accent": ACCENT,
    "accent_bright": ACCENT_BRIGHT,
    "teal": TEAL,
    "blue": BLUE,
    "red": RED,
}


def qcolor(name: str, alpha: int | None = None) -> QColor:
    c = QColor(COLOR_MAP.get(name, name))
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def app_font(size: int = 10, weight: int = 400) -> QFont:
    font = QFont("Microsoft YaHei UI" if "Microsoft YaHei UI" in QFontDatabase.families() else "Segoe UI")
    font.setPointSize(size)
    font.setWeight(QFont.Weight(weight))
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


STYLESHEET = f"""
* {{
    outline: none;
}}
QMainWindow, QDialog {{
    background: {BG};
    color: {TEXT};
}}
QWidget {{
    color: {TEXT};
    font-family: {CHINESE_FONT};
    font-size: 12px;
}}
QWidget#Root {{
    background: {BG};
}}
QWidget#Sidebar {{
    background: {PANEL};
    border-right: 1px solid {LINE_SOFT};
}}
QWidget#Inspector {{
    background: {PANEL};
    border-left: 1px solid {LINE_SOFT};
}}
QWidget#TopBar {{
    background: {PANEL};
    border-bottom: 1px solid {LINE_SOFT};
}}
QWidget#MetaBar {{
    background: {BG};
    border-bottom: 1px solid {LINE_SOFT};
}}
QWidget#Transport {{
    background: {PANEL};
    border-top: 1px solid {LINE_SOFT};
}}
QWidget#PanelHeader {{
    background: {PANEL_RAISED};
    border-bottom: 1px solid {LINE_SOFT};
}}
QWidget#PianoHeader {{
    background: {PANEL_RAISED};
    border-bottom: 1px solid {LINE_SOFT};
}}
QLabel {{
    background: transparent;
}}
QLabel#Brand {{
    font-family: {DISPLAY_FONT};
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 2px;
    color: {TEXT};
}}
QLabel#BrandSub {{
    font-family: {DISPLAY_FONT};
    font-size: 8px;
    letter-spacing: 3px;
    color: {MUTED};
}}
QLabel#SectionLabel {{
    font-family: {DISPLAY_FONT};
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: {MUTED};
}}
QLabel#SectionTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#FileName {{
    font-size: 14px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#Muted {{
    color: {MUTED};
}}
QLabel#Faint {{
    color: {FAINT};
}}
QLabel#Mono {{
    font-family: {MONO_FONT};
    color: {TEXT_SOFT};
}}
QLabel#Value {{
    font-family: {MONO_FONT};
    font-size: 11px;
    color: {TEXT_SOFT};
}}
QLabel#AccentValue {{
    font-family: {MONO_FONT};
    font-size: 11px;
    color: {ACCENT};
}}
QLabel#KeyName {{
    font-family: {DISPLAY_FONT};
    font-size: 29px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#KeyMode {{
    font-size: 11px;
    color: {MUTED};
}}
QLabel#BigNumber {{
    font-family: {DISPLAY_FONT};
    font-size: 21px;
    color: {TEXT};
}}
QLabel#Camelot {{
    font-family: {DISPLAY_FONT};
    font-size: 11px;
    font-weight: 600;
    color: {TEAL};
    background: rgba(89, 216, 200, 0.10);
    border: 1px solid rgba(89, 216, 200, 0.30);
    border-radius: 8px;
    padding: 2px 8px;
}}
QLabel#Pill {{
    font-family: {DISPLAY_FONT};
    font-size: 9px;
    letter-spacing: 0.8px;
    color: {TEXT_SOFT};
    background: {PANEL_SOFT};
    border: 1px solid {LINE};
    border-radius: 8px;
    padding: 2px 8px;
}}
QLabel#PillAccent {{
    font-family: {DISPLAY_FONT};
    font-size: 9px;
    letter-spacing: 0.8px;
    color: {ACCENT_BRIGHT};
    background: rgba(240, 162, 74, 0.10);
    border: 1px solid rgba(240, 162, 74, 0.28);
    border-radius: 8px;
    padding: 2px 8px;
}}
QLabel#PillTeal {{
    font-family: {DISPLAY_FONT};
    font-size: 9px;
    letter-spacing: 0.8px;
    color: {TEAL};
    background: rgba(89, 216, 200, 0.08);
    border: 1px solid rgba(89, 216, 200, 0.24);
    border-radius: 8px;
    padding: 2px 8px;
}}
QLabel#Hint {{
    color: {FAINT};
    font-size: 10px;
}}
QLabel#Error {{
    color: {RED};
    font-size: 10px;
}}

QPushButton {{
    color: {TEXT_SOFT};
    background: {PANEL_SOFT};
    border: 1px solid {LINE};
    border-radius: 5px;
    padding: 6px 12px;
}}
QPushButton:hover {{
    background: #262c34;
    border-color: #3a424c;
    color: {TEXT};
}}
QPushButton:pressed {{
    background: #111418;
}}
QPushButton:disabled {{
    color: {FAINT};
    background: #181b20;
    border-color: {LINE_SOFT};
}}
QPushButton#Primary {{
    color: #20160a;
    background: {ACCENT};
    border: 1px solid {ACCENT_BRIGHT};
    border-radius: 5px;
    font-weight: 600;
    padding: 8px 16px;
}}
QPushButton#Primary:hover {{
    background: {ACCENT_BRIGHT};
}}
QPushButton#Primary:pressed {{
    background: #d98d37;
}}
QPushButton#Primary:disabled {{
    color: #6e563b;
    background: #493a2a;
    border-color: #57432e;
}}
QPushButton#Ghost {{
    background: transparent;
    border: 1px solid transparent;
    color: {TEXT_SOFT};
    padding: 5px 8px;
}}
QPushButton#Ghost:hover {{
    background: {PANEL_SOFT};
    border-color: {LINE_SOFT};
    color: {TEXT};
}}
QPushButton#BassButton {{
    background: rgba(79, 157, 255, 0.07);
    border: 1px solid rgba(79, 157, 255, 0.32);
    border-radius: 4px;
    color: {BLUE};
    padding: 4px 8px;
    font-size: 10px;
}}
QPushButton#BassButton:hover {{
    background: rgba(79, 157, 255, 0.14);
    border-color: rgba(112, 180, 255, 0.62);
    color: #b9dcff;
}}
QPushButton#BassButton:checked {{
    background: rgba(79, 157, 255, 0.20);
    border-color: {BLUE};
}}
QWidget#BassHeader {{
    background: #0e141c;
    border-top: 1px solid {LINE_SOFT};
    border-bottom: 1px solid {LINE_SOFT};
}}
QWidget#BassPanel {{
    background: #0a0e13;
    border-top: 1px solid {LINE_SOFT};
}}
QPushButton#Outline {{
    background: transparent;
    border: 1px solid {LINE};
    border-radius: 5px;
    color: {TEXT_SOFT};
    padding: 6px 12px;
}}
QPushButton#Outline:hover {{
    border-color: {ACCENT_DARK};
    color: {TEXT};
    background: rgba(240, 162, 74, 0.05);
}}
QPushButton#Segment {{
    background: {PANEL_SOFT};
    border: 1px solid {LINE};
    border-radius: 4px;
    color: {MUTED};
    padding: 5px 6px;
    font-size: 10px;
}}
QPushButton#Segment:hover {{
    color: {TEXT};
    border-color: #3a424c;
}}
QPushButton#Segment:checked {{
    background: rgba(240, 162, 74, 0.12);
    border-color: rgba(240, 162, 74, 0.55);
    color: {ACCENT_BRIGHT};
    font-weight: 600;
}}
QPushButton#DangerGhost {{
    background: transparent;
    border: 1px solid transparent;
    color: {MUTED};
    padding: 4px 8px;
}}
QPushButton#DangerGhost:hover {{
    color: {RED};
    border-color: rgba(255, 116, 111, 0.28);
    background: rgba(255, 116, 111, 0.06);
}}

QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px;
}}
QToolButton:hover {{
    background: {PANEL_SOFT};
    border-color: {LINE_SOFT};
}}
QToolButton:pressed {{
    background: #0c0e11;
}}
QToolButton:checked {{
    background: rgba(240, 162, 74, 0.10);
    border-color: rgba(240, 162, 74, 0.25);
}}

QComboBox {{
    background: {PANEL_SOFT};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 5px 9px;
    color: {TEXT_SOFT};
    combobox-popup: 0;
    min-height: 17px;
}}
QComboBox:hover {{
    border-color: #3a424c;
    color: {TEXT};
}}
QComboBox:focus {{
    border-color: {ACCENT_DARK};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: url("{CHEVRON_SVG}");
    width: 10px;
    height: 10px;
    margin-right: 6px;
}}
QDoubleSpinBox {{
    background: {PANEL_SOFT};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 4px 6px;
    color: {TEXT_SOFT};
    font-family: {MONO_FONT};
    font-size: 10px;
}}
QDoubleSpinBox:hover {{
    border-color: #3a424c;
    color: {TEXT};
}}
QDoubleSpinBox:focus {{
    border-color: {ACCENT_DARK};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 12px;
    background: transparent;
    border: none;
}}
QDoubleSpinBox::up-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {MUTED};
}}
QDoubleSpinBox::down-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {MUTED};
}}
QComboBox QAbstractItemView {{
    background: {PANEL_RAISED};
    color: {TEXT_SOFT};
    border: 1px solid {LINE};
    border-radius: 4px;
    selection-background-color: rgba(240, 162, 74, 0.16);
    selection-color: {TEXT};
    outline: none;
    padding: 3px;
}}

QSlider::groove:horizontal {{
    height: 3px;
    background: #2c323a;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 11px;
    margin: -5px 0;
    border-radius: 5px;
    background: #f0f1f2;
    border: 1px solid #0c0e11;
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_BRIGHT};
}}
QSlider::groove:horizontal:disabled {{
    background: #23272e;
}}
QSlider::sub-page:horizontal:disabled {{
    background: #4b4135;
}}
QSlider::handle:horizontal:disabled {{
    background: #5a5f66;
}}

QCheckBox {{
    spacing: 8px;
    color: {TEXT_SOFT};
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {LINE};
    border-radius: 3px;
    background: {PANEL_SOFT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT_BRIGHT};
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    width: 8px;
    background: transparent;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #343b44;
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: #4a525e;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    height: 8px;
    background: transparent;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #343b44;
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

QSplitter::handle {{
    background: {LINE_SOFT};
}}
QSplitter::handle:vertical {{
    height: 1px;
}}
QSplitter::handle:hover {{
    background: {ACCENT_DARK};
}}
QMenu {{
    background: {PANEL_RAISED};
    border: 1px solid {LINE};
    color: {TEXT_SOFT};
    padding: 5px;
}}
QMenu::item {{
    padding: 6px 22px 6px 12px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background: rgba(240, 162, 74, 0.12);
    color: {TEXT};
}}
QToolTip {{
    background: #1c2128;
    color: {TEXT};
    border: 1px solid {LINE};
    padding: 5px 7px;
}}
QProgressBar {{
    background: #252a31;
    border: none;
    border-radius: 2px;
    max-height: 3px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 2px;
}}
QMessageBox {{
    background: {PANEL};
}}
QMessageBox QLabel {{
    color: {TEXT};
}}
"""
