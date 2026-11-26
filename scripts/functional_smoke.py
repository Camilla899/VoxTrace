"""Functional Qt smoke test: precision switch, playback, export, undo."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication, QFileDialog

from voxtrace.ui import theme
from voxtrace.ui.main_window import MainWindow

_APP = None


def wait_for(predicate, timeout_ms=30000, step=80, label=""):
    elapsed = 0
    while not predicate() and elapsed < timeout_ms:
        time.sleep(step / 1000.0)
        _APP.processEvents()
        elapsed += step
    if not predicate():
        raise TimeoutError(f"Timed out waiting for {label}")
    return True


def main() -> int:
    global _APP
    app = QApplication(sys.argv[:1])
    _APP = app
    app.setStyle("Fusion")
    app.setFont(theme.app_font(9))
    app.setStyleSheet(theme.STYLESHEET)
    window = MainWindow(auto_demo=True)
    window.show()
    wait_for(lambda: window.result is not None, 40000, label="initial analysis")
    print("initial", len(window.result.notes), window.result.precision, window.result.pitch_source)

    # Precision switch -> re-analysis.
    window.inspector.set_precision("precise")
    wait_for(lambda: not window._busy, 5000, label="unbusy")
    window.start_analysis()
    wait_for(lambda: window.result is not None and window.result.precision == "precise" and not window._busy, 40000, label="precise analysis")
    print("precise", len(window.result.notes), window.result.precision)

    # Breath gate must not require an analysis, and the sidebar shortcut must
    # stay in sync with the inspector control.
    before = len(window.notes)
    window.inspector.breath_toggle.setChecked(False)
    wait_for(lambda: not window._breath_timer.isActive(), 3000, label="breath filter")
    if window.sidebar.breath_switch.isChecked():
        raise RuntimeError("sidebar breath switch did not sync")
    print("breath-toggle off", before, "->", len(window.notes))
    window.sidebar.breath_switch.setChecked(True)
    wait_for(lambda: not window._breath_timer.isActive(), 3000, label="breath filter")
    if not window.inspector.breath_toggle.isChecked():
        raise RuntimeError("inspector breath switch did not sync")
    if not window.sidebar.export_btn.isEnabled():
        raise RuntimeError("sidebar export action not enabled")

    # Manual BPM correction updates the shared timeline grid.
    window.bpm_spin.setValue(97.0)
    app.processEvents()
    if abs(window.view.bpm - 97.0) > 0.01 or abs(window.result.key.bpm - 97.0) > 0.01:
        raise RuntimeError("BPM override did not reach timeline and result")

    # Playback position advances.
    window.seek_to(1.0)
    window.toggle_play()
    time.sleep(0.9)
    app.processEvents()
    state = window.player.playbackState()
    pos = window.player.position()
    window.player.pause()
    print("playback", state, pos)
    if state != QMediaPlayer.PlaybackState.PlayingState or pos < 300:
        raise RuntimeError("Playback did not advance")

    # Export through a stubbed file dialog.
    target = Path(tempfile.gettempdir()) / "voxtrace_functional.mid"
    target.unlink(missing_ok=True)
    original = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(target), "MIDI 文件 (*.mid)"))
    try:
        window.export_midi()
    finally:
        QFileDialog.getSaveFileName = original
    print("export", target.exists(), target.stat().st_size if target.exists() else 0)
    if not target.exists() or target.stat().st_size < 64:
        raise RuntimeError("MIDI export failed")

    # Note edit and undo.
    changed = [n.copy() for n in window.notes]
    original_pitch = changed[0].pitch
    changed[0].pitch = max(21, changed[0].pitch - 1)
    window._notes_edited(changed, "smoke test")
    print("edit", original_pitch, "->", changed[0].pitch, "result", window.notes[0].pitch)
    if window.notes[0].pitch != changed[0].pitch:
        raise RuntimeError("note edit did not stick")
    window.undo()
    print("undo", len(window.notes), window.notes[0].pitch)

    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
