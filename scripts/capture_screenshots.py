"""Regenerate the documentation screenshots with the bundled demos.

Run from the repository root or from this directory:

    python scripts/capture_screenshots.py

Outputs:
    docs/screenshot.png        Vocal workspace with a finished analysis
    docs/screenshot_bass.png   Bass workspace with a finished analysis
    docs/empty_bass.png        Empty Bass workspace
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from voxtrace.audio_io import load_audio
from voxtrace.engine import analyze_audio
from voxtrace.notes import AnalysisSettings
from voxtrace.ui import theme
from voxtrace.ui.main_window import MainWindow

DOCS = ROOT / "docs"
DEMO = ROOT / "demo" / "voxtrace_demo.wav"
BASS_DEMO = ROOT / "demo" / "voxtrace_bass_demo.wav"


def _analyze(window: MainWindow, path: Path, settings: AnalysisSettings, playhead=0.0) -> None:
    audio = load_audio(path)
    result = analyze_audio(audio, settings, workers=8)
    window._on_analysis_done(result)
    QApplication.processEvents()
    if hasattr(window, "_toast"):
        window._toast.hide()
    if playhead:
        window.view.set_playhead(playhead)
    QApplication.processEvents()


def main() -> None:
    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(theme.app_font(9))
    app.setStyleSheet(theme.STYLESHEET)
    DOCS.mkdir(exist_ok=True)

    # Empty Bass workspace: verifies that no vocal copy leaks into this mode.
    window = MainWindow(auto_demo=False)
    window.resize(1480, 940)
    window.show()
    QApplication.processEvents()
    window._switch_mode("bass")
    QApplication.processEvents()
    window._toast.hide()
    window.grab().save(str(DOCS / "empty_bass.png"))
    window.close()

    # Vocal workspace.
    window = MainWindow(auto_demo=False)
    window.resize(1480, 940)
    window.show()
    QApplication.processEvents()
    window.load_audio_path(DEMO, auto_analyze=False)
    _analyze(window, DEMO, AnalysisSettings(), playhead=5.0)
    window.grab().save(str(DOCS / "screenshot.png"))
    window.close()

    # Bass workspace.
    window = MainWindow(auto_demo=False)
    window.resize(1480, 940)
    window.show()
    QApplication.processEvents()
    window._switch_mode("bass")
    window.load_audio_path(BASS_DEMO, auto_analyze=False)
    _analyze(
        window,
        BASS_DEMO,
        AnalysisSettings(
            engine="bass",
            precision="balanced",
            min_midi=28,
            max_midi=55,
            min_note_ms=70,
            ignore_breath=False,
            noise_gate=0.2,
        ),
    )
    window.grab().save(str(DOCS / "screenshot_bass.png"))
    window.close()
    print(f"Wrote screenshots to {DOCS}")


if __name__ == "__main__":
    main()
