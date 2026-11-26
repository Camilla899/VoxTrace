"""Background analysis thread."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from ..audio_io import AudioFile
from ..engine import AnalysisCancelled, analyze_audio
from ..notes import AnalysisSettings


class AnalysisWorker(QThread):
    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, audio: AudioFile, settings: AnalysisSettings, parent=None):
        super().__init__(parent)
        self.audio = audio
        self.settings = settings
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result = analyze_audio(
                self.audio,
                self.settings,
                progress=lambda value, message: self.progress.emit(int(value), str(message)),
                cancel=lambda: self._cancel,
                workers=self.settings.workers,
            )
            if not self._cancel:
                self.completed.emit(result)
        except AnalysisCancelled:
            return
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail or str(exc))
