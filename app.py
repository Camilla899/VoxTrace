"""VoxTrace Studio launcher."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

# Make sibling imports work when launched from any cwd.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from voxtrace import APP_NAME, __version__
from voxtrace.ui import theme
from voxtrace.ui.main_window import MainWindow


def _log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "VoxTrace"
    base.mkdir(parents=True, exist_ok=True)
    return base / "voxtrace.log"


def _install_exception_handler(app: QApplication) -> None:
    def hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.error("Unhandled exception:\n%s", text)
        QMessageBox.critical(None, "VoxTrace 发生错误", f"{exc_value}\n\n详细信息已写入日志：\n{_log_path()}")

    sys.excepthook = hook


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VoxTrace.Studio.1")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} {__version__} — 人声转 MIDI 工作站")
    parser.add_argument("audio", nargs="?", help="启动时打开的音频文件")
    parser.add_argument("--empty", action="store_true", help="不加载示例音频，显示空白工作区")
    parser.add_argument("--no-demo", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_self_test() -> int:
    """Packaged-build verification: decode + analyze the demo without a GUI."""
    from voxtrace.audio_io import load_audio
    from voxtrace.engine import analyze_audio
    from voxtrace.midi_io import read_midi_notes
    from voxtrace.notes import AnalysisSettings

    base = Path(getattr(sys, "_MEIPASS", ROOT))
    demo = base / "demo" / "voxtrace_demo.wav"
    bass_demo_audio = base / "demo" / "voxtrace_bass_demo.wav"
    bass_demo = base / "demo" / "voxtrace_demo_bass.mid"
    report = {"ok": False, "demo": str(demo), "frozen": bool(getattr(sys, "frozen", False))}
    try:
        if not demo.exists():
            raise FileNotFoundError(str(demo))
        audio = load_audio(demo)
        result = analyze_audio(audio, AnalysisSettings(), workers=4)
        bass_data = read_midi_notes(bass_demo)
        bass_audio = load_audio(bass_demo_audio)
        bass_settings = AnalysisSettings(
            engine="bass",
            precision="balanced",
            min_midi=28,
            max_midi=55,
            min_note_ms=70,
            ignore_breath=False,
            noise_gate=0.2,
        )
        bass_result = analyze_audio(bass_audio, bass_settings, workers=4)
        report.update(
            {
                "ok": True,
                "duration": round(audio.duration, 2),
                "notes": len(result.notes),
                "key": result.key.name,
                "bpm": round(result.key.bpm, 1),
                "pitch_source": result.pitch_source,
                "bass_midi_notes": len(bass_data.all_notes),
                "bass_midi_bpm": round(bass_data.bpm, 1),
                "bass_mode_notes": len(bass_result.notes),
                "bass_mode_pitch": bass_result.pitch_source,
                "elapsed": round(result.elapsed, 2),
                "bass_elapsed": round(bass_result.elapsed, 2),
            }
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("Self-test failed")
    targets = [
        _log_path().with_name("selftest.json"),
        Path(tempfile.gettempdir()) / "VoxTrace_selftest.json",
        Path(getattr(sys, "_MEIPASS", ROOT)) / "VoxTrace_selftest.json",
    ]
    written = None
    for target in targets:
        try:
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            written = target
            break
        except OSError:
            continue
    report["report_path"] = str(written) if written else ""
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def _log_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = []
    target = _log_path()
    try:
        handlers.append(logging.FileHandler(target, encoding="utf-8"))
    except OSError:
        # A source build may already hold the default log file open. Keep the
        # packaged app launchable instead of dying before the UI appears.
        try:
            handlers.append(logging.FileHandler(target.with_name(f"voxtrace-{os.getpid()}.log"), encoding="utf-8"))
        except OSError:
            pass
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    return handlers


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=_log_handlers(),
    )
    # Basic Pitch emits optional-backend warnings at import time; those are not
    # relevant when using its bundled ONNX model.
    logging.getLogger().setLevel(logging.ERROR)

    args = parse_args()
    _set_windows_app_id()

    if args.self_test:
        return _run_self_test()

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("VoxTrace")
    app.setStyle("Fusion")
    app.setFont(theme.app_font(9))
    app.setStyleSheet(theme.STYLESHEET)
    _install_exception_handler(app)

    audio_path = None
    if args.audio:
        candidate = Path(args.audio).expanduser()
        if candidate.exists():
            audio_path = candidate
        else:
            logging.warning("Audio path not found: %s", candidate)

    auto_demo = not (args.empty or args.no_demo or audio_path is not None)
    window = MainWindow(auto_demo=auto_demo)
    window.show()
    if audio_path is not None:
        window.load_audio_path(audio_path)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
