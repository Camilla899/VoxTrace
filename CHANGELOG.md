# Changelog

## Source maintenance — 2026-11-26

- Removed leftover debug screenshots and one-off UI/scroll test scripts.
- Renamed demo generators to make their purpose explicit; added
  `scripts/README.md` and a single `capture_screenshots.py` entry point.
- Added `docs/DEVELOPMENT.md`, this changelog and a complete `.gitignore`.
- Fixed `requirements-dev.txt` to include PyInstaller.
- Fixed `run.bat`'s stale `rmvpe_onnx` dependency check and removed the
  duplicate Mido install from `setup_env.bat`.
- Removed unused imports and dead local variables across the package.
- No runtime behavior changes beyond the 1.1.3 scroll release.

## 1.1.3 — 2026-11-26

- Piano roll supports vertical pitch scrolling with the mouse wheel.
- `PageUp` / `PageDown` move the pitch window by one octave.
- `Shift+wheel` pans the timeline; `Ctrl+Shift+wheel` zooms the pitch range.
- `Ctrl+0` / “显示全部” now fits both time and pitch.
- Editing, undo and noise-gate refiltering preserve the current pitch window.

## 1.1.2 — 2026-11-26

- Fixed vocal-only text left in the empty Bass workspace.
- Empty Bass state now says “Bass 音频”, “独立低音工作区” and “低音音符、调性与 BPM”.
- Bass file header and status now identify the Bass workspace.

## 1.1.1 — 2026-11-26

- Vocal and Bass workflows now keep separate audio files, results, notes,
  undo history, BPM, viewport and playhead.
- Switching workflows no longer reuses or copies the same audio file.

## 1.1.0 — 2026-11-26

- Added a dedicated Bass-to-MIDI workflow.
- Bass pipeline: Basic Pitch notes + low-register pYIN validation, octave
  correction, monophonic cleanup, E1–G3 default range.
- Added Bass presets, blue Bass piano roll, separate `_Bass` MIDI export.
- Added optional external Bass MIDI reference lane in the vocal workspace.
- Added synthetic Bass demo and Bass engine tests.

## 1.0.0 — 2026-11-26

- Initial portable Windows release.
- Vocal-to-MIDI: Basic Pitch + RMVPE, pYIN fallback, breath detection,
  noise gate, key/BPM detection, editable piano roll, MIDI export.
- Portable PyInstaller build with bundled models and offline inference.
