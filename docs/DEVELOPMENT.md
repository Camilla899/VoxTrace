# Development guide

## Source layout

```text
VoxTrace/
├─ app.py                    launcher, logging, --self-test
├─ voxtrace/
│  ├─ audio_io.py            decode, resample, temporary playback WAV
│  ├─ basicpitch_decode.py   vendored Apache-2.0 note decoder
│  ├─ engine.py              parallel inference and note post-processing
│  ├─ midi_io.py             external Bass MIDI parsing for the reference lane
│  ├─ notes.py               note/key data model and Standard MIDI writer
│  ├─ rmvpe.py               vendored RMVPE ONNX estimator
│  └─ ui/                    PySide6 window, widgets, theme, worker
├─ models/                   basic_pitch.onnx + rmvpe.onnx
├─ demo/                     synthetic vocal/Bass examples + public-domain audio
├─ tests/                    unittest suite
├─ scripts/                  demo generation, screenshots, UI smoke test
├─ docs/                     screenshots and this guide
├─ licenses/                 third-party license texts
├─ build_exe.ps1             portable PyInstaller build
└─ requirements*.txt
```

## Two isolated workflows

`MainWindow` owns one `ModeSession` per workflow (`vocal`, `bass`). A session
contains its own:

- `AudioFile`
- `AnalysisResult` and editable notes
- undo/redo lists
- BPM, viewport and playhead
- player source/position

`_activate_session()` swaps these values into the active window. Loading a new
file only resets the active session. Switching modes must never copy audio or
results to the other session.

## Analysis pipelines

**Vocal** (`AnalysisSettings.engine = "vocal"`):

1. Parallel Basic Pitch windows produce note/onset candidates.
2. RMVPE supplies a neural F0 track (standard/precise); pYIN is the fast and
   fallback tracker.
3. Note cleanup: legato merge, octave-ghost suppression, containment removal,
   pitch refinement, monophonic enforcement.
4. Breath regions and spectral features are detected.
5. Noise gate, key profile and tempo refinement run last.

**Bass** (`engine = "bass"`):

1. Basic Pitch runs over the bass register.
2. A 4096-point pYIN track (`fmin` about 25 Hz) validates the fundamental.
3. Notes are snapped to/removed by pYIN support, octave errors are corrected,
   and the result is forced monophonic.
4. No RMVPE or breath filtering is used; the noise gate still applies.

**Mono** (`engine = "mono"`): pYIN-only note segmentation for clean monophonic
input.

## UI extension points

- `voxtrace/ui/piano_roll.py` — editable notes, pitch window, wheel handling.
- `voxtrace/ui/scope.py` — waveform/spectrogram; empty-state text is per-mode.
- `voxtrace/ui/inspector.py` — engine parameters and mode-specific controls.
- `voxtrace/ui/sidebar.py` — workflow entries and presets.
- `voxtrace/ui/theme.py` — colors and Qt stylesheet.

The piano roll keeps a visible pitch window (`_low`/`_high`). `set_notes(...,
fit_range=False)` must be used after edits/filtering so manual vertical
scrolling is not reset; a new analysis uses the default `fit_range=True`.

## Tests

```text
python tests/run_tests.py
python scripts/functional_smoke.py
```

The unit suite covers MIDI writing, note cleanup, key signatures, breath gate,
noise gate, tempo refinement, vocal integration, Bass MIDI parsing and the Bass
pipeline. The Qt smoke test covers threaded analysis, precision switching,
playback, export and undo.

## Build

Install dependencies and PyInstaller:

```text
python -m pip install -r requirements-dev.txt
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

The build is a portable `onedir` app; `dist/VoxTrace/VoxTrace.exe` requires the
adjacent `_internal` directory. `--self-test` verifies the frozen app without
opening the GUI and exercises both workflows.

## Conventions

- Keep model paths resolved from `sys._MEIPASS` when frozen.
- Never assume `__file__` is usable for data paths in the frozen app.
- Do not add network calls to the analysis path.
- Run the full test suite after touching `engine.py` or the frozen-app setup.
- Keep mode-specific UI text in mode branches; no “人声” labels in Bass mode.
