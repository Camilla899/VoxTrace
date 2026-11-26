# Development scripts

These helpers are not needed to run the app. They generate the bundled demo
assets, refresh documentation images, and exercise the Qt UI end to end.

| Script | Purpose |
|---|---|
| `make_vocal_demo.py` | Rebuild `demo/voxtrace_demo.wav` (synthetic vocal with breaths) |
| `make_bass_audio_demo.py` | Rebuild `demo/voxtrace_bass_demo.wav` (synthetic electric bass) |
| `make_bass_reference_midi.py` | Rebuild `demo/voxtrace_demo_bass.mid` (external Bass MIDI reference) |
| `make_icon.py` | Rebuild the application icon from the source drawing |
| `capture_screenshots.py` | Rebuild the three screenshots in `docs/` |
| `functional_smoke.py` | Qt smoke test: analysis, precision switch, playback, export, undo |

Run them from the repository root with the project environment active:

```text
python scripts/capture_screenshots.py
python scripts/functional_smoke.py
```

`capture_screenshots.py` uses private window methods to load a result
synchronously; it is intentionally a documentation tool, not an API example.
