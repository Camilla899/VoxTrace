"""Generate a short synthetic electric-bass phrase for Bass-to-MIDI testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "voxtrace_bass_demo.wav"
SR = 44100
BPM = 92.0
BEAT = 60.0 / BPM
NOTES = [
    (0.0, 1.5, 33), (1.5, 0.5, 33), (2.0, 1.0, 36), (3.0, 1.0, 40),
    (4.0, 1.5, 38), (5.5, 0.5, 36), (6.0, 1.0, 33), (7.0, 1.0, 36),
    (8.0, 1.5, 31), (9.5, 0.5, 33), (10.0, 1.0, 36), (11.0, 1.0, 38),
    (12.0, 1.5, 33), (13.5, 0.5, 36), (14.0, 2.0, 40), (16.0, 2.0, 33),
    (18.0, 1.0, 36), (19.0, 1.0, 38), (20.0, 1.5, 33), (21.5, 2.0, 33),
]


def midi_hz(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def main() -> None:
    duration = 24.0 * BEAT + 1.0
    total = int(duration * SR)
    out = np.zeros(total, dtype=np.float64)
    rng = np.random.default_rng(42)
    for start_beat, length_beats, pitch in NOTES:
        start = int(start_beat * BEAT * SR)
        length = int(length_beats * BEAT * SR)
        t = np.arange(length) / SR
        f0 = midi_hz(pitch)
        # Electric bass: fundamental plus a few decaying harmonics and a pluck.
        tone = np.zeros(length)
        for harmonic in range(1, 7):
            tone += (1.0 / harmonic**1.35) * np.sin(2 * np.pi * f0 * harmonic * t + rng.uniform(0, 0.3))
        tone += 0.04 * np.sin(2 * np.pi * 2 * f0 * t)
        attack = np.exp(-t / 0.035) * 0.35
        body = np.exp(-t / 1.1)
        env = np.clip(attack + body, 0, 1)
        click = rng.normal(0, 1, min(length, int(0.012 * SR)))
        click = sosfilt(butter(2, [300, 1800], btype="bandpass", fs=SR, output="sos"), click)
        tone[: len(click)] += click * np.linspace(0.08, 0, len(click))
        tone *= env
        end = min(total, start + length)
        out[start:end] += tone[: end - start]
    sos = butter(4, 2200, btype="lowpass", fs=SR, output="sos")
    out = sosfilt(sos, out)
    out /= max(1e-9, np.max(np.abs(out)))
    out *= 0.82
    fade = int(0.05 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    sf.write(OUT, out.astype(np.float32), SR, subtype="PCM_16")
    print(OUT)
    print(f"{len(out)/SR:.2f}s")


if __name__ == "__main__":
    main()
