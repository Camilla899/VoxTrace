"""Generate a royalty-free synthesized vocal demo for VoxTrace.

The signal is a formant-synthesized legato voice with vibrato, consonant
noise and audible breath tails, useful for exercising the full analysis UI.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100
BPM = 92.0
BEAT = 60.0 / BPM


def midi_hz(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


FORMANTS = {
    "a": [(800, 90, 1.0), (1150, 110, 0.62), (2800, 140, 0.26), (3800, 180, 0.12)],
    "e": [(500, 80, 1.0), (1900, 120, 0.55), (2600, 140, 0.3), (3600, 180, 0.12)],
    "i": [(300, 70, 1.0), (2300, 120, 0.5), (3000, 150, 0.25), (3900, 190, 0.1)],
    "o": [(520, 80, 1.0), (900, 100, 0.65), (2500, 140, 0.22), (3400, 180, 0.1)],
    "u": [(350, 70, 1.0), (720, 90, 0.5), (2400, 120, 0.15), (3300, 160, 0.08)],
}


def formant_gain(freqs: np.ndarray, vowel: str) -> np.ndarray:
    gain = np.zeros_like(freqs)
    for center, width, level in FORMANTS[vowel]:
        gain += level * np.exp(-0.5 * ((freqs - center) / (width * 1.8)) ** 2)
    return gain + 0.012


def synth_note(f0: float, duration: float, vowel: str, rng: np.random.Generator) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    # Gentle scoop + delayed vibrato, like a sustained sung note.
    scoop = 0.55 * np.exp(-t / 0.09) * (1 if f0 < 330 else 0.6)
    vibrato_depth = 0.16 + 0.11 * np.clip((t - 0.18) / 0.5, 0, 1)
    vib = vibrato_depth * np.sin(2 * np.pi * 5.1 * t + 0.3) + 0.025 * np.sin(2 * np.pi * 7.2 * t)
    phase = 2 * np.pi * np.cumsum(f0 * 2 ** ((scoop + vib) / 12)) / SR
    harmonics = np.arange(1, 36)
    freqs = f0 * harmonics
    gains = formant_gain(freqs, vowel) / harmonics**0.92
    gains *= np.exp(-freqs / 7200.0)
    signal = np.zeros(n, dtype=np.float64)
    for harmonic, gain in zip(harmonics, gains):
        signal += gain * np.sin(harmonic * phase + rng.uniform(0, 0.35))
    # Tiny turbulent component keeps the voice from sounding like a pure synth.
    noise = rng.normal(0, 1, n)
    sos = butter(2, [900, 6000], btype="bandpass", fs=SR, output="sos")
    signal += 0.028 * sosfilt(sos, noise)
    attack = 0.035
    release = 0.075
    env = np.ones(n)
    a = max(2, int(attack * SR))
    env[:a] = np.linspace(0, 1, a) ** 0.7
    r = max(2, int(release * SR))
    env[-r:] *= np.linspace(1, 0, r) ** 1.5
    # Slow dynamic arch.
    env *= 0.82 + 0.18 * np.sin(np.pi * np.clip(t / max(duration, 1e-3), 0, 1))
    return signal * env


def breath(duration: float, rng: np.random.Generator, strength: float = 0.55) -> np.ndarray:
    n = int(duration * SR)
    noise = rng.normal(0, 1, n)
    sos = butter(3, [1400, 8500], btype="bandpass", fs=SR, output="sos")
    shaped = sosfilt(sos, noise)
    env = np.sin(np.linspace(0, np.pi, n)) ** 1.6
    return strength * shaped * env


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "demo" / "voxtrace_demo.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261125)

    # A-minor phrase, deliberately including a low chest note, a leap and a
    # soft breathy pickup.  (midi, beats, vowel)
    phrase = [
        (69, 1.0, "a"), (72, 0.75, "a"), (76, 1.25, "e"), (74, 0.75, "o"),
        (72, 0.5, "a"), (69, 0.75, "u"), (67, 0.75, "a"), (69, 1.5, "a"),
        (64, 0.75, "o"), (67, 0.75, "e"), (69, 2.0, "a"),
    ]
    phrase_b = [
        (72, 0.5, "e"), (74, 0.5, "e"), (76, 1.0, "a"), (79, 0.75, "i"),
        (77, 0.75, "a"), (76, 0.5, "o"), (74, 0.75, "a"), (72, 0.5, "u"),
        (69, 1.75, "a"), (67, 0.5, "o"), (69, 1.25, "a"),
    ]

    total_beats = sum(x[1] for x in phrase) + sum(x[1] for x in phrase_b) + 8
    total = int((total_beats * BEAT + 2.5) * SR)
    mix = np.zeros(total, dtype=np.float64)
    cursor = 1.3

    def render_phrase(notes):
        nonlocal cursor
        # Audible inhale before every phrase.
        b = breath(0.42, rng, 0.32)
        start = int((cursor - 0.5) * SR)
        mix[start : start + len(b)] += b
        for idx, (midi, beats, vowel) in enumerate(notes):
            dur = beats * BEAT * 0.93
            note = synth_note(midi_hz(midi), dur, vowel, rng)
            # consonant transient
            click_n = int(0.035 * SR)
            click = rng.normal(0, 1, click_n) * np.linspace(0.05, 0, click_n)
            csos = butter(2, [2500, 8000], btype="bandpass", fs=SR, output="sos")
            click = sosfilt(csos, click)
            note[:click_n] += click * 0.35
            start = int(cursor * SR)
            mix[start : start + len(note)] += note
            cursor += beats * BEAT
            # small breath / phrase tail sometimes
            if idx in {3, 7}:
                gap = 0.2
                b = breath(gap, rng, 0.11)
                s = int(cursor * SR)
                mix[s : s + len(b)] += b
                cursor += gap

    render_phrase(phrase)
    cursor += 0.65
    render_phrase(phrase_b)
    cursor += 0.8

    # Soft airy tail.
    tail = breath(1.4, rng, 0.12)
    mix[int(cursor * SR) : int(cursor * SR) + len(tail)] += tail

    # Light plate reverb, tasteful not washy.
    ir_len = int(0.9 * SR)
    ir_t = np.arange(ir_len) / SR
    ir = rng.normal(0, 1, ir_len) * np.exp(-ir_t / 0.28)
    ir[0] = 0
    ir /= np.max(np.abs(ir)) + 1e-9
    wet = np.convolve(mix, ir, mode="full")[: len(mix)]
    mix = mix + 0.12 * wet

    # Remove DC, normalize, and add a very light fade.
    sos = butter(2, 55, btype="highpass", fs=SR, output="sos")
    mix = sosfilt(sos, mix)
    mix /= max(1e-9, np.max(np.abs(mix)))
    mix *= 0.88
    fade = int(0.08 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)
    sf.write(out, mix.astype(np.float32), SR, subtype="PCM_16")
    print(out)
    print(f"{len(mix) / SR:.2f}s, expected {len(phrase) + len(phrase_b)} sung notes")


if __name__ == "__main__":
    main()
