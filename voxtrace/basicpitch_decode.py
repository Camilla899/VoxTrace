"""Basic Pitch note decoder, vendored for a pure ONNX Runtime deployment.

Adapted from Spotify's Basic Pitch (Apache-2.0):
https://github.com/spotify/basic-pitch
Copyright 2022 Spotify AB.

Only the post-processing needed by VoxTrace is kept here.  The model itself is
distributed separately in ``models/basic_pitch.onnx`` under the same license.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import argrelmax

AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
AUDIO_N_SAMPLES = 22050 * 2 - 256
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOT_N_FRAMES = ANNOTATIONS_FPS * 2
MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
INFERENCE_OVERLAP_FRAMES = 30


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(1e-9, float(hz)) / 440.0)


def get_infered_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1]), dtype=frames.dtype), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    peak = float(np.max(frame_diff))
    if peak > 0 and np.max(onsets) > 0:
        frame_diff = np.max(onsets) * frame_diff / peak
    else:
        frame_diff[:] = 0
    return np.maximum(onsets, frame_diff)


def constrain_frequency(
    onsets: np.ndarray,
    frames: np.ndarray,
    max_freq: Optional[float],
    min_freq: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if max_freq is not None:
        max_freq_idx = int(np.round(hz_to_midi(max_freq) - MIDI_OFFSET))
        if 0 <= max_freq_idx < frames.shape[1]:
            onsets[:, max_freq_idx:] = 0
            frames[:, max_freq_idx:] = 0
    if min_freq is not None:
        min_freq_idx = int(np.round(hz_to_midi(min_freq) - MIDI_OFFSET))
        if 0 < min_freq_idx < frames.shape[1]:
            onsets[:, :min_freq_idx] = 0
            frames[:, :min_freq_idx] = 0
    return onsets, frames


def model_frames_to_time(n_frames: int) -> np.ndarray:
    original_times = np.arange(n_frames, dtype=np.float64) * FFT_HOP / AUDIO_SAMPLE_RATE
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + 0.0018
    return original_times - (window_offset * window_numbers)


def output_to_notes_polyphonic(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    max_freq: Optional[float],
    min_freq: Optional[float],
    melodia_trick: bool = True,
    energy_tol: int = 11,
) -> List[Tuple[int, int, int, float]]:
    """Decode note/frame activations into (start_frame, end_frame, midi, amplitude)."""
    n_frames = frames.shape[0]
    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]
    onset_freq_idx = onset_idx[1][::-1]

    remaining_energy = frames.copy()
    note_events: List[Tuple[int, int, int, float]] = []

    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        if note_start_idx >= n_frames - 1:
            continue
        i = note_start_idx + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1
        i -= k
        if i - note_start_idx <= min_note_len:
            continue
        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0
        amplitude = float(np.mean(frames[note_start_idx:i, freq_idx]))
        note_events.append((note_start_idx, i, freq_idx + MIDI_OFFSET, amplitude))

    if melodia_trick:
        energy_shape = remaining_energy.shape
        while float(np.max(remaining_energy)) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0
            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i += 1
            i_end = i - 1 - k
            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i -= 1
            i_start = i + 1 + k
            if i_end - i_start <= min_note_len:
                continue
            amplitude = float(np.mean(frames[i_start:i_end, freq_idx]))
            note_events.append((i_start, i_end, freq_idx + MIDI_OFFSET, amplitude))

    return note_events


def decode_model_output(
    output: dict,
    onset_threshold: float,
    frame_threshold: float,
    min_note_ms: int,
    min_midi: int,
    max_midi: int,
) -> List[Tuple[float, float, int, float]]:
    frames = output["note"]
    onsets = output["onset"]
    min_frames = max(2, int(round(min_note_ms / 1000.0 * ANNOTATIONS_FPS)))
    events = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=float(onset_threshold),
        frame_thresh=float(frame_threshold),
        min_note_len=min_frames,
        infer_onsets=True,
        max_freq=440.0 * 2.0 ** ((max_midi - 69.0) / 12.0),
        min_freq=440.0 * 2.0 ** ((min_midi - 69.0) / 12.0),
        melodia_trick=True,
    )
    times = model_frames_to_time(frames.shape[0])
    decoded = []
    for start, end, pitch, amplitude in events:
        if end <= start or start >= len(times) or end >= len(times):
            continue
        decoded.append((float(times[start]), float(times[end]), int(pitch), float(amplitude)))
    return decoded
