"""Multi-core vocal and Bass analysis engine.

The transcription front-end is Spotify's Basic Pitch ONNX model, run across
several CPU cores.  RMVPE/pYIN supply continuous pitch tracks used for breath
classification, Bass octave correction and monophonic note segmentation.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_closing, median_filter

from .audio_io import ANALYSIS_SR, AudioFile
from .notes import (
    AnalysisResult,
    AnalysisSettings,
    BreathRegion,
    KeyInfo,
    NoteEvent,
    hz_to_midi,
    merge_legato,
    midi_to_hz,
    suppress_octave_ghosts,
)

ProgressFn = Callable[[int, str], None]
CancelFn = Callable[[], bool]

_NUMBA_FROZEN_PATCHED = False


def _prepare_frozen_numba() -> None:
    """Numba's cache=True cannot locate sources inside a PyInstaller archive.

    Librosa decorates several helpers with ``numba.jit(..., cache=True)``.
    Disabling only the disk cache keeps JIT speed while allowing frozen apps to
    import and run librosa normally.
    """
    global _NUMBA_FROZEN_PATCHED
    if _NUMBA_FROZEN_PATCHED:
        return
    _NUMBA_FROZEN_PATCHED = True
    if not getattr(sys, "frozen", False):
        return
    try:
        import numba
        import numba.core.decorators as decorators

        def without_cache(original):
            def wrapper(*args, **kwargs):
                kwargs.pop("cache", None)
                return original(*args, **kwargs)

            return wrapper

        # Librosa uses jit, njit, guvectorize and vectorize together. All of
        # them default to cache=True in places; strip that flag before librosa
        # is imported so numba never asks PyInstaller's archive for a source
        # locator.
        for module in (numba, decorators):
            for name in ("jit", "njit", "guvectorize", "vectorize"):
                original = getattr(module, name, None)
                if original is not None:
                    setattr(module, name, without_cache(original))
    except Exception:
        pass


class AnalysisCancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# Basic Pitch neural front-end
# ---------------------------------------------------------------------------

_BP_CONST = {
    "sr": 22050,
    "fft_hop": 256,
    "audio_n_samples": 22050 * 2 - 256,
    "annotations_fps": 22050 // 256,
    "annot_n_frames": (22050 // 256) * 2,
}
_OVERLAP_FRAMES = 30
_OVERLAP_SAMPLES = _OVERLAP_FRAMES * _BP_CONST["fft_hop"]


def _model_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "models" / "basic_pitch.onnx",
        Path(__file__).resolve().parent / "models" / "basic_pitch.onnx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到人声 AI 模型 models/basic_pitch.onnx")


def _create_session(threads: int = 1):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(threads))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(_model_path()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _window_audio(audio: np.ndarray) -> Tuple[List[np.ndarray], int, int]:
    n_samples = _BP_CONST["audio_n_samples"]
    hop = n_samples - _OVERLAP_SAMPLES
    padded = np.pad(audio.astype(np.float32, copy=False), ( _OVERLAP_SAMPLES // 2, 0))
    original_length = len(audio)
    windows: List[np.ndarray] = []
    for start in range(0, len(padded), hop):
        chunk = padded[start : start + n_samples]
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        windows.append(chunk[None, :, None].astype(np.float32, copy=False))
    return windows, original_length, len(windows)


def _unwrap_output(output: np.ndarray, original_length: int) -> Optional[np.ndarray]:
    if output.ndim != 3:
        return None
    n_olap = int(0.5 * _OVERLAP_FRAMES)
    if n_olap:
        output = output[:, n_olap:-n_olap, :]
    shape = output.shape
    flat = output.reshape(shape[0] * shape[1], shape[2])
    n_frames = int(np.floor(original_length * (_BP_CONST["annotations_fps"] / _BP_CONST["sr"])))
    return flat[:n_frames, :]


def _run_basic_pitch(
    audio: np.ndarray,
    settings: AnalysisSettings,
    workers: int,
    progress: ProgressFn,
    cancel: CancelFn,
) -> Dict[str, np.ndarray]:
    import logging

    logging.getLogger().setLevel(logging.ERROR)
    import onnxruntime as ort  # noqa: F401  (ensures a clear error if missing)
    from .basicpitch_decode import decode_model_output

    windows, original_length, count = _window_audio(audio)
    if not windows:
        raise RuntimeError("音频太短，无法分析。")
    session = _create_session(threads=1)
    input_name = session.get_inputs()[0].name
    output_names = ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"]

    def infer(window: np.ndarray):
        return session.run(output_names, {input_name: window})

    results: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * count
    completed = 0
    using = max(1, min(workers, count))
    if using <= 1:
        for index, window in enumerate(windows):
            if cancel():
                raise AnalysisCancelled()
            results[index] = infer(window)
            completed += 1
            progress(8 + int(34 * completed / count), f"AI 神经网络转录 · {completed}/{count} 帧")
    else:
        with ThreadPoolExecutor(max_workers=using, thread_name_prefix="bp-infer") as pool:
            future_map = {pool.submit(infer, window): index for index, window in enumerate(windows)}
            for future in as_completed(future_map):
                if cancel():
                    for pending in future_map:
                        pending.cancel()
                    raise AnalysisCancelled()
                index = future_map[future]
                results[index] = future.result()
                completed += 1
                progress(8 + int(34 * completed / count), f"AI 神经网络转录 · {using} 核并行 · {completed}/{count}")

    if any(result is None for result in results):
        raise RuntimeError("神经网络推理未完成。")
    note_out = _unwrap_output(np.concatenate([r[0] for r in results], axis=0), original_length)  # type: ignore[index]
    onset_out = _unwrap_output(np.concatenate([r[1] for r in results], axis=0), original_length)  # type: ignore[index]
    if note_out is None or onset_out is None:
        raise RuntimeError("AI 模型输出无效。")

    progress(44, "解码音高与起音事件")
    model_output = {
        "note": note_out.astype(np.float32),
        "onset": onset_out.astype(np.float32),
    }
    events = decode_model_output(
        model_output,
        onset_threshold=settings.onset_threshold,
        frame_threshold=settings.frame_threshold,
        min_note_ms=settings.min_note_ms,
        min_midi=settings.min_midi,
        max_midi=settings.max_midi,
    )
    if settings.precision == "precise":
        # High-precision recall pass: lower onset threshold, then keep only
        # candidates that were not already found by the primary decode.
        progress(46, "高精度二次解码 · 补全弱起音")
        extra_threshold = max(0.20, float(settings.onset_threshold) - 0.14)
        extra = decode_model_output(
            model_output,
            onset_threshold=extra_threshold,
            frame_threshold=max(0.18, float(settings.frame_threshold) - 0.04),
            min_note_ms=max(55, int(settings.min_note_ms * 0.72)),
            min_midi=settings.min_midi,
            max_midi=settings.max_midi,
        )
        base = [tuple(e) for e in events]
        for candidate in extra:
            cs, ce, cp, ca = candidate
            duplicate = False
            for bs, be, bp, ba in base:
                if cp != bp:
                    continue
                inter = max(0.0, min(ce, be) - max(cs, bs))
                union = max(ce, be) - min(cs, bs)
                if union > 0 and inter / union >= 0.22:
                    duplicate = True
                    break
            if not duplicate:
                base.append(candidate)
        events = sorted(base, key=lambda e: (e[0], e[2]))

    notes: List[NoteEvent] = []
    for start, end, pitch, amplitude in events:
        if end <= start:
            continue
        velocity = int(round(np.clip(38.0 + 96.0 * float(amplitude), 28, 127)))
        confidence = float(np.clip(0.35 + 0.9 * float(amplitude), 0.1, 0.995))
        notes.append(
            NoteEvent(
                start=float(start),
                end=float(end),
                pitch=int(pitch),
                velocity=velocity,
                confidence=confidence,
                source="neural",
            )
        )
    return {"notes": notes, "cores": using}


# ---------------------------------------------------------------------------
# pYIN vocal pitch track (parallel chunks)
# ---------------------------------------------------------------------------

_PYIN_HOP = 512
_PYIN_FRAME = 2048
_CHUNK_SECONDS = 24.0
_CHUNK_OVERLAP = 1.0
_RMVPE_MODEL = None


def _rmvpe_path() -> Optional[Path]:
    path = Path(__file__).resolve().parent.parent / "models" / "rmvpe.onnx"
    return path if path.exists() and path.stat().st_size > 1_000_000 else None


def _track_pitch_rmvpe(
    audio: np.ndarray,
    settings: AnalysisSettings,
    workers: int,
    progress: ProgressFn,
    cancel: CancelFn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neural vocal F0 (RMVPE, 10 ms frames). This is the primary tracker."""
    global _RMVPE_MODEL
    path = _rmvpe_path()
    if path is None:
        raise FileNotFoundError("RMVPE model missing")
    if _RMVPE_MODEL is None:
        progress(54, "加载 RMVPE 神经音高模型")
        from .rmvpe import RMVPE

        _RMVPE_MODEL = RMVPE(model_path=str(path), threads=min(8, workers))
    if cancel():
        raise AnalysisCancelled()
    progress(62, "RMVPE 神经音高追踪 · 全频带人声")
    times, hz, confidence = _RMVPE_MODEL.predict(audio=audio, sr=ANALYSIS_SR)
    times = np.asarray(times, dtype=np.float64)
    hz = np.asarray(hz, dtype=np.float32)
    prob = np.asarray(confidence, dtype=np.float32)
    # Reject sub-audio / out-of-range and low-confidence excursions.
    lo = midi_to_hz(settings.min_midi) * 0.88
    hi = midi_to_hz(settings.max_midi) * 1.12
    invalid = (~np.isfinite(hz)) | (hz < lo) | (hz > hi) | (prob < 0.22)
    hz = hz.copy()
    hz[invalid] = np.nan
    prob = prob.copy()
    prob[invalid] = 0.0
    progress(78, f"RMVPE 完成 · {len(times)} 个音高帧")
    return times, hz, prob


def _pyin_chunk(args: Tuple[np.ndarray, int, int, float, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa

    audio, sr, fmin, fmax, frame_length, hop_length = args
    f0, voiced, probability = librosa.pyin(
        audio,
        fmin=float(fmin),
        fmax=float(fmax),
        sr=int(sr),
        frame_length=int(frame_length),
        hop_length=int(hop_length),
        fill_na=np.nan,
        center=True,
        n_thresholds=80,
        max_transition_rate=40,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, np.asarray(f0, dtype=np.float32), np.asarray(probability, dtype=np.float32)


def _track_pitch(
    audio: np.ndarray,
    settings: AnalysisSettings,
    workers: int,
    progress: ProgressFn,
    cancel: CancelFn,
    frame_length: int = _PYIN_FRAME,
    hop_length: int = _PYIN_HOP,
    min_hz: Optional[float] = None,
    max_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times, f0 (Hz), voiced probability on a regular pYIN grid."""
    duration = len(audio) / ANALYSIS_SR
    fmin = min_hz if min_hz is not None else max(50.0, float(midi_to_hz(settings.min_midi)) * 0.92)
    fmax = max_hz if max_hz is not None else min(1600.0, float(midi_to_hz(settings.max_midi)) * 1.08)
    chunks: List[Tuple[float, float, np.ndarray]] = []
    if duration <= _CHUNK_SECONDS + 4 or workers <= 1:
        chunks.append((0.0, duration, audio))
    else:
        step = _CHUNK_SECONDS
        start = 0.0
        while start < duration:
            end = min(duration, start + step)
            s = max(0.0, start - _CHUNK_OVERLAP)
            e = min(duration, end + _CHUNK_OVERLAP)
            segments = audio[int(s * ANALYSIS_SR) : int(e * ANALYSIS_SR)]
            chunks.append((s, e, segments))
            if end >= duration:
                break
            start += step - _CHUNK_OVERLAP

    args = [
        (segment, ANALYSIS_SR, fmin, fmax, frame_length, hop_length)
        for _, _, segment in chunks
    ]
    results: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = [None] * len(chunks)
    done = 0
    use_workers = max(1, min(workers, len(chunks)))
    if use_workers <= 1:
        for i, arg in enumerate(args):
            if cancel():
                raise AnalysisCancelled()
            results[i] = _pyin_chunk(arg)
            done += 1
            progress(56 + int(22 * done / len(chunks)), f"pYIN 音高追踪 · {done}/{len(chunks)}")
    else:
        with ThreadPoolExecutor(max_workers=use_workers, thread_name_prefix="pyin") as pool:
            future_map = {pool.submit(_pyin_chunk, arg): i for i, arg in enumerate(args)}
            for future in as_completed(future_map):
                if cancel():
                    for pending in future_map:
                        pending.cancel()
                    raise AnalysisCancelled()
                i = future_map[future]
                results[i] = future.result()
                done += 1
                progress(56 + int(22 * done / len(chunks)), f"pYIN 音高追踪 · {use_workers} 线程 · {done}/{len(chunks)}")

    times_all: List[np.ndarray] = []
    f0_all: List[np.ndarray] = []
    prob_all: List[np.ndarray] = []
    for (s, e, seg), result in zip(chunks, results):
        if result is None:
            continue
        local_times, f0, prob = result
        abs_times = local_times + s
        # Keep only this chunk's core, to avoid seams in the overlap region.
        sel = (abs_times >= s + (_CHUNK_OVERLAP * 0.5 if s > 0 else 0.0)) & (
            abs_times <= e - (_CHUNK_OVERLAP * 0.5 if e < duration else 0.0)
        )
        times_all.append(abs_times[sel])
        f0_all.append(f0[sel])
        prob_all.append(prob[sel])

    if not times_all:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    times = np.concatenate(times_all)
    f0 = np.concatenate(f0_all)
    prob = np.concatenate(prob_all)
    order = np.argsort(times, kind="stable")
    times = times[order]
    f0 = f0[order]
    prob = prob[order]

    # Resample the (possibly variable) chunk grid onto a uniform hop grid.
    target_times = np.arange(0.0, duration, hop_length / ANALYSIS_SR, dtype=np.float64)
    if len(times) >= 2:
        target_times = target_times[target_times <= times[-1]]
        f0 = np.interp(target_times, times, np.nan_to_num(f0, nan=0.0)).astype(np.float32)
        f0[f0 <= 0] = np.nan
        prob = np.interp(target_times, times, prob).astype(np.float32)
    return target_times.astype(np.float64), f0.astype(np.float32), prob.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature analysis: breath regions, key, tempo
# ---------------------------------------------------------------------------

def _spectral_features(audio: np.ndarray) -> Dict[str, np.ndarray]:
    n_fft = 1024
    hop = _PYIN_HOP
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop].astype(np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / ANALYSIS_SR)
    eps = 1e-10
    flatness = np.exp(np.mean(np.log(power + eps), axis=1)) / (np.mean(power, axis=1) + eps)
    total = np.sum(power, axis=1) + eps
    high = np.sum(power[:, freqs > 2200], axis=1) / total
    vocal_band = np.sum(power[:, (freqs > 150) & (freqs < 8000)], axis=1) + eps
    centroid = np.sum(power * freqs[None, :], axis=1) / total
    rms = np.sqrt(np.mean(frames**2, axis=1) + eps)
    db = 20.0 * np.log10(rms + 1e-8)
    times = (np.arange(frames.shape[0]) * hop + n_fft / 2) / ANALYSIS_SR
    return {
        "times": times.astype(np.float64),
        "flatness": np.clip(flatness, 0, 1),
        "high_ratio": np.clip(high, 0, 1),
        "centroid": centroid,
        "db": db,
        "vocal_energy": vocal_band,
        "power": power,
        "freqs": freqs,
    }


def _detect_breaths(
    audio: np.ndarray,
    f0_times: np.ndarray,
    f0_prob: np.ndarray,
    strength: float,
) -> Tuple[List[BreathRegion], Dict[str, np.ndarray]]:
    features = _spectral_features(audio)
    times = features["times"]
    db = features["db"]
    if len(db) == 0:
        return [], features
    # Align pYIN voiced probability to the spectral frame grid.
    if len(f0_times) and len(f0_prob):
        prob = np.interp(times, f0_times, np.nan_to_num(f0_prob, nan=0.0))
    else:
        prob = np.zeros_like(times)
    active_floor = max(float(np.percentile(db, 70)) - 30.0, -58.0)
    active = db > active_floor
    flat = np.clip((features["flatness"] - 0.16) / 0.40, 0, 1)
    high = np.clip((features["high_ratio"] - 0.22) / 0.55, 0, 1)
    low_voiced = np.clip((0.48 - prob) / 0.48, 0, 1)
    centroid = np.clip((features["centroid"] - 1500.0) / 3500.0, 0, 1)
    energy = np.clip((db - (active_floor - 2.0)) / 26.0, 0, 1)
    score = 0.40 * flat + 0.24 * high + 0.23 * low_voiced + 0.08 * centroid + 0.05 * energy
    threshold = float(np.clip(0.70 - 0.34 * strength, 0.30, 0.66))
    raw = (score > threshold) & active
    # Close tiny holes, then discard very short noise blips.
    raw = binary_closing(raw, structure=np.ones(3))
    min_frames = max(3, int(round(0.075 / (hop_seconds := _PYIN_HOP / ANALYSIS_SR))))
    regions: List[BreathRegion] = []
    indices = np.flatnonzero(raw)
    if len(indices):
        splits = np.where(np.diff(indices) > 1)[0]
        groups = np.split(indices, splits + 1)
        for group in groups:
            if len(group) < min_frames:
                continue
            start = max(0.0, float(times[group[0]] - hop_seconds * 0.5))
            end = float(times[group[-1]] + hop_seconds * 0.5)
            regions.append(BreathRegion(start, end, float(np.mean(score[group]))))
    # Merge close regions (a breath often has a soft onset and tail).
    merged: List[BreathRegion] = []
    for region in regions:
        if merged and region.start - merged[-1].end < 0.075:
            merged[-1].end = region.end
            merged[-1].score = max(merged[-1].score, region.score)
        else:
            merged.append(region)
    return merged, {**features, "prob": prob, "breath_score": score}


def _annotate_note_features(
    notes: Sequence[NoteEvent],
    audio: np.ndarray,
    f0_times: np.ndarray,
    f0_prob: np.ndarray,
) -> Tuple[List[NoteEvent], float]:
    """Attach local loudness and neural voicing support to every note."""
    n_fft = 1024
    hop = 256
    padded = np.pad(audio, (0, max(0, n_fft - len(audio))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    frame_times = (np.arange(len(frame_rms)) * hop + n_fft / 2) / ANALYSIS_SR
    loud_db = float(20.0 * np.log10(max(1e-9, np.percentile(frame_rms, 90))))
    annotated: List[NoteEvent] = []
    for note in notes:
        copy = note.copy()
        level_mask = (frame_times >= note.start) & (frame_times <= note.end)
        if np.any(level_mask):
            level = float(np.percentile(frame_rms[level_mask], 60))
            copy.level_db = float(20.0 * np.log10(max(1e-9, level)))
        else:
            copy.level_db = -120.0
        support_mask = (f0_times >= note.start) & (f0_times <= note.end) & np.isfinite(f0_prob)
        copy.pitch_support = float(np.median(f0_prob[support_mask])) if np.any(support_mask) else 0.0
        annotated.append(copy)
    return annotated, loud_db


def _noise_gate_db(loud_db: float, strength: float) -> float:
    if strength <= 0.001:
        return -120.0
    # Higher strength raises the gate. The relative margin widens with the
    # loudness of the input, so quiet/whispered performances stay usable.
    relative = 42.0 - 22.0 * max(0.0, min(1.0, strength))
    return float(max(-55.0, min(-30.0, loud_db - relative)))


def _filter_noise_notes(notes: Sequence[NoteEvent], gate_db: float) -> Tuple[List[NoteEvent], List[NoteEvent]]:
    kept: List[NoteEvent] = []
    removed: List[NoteEvent] = []
    for note in notes:
        if gate_db > -119.0 and note.level_db < gate_db:
            removed.append(note)
            continue
        if gate_db > -119.0 and note.pitch_support < 0.06 and note.level_db < gate_db + 8.0:
            removed.append(note)
            continue
        kept.append(note)
    return kept, removed


def _breath_overlap(note: NoteEvent, breaths: Sequence[BreathRegion]) -> float:
    if note.duration <= 0 or not breaths:
        return 0.0
    overlap = 0.0
    for region in breaths:
        overlap += max(0.0, min(note.end, region.end) - max(note.start, region.start))
    return min(1.0, overlap / note.duration)


def _filter_breaths(
    notes: Sequence[NoteEvent],
    breaths: Sequence[BreathRegion],
    strength: float,
) -> Tuple[List[NoteEvent], List[NoteEvent]]:
    kept: List[NoteEvent] = []
    removed: List[NoteEvent] = []
    limit = float(np.clip(0.72 - 0.50 * strength, 0.20, 0.72))
    for note in notes:
        fraction = _breath_overlap(note, breaths)
        if fraction >= limit:
            removed.append(note)
        else:
            kept.append(note)
    return kept, removed


def refilter_breath(result: AnalysisResult, settings: AnalysisSettings) -> List[NoteEvent]:
    """Fast noise/breath re-gating using cached candidates (no re-inference)."""
    candidates = result.pre_breath_notes or result.all_notes or result.notes
    gate_db = _noise_gate_db(result.noise_reference_db, settings.noise_gate)
    kept, removed_noise = _filter_noise_notes(candidates, gate_db)
    if settings.ignore_breath:
        kept, removed_breath = _filter_breaths(kept, result.breaths, settings.breath_strength)
    else:
        removed_breath = []
    result.notes = sorted(kept, key=lambda n: (n.start, n.pitch))
    result.removed_breath_notes = removed_breath
    result.removed_noise_notes = removed_noise
    return result.notes


def _add_pitch_recovered_notes(
    notes: Sequence[NoteEvent],
    f0_times: np.ndarray,
    f0_hz: np.ndarray,
    f0_prob: np.ndarray,
    audio: np.ndarray,
    settings: AnalysisSettings,
    hop_seconds: float,
) -> List[NoteEvent]:
    """High-precision recall: fill note gaps with stable RMVPE pitch segments."""
    if len(f0_times) < 5:
        return list(notes)
    candidates = _monophonic_notes(f0_times, f0_hz, f0_prob, audio, settings, hop_seconds=hop_seconds)
    min_add = max(0.15, settings.min_note_ms / 1000.0 * 1.25)
    base = [n.copy() for n in notes]
    additions: List[NoteEvent] = []
    for candidate in candidates:
        if candidate.duration < min_add:
            continue
        overlap = 0.0
        for existing in base:
            overlap += max(0.0, min(candidate.end, existing.end) - max(candidate.start, existing.start))
        if overlap / candidate.duration > 0.30:
            continue
        candidate.source = "rmvpe"
        candidate.confidence = 0.62
        additions.append(candidate)
    # Precision is allowed to add a handful of notes, not to swamp the result.
    max_additions = max(2, min(24, len(base) // 4))
    additions.sort(key=lambda n: n.duration, reverse=True)
    additions = additions[:max_additions]
    merged = merge_legato(base + additions, max_gap=0.05)
    merged = _remove_contained_notes(merged)
    return _enforce_monophony(merged)


def _refine_bass_notes(
    notes: Sequence[NoteEvent],
    f0_times: np.ndarray,
    f0_hz: np.ndarray,
    f0_prob: np.ndarray,
    settings: AnalysisSettings,
) -> List[NoteEvent]:
    """Use a low-register pYIN track to validate/correct Basic Pitch bass notes."""
    out: List[NoteEvent] = []
    low, high = int(settings.min_midi), int(settings.max_midi)
    for note in notes:
        if not (low - 1 <= note.pitch <= high + 1):
            continue
        lo = note.start + min(0.05, note.duration * 0.18)
        hi = note.end - min(0.05, note.duration * 0.18)
        if hi <= lo:
            lo, hi = note.start, note.end
        mask = (
            (f0_times >= lo)
            & (f0_times <= hi)
            & np.isfinite(f0_hz)
            & (f0_prob >= 0.30)
        )
        count = int(np.sum(mask))
        refined = note.copy()
        if count >= 3:
            values = hz_to_midi(f0_hz[mask])
            support = f0_prob[mask].astype(np.float64)
            target = int(round(float(np.median(values))))
            if low <= target <= high:
                refined.pitch = target
                refined.confidence = min(0.99, max(note.confidence, float(np.mean(support)) * 0.98))
                out.append(refined)
            continue
        # Keep very short attacks that the monophonic tracker could miss, but
        # only when Basic Pitch itself is confident.
        if count >= 1 or (note.duration <= 0.14 and note.confidence >= 0.9):
            out.append(refined)
    return out


def _remove_contained_notes(notes: Sequence[NoteEvent]) -> List[NoteEvent]:
    """Drop short harmonic/portamento notes sitting inside a stronger note."""
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
    removed = set()
    for i, outer in enumerate(ordered):
        if i in removed:
            continue
        for j, inner in enumerate(ordered):
            if i == j or j in removed or inner.pitch == outer.pitch:
                continue
            if inner.duration >= outer.duration * 0.82:
                continue
            contained = inner.start >= outer.start - 0.025 and inner.end <= outer.end + 0.025
            if contained and inner.confidence <= outer.confidence + 0.15:
                removed.add(j)
    return [n for i, n in enumerate(ordered) if i not in removed]


def _enforce_monophony(notes: Sequence[NoteEvent]) -> List[NoteEvent]:
    """Keep one lead line: trim tiny legato overlaps so exported MIDI is clean."""
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
    for i in range(len(ordered) - 1):
        current = ordered[i]
        nxt = ordered[i + 1]
        if current.end > nxt.start + 0.012 and nxt.start > current.start:
            current.end = max(current.start + 0.03, nxt.start - 0.008)
    return [n for n in ordered if n.duration > 0.03]


def _refine_with_pitch_track(
    notes: Sequence[NoteEvent],
    f0_times: np.ndarray,
    f0_hz: np.ndarray,
    f0_prob: np.ndarray,
) -> List[NoteEvent]:
    if len(f0_times) == 0:
        return [n.copy() for n in notes]
    out: List[NoteEvent] = []
    for note in notes:
        refined = note.copy()
        lo = note.start + min(0.04, note.duration * 0.15)
        hi = note.end - min(0.04, note.duration * 0.15)
        if hi <= lo:
            out.append(refined)
            continue
        mask = (f0_times >= lo) & (f0_times <= hi) & np.isfinite(f0_hz) & (f0_prob > 0.45)
        count = int(np.sum(mask))
        if count >= 4:
            pitches = hz_to_midi(f0_hz[mask])
            weights = f0_prob[mask].astype(np.float64)
            median = float(np.median(pitches))
            target = int(round(median))
            # Only nudge by a semitone: this fixes partial mistuning without
            # importing an octave error from the monophonic tracker.
            if abs(target - note.pitch) == 1:
                refined.pitch = target
                refined.confidence = float(np.clip(max(note.confidence, np.mean(weights)) * 0.99, 0.1, 0.995))
            elif target == note.pitch:
                refined.confidence = float(np.clip(max(note.confidence, np.mean(weights) * 0.95), 0.1, 0.995))
        out.append(refined)
    return out


def _monophonic_notes(
    f0_times: np.ndarray,
    f0_hz: np.ndarray,
    f0_prob: np.ndarray,
    audio: np.ndarray,
    settings: AnalysisSettings,
    hop_seconds: float = _PYIN_HOP / ANALYSIS_SR,
) -> List[NoteEvent]:
    if len(f0_times) < 3:
        return []
    hop_s = float(hop_seconds) if hop_seconds > 0 else _PYIN_HOP / ANALYSIS_SR
    valid = np.isfinite(f0_hz) & (f0_prob >= 0.34)
    midi = np.full(len(f0_hz), np.nan, dtype=np.float64)
    midi[valid] = hz_to_midi(f0_hz[valid])
    # Remove short pitch spikes before quantization (~160 ms window).
    if np.sum(valid) >= 5:
        filled = np.where(np.isfinite(midi), midi, np.nanmedian(midi[valid]))
        filter_size = max(3, int(round(0.16 / hop_s)) | 1)
        smooth = median_filter(filled, size=filter_size, mode="nearest")
        smooth[~valid] = np.nan
        midi = smooth

    # RMS per frame for velocity (use a fixed 512-hop analysis grid and interp).
    n_fft = 1024
    analysis_hop = _PYIN_HOP
    frames = np.lib.stride_tricks.sliding_window_view(
        np.pad(audio, (0, max(0, n_fft - len(audio)))), n_fft
    )[::analysis_hop]
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-10)
    rms_at = np.interp(
        f0_times,
        (np.arange(len(frame_rms)) * analysis_hop + n_fft / 2) / ANALYSIS_SR,
        frame_rms,
    )
    ref = float(np.percentile(frame_rms, 85)) if len(frame_rms) else 1.0
    hold_frames = max(2, int(round(0.04 / hop_s)))

    notes: List[NoteEvent] = []
    current_pitch: Optional[int] = None
    start_time = 0.0
    gap_start: Optional[float] = None
    candidate: Optional[int] = None
    candidate_count = 0
    for i, value in enumerate(midi):
        t = float(f0_times[i]) if i < len(f0_times) else i * hop_s
        if not np.isfinite(value):
            if current_pitch is not None and gap_start is None:
                gap_start = t
            elif current_pitch is not None and gap_start is not None and t - gap_start > 0.06:
                end = gap_start
                if end - start_time >= settings.min_note_ms / 1000.0:
                    seg = (f0_times >= start_time) & (f0_times <= end)
                    vel = int(np.clip(45 + 90 * (np.median(rms_at[seg]) / (ref + 1e-9)), 35, 127)) if np.any(seg) else 90
                    notes.append(NoteEvent(start_time, end, current_pitch, vel, 0.75, "pyin"))
                current_pitch = None
                gap_start = None
                candidate = None
                candidate_count = 0
            continue

        pitch = int(round(float(value)))
        if current_pitch is None:
            current_pitch = pitch
            start_time = t
            gap_start = None
            candidate = None
            candidate_count = 0
            continue
        gap_start = None
        if pitch == current_pitch:
            candidate = None
            candidate_count = 0
            continue
        if pitch == candidate:
            candidate_count += 1
        else:
            candidate = pitch
            candidate_count = 1
        if candidate_count >= hold_frames:
            change_time = t - (candidate_count - 1) * hop_s
            if change_time - start_time >= settings.min_note_ms / 1000.0:
                seg = (f0_times >= start_time) & (f0_times <= change_time)
                vel = int(np.clip(45 + 90 * (np.median(rms_at[seg]) / (ref + 1e-9)), 35, 127)) if np.any(seg) else 90
                notes.append(NoteEvent(start_time, change_time, current_pitch, vel, 0.75, "pyin"))
                start_time = change_time
            else:
                start_time = change_time
            current_pitch = pitch
            candidate = None
            candidate_count = 0

    if current_pitch is not None:
        end = float(f0_times[-1] + hop_s)
        if end - start_time >= settings.min_note_ms / 1000.0:
            seg = f0_times >= start_time
            vel = int(np.clip(45 + 90 * (np.median(rms_at[seg]) / (ref + 1e-9)), 35, 127)) if np.any(seg) else 90
            notes.append(NoteEvent(start_time, end, current_pitch, vel, 0.75, "pyin"))

    # Coalesce notes repeated after a tiny unvoiced hiccup.
    return merge_legato(notes, max_gap=0.045)


_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _key_from_audio(
    f0_hz: np.ndarray,
    f0_prob: np.ndarray,
    features: Dict[str, np.ndarray],
) -> Tuple[KeyInfo, np.ndarray]:
    pitch_hist = np.zeros(12, dtype=np.float64)
    if len(f0_hz):
        valid = np.isfinite(f0_hz) & (f0_prob > 0.35)
        if np.any(valid):
            pc = np.mod(np.round(hz_to_midi(f0_hz[valid])).astype(int), 12)
            weights = f0_prob[valid].astype(np.float64) ** 2
            np.add.at(pitch_hist, pc, weights)
    if pitch_hist.sum() > 0:
        pitch_hist /= pitch_hist.sum()

    power = features["power"]
    freqs = features["freqs"]
    band = (freqs >= 65) & (freqs <= 2200)
    band_power = power[:, band]
    band_freqs = freqs[band]
    frame_energy = band_power.sum(axis=1)
    if len(frame_energy):
        threshold = np.percentile(frame_energy, 55)
        use = frame_energy > threshold
        band_power = band_power[use]
    spectral_hist = np.zeros(12, dtype=np.float64)
    if band_power.size:
        midi_bins = hz_to_midi(np.maximum(band_freqs, 1e-6))
        classes = np.mod(np.round(midi_bins).astype(int), 12)
        weights = np.sqrt(band_power).sum(axis=0)
        np.add.at(spectral_hist, classes, weights)
    if spectral_hist.sum() > 0:
        spectral_hist /= spectral_hist.sum()
    chroma = 0.62 * pitch_hist + 0.38 * spectral_hist
    if chroma.sum() <= 0:
        return KeyInfo(0, "major", 0.0), chroma

    candidates = []
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(profile, tonic)
            corr = float(np.corrcoef(chroma, rotated)[0, 1])
            candidates.append((corr, tonic, mode))
    candidates.sort(reverse=True)
    best, tonic, mode = candidates[0]
    second = candidates[1][0]
    margin = max(0.0, best - second)
    confidence = float(np.clip(52.0 + margin * 340.0 + max(0.0, best - 0.55) * 55.0, 30.0, 98.0))
    return KeyInfo(tonic=tonic, mode=mode, confidence=confidence), chroma


def _note_onset_grid_score(onsets: np.ndarray, bpm: float) -> float:
    """How tightly note attacks align to a beat + eighth-note grid."""
    if len(onsets) < 8:
        return 0.0
    beat = 60.0 / max(1e-6, bpm)
    best = 0.0
    for phase in np.linspace(0.0, beat, 41, endpoint=False):
        distance = np.abs(((onsets - phase + beat / 2.0) % beat) - beat / 2.0) / beat
        error = np.minimum(distance, np.abs(distance - 0.5))
        score = float(np.mean(np.exp(-((error / 0.045) ** 2))))
        if score > best:
            best = score
    return best


def _refine_tempo_with_notes(base_bpm: float, notes: Sequence[NoteEvent], duration: float) -> float:
    """Vocal-only beat tracking is ambiguous; use note attacks to disambiguate.

    A sung vocal has no kick drum, so librosa alone tends to report a
    triplet-related tempo (e.g. 129 for a 97 BPM song).  We score nearby BPM
    candidates against the observed note attacks; real melodies generally
    land on the half-beat grid far more consistently than the wrong triplets.
    """
    if len(notes) < 40 or duration < 30.0 or not np.isfinite(base_bpm):
        return base_bpm
    onsets = np.array(sorted({round(float(n.start), 3) for n in notes}), dtype=np.float64)
    if len(onsets) > 1:
        keep = np.r_[True, np.diff(onsets) > 0.055]
        onsets = onsets[keep]
    if len(onsets) < 30:
        return base_bpm
    low = np.floor(max(55.0, base_bpm - 10.0) * 2.0) / 2.0
    high = np.ceil(min(190.0, base_bpm + 10.0) * 2.0) / 2.0
    candidates = np.arange(low, high + 0.01, 0.5)
    scored = []
    for bpm in candidates:
        grid_score = _note_onset_grid_score(onsets, float(bpm))
        prior = float(np.exp(-0.5 * ((bpm - base_bpm) / 3.2) ** 2))
        scored.append((grid_score * (0.72 + 0.28 * prior), float(bpm), grid_score))
    scored.sort(reverse=True)
    best_combined, best_bpm, best_grid = scored[0]
    if best_grid < 0.145:
        return base_bpm
    return float(round(best_bpm))


def _estimate_tempo(audio: np.ndarray, notes: Sequence[NoteEvent] = ()) -> float:
    try:
        import librosa

        onset_env = librosa.onset.onset_strength(y=audio, sr=ANALYSIS_SR, hop_length=_PYIN_HOP)
        tempo = librosa.feature.tempo(
            onset_envelope=onset_env,
            sr=ANALYSIS_SR,
            hop_length=_PYIN_HOP,
            start_bpm=100.0,
            std_bpm=1.0,
            aggregate=np.median,
        )
        value = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(value) or not 35 <= value <= 220:
            return 120.0
        return _refine_tempo_with_notes(value, notes, len(audio) / ANALYSIS_SR)
    except Exception:
        return 120.0


def _engine_label(engine: str) -> str:
    return {
        "vocal": "Vocal AI · Basic Pitch",
        "poly": "Polyphonic AI · Basic Pitch",
        "mono": "Monophonic",
        "bass": "Bass AI · Basic Pitch",
    }.get(engine, "Vocal AI")


def analyze_audio(
    audio_file: AudioFile,
    settings: AnalysisSettings,
    progress: Optional[ProgressFn] = None,
    cancel: Optional[CancelFn] = None,
    workers: int = 0,
) -> AnalysisResult:
    started = time.perf_counter()
    _prepare_frozen_numba()
    progress = progress or (lambda p, m: None)
    cancel = cancel or (lambda: False)
    audio = np.asarray(audio_file.analysis_samples, dtype=np.float32)
    if workers <= 0:
        workers = min(8, os.cpu_count() or 4)
    workers = max(1, int(workers))
    result = AnalysisResult(settings=settings, duration=float(len(audio) / ANALYSIS_SR))
    result.engine_label = _engine_label(settings.engine)

    progress(2, "读取并重采样音频")
    if len(audio) < ANALYSIS_SR // 2:
        raise ValueError("音频短于 0.5 秒，无法分析。")
    if cancel():
        raise AnalysisCancelled()

    raw_notes: List[NoteEvent] = []
    if settings.engine in ("vocal", "poly", "bass"):
        progress(6, "初始化音符神经网络")
        bp = _run_basic_pitch(audio, settings, workers, progress, cancel)
        raw_notes = bp["notes"]
        cores_used = int(bp["cores"])
        progress(47, f"神经网络完成 · {len(raw_notes)} 个候选音符")
    else:
        cores_used = workers

    if cancel():
        raise AnalysisCancelled()
    progress(52, "启动低频音高追踪" if settings.engine == "bass" else "启动神经音高追踪")
    if settings.engine == "bass":
        pitch_source = "pYIN"
        progress(54, "Bass 低频音高追踪 · pYIN")
        max_bass_hz = max(300.0, min(700.0, float(midi_to_hz(settings.max_midi)) * 2.2))
        f0_times, f0_hz, f0_prob = _track_pitch(
            audio,
            settings,
            workers,
            progress,
            cancel,
            frame_length=4096,
            hop_length=512,
            min_hz=25.0,
            max_hz=max_bass_hz,
        )
    elif settings.precision == "fast":
        pitch_source = "pYIN"
        progress(54, "快速音高追踪 · pYIN")
        f0_times, f0_hz, f0_prob = _track_pitch(audio, settings, workers, progress, cancel)
    else:
        pitch_source = "RMVPE"
        try:
            if _rmvpe_path() is None:
                raise FileNotFoundError("RMVPE model missing")
            f0_times, f0_hz, f0_prob = _track_pitch_rmvpe(audio, settings, workers, progress, cancel)
        except AnalysisCancelled:
            raise
        except Exception:
            pitch_source = "pYIN"
            progress(54, "RMVPE 不可用 · 切换 pYIN 音高追踪")
            f0_times, f0_hz, f0_prob = _track_pitch(audio, settings, workers, progress, cancel)
    result.f0_times = f0_times
    result.f0_hz = f0_hz
    result.f0_prob = f0_prob
    result.pitch_source = pitch_source
    result.precision = settings.precision
    result.engine_label = _engine_label(settings.engine) + f" · {pitch_source}"
    hop_seconds = float(np.median(np.diff(f0_times))) if len(f0_times) > 1 else _PYIN_HOP / ANALYSIS_SR

    if cancel():
        raise AnalysisCancelled()
    if settings.engine == "bass":
        progress(80, "低频噪声门限")
        breaths = []
        features = _spectral_features(audio)
    else:
        progress(80, "检测气声与噪声段")
        breaths, features = _detect_breaths(
            audio, f0_times, f0_prob, settings.breath_strength if settings.ignore_breath else 0.45
        )
    result.breaths = breaths
    result.breath_count = len(breaths)
    result.breath_time = float(sum(b.duration for b in breaths))

    if settings.engine == "mono":
        notes = _monophonic_notes(f0_times, f0_hz, f0_prob, audio, settings, hop_seconds=hop_seconds)
        result.all_notes = [n.copy() for n in notes]
    else:
        notes = [n.copy() for n in raw_notes]
        result.all_notes = [n.copy() for n in raw_notes]
        if settings.engine == "vocal":
            notes = merge_legato(notes, max_gap=0.055)
            notes = suppress_octave_ghosts(notes, iou_threshold=0.58)
            notes = _remove_contained_notes(notes)
            if settings.pitch_refine:
                notes = _refine_with_pitch_track(notes, f0_times, f0_hz, f0_prob)
            notes = merge_legato(notes, max_gap=0.055)
            notes = _enforce_monophony(notes)
            if settings.precision == "precise":
                progress(86, "高精度音高补全 · 找回弱音")
                notes = _add_pitch_recovered_notes(
                    notes, f0_times, f0_hz, f0_prob, audio, settings, hop_seconds
                )
        elif settings.engine == "bass":
            progress(84, "Bass 音高校验与八度修正")
            notes = merge_legato(notes, max_gap=0.075)
            notes = suppress_octave_ghosts(notes, iou_threshold=0.55)
            notes = _remove_contained_notes(notes)
            notes = _refine_bass_notes(notes, f0_times, f0_hz, f0_prob, settings)
            notes = merge_legato(notes, max_gap=0.06)
            notes = _enforce_monophony(notes)

    progress(88, "噪声门限与音符清理")
    notes = [n for n in notes if n.duration >= settings.min_note_ms / 1000.0 * 0.90 and n.end > 0]
    notes, loud_db = _annotate_note_features(notes, audio, f0_times, f0_prob)
    result.noise_reference_db = loud_db
    result.pre_breath_notes = [n.copy() for n in notes]
    gate_db = _noise_gate_db(loud_db, settings.noise_gate)
    notes, removed_noise = _filter_noise_notes(notes, gate_db)
    removed_breath: List[NoteEvent] = []
    if settings.ignore_breath and settings.engine != "bass":
        notes, removed_breath = _filter_breaths(notes, breaths, settings.breath_strength)
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    result.notes = notes
    result.removed_breath_notes = removed_breath
    result.removed_noise_notes = removed_noise

    progress(93, "识别调性与速度")
    key, _chroma = _key_from_audio(f0_hz, f0_prob, features)
    if settings.detect_bpm:
        key.bpm = _estimate_tempo(audio, notes)
    result.key = key
    result.cores_used = int(cores_used)
    result.elapsed = float(time.perf_counter() - started)
    progress(100, f"分析完成 · {len(notes)} 个音符")
    return result
