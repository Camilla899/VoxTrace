"""Audio decoding, resampling and temporary playback file management."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PLAYBACK_SR = 44100
ANALYSIS_SR = 22050
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma", ".mp4"
}


def _ffmpeg_candidates() -> list[str]:
    candidates = []
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local.exists():
        candidates.append(str(local))
    candidates.append("ffmpeg")
    return candidates


def find_ffmpeg() -> Optional[str]:
    for candidate in _ffmpeg_candidates():
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _ffmpeg_decode(path: Path, target: Path, sr: int = PLAYBACK_SR) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("无法解码此格式：未找到 ffmpeg。请安装 ffmpeg 或使用 WAV 文件。")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le", str(target),
    ]
    result = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0 or not target.exists() or target.stat().st_size < 64:
        msg = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"音频解码失败：{msg or '未知错误'}")


def _read_mono(path: Path) -> Tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(str(path), always_2d=True, dtype="float64")
        mono = data.mean(axis=1)
        if not np.all(np.isfinite(mono)):
            mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        return mono.astype(np.float32), int(sr)
    except Exception:
        # Formats libsndfile cannot handle (M4A/AAC/MP4/WMA) go through ffmpeg first.
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            _ffmpeg_decode(path, tmp, PLAYBACK_SR)
            data, sr = sf.read(str(tmp), always_2d=True, dtype="float64")
            return data.mean(axis=1).astype(np.float32), int(sr)
        finally:
            tmp.unlink(missing_ok=True)


def resample_audio(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(samples, dtype=np.float32)
    gcd = np.gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // int(gcd)
    down = int(sr_in) // int(gcd)
    out = resample_poly(samples, up, down)
    return np.asarray(out, dtype=np.float32)


@dataclass
class AudioFile:
    source_path: str
    display_name: str
    samples: np.ndarray          # mono float32 at PLAYBACK_SR
    sample_rate: int
    duration: float
    playback_path: str
    analysis_samples: np.ndarray  # mono float32 at ANALYSIS_SR

    @property
    def channel_label(self) -> str:
        return "MONO · 44.1 kHz"


_CACHE_DIR: Optional[Path] = None


def cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = Path(tempfile.mkdtemp(prefix="voxtrace_"))
    return _CACHE_DIR


def cleanup_cache() -> None:
    global _CACHE_DIR
    if _CACHE_DIR is not None:
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)
        _CACHE_DIR = None


def _write_playback_wav(samples: np.ndarray, sr: int, path: Path) -> None:
    # QMediaPlayer is most reliable with a plain PCM WAV.
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    data = samples
    if peak > 0.98:
        data = data * (0.98 / peak)
    pcm = np.clip(data * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def load_audio(path: str | Path) -> AudioFile:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"音频文件不存在：{source}")
    samples, sr = _read_mono(source)
    if samples.size == 0:
        raise RuntimeError("音频为空或无法读取。")
    samples = resample_audio(samples, sr, PLAYBACK_SR)
    duration = float(len(samples) / PLAYBACK_SR)
    analysis = resample_audio(samples, PLAYBACK_SR, ANALYSIS_SR)
    playback = cache_dir() / f"play_{abs(hash(str(source))) & 0xFFFFFFFFFFFF:x}.wav"
    _write_playback_wav(samples, PLAYBACK_SR, playback)
    display = source.stem
    return AudioFile(
        source_path=str(source),
        display_name=display,
        samples=samples,
        sample_rate=PLAYBACK_SR,
        duration=duration,
        playback_path=str(playback),
        analysis_samples=analysis,
    )
