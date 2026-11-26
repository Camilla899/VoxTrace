"""RMVPE neural vocal pitch estimator, ONNX Runtime implementation.

Adapted for VoxTrace from ``rmvpe-onnx`` (MIT) and the RVC project.
Original code: https://github.com/NewComer00/rmvpe-onnx
RMVPE algorithm: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
Model weights: https://huggingface.co/lj1995/VoiceConversionWebUI

Copyright (c) 2023 liujing04, 源文雨, Ftps
Copyright (c) 2022 lj1995
MIT License — see licenses/RMVPE_LICENSE.txt and RMVPE_ONNX_LICENSE.txt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from scipy.signal import get_window

MEL_CHANNELS = 128
RMVPE_SAMPLE_RATE = 16_000
WIN_LENGTH = 1024
HOP_LENGTH = 160
MEL_FMIN = 30
MEL_FMAX = 8000
N_CLASS = 360
CENTS_OFFSET = 1997.3794084376191


class MelSpectrogram:
    def __init__(
        self,
        n_mel_channels: int = MEL_CHANNELS,
        sampling_rate: int = RMVPE_SAMPLE_RATE,
        win_length: int = WIN_LENGTH,
        hop_length: int = HOP_LENGTH,
        n_fft: int | None = None,
        mel_fmin: float = MEL_FMIN,
        mel_fmax: float = MEL_FMAX,
        clamp: float = 1e-5,
    ):
        from librosa.filters import mel as librosa_mel

        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft or win_length
        self.clamp = clamp
        self.mel_basis = librosa_mel(
            sr=sampling_rate,
            n_fft=self.n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        ).astype(np.float32)

    def __call__(self, audio: np.ndarray, center: bool = True) -> np.ndarray:
        if center:
            audio = np.pad(audio, self.n_fft // 2, mode="reflect")
        window = get_window("hann", self.win_length, fftbins=True).astype(np.float32)
        if self.n_fft > self.win_length:
            pad = self.n_fft - self.win_length
            window = np.pad(window, (pad // 2, pad - pad // 2))
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        shape = ((audio.size - self.n_fft) // self.hop_length + 1, self.n_fft)
        strides = (audio.strides[0] * self.hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        fft_out = np.fft.rfft(frames * window, n=self.n_fft)
        magnitude = np.abs(fft_out).T.astype(np.float32)
        mel_out = self.mel_basis @ magnitude
        return np.log(np.clip(mel_out, self.clamp, None))


class RMVPE:
    """Small, dependency-light RMVPE ONNX front-end used by VoxTrace."""

    def __init__(self, model_path: str | Path, threads: int = 0):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            options.intra_op_num_threads = int(threads)
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.mel_extractor = MelSpectrogram()
        cents = 20 * np.arange(N_CLASS) + CENTS_OFFSET
        self.cents_mapping = np.pad(cents, (4, 4))

    def _mel_to_salience(self, mel: np.ndarray) -> np.ndarray:
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, n_pad)))
        hidden = self.session.run([self.output_name], {self.input_name: mel})[0]
        return hidden[:, :n_frames]

    def _local_average_cents(self, salience: np.ndarray) -> np.ndarray:
        center = np.argmax(salience, axis=1)
        padded = np.pad(salience, ((0, 0), (4, 4)))
        center = center + 4
        frame_idx = np.arange(padded.shape[0])[:, None]
        bin_offsets = np.arange(9)[None, :]
        bin_idx = (center[:, None] - 4) + bin_offsets
        window_sal = padded[frame_idx, bin_idx]
        window_cent = self.cents_mapping[bin_idx]
        return (window_sal * window_cent).sum(1) / np.maximum(1e-9, window_sal.sum(1))

    def predict(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import librosa

        if audio.ndim > 1:
            audio = librosa.to_mono(audio.T)
        if sr != RMVPE_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=RMVPE_SAMPLE_RATE)
        audio = np.asarray(audio, dtype=np.float32)
        mel = self.mel_extractor(audio)[None]
        salience = self._mel_to_salience(mel)[0].astype(np.float32)
        confidence = salience.max(axis=1)
        cents = self._local_average_cents(salience)
        frequency = 10 * (2 ** (cents / 1200))
        times = np.arange(len(frequency), dtype=np.float64) * (HOP_LENGTH / RMVPE_SAMPLE_RATE)
        return times, frequency.astype(np.float32), confidence.astype(np.float32)
