"""Note and key primitives shared by the analysis engine and the UI."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
MAJOR_CODE = "major"
MINOR_CODE = "minor"


def midi_to_name(pitch: int, flats: bool = False) -> str:
    names = FLAT_NAMES if flats else NOTE_NAMES
    return f"{names[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def midi_to_hz(pitch: float) -> float:
    return float(440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0)))


def hz_to_midi(hz: float):
    """Scalar or vectorized Hz -> (fractional) MIDI pitch."""
    arr = np.asarray(hz, dtype=np.float64)
    out = 69.0 + 12.0 * np.log2(np.maximum(arr, 1e-8) / 440.0)
    if out.ndim == 0:
        return float(out)
    return out


def is_black_key(pitch: int) -> bool:
    return int(pitch) % 12 in {1, 3, 6, 8, 10}


@dataclass
class NoteEvent:
    """A single MIDI note on the timeline (seconds)."""

    start: float
    end: float
    pitch: int
    velocity: int = 96
    confidence: float = 0.85
    source: str = "ai"
    level_db: float = 0.0       # local RMS level, annotated by the analyzer
    pitch_support: float = 0.0  # median neural voicing probability

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def name(self) -> str:
        return midi_to_name(self.pitch)

    def copy(self) -> "NoteEvent":
        return dataclasses.replace(self)

    def overlaps(self, start: float, end: float) -> float:
        lo = max(self.start, start)
        hi = min(self.end, end)
        return max(0.0, hi - lo)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NoteEvent":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            pitch=int(data["pitch"]),
            velocity=int(data.get("velocity", 96)),
            confidence=float(data.get("confidence", 0.85)),
            source=str(data.get("source", "ai")),
            level_db=float(data.get("level_db", 0.0)),
            pitch_support=float(data.get("pitch_support", 0.0)),
        )


@dataclass
class BreathRegion:
    start: float
    end: float
    score: float = 0.7

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BreathRegion":
        return cls(float(data["start"]), float(data["end"]), float(data.get("score", 0.7)))


@dataclass
class KeyInfo:
    tonic: int = 0
    mode: str = MAJOR_CODE
    confidence: float = 0.0
    bpm: float = 120.0

    @property
    def name(self) -> str:
        flats = self.mode == MINOR_CODE and self.tonic in {1, 3, 8, 10}
        names = FLAT_NAMES if flats else NOTE_NAMES
        return f"{names[self.tonic % 12]} {self.mode}"

    @property
    def chinese_mode(self) -> str:
        return "大调" if self.mode == MAJOR_CODE else "小调"

    @property
    def camelot(self) -> str:
        # Camelot wheel. Values are (B-major wheel position, A-minor wheel position).
        major = {0: "8B", 7: "9B", 2: "10B", 9: "11B", 4: "12B", 11: "1B", 6: "2B",
                 1: "3B", 8: "4B", 3: "5B", 10: "6B", 5: "7B"}
        minor = {9: "8A", 4: "9A", 11: "10A", 6: "11A", 1: "12A", 8: "1A", 3: "2A",
                 10: "3A", 5: "4A", 0: "5A", 7: "6A", 2: "7A"}
        return (major if self.mode == MAJOR_CODE else minor)[self.tonic % 12]

    @property
    def signature(self) -> str:
        """Number of sharps/flats for the key signature, e.g. '2♯' or '3♭'."""
        # Circle of fifths for major tonics (flats are negative).
        fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
        count = fifths[self.tonic % 12]
        if self.mode == MINOR_CODE:
            count = fifths[(self.tonic + 3) % 12]
        if count > 0:
            return f"{count}♯"
        if count < 0:
            return f"{-count}♭"
        return "0"


@dataclass
class AnalysisSettings:
    engine: str = "vocal"  # vocal | poly | mono | bass
    precision: str = "balanced"  # fast | balanced | precise
    onset_threshold: float = 0.48
    frame_threshold: float = 0.30
    min_note_ms: int = 90
    min_midi: int = 43  # G2
    max_midi: int = 86  # D6
    ignore_breath: bool = True
    breath_strength: float = 0.62
    noise_gate: float = 0.70
    detect_bpm: bool = True
    pitch_refine: bool = True
    workers: int = 0  # 0 = automatic (up to 8 cores)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class AnalysisResult:
    notes: List[NoteEvent] = field(default_factory=list)
    all_notes: List[NoteEvent] = field(default_factory=list)
    pre_breath_notes: List[NoteEvent] = field(default_factory=list)
    removed_breath_notes: List[NoteEvent] = field(default_factory=list)
    removed_noise_notes: List[NoteEvent] = field(default_factory=list)
    noise_reference_db: float = -120.0
    breaths: List[BreathRegion] = field(default_factory=list)
    key: KeyInfo = field(default_factory=KeyInfo)
    f0_times: Optional[object] = None
    f0_hz: Optional[object] = None
    f0_prob: Optional[object] = None
    duration: float = 0.0
    breath_count: int = 0
    breath_time: float = 0.0
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    engine_label: str = ""
    pitch_source: str = "pYIN"
    precision: str = "balanced"
    elapsed: float = 0.0
    cores_used: int = 1

    @property
    def note_count(self) -> int:
        return len(self.notes)


def merge_legato(notes: Sequence[NoteEvent], max_gap: float = 0.055) -> List[NoteEvent]:
    """Merge same-pitch notes separated by a tiny gap (scoop / onset jitter)."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch, n.end))
    out: List[NoteEvent] = []
    for note in ordered:
        if out:
            prev = out[-1]
            if (
                prev.pitch == note.pitch
                and note.start - prev.end <= max_gap
                and note.start >= prev.start
            ):
                prev.end = max(prev.end, note.end)
                prev.velocity = max(prev.velocity, note.velocity)
                prev.confidence = max(prev.confidence, note.confidence)
                continue
        out.append(note.copy())
    return out


def suppress_octave_ghosts(notes: Sequence[NoteEvent], iou_threshold: float = 0.55) -> List[NoteEvent]:
    """Remove doubled octave / twelfth ghosts that neural transcription can emit for a solo voice."""
    ordered = sorted(notes, key=lambda n: n.confidence, reverse=True)
    kept: List[NoteEvent] = []
    for note in ordered:
        drop = False
        for other in kept:
            lo = max(note.start, other.start)
            hi = min(note.end, other.end)
            inter = max(0.0, hi - lo)
            union = max(note.end, other.end) - min(note.start, other.start)
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold and abs(note.pitch - other.pitch) in (12, 19, 24, 31, 36):
                if note.confidence <= other.confidence + 0.08:
                    drop = True
                    break
        if not drop:
            kept.append(note)
    return sorted(kept, key=lambda n: (n.start, n.pitch))


def quantize_time(value: float, division: float, enabled: bool = True) -> float:
    if not enabled or division <= 0:
        return value
    return round(value / division) * division


def note_to_midi_bytes(
    notes: Iterable[NoteEvent],
    bpm: float = 120.0,
    track_name: str = "VoxTrace Vocal",
    key: Optional[KeyInfo] = None,
) -> bytes:
    """Render a type-1 Standard MIDI File with a tempo map and one vocal track."""
    notes = sorted((n for n in notes if n.end > n.start), key=lambda n: (n.start, n.pitch))
    ticks_per_beat = 480

    def tick(seconds: float) -> int:
        return max(0, int(round(seconds * bpm / 60.0 * ticks_per_beat)))

    def vlq(value: int) -> bytes:
        value = max(0, int(value))
        buf = [value & 0x7F]
        value >>= 7
        while value:
            buf.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(buf))

    def meta(kind: int, data: bytes) -> bytes:
        return bytes([0xFF, kind]) + vlq(len(data)) + data

    events: List[Tuple[int, int, bytes]] = []
    events.append((tick(0), 0, meta(0x03, track_name.encode("utf-8", "replace"))))
    if key is not None:
        # sf: signed number of sharps/flats, mi: 0 major / 1 minor.
        sig = key.signature
        signed = 0
        if sig.endswith("♯"):
            signed = int(sig[:-1])
        elif sig.endswith("♭"):
            signed = -int(sig[:-1])
        sf = signed if -128 <= signed <= 127 else 0
        events.append((tick(0), 1, meta(0x59, bytes([sf & 0xFF, 0 if key.mode == MAJOR_CODE else 1]))))
    if key is not None:
        use_flat = key.mode == MINOR_CODE and key.tonic in {1, 3, 8, 10}
        names = FLAT_NAMES if use_flat else NOTE_NAMES
        text = f"{names[key.tonic % 12]} {key.mode} | VoxTrace".encode("utf-8")
        events.append((tick(0), 2, meta(0x01, text)))

    for idx, note in enumerate(notes):
        pitch = max(0, min(127, int(note.pitch)))
        vel = max(1, min(127, int(note.velocity)))
        channel = 0
        events.append((tick(note.start), 10 + idx * 2, bytes([0x90 | channel, pitch, vel])))
        events.append((tick(note.end), 11 + idx * 2, bytes([0x80 | channel, pitch, 0])))

    events.append((tick(max((n.end for n in notes), default=0.0) + 0.5), 99999, meta(0x2F, b"")))

    # Stable ordering: meta first, then note-offs before note-ons at the same tick.
    def _priority(event: Tuple[int, int, bytes]) -> int:
        status = event[2][0]
        if status == 0xFF:
            return 0
        if status == 0x80:
            return 1
        return 2

    events.sort(key=lambda e: (e[0], _priority(e), e[1]))

    track = bytearray()
    prev_tick = 0
    for abs_tick, _, data in events:
        track.extend(vlq(abs_tick - prev_tick))
        track.extend(data)
        prev_tick = abs_tick

    # Format 1, two tracks: a conductor track with tempo, and the vocal track.
    header = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + (2).to_bytes(2, "big") + ticks_per_beat.to_bytes(2, "big")
    tempo_us = int(round(60_000_000.0 / max(1.0, float(bpm))))
    conductor_data = bytearray()
    conductor_data.extend(vlq(0) + meta(0x03, b"VoxTrace Tempo"))
    conductor_data.extend(vlq(0) + meta(0x51, tempo_us.to_bytes(3, "big")))
    conductor_data.extend(vlq(0) + meta(0x58, bytes([4, 2, 24, 8])))
    conductor_data.extend(vlq(0) + meta(0x2F, b""))
    conductor = b"MTrk" + len(conductor_data).to_bytes(4, "big") + bytes(conductor_data)
    vocal = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    return header + conductor + vocal
