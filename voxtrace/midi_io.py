"""Read Standard MIDI Files for the Bass MIDI reference lane."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .notes import NoteEvent


@dataclass
class MidiTrackData:
    index: int
    name: str
    notes: List[NoteEvent] = field(default_factory=list)
    is_drum: bool = False

    @property
    def median_pitch(self) -> float:
        if not self.notes:
            return 60.0
        pitches = sorted(n.pitch for n in self.notes)
        return float(pitches[len(pitches) // 2])

    def merged_notes(self) -> List[NoteEvent]:
        return sorted((n.copy() for n in self.notes), key=lambda n: (n.start, n.pitch))


@dataclass
class MidiFileData:
    path: str
    filename: str
    tracks: List[MidiTrackData]
    duration: float
    bpm: float

    @property
    def all_notes(self) -> List[NoteEvent]:
        return sorted(
            (n.copy() for t in self.tracks if not t.is_drum for n in t.notes),
            key=lambda n: (n.start, n.pitch),
        )

    def preferred_track_index(self) -> int:
        """Prefer a track whose name says bass, otherwise the lowest non-drum part."""
        for track in self.tracks:
            lowered = track.name.casefold()
            if any(token in lowered for token in ("bass", "ベース", "低音", "basso")):
                return track.index
        candidates = [t for t in self.tracks if not t.is_drum and t.notes]
        if not candidates:
            candidates = [t for t in self.tracks if t.notes]
        if not candidates:
            return -1
        return min(candidates, key=lambda t: t.median_pitch).index

    def track_by_index(self, index: int) -> MidiTrackData | None:
        for track in self.tracks:
            if track.index == index:
                return track
        return None


def _tempo_converter(tempos: Sequence[Tuple[int, int]], ticks_per_beat: int, default_tempo: int = 500000):
    unique: Dict[int, int] = {0: default_tempo}
    for tick, tempo in tempos:
        if tick >= 0 and tempo > 0:
            unique[int(tick)] = int(tempo)
    ticks = sorted(unique)
    tempos_us = [unique[t] for t in ticks]
    seconds = [0.0]
    for i in range(1, len(ticks)):
        delta_ticks = ticks[i] - ticks[i - 1]
        seconds.append(seconds[-1] + delta_ticks * tempos_us[i - 1] / 1_000_000.0 / ticks_per_beat)

    def convert(tick: int) -> float:
        idx = max(0, bisect.bisect_right(ticks, tick) - 1)
        return seconds[idx] + (tick - ticks[idx]) * tempos_us[idx] / 1_000_000.0 / ticks_per_beat

    return convert


def read_midi_notes(path: str | Path) -> MidiFileData:
    """Parse note events, track names and a global tempo map from a MIDI file."""
    import mido

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    midi = mido.MidiFile(str(source))
    ticks_per_beat = int(midi.ticks_per_beat or 480)
    if ticks_per_beat <= 0:
        ticks_per_beat = 480

    # Global tempo map, independent of track ordering.
    tempo_events: List[Tuple[int, int]] = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.is_meta and message.type == "set_tempo":
                tempo_events.append((tick, int(message.tempo)))
    tempo_events.sort()
    tick_to_seconds = _tempo_converter(tempo_events, ticks_per_beat)
    initial_tempo = tempo_events[0][1] if tempo_events else 500000

    tracks: List[MidiTrackData] = []
    for index, midi_track in enumerate(midi.tracks):
        name = ""
        tick = 0
        active: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        raw_notes: List[Tuple[int, int, int, int, int]] = []
        drum_channel = False
        for message in midi_track:
            tick += int(message.time)
            if message.is_meta:
                if message.type == "track_name" and not name:
                    name = str(message.name)
                continue
            if not hasattr(message, "channel"):
                continue
            channel = int(message.channel)
            if channel == 9:
                drum_channel = True
            key = (channel, int(message.note)) if hasattr(message, "note") else None
            if key is None:
                continue
            if message.type == "note_on" and int(message.velocity) > 0:
                active.setdefault(key, []).append((tick, int(message.velocity), channel))
            elif message.type in ("note_off", "note_on"):
                starts = active.get(key)
                if not starts:
                    continue
                start_tick, velocity, note_channel = starts.pop(0)
                raw_notes.append((start_tick, tick, key[1], velocity, note_channel))
        if not name:
            name = f"Track {index + 1}"
        notes: List[NoteEvent] = []
        for start_tick, end_tick, pitch, velocity, channel in raw_notes:
            if end_tick <= start_tick:
                continue
            start = tick_to_seconds(start_tick)
            end = tick_to_seconds(end_tick)
            notes.append(
                NoteEvent(
                    start=round(max(0.0, start), 4),
                    end=round(max(start, end), 4),
                    pitch=max(0, min(127, pitch)),
                    velocity=max(1, min(127, velocity)),
                    confidence=0.8,
                    source="midi",
                )
            )
        notes.sort(key=lambda n: (n.start, n.pitch))
        tracks.append(MidiTrackData(index=index, name=name, notes=notes, is_drum=drum_channel))

    tracks = [t for t in tracks if t.notes]
    if not tracks:
        raise ValueError("MIDI 文件中没有可显示的音符。")
    duration = max((n.end for t in tracks for n in t.notes), default=0.0)
    return MidiFileData(
        path=str(source),
        filename=source.name,
        tracks=tracks,
        duration=float(duration),
        bpm=float(round(60_000_000.0 / initial_tempo, 1)),
    )
