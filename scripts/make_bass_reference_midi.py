"""Generate a small Bass MIDI reference track for the bundled demo."""

from __future__ import annotations

from pathlib import Path

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "voxtrace_demo_bass.mid"

BPM = 92.0
BEAT = 60.0 / BPM
# (start_beat, length_beats, midi)
NOTES = [
    (0.0, 1.5, 33), (1.5, 0.5, 33), (2.0, 1.0, 36), (3.0, 1.0, 40),
    (4.0, 1.5, 38), (5.5, 0.5, 36), (6.0, 1.0, 33), (7.0, 1.0, 36),
    (8.0, 1.5, 31), (9.5, 0.5, 33), (10.0, 1.0, 36), (11.0, 1.0, 38),
    (12.0, 1.5, 33), (13.5, 0.5, 36), (14.0, 2.0, 40), (16.0, 2.0, 33),
    (18.0, 1.0, 36), (19.0, 1.0, 38), (20.0, 1.5, 33), (21.5, 2.0, 33),
]


def main() -> None:
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("track_name", name="Bass", time=0))
    track.append(MetaMessage("set_tempo", tempo=int(60_000_000 / BPM), time=0))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    events = []
    for start_beat, length, pitch in NOTES:
        start = int(round(start_beat * 480))
        end = int(round((start_beat + length) * 480))
        velocity = 96 if start_beat % 4 == 0 else 82
        events.append((start, 1, Message("note_on", note=pitch, velocity=velocity, channel=0)))
        events.append((end, 0, Message("note_off", note=pitch, velocity=0, channel=0)))
    events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tick, _, message in events:
        track.append(message.copy(time=tick - last))
        last = tick
    track.append(MetaMessage("end_of_track", time=480))
    mid.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
