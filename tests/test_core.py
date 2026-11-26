"""Core unit and integration tests for VoxTrace Studio."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voxtrace.audio_io import load_audio
from voxtrace.engine import (
    _filter_breaths,
    _filter_noise_notes,
    _noise_gate_db,
    _refine_tempo_with_notes,
    analyze_audio,
    refilter_breath,
)
from voxtrace.midi_io import read_midi_notes
from voxtrace.notes import (
    AnalysisResult,
    AnalysisSettings,
    BreathRegion,
    KeyInfo,
    NoteEvent,
    merge_legato,
    note_to_midi_bytes,
    suppress_octave_ghosts,
)

DEMO = ROOT / "demo" / "voxtrace_demo.wav"
BASS_DEMO = ROOT / "demo" / "voxtrace_bass_demo.wav"


class NoteTests(unittest.TestCase):
    def test_key_signature_and_camelot(self):
        self.assertEqual(KeyInfo(9, "minor").name, "A minor")
        self.assertEqual(KeyInfo(9, "minor").camelot, "8A")
        self.assertEqual(KeyInfo(0, "major").signature, "0")
        self.assertEqual(KeyInfo(7, "major").signature, "1♯")
        self.assertEqual(KeyInfo(5, "major").signature, "1♭")
        self.assertEqual(KeyInfo(9, "minor").signature, "0")

    def test_merge_legato(self):
        notes = [NoteEvent(0.0, 0.5, 60), NoteEvent(0.52, 1.0, 60), NoteEvent(1.2, 1.6, 62)]
        merged = merge_legato(notes, max_gap=0.04)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0].end, 1.0)

    def test_octave_ghost_suppression(self):
        notes = [
            NoteEvent(0.0, 1.0, 60, confidence=0.9),
            NoteEvent(0.02, 0.98, 72, confidence=0.6),
        ]
        kept = suppress_octave_ghosts(notes)
        self.assertEqual([n.pitch for n in kept], [60])

    def test_breath_gate(self):
        notes = [NoteEvent(0.55, 1.35, 64, confidence=0.8), NoteEvent(2.0, 2.5, 67)]
        breaths = [BreathRegion(0.5, 1.5, 0.9)]
        kept, removed = _filter_breaths(notes, breaths, strength=0.62)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].pitch, 67)
        self.assertEqual(len(removed), 1)
        result = AnalysisResult(
            notes=kept,
            pre_breath_notes=[n.copy() for n in notes],
            breaths=breaths,
        )
        restored = refilter_breath(result, AnalysisSettings(ignore_breath=False))
        self.assertEqual(len(restored), 2)

    def test_noise_gate_removes_silent_notes(self):
        gate = _noise_gate_db(-12.0, 0.70)
        quiet = NoteEvent(1.0, 1.2, 60, level_db=-50.0, pitch_support=0.8)
        supported = NoteEvent(2.0, 2.4, 64, level_db=-15.0, pitch_support=0.9)
        weak = NoteEvent(3.0, 3.2, 67, level_db=gate + 2.0, pitch_support=0.01)
        kept, removed = _filter_noise_notes([quiet, supported, weak], gate)
        self.assertIn(supported, kept)
        self.assertIn(quiet, removed)
        self.assertIn(weak, removed)
        self.assertEqual(_noise_gate_db(-12.0, 0.0), -120.0)

    def test_midi_export(self):
        try:
            from mido import MidiFile
        except ImportError:
            self.skipTest("mido not installed")
        import tempfile

        notes = [NoteEvent(0.0, 0.5, 60, 100), NoteEvent(0.5, 1.25, 64, 90)]
        data = note_to_midi_bytes(notes, bpm=92, key=KeyInfo(9, "minor"))
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
            path = Path(handle.name)
            handle.write(data)
        try:
            midi = MidiFile(path)
            self.assertEqual(midi.type, 1)
            self.assertEqual(len(midi.tracks), 2)
            key_events = [m for m in midi.tracks[1] if m.is_meta and m.type == "key_signature"]
            self.assertTrue(key_events)
            self.assertEqual(key_events[0].key, "Am")
            note_ons = [m for m in midi.tracks[1] if m.type == "note_on" and m.velocity > 0]
            self.assertEqual([m.note for m in note_ons], [60, 64])
            self.assertAlmostEqual(midi.length, 1.75, places=1)
        finally:
            path.unlink(missing_ok=True)


class TempoTests(unittest.TestCase):
    def test_note_grid_disambiguates_vocal_tempo(self):
        # 97 BPM attacks on the eighth-note grid; the base estimate is the
        # triplet-biased 95.7 value librosa often reports for a solo voice.
        beat = 60.0 / 97.0
        notes = [
            NoteEvent(index * beat / 2.0 + 0.004 * (index % 3), index * beat / 2.0 + 0.25, 60)
            for index in range(160)
        ]
        refined = _refine_tempo_with_notes(95.703125, notes, 60.0)
        self.assertAlmostEqual(refined, 97.0, places=1)


class BassMidiTests(unittest.TestCase):
    def test_read_demo_bass_midi(self):
        path = ROOT / "demo" / "voxtrace_demo_bass.mid"
        data = read_midi_notes(path)
        self.assertEqual(len(data.tracks), 1)
        self.assertEqual(data.tracks[0].name, "Bass")
        self.assertEqual(len(data.all_notes), 20)
        self.assertAlmostEqual(data.bpm, 92.0, places=1)
        self.assertEqual(data.preferred_track_index(), data.tracks[0].index)
        self.assertTrue(all(28 <= n.pitch <= 48 for n in data.all_notes))
        self.assertTrue(all(n.end > n.start for n in data.all_notes))


class BassEngineTests(unittest.TestCase):
    def test_bass_mode_transcribes_low_notes(self):
        audio = load_audio(BASS_DEMO)
        settings = AnalysisSettings(
            engine="bass",
            precision="balanced",
            min_midi=28,
            max_midi=55,
            min_note_ms=70,
            ignore_breath=False,
            noise_gate=0.2,
        )
        result = analyze_audio(audio, settings, workers=8)
        self.assertGreaterEqual(len(result.notes), 10)
        self.assertLessEqual(len(result.notes), 26)
        self.assertEqual(result.pitch_source, "pYIN")
        self.assertTrue(all(24 <= n.pitch <= 60 for n in result.notes))
        self.assertEqual(result.breath_count, 0)
        self.assertFalse(any(n.pitch > 55 for n in result.notes))


class AudioTests(unittest.TestCase):
    def test_demo_load(self):
        audio = load_audio(DEMO)
        self.assertAlmostEqual(audio.duration, 20.43, places=1)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertGreater(len(audio.analysis_samples), 1000)
        self.assertTrue(Path(audio.playback_path).exists())


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio = load_audio(DEMO)
        cls.result = analyze_audio(cls.audio, AnalysisSettings(), workers=8)

    def test_detects_notes_and_key(self):
        self.assertGreaterEqual(len(self.result.notes), 15)
        self.assertLessEqual(len(self.result.notes), 28)
        self.assertEqual(self.result.key.name, "A minor")
        self.assertGreater(self.result.key.confidence, 70)
        pitches = {n.pitch for n in self.result.notes}
        self.assertTrue({69, 72, 76}.issubset(pitches))
        self.assertEqual(self.result.pitch_source, "RMVPE")

    def test_breath_regions_present(self):
        self.assertGreater(len(self.result.breaths), 0)
        self.assertGreater(self.result.breath_time, 0.5)
        for breath in self.result.breaths:
            self.assertLess(breath.start, breath.end)

    def test_note_timing_is_sane(self):
        for note in self.result.notes:
            self.assertGreaterEqual(note.start, 0.0)
            self.assertLessEqual(note.end, self.audio.duration + 0.01)
            self.assertGreater(note.end, note.start)
            self.assertGreaterEqual(note.velocity, 1)
            self.assertLessEqual(note.velocity, 127)
        for left, right in zip(self.result.notes, self.result.notes[1:]):
            self.assertGreaterEqual(right.start + 0.001, left.start)


if __name__ == "__main__":
    unittest.main(verbosity=2)
