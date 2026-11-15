import numpy as np
import pandas as pd
import librosa

from sonata.analyze import FEATURE_COLUMNS, SAMPLE_RATE, add_labels, analyze_audio, tempo_label


def note_sequence(midi_notes, amplitudes, note_seconds=0.5, sr=SAMPLE_RATE):
    """Sine notes with sharp attacks and silence between them, so each one is a clear onset.

    Starts with 0.25 s of silence (an onset at frame 0 has nothing before it to compare to),
    and each note fades out smoothly so its ending doesn't click like a second onset.
    """
    gap = np.zeros(int(0.25 * sr))
    pieces = [gap]
    for note, amplitude in zip(midi_notes, amplitudes):
        t = np.arange(int(note_seconds * sr)) / sr
        release = np.clip((note_seconds - t) / 0.1, 0, 1)
        tone = amplitude * np.sin(2 * np.pi * librosa.midi_to_hz(note) * t) * np.exp(-3 * t) * release
        pieces += [tone, gap]
    return np.concatenate(pieces).astype(np.float32)


def test_star_map_finds_each_note_with_its_pitch_and_loudness():
    notes, amps = [60, 67, 72, 64], [0.3, 0.9, 0.6, 0.15]
    stars = analyze_audio(note_sequence(notes, amps))["stars"]

    assert [s["pitch"] for s in stars] == notes
    # Notes start every 0.75 s after the leading gap; onsets should land within ~50 ms
    assert np.allclose([s["t"] for s in stars], [0.25, 1.0, 1.75, 2.5], atol=0.05)
    sizes = [s["size"] for s in stars]
    assert all(0 <= size <= 1 for size in sizes)
    assert sizes.index(max(sizes)) == 1 and sizes.index(min(sizes)) == 3


def test_feature_vector_has_all_columns():
    features = analyze_audio(note_sequence([60, 64, 67], [0.5, 0.5, 0.5]))["features"]
    assert list(features) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 3 + 13 + 12
    assert all(np.isfinite(value) for value in features.values())


def test_labels():
    assert [tempo_label(bpm) for bpm in (70, 90, 119.9, 120, 160)] == ["slow", "medium", "medium", "fast", "fast"]
    features = pd.DataFrame({"deezer_id": [1, 2, 3, 4, 5, 6], "tempo": [100] * 6,
                             "rms_mean": [0.1, 0.6, 0.3, 0.5, 0.2, 0.4],
                             "spectral_centroid": [3000, 1000, 2000, 1500, 2500, 500]})
    labels = add_labels(features)
    assert list(labels["energy_label"]) == ["low", "high", "medium", "high", "low", "medium"]
    assert list(labels["brightness_label"]) == ["bright", "dark", "warm", "warm", "bright", "dark"]
