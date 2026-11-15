"""
Analyzes each preview in data/library.json with librosa.

For every track this produces:
  (a) a star map, data/stars/<deezer_id>.json: one star per detected onset,
      {t: seconds, pitch: MIDI note of the strongest CQT bin, size: RMS loudness 0-1}
  (b) a feature vector (tempo, mean RMS, spectral centroid, 13 MFCC means,
      12 chroma means), all collected into data/features.parquet
and adds readable labels to library.json: tempo_label (slow/medium/fast),
energy_label (low/medium/high) and brightness_label (dark/warm/bright).

Run from the project root with: python -m sonata.analyze
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = PROJECT_ROOT / "data" / "library.json"
STARS_DIR = PROJECT_ROOT / "data" / "stars"
FEATURES_PATH = PROJECT_ROOT / "data" / "features.parquet"

SAMPLE_RATE = 22050
HOP = 512                    # shared by every frame-based feature so frames line up
CQT_FMIN_MIDI = 24           # C1, librosa's default CQT starting note
CQT_BINS = 84                # 7 octaves, one bin per semitone: MIDI 24-107
PITCH_WINDOW = 3             # frames after an onset to read the pitch from (~70 ms)
N_MFCC = 13
CHROMA_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

# Tempo has a natural absolute scale, so it uses fixed BPM cut-offs.
SLOW_BELOW_BPM = 90
FAST_FROM_BPM = 120
# RMS and spectral centroid depend heavily on mixing and mastering, so energy and
# brightness are split into thirds of this library: the labels are relative to your songs.
ENERGY_LABELS = ("low", "medium", "high")
BRIGHTNESS_LABELS = ("dark", "warm", "bright")

FEATURE_COLUMNS = (
    ["tempo", "rms_mean", "spectral_centroid"]
    + [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
    + [f"chroma_{name}" for name in CHROMA_NAMES]
)

log = logging.getLogger("analyze")


def analyze_audio(y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict:
    """Returns {"stars": [...], "features": {...}} for one mono signal."""
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, n_bins=CQT_BINS,
                             fmin=librosa.midi_to_hz(CQT_FMIN_MIDI)))
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP)

    n_frames = min(cqt.shape[1], len(rms))
    peak_rms = float(rms.max()) or 1.0
    stars = []
    for frame in onsets:
        if frame >= n_frames:
            continue
        # The attack itself is noisy, so read the strongest bin over the next few frames
        window = cqt[:, frame:min(frame + PITCH_WINDOW, n_frames)].sum(axis=1)
        stars.append({
            "t": round(float(librosa.frames_to_time(frame, sr=sr, hop_length=HOP)), 3),
            "pitch": CQT_FMIN_MIDI + int(window.argmax()),
            "size": round(float(rms[frame]) / peak_rms, 3),
        })

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
    chroma = librosa.feature.chroma_cqt(C=cqt, sr=sr, hop_length=HOP, bins_per_octave=12,
                                        fmin=librosa.midi_to_hz(CQT_FMIN_MIDI))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP)[0]

    values = [float(tempo), float(rms.mean()), float(centroid.mean()),
              *mfcc.mean(axis=1).tolist(), *chroma.mean(axis=1).tolist()]
    return {"stars": stars, "features": dict(zip(FEATURE_COLUMNS, values))}


def tempo_label(bpm: float) -> str:
    if bpm < SLOW_BELOW_BPM:
        return "slow"
    return "fast" if bpm >= FAST_FROM_BPM else "medium"


def tertile_labels(values: pd.Series, labels: tuple) -> pd.Series:
    """Splits values into thirds (lowest third gets labels[0])."""
    ranks = values.rank(method="first", pct=True)
    return pd.cut(ranks, bins=[0, 1 / 3, 2 / 3, 1], labels=list(labels)).astype(str)


def add_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Returns deezer_id plus the three readable labels for each analyzed track."""
    return pd.DataFrame({
        "deezer_id": features["deezer_id"],
        "tempo_label": features["tempo"].map(tempo_label),
        "energy_label": tertile_labels(features["rms_mean"], ENERGY_LABELS),
        "brightness_label": tertile_labels(features["spectral_centroid"], BRIGHTNESS_LABELS),
    })


def analyze_track(track: Dict) -> Optional[Dict]:
    """Loads one preview, writes its star map, and returns its feature row (None on failure)."""
    path = PROJECT_ROOT / track["preview_path"]
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as error:  # missing file, corrupt MP3, decoder error
        log.warning("Could not load %s (%s - %s): %s", path.name, track["artist"], track["title"], error)
        return None
    if y.size == 0 or not np.any(y):
        log.warning("Silent or empty preview, skipped: %s - %s", track["artist"], track["title"])
        return None

    result = analyze_audio(y, sr)
    star_path = STARS_DIR / f"{track['deezer_id']}.json"
    star_path.write_text(json.dumps(result["stars"], separators=(",", ":")) + "\n", encoding="utf-8")
    return {"deezer_id": track["deezer_id"], **result["features"]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    library: List[Dict] = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    STARS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for number, track in enumerate(library, start=1):
        row = analyze_track(track)
        if row:
            rows.append(row)
            log.info("[%d/%d] %s - %s: %.0f BPM", number, len(library), track["artist"], track["title"], row["tempo"])

    features = pd.DataFrame(rows, columns=["deezer_id", *FEATURE_COLUMNS])
    features.to_parquet(FEATURES_PATH, index=False)

    labels = add_labels(features).set_index("deezer_id").to_dict("index") if rows else {}
    for track in library:
        # Tracks that failed analysis get no labels rather than stale ones
        for key in ("tempo_label", "energy_label", "brightness_label"):
            track.pop(key, None)
        track.update(labels.get(track["deezer_id"], {}))
    LIBRARY_PATH.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Star maps of songs that are no longer in the library (or failed analysis this run)
    analyzed = {f"{row['deezer_id']}.json" for row in rows}
    stale = [path for path in STARS_DIR.glob("*.json") if path.name not in analyzed]
    for path in stale:
        path.unlink()
    if stale:
        log.info("Removed %d star maps no longer in the library", len(stale))

    log.info("Analyzed %d of %d tracks; wrote %s, %s and labels in %s", len(rows), len(library),
             STARS_DIR.relative_to(PROJECT_ROOT), FEATURES_PATH.relative_to(PROJECT_ROOT),
             LIBRARY_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
