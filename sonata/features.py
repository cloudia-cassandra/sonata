"""
The audio feature table's location and column layout.

Kept separate from sonata.analyze so that code which only reads
data/features.parquet (recommender, rag, evaluate) doesn't import librosa.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features.parquet"

N_MFCC = 13
CHROMA_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

FEATURE_COLUMNS = (
    ["tempo", "rms_mean", "spectral_centroid"]
    + [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
    + [f"chroma_{name}" for name in CHROMA_NAMES]
)
