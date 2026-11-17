"""
Content-based recommender on the audio features from sonata.analyze.

Every feature column in data/features.parquet is standardized (z-scored) across
the whole library, so tempo in BPM and chroma energy in 0-1 count on the same
scale. Each candidate (liked=false) is compared with every liked song by cosine
similarity, scored by its closest liked song, and explained by the two features
that contribute most to that similarity, i.e. where both songs sit furthest
from average in the same direction.

Run from the project root with: python -m sonata.recommender [-k 10]
"""

import argparse
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from sonata.features import CHROMA_NAMES, FEATURES_PATH, FEATURE_COLUMNS
from sonata.catalog import LIBRARY_PATH, load_library

SHARED_FEATURES = 2

# Words for "both songs are above / below the library average" on each feature
_FEATURE_WORDS = {
    "tempo": ("tempo", "faster", "slower"),
    "rms_mean": ("loudness", "louder", "quieter"),
    "spectral_centroid": ("brightness", "brighter", "darker"),
    **{f"mfcc_{i}": (f"timbre (MFCC {i})", "higher", "lower") for i in range(1, 14)},
    **{f"chroma_{name}": (f"{name.replace('s', '♯')} in the harmony", "more", "less")
       for name in CHROMA_NAMES},
}


def standardize(features: pd.DataFrame) -> pd.DataFrame:
    """Z-scores each feature column; a constant column becomes all zeros instead of NaN."""
    values = features[FEATURE_COLUMNS]
    std = values.std(ddof=0).replace(0, 1)
    return (values - values.mean()) / std


def describe_shared(feature: str, z_candidate: float, z_liked: float) -> str:
    """Readable phrase like "brightness: both brighter than average (+1.3σ, +1.1σ)"."""
    name, above, below = _FEATURE_WORDS[feature]
    if np.sign(z_candidate) == np.sign(z_liked) and z_candidate != 0:
        direction = f"both {above if z_candidate > 0 else below} than average"
    else:
        direction = "close to each other"
    return f"{name}: {direction} ({z_candidate:+.1f}σ, {z_liked:+.1f}σ)"


def recommend(features: pd.DataFrame, library: List[Dict], k: int = 10) -> List[Dict]:
    """Ranks liked=false songs by cosine similarity to their closest liked song.

    Returns up to k dicts with the song, its similarity, the closest liked song,
    and the two features where the pair is most alike.
    """
    tracks = {track["deezer_id"]: track for track in library}
    features = features[features["deezer_id"].isin(tracks)].reset_index(drop=True)
    z = standardize(features).to_numpy()
    ids = features["deezer_id"].to_numpy()

    liked_mask = np.array([tracks[i]["liked"] for i in ids])
    if not liked_mask.any() or liked_mask.all():
        return []

    norms = np.linalg.norm(z, axis=1, keepdims=True)
    unit = z / np.where(norms == 0, 1, norms)
    candidates, liked = unit[~liked_mask], unit[liked_mask]
    candidate_ids, liked_ids = ids[~liked_mask], ids[liked_mask]
    z_candidates, z_liked = z[~liked_mask], z[liked_mask]

    similarity = candidates @ liked.T                   # (n_candidates, n_liked)
    closest = similarity.argmax(axis=1)
    best = similarity[np.arange(len(candidates)), closest]
    order = np.argsort(-best, kind="stable")[:k]

    results = []
    for row in order:
        match = closest[row]
        # Per-feature terms of the cosine; they sum to the similarity
        contributions = candidates[row] * liked[match]
        top = np.argsort(-contributions, kind="stable")[:SHARED_FEATURES]
        song, liked_song = tracks[candidate_ids[row]], tracks[liked_ids[match]]
        results.append({
            "song": song,
            "similarity": float(best[row]),
            "closest_liked": liked_song,
            "shared_features": [
                {
                    "feature": FEATURE_COLUMNS[i],
                    "candidate_z": float(z_candidates[row, i]),
                    "liked_z": float(z_liked[match, i]),
                    "contribution": float(contributions[i]),
                    "description": describe_shared(FEATURE_COLUMNS[i], z_candidates[row, i], z_liked[match, i]),
                }
                for i in top
            ],
        })
    return results


def load_and_recommend(k: int = 10,
                       features_path: Union[str, Path] = FEATURES_PATH,
                       library_path: Union[str, Path] = LIBRARY_PATH) -> List[Dict]:
    return recommend(pd.read_parquet(features_path), load_library(library_path), k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend new songs that sound like the ones you like")
    parser.add_argument("-k", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()

    print(format_recommendations(load_and_recommend(args.k)))


def format_recommendations(results: List[Dict]) -> str:
    """Readable text: each song, its closest liked song, and the two shared features."""
    lines = []
    for rank, result in enumerate(results, start=1):
        song, liked = result["song"], result["closest_liked"]
        lines += ["" if rank > 1 else None,
                  f"{rank}. {song['title']} by {song['artist']} ({song['genre'] or 'unknown genre'})",
                  f"   Similarity {result['similarity']:.2f} to {liked['title']} by {liked['artist']}"]
        lines += [f"     - {shared['description']}" for shared in result["shared_features"]]
    return "\n".join(line for line in lines if line is not None)


if __name__ == "__main__":
    main()
