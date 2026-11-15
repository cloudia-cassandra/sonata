import numpy as np
import pandas as pd
import pytest

from sonata.analyze import FEATURE_COLUMNS
from sonata.recommender import describe_shared, recommend, standardize


def make_data(vectors, liked):
    """vectors: {deezer_id: {feature: value}}; unspecified features are 0."""
    rows = [{"deezer_id": i, **{c: 0.0 for c in FEATURE_COLUMNS}, **v} for i, v in vectors.items()]
    library = [{"deezer_id": i, "title": f"Song {i}", "artist": "A", "genre": None, "liked": i in liked}
               for i in vectors]
    return pd.DataFrame(rows), library


def test_standardize_zero_mean_unit_std_and_constant_columns_are_zero():
    features, _ = make_data({1: {"tempo": 80}, 2: {"tempo": 120}, 3: {"tempo": 160}}, liked=set())
    z = standardize(features)
    assert np.allclose(z["tempo"], [-1.2247, 0, 1.2247], atol=1e-3)
    assert (z["rms_mean"] == 0).all()


def test_ranks_only_candidates_by_closest_liked_song():
    features, library = make_data({
        1: {"tempo": 170, "spectral_centroid": 3500},   # liked: fast and bright
        2: {"tempo": 70, "rms_mean": 0.6},              # liked: slow and loud
        3: {"tempo": 165, "spectral_centroid": 3400},   # candidate like 1
        4: {"tempo": 75, "rms_mean": 0.55},             # candidate like 2
        5: {"mfcc_5": 9},                               # candidate like neither
    }, liked={1, 2})

    results = recommend(features, library, k=10)

    assert [r["song"]["deezer_id"] for r in results][:2] in ([3, 4], [4, 3])
    assert results[-1]["song"]["deezer_id"] == 5
    assert all(not r["song"]["liked"] for r in results)
    closest = {r["song"]["deezer_id"]: r["closest_liked"]["deezer_id"] for r in results}
    assert closest[3] == 1 and closest[4] == 2
    assert all(-1 - 1e-9 <= r["similarity"] <= 1 + 1e-9 for r in results)


def test_shared_features_are_the_top_two_cosine_terms():
    features, library = make_data({
        1: {"tempo": 170, "spectral_centroid": 3500},
        2: {"tempo": 70},
        3: {"tempo": 165, "spectral_centroid": 3400},
        4: {"rms_mean": 0.9},
    }, liked={1, 2})

    top = next(r for r in recommend(features, library) if r["song"]["deezer_id"] == 3)
    assert {s["feature"] for s in top["shared_features"]} == {"tempo", "spectral_centroid"}
    contributions = [s["contribution"] for s in top["shared_features"]]
    assert contributions == sorted(contributions, reverse=True)


def test_k_limits_results_and_no_liked_songs_gives_nothing():
    features, library = make_data({i: {"tempo": 60 + 10 * i} for i in range(1, 7)}, liked={1})
    assert len(recommend(features, library, k=3)) == 3
    features, library = make_data({1: {"tempo": 80}, 2: {"tempo": 90}}, liked=set())
    assert recommend(features, library) == []


@pytest.mark.parametrize("feature,zc,zl,expected", [
    ("spectral_centroid", 1.3, 1.1, "brightness: both brighter than average (+1.3σ, +1.1σ)"),
    ("tempo", -0.8, -1.2, "tempo: both slower than average (-0.8σ, -1.2σ)"),
    ("chroma_Fs", 2.0, 1.5, "F♯ in the harmony: both more than average (+2.0σ, +1.5σ)"),
])
def test_describe_shared(feature, zc, zl, expected):
    assert describe_shared(feature, zc, zl) == expected
