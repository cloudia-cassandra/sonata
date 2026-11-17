"""The galaxy layout: deterministic UMAP, scaled into [-1, 1], written as x/y per song."""

import numpy as np
import pandas as pd

from sonata.compute_galaxy import add_positions, embed_2d, scale_to_unit
from sonata.features import FEATURE_COLUMNS


def synthetic_features(n=40, seed=0):
    """Two clearly separate groups of songs."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    values[: n // 2] += 4
    return pd.DataFrame(values, columns=FEATURE_COLUMNS).assign(deezer_id=np.arange(1, n + 1))


def test_scale_to_unit_fills_the_widest_axis_and_keeps_the_aspect_ratio():
    scaled = scale_to_unit(np.array([[10.0, 5.0], [14.0, 6.0], [12.0, 7.0]]))
    assert scaled[:, 0].min() == -1 and scaled[:, 0].max() == 1
    assert np.allclose(scaled[:, 1], [-0.5, 0, 0.5])      # 2 wide in y vs 4 in x


def test_scale_to_unit_handles_a_single_point():
    assert np.allclose(scale_to_unit(np.array([[3.0, 4.0]])), [[0, 0]])


def test_embed_2d_is_deterministic_and_separates_groups():
    features = synthetic_features()
    first, second = embed_2d(features), embed_2d(features)
    pd.testing.assert_frame_equal(first, second)
    xy = first[["x", "y"]].to_numpy()
    assert np.abs(xy).max() <= 1
    # Songs in the same group sit closer to their own group's center than to the other's
    a, b = xy[:20].mean(axis=0), xy[20:].mean(axis=0)
    assert all(np.linalg.norm(p - a) < np.linalg.norm(p - b) for p in xy[:20])


def test_add_positions_writes_xy_and_clears_stale_ones():
    library = [{"deezer_id": 1, "x": 9, "y": 9}, {"deezer_id": 2, "x": 9, "y": 9}]
    placed = add_positions(library, pd.DataFrame({"deezer_id": [1], "x": [0.5], "y": [-0.25]}))
    assert placed == 1
    assert library[0] == {"deezer_id": 1, "x": 0.5, "y": -0.25}
    assert library[1] == {"deezer_id": 2}
