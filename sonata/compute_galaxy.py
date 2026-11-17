"""
Lays the whole library out as a galaxy: one star per song, placed so songs that
sound alike sit close together.

The 28 audio features in data/features.parquet are standardized (the same
z-scores the content recommender uses), reduced to 2D with UMAP using cosine
distance, and scaled into [-1, 1] keeping the aspect ratio. Each song's position
is saved as x and y in data/library.json; songs without features get none and
are left out of the galaxy.

UMAP is run with a fixed random_state, so the layout is the same on every run.

Run from the project root with: python -m sonata.compute_galaxy
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sonata.features import FEATURES_PATH
from sonata.recommender import standardize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = PROJECT_ROOT / "data" / "library.json"

RANDOM_STATE = 42
N_NEIGHBORS = 15     # how many similar songs shape each song's neighborhood
MIN_DIST = 0.1       # how tightly similar songs may pack together
DECIMALS = 4

log = logging.getLogger("compute_galaxy")


def scale_to_unit(points: np.ndarray) -> np.ndarray:
    """Centers the points and scales them so the widest axis spans [-1, 1]; keeps the aspect ratio."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    centered = points - (lo + hi) / 2
    extent = float((hi - lo).max()) / 2
    return centered / extent if extent > 0 else centered


def embed_2d(features: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Returns deezer_id, x, y for every row of the feature table."""
    import umap   # imported here: it's slow to load and only this script needs it

    z = standardize(features).to_numpy()
    reducer = umap.UMAP(n_components=2, n_neighbors=min(N_NEIGHBORS, len(z) - 1), min_dist=MIN_DIST,
                        metric="cosine", random_state=random_state,
                        n_jobs=1,                    # a fixed seed runs single-threaded anyway
                        compatibility_layout=True)   # pin the layout algorithm so a UMAP upgrade can't move the stars
    xy = scale_to_unit(reducer.fit_transform(z))
    return pd.DataFrame({"deezer_id": features["deezer_id"].to_numpy(),
                         "x": xy[:, 0].round(DECIMALS), "y": xy[:, 1].round(DECIMALS)})


def add_positions(library: List[Dict], positions: pd.DataFrame) -> int:
    """Writes x, y into each track in place; tracks without a position lose any stale one. Returns how many got one."""
    by_id = {int(row.deezer_id): (float(row.x), float(row.y)) for row in positions.itertuples()}
    placed = 0
    for track in library:
        track.pop("x", None)
        track.pop("y", None)
        position: Optional[tuple] = by_id.get(track["deezer_id"])
        if position:
            track["x"], track["y"] = position
            placed += 1
    return placed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    library: List[Dict] = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    features = pd.read_parquet(FEATURES_PATH)
    features = features[features["deezer_id"].isin({t["deezer_id"] for t in library})].reset_index(drop=True)
    if len(features) < 3:
        raise SystemExit(f"Need at least 3 analyzed songs for a galaxy, found {len(features)}. Run python -m sonata.analyze first.")

    log.info("Running UMAP on %d songs x %d features (random_state=%d)", len(features), features.shape[1] - 1, RANDOM_STATE)
    placed = add_positions(library, embed_2d(features))
    LIBRARY_PATH.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Placed %d of %d songs; wrote x, y into %s", placed, len(library), LIBRARY_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
