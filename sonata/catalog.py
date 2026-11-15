"""
Loads Sonata's song catalog.

data/library.json (built by sonata.build_library) holds real tracks but no audio
features yet, so the CLI still scores the placeholder songs in data/songs.csv.
Once audio analysis adds mood/energy/valence/acousticness, the CLI switches to
load_library() and songs.csv is removed.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = PROJECT_ROOT / "data" / "library.json"

REQUIRED_FIELDS = ("deezer_id", "title", "artist", "genre", "cover", "preview_path", "liked")


def load_library(path: Union[str, Path] = LIBRARY_PATH) -> List[Dict]:
    """Returns the library's tracks, each with an `id` (the Deezer id) like the CSV songs have."""
    path = Path(path)
    tracks = json.loads(path.read_text(encoding="utf-8"))
    for index, track in enumerate(tracks):
        missing = [name for name in REQUIRED_FIELDS if name not in track]
        if missing:
            raise ValueError(f"{path}: track {index} is missing {', '.join(missing)}")
        track["id"] = track["deezer_id"]
    return tracks
