import json

import pytest

from sonata.catalog import load_library


def write_library(tmp_path, tracks):
    path = tmp_path / "library.json"
    path.write_text(json.dumps(tracks), encoding="utf-8")
    return path


def track(**overrides):
    base = {"deezer_id": 1, "title": "Song", "artist": "Artist", "genre": "Pop",
            "cover": "https://example.com/c.jpg", "preview_path": "data/previews/1.mp3", "liked": True}
    return {**base, **overrides}


def test_load_library_adds_id_from_deezer_id(tmp_path):
    path = write_library(tmp_path, [track(deezer_id=7), track(deezer_id=8, liked=False, genre=None)])
    tracks = load_library(path)
    assert [t["id"] for t in tracks] == [7, 8]
    assert [t["liked"] for t in tracks] == [True, False]


def test_load_library_rejects_missing_fields(tmp_path):
    bad = track()
    del bad["preview_path"]
    with pytest.raises(ValueError, match="preview_path"):
        load_library(write_library(tmp_path, [bad]))
