"""The web server runs from the committed data; previews are optional and can't be used to read other files."""

import json

import pytest
from fastapi.testclient import TestClient

import sonata.rag as rag
import sonata.server as server

SONG_ID = 111
STARS = [{"t": 0.5, "pitch": 60, "size": 0.8}, {"t": 1.25, "pitch": 64, "size": 0.3}]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A tiny library of two songs; only the first has a downloaded preview."""
    library = [
        {"deezer_id": SONG_ID, "title": "Lit", "artist": "A", "genre": "Pop", "cover": "",
         "preview_path": f"data/previews/{SONG_ID}.mp3", "liked": True, "tempo_label": "fast",
         "x": 0.25, "y": -0.5},
        {"deezer_id": 222, "title": "Silent", "artist": "B", "genre": "Rap", "cover": "",
         "preview_path": "data/previews/222.mp3", "liked": False},
    ]
    (tmp_path / "library.json").write_text(json.dumps(library), encoding="utf-8")
    (tmp_path / "stars").mkdir()
    (tmp_path / "stars" / f"{SONG_ID}.json").write_text(json.dumps(STARS), encoding="utf-8")
    (tmp_path / "previews").mkdir()
    (tmp_path / "previews" / f"{SONG_ID}.mp3").write_bytes(b"ID3" + bytes(range(256)) * 4)
    (tmp_path / "secret.txt").write_text("not for the browser", encoding="utf-8")

    monkeypatch.setattr(server, "LIBRARY_PATH", tmp_path / "library.json")
    monkeypatch.setattr(server, "STARS_DIR", tmp_path / "stars")
    monkeypatch.setattr(server, "PREVIEWS_DIR", tmp_path / "previews")
    monkeypatch.setattr(server, "FEATURES_PATH", tmp_path / "missing.parquet")
    return TestClient(server.app)


def test_library_marks_which_songs_have_a_preview(client):
    songs = {song["deezer_id"]: song for song in client.get("/api/library").json()}
    assert songs[SONG_ID]["preview_url"] == f"/previews/{SONG_ID}.mp3"
    assert songs[222]["preview_url"] is None
    assert songs[SONG_ID]["tempo_label"] == "fast" and songs[222]["tempo_label"] is None
    assert "preview_path" not in songs[SONG_ID]          # local paths stay on the server
    assert songs[SONG_ID]["x"] == 0.25 and songs[SONG_ID]["y"] == -0.5
    assert songs[222]["x"] is None                       # not placed in the galaxy


def test_stars_returns_the_star_map(client):
    assert client.get(f"/api/stars/{SONG_ID}").json() == STARS


def test_unknown_or_malformed_star_ids_are_404_or_422(client):
    assert client.get("/api/stars/999").status_code == 404
    assert client.get("/api/stars/abc").status_code == 422


def test_preview_supports_range_requests_for_seeking(client):
    full = client.get(f"/previews/{SONG_ID}.mp3")
    assert full.status_code == 200 and full.headers["content-type"] == "audio/mpeg"
    part = client.get(f"/previews/{SONG_ID}.mp3", headers={"Range": "bytes=10-19"})
    assert part.status_code == 206 and part.content == full.content[10:20]


@pytest.mark.parametrize("name", ["222.mp3", "secret.txt", "..%2Fsecret.txt", "111.mp3.bak"])
def test_preview_only_serves_downloaded_mp3s(client, name):
    assert client.get(f"/previews/{name}").status_code == 404


def test_web_ui_is_served_at_root(client):
    page = client.get("/")
    assert page.status_code == 200 and '<canvas id="stage"' in page.text and 'id="galaxy-view"' in page.text
    for asset in ("app.js", "galaxy.js", "sky.js", "util.js", "style.css"):
        assert client.get(f"/{asset}").status_code == 200, asset
    # ES modules must be served as JavaScript or the browser refuses to run them
    assert "javascript" in client.get("/app.js").headers["content-type"]
    # ...and re-checked on every load, so an edited module is never served stale from cache
    assert client.get("/sky.js").headers["cache-control"] == "no-cache"
    assert client.get("/").headers["cache-control"] == "no-cache"


def test_real_library_loads():
    """The committed library and features work through the API, previews or not."""
    songs = TestClient(server.app).get("/api/library").json()
    assert len(songs) > 0
    assert all(isinstance(song["tempo"], int) for song in songs)
    assert all(-1 <= song["x"] <= 1 and -1 <= song["y"] <= 1 for song in songs)   # the committed galaxy


# ---------- recommendations and Ask Sonata: thin wrappers over sonata.recommender and sonata.rag ----------

@pytest.fixture
def real_client(monkeypatch):
    """The committed library, mock LLM (conftest) and an offline fake embedder instead of MiniLM."""
    from test_rag import FakeEmbedder

    monkeypatch.setattr(rag.ask, "__kwdefaults__", {**rag.ask.__kwdefaults__, "embedder": FakeEmbedder()})
    monkeypatch.setattr(rag, "MIN_RELEVANCE", 0.0)
    monkeypatch.setattr(rag, "STRONG_MATCH", 0.0)
    return TestClient(server.app)


def test_recommendations_match_the_recommender(real_client):
    from sonata.recommender import load_and_recommend

    expected = load_and_recommend(5)
    got = real_client.get("/api/recommendations").json()
    assert [r["deezer_id"] for r in got] == [r["song"]["deezer_id"] for r in expected]
    assert [r["closest_liked"]["deezer_id"] for r in got] == [r["closest_liked"]["deezer_id"] for r in expected]
    assert all(len(r["shared_features"]) == 2 and 0 < r["similarity"] <= 1 for r in got)


def test_recommendations_k_is_bounded(real_client):
    assert len(real_client.get("/api/recommendations?k=3").json()) == 3
    assert real_client.get("/api/recommendations?k=0").status_code == 422


def test_ask_returns_grounded_picks_with_reasons_and_retrieval_similarity(real_client):
    body = real_client.post("/api/ask", json={"query": "reggaeton for a party"}).json()
    assert body["source"] == "mock"
    assert 3 <= len(body["recommendations"]) <= 5
    assert all(r["deezer_id"] in body["retrieved_ids"] for r in body["recommendations"])
    assert all(r["reason"] and 0 <= r["retrieval_similarity"] <= 1 for r in body["recommendations"])
    assert body["strong_matches"] == len(body["retrieved_ids"])     # STRONG_MATCH is 0 here


@pytest.mark.parametrize("query", ["   ", "x" * 501])
def test_ask_rejects_bad_queries_with_rags_message(real_client, query):
    response = real_client.post("/api/ask", json={"query": query})
    assert response.status_code == 400
    assert response.json()["detail"] == rag.validate_query(query)


def test_ask_requires_a_query_field(real_client):
    assert real_client.post("/api/ask", json={}).status_code == 422
