"""
Sonata's web server: the galaxy, song skies, recommendations and Ask Sonata.

  python -m sonata.server [--port 8000]      then open http://127.0.0.1:8000

Endpoints
  GET /api/library            every song with its labels, tempo, galaxy position and preview URL
  GET /api/stars/{deezer_id}  the song's star map: [{t, pitch, size}, ...]
  GET /api/recommendations    content-based picks (sonata.recommender), ?k=5
  POST /api/ask {"query"}     RAG answer (sonata.rag.ask): songs, reasons, retrieval similarity
  GET /previews/{file}        a 30-second MP3 preview, e.g. /previews/3135556.mp3
  GET /                       the web UI in web/ (plain HTML, CSS and JS, no build step)

Runs from the committed data. The MP3 previews are optional: a song without one
gets preview_url null, and the page lets you scrub through its stars by hand.
The two recommendation endpoints only reshape what the sonata functions return
for JSON; all ranking, validation, caching and quota logic stays in those modules.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from sonata import rag
from sonata.catalog import load_library
from sonata.features import FEATURES_PATH
from sonata.recommender import load_and_recommend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = PROJECT_ROOT / "data" / "library.json"
STARS_DIR = PROJECT_ROOT / "data" / "stars"
PREVIEWS_DIR = PROJECT_ROOT / "data" / "previews"
WEB_DIR = PROJECT_ROOT / "web"

# Only "<digits>.mp3": no other characters, so no path can leave PREVIEWS_DIR
PREVIEW_NAME = re.compile(r"\d+\.mp3")

# Paths are module globals read at request time, so tests can point them elsewhere
app = FastAPI(title="Sonata", summary="Your library as a galaxy: star maps, previews and recommendations")


def _tempos() -> Dict[int, int]:
    """Deezer id -> tempo in BPM from the audio features (empty if they haven't been built)."""
    if not FEATURES_PATH.exists():
        return {}
    features = pd.read_parquet(FEATURES_PATH, columns=["deezer_id", "tempo"])
    return {int(i): int(round(t)) for i, t in zip(features["deezer_id"], features["tempo"])}


@app.get("/api/library")
def library() -> List[Dict]:
    tempos = _tempos()
    songs = []
    for track in load_library(LIBRARY_PATH):
        preview = Path(track["preview_path"]).name
        has_preview = bool(PREVIEW_NAME.fullmatch(preview)) and (PREVIEWS_DIR / preview).is_file()
        songs.append({
            "deezer_id": track["deezer_id"],
            "title": track["title"],
            "artist": track["artist"],
            "genre": track["genre"],
            "cover": track["cover"],
            "liked": track["liked"],
            "tempo": tempos.get(track["deezer_id"]),
            "tempo_label": track.get("tempo_label"),
            "energy_label": track.get("energy_label"),
            "brightness_label": track.get("brightness_label"),
            "preview_url": f"/previews/{preview}" if has_preview else None,
            # Galaxy position from sonata.compute_galaxy, in [-1, 1]; null if not computed
            "x": track.get("x"),
            "y": track.get("y"),
        })
    return songs


def _song_ref(track: Dict) -> Dict:
    """The few song fields a recommendation carries; the page joins the rest from /api/library."""
    return {"deezer_id": track["deezer_id"], "title": track["title"], "artist": track["artist"]}


@app.get("/api/recommendations")
def recommendations(k: int = Query(5, ge=1, le=50)) -> List[Dict]:
    """Content-based top k: each unheard song, its closest liked song and the two features they share most."""
    return [
        {
            **_song_ref(result["song"]),
            "similarity": round(result["similarity"], 3),
            "closest_liked": _song_ref(result["closest_liked"]),
            "shared_features": [shared["description"] for shared in result["shared_features"]],
        }
        for result in load_and_recommend(k, FEATURES_PATH, LIBRARY_PATH)
    ]


class AskRequest(BaseModel):
    query: str


# A plain def: FastAPI runs it in a worker thread, so a slow model call doesn't block other requests
@app.post("/api/ask")
def ask(request: AskRequest) -> Dict:
    """Runs sonata.rag.ask. Empty or too-long queries get 400 with rag's message; API problems
    come back as source "fallback" with retrieval-only picks and a message, never as an error."""
    result = rag.ask(request.query)
    if result["source"] == "rejected":
        raise HTTPException(status_code=400, detail=result["message"])
    return {
        "query": result["query"],
        "source": result["source"],
        "message": result["message"],
        "recommendations": [
            {**_song_ref(pick["song"]), "reason": pick["reason"],
             # how closely the song matched the request in retrieval, not a measure of correctness
             "retrieval_similarity": pick["retrieval_similarity"]}
            for pick in result["recommendations"]
        ],
        "retrieved_ids": result["retrieved_ids"],
        "strong_matches": result["strong_matches"],
        "latency_ms": result["latency_ms"],
    }


@app.get("/api/stars/{deezer_id}")
def stars(deezer_id: int) -> FileResponse:
    path = STARS_DIR / f"{deezer_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No star map for song {deezer_id}")
    return FileResponse(path, media_type="application/json")


@app.get("/previews/{file}")
def preview(file: str) -> FileResponse:
    if not PREVIEW_NAME.fullmatch(file):
        raise HTTPException(status_code=404, detail="Previews are named <deezer_id>.mp3")
    path = PREVIEWS_DIR / file
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Preview {file} hasn't been downloaded")
    # FileResponse answers Range requests, so the browser can seek inside the MP3
    return FileResponse(path, media_type="audio/mpeg")


# Last, so the API routes above take precedence over static files
class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must re-check on every load. Without a Cache-Control header,
    browsers may keep using a stale cached copy of a changed script (notably ES modules like
    sky.js, imported by app.js), even across reloads. "no-cache" still lets them keep the copy;
    they just ask first, and an unchanged file costs only a 304."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", RevalidatedStaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve Sonata's song sky")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Sonata's sky: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
