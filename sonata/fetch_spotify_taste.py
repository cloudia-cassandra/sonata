"""
Fetches your Spotify listening taste and saves it to data/my_taste.json.

Collects your top 50 tracks for each time range (short_term ≈ 4 weeks,
medium_term ≈ 6 months, long_term ≈ 1 year) plus your 50 most recently saved
tracks, de-duplicated by Spotify track id.

Setup: copy .env.example to .env and fill in the keys from your app at
https://developer.spotify.com/dashboard (the redirect URI must also be
registered there). The first run opens a browser to log in; the token is then
cached in .spotify_token_cache and refreshed automatically.

Run from the project root with: python -m sonata.fetch_spotify_taste
"""

import json
import os
import sys
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
TOKEN_CACHE_PATH = PROJECT_ROOT / ".spotify_token_cache"
OUTPUT_PATH = PROJECT_ROOT / "data" / "my_taste.json"

SCOPES = "user-top-read user-library-read"
TIME_RANGES = ("short_term", "medium_term", "long_term")
LIMIT = 50  # Spotify's maximum page size for both endpoints
REQUIRED_ENV = ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REDIRECT_URI")


def make_client() -> spotipy.Spotify:
    """Builds an authenticated client from .env settings, reusing the cached token."""
    load_dotenv(ENV_PATH)
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        sys.exit(f"Missing {', '.join(missing)} in {ENV_PATH} (see .env.example)")

    auth = SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],
        scope=SCOPES,
        cache_handler=CacheFileHandler(cache_path=str(TOKEN_CACHE_PATH)),
    )
    return spotipy.Spotify(auth_manager=auth)


def track_record(track: dict) -> dict:
    """Keeps only the fields Sonata needs from a Spotify track object."""
    images = track["album"].get("images") or []
    return {
        "spotify_id": track["id"],
        "title": track["name"],
        "artists": [artist["name"] for artist in track["artists"]],
        "album": track["album"]["name"],
        # Spotify lists album images largest first
        "cover_url": images[0]["url"] if images else None,
        "time_ranges": [],
        "saved": False,
    }


def fetch_taste(sp: spotipy.Spotify) -> list:
    """Returns top and saved tracks merged by id, in first-seen order."""
    tracks = {}

    def add(track: dict) -> dict:
        # Local files and unavailable tracks come back without an id
        if not track or not track.get("id"):
            return None
        return tracks.setdefault(track["id"], track_record(track))

    for time_range in TIME_RANGES:
        for track in sp.current_user_top_tracks(limit=LIMIT, time_range=time_range)["items"]:
            record = add(track)
            if record is not None:
                record["time_ranges"].append(time_range)

    # Saved tracks come back most recently saved first
    for item in sp.current_user_saved_tracks(limit=LIMIT)["items"]:
        record = add(item["track"])
        if record is not None:
            record["saved"] = True

    return list(tracks.values())


def save_taste(tracks: list, path: Path = OUTPUT_PATH) -> None:
    """Writes the tracks as pretty-printed UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tracks, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    tracks = fetch_taste(make_client())
    save_taste(tracks)
    saved = sum(track["saved"] for track in tracks)
    print(f"Saved {len(tracks)} unique tracks ({saved} from your library) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
