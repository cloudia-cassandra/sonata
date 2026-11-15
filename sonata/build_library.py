"""
Builds data/library.json from your Spotify taste using the public Deezer API.

1. Liked tracks: each track in data/my_taste.json is searched on Deezer
   (artist:"X" track:"Y"), the best match is kept, its 30-second preview is
   downloaded to data/previews/ and its genre is read from the album.
2. Candidate pool: for your top 15 artists, 3 related artists each, and each
   related artist's top 5 tracks. Anything already in your taste list is dropped.

Tracks without a preview are skipped and logged. Previews already on disk are
not downloaded again, so the script can be re-run after an interruption.

Run from the project root with: python -m sonata.build_library
"""

import json
import logging
import re
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASTE_PATH = PROJECT_ROOT / "data" / "my_taste.json"
LIBRARY_PATH = PROJECT_ROOT / "data" / "library.json"
PREVIEW_DIR = PROJECT_ROOT / "data" / "previews"

API = "https://api.deezer.com"
REQUEST_DELAY = 0.2         # seconds between requests
TIMEOUT = 15
RETRIES = 3
QUOTA_ERROR_CODE = 4        # Deezer's "too many requests" error
MIN_ARTIST_SIMILARITY = 0.85  # normalized artist names must be at least this close
MIN_TITLE_SIMILARITY = 0.75   # below this, a same-artist result is treated as no match

TOP_ARTISTS = 15
RELATED_PER_ARTIST = 3
TOP_TRACKS_PER_ARTIST = 5

log = logging.getLogger("build_library")


class Deezer:
    """Small Deezer client: rate-limited, retries transient errors, never raises on network failure."""

    def __init__(self):
        self.session = requests.Session()
        self.album_genres: Dict[int, Optional[str]] = {}

    def _wait(self):
        time.sleep(REQUEST_DELAY)

    def get(self, path: str, **params) -> Optional[dict]:
        """GETs an API path and returns its JSON, or None after logging the failure."""
        for attempt in range(1, RETRIES + 1):
            self._wait()
            try:
                response = self.session.get(f"{API}{path}", params=params, timeout=TIMEOUT)
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as error:
                log.warning("GET %s failed (attempt %d/%d): %s", path, attempt, RETRIES, error)
                time.sleep(attempt)
                continue
            # Deezer reports API errors with HTTP 200 and an "error" object
            error = data.get("error") if isinstance(data, dict) else None
            if error:
                if error.get("code") == QUOTA_ERROR_CODE and attempt < RETRIES:
                    log.info("Deezer quota hit, backing off")
                    time.sleep(5 * attempt)
                    continue
                log.warning("GET %s returned an error: %s", path, error)
                return None
            return data
        return None

    def download(self, url: str, dest: Path) -> bool:
        """Downloads url to dest (via a temp file so partial downloads never look finished)."""
        if dest.exists() and dest.stat().st_size > 0:
            return True
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(1, RETRIES + 1):
            self._wait()
            try:
                with self.session.get(url, timeout=TIMEOUT, stream=True) as response:
                    response.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in response.iter_content(64 * 1024):
                            f.write(chunk)
                tmp.replace(dest)
                return True
            except (requests.RequestException, OSError) as error:
                log.warning("Download of %s failed (attempt %d/%d): %s", dest.name, attempt, RETRIES, error)
                tmp.unlink(missing_ok=True)
                time.sleep(attempt)
        return False

    def genre(self, album_id: int) -> Optional[str]:
        """Returns the album's first genre name, cached per album."""
        if album_id not in self.album_genres:
            album = self.get(f"/album/{album_id}") or {}
            genres = (album.get("genres") or {}).get("data") or []
            self.album_genres[album_id] = genres[0]["name"] if genres else None
        return self.album_genres[album_id]


VERSION_WORDS = r"explicit|clean|remaster(ed)?|version|edit|live|mono|stereo"


def normalize(text: str) -> str:
    """Lowercases, strips accents, and drops feat./explicit/remaster/version tags for comparison.

    "Headlines (Explicit Version)", "Headlines - 2011 Remaster" and "Headlines" all become
    "headlines"; remixes keep the word "remix" so they stay distinct from the original.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\s*[\(\[](feat|ft|with|prod)\b\.?[^\)\]]*[\)\]]", "", text)
    text = re.sub(rf"\s*[\(\[][^\)\]]*\b({VERSION_WORDS})\b[^\)\]]*[\)\]]", "", text)
    text = re.sub(rf"\s+-\s+[^-]*\b({VERSION_WORDS})\b.*$", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def same_artist(a: str, b: str) -> bool:
    """True for the same artist despite accents/punctuation ("Fred again.." vs "Fred again")."""
    a, b = normalize(a), normalize(b)
    return bool(a and b) and (a == b or SequenceMatcher(None, a, b).ratio() >= MIN_ARTIST_SIMILARITY)


def best_match(results: List[dict], artist: str, title: str, also_by: Sequence[str] = ()) -> Optional[dict]:
    """Picks the result by the same artist whose title is closest to the Spotify track.

    The artist must match (the main artist or, for collaborations, any of also_by):
    covers, karaoke and lullaby versions often have the exact title.
    """
    artists = [artist, *also_by]
    scored = [(similarity(result["title"], title), result) for result in results
              if result.get("readable", True) and any(same_artist(result["artist"]["name"], a) for a in artists)]
    if not scored:
        return None
    top_score, top = max(scored, key=lambda pair: pair[0])
    return top if top_score >= MIN_TITLE_SIMILARITY else None


def search_track(dz: Deezer, artist: str, title: str, also_by: Sequence[str] = ()) -> Optional[dict]:
    """Strict artist/track search first, then a looser free-text search as a fallback."""
    clean_title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip() or title
    for query in (f'artist:"{artist}" track:"{clean_title}"', f"{artist} {clean_title}"):
        data = dz.get("/search", q=query)
        match = best_match((data or {}).get("data") or [], artist, title, also_by)
        if match:
            return match
    return None


def library_entry(dz: Deezer, track: dict, liked: bool) -> Optional[dict]:
    """Downloads the preview and builds the library record, or returns None (and logs why)."""
    label = f"{track['artist']['name']} - {track['title']}"
    if not track.get("preview"):
        log.info("Skipped (no preview): %s", label)
        return None
    dest = PREVIEW_DIR / f"{track['id']}.mp3"
    if not dz.download(track["preview"], dest):
        log.info("Skipped (preview download failed): %s", label)
        return None
    album = track.get("album") or {}
    return {
        "deezer_id": track["id"],
        "title": track["title"],
        "artist": track["artist"]["name"],
        "genre": dz.genre(album["id"]) if album.get("id") else None,
        "cover": album.get("cover_xl") or album.get("cover_big") or album.get("cover"),
        "preview_path": str(dest.relative_to(PROJECT_ROOT)),
        "liked": liked,
    }


def top_artist_names(taste: List[dict], n: int) -> List[str]:
    """Ranks primary artists by how many of your tracks they lead, ties broken by first appearance."""
    counts = Counter(track["artists"][0] for track in taste if track["artists"])
    first_seen = {}
    for index, track in enumerate(taste):
        if track["artists"]:
            first_seen.setdefault(track["artists"][0], index)
    return sorted(counts, key=lambda name: (-counts[name], first_seen[name]))[:n]


def build_liked(dz: Deezer, taste: List[dict]) -> List[dict]:
    """Matches each taste track on Deezer; also remembers each artist's Deezer id."""
    liked, seen = [], set()
    for number, item in enumerate(taste, start=1):
        artist, title = item["artists"][0], item["title"]
        match = search_track(dz, artist, title, item["artists"][1:])
        if not match:
            log.info("[%d/%d] No Deezer match: %s - %s", number, len(taste), artist, title)
            continue
        if match["id"] in seen:
            log.info("[%d/%d] Same Deezer track as an earlier one, skipped: %s - %s",
                     number, len(taste), artist, title)
            continue
        entry = library_entry(dz, match, liked=True)
        if entry:
            entry["_artist_id"] = match["artist"]["id"]
            entry["_spotify_artist"] = artist
            liked.append(entry)
            seen.add(match["id"])
            log.info("[%d/%d] Liked: %s - %s (%s)", number, len(taste), entry["artist"], entry["title"], entry["genre"])
    return liked


def build_candidates(dz: Deezer, liked: List[dict], taste: List[dict]) -> List[dict]:
    """Top tracks of artists related to your top artists, minus anything already in your taste."""
    artist_ids = {}
    for entry in liked:
        artist_ids.setdefault(entry["_spotify_artist"], entry["_artist_id"])

    taste_keys = {(normalize(t["artists"][0]), normalize(t["title"])) for t in taste if t["artists"]}
    seen_ids = {entry["deezer_id"] for entry in liked}
    source_ids = set(artist_ids.values())
    used_related = set()
    candidates = []

    for name in top_artist_names(taste, TOP_ARTISTS):
        artist_id = artist_ids.get(name)
        if artist_id is None:
            log.info("No Deezer id for top artist %s, skipping its related artists", name)
            continue
        related = ((dz.get(f"/artist/{artist_id}/related") or {}).get("data")) or []
        picks = [a for a in related if a["id"] not in used_related and a["id"] not in source_ids][:RELATED_PER_ARTIST]
        log.info("Related to %s: %s", name, ", ".join(a["name"] for a in picks) or "(none)")

        for related_artist in picks:
            used_related.add(related_artist["id"])
            top = (dz.get(f"/artist/{related_artist['id']}/top", limit=TOP_TRACKS_PER_ARTIST) or {}).get("data") or []
            for track in top[:TOP_TRACKS_PER_ARTIST]:
                key = (normalize(track["artist"]["name"]), normalize(track["title"]))
                if track["id"] in seen_ids or key in taste_keys:
                    log.info("Dropped (already in your taste): %s - %s", track["artist"]["name"], track["title"])
                    continue
                entry = library_entry(dz, track, liked=False)
                if entry:
                    candidates.append(entry)
                    seen_ids.add(track["id"])
    return candidates


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    taste = json.loads(TASTE_PATH.read_text(encoding="utf-8"))
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    dz = Deezer()

    liked = build_liked(dz, taste)
    candidates = build_candidates(dz, liked, taste)
    library = [{k: v for k, v in entry.items() if not k.startswith("_")} for entry in liked + candidates]

    LIBRARY_PATH.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Previews from earlier runs that are no longer in the library (e.g. replaced matches)
    in_use = {PROJECT_ROOT / entry["preview_path"] for entry in library}
    stale = [path for path in PREVIEW_DIR.glob("*.mp3") if path not in in_use]
    for path in stale:
        path.unlink()
    if stale:
        log.info("Removed %d previews no longer in the library", len(stale))

    log.info("Saved %d tracks (%d liked, %d candidates) to %s",
             len(library), len(liked), len(candidates), LIBRARY_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
