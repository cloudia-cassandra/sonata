"""
Hand-labeled moods for Sonata's songs, and moods estimated from them.

  python -m sonata.label                    label songs one at a time (liked songs first)
  python -m sonata.label --revisit-skipped  go back through songs you skipped
  python -m sonata.label --rebuild          only refresh the embeddings cache

Labeling: each song shows its title and artist. Type moods from MOODS by name,
number or first letter, separated by spaces or commas ("chill focus", "1 2", "c f").
Enter skips, q saves and quits. Every answer is saved straight away, so it also
survives Ctrl+C, and the next run resumes at the first song not yet labeled or skipped.

  data/mood_labels.csv        deezer_id, title, artist, moods ("chill;focus"): the
                              project's own dataset, committed with the code
  data/.cache/mood_label_skips.json   songs skipped so far (local, not committed)

How sonata.rag uses them (with_moods), in order of precedence:
  1. hand labels (data/mood_labels.csv): "Moods: chill, focus." Hand labels always win.
  2. agent labels (data/auto_labels.json, from sonata.label_agent): "Moods (estimated): ..."
  3. otherwise its INFER_NEIGHBORS nearest labeled songs by cosine similarity on
     standardized audio features; each mood at least INFER_MIN_AGREE of them share
     is kept: "Moods (estimated): ..."
After labeling, the descriptions' embeddings are rebuilt, so the next ask() uses them.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from sonata.catalog import LIBRARY_PATH, load_library
from sonata.features import FEATURES_PATH
from sonata.recommender import standardize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "data" / "mood_labels.csv"
AUTO_LABELS_PATH = PROJECT_ROOT / "data" / "auto_labels.json"
SKIPS_PATH = PROJECT_ROOT / "data" / ".cache" / "mood_label_skips.json"

MOODS = ("chill", "focus", "hype", "party", "sad", "romantic", "dark", "uplifting", "workout", "late-night")
INFER_NEIGHBORS = 5
INFER_MIN_AGREE = 3
CSV_FIELDS = ("deezer_id", "title", "artist", "moods")


# ---------- the dataset ----------

def load_labels(path: Optional[Path] = None) -> Dict[int, List[str]]:
    """deezer_id -> its moods, in MOODS order. A missing file means no labels yet."""
    path = Path(path or LABELS_PATH)
    if not path.exists():
        return {}
    labels = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            moods = [mood for mood in row["moods"].split(";") if mood]
            unknown = [mood for mood in moods if mood not in MOODS]
            if unknown:
                raise ValueError(f"{path}: song {row['deezer_id']} has unknown moods {unknown}")
            labels[int(row["deezer_id"])] = sorted(moods, key=MOODS.index)
    return labels


def save_labels(labels: Dict[int, List[str]], songs: Dict[int, Dict], path: Optional[Path] = None) -> None:
    """Writes the CSV in labeling order, atomically (a crash can't leave half a file)."""
    path = Path(path or LABELS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_FIELDS)
        for deezer_id, moods in labels.items():
            song = songs.get(deezer_id, {})
            writer.writerow([deezer_id, song.get("title", ""), song.get("artist", ""), ";".join(moods)])
    os.replace(temporary, path)


def load_auto_labels(path: Optional[Path] = None) -> Dict[int, List[str]]:
    """deezer_id -> moods from sonata.label_agent (data/auto_labels.json); empty if not run yet."""
    path = Path(path or AUTO_LABELS_PATH)
    if not path.exists():
        return {}
    songs = json.loads(path.read_text(encoding="utf-8"))["songs"]
    return {int(i): [m for m in entry["moods"] if m in MOODS] for i, entry in songs.items() if entry.get("moods")}


def _load_skips(path: Path) -> List[int]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _save_skips(skips: Sequence[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(skips)), encoding="utf-8")


# ---------- estimated moods ----------

def infer_moods(features: pd.DataFrame, labels: Dict[int, List[str]],
                neighbors: int = INFER_NEIGHBORS, min_agree: int = INFER_MIN_AGREE) -> Dict[int, List[str]]:
    """Moods for every unlabeled song with audio features: the moods that at least min_agree of
    its `neighbors` most similar labeled songs (cosine on standardized features) share.
    Songs with no such mood are left out."""
    features = features.reset_index(drop=True)
    ids = features["deezer_id"].astype(int).to_numpy()
    labeled_rows = np.flatnonzero(np.isin(ids, [i for i, moods in labels.items() if moods]))
    if len(labeled_rows) < min_agree:
        return {}

    z = standardize(features).to_numpy()
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    unit = z / np.where(norms == 0, 1, norms)
    similarity = unit @ unit[labeled_rows].T

    inferred = {}
    for row, deezer_id in enumerate(ids):
        if int(deezer_id) in labels:
            continue
        nearest = labeled_rows[np.argsort(-similarity[row], kind="stable")[:neighbors]]
        votes = [mood for neighbor in nearest for mood in labels[int(ids[neighbor])]]
        agreed = [mood for mood in MOODS if votes.count(mood) >= min_agree]
        if agreed:
            inferred[int(deezer_id)] = agreed
    return inferred


def with_moods(library: List[Dict], features: Optional[pd.DataFrame],
               labels: Optional[Dict[int, List[str]]] = None,
               auto_labels: Optional[Dict[int, List[str]]] = None) -> List[Dict]:
    """Copies of the songs with "moods" (hand-labeled) or "moods_estimated" added. Hand labels
    win, then agent labels, then the nearest-neighbor estimate. labels and auto_labels default to
    data/mood_labels.csv and data/auto_labels.json; without features there is no estimate."""
    labels = load_labels() if labels is None else labels
    auto_labels = load_auto_labels() if auto_labels is None else auto_labels
    inferred = infer_moods(features, labels) if features is not None and labels else {}
    songs = []
    for song in library:
        song = dict(song)
        song_id = song["deezer_id"]
        if labels.get(song_id):
            song["moods"] = labels[song_id]
        elif auto_labels.get(song_id):
            song["moods_estimated"] = auto_labels[song_id]
        elif song_id in inferred:
            song["moods_estimated"] = inferred[song_id]
        songs.append(song)
    return songs


def rebuild_embeddings(library_path: Path = LIBRARY_PATH, features_path: Path = FEATURES_PATH) -> int:
    """Re-embeds the descriptions (with moods) into the cache rag uses. Returns how many songs."""
    from sonata import rag    # rag imports this module, so import it only when needed

    features = pd.read_parquet(features_path) if Path(features_path).exists() else None
    songs = with_moods(load_library(library_path), features)
    rag.embed_descriptions([rag.describe(song) for song in songs], rag.minilm_embedder)
    return len(songs)


# ---------- labeling in the terminal ----------

def parse_moods(text: str) -> List[str]:
    """Moods from names, numbers (1-10) or unique prefixes ("c", "late"). Raises ValueError."""
    moods = []
    for token in re.split(r"[\s,;]+", text.strip().lower()):
        if not token:
            continue
        if token.isdigit() and 1 <= int(token) <= len(MOODS):
            mood = MOODS[int(token) - 1]
        else:
            matches = [m for m in MOODS if m == token] or [m for m in MOODS if m.startswith(token)]
            if len(matches) != 1:
                raise ValueError(f"'{token}' isn't one of the moods" if not matches
                                 else f"'{token}' could be {' or '.join(matches)}")
            mood = matches[0]
        if mood not in moods:
            moods.append(mood)
    return sorted(moods, key=MOODS.index)


def labeling_order(library: List[Dict]) -> List[Dict]:
    """Liked songs first, then the rest, each in library order."""
    return [s for s in library if s["liked"]] + [s for s in library if not s["liked"]]


MENU = "  ".join(f"{n} {mood}" for n, mood in enumerate(MOODS, start=1))


def run(library: List[Dict], labels_path: Optional[Path] = None, skips_path: Optional[Path] = None,
        revisit_skipped: bool = False, read: Callable[[str], str] = input,
        write: Callable[[str], None] = print) -> int:
    """The labeling loop. Returns how many songs were labeled in this session."""
    labels_path = Path(labels_path or LABELS_PATH)
    skips_path = Path(skips_path or SKIPS_PATH)
    labels = load_labels(labels_path)
    skips = _load_skips(skips_path)
    songs = {song["deezer_id"]: song for song in library}
    order = labeling_order(library)
    if revisit_skipped:
        queue = [s for s in order if s["deezer_id"] in skips and s["deezer_id"] not in labels]
    else:
        queue = [s for s in order if s["deezer_id"] not in labels and s["deezer_id"] not in skips]

    write(f"Mood labels: {len(labels)} of {len(library)} songs labeled, {len(skips)} skipped, {len(queue)} to go.")
    write("Type moods by name, number or first letter (\"chill focus\", \"1 2\", \"c f\"). "
          "Enter skips, q saves and quits.\n")
    labeled_now = 0
    for song in queue:
        position = len(labels) + 1
        write(f"[{position}/{len(library)}] {'♥ ' if song['liked'] else ''}{song['title']} — {song['artist']}")
        write(f"  {MENU}")
        while True:
            try:
                answer = read("  moods> ")
            except (EOFError, KeyboardInterrupt):
                answer = "q"
            if answer.strip().lower() in ("q", "quit"):
                write(f"\nSaved. {labeled_now} labeled this session, {len(labels)} in total.")
                return labeled_now
            if not answer.strip():
                if song["deezer_id"] not in skips:
                    skips.append(song["deezer_id"])
                    _save_skips(skips, skips_path)
                write("  skipped\n")
                break
            try:
                moods = parse_moods(answer)
            except ValueError as error:
                write(f"  {error}. Moods: {', '.join(MOODS)}")
                continue
            labels[song["deezer_id"]] = moods
            if song["deezer_id"] in skips:
                skips.remove(song["deezer_id"])
                _save_skips(skips, skips_path)
            save_labels(labels, songs, labels_path)
            labeled_now += 1
            write(f"  ✓ {', '.join(moods)}\n")
            break
    write(f"Done: no songs left{' among the skipped ones' if revisit_skipped else ''}. "
          f"{labeled_now} labeled this session, {len(labels)} in total.")
    if skips and not revisit_skipped:
        write("Run with --revisit-skipped to go back through the skipped songs.")
    return labeled_now


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sonata.label", description="Label your songs' moods")
    parser.add_argument("--revisit-skipped", action="store_true", help="go through the songs you skipped")
    parser.add_argument("--rebuild", action="store_true", help="only refresh the embeddings cache")
    parser.add_argument("--no-embed", action="store_true", help="don't refresh the embeddings cache afterwards")
    args = parser.parse_args(argv)

    labeled = 0
    if not args.rebuild:
        labeled = run(load_library(LIBRARY_PATH), revisit_skipped=args.revisit_skipped)
    if args.rebuild or (labeled and not args.no_embed):
        print("Updating the description embeddings (moods included)…", flush=True)
        try:
            print(f"Embeddings cache up to date for {rebuild_embeddings()} songs.")
        except Exception as error:   # labels are already saved; the next ask() re-embeds anyway
            print(f"Couldn't update the embeddings ({error}); the next ask will do it.", file=sys.stderr)


if __name__ == "__main__":
    main()
