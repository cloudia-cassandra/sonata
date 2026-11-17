"""
Review the songs the labeling agent couldn't settle (status needs_review).

  python -m sonata.review

Each song shows its title and artist, the model's moods and reason, and why the check
step flagged it. Enter accepts the model's moods; typing moods (numbers, names or first
letters, the same list as python -m sonata.label) replaces them; s skips; q quits.
Every answer is saved to data/mood_labels.csv as a hand label straight away, and hand
labels always win over agent labels.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from sonata import label
from sonata.catalog import LIBRARY_PATH, load_library


def review_queue(auto: Dict, hand_labels: Dict[int, List[str]]) -> List[int]:
    """Songs still marked needs_review that don't have a hand label yet, in file order."""
    return [int(i) for i, entry in auto["songs"].items()
            if entry.get("status") == "needs_review" and int(i) not in hand_labels]


def run(library: List[Dict], auto_path: Optional[Path] = None, labels_path: Optional[Path] = None,
        read: Callable[[str], str] = input, write: Callable[[str], None] = print) -> int:
    """The review loop. Returns how many songs were saved as hand labels."""
    auto_path = Path(auto_path or label.AUTO_LABELS_PATH)
    labels_path = Path(labels_path or label.LABELS_PATH)
    if not auto_path.exists():
        write("No agent labels yet: run python -m sonata.label_agent first.")
        return 0
    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    hand = label.load_labels(labels_path)
    songs = {song["deezer_id"]: song for song in library}
    queue = [i for i in review_queue(auto, hand) if i in songs]
    write(f"{len(queue)} song(s) to review. Enter accepts the model's moods, moods replace them "
          "(numbers, names or first letters), s skips, q quits.\n")

    saved = 0
    for position, song_id in enumerate(queue, start=1):
        entry = auto["songs"][str(song_id)]
        song = songs[song_id]
        write(f"[{position}/{len(queue)}] {song['title']} — {song['artist']}")
        write(f"  Model: {', '.join(entry['moods'])} (confidence {entry.get('confidence', 0):.2f})")
        write(f"  Reason: {entry.get('reason', '')}")
        for flag in entry.get("remaining_flags") or entry.get("flags") or []:
            write(f"  Flag: {flag}")
        write(f"  {label.MENU}")
        while True:
            try:
                answer = read("  moods> ")
            except (EOFError, KeyboardInterrupt):
                answer = "q"
            command = answer.strip().lower()
            if command in ("q", "quit"):
                write(f"\nSaved {saved} label(s).")
                return saved
            if command == "s":
                write("  skipped\n")
                break
            try:
                moods = label.parse_moods(answer) if command else list(entry["moods"])
            except ValueError as error:
                write(f"  {error}. Moods: {', '.join(label.MOODS)}")
                continue
            if not moods:
                write("  Type at least one mood, or s to skip.")
                continue
            hand[song_id] = moods
            label.save_labels(hand, songs, labels_path)
            saved += 1
            write(f"  ✓ {', '.join(moods)} ({'accepted' if not command else 'replaced'})\n")
            break
    write(f"Done. Saved {saved} label(s).")
    return saved


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sonata.review", description="Review flagged agent labels")
    parser.add_argument("--no-embed", action="store_true", help="don't refresh the embeddings cache afterwards")
    args = parser.parse_args(argv)
    if run(load_library(LIBRARY_PATH)) and not args.no_embed:
        print("Updating the description embeddings…", flush=True)
        try:
            print(f"Embeddings cache up to date for {label.rebuild_embeddings()} songs.")
        except Exception as error:
            print(f"Couldn't update the embeddings ({error}); the next ask will do it.")


if __name__ == "__main__":
    main()
