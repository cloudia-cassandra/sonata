"""Mood labeling: parsing, the resumable terminal loop, the CSV, and estimated moods."""

import csv

import numpy as np
import pandas as pd
import pytest

import sonata.label as label
import sonata.rag as rag
from sonata.analyze import FEATURE_COLUMNS


def track(i, liked=False, title=None):
    return {"deezer_id": 100 + i, "title": title or f"Song {i}", "artist": f"Artist {i}", "genre": "Pop",
            "cover": None, "preview_path": "", "liked": liked}


def scripted(*answers):
    """A stand-in for input() that gives these answers in turn, then acts like Ctrl+D."""
    queue = list(answers)

    def read(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)
    return read


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# ---------- parsing ----------

@pytest.mark.parametrize("text, moods", [
    ("chill focus", ["chill", "focus"]),
    ("2, 1", ["chill", "focus"]),                 # numbers, and always in MOODS order
    ("c f late", ["chill", "focus", "late-night"]),
    ("10 late-night LATE", ["late-night"]),       # duplicates collapse, case doesn't matter
    ("sad;dark", ["sad", "dark"]),
])
def test_parse_moods_accepts_names_numbers_and_prefixes(text, moods):
    assert label.parse_moods(text) == moods


@pytest.mark.parametrize("text", ["calm", "11", "0"])
def test_parse_moods_rejects_anything_else(text):
    with pytest.raises(ValueError):
        label.parse_moods(text)


# ---------- the labeling loop ----------

def test_liked_songs_come_first():
    library = [track(1), track(2, liked=True), track(3), track(4, liked=True)]
    assert [s["deezer_id"] for s in label.labeling_order(library)] == [102, 104, 101, 103]


def test_label_skip_and_quit_then_resume(tmp_path):
    library = [track(1), track(2, liked=True, title='Comma, "quoted"; title'), track(3), track(4)]
    labels_path, skips_path = tmp_path / "mood_labels.csv", tmp_path / "skips.json"
    out = []

    # Liked song 102 first: a typo is asked again, then labeled; 101 skipped; q on 103
    labeled = label.run(library, labels_path, skips_path, read=scripted("calm", "c 2", "", "q"), write=out.append)
    assert labeled == 1
    assert any("isn't one of the moods" in line for line in out)
    assert csv_rows(labels_path) == [{"deezer_id": "102", "title": 'Comma, "quoted"; title',
                                      "artist": "Artist 2", "moods": "chill;focus"}]
    assert label.load_labels(labels_path) == {102: ["chill", "focus"]}

    # Resumes at 103: labeled and skipped songs aren't shown again
    shown = []
    label.run(library, labels_path, skips_path, read=scripted("hype", "party sad"), write=shown.append)
    titles = [line for line in shown if line.startswith("[")]
    assert [t.split("] ")[1].split(" —")[0] for t in titles] == ["Song 3", "Song 4"]
    assert label.load_labels(labels_path) == {102: ["chill", "focus"], 103: ["hype"], 104: ["party", "sad"]}

    # The skipped song can be revisited, and labeling it clears the skip
    label.run(library, labels_path, skips_path, revisit_skipped=True, read=scripted("u"), write=out.append)
    assert label.load_labels(labels_path)[101] == ["uplifting"]
    assert label.run(library, labels_path, skips_path, revisit_skipped=True, read=scripted(), write=out.append) == 0


def test_ctrl_d_or_ctrl_c_quits_without_losing_labels(tmp_path):
    labels_path = tmp_path / "mood_labels.csv"
    label.run([track(1), track(2)], labels_path, tmp_path / "s.json", read=scripted("w"), write=lambda _: None)
    assert label.load_labels(labels_path) == {101: ["workout"]}


def test_unknown_moods_in_the_csv_are_an_error(tmp_path):
    path = tmp_path / "mood_labels.csv"
    path.write_text("deezer_id,title,artist,moods\n1,A,B,chill;calm\n", encoding="utf-8")
    with pytest.raises(ValueError, match="calm"):
        label.load_labels(path)


# ---------- estimated moods ----------

def features_for(vectors):
    """Feature rows whose standardized values point the way the given 2D vectors do."""
    rows = []
    for deezer_id, (a, b) in vectors.items():
        values = np.zeros(len(FEATURE_COLUMNS))
        values[0], values[1] = a, b
        rows.append({"deezer_id": deezer_id, **dict(zip(FEATURE_COLUMNS, values))})
    return pd.DataFrame(rows)


def test_a_mood_is_estimated_when_3_of_the_5_nearest_labeled_songs_agree():
    # Labeled songs 1-5 sit near (1, 0); 6-8 are far away near (-1, 0)
    vectors = {1: (1, 0.1), 2: (1, 0.2), 3: (1, -0.1), 4: (1, -0.2), 5: (1, 0.3),
               6: (-1, 0.1), 7: (-1, 0.2), 8: (-1, -0.1),
               10: (1, 0), 11: (-1, 0)}
    labels = {1: ["chill", "focus"], 2: ["chill"], 3: ["chill", "sad"], 4: ["focus", "sad"], 5: ["hype"],
              6: ["party"], 7: ["party"], 8: ["party"]}
    inferred = label.infer_moods(features_for(vectors), labels)
    assert inferred[10] == ["chill"]          # chill 3/5 kept; focus and sad only 2/5
    assert inferred[11] == ["party"]          # its 5 nearest: the 3 party songs + 2 others
    assert set(inferred) == {10, 11}          # labeled songs aren't estimated


def test_no_estimates_from_fewer_labeled_songs_than_must_agree():
    vectors = {1: (1, 0), 2: (1, 0.1), 10: (1, 0.05)}
    assert label.infer_moods(features_for(vectors), {1: ["chill"], 2: ["chill"]}) == {}


def test_descriptions_carry_hand_labeled_or_estimated_moods():
    library = [track(1), track(2), track(3), track(4), track(10)]
    vectors = {101: (1, 0), 102: (1, 0.1), 103: (1, -0.1), 104: (1, 0.2), 110: (1, 0.05)}
    labels = {101: ["chill"], 102: ["chill", "focus"], 103: ["chill"]}
    songs = {s["deezer_id"]: s for s in label.with_moods(library, features_for(vectors), labels)}

    assert rag.describe(songs[102]) == "Song 2 by Artist 2. Genre: Pop. Moods: chill, focus."
    assert rag.describe(songs[110]) == "Song 10 by Artist 10. Genre: Pop. Moods (estimated): chill."
    assert rag.describe(songs[104]) == "Song 4 by Artist 4. Genre: Pop. Moods (estimated): chill."
    assert "moods" not in library[0]          # the caller's songs aren't modified


def test_ask_embeds_the_moods(tmp_path):
    from test_rag import FakeEmbedder

    library = [track(i, liked=i < 3) for i in range(1, 7)]
    vectors = {100 + i: (1, i / 10) for i in range(1, 7)}
    embedder = FakeEmbedder()
    rag.ask("chill music", library=library, features=features_for(vectors), embedder=embedder,
            mood_labels={101: ["chill"], 102: ["chill"], 103: ["chill"]})
    assert "Song 1 by Artist 1. Genre: Pop. Moods: chill." in embedder.embedded
    assert "Song 6 by Artist 6. Genre: Pop. Moods (estimated): chill." in embedder.embedded


def test_without_labels_descriptions_are_unchanged():
    library = [track(1), track(2)]
    assert label.with_moods(library, features_for({101: (1, 0), 102: (0, 1)}), {}) == library
