"""The mood-labeling agent: flag rules, batching and retries, the fixed split, hands-win, review."""

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import sonata.label as label
import sonata.label_agent as agent
import sonata.rag as rag
import sonata.review as review
from sonata.analyze import FEATURE_COLUMNS


def song(i, **extra):
    return {"deezer_id": i, "title": f"Song {i}", "artist": f"Artist {i}", "genre": "Pop", "cover": None,
            "preview_path": "", "liked": False, "tempo_label": "medium", "energy_label": "medium",
            "brightness_label": "warm", **extra}


def features_table(directions):
    """Feature rows pointing the given 2D directions (after standardizing), plus tempo and RMS."""
    rows = []
    for song_id, (a, b, tempo, rms) in directions.items():
        values = dict.fromkeys(FEATURE_COLUMNS, 0.0)
        values.update({FEATURE_COLUMNS[3]: a, FEATURE_COLUMNS[4]: b, "tempo": tempo, "rms_mean": rms})
        rows.append({"deezer_id": song_id, **values})
    return pd.DataFrame(rows)


def profile(song_id, tempo=100.0, energy=0.2, sps=3.0):
    return {"deezer_id": song_id, "title": f"Song {song_id}", "artist": "A", "genre": "Pop", "tempo_label": "medium",
            "energy_label": "medium", "brightness_label": "warm", "stars_per_second": sps, "tempo": tempo, "energy": energy}


@pytest.fixture
def reference():
    """Hand labels: songs 1-4 chill (slow, close together in sound), 5-9 party (fast, on the other side).
    Song 10 sounds like the chill songs, song 11 like the party songs."""
    directions = {1: (1, 0.0, 70, 0.1), 2: (1, 0.1, 72, 0.1), 3: (1, -0.1, 74, 0.1), 4: (1, 0.2, 76, 0.1),
                  5: (-1, 0.0, 128, 0.4), 6: (-1, 0.1, 130, 0.4), 7: (-1, -0.1, 132, 0.4), 8: (-1, 0.2, 134, 0.4),
                  9: (-1, -0.2, 136, 0.4), 10: (1, 0.05, 73, 0.1), 11: (-1, 0.05, 131, 0.4)}
    features = features_table(directions)
    profiles = {i: profile(i, tempo=t, energy=r) for i, (_, _, t, r) in directions.items()}
    labels = {1: ["chill"], 2: ["chill", "focus"], 3: ["chill"], 4: ["chill", "sad"],
              5: ["party"], 6: ["party"], 7: ["party"], 8: ["party", "hype"], 9: ["party"]}
    return agent.Reference(labels, profiles, features), profiles


def entry(moods, confidence=0.9):
    return {"moods": moods, "confidence": confidence, "reason": "r"}


def rules(flags):
    return sorted(f["rule"] for f in flags)


# ---------- Step 2: the flag rules ----------

@pytest.mark.parametrize("moods,flagged", [
    (["chill", "hype"], True), (["chill", "party"], True), (["chill", "workout"], True),
    (["sad", "party"], True), (["sad", "uplifting"], True),
    (["chill", "sad"], False), (["hype", "party"], False), (["sad", "dark"], False),
])
def test_rule_a_contradictory_moods(reference, moods, flagged):
    ref, profiles = reference
    flags, _ = agent.check(10, entry(moods), profiles[10], ref)
    assert ("a" in rules(flags)) is flagged


def test_rule_b_low_confidence(reference):
    ref, profiles = reference
    assert "b" in rules(agent.check(10, entry(["chill"], 0.59), profiles[10], ref)[0])
    assert "b" not in rules(agent.check(10, entry(["chill"], 0.6), profiles[10], ref)[0])


def test_rule_c_learns_ranges_per_mood_from_hand_labels(reference):
    ref, profiles = reference
    # chill tempos 70, 72, 74, 76: quartiles 71.5 and 74.5, so 1.5 x IQR allows 67-79
    assert ref.ranges["chill"]["tempo"] == {"n": 4, "q1": 71.5, "q3": 74.5, "low": 67.0, "high": 79.0}
    assert "sad" not in ref.ranges and "focus" not in ref.ranges          # 1 song each: no range
    inside, _ = agent.check(10, entry(["chill"]), profiles[10], ref)
    outside, evidence = agent.check(10, entry(["chill"]), {**profiles[10], "tempo": 120.0}, ref)
    assert "c" not in rules(inside)
    assert rules(outside) == ["c"] and "tempo (BPM) 120 is outside the chill range" in outside[0]["reason"]
    assert evidence["mood_ranges"]["chill"]["tempo"]["high"] == 79.0


def test_rule_d_none_of_the_moods_among_the_nearest_labeled_songs(reference):
    ref, profiles = reference
    near = [n["deezer_id"] for n in ref.neighbors(10)]
    assert near[:4] == [1, 2, 3, 4] or set(near[:4]) == {1, 2, 3, 4}  # the 4 chill songs sound closest
    assert "d" not in rules(agent.check(10, entry(["focus"]), profiles[10], ref)[0])   # song 2 has focus
    flags, evidence = agent.check(10, entry(["workout"]), profiles[10], ref)
    assert "d" in rules(flags) and "workout" in flags[-1]["reason"]
    assert len(evidence["neighbors"]) == 5


def test_every_reason_is_recorded(reference):
    ref, profiles = reference
    # Song 11 sounds like the party songs. "chill, workout" contradicts itself, is unsure, has a
    # tempo and an energy far from my chill songs (one reason each), and its 5 nearest labeled
    # songs are party/hype songs, so neither mood appears among theirs
    flags, _ = agent.check(11, entry(["chill", "workout"], 0.3), profiles[11], ref)
    assert rules(flags) == ["a", "b", "c", "c", "d"]
    assert [f["name"] for f in flags] == ["contradiction", "low_confidence", "out_of_range", "out_of_range",
                                          "neighbors_disagree"]
    assert "energy (RMS) 0.4" in flags[2]["reason"] and "tempo (BPM) 131" in flags[3]["reason"]
    # "chill, hype": hype is among the neighbors' moods, so rule d stays quiet
    flags, _ = agent.check(11, entry(["chill", "hype"], 0.3), profiles[11], ref)
    assert rules(flags) == ["a", "b", "c", "c"]


# ---------- Step 1: validation, batching and retries ----------

def test_parse_labels_drops_moods_outside_the_vocabulary():
    text = json.dumps({"labels": [
        {"deezer_id": 1, "moods": ["Chill", "calm", "focus", "sad", "dark"], "confidence": 1.4, "reason": "a\nb"},
        {"deezer_id": 2, "moods": ["melancholy"], "confidence": 0.5, "reason": "x"},
        {"deezer_id": 99, "moods": ["chill"], "confidence": 0.5, "reason": "x"},
        {"deezer_id": 1, "moods": ["hype"], "confidence": 0.5, "reason": "again"},
        {"deezer_id": 3, "moods": ["hype"], "confidence": "high", "reason": ""},
    ]})
    parsed = agent.parse_labels(text, [1, 2, 3])
    assert parsed["labels"][1] == {"moods": ["chill", "focus", "sad"], "confidence": 1.0, "reason": "a"}
    assert parsed["labels"][3] == {"moods": ["hype"], "confidence": 0.0, "reason": "(no reason given)"}
    assert 2 not in parsed["labels"]                                      # nothing valid left
    problems = " | ".join(parsed["problems"])
    for expected in ("'calm'", "first 3 of 4", "song 2: no valid moods", "song 99 wasn't in the batch",
                     "labeled twice", "no confidence"):
        assert expected in problems
    assert agent.parse_labels("nope", [1])["error"]


def fake_model(monkeypatch, answer):
    """Replaces the model call; answer(batch_ids, call_number) returns the JSON text."""
    calls = []

    def respond(task, system, prompt, schema, mock, is_usable, use_cache=True):
        ids = [s["deezer_id"] for s in json.loads(prompt.split("Songs to label:\n")[1])]
        calls.append(ids)
        return {"text": answer(ids, len(calls)), "source": "gemini"}

    monkeypatch.setattr(rag, "structured_response", respond)
    return calls


def good(ids):
    return json.dumps({"labels": [{"deezer_id": i, "moods": ["chill"], "confidence": 0.9, "reason": "r"} for i in ids]})


def test_songs_are_labeled_in_batches_of_25(monkeypatch):
    calls = fake_model(monkeypatch, lambda ids, n: good(ids))
    profiles = {i: profile(i) for i in range(1, 61)}
    counter = Counter()
    labels, problems = agent.label_songs(list(profiles), profiles, None, counter)
    assert [len(c) for c in calls] == [25, 25, 10] and sum(calls, []) == list(range(1, 61))
    assert len(labels) == 60 and problems == [] and counter["gemini"] == 3


def test_a_failed_batch_is_retried_once(monkeypatch):
    calls = fake_model(monkeypatch, lambda ids, n: "not json" if n == 1 else good(ids))
    profiles = {i: profile(i) for i in range(1, 4)}
    labels, problems = agent.label_songs([1, 2, 3], profiles, None, Counter())
    assert len(calls) == 2 and set(labels) == {1, 2, 3}

    calls = fake_model(monkeypatch, lambda ids, n: "not json")
    labels, problems = agent.label_songs([1, 2, 3], profiles, None, Counter())
    assert len(calls) == 2 and labels == {}                              # only once, then give up
    assert any("no label after a retry for 3 song(s)" in p for p in problems)


def test_the_retry_only_resends_the_missing_songs(monkeypatch):
    calls = fake_model(monkeypatch, lambda ids, n: good(ids[:2]) if n == 1 else good(ids))
    profiles = {i: profile(i) for i in range(1, 5)}
    labels, _ = agent.label_songs([1, 2, 3, 4], profiles, None, Counter())
    assert calls == [[1, 2, 3, 4], [3, 4]] and set(labels) == {1, 2, 3, 4}


def test_api_errors_are_retried_but_the_daily_limit_stops_the_run(monkeypatch):
    attempts = []

    def flaky(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise rag.LLMUnavailable("Gemini API error 500")
        return {"text": good([1]), "source": "gemini"}

    monkeypatch.setattr(rag, "structured_response", flaky)
    labels, problems = agent.label_songs([1], {1: profile(1)}, None, Counter())
    assert len(attempts) == 2 and 1 in labels and "500" in problems[0]

    def exhausted(*args, **kwargs):
        raise rag.DailyLimitReached("Daily limit reached (18/18 Gemini calls today)")

    monkeypatch.setattr(rag, "structured_response", exhausted)
    with pytest.raises(agent.BudgetStop, match="Daily limit"):
        agent.label_songs([1], {1: profile(1)}, None, Counter())


def test_examples_are_sent_as_few_shot_examples():
    prompt = agent.label_prompt([{"deezer_id": 1}], [{"deezer_id": 9, "moods": ["sad"]}])
    assert prompt.index("Examples labeled by the listener") < prompt.index("Songs to label")
    assert "Examples" not in agent.label_prompt([{"deezer_id": 1}], None)


# ---------- the real-call path: cache and budget shared with rag ----------

def test_structured_calls_use_the_cache_and_the_daily_budget(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sent = []
    monkeypatch.setattr(rag, "gemini_generate", lambda *args: sent.append(args) or good([1]))
    call = lambda prompt: rag.structured_response("label", "sys", prompt, {}, mock=lambda: "", is_usable=lambda t: True)

    assert call("p")["source"] == "gemini" and call("p")["source"] == "cache" and len(sent) == 1
    for n in range(rag.MAX_CALLS_PER_DAY - 1):
        call(f"other {n}")
    with pytest.raises(rag.DailyLimitReached):
        call("one too many")
    assert len(sent) == rag.MAX_CALLS_PER_DAY


# ---------- the split, hands-win, and the whole run ----------

def test_the_split_is_made_once_and_never_changes(tmp_path):
    path = tmp_path / "split.json"
    labels = {i: ["chill"] for i in range(100, 157)}
    split = agent.load_or_create_split(labels, path)
    assert len(split["examples"]) == 15 and len(split["test"]) == 42 and split["seed"] == agent.SPLIT_SEED
    assert agent.load_or_create_split({**labels, 999: ["sad"]}, path) == split           # more labels later: same split
    other = tmp_path / "again.json"
    assert agent.load_or_create_split(labels, other)["examples"] == split["examples"]    # the seed is fixed
    with pytest.raises(SystemExit, match="more than 15"):
        agent.load_or_create_split({1: ["sad"]}, tmp_path / "too_few.json")


def test_hand_labels_win_over_agent_labels_and_estimates():
    library = [song(i) for i in (1, 2, 3)]
    songs = {s["deezer_id"]: s for s in label.with_moods(library, None, labels={1: ["sad"]},
                                                         auto_labels={1: ["party"], 2: ["hype"]})}
    assert songs[1]["moods"] == ["sad"] and "moods_estimated" not in songs[1]
    assert songs[2]["moods_estimated"] == ["hype"] and "moods" not in songs[2]
    assert "moods" not in songs[3] and "moods_estimated" not in songs[3]
    assert rag.describe(songs[2]).endswith("Moods (estimated): hype.")


def small_world(tmp_path, n_songs=50, n_hand=20):
    rng = np.random.default_rng(1)
    library = [song(i, energy_label=("low", "medium", "high")[i % 3], tempo_label=("slow", "fast")[i % 2])
               for i in range(1, n_songs + 1)]
    features = pd.DataFrame([{"deezer_id": s["deezer_id"], **dict(zip(FEATURE_COLUMNS, rng.normal(size=len(FEATURE_COLUMNS))))}
                             for s in library])
    stars = tmp_path / "stars"
    stars.mkdir()
    for s in library:
        (stars / f"{s['deezer_id']}.json").write_text(json.dumps([{"t": 0.5, "pitch": 60, "size": 1}] * (s["deezer_id"] % 7 + 1)))
    moods = ["chill", "focus", "hype", "party", "sad"]
    hand = {i: [moods[i % 5], moods[(i + 2) % 5]] for i in range(1, n_hand + 1)}
    return library, features, hand, stars


def test_the_agent_never_labels_hand_labeled_songs_and_runs_end_to_end(tmp_path):
    library, features, hand, stars = small_world(tmp_path)
    result = agent.run(library, features, hand, split_path=tmp_path / "split.json", stars_dir=stars)
    assert set(result["target_ids"]) == set(range(21, 51))             # only songs without a hand label
    assert len(result["test_ids"]) == 5 and len(result["example_ids"]) == 15
    assert set(result["records"]) == set(result["target_ids"]) | set(result["test_ids"])
    assert result["calls"]["mock"] >= 3 and result["calls"]["gemini"] == 0
    assert result["profiles"][3]["stars_per_second"] == round(4 / 30, 2)

    out = {name: tmp_path / "out" / file for name, file in
           (("auto_labels", "auto.json"), ("trace", "trace.jsonl"), ("report", "eval.md"), ("interactions", "ai.md"))}
    agent.write_outputs(result, "mock", "", out)
    auto = json.loads(out["auto_labels"].read_text())["songs"]
    assert set(map(int, auto)) == set(result["target_ids"])            # test songs aren't output labels
    trace = [json.loads(line) for line in out["trace"].read_text().splitlines()]
    assert len(trace) == 35 and {"initial", "flags", "revision", "final", "source"} <= set(trace[0])
    report = out["report"].read_text()
    for row in ("Model, no examples", "with my 15 examples", "After the check and fix steps",
                "Nearest-neighbor estimate", "Test songs:** 5", "API calls:"):
        assert row in report
    assert "Five example traces" in out["interactions"].read_text()


def test_agreement_scores():
    truth = {1: ["chill", "sad"], 2: ["hype"], 3: ["party"]}
    score = agent.agreement({1: ["sad"], 2: ["chill"]}, truth, [1, 2, 3])
    assert score == {"n": 3, "covered": 2, "any_match": pytest.approx(1 / 3), "overlap": pytest.approx(0.5 / 3)}


# ---------- Step 4: human review ----------

def scripted(*answers):
    queue = list(answers)
    return lambda prompt: queue.pop(0) if queue else (_ for _ in ()).throw(EOFError)


def test_review_accepts_replaces_skips_and_saves_hand_labels(tmp_path):
    library = [song(i) for i in (1, 2, 3, 4)]
    auto = {"songs": {
        "1": {"moods": ["chill", "party"], "status": "needs_review", "reason": "r", "flags": ["chill contradicts party"]},
        "2": {"moods": ["hype"], "status": "needs_review", "reason": "r", "flags": []},
        "3": {"moods": ["sad"], "status": "accepted", "reason": "r"},
        "4": {"moods": ["focus"], "status": "needs_review", "reason": "r", "flags": []},
    }}
    auto_path, labels_path = tmp_path / "auto.json", tmp_path / "labels.csv"
    auto_path.write_text(json.dumps(auto))
    out = []
    saved = review.run(library, auto_path, labels_path, read=scripted("", "s", "5 7"), write=out.append)
    assert saved == 2
    assert label.load_labels(labels_path) == {1: ["chill", "party"], 4: ["sad", "dark"]}
    assert any("Flag: chill contradicts party" in line for line in out)

    # Reviewed songs are hand labels now: they leave the queue, and they win in descriptions
    assert review.review_queue(auto, label.load_labels(labels_path)) == [2]
    described = label.with_moods(library, None, label.load_labels(labels_path), label.load_auto_labels(auto_path))
    assert described[3]["moods"] == ["sad", "dark"]
