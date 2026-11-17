import json
import re
import zlib

import numpy as np
import pandas as pd
import pytest

import sonata.rag as rag
from sonata.analyze import FEATURE_COLUMNS

DIM = 256


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts that share words are similar. Counts what it embeds."""

    def __init__(self):
        self.embedded = []

    def __call__(self, texts):
        self.embedded += texts
        vectors = np.zeros((len(texts), DIM))
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % DIM] += 1
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


GENRES = ["Reggaeton", "Rap", "Pop", "Jazz", "Electro"]


def song(i, genre, liked, energy="medium"):
    return {"deezer_id": 1000 + i, "title": f"Track {i}", "artist": f"Artist {i}", "genre": genre,
            "cover": None, "preview_path": f"data/previews/{1000 + i}.mp3", "liked": liked,
            "tempo_label": "fast", "energy_label": energy, "brightness_label": "warm"}


@pytest.fixture
def library():
    return [song(i, GENRES[i % len(GENRES)], liked=i < 6, energy=("low", "medium", "high")[i % 3])
            for i in range(20)]


@pytest.fixture
def features(library):
    rng = np.random.default_rng(0)
    rows = [{"deezer_id": s["deezer_id"], **dict(zip(FEATURE_COLUMNS, rng.normal(size=len(FEATURE_COLUMNS))))}
            for s in library]
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def no_relevance_gate(monkeypatch):
    """The fake embedder scores lower than MiniLM; the relevance gate and strong matches have their own tests."""
    monkeypatch.setattr(rag, "MIN_RELEVANCE", 0.0)
    monkeypatch.setattr(rag, "STRONG_MATCH", 0.0)


def ask(query, library, features, embedder=None, **kwargs):
    return rag.ask(query, library=library, features=features, embedder=embedder or FakeEmbedder(), **kwargs)


def use_gemini(monkeypatch, response=None, calls=None):
    """Switches to the gemini provider with a fake key and a fake API call."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = calls if calls is not None else []

    def fake(query, retrieved, model, api_key):
        calls.append(query)
        if isinstance(response, Exception):
            raise response
        ids = [item["song"]["deezer_id"] for item in retrieved]
        return response if response is not None else json.dumps(
            {"recommendations": [{"deezer_id": i, "reason": "Fits the request."} for i in ids[:4]]})

    monkeypatch.setattr(rag, "gemini_response", fake)
    return calls


# ---------- descriptions, embeddings, retrieval ----------

def test_describe_uses_title_artist_genre_and_labels():
    assert rag.describe(song(1, "Reggaeton", True, energy="high")) == \
        "Track 1 by Artist 1. Genre: Reggaeton. Tempo fast, energy high, brightness warm."
    bare = {"title": "X", "artist": "Y", "genre": None}
    assert rag.describe(bare) == "X by Y. Genre: unknown."


def test_embeddings_are_cached_and_only_new_descriptions_are_embedded(tmp_path):
    cache = tmp_path / "emb.npz"
    first, second = FakeEmbedder(), FakeEmbedder()
    texts = ["a b", "c d", "e f"]
    vectors = rag.embed_descriptions(texts, first, cache)
    again = rag.embed_descriptions(texts + ["g h"], second, cache)
    assert first.embedded == texts and second.embedded == ["g h"]
    assert np.allclose(vectors, again[:3])


def test_text_retrieval_returns_8_songs_matching_the_query(library, features):
    retrieved = rag.retrieve("reggaeton", library, rag.embed_descriptions(
        [rag.describe(s) for s in library], FakeEmbedder()), FakeEmbedder())
    assert len(retrieved) == rag.RETRIEVE_K
    assert all(item["song"]["genre"] == "Reggaeton" for item in retrieved[:4])
    assert all(item["audio_similarity"] is None for item in retrieved)
    scores = [item["score"] for item in retrieved]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 1 for s in scores)


def test_audio_blend_is_85_15(library, features):
    vectors = rag.embed_descriptions([rag.describe(s) for s in library], FakeEmbedder())
    audio = rag.audio_similarity(features, library)
    blended = rag.retrieve("reggaeton energy high", library, vectors, FakeEmbedder(), audio)
    for item in blended:
        expected = 0.85 * item["text_similarity"] + 0.15 * item["audio_similarity"]
        assert item["score"] == pytest.approx(expected)
    scores = [item["score"] for item in blended]
    assert scores == sorted(scores, reverse=True)


def test_liked_songs_are_not_compared_with_themselves(library, features):
    audio = rag.audio_similarity(features, library)
    liked = [s["deezer_id"] for s in library if s["liked"]]
    assert all(audio[i] < 0.999 for i in liked)
    assert set(audio) == {s["deezer_id"] for s in library}


# ---------- mock mode and validation ----------

def test_mock_mode_is_the_default_and_returns_3_to_5_retrieved_songs(library, features, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    result = ask("chill jazz", library, features)
    assert result["source"] == "mock" and result["message"] is None
    assert 3 <= len(result["recommendations"]) <= 5
    for pick in result["recommendations"]:
        assert pick["song"] in library and pick["reason"] and 0 <= pick["retrieval_similarity"] <= 1
        assert "confidence" not in pick


def test_use_audio_false_gives_text_only_similarity(library, features):
    result = ask("chill jazz", library, features, use_audio=False)
    assert result["use_audio"] is False and result["source"] == "mock"


def retrieved_for(query, library):
    vectors = rag.embed_descriptions([rag.describe(s) for s in library], FakeEmbedder())
    return rag.retrieve(query, library, vectors, FakeEmbedder())


def test_validate_response_drops_and_labels_bad_picks(library, caplog):
    retrieved = retrieved_for("pop", library)
    ids = [item["song"]["deezer_id"] for item in retrieved]
    response = {"recommendations": [
        {"deezer_id": 999999, "reason": "invented"},
        {"deezer_id": ids[0], "reason": "good"},
        {"deezer_id": ids[0], "reason": "repeat"},
        {"deezer_id": str(ids[1]), "reason": "id as string"},
        {"deezer_id": ids[2], "reason": "   "},
        {"deezer_id": "abc", "reason": "not a number"},
        {"deezer_id": ids[3], "reason": "fine"},
        {"deezer_id": ids[4], "reason": "fine"},
        {"deezer_id": ids[5], "reason": "fine"},
        {"deezer_id": ids[6], "reason": "sixth valid, over the limit"},
    ]}
    with caplog.at_level("WARNING", logger="rag"):
        checked = rag.validate_response(json.dumps(response), retrieved)

    assert checked["error"] is None
    assert [p["song"]["deezer_id"] for p in checked["picks"]] == [ids[0], ids[1], ids[3], ids[4], ids[5]]
    assert checked["dropped"] == [
        {"deezer_id": 999999, "why": "hallucination"}, {"deezer_id": ids[0], "why": "duplicate"},
        {"deezer_id": ids[2], "why": "no_reason"}, {"deezer_id": "abc", "why": "bad_id"},
        {"deezer_id": ids[6], "why": "over_limit"}]
    assert "Hallucination" in caplog.text and "999999" in caplog.text
    assert all(0 <= p["retrieval_similarity"] <= 1 for p in checked["picks"])


@pytest.mark.parametrize("text,error", [
    ("not json", "not valid JSON"),
    ("[1, 2, 3]", "no \"recommendations\" list"),
    ('{"recommendations": "Track 1"}', "no \"recommendations\" list"),
    (None, "not valid JSON"),
])
def test_validate_response_rejects_malformed_answers(library, text, error):
    checked = rag.validate_response(text, retrieved_for("pop", library))
    assert error in checked["error"] and checked["picks"] == []


@pytest.mark.parametrize("query,expected", [("", "type what"), ("   ", "type what"), (None, "type what"),
                                            ("x" * 501, "501 characters long")])
def test_bad_queries_are_rejected_with_a_message_and_no_model_call(library, features, monkeypatch, query, expected):
    calls = use_gemini(monkeypatch)
    result = ask(query, library, features)
    assert result["source"] == "rejected" and result["recommendations"] == [] and calls == []
    assert expected in result["message"]


def test_a_500_character_query_is_accepted():
    assert rag.validate_query("x" * 500) is None
    assert "500 characters or fewer" in rag.validate_query("x" * 501)


def test_hallucinated_picks_are_removed_but_valid_ones_kept(library, features, monkeypatch):
    def answer(ids):
        return json.dumps({"recommendations": [{"deezer_id": 42, "reason": "made up"}] +
                           [{"deezer_id": i, "reason": "fits"} for i in ids[:3]]})

    calls = []
    use_gemini(monkeypatch, calls=calls)
    monkeypatch.setattr(rag, "gemini_response",
                        lambda q, retrieved, m, k: answer([it["song"]["deezer_id"] for it in retrieved]))
    result = ask("pop", library, features)
    assert result["source"] == "gemini" and len(result["recommendations"]) == 3
    assert result["dropped"] == [{"deezer_id": 42, "why": "hallucination"}]
    assert "Removed 1 recommendation" in result["message"]


def test_too_few_valid_picks_falls_back_to_top_retrieved(library, features, monkeypatch):
    use_gemini(monkeypatch, response=json.dumps({"recommendations": [{"deezer_id": 42, "reason": "x"}]}))
    result = ask("pop", library, features)
    assert result["source"] == "fallback" and "only 0 valid songs" in result["message"]
    assert [p["song"]["deezer_id"] for p in result["recommendations"]] == result["retrieved_ids"][:5]
    assert result["dropped"] == [{"deezer_id": 42, "why": "hallucination"}]


def test_requests_matching_nothing_skip_the_model(library, features, monkeypatch):
    monkeypatch.setattr(rag, "MIN_RELEVANCE", 0.4)
    calls = use_gemini(monkeypatch)
    result = ask("qwxz vbnm plork", library, features)     # shares no words with any description
    assert result["source"] == "fallback" and "Nothing in your library closely matches" in result["message"]
    assert calls == [] and 3 <= len(result["recommendations"]) <= 5


def test_every_request_is_logged(library, features, monkeypatch):
    use_gemini(monkeypatch)
    ask("late night rap", library, features)
    ask("", library, features)
    entries = [json.loads(line) for line in rag.REQUEST_LOG_PATH.read_text().splitlines()]
    assert len(entries) == 2
    first, second = entries
    assert set(first) >= {"timestamp", "query", "retrieved_ids", "returned_ids", "dropped_ids",
                          "hallucinated_ids", "latency_ms", "source"}
    assert first["query"] == "late night rap" and len(first["retrieved_ids"]) == rag.RETRIEVE_K
    assert set(first["returned_ids"]) <= set(first["retrieved_ids"]) and first["latency_ms"] >= 0
    assert second["source"] == "rejected" and second["retrieved_ids"] == [] and second["returned_ids"] == []


def test_use_cache_false_makes_a_fresh_call(library, features, monkeypatch):
    calls = use_gemini(monkeypatch)
    ask("pop songs", library, features)
    fresh = ask("pop songs", library, features, use_cache=False)
    assert fresh["source"] == "gemini" and len(calls) == 2


# ---------- Gemini: cache, limits, errors ----------

def test_repeated_request_costs_zero_calls(library, features, monkeypatch):
    calls = use_gemini(monkeypatch)
    first = ask("late night rap", library, features)
    second = ask("late night rap", library, features)
    assert first["source"] == "gemini" and second["source"] == "cache"
    assert len(calls) == 1 and rag.calls_today() == 1
    assert [p["song"] for p in first["recommendations"]] == [p["song"] for p in second["recommendations"]]
    assert len(list(rag.LLM_CACHE_DIR.glob("*.json"))) == 1


def test_daily_limit_blocks_the_19th_call(library, features, monkeypatch):
    calls = use_gemini(monkeypatch)
    for _ in range(rag.MAX_CALLS_PER_DAY):
        rag.record_call()
    result = ask("something new", library, features)
    assert calls == [] and rag.calls_today() == rag.MAX_CALLS_PER_DAY
    assert result["source"] == "fallback" and "Daily limit reached (18/18" in result["message"]
    assert 3 <= len(result["recommendations"]) <= 5


def test_cached_answers_still_work_past_the_limit(library, features, monkeypatch):
    calls = use_gemini(monkeypatch)
    ask("pop songs", library, features)
    for _ in range(rag.MAX_CALLS_PER_DAY):
        rag.record_call()
    assert ask("pop songs", library, features)["source"] == "cache" and len(calls) == 1


def test_rate_limit_falls_back_without_crashing(library, features, monkeypatch):
    use_gemini(monkeypatch, response=rag.LLMUnavailable("Gemini rate limit or daily quota reached"))
    result = ask("pop songs", library, features)
    assert result["source"] == "fallback" and "rate limit" in result["message"]
    assert rag.calls_today() == 1               # the failed call still counts
    assert list(rag.LLM_CACHE_DIR.glob("*.json")) == []


def test_invalid_model_answer_falls_back_and_is_not_cached(library, features, monkeypatch):
    use_gemini(monkeypatch, response=json.dumps({"recommendations": [{"deezer_id": 1, "reason": "made up"}]}))
    result = ask("pop songs", library, features)
    assert result["source"] == "fallback" and "valid songs" in result["message"]
    assert list(rag.LLM_CACHE_DIR.glob("*.json")) == []


@pytest.mark.parametrize("missing,expected", [("GEMINI_API_KEY", "GEMINI_API_KEY is not set"),
                                              ("LLM_MODEL", "LLM_MODEL is not set")])
def test_missing_settings_fall_back_without_a_call(library, features, monkeypatch, missing, expected):
    calls = use_gemini(monkeypatch)
    monkeypatch.delenv(missing)
    result = ask("pop songs", library, features)
    assert result["source"] == "fallback" and expected in result["message"]
    assert calls == [] and rag.calls_today() == 0


def fake_sdk(monkeypatch, behaviour):
    """Replaces genai.Client so the real gemini_response runs without network access."""
    from google import genai

    class FakeModels:
        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return behaviour(kwargs)

    class FakeClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)


def test_gemini_429_is_reported_as_rate_limit(monkeypatch, real_gemini_response):
    from google.genai import errors

    def quota(kwargs):
        raise errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded",
                                                 "status": "RESOURCE_EXHAUSTED"}})

    fake_sdk(monkeypatch, quota)
    with pytest.raises(rag.LLMUnavailable, match="rate limit"):
        real_gemini_response("q", [], "test-model", "test-key")


@pytest.mark.parametrize("behaviour,match", [
    (lambda kwargs: (_ for _ in ()).throw(ConnectionError("offline")), "Could not reach Gemini"),
    (lambda kwargs: type("Response", (), {"text": None})(), "empty response"),
])
def test_gemini_network_errors_and_empty_answers_raise_llm_unavailable(monkeypatch, real_gemini_response,
                                                                      behaviour, match):
    fake_sdk(monkeypatch, behaviour)
    with pytest.raises(rag.LLMUnavailable, match=match):
        real_gemini_response("q", [], "test-model", "test-key")


def test_gemini_request_uses_system_prompt_and_json_schema(monkeypatch, real_gemini_response):
    seen = {}

    def ok(kwargs):
        seen.update(kwargs)
        return type("Response", (), {"text": '{"recommendations": []}'})()

    fake_sdk(monkeypatch, ok)
    assert real_gemini_response("q", [], "test-model", "test-key") == '{"recommendations": []}'
    config = seen["config"]
    assert seen["model"] == "test-model" and "ONLY from the provided list" in config.system_instruction
    assert config.response_mime_type == "application/json" and config.response_json_schema == rag.RESPONSE_SCHEMA


def test_gemini_request_disables_automatic_function_calling(monkeypatch, real_gemini_response):
    seen = {}
    fake_sdk(monkeypatch, lambda kwargs: seen.update(kwargs) or type("R", (), {"text": "{}"})())
    real_gemini_response("q", [], "test-model", "test-key")
    assert seen["config"].automatic_function_calling.disable is True


def test_very_long_queries_are_truncated_in_the_log(library, features):
    ask("y" * 5000, library, features)
    entry = json.loads(rag.REQUEST_LOG_PATH.read_text().splitlines()[-1])
    assert entry["source"] == "rejected" and entry["query"].endswith("… (5000 chars)")
    assert len(entry["query"]) < 1100


# ---------- relevance first: strong matches, taste as a tie-breaker ----------

class FixedEmbedder:
    """The query is always (1, 0); song vectors are made so their text similarity is exact."""

    def __call__(self, texts):
        return np.array([[1.0, 0.0]] * len(texts))


def at_similarity(*similarities):
    return np.array([[s, np.sqrt(1 - s * s)] for s in similarities])


def ranked(library, text, audio, **kwargs):
    retrieved = rag.retrieve("q", library, at_similarity(*text), FixedEmbedder(), audio, **kwargs)
    return [item["song"]["deezer_id"] for item in retrieved], retrieved


def test_taste_only_reorders_strong_matches(library, monkeypatch):
    monkeypatch.setattr(rag, "STRONG_MATCH", 0.5)
    songs = library[:5]
    ids = [s["deezer_id"] for s in songs]
    #           strong        strong        weak          weak          strong
    text = [0.60, 0.58, 0.49, 0.45, 0.52]
    audio = {ids[0]: 0.0, ids[1]: 1.0, ids[2]: 1.0, ids[3]: 0.0, ids[4]: 0.9}
    order, retrieved = ranked(songs, text, audio, k=5)
    # Among strong matches taste decides (0.85*.58+.15 > 0.85*.52+.135 > 0.85*.60); the weak
    # ones follow in text order, even though song 2 sounds exactly like a liked song
    assert order == [ids[1], ids[4], ids[0], ids[2], ids[3]]
    assert [item["strong"] for item in retrieved] == [True, True, True, False, False]
    assert rag.strong_count(retrieved) == 3 and rag.min_picks(retrieved) == 3


def test_a_weak_match_never_outranks_a_strong_one_however_it_sounds(library, monkeypatch):
    monkeypatch.setattr(rag, "STRONG_MATCH", 0.5)
    songs = library[:2]
    order, retrieved = ranked(songs, [0.50, 0.4999], {songs[0]["deezer_id"]: 0.0, songs[1]["deezer_id"]: 1.0}, k=2)
    assert order == [songs[0]["deezer_id"], songs[1]["deezer_id"]]
    assert retrieved[1]["score"] > retrieved[0]["score"]          # higher blend, still second


def few_strong_setup(library, monkeypatch, strong=1):
    """MiniLM-like scores: `strong` songs at 0.6, the rest at 0.42 (above the nonsense gate, below strong)."""
    monkeypatch.setattr(rag, "STRONG_MATCH", 0.45)
    monkeypatch.setattr(rag, "MIN_RELEVANCE", 0.40)
    sims = [0.6] * strong + [0.42] * (len(library) - strong)
    monkeypatch.setattr(rag, "embed_descriptions", lambda texts, embedder, cache=None: at_similarity(*sims))
    return library[:strong]


def test_few_strong_matches_tell_the_model_and_allow_a_shorter_answer(library, features, monkeypatch):
    strong = few_strong_setup(library, monkeypatch, strong=2)
    prompts = []
    use_gemini(monkeypatch)

    def fake(query, retrieved, model, api_key):
        prompts.append(rag.user_prompt(query, retrieved))
        return json.dumps({"recommendations": [{"deezer_id": strong[0]["deezer_id"], "reason": "Fits."}],
                           "note": "Nothing else in your library is really chill.\nsecond line ignored"})

    monkeypatch.setattr(rag, "gemini_response", fake)

    result = ask("chill", library, features, embedder=FixedEmbedder())
    assert "Few strong matches: only 2 of these songs" in prompts[0]
    assert '"strong_match": true' in prompts[0] and '"strong_match": false' in prompts[0]
    assert result["source"] == "gemini" and result["strong_matches"] == 2
    assert [p["song"]["deezer_id"] for p in result["recommendations"]] == [strong[0]["deezer_id"]]
    assert result["message"] == "Nothing else in your library is really chill."


def test_few_strong_matches_get_a_default_note_if_the_model_gives_none(library, features, monkeypatch):
    strong = few_strong_setup(library, monkeypatch, strong=1)
    use_gemini(monkeypatch, response=json.dumps({"recommendations": [{"deezer_id": strong[0]["deezer_id"], "reason": "Fits."}]}))
    result = ask("chill", library, features, embedder=FixedEmbedder())
    assert len(result["recommendations"]) == 1
    assert result["message"].startswith("No strong match in your library for this request: only 1 song fit")


def test_enough_strong_matches_still_need_3_picks(library, features, monkeypatch):
    strong = few_strong_setup(library, monkeypatch, strong=5)
    use_gemini(monkeypatch, response=json.dumps({"recommendations": [{"deezer_id": strong[0]["deezer_id"], "reason": "Fits."}]}))
    result = ask("chill", library, features, embedder=FixedEmbedder())
    assert result["source"] == "fallback" and "(needs 3)" in result["message"]
    assert len(result["recommendations"]) == 5


def test_few_strong_fallback_does_not_pad_the_list(library, features, monkeypatch):
    few_strong_setup(library, monkeypatch, strong=2)
    use_gemini(monkeypatch, response="not json")
    result = ask("chill", library, features, embedder=FixedEmbedder())
    assert result["source"] == "fallback" and len(result["recommendations"]) == 2
    assert "No strong match" in result["message"]


def test_mock_mode_follows_the_few_strong_rule(library, features, monkeypatch):
    few_strong_setup(library, monkeypatch, strong=1)
    result = ask("chill", library, features, embedder=FixedEmbedder())
    assert result["source"] == "mock" and len(result["recommendations"]) == 1
    assert "no strong match" in result["message"]
