"""The command line runs from the committed data only: no Spotify, no MP3s, no model calls."""

import pytest

import sonata.rag as rag
from sonata.__main__ import build_parser, main

from test_rag import FakeEmbedder


@pytest.fixture(autouse=True)
def offline_embedder(monkeypatch):
    """Keep CLI tests fast and offline: a fake embedder instead of downloading MiniLM."""
    monkeypatch.setattr(rag, "minilm_embedder", FakeEmbedder())
    monkeypatch.setattr(rag.ask, "__kwdefaults__", {**rag.ask.__kwdefaults__, "embedder": FakeEmbedder()})
    monkeypatch.setattr(rag, "MIN_RELEVANCE", 0.0)


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_recommend_prints_top_5(capsys):
    assert main(["recommend"]) == 0
    out = capsys.readouterr().out
    assert "5. " in out and "6. " not in out and "Similarity" in out


def test_ask_prints_songs_reasons_and_retrieval_similarity(capsys):
    assert main(["ask", "reggaeton for a party"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Source: mock") and "(retrieval similarity " in out and "1. " in out


def test_ask_rejects_an_empty_request(capsys):
    assert main(["ask", "   "]) == 2
    assert "type what you'd like to hear" in capsys.readouterr().err


def test_recommend_rejects_k_below_1(capsys):
    assert main(["recommend", "-k", "0"]) == 2
