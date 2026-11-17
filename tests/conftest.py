import pytest

import sonata.label as label
import sonata.rag as rag

REAL_GEMINI_RESPONSE = rag.gemini_response


@pytest.fixture
def real_gemini_response():
    """The unpatched function, for tests that fake the SDK client underneath it."""
    return REAL_GEMINI_RESPONSE


@pytest.fixture(autouse=True)
def offline_llm(tmp_path, monkeypatch):
    """Every test runs in mock mode, with no real key, and never touches the repo's caches or usage log."""
    monkeypatch.setattr(rag, "ENV_PATH", tmp_path / "no.env")   # the real .env is never loaded
    monkeypatch.setattr(rag, "LLM_CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(rag, "USAGE_PATH", tmp_path / "logs" / "api_usage.json")
    monkeypatch.setattr(rag, "REQUEST_LOG_PATH", tmp_path / "logs" / "sonata.log")
    monkeypatch.setattr(rag, "EMBEDDING_CACHE_PATH", tmp_path / "cache" / "text_embeddings.npz")
    # No hand mood labels unless a test passes them, so results don't depend on data/mood_labels.csv
    monkeypatch.setattr(label, "LABELS_PATH", tmp_path / "no_mood_labels.csv")
    monkeypatch.setattr(label, "SKIPS_PATH", tmp_path / "cache" / "mood_label_skips.json")
    monkeypatch.setattr(label, "AUTO_LABELS_PATH", tmp_path / "no_auto_labels.json")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def no_real_calls(*args, **kwargs):
        raise AssertionError("a test tried to make a real Gemini call")

    monkeypatch.setattr(rag, "gemini_response", no_real_calls)
    monkeypatch.setattr(rag, "gemini_generate", no_real_calls)
