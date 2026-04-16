import pytest

from app.config import settings


def test_validate_runtime_requires_remote_keys(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_RAG", False)
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    monkeypatch.setattr(settings, "PINECONE_API_KEY", None)

    with pytest.raises(RuntimeError):
        settings.validate_runtime()
