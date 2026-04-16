from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ingest.document_registry as document_registry
import memory.sqlite_checkpoint as sqlite_checkpoint
import vectorstore.local_store as local_store
from api.chat_routers import get_graph
from app.config import settings
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    upload_dir = tmp_path / "uploads"
    db_dir.mkdir()
    upload_dir.mkdir()

    monkeypatch.setattr(settings, "USE_LOCAL_RAG", True)
    monkeypatch.setattr(settings, "DATA_DIR", db_dir)
    monkeypatch.setattr(settings, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(settings, "LOCAL_CHUNKS_PATH", db_dir / "local_chunks.json")
    monkeypatch.setattr(settings, "DOCUMENT_REGISTRY_PATH", db_dir / "documents.json")
    monkeypatch.setattr(settings, "CHECKPOINT_DB_PATH", db_dir / "checkpoints.db")
    monkeypatch.setattr(settings, "DEFAULT_DUPLICATE_STRATEGY", "replace")

    monkeypatch.setattr(local_store, "STORE_PATH", settings.LOCAL_CHUNKS_PATH)
    monkeypatch.setattr(document_registry, "REGISTRY_PATH", settings.DOCUMENT_REGISTRY_PATH)
    monkeypatch.setattr(sqlite_checkpoint, "DB_PATH", settings.CHECKPOINT_DB_PATH)

    get_graph.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_graph.cache_clear()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "local"}


def test_upload_list_chat_delete_document_flow(client):
    upload_response = client.post(
        "/upload",
        files={"file": ("note.md", "# note\n苹果是红色的，香蕉是黄色的。".encode("utf-8"), "text/markdown")},
        data={"duplicate_strategy": "replace"},
    )

    assert upload_response.status_code == 200
    assert upload_response.json()["status"] == "success"

    documents_response = client.get("/documents")
    payload = documents_response.json()
    assert payload["count"] == 1
    assert payload["documents"][0]["filename"] == "note.md"

    chat_response = client.post(
        "/chat",
        json={"query": "苹果是什么颜色？", "session_id": "api-test"},
    )

    assert chat_response.status_code == 200
    chat_payload = chat_response.json()
    assert "红色" in chat_payload["answer"] or "红色" in chat_payload["debug_context"]
    assert chat_payload["retrieval_query"]
    assert chat_payload["retrieval_scores"]

    delete_response = client.delete("/documents/note.md")
    assert delete_response.status_code == 200
    assert delete_response.json()["chunks_deleted"] > 0

    documents_response = client.get("/documents")
    assert documents_response.json()["count"] == 0


def test_duplicate_strategy_skip_and_reject(client):
    first_upload = client.post(
        "/upload",
        files={"file": ("dup.md", "第一版内容".encode("utf-8"), "text/markdown")},
        data={"duplicate_strategy": "replace"},
    )
    assert first_upload.status_code == 200

    second_upload = client.post(
        "/upload",
        files={"file": ("dup.md", "第一版内容".encode("utf-8"), "text/markdown")},
        data={"duplicate_strategy": "skip"},
    )
    assert second_upload.status_code == 200
    assert second_upload.json()["status"] == "skipped"

    third_upload = client.post(
        "/upload",
        files={"file": ("dup.md", "第二版内容".encode("utf-8"), "text/markdown")},
        data={"duplicate_strategy": "reject"},
    )
    assert third_upload.status_code == 409


def test_rebuild_endpoint_reindexes_uploaded_files(client):
    client.post(
        "/upload",
        files={"file": ("a.md", "A 文档内容".encode("utf-8"), "text/markdown")},
        data={"duplicate_strategy": "replace"},
    )
    client.post(
        "/upload",
        files={"file": ("b.md", "B 文档内容".encode("utf-8"), "text/markdown")},
        data={"duplicate_strategy": "replace"},
    )

    rebuild_response = client.post("/documents/rebuild")
    payload = rebuild_response.json()

    assert rebuild_response.status_code == 200
    assert payload["status"] == "success"
    assert payload["documents"] == 2
    assert set(payload["filenames"]) == {"a.md", "b.md"}
