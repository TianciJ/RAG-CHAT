import importlib
from pathlib import Path

from langchain_core.documents import Document

import vectorstore.local_store as local_store


def test_local_store_persists_across_reload(monkeypatch):
    store_path = Path("db/test_local_chunks.json")
    if store_path.exists():
        store_path.unlink()

    monkeypatch.setattr(local_store, "STORE_PATH", store_path)

    docs = [
        Document(page_content="list.pdf 包含链接和摘要。"),
        Document(page_content="FastAPI 提供了 chat 和 upload 接口。"),
    ]

    local_store.save_chunks(docs)
    first_pass = local_store.retrieve_chunks("list.pdf 里有什么内容？", k=2)

    reloaded = importlib.reload(local_store)
    monkeypatch.setattr(reloaded, "STORE_PATH", store_path)

    second_pass = reloaded.retrieve_chunks("list.pdf 里有什么内容？", k=2)

    assert first_pass
    assert second_pass
    assert second_pass[0].page_content == first_pass[0].page_content

    if store_path.exists():
        store_path.unlink()
