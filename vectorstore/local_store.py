import json
import re
from pathlib import Path

from langchain_core.documents import Document


STORE_PATH = Path("db/local_chunks.json")


def _extract_terms(text):
    normalized = (text or "").lower()
    terms = set()

    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized):
        if len(token) <= 2:
            terms.add(token)
            continue

        terms.add(token)

        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            terms.update(token[i : i + 2] for i in range(len(token) - 1))

    return terms


def _load_raw_chunks():
    if not STORE_PATH.exists():
        return []

    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_chunks(documents):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]
    STORE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def retrieve_chunks(query, k=4):
    chunks = _load_raw_chunks()
    if not chunks:
        return []

    tokens = _extract_terms(query)
    scored = []

    for chunk in chunks:
        text = chunk.get("page_content", "")
        words = _extract_terms(text)
        score = len(tokens & words)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    docs = [
        Document(
            page_content=item[1].get("page_content", ""),
            metadata=item[1].get("metadata", {})
        )
        for item in scored[:k]
        if item[1].get("page_content")
    ]

    return docs
