"""
文件名：rag/retrieval.py
最后修改时间：2026-04-16
模块功能：提供检索 query 改写、混合检索重排、知识库目录摘要和上下文格式化能力。
模块相关技术：文本特征匹配、混合检索、轻量 rerank、LangChain 文档对象。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


CATALOG_KEYWORDS = {
    "知识库",
    "文档",
    "文章",
    "资料",
    "目录",
    "文件",
    "有哪些",
    "多少",
    "包含",
    "收录",
}


def build_retriever(vectorstore, *, search_type: str = "mmr", k: int = 8, fetch_k: int = 24):
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )


def build_retrieval_query(
    question: str,
    messages: Sequence[BaseMessage] | None = None,
    max_history_turns: int = 2,
) -> str:
    question = (question or "").strip()
    if not messages:
        return question

    recent_messages = [
        message
        for message in messages[:-1]
        if getattr(message, "content", "")
    ][-max_history_turns * 2 :]

    if not recent_messages:
        return question

    history_lines = []
    for message in recent_messages:
        role = "用户" if message.type in {"human", "user"} else "助手"
        history_lines.append(f"- {role}：{str(message.content).strip()}")

    history_block = "\n".join(history_lines)
    return f"历史相关对话：\n{history_block}\n\n当前问题：{question}"


def _extract_terms(text: str) -> set[str]:
    normalized = (text or "").lower()
    terms: set[str] = set()

    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized):
        if not token:
            continue
        terms.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) >= 2:
            terms.update(token[index : index + 2] for index in range(len(token) - 1))

    return terms


def is_catalog_question(question: str) -> bool:
    terms = _extract_terms(question)
    return len(terms & CATALOG_KEYWORDS) >= 2 or ("知识库" in question and "哪些" in question)


def build_catalog_documents(registry_documents: Sequence[dict]) -> list[Document]:
    if not registry_documents:
        return []

    lines = [f"当前知识库共收录 {len(registry_documents)} 个文件："]
    for index, document in enumerate(registry_documents, start=1):
        filename = document.get("filename", "未知文件")
        chunks = document.get("chunks", 0)
        updated_at = document.get("updated_at", "未知时间")
        lines.append(f"{index}. {filename}，共 {chunks} 个片段，最近更新时间：{updated_at}")

    return [
        Document(
            page_content="\n".join(lines),
            metadata={
                "source_file": "知识库目录",
                "chunk_index": 0,
                "_keyword_score": 100.0,
                "_catalog_doc": True,
            },
        )
    ]


def _document_key(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    source_file = metadata.get("source_file") or metadata.get("source") or "unknown"
    chunk_index = metadata.get("chunk_index") or metadata.get("page") or 0
    content_hash = hashlib.sha1((doc.page_content or "").encode("utf-8")).hexdigest()[:12]
    return f"{source_file}:{chunk_index}:{content_hash}"


def score_document(question: str, doc: Document) -> float:
    question_terms = _extract_terms(question)
    content = doc.page_content or ""
    content_terms = _extract_terms(content)
    metadata = dict(doc.metadata or {})

    overlap = len(question_terms & content_terms)
    score = float(overlap)

    if question and question in content:
        score += 8.0

    if question_terms and any(term in content.lower() for term in question_terms if len(term) > 2):
        score += 1.5

    source_file = str(metadata.get("source_file") or metadata.get("source") or "")
    if source_file:
        score += 0.5 * len(question_terms & _extract_terms(source_file))

    if metadata.get("_catalog_doc"):
        score += 20.0

    return score


def _normalize_rank_scores(docs: Sequence[Document]) -> dict[str, float]:
    if not docs:
        return {}

    max_index = max(len(docs) - 1, 1)
    scores = {}
    for index, doc in enumerate(docs):
        rank_score = 1.0 - (index / max_index)
        scores[_document_key(doc)] = round(rank_score, 4)
    return scores


def rerank_documents(
    question: str,
    dense_docs: Sequence[Document],
    keyword_docs: Sequence[Document],
    top_k: int = 6,
) -> list[Document]:
    dense_scores = _normalize_rank_scores(dense_docs)
    keyword_raw_scores = {
        _document_key(doc): max(
            float(dict(doc.metadata or {}).get("_keyword_score", 0.0)),
            score_document(question, doc),
        )
        for doc in keyword_docs
    }

    max_keyword_score = max(keyword_raw_scores.values(), default=1.0)
    merged_docs: dict[str, Document] = {}

    for doc in list(dense_docs) + list(keyword_docs):
        key = _document_key(doc)
        if key not in merged_docs:
            merged_docs[key] = Document(
                page_content=doc.page_content,
                metadata=dict(doc.metadata or {}),
            )

    ranked_items = []
    for key, doc in merged_docs.items():
        dense_score = dense_scores.get(key, 0.0)
        keyword_score_raw = keyword_raw_scores.get(key, 0.0)
        keyword_score = keyword_score_raw / max_keyword_score if max_keyword_score else 0.0
        retrieval_score = round(0.4 * dense_score + 0.6 * keyword_score, 4)

        doc.metadata["_dense_score"] = round(dense_score, 4)
        doc.metadata["_keyword_score"] = round(keyword_score_raw, 4)
        doc.metadata["_retrieval_score"] = retrieval_score

        ranked_items.append((retrieval_score, doc))

    ranked_items.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked_items[:top_k]]


def format_documents(docs: Sequence[Document]) -> str:
    if not docs:
        return ""

    sections = []
    for index, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata or {})
        source_file = metadata.get("source_file") or metadata.get("source") or "未知来源"
        chunk_index = metadata.get("chunk_index")
        page = metadata.get("page")
        retrieval_score = metadata.get("_retrieval_score")

        header_parts = [f"片段{index}", f"来源: {source_file}"]
        if page is not None:
            header_parts.append(f"页码: {page}")
        if chunk_index is not None:
            header_parts.append(f"段号: {chunk_index}")
        if retrieval_score is not None:
            header_parts.append(f"综合分数: {retrieval_score}")

        header = "【" + " | ".join(header_parts) + "】"
        sections.append(f"{header}\n{(doc.page_content or '').strip()}")

    return "\n\n".join(sections)


def summarize_documents(docs: Sequence[Document]) -> list[dict]:
    summary = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        summary.append(
            {
                "source_file": metadata.get("source_file") or metadata.get("source"),
                "chunk_index": metadata.get("chunk_index"),
                "page": metadata.get("page"),
                "dense_score": metadata.get("_dense_score", 0.0),
                "keyword_score": metadata.get("_keyword_score", 0.0),
                "retrieval_score": metadata.get("_retrieval_score", 0.0),
            }
        )
    return summary
