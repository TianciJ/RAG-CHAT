from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


def build_retriever(vectorstore, *, search_type: str = "mmr", k: int = 6, fetch_k: int = 20):
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )


def build_retrieval_query(question: str, messages: Sequence[BaseMessage] | None = None, max_history_turns: int = 2) -> str:
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


def format_documents(docs: Sequence[Document]) -> str:
    if not docs:
        return ""

    sections = []
    for index, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata or {})
        source_file = metadata.get("source_file") or metadata.get("source") or "未知来源"
        chunk_index = metadata.get("chunk_index")
        page = metadata.get("page")

        header_parts = [f"片段{index}", f"来源: {source_file}"]
        if page is not None:
            header_parts.append(f"页码: {page}")
        if chunk_index is not None:
            header_parts.append(f"段号: {chunk_index}")

        header = "【" + " | ".join(header_parts) + "】"
        content = (doc.page_content or "").strip()
        sections.append(f"{header}\n{content}")

    return "\n\n".join(sections)
