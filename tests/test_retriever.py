from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from rag.retrieval import (
    build_catalog_documents,
    build_retrieval_query,
    format_documents,
    is_catalog_question,
    rerank_documents,
    summarize_documents,
)


def test_build_retrieval_query_includes_recent_history():
    messages = [
        HumanMessage(content="这份文件主要讲什么"),
        AIMessage(content="它主要讲了向量检索。"),
        HumanMessage(content="里面提到了召回吗"),
        AIMessage(content="提到了基础召回。"),
        HumanMessage(content="那具体怎么做的"),
    ]

    query = build_retrieval_query(messages[-1].content, messages)

    assert "历史相关对话" in query
    assert "这份文件主要讲什么" in query
    assert "里面提到了召回吗" in query
    assert "那具体怎么做的" in query


def test_catalog_question_detection_and_catalog_document():
    assert is_catalog_question("你知道你的RAG知识库里有哪些文章吗")

    docs = build_catalog_documents(
        [
            {"filename": "a.pdf", "chunks": 3, "updated_at": "2026-04-16 10:00:00"},
            {"filename": "b.md", "chunks": 2, "updated_at": "2026-04-16 11:00:00"},
        ]
    )

    assert docs
    assert "当前知识库共收录 2 个文件" in docs[0].page_content
    assert "a.pdf" in docs[0].page_content
    assert docs[0].metadata["_catalog_doc"] is True


def test_rerank_documents_combines_dense_and_keyword_scores():
    question = "这份文档有没有讲召回和重排"
    dense_docs = [
        Document(
            page_content="这里介绍向量召回和重排，先检索后 rerank，再把结果送入模型。",
            metadata={"source_file": "rag.md", "chunk_index": 1},
        )
    ]
    keyword_docs = [
        Document(
            page_content="这里在讲前端页面和按钮样式。",
            metadata={"source_file": "ui.md", "chunk_index": 3, "_keyword_score": 1.0},
        ),
        Document(
            page_content="这里介绍向量召回和重排，先检索后 rerank，再把结果送入模型。",
            metadata={"source_file": "rag.md", "chunk_index": 1, "_keyword_score": 6.0},
        ),
    ]

    ranked = rerank_documents(question, dense_docs, keyword_docs, top_k=2)
    summary = summarize_documents(ranked)

    assert ranked[0].metadata["source_file"] == "rag.md"
    assert summary[0]["retrieval_score"] >= summary[1]["retrieval_score"]


def test_format_documents_contains_source_metadata_and_scores():
    docs = [
        Document(
            page_content="这是第一段内容",
            metadata={
                "source_file": "list.pdf",
                "page": 2,
                "chunk_index": 1,
                "_retrieval_score": 0.92,
            },
        )
    ]

    context = format_documents(docs)

    assert "list.pdf" in context
    assert "页码: 2" in context
    assert "段号: 1" in context
    assert "综合分数: 0.92" in context
