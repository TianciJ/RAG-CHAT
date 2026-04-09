from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from rag.retrieval import build_retrieval_query, format_documents


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


def test_format_documents_contains_source_metadata():
    docs = [
        Document(
            page_content="这是第一段内容",
            metadata={
                "source_file": "list.pdf",
                "page": 2,
                "chunk_index": 1,
            },
        )
    ]

    context = format_documents(docs)

    assert "list.pdf" in context
    assert "页码: 2" in context
    assert "段号: 1" in context
    assert "这是第一段内容" in context
