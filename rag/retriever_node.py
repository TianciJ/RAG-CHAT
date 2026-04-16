"""
文件名：rag/retriever_node.py
最后修改时间：2026-04-16
模块功能：执行检索节点逻辑，完成查询改写、混合检索、知识库目录补充、重排和上下文组织。
模块相关技术：LangGraph 节点、Pinecone、混合检索、关键词检索、rerank。
"""

from app.config import settings
from ingest.document_registry import list_documents
from rag.retrieval import (
    build_catalog_documents,
    build_retrieval_query,
    build_retriever,
    format_documents,
    is_catalog_question,
    rerank_documents,
    summarize_documents,
)
from vectorstore.local_store import retrieve_chunks
from vectorstore.pinecone_store import get_vectorstore


def retrieve_node(state):
    messages = state["messages"]
    question = messages[-1].content
    retrieval_query = build_retrieval_query(question, messages)

    dense_docs = []
    keyword_docs = retrieve_chunks(question, k=settings.KEYWORD_TOP_K)

    if not settings.USE_LOCAL_RAG:
        vectorstore = get_vectorstore()
        try:
            retriever = build_retriever(
                vectorstore,
                k=settings.VECTOR_TOP_K,
                fetch_k=settings.VECTOR_FETCH_K,
            )
            dense_docs = retriever.invoke(retrieval_query)
        except Exception:
            dense_docs = vectorstore.similarity_search(
                retrieval_query,
                k=settings.VECTOR_TOP_K,
            )

    if is_catalog_question(question):
        keyword_docs = list(keyword_docs) + build_catalog_documents(list_documents())

    ranked_docs = rerank_documents(
        question=question,
        dense_docs=dense_docs,
        keyword_docs=keyword_docs,
        top_k=settings.FINAL_TOP_K,
    )

    return {
        "retrieval_query": retrieval_query,
        "retrieval_scores": summarize_documents(ranked_docs),
        "context": format_documents(ranked_docs),
    }
