from app.config import settings
from rag.retrieval import build_retrieval_query, format_documents, build_retriever
from vectorstore.local_store import retrieve_chunks
from vectorstore.pinecone_store import get_vectorstore


def retrieve_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content
    retrieval_query = build_retrieval_query(query, messages)

    if settings.USE_LOCAL_RAG:
        docs = retrieve_chunks(retrieval_query)
    else:
        vectorstore = get_vectorstore()
        try:
            retriever = build_retriever(vectorstore)
            docs = retriever.invoke(retrieval_query)
        except Exception:
            docs = vectorstore.similarity_search(retrieval_query, k=6)

    context = format_documents(docs)

    return {
        "retrieval_query": retrieval_query,
        "retrieved_documents": docs,
        "context": context
    }
