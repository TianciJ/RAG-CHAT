from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from ingest.embedder import get_embedding
from app.config import settings


@lru_cache(maxsize=1)
def init_pinecone_index():
    pc = Pinecone(
        api_key=settings.PINECONE_API_KEY
    )

    index_name = settings.PINECONE_INDEX_NAME

    # 如果索引不存在，就先创建索引。
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # MiniLM-L6-v2 的向量维度是 384
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # 免费层可用区域
            )
        )

    return pc.Index(index_name)


@lru_cache(maxsize=1)
def get_vectorstore():
    embedding = get_embedding()

    index = init_pinecone_index()

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding
    )

    return vectorstore
