from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = "default"


@lru_cache(maxsize=1)
def get_graph():
    from rag.graph import build_graph

    return build_graph()


@router.post("/chat")
def chat(payload: ChatRequest):
    from langchain_core.messages import HumanMessage

    graph = get_graph()

    response = graph.invoke(
        {
            "messages": [
                HumanMessage(content=payload.query)
            ]
        },
        config={
            "configurable": {
                "thread_id": payload.session_id
            }
        }
    )

    return {
        "answer": response["messages"][-1].content,
        "session_id": payload.session_id,
        "debug_context": response.get("context", "")
    }
