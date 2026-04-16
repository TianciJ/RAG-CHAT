"""
文件名：rag/generation_node.py
最后修改时间：2026-04-16
模块功能：执行生成节点逻辑，将历史消息与检索上下文交给模型生成最终回答。
模块相关技术：ChatOpenAI、ChatGroq、LangChain 消息、Prompt Template、中文回答约束。
"""

from functools import lru_cache

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import settings
from rag.prompt import get_prompt


@lru_cache(maxsize=1)
def get_llm():
    model_name = settings.resolve_llm_model()

    if settings.LLM_PROVIDER == "openai":
        kwargs = {
            "model": model_name,
            "api_key": settings.OPENAI_API_KEY,
            "temperature": 0,
        }
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    return ChatGroq(
        model=model_name,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )


def _build_local_answer(question: str, context: str) -> str:
    cleaned_context = (context or "").strip()
    if not cleaned_context:
        return "上下文中没有找到相关信息。"

    snippet = " ".join(
        block.strip()
        for block in cleaned_context.split("\n\n")
        if block.strip()
    )
    if len(snippet) > 500:
        snippet = snippet[:500] + "..."

    return f"根据检索到的上下文，与你的问题“{question}”相关的信息如下：{snippet}"


def generation_node(state):
    messages = state["messages"]
    context = state.get("context", "")
    question = messages[-1].content
    history = "\n".join(
        f"{message.type}: {message.content}"
        for message in messages[:-1]
    )

    if settings.USE_LOCAL_RAG:
        return {
            "messages": [
                AIMessage(content=_build_local_answer(question, context))
            ]
        }

    prompt = get_prompt()
    response = get_llm().invoke(
        prompt.format_messages(
            history=history or "（暂无历史消息）",
            context=context or "（暂无检索到的上下文）",
            question=question,
        )
    )

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }
