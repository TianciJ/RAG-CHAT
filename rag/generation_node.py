from functools import lru_cache

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from app.config import settings
from rag.prompt import get_prompt


@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(
        model=settings.LLM_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )


def _build_local_answer(question: str, context: str) -> str:
    cleaned_context = (context or "").strip()
    if not cleaned_context:
        return "上下文中没有找到相关信息。"

    lines = []
    for block in cleaned_context.split("\n\n"):
        block = block.strip()
        if block:
            lines.append(block)

    snippet = " ".join(lines)
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
        answer = _build_local_answer(question, context)
        return {
            "messages": [
                AIMessage(content=answer)
            ]
        }

    llm = get_llm()
    prompt = get_prompt()
    prompt_messages = prompt.format_messages(
        history=history or "（暂无历史消息）",
        context=context or "（暂无检索到的上下文）",
        question=question,
    )

    response = llm.invoke(prompt_messages)

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }
