from functools import lru_cache

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from app.config import settings


@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(
        model=settings.LLM_MODEL,
        api_key=settings.GROQ_API_KEY
    )


def generation_node(state):

    messages = state["messages"]

    context = state["context"]

    question = messages[-1].content
    history = "\n".join(
        f"{message.type}: {message.content}"
        for message in messages[:-1]
    )

    if settings.USE_LOCAL_RAG:
        if context.strip():
            answer = (
                "根据上传的上下文，给你一个简洁回答：\n\n"
                f"{context}"
            )
        else:
            answer = "我没有在上传的文档中找到相关信息。"
        return {
            "messages": [
                AIMessage(content=answer)
            ]
        }

    llm = get_llm()

    prompt = f"""
你是一个中文 RAG Chat 助手。请结合对话历史和检索到的上下文回答用户问题。
如果用户在当前会话里追问之前的内容，请优先参考对话历史。
如果上下文里没有答案，请明确说明“不确定”或“上下文中没有相关信息”，不要编造。

对话历史：
{history or "（暂无历史消息）"}

检索到的上下文：
{context or "（暂无检索上下文）"}

问题：
{question}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }
