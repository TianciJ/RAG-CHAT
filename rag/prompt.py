from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    return ChatPromptTemplate.from_template(
        """
你是一个中文 RAG Chat 助手。请严格根据给定上下文回答问题。
如果上下文里没有明确答案，请直接说明“不确定”或“上下文中没有相关信息”，不要编造。
回答时优先使用中文，表达尽量简洁、清楚。

{context}

问题：
{question}
"""
    )
