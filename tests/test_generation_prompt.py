from langchain_core.messages import AIMessage, HumanMessage

import rag.generation_node as generation_node


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return AIMessage(content="模拟回答")


def test_generation_prompt_includes_history_and_context(monkeypatch):
    fake_llm = FakeLLM()
    monkeypatch.setattr(generation_node.settings, "USE_LOCAL_RAG", False)
    monkeypatch.setattr(generation_node, "get_llm", lambda: fake_llm)

    state = {
        "messages": [
            HumanMessage(content="我叫小明"),
            HumanMessage(content="我的名字是什么？"),
        ],
        "context": "list.pdf 里写着用户的名字是小明。",
    }

    result = generation_node.generation_node(state)

    assert result["messages"][0].content == "模拟回答"
    assert "我叫小明" in fake_llm.last_prompt
    assert "list.pdf 里写着用户的名字是小明。" in fake_llm.last_prompt
    assert "我的名字是什么？" in fake_llm.last_prompt
