from pathlib import Path

from langchain_core.messages import HumanMessage

import memory.sqlite_checkpoint as sqlite_checkpoint
from app.config import settings
from rag.graph import build_graph


def test_messages_accumulate_with_same_thread(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoints.db"
    monkeypatch.setattr(settings, "USE_LOCAL_RAG", True)
    monkeypatch.setattr(sqlite_checkpoint, "DB_PATH", checkpoint_path)

    graph = build_graph()
    config = {"configurable": {"thread_id": "memory-check"}}

    graph.invoke({"messages": [HumanMessage(content="第一轮：请记住我叫小明。")]}, config=config)
    graph.invoke({"messages": [HumanMessage(content="第二轮：我叫什么名字？")]}, config=config)

    state = graph.get_state(config)
    assert len(state.values["messages"]) >= 4


def test_messages_do_not_mix_across_threads(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoints.db"
    monkeypatch.setattr(settings, "USE_LOCAL_RAG", True)
    monkeypatch.setattr(sqlite_checkpoint, "DB_PATH", checkpoint_path)

    graph = build_graph()
    config_a = {"configurable": {"thread_id": "memory-a"}}
    config_b = {"configurable": {"thread_id": "memory-b"}}

    graph.invoke({"messages": [HumanMessage(content="第一轮：A 说他叫小红。")]}, config=config_a)
    graph.invoke({"messages": [HumanMessage(content="第一轮：B 说他叫小蓝。")]}, config=config_b)

    state_a = graph.get_state(config_a)
    state_b = graph.get_state(config_b)

    assert state_a.values["messages"] != state_b.values["messages"]
