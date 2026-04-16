"""
文件名：memory/sqlite_checkpoint.py
最后修改时间：2026-04-16
模块功能：为 LangGraph 提供 SQLite 持久化检查点，实现会话记忆保存。
模块相关技术：SQLite、LangGraph Checkpointer、Python 数据库连接。
"""

import sqlite3

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    use_sqlite = True
except ImportError:
    try:
        from langgraph.checkpoints.sqlite import SqliteSaver
        use_sqlite = True
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver
        use_sqlite = False

from app.config import settings


DB_PATH = settings.CHECKPOINT_DB_PATH


def build_checkpointer():
    if not use_sqlite:
        return MemorySaver()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)
