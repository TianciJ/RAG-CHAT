# 中文 RAG-Chat

这是一个中文 RAG Chat 项目，包含以下核心能力：

- 文档导入与切分
- 基于向量库的检索
- LangGraph 多节点问答流程
- SQLite 持久化记忆
- FastAPI 后端与 Streamlit 前端

## 快速开始

1. 创建虚拟环境。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 在 `.env` 文件中配置环境变量：

```bash
GROQ_API_KEY=your_key
PINECONE_API_KEY=your_key
```

4. 启动后端：

```bash
python -m app.main
```

5. 在另一个终端启动前端：

```bash
streamlit run ui/stramlit_app.py
```

## 建议先测什么

1. `GET /health` 返回 `{"status": "ok"}`。
2. `POST /chat` 接收类似 `{"query": "你好", "session_id": "demo"}` 的 JSON。
3. 上传一个小的 PDF 或 Markdown，然后问一个必须依赖文档内容的问题。
4. 在同一个 `session_id` 下连续问两句，确认第二句能接着第一句理解。
5. 换一个新的 `session_id` 再问同样的问题，确认不会串会话。

## 说明

- 当前项目支持本地测试模式，也支持 Groq + Pinecone 的远程模式。
- 如果你要切到真实在线模式，需要确保 `.env` 中的 API Key 正确，并关闭本地模式开关。
