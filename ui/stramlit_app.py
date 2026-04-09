import streamlit as st
import requests

st.title("中文 RAG Chat")

session_id = st.session_state.setdefault("session_id", "streamlit")
query = st.text_input("请输入问题")

if query:

    response = requests.post(
        "http://localhost:8000/chat",
        json={"query": query, "session_id": session_id},
        timeout=60
    )

    if response.ok:
        st.write(response.json()["answer"])
    else:
        st.error(f"请求失败：{response.status_code}")
