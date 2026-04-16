"""
文件名：app/config.py
最后修改时间：2026-04-16
模块功能：集中管理环境变量与项目运行配置，统一控制模型、向量库、数据目录和部署参数。
模块相关技术：python-dotenv、环境变量管理、Python 配置类、Path 路径管理。
"""

from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


def _read_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


class Settings:
    APP_NAME = os.getenv("APP_NAME", "中文 RAG Chat")

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = _read_int("API_PORT", 8000)
    FRONTEND_API_BASE = os.getenv("FRONTEND_API_BASE", f"http://127.0.0.1:{API_PORT}")

    DATA_DIR = Path(os.getenv("DATA_DIR", "db"))
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
    LOCAL_CHUNKS_PATH = DATA_DIR / "local_chunks.json"
    DOCUMENT_REGISTRY_PATH = DATA_DIR / "documents.json"
    CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.db"
    SQLITE_DB = f"sqlite:///{(DATA_DIR / 'chat.db').as_posix()}"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chat")

    LLM_PROVIDER = os.getenv(
        "LLM_PROVIDER",
        "openai" if os.getenv("OPENAI_API_KEY") else "groq",
    ).strip().lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    USE_LOCAL_RAG = _read_bool("USE_LOCAL_RAG", _read_bool("DEV_MODE", False))
    DEFAULT_DUPLICATE_STRATEGY = os.getenv("DEFAULT_DUPLICATE_STRATEGY", "replace")

    CHUNK_SIZE = _read_int("CHUNK_SIZE", 500)
    CHUNK_OVERLAP = _read_int("CHUNK_OVERLAP", 50)
    VECTOR_TOP_K = _read_int("VECTOR_TOP_K", 8)
    VECTOR_FETCH_K = _read_int("VECTOR_FETCH_K", 24)
    KEYWORD_TOP_K = _read_int("KEYWORD_TOP_K", 8)
    FINAL_TOP_K = _read_int("FINAL_TOP_K", 6)

    ALLOWED_EXTENSIONS = {".pdf", ".md"}
    VALID_DUPLICATE_STRATEGIES = {"replace", "skip", "reject"}
    VALID_LLM_PROVIDERS = {"openai", "groq"}

    def ensure_directories(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def runtime_mode(self) -> str:
        return "local" if self.USE_LOCAL_RAG else "remote"

    def resolve_llm_model(self) -> str:
        if self.LLM_MODEL:
            return self.LLM_MODEL
        if self.LLM_PROVIDER == "openai":
            return "gpt-4o-mini"
        return "llama-3.3-70b-versatile"

    def validate_runtime(self) -> None:
        self.ensure_directories()

        if self.DEFAULT_DUPLICATE_STRATEGY not in self.VALID_DUPLICATE_STRATEGIES:
            raise RuntimeError(
                "DEFAULT_DUPLICATE_STRATEGY 只能是 replace、skip 或 reject。"
            )

        if self.LLM_PROVIDER not in self.VALID_LLM_PROVIDERS:
            raise RuntimeError("LLM_PROVIDER 只能是 openai 或 groq。")

        if self.USE_LOCAL_RAG:
            return

        missing = []
        PLACEHOLDER_KEY = {"your_groq_api_key","your_pinecone_api_key"}
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if self.LLM_PROVIDER == "groq" and (not self.GROQ_API_KEY or self.GROQ_API_KEY in PLACEHOLDER_KEY):
            missing.append("GROQ_API_KEY")
        if not self.PINECONE_API_KEY or self.PINECONE_API_KEY in PLACEHOLDER_KEY:
            missing.append("PINECONE_API_KEY")

        if missing:
            raise RuntimeError(
                "当前处于在线模式，但缺少必要配置："
                + "、".join(missing)
                + "。请补齐 .env，或把 USE_LOCAL_RAG 设为 true。"
            )


settings = Settings()
