from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI

from api.chat_routers import router as chat_router
from api.ingest_routers import router as ingest_router


app = FastAPI(title="RAG Chat", version="0.1.0")

app.include_router(chat_router)
app.include_router(ingest_router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # 直接运行脚本时保持简单稳定。
    # 如果需要热重载，建议使用 `uvicorn app.main:app --reload`。
    uvicorn.run(app, host="0.0.0.0", port=8000)
