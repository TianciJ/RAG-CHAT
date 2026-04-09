import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()
UPLOAD_DIR = Path("uploads")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in {".pdf", ".md"}:
        raise HTTPException(
            status_code=400,
            detail="只支持 PDF 和 Markdown 文件。"
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    target_path = UPLOAD_DIR / file.filename

    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from ingest.ingest_pipeline import ingest_file

    chunk_count = ingest_file(str(target_path))

    return {
        "status": "success",
        "filename": file.filename,
        "chunks": chunk_count
    }
