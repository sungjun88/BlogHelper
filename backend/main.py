from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from backend.image_classifier import (
        ALLOWED_IMAGE_EXTENSIONS,
        classify_image,
        get_category_metadata,
        get_enabled_categories,
        list_uploaded_images,
    )
except ModuleNotFoundError:
    from image_classifier import (
        ALLOWED_IMAGE_EXTENSIONS,
        classify_image,
        get_category_metadata,
        get_enabled_categories,
        list_uploaded_images,
    )


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

UPLOAD_JOBS: dict[str, dict] = {}


def _build_image_url(filename: str) -> str:
    return f"/uploads/{quote(filename)}"


def _serialize_analysis(analysis: dict) -> dict:
    return {
        **analysis,
        "image_url": _build_image_url(analysis["filename"]),
    }


def _clear_upload_dir() -> None:
    for path in UPLOAD_DIR.iterdir():
        if path.is_file():
            path.unlink()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _build_job_snapshot(job: dict) -> dict:
    total_units = max(job["total_files"] + 1, 1)
    completed_units = job["transfer_progress"] + job["processed_files"]
    progress_ratio = min(max(completed_units / total_units, 0.0), 1.0)

    if job["status"] == "completed":
        progress_ratio = 1.0
    elif job["status"] == "failed":
        progress_ratio = min(max(progress_ratio, 0.0), 0.99)

    elapsed_seconds = (_utc_now() - job["started_at"]).total_seconds()
    estimated_total_seconds = None
    remaining_seconds = None
    if progress_ratio > 0:
        estimated_total_seconds = elapsed_seconds / progress_ratio
        remaining_seconds = max(estimated_total_seconds - elapsed_seconds, 0.0)

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "message": job["message"],
        "current_file": job["current_file"],
        "total_files": job["total_files"],
        "processed_files": job["processed_files"],
        "uploaded_count": job["uploaded_count"],
        "ignored_count": job["ignored_count"],
        "failed_count": job["failed_count"],
        "transfer_progress": round(job["transfer_progress"], 4),
        "progress_percent": round(progress_ratio * 100, 1),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "estimated_total_seconds": round(estimated_total_seconds, 1) if estimated_total_seconds is not None else None,
        "remaining_seconds": round(remaining_seconds, 1) if remaining_seconds is not None else None,
        "started_at": _isoformat(job["started_at"]),
        "updated_at": _isoformat(job["updated_at"]),
    }


def _create_upload_job() -> dict:
    now = _utc_now()
    job = {
        "job_id": uuid4().hex,
        "status": "pending",
        "stage": "queued",
        "message": "업로드 준비 중입니다.",
        "current_file": None,
        "total_files": 0,
        "processed_files": 0,
        "uploaded_count": 0,
        "ignored_count": 0,
        "failed_count": 0,
        "transfer_progress": 0.0,
        "started_at": now,
        "updated_at": now,
    }
    UPLOAD_JOBS[job["job_id"]] = job
    return job


def _update_upload_job(job: dict, **changes) -> None:
    job.update(changes)
    job["updated_at"] = _utc_now()


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/results")
def results_page():
    return FileResponse(FRONTEND_DIR / "results.html")


@app.get("/health")
def health():
    return {"message": "API running"}


@app.post("/upload/init")
def init_upload():
    job = _create_upload_job()
    return _build_job_snapshot(job)


@app.get("/upload/status/{job_id}")
def get_upload_status(job_id: str):
    job = UPLOAD_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Upload job not found")
    return _build_job_snapshot(job)


@app.get("/categories")
def get_categories():
    return {
        "categories": get_category_metadata(),
        "enabled_category_keys": [category.key for category in get_enabled_categories()],
    }


@app.get("/test-upload", response_class=HTMLResponse)
async def test_upload_page():
    return """
    <html>
        <head>
            <meta charset="utf-8">
            <title>파일 업로드 테스트</title>
        </head>
        <body>
            <h2>여러 이미지 업로드</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple>
                <button type="submit">업로드</button>
            </form>
        </body>
    </html>
    """


@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...), job_id: str | None = None):
    job = UPLOAD_JOBS.get(job_id) if job_id else None
    if job is None:
        job = _create_upload_job()

    uploaded_files = []
    ignored_files = []
    failed_files = []

    _update_upload_job(
        job,
        status="processing",
        stage="preparing",
        message="업로드된 파일을 정리하고 있습니다.",
        total_files=len(files),
        transfer_progress=1.0,
    )

    _clear_upload_dir()

    for index, file in enumerate(files, start=1):
        _update_upload_job(
            job,
            stage="processing",
            message=f"{index}/{len(files)}번째 파일을 분석하고 있습니다.",
            current_file=file.filename,
        )

        file_extension = Path(file.filename or "").suffix.lower()
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            ignored_files.append(
                {
                    "filename": file.filename,
                    "reason": "이미지 파일만 업로드할 수 있습니다.",
                }
            )
            _update_upload_job(
                job,
                processed_files=index,
                ignored_count=len(ignored_files),
                message=f"{index}/{len(files)}번째 파일은 이미지가 아니어서 제외했습니다.",
            )
            continue

        file_path = UPLOAD_DIR / file.filename
        content = await file.read()
        file_path.write_bytes(content)

        try:
            uploaded_files.append(_serialize_analysis(classify_image(file_path)))
            _update_upload_job(
                job,
                processed_files=index,
                uploaded_count=len(uploaded_files),
                message=f"{index}/{len(files)}번째 파일 분석을 완료했습니다.",
            )
        except Exception:
            if file_path.exists():
                file_path.unlink()
            failed_files.append(
                {
                    "filename": file.filename,
                    "reason": "이미지 분석에 실패했습니다.",
                }
            )
            _update_upload_job(
                job,
                processed_files=index,
                failed_count=len(failed_files),
                message=f"{index}/{len(files)}번째 파일 분석에 실패했습니다.",
            )

    result = {
        "uploaded_count": len(uploaded_files),
        "ignored_count": len(ignored_files),
        "failed_count": len(failed_files),
        "categories": get_category_metadata(),
        "files": uploaded_files,
        "ignored_files": ignored_files,
        "failed_files": failed_files,
    }

    _update_upload_job(
        job,
        status="completed",
        stage="completed",
        message="모든 파일 분류가 완료되었습니다.",
        current_file=None,
        processed_files=len(files),
        uploaded_count=len(uploaded_files),
        ignored_count=len(ignored_files),
        failed_count=len(failed_files),
    )

    return result


@app.get("/images")
def get_uploaded_images():
    files = [path.name for path in list_uploaded_images(UPLOAD_DIR)]
    return {"images": files}


@app.get("/images/categorized")
def get_categorized_images():
    analyses = [
        _serialize_analysis(classify_image(path))
        for path in list_uploaded_images(UPLOAD_DIR)
    ]
    grouped_images = {category.key: [] for category in get_enabled_categories()}
    for analysis in analyses:
        grouped_images[analysis["category"]].append(analysis)

    return {
        "total_count": len(analyses),
        "categories": get_category_metadata(),
        "grouped_images": grouped_images,
    }
