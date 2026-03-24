import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from backend.image_classifier import (
        ALLOWED_IMAGE_EXTENSIONS,
        CATEGORY_DEFINITIONS,
        classify_image,
        get_category_metadata,
        get_enabled_categories,
        list_uploaded_images,
        train_embedding_classifier,
    )
except ModuleNotFoundError:
    from image_classifier import (
        ALLOWED_IMAGE_EXTENSIONS,
        CATEGORY_DEFINITIONS,
        classify_image,
        get_category_metadata,
        get_enabled_categories,
        list_uploaded_images,
        train_embedding_classifier,
    )


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
LABELS_FILE = BASE_DIR / "tuning_labels.json"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VIDEO_THUMBNAIL_SUFFIX = "__thumb.jpg"
UPLOAD_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

UPLOAD_JOBS: dict[str, dict] = {}
VALID_CATEGORY_KEYS = {category.key for category in CATEGORY_DEFINITIONS}
TRAINABLE_CATEGORY_KEYS = {category.key for category in CATEGORY_DEFINITIONS if category.trainable}


class LabelUpdateRequest(BaseModel):
    assignments: dict[str, str]


class TrainRequest(BaseModel):
    assignments: dict[str, str] | None = None


def _build_image_url(filename: str) -> str:
    return f"/uploads/{quote(filename)}"


def _is_video_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def _is_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def _is_uploaded_image_path(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        and not path.name.endswith(VIDEO_THUMBNAIL_SUFFIX)
    )


def _list_uploaded_media() -> list[Path]:
    return sorted(
        path
        for path in UPLOAD_DIR.iterdir()
        if _is_uploaded_image_path(path) or _is_video_path(path)
    )


def _video_thumbnail_name(filename: str) -> str:
    return f"{Path(filename).stem}{VIDEO_THUMBNAIL_SUFFIX}"


def _video_thumbnail_path(video_path: Path) -> Path:
    return UPLOAD_DIR / _video_thumbnail_name(video_path.name)


def _extract_video_thumbnail(video_path: Path) -> Path:
    thumbnail_path = _video_thumbnail_path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path.name}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        candidate_indices = [0]
        if frame_count > 0:
            candidate_indices = sorted({0, max(frame_count // 3, 0), max(frame_count // 2, 0)})

        frame = None
        for frame_index in candidate_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, current_frame = capture.read()
            if ok and current_frame is not None:
                frame = current_frame
                break

        if frame is None:
            raise RuntimeError(f"Could not extract thumbnail frame from: {video_path.name}")

        if not cv2.imwrite(str(thumbnail_path), frame):
            raise RuntimeError(f"Could not write thumbnail image for: {video_path.name}")
    finally:
        capture.release()

    return thumbnail_path


def _get_analysis_target(media_path: Path) -> Path:
    if _is_video_path(media_path):
        return _video_thumbnail_path(media_path)
    return media_path


def _load_labels_file() -> dict[str, dict[str, str]]:
    if not LABELS_FILE.exists():
        return {}

    data = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Label file must be a JSON object keyed by filename.")
    return data


def _save_labels_file(labels: dict[str, dict[str, str]]) -> None:
    LABELS_FILE.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_uploaded_label_assignments() -> dict[str, str]:
    labels = _load_labels_file()
    uploaded_filenames = {path.name for path in _list_uploaded_media()}
    assignments: dict[str, str] = {}
    for filename, entry in labels.items():
        label = entry.get("label", "")
        if filename in uploaded_filenames and label in VALID_CATEGORY_KEYS:
            assignments[filename] = label
    return assignments


def _persist_label_assignments(assignments: dict[str, str]) -> dict[str, str]:
    labels = _load_labels_file()
    uploaded_filenames = {path.name for path in _list_uploaded_media()}

    for filename, label in assignments.items():
        if filename not in uploaded_filenames:
            raise HTTPException(status_code=400, detail=f"Unknown uploaded file: {filename}")
        if label not in VALID_CATEGORY_KEYS:
            raise HTTPException(status_code=400, detail=f"Invalid category label: {label}")

    for filename in uploaded_filenames:
        current = labels.get(filename, {})
        label = assignments.get(filename, current.get("label", ""))
        notes = current.get("notes", "")
        labels[filename] = {
            "label": label if label in VALID_CATEGORY_KEYS else "",
            "notes": notes,
        }

    _save_labels_file(labels)
    return _get_uploaded_label_assignments()


def _apply_manual_label(
    analysis: dict[str, Any],
    assignments: dict[str, str],
    lookup_filename: str | None = None,
) -> dict[str, Any]:
    manual_label = assignments.get(lookup_filename or analysis["filename"])
    if not manual_label:
        return {
            **analysis,
            "is_manual_label": False,
        }

    category_lookup = {category.key: category for category in CATEGORY_DEFINITIONS}
    return {
        **analysis,
        "category": manual_label,
        "category_label": category_lookup[manual_label].label,
        "manual_label": manual_label,
        "is_manual_label": True,
    }


def _serialize_analysis(analysis: dict) -> dict:
    return {
        **analysis,
        "image_url": _build_image_url(analysis["filename"]),
    }


def _serialize_media_analysis(analysis: dict, media_path: Path) -> dict:
    preview_path = _get_analysis_target(media_path)
    return {
        **analysis,
        "filename": media_path.name,
        "image_url": _build_image_url(preview_path.name),
        "media_type": "video" if _is_video_path(media_path) else "image",
        "preview_filename": preview_path.name,
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


@app.get("/labels")
def get_labels():
    return {"assignments": _get_uploaded_label_assignments()}


@app.post("/labels")
def update_labels(payload: LabelUpdateRequest):
    assignments = _persist_label_assignments(payload.assignments)
    return {
        "saved_count": len(assignments),
        "assignments": assignments,
    }


@app.post("/train")
def train_current_uploads(payload: TrainRequest):
    if payload.assignments is not None:
        assignments = _persist_label_assignments(payload.assignments)
    else:
        assignments = _get_uploaded_label_assignments()

    labeled_items = []
    for path in _list_uploaded_media():
        label = assignments.get(path.name)
        if label in TRAINABLE_CATEGORY_KEYS:
            labeled_items.append((_get_analysis_target(path), label))

    if not labeled_items:
        raise HTTPException(status_code=400, detail="No labeled uploaded images are available for training.")

    try:
        summary = train_embedding_classifier(labeled_items)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "trained_count": len(labeled_items),
        "summary": summary,
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
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS and file_extension not in ALLOWED_VIDEO_EXTENSIONS:
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
            analysis_target = file_path
            if file_extension in ALLOWED_VIDEO_EXTENSIONS:
                analysis_target = _extract_video_thumbnail(file_path)
            uploaded_files.append(_serialize_media_analysis(classify_image(analysis_target), file_path))
            _update_upload_job(
                job,
                processed_files=index,
                uploaded_count=len(uploaded_files),
                message=f"{index}/{len(files)}번째 파일 분석을 완료했습니다.",
            )
        except Exception:
            if file_path.exists():
                file_path.unlink()
            thumbnail_path = _video_thumbnail_path(file_path)
            if thumbnail_path.exists():
                thumbnail_path.unlink()
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
    files = [path.name for path in _list_uploaded_media()]
    return {"images": files}


@app.get("/images/categorized")
def get_categorized_images():
    assignments = _get_uploaded_label_assignments()
    analyses = [
        _serialize_media_analysis(
            _apply_manual_label(
                classify_image(_get_analysis_target(path)),
                assignments,
                lookup_filename=path.name,
            ),
            path,
        )
        for path in _list_uploaded_media()
    ]
    grouped_images = {category.key: [] for category in CATEGORY_DEFINITIONS}
    for analysis in analyses:
        grouped_images[analysis["category"]].append(analysis)

    return {
        "total_count": len(analyses),
        "categories": get_category_metadata(),
        "grouped_images": grouped_images,
        "assignments": assignments,
    }
