import json
import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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
    from backend.place_lookup import (
        add_request_step_timing,
        cluster_media_by_gps,
        extract_capture_datetime,
        log_request_step_timings,
        lookup_place_info,
        reset_request_step_timings,
        restore_request_step_timings,
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
    from place_lookup import (
        add_request_step_timing,
        cluster_media_by_gps,
        extract_capture_datetime,
        log_request_step_timings,
        lookup_place_info,
        reset_request_step_timings,
        restore_request_step_timings,
    )


app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("bloghelper")

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
MEDIA_METADATA_CACHE: dict[str, dict[str, Any]] = {}
VALID_CATEGORY_KEYS = {category.key for category in CATEGORY_DEFINITIONS}
TRAINABLE_CATEGORY_KEYS = {category.key for category in CATEGORY_DEFINITIONS if category.trainable}
TIMESTAMP_FILENAME_PATTERN = re.compile(r"^\d{8}_\d{6}$")
PLACE_LOOKUP_INFERENCE_WINDOW_HOURS = 1
PLACE_LOOKUP_INFERENCE_WINDOW = timedelta(hours=PLACE_LOOKUP_INFERENCE_WINDOW_HOURS)


class LabelUpdateRequest(BaseModel):
    assignments: dict[str, str]


class TrainRequest(BaseModel):
    assignments: dict[str, str] | None = None


@contextmanager
def _record_timed_step(step_name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start
        add_request_step_timing(step_name, elapsed_seconds)


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    start = time.perf_counter()
    timing_token = reset_request_step_timings()
    LOGGER.info("요청 처리 시작 %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_seconds = time.perf_counter() - start
        log_request_step_timings()
        restore_request_step_timings(timing_token)
        LOGGER.exception("요청 처리 실패 %s %s / 걸린시간 %.3fs", request.method, request.url.path, elapsed_seconds)
        raise

    elapsed_seconds = time.perf_counter() - start
    log_request_step_timings()
    restore_request_step_timings(timing_token)
    LOGGER.info(
        "요청 처리 완료 %s %s status=%s / 걸린시간 %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_seconds,
    )
    return response


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
    with _record_timed_step("비디오 썸네일 추출"):
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


def _get_target_mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def _cache_place_info(media_path: Path, analysis_target: Path, place_info: dict[str, Any] | None) -> None:
    MEDIA_METADATA_CACHE[media_path.name] = {
        "target_mtime": _get_target_mtime(analysis_target),
        "place_info": place_info,
    }


def _get_cached_place_info(media_path: Path, analysis_target: Path) -> dict[str, Any] | None:
    cache_key = media_path.name
    cache_entry = MEDIA_METADATA_CACHE.get(cache_key)
    target_mtime = _get_target_mtime(analysis_target)
    if cache_entry and cache_entry.get("target_mtime") == target_mtime:
        return cache_entry.get("place_info")

    with _record_timed_step("장소 정보 조회"):
        try:
            place_info = lookup_place_info(analysis_target)
        except Exception:
            place_info = None
    _cache_place_info(media_path, analysis_target, place_info)
    return place_info


def _build_classified_media_analysis(
    media_path: Path,
    assignments: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    analysis_target = _get_analysis_target(media_path)
    with _record_timed_step("이미지 분류"):
        analysis = classify_image(analysis_target)
    if assignments is not None:
        analysis = _apply_manual_label(
            analysis,
            assignments,
            lookup_filename=media_path.name,
        )

    serialized = _serialize_media_analysis(analysis, media_path)
    return analysis_target, serialized


def _build_media_analysis(media_path: Path, assignments: dict[str, str] | None = None) -> dict[str, Any]:
    analysis_target, serialized = _build_classified_media_analysis(media_path, assignments)
    serialized["place_info"] = _get_cached_place_info(media_path, analysis_target)
    return serialized


def _get_media_capture_time(media_path: Path) -> datetime | None:
    capture_time = extract_capture_datetime(media_path)
    if capture_time is not None:
        return capture_time

    analysis_target = _get_analysis_target(media_path)
    if analysis_target != media_path:
        return extract_capture_datetime(analysis_target)
    return None


def _build_time_window_place_info(
    anchor_path: Path,
    anchor_capture_time: datetime,
    anchor_place_info: dict[str, Any],
    media_path: Path,
    capture_time: datetime,
) -> dict[str, Any]:
    nearest_place = anchor_place_info.get("nearest_place")
    reverse_geocode = anchor_place_info.get("reverse_geocode")
    nearby_places = anchor_place_info.get("nearby_places")
    gps = anchor_place_info.get("gps")
    place_name = None
    address = None
    if isinstance(nearest_place, dict):
        place_name = nearest_place.get("name")
    if not place_name and isinstance(reverse_geocode, dict):
        place_name = reverse_geocode.get("name")
    if isinstance(reverse_geocode, dict):
        address = reverse_geocode.get("display_name")

    inferred_place_info: dict[str, Any] = {
        "inferred_from_time_window": True,
        "inferred_source_filename": anchor_path.name,
        "inferred_place_name": place_name,
        "inferred_address": address,
        "inferred_time_delta_minutes": round(abs((capture_time - anchor_capture_time).total_seconds()) / 60.0, 1),
    }
    if isinstance(gps, dict):
        inferred_place_info["gps"] = dict(gps)
    if isinstance(reverse_geocode, dict):
        inferred_place_info["reverse_geocode"] = dict(reverse_geocode)
    if isinstance(nearest_place, dict):
        inferred_place_info["nearest_place"] = dict(nearest_place)
    if isinstance(nearby_places, list):
        inferred_place_info["nearby_places"] = [
            dict(place)
            for place in nearby_places
            if isinstance(place, dict)
        ]
    return inferred_place_info


def _resolve_batch_place_info(media_paths: list[Path]) -> dict[str, dict[str, Any] | None]:
    with _record_timed_step("장소 버킷 해석"):
        contexts = []
        for index, media_path in enumerate(media_paths):
            analysis_target = _get_analysis_target(media_path)
            contexts.append(
                {
                    "index": index,
                    "media_path": media_path,
                    "analysis_target": analysis_target,
                    "capture_time": _get_media_capture_time(media_path),
                }
            )

        ordered_contexts = sorted(
            contexts,
            key=lambda context: (
                context["capture_time"] is None,
                context["capture_time"] or datetime.max,
                context["index"],
            ),
        )

        resolved: dict[str, dict[str, Any] | None] = {}
        processed_filenames: set[str] = set()

        timed_contexts = [
            context
            for context in ordered_contexts
            if context["capture_time"] is not None
        ]
        untimed_contexts = [
            context
            for context in ordered_contexts
            if context["capture_time"] is None
        ]

        index = 0
        while index < len(timed_contexts):
            seed_context = timed_contexts[index]
            seed_filename = seed_context["media_path"].name
            if seed_filename in processed_filenames:
                index += 1
                continue

            seed_capture_time = seed_context["capture_time"]
            bucket_contexts = []
            bucket_end = seed_capture_time + PLACE_LOOKUP_INFERENCE_WINDOW
            lookahead = index
            while lookahead < len(timed_contexts):
                candidate_context = timed_contexts[lookahead]
                candidate_filename = candidate_context["media_path"].name
                candidate_capture_time = candidate_context["capture_time"]
                if candidate_capture_time is None or candidate_capture_time > bucket_end:
                    break
                if candidate_filename not in processed_filenames:
                    bucket_contexts.append(candidate_context)
                lookahead += 1

            anchor_context = None
            anchor_place_info = None
            for context in bucket_contexts:
                media_path = context["media_path"]
                place_info = _get_cached_place_info(media_path, context["analysis_target"])
                resolved[media_path.name] = place_info

                gps = place_info.get("gps") if isinstance(place_info, dict) else None
                if isinstance(gps, dict):
                    anchor_context = context
                    anchor_place_info = place_info
                    break

            if anchor_context is None or anchor_place_info is None:
                for context in bucket_contexts:
                    processed_filenames.add(context["media_path"].name)
                index = lookahead
                continue

            anchor_capture_time = anchor_context["capture_time"]
            anchor_media_path = anchor_context["media_path"]
            for context in timed_contexts:
                media_path = context["media_path"]
                filename = media_path.name
                if filename in processed_filenames:
                    continue

                capture_time = context["capture_time"]
                if capture_time is None:
                    continue
                if abs(capture_time - anchor_capture_time) > PLACE_LOOKUP_INFERENCE_WINDOW:
                    continue

                if filename == anchor_media_path.name:
                    resolved[filename] = anchor_place_info
                    processed_filenames.add(filename)
                    continue

                inferred_place_info = _build_time_window_place_info(
                    anchor_media_path,
                    anchor_capture_time,
                    anchor_place_info,
                    media_path,
                    capture_time,
                )
                resolved[filename] = inferred_place_info
                _cache_place_info(media_path, context["analysis_target"], inferred_place_info)
                processed_filenames.add(filename)

            index += 1

        for context in untimed_contexts:
            media_path = context["media_path"]
            filename = media_path.name
            if filename in processed_filenames:
                continue

            place_info = _get_cached_place_info(media_path, context["analysis_target"])
            resolved[filename] = place_info
            processed_filenames.add(filename)

        return resolved


def _build_media_analyses(
    media_paths: list[Path],
    assignments: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    place_info_by_filename = _resolve_batch_place_info(media_paths)
    analyses = []
    for media_path in media_paths:
        _, serialized = _build_classified_media_analysis(media_path, assignments)
        serialized["place_info"] = place_info_by_filename.get(media_path.name)
        analyses.append(serialized)
    return analyses


def _is_timestamp_filename(filename: str) -> bool:
    return bool(TIMESTAMP_FILENAME_PATTERN.fullmatch(Path(filename).stem))


def _infer_location_groups_from_neighbors(
    analyses: list[dict[str, Any]],
    location_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filename_to_group_id: dict[str, str] = {}
    groups_by_id = {group["group_id"]: group for group in location_groups}

    for group in location_groups:
        for item in group["items"]:
            filename_to_group_id[item["filename"]] = group["group_id"]

    def find_neighbor_group_id(start_index: int, step: int) -> str | None:
        neighbor_index = start_index + step
        if neighbor_index < 0 or neighbor_index >= len(analyses):
            return None

        neighbor = analyses[neighbor_index]
        return filename_to_group_id.get(neighbor["filename"])

    for index, analysis in enumerate(analyses):
        filename = analysis["filename"]
        if filename in filename_to_group_id or not _is_timestamp_filename(filename):
            continue

        previous_group_id = find_neighbor_group_id(index, -1)
        next_group_id = find_neighbor_group_id(index, 1)

        target_group_id = None
        if previous_group_id and next_group_id:
            if previous_group_id == next_group_id:
                target_group_id = previous_group_id
        else:
            target_group_id = previous_group_id or next_group_id

        if not target_group_id:
            continue

        target_group = groups_by_id.get(target_group_id)
        if not target_group:
            continue

        place_info = dict(analysis.get("place_info") or {})
        place_info["inferred_from_neighbors"] = True
        place_info["inferred_group_id"] = target_group_id
        place_info["inferred_place_name"] = target_group.get("place_name")
        place_info["inferred_address"] = target_group.get("address")
        place_info["neighbor_filenames"] = {
            "previous": analyses[index - 1]["filename"] if index > 0 else None,
            "next": analyses[index + 1]["filename"] if index + 1 < len(analyses) else None,
        }
        analysis["place_info"] = place_info
        target_group["items"].append(analysis)

    for group in location_groups:
        group["count"] = len(group["items"])

    location_groups.sort(key=lambda group: group["count"], reverse=True)
    return location_groups


def _clear_upload_dir() -> None:
    MEDIA_METADATA_CACHE.clear()
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
    uploaded_media_paths: list[Path] = []
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
            if file_extension in ALLOWED_VIDEO_EXTENSIONS:
                _extract_video_thumbnail(file_path)
            uploaded_media_paths.append(file_path)
            _update_upload_job(
                job,
                processed_files=index,
                uploaded_count=len(uploaded_media_paths),
                message=f"{index}/{len(files)}번째 파일 저장을 완료했습니다.",
            )
        except Exception:
            if file_path.exists():
                try:
                    file_path.unlink()
                except PermissionError:
                    pass
            thumbnail_path = _video_thumbnail_path(file_path)
            if thumbnail_path.exists():
                try:
                    thumbnail_path.unlink()
                except PermissionError:
                    pass
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

    if uploaded_media_paths:
        _update_upload_job(
            job,
            stage="analyzing",
            message="GPS 기준 파일만 확인하고 같은 시간대 파일은 장소 조회를 건너뛰는 중입니다.",
            current_file=None,
        )
        place_info_by_filename = _resolve_batch_place_info(uploaded_media_paths)

        for index, media_path in enumerate(uploaded_media_paths, start=1):
            _update_upload_job(
                job,
                stage="analyzing",
                message=f"{index}/{len(uploaded_media_paths)}번째 파일 분류를 완료하는 중입니다.",
                current_file=media_path.name,
            )
            try:
                _, serialized = _build_classified_media_analysis(media_path)
                serialized["place_info"] = place_info_by_filename.get(media_path.name)
                uploaded_files.append(serialized)
                _update_upload_job(
                    job,
                    uploaded_count=len(uploaded_files),
                )
            except Exception:
                if media_path.exists():
                    try:
                        media_path.unlink()
                    except PermissionError:
                        pass
                thumbnail_path = _video_thumbnail_path(media_path)
                if thumbnail_path.exists():
                    try:
                        thumbnail_path.unlink()
                    except PermissionError:
                        pass
                failed_files.append(
                    {
                        "filename": media_path.name,
                        "reason": "이미지 분석에 실패했습니다.",
                    }
                )
                _update_upload_job(
                    job,
                    failed_count=len(failed_files),
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
    analyses = _build_media_analyses(_list_uploaded_media(), assignments)
    location_groups = _infer_location_groups_from_neighbors(
        analyses,
        cluster_media_by_gps(analyses, max_distance_meters=10.0),
    )
    grouped_images = {category.key: [] for category in CATEGORY_DEFINITIONS}
    for analysis in analyses:
        grouped_images[analysis["category"]].append(analysis)

    return {
        "total_count": len(analyses),
        "categories": get_category_metadata(),
        "grouped_images": grouped_images,
        "location_groups": location_groups,
        "assignments": assignments,
    }
